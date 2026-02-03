import Foundation

// MARK: - Graph-Based Training Framework
//
// This training framework uses the tinygrad-style gradient computation where
// gradients are built as LazyOps in the graph rather than emitting backward IR.
// Key differences from Training.swift:
// - Gradients computed via graph.computeGradients() before compilation
// - Uses memoryAccumulate for atomic per-frame gradient accumulation
// - Forward and backward are unified - just one compilation pass

/// A trainable parameter for the graph-based training system
public class GraphParameter {
    public let cellId: CellID
    public let nodeId: NodeID
    public var value: Float
    public let name: String?

    // Set after prepareForTraining - the memory cell where gradient accumulates
    var gradientCell: CellID?

    // Gradient value (read after each step)
    public var grad: Float = 0.0

    public init(graph: Graph, value: Float, name: String? = nil) {
        self.cellId = graph.alloc()
        self.value = value
        self.name = name
        self.nodeId = graph.n(.param(cellId))
    }

    public func node() -> NodeID {
        return nodeId
    }
}

/// Optimizer protocol for graph-based training
public protocol GraphOptimizer {
    mutating func update(parameters: inout [GraphParameter], learningRate: Float)
    mutating func updateTensors(tensorParameters: inout [TensorParameter], learningRate: Float)
}

/// Simple SGD optimizer
public struct GraphSGD: GraphOptimizer {
    public init() {}

    public mutating func update(parameters: inout [GraphParameter], learningRate: Float) {
        for param in parameters {
            param.value -= learningRate * param.grad
        }
    }

    public mutating func updateTensors(
        tensorParameters: inout [TensorParameter], learningRate: Float
    ) {
        for param in tensorParameters {
            for i in 0..<param.data.count {
                param.data[i] -= learningRate * param.grads[i]
            }
        }
    }
}

/// Adam optimizer
public struct GraphAdam: GraphOptimizer {
    private var m: [Float] = []
    private var v: [Float] = []
    private var t: Int = 0
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float

    // For tensor parameters - flattened momentum arrays
    private var tensorM: [[Float]] = []
    private var tensorV: [[Float]] = []

    public init(beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    public mutating func update(parameters: inout [GraphParameter], learningRate: Float) {
        if m.isEmpty && !parameters.isEmpty {
            m = Array(repeating: 0.0, count: parameters.count)
            v = Array(repeating: 0.0, count: parameters.count)
        }

        // Note: t is incremented in updateTensors() which is always called after this
        // We use (t+1) here to get the correct timestep for this step
        let currentT = t + 1

        for i in 0..<parameters.count {
            let grad = parameters[i].grad

            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad

            let mHat = m[i] / (1 - pow(beta1, Float(currentT)))
            let vHat = v[i] / (1 - pow(beta2, Float(currentT)))

            parameters[i].value -= learningRate * mHat / (sqrt(vHat) + epsilon)
        }
    }

    public mutating func updateTensors(
        tensorParameters: inout [TensorParameter], learningRate: Float
    ) {
        // Initialize momentum arrays if needed
        if tensorM.isEmpty && !tensorParameters.isEmpty {
            tensorM = tensorParameters.map { Array(repeating: Float(0.0), count: $0.data.count) }
            tensorV = tensorParameters.map { Array(repeating: Float(0.0), count: $0.data.count) }
        }

        // Increment timestep for tensor updates
        t += 1

        for (pi, param) in tensorParameters.enumerated() {
            for i in 0..<param.data.count {
                let grad = param.grads[i]

                tensorM[pi][i] = beta1 * tensorM[pi][i] + (1 - beta1) * grad
                tensorV[pi][i] = beta2 * tensorV[pi][i] + (1 - beta2) * grad * grad

                let mHat = tensorM[pi][i] / (1 - pow(beta1, Float(t)))
                let vHat = tensorV[pi][i] / (1 - pow(beta2, Float(t)))

                param.data[i] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }
}

// MARK: - Graph Training Context

/// Manages training using graph-based gradient computation
public class GraphTrainingContext {
    public var parameters: [GraphParameter]
    public var tensorParameters: [TensorParameter]
    private var optimizer: GraphOptimizer
    public let learningRate: Float

    private var graph: Graph
    private let lossNode: NodeID
    private var runtime: MetalCompiledKernel?
    private var cellAllocations: CellAllocations?
    private var frameCount: Int
    private var kernelDebugOutput: String?

    // Physical cell indices for fast access (scalar params)
    private var paramPhysicalCells: [Int] = []
    private var gradPhysicalCells: [Int] = []

    // Physical cell indices for tensor params
    private var tensorPhysicalCells: [(base: Int, size: Int)] = []
    private var tensorGradPhysicalCells: [(base: Int, size: Int)] = []

    /// Initialize training context
    /// - Parameters:
    ///   - graph: The computation graph (will be modified to add gradient nodes)
    ///   - loss: The loss node to compute gradients from
    ///   - parameters: Trainable scalar parameters
    ///   - tensorParameters: Trainable tensor parameters
    ///   - optimizer: Optimization algorithm
    ///   - learningRate: Learning rate for optimization
    ///   - frameCount: Number of frames per training step
    ///   - kernelDebugOutput: Optional path to write generated kernels for debugging
    public init(
        graph: Graph,
        loss: NodeID,
        parameters: [GraphParameter] = [],
        tensorParameters: [TensorParameter] = [],
        optimizer: GraphOptimizer = GraphSGD(),
        learningRate: Float = 0.001,
        frameCount: Int = 64,
        kernelDebugOutput: String? = nil
    ) throws {
        precondition(
            frameCount <= graph.maxFrameCount,
            "frameCount (\(frameCount)) exceeds graph.maxFrameCount (\(graph.maxFrameCount)). " +
            "Set graph.maxFrameCount to at least \(frameCount) before creating GraphTraining."
        )
        self.graph = graph
        self.lossNode = loss
        self.parameters = parameters
        self.tensorParameters = tensorParameters
        self.optimizer = optimizer
        self.learningRate = learningRate
        self.frameCount = frameCount
        self.kernelDebugOutput = kernelDebugOutput

        // Prepare gradients and compile
        try prepareAndCompile()
    }

    /// Prepare gradients and compile the graph
    private func prepareAndCompile() throws {
        // Record the last forward node ID before adding gradient nodes
        // This ensures gradient nodes don't affect forward node ordering
        let lastForwardNodeId = graph.nodes.keys.max() ?? 0
        graph.lastForwardNodeId = lastForwardNodeId

        // Step 1: Compute gradient nodes for all parameters (scalar + tensor)
        var targetNodes = Set(parameters.map { $0.nodeId })
        for tp in tensorParameters {
            targetNodes.insert(tp.nodeId)
        }
        let gradients = graph.computeGradients(loss: lossNode, targets: targetNodes)

        // Step 2: For each scalar parameter, create atomic accumulator for its gradient
        let zero = graph.n(.constant(0.0))
        for param in parameters {
            guard var gradNode = gradients[param.nodeId] else {
                print("Warning: No gradient found for parameter \(param.name ?? "unnamed")")
                continue
            }

            // Chain any gradient side effects with this gradient node
            // This ensures temporal gradient carry operations are scheduled
            for sideEffect in graph.gradientSideEffects {
                gradNode = graph.n(.seq, [sideEffect, gradNode])
            }

            // Allocate cell for accumulated gradient
            let gradCell = graph.alloc()
            param.gradientCell = gradCell

            // Insert atomic accumulation
            _ = graph.n(.memoryAccumulate(gradCell), [zero, gradNode])
        }

        // Step 2b: For tensor parameters, use shared gradient setup
        let tensorNodes = tensorParameters.map { ($0.nodeId, $0.size) }
        let _ = graph.setupTensorGradients(gradients: gradients, tensorNodes: tensorNodes)

        // Ensure gradient side effects are scheduled
        if !graph.gradientSideEffects.isEmpty {
            // Find the output node and make it depend on all side effects
            // This ensures the scatter operations get scheduled
            if let outputNodeId = graph.nodes.first(where: {
                if case .output(_) = $0.value.op { return true }
                return false
            })?.key {
                // Chain all side effects before the output
                var chainedValue = lossNode
                for sideEffect in graph.gradientSideEffects {
                    chainedValue = graph.n(.seq, [sideEffect, chainedValue])
                }
                // Update the output node to use the chained value
                // We need to create a new output that depends on the side effects
                let newOutput = graph.n(.output(0), [chainedValue])
                // The old output is orphaned, new one will be used
                _ = newOutput
            }
        }

        // Clear side effects after use
        graph.gradientSideEffects = []

        // Step 3: Compile the graph
        let result = try CompilationPipeline.compile(
            graph: graph,
            backend: .metal,
            options: .init(frameCount: frameCount)
        )

        // Write kernels to disk if debug output path provided
        if let outputPath = kernelDebugOutput {
            writeKernelsToDisk(result, outputPath)
        }

        self.cellAllocations = result.cellAllocations

        // Step 4: Create runtime
        self.runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Step 5: Cache physical cell indices for scalar params
        for param in parameters {
            let physicalParam = result.cellAllocations.cellMappings[param.cellId] ?? param.cellId
            paramPhysicalCells.append(physicalParam)

            if let gradCell = param.gradientCell {
                let physicalGrad = result.cellAllocations.cellMappings[gradCell] ?? gradCell
                gradPhysicalCells.append(physicalGrad)
            } else {
                gradPhysicalCells.append(-1)
            }
        }

        // Step 5b: Cache physical cell indices for tensor params
        for tp in tensorParameters {
            let physicalBase = result.cellAllocations.cellMappings[tp.cellId] ?? tp.cellId
            tensorPhysicalCells.append((base: physicalBase, size: tp.size))

            // First: check tensorGradCells (from tensorAccumulate)
            if let gradCell = graph.tensorGradCells[tp.nodeId] {
                let physical = result.cellAllocations.cellMappings[gradCell] ?? gradCell
                tensorGradPhysicalCells.append((base: physical, size: tp.size))
            }
            // Fallback: scatter-based (direct peekRow)
            else if let gradCellId = graph.gradCarryCells[tp.cellId] {
                let physicalGradBase = result.cellAllocations.cellMappings[gradCellId] ?? gradCellId
                tensorGradPhysicalCells.append((base: physicalGradBase, size: tp.size))
            } else {
                // No gradient cell found - tensor wasn't read with gradient-aware ops
                tensorGradPhysicalCells.append((base: -1, size: tp.size))
            }
        }

        // Step 6: Initialize parameter values in memory
        initializeMemory()
    }

    /// Initialize parameter values in GPU memory
    private func initializeMemory() {
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }
        guard let cellAllocs = cellAllocations else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        // Initialize scalar parameters
        for (i, param) in parameters.enumerated() {
            memPtr[paramPhysicalCells[i]] = param.value
        }

        // Initialize tensor parameters
        for (i, tp) in tensorParameters.enumerated() {
            let base = tensorPhysicalCells[i].base
            for j in 0..<tp.data.count {
                memPtr[base + j] = tp.data[j]
            }
        }

        // Initialize all tensors with initial data (not just TensorParameters)
        // Skip tensors that belong to TensorParameters - those were already written above
        let tensorParamCellIds = Set(tensorParameters.map { $0.cellId })
        for (_, tensor) in graph.tensors {
            // Skip if this tensor belongs to a TensorParameter (already handled above)
            if tensorParamCellIds.contains(tensor.cellId) {
                continue
            }
            if let data = tensor.data {
                let physicalBase = cellAllocs.cellMappings[tensor.cellId] ?? tensor.cellId
                for j in 0..<data.count {
                    memPtr[physicalBase + j] = data[j]
                }
            }
        }
    }

    /// Zero the gradient accumulators
    public func zeroGrad() {
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        // Zero scalar param gradients
        for i in 0..<parameters.count {
            if gradPhysicalCells[i] >= 0 {
                memPtr[gradPhysicalCells[i]] = 0.0
            }
        }

        // Zero tensor param gradients
        for i in 0..<tensorParameters.count {
            let gradInfo = tensorGradPhysicalCells[i]
            if gradInfo.base >= 0 {
                for j in 0..<gradInfo.size {
                    memPtr[gradInfo.base + j] = 0.0
                }
            }
        }
    }

    /// Reset all state memory (phasor state, scratch cells, etc.) while preserving parameters.
    /// Call this at the start of each training step for deterministic gradients.
    public func resetState(debug: Bool = false) {
        guard let runtime = runtime else { return }

        runtime.zeroAllBuffers()
        if debug { print("[resetState] Zeroed all GPU buffers") }

        initializeMemory()

        if debug, let memBuffer = runtime.getBuffer(name: "memory") {
            let bufferSize = memBuffer.length / MemoryLayout<Float>.size
            let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
            var nonZero = 0
            var sum: Float = 0
            for i in 0..<bufferSize {
                if memPtr[i] != 0 {
                    nonZero += 1
                    sum += memPtr[i]
                }
            }
            print("[resetState] After init: nonZero=\(nonZero), sum=\(sum)")
        }
    }

    /// Run forward pass and accumulate gradients
    /// - Returns: Mean loss across frames
    public func forward() -> Float {
        guard let runtime = runtime else { return 0.0 }

        // Run the compiled graph (forward + gradient accumulation)
        runtime.runNoCopy(frameCount: frameCount)

        // Read gradients from memory
        if let memBuffer = runtime.getBuffer(name: "memory") {
            let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

            // Read scalar param gradients (normalize by frame count to get mean gradient)
            let scalarGradScale = 1.0 / Float(frameCount)
            for (i, param) in parameters.enumerated() {
                if gradPhysicalCells[i] >= 0 {
                    param.grad = memPtr[gradPhysicalCells[i]] * scalarGradScale
                }
            }

            // Read tensor param gradients (normalize by frame count to get mean gradient)
            let gradScale = 1.0 / Float(frameCount)
            for (i, tp) in tensorParameters.enumerated() {
                let gradInfo = tensorGradPhysicalCells[i]
                if gradInfo.base >= 0 {
                    for j in 0..<gradInfo.size {
                        tp.grads[j] = memPtr[gradInfo.base + j] * gradScale
                    }
                }
            }
        }

        // Return mean loss
        let outputs = runtime.getOutputBuffer()
        if outputs.isEmpty { return 0.0 }
        return outputs.reduce(0.0, +) / Float(outputs.count)
    }

    /// Apply optimizer to update parameters
    public func step() {
        // Update scalar parameters using optimizer
        optimizer.update(parameters: &parameters, learningRate: learningRate)

        // Update tensor parameters using optimizer
        optimizer.updateTensors(tensorParameters: &tensorParameters, learningRate: learningRate)

        // Write updated values back to GPU memory
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        // Write scalar params
        for (i, param) in parameters.enumerated() {
            memPtr[paramPhysicalCells[i]] = param.value
        }

        // Write tensor params
        for (i, tp) in tensorParameters.enumerated() {
            let base = tensorPhysicalCells[i].base
            for j in 0..<tp.data.count {
                memPtr[base + j] = tp.data[j]
            }
        }
    }

    /// Run a complete training step: zero grad, forward, step
    /// - Parameters:
    ///   - fullReset: If true, resets all memory (state, scratch cells) before each step for determinism.
    ///                If false (default), only zeros gradient accumulators.
    /// - Returns: Mean loss from this step
    @discardableResult
    public func trainStep(fullReset: Bool = false) -> Float {
        if fullReset {
            resetState()
        }
        zeroGrad()
        let loss = forward()
        step()
        return loss
    }

    /// Get current parameter values
    public func getParameterValues() -> [Float] {
        return parameters.map { $0.value }
    }

    /// Get current gradients
    public func getGradients() -> [Float] {
        return parameters.map { $0.grad }
    }

    /// Configure memory before forward pass (e.g., set input values)
    public func configureMemory(_ configure: (UnsafeMutablePointer<Float>) -> Void) {
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        configure(memPtr)
    }

    /// Get the output buffer after forward pass
    public func getOutputs() -> [Float] {
        return runtime?.getOutputBuffer() ?? []
    }

    /// Get memory statistics for debugging
    public func getMemoryStats(maxElements: Int = 10000) -> (min: Float, max: Float, mean: Float, nonZeroCount: Int, totalCount: Int) {
        guard let runtime = runtime,
              let memBuffer = runtime.getBuffer(name: "memory") else {
            return (0, 0, 0, 0, 0)
        }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        let totalCount = min(memBuffer.length / MemoryLayout<Float>.size, maxElements)

        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude
        var sum: Float = 0
        var nonZeroCount = 0

        for i in 0..<totalCount {
            let val = memPtr[i]
            if !val.isNaN && !val.isInfinite {
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
                sum += val
                if val != 0 { nonZeroCount += 1 }
            }
        }

        let mean = totalCount > 0 ? sum / Float(totalCount) : 0
        return (minVal, maxVal, mean, nonZeroCount, totalCount)
    }

    /// Get tensor parameter gradients (for debugging)
    public func getTensorGradients() -> [Float] {
        return tensorParameters.flatMap { $0.grads }
    }

    /// Debug: print tensor physical cell info
    public func debugTensorCells() {
        print("Tensor physical cells:")
        for (i, tp) in tensorParameters.enumerated() {
            let base = tensorPhysicalCells[i].base
            let gradBase = tensorGradPhysicalCells[i].base
            print("  \(tp.name ?? "tensor\(i)"): base=\(base), size=\(tp.size), gradBase=\(gradBase)")
        }
    }

    /// Debug: read memory at specific location
    public func debugReadMemory(at offset: Int, count: Int = 5) -> [Float] {
        guard let runtime = runtime,
              let memBuffer = runtime.getBuffer(name: "memory") else { return [] }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        return (0..<count).map { memPtr[offset + $0] }
    }
}

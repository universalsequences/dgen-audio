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
}

/// Simple SGD optimizer
public struct GraphSGD: GraphOptimizer {
    public init() {}

    public mutating func update(parameters: inout [GraphParameter], learningRate: Float) {
        for param in parameters {
            param.value -= learningRate * param.grad
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

    public init(beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8) {
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    public mutating func update(parameters: inout [GraphParameter], learningRate: Float) {
        if m.isEmpty {
            m = Array(repeating: 0.0, count: parameters.count)
            v = Array(repeating: 0.0, count: parameters.count)
        }

        t += 1

        for i in 0..<parameters.count {
            let grad = parameters[i].grad

            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad

            let mHat = m[i] / (1 - pow(beta1, Float(t)))
            let vHat = v[i] / (1 - pow(beta2, Float(t)))

            parameters[i].value -= learningRate * mHat / (sqrt(vHat) + epsilon)
        }
    }
}

// MARK: - Graph Training Context

/// Manages training using graph-based gradient computation
public class GraphTrainingContext {
    public var parameters: [GraphParameter]
    private var optimizer: GraphOptimizer
    public let learningRate: Float

    private var graph: Graph
    private let lossNode: NodeID
    private var runtime: MetalCompiledKernel?
    private var cellAllocations: CellAllocations?
    private var frameCount: Int

    // Physical cell indices for fast access
    private var paramPhysicalCells: [Int] = []
    private var gradPhysicalCells: [Int] = []

    /// Initialize training context
    /// - Parameters:
    ///   - graph: The computation graph (will be modified to add gradient nodes)
    ///   - loss: The loss node to compute gradients from
    ///   - parameters: Trainable parameters
    ///   - optimizer: Optimization algorithm
    ///   - learningRate: Learning rate for optimization
    ///   - frameCount: Number of frames per training step
    public init(
        graph: Graph,
        loss: NodeID,
        parameters: [GraphParameter],
        optimizer: GraphOptimizer = GraphSGD(),
        learningRate: Float = 0.001,
        frameCount: Int = 64
    ) throws {
        self.graph = graph
        self.lossNode = loss
        self.parameters = parameters
        self.optimizer = optimizer
        self.learningRate = learningRate
        self.frameCount = frameCount

        // Prepare gradients and compile
        try prepareAndCompile()
    }

    /// Prepare gradients and compile the graph
    private func prepareAndCompile() throws {
        // Record the last forward node ID before adding gradient nodes
        // This ensures gradient nodes don't affect forward node ordering
        let lastForwardNodeId = graph.nodes.keys.max() ?? 0
        graph.lastForwardNodeId = lastForwardNodeId

        // Step 1: Compute gradient nodes for all parameters
        let targetNodes = Set(parameters.map { $0.nodeId })
        let gradients = graph.computeGradients(loss: lossNode, targets: targetNodes)

        // Step 2: For each parameter, create atomic accumulator for its gradient
        // Chain with any side-effect nodes (like gradient carry writes) to ensure they execute
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

        // Clear side effects after use
        graph.gradientSideEffects = []

        // Step 3: Compile the graph
        let result = try CompilationPipeline.compile(
            graph: graph,
            backend: .metal,
            options: .init(frameCount: frameCount)
        )

        self.cellAllocations = result.cellAllocations

        // Step 4: Create runtime
        self.runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Step 5: Cache physical cell indices
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

        // Step 6: Initialize parameter values in memory
        initializeMemory()
    }

    /// Initialize parameter values in GPU memory
    private func initializeMemory() {
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        for (i, param) in parameters.enumerated() {
            memPtr[paramPhysicalCells[i]] = param.value
        }
    }

    /// Zero the gradient accumulators
    public func zeroGrad() {
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        for i in 0..<parameters.count {
            if gradPhysicalCells[i] >= 0 {
                memPtr[gradPhysicalCells[i]] = 0.0
            }
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

            for (i, param) in parameters.enumerated() {
                if gradPhysicalCells[i] >= 0 {
                    param.grad = memPtr[gradPhysicalCells[i]]
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
        // Update parameters using optimizer
        optimizer.update(parameters: &parameters, learningRate: learningRate)

        // Write updated values back to GPU memory
        guard let runtime = runtime else { return }
        guard let memBuffer = runtime.getBuffer(name: "memory") else { return }

        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        for (i, param) in parameters.enumerated() {
            memPtr[paramPhysicalCells[i]] = param.value
        }
    }

    /// Run a complete training step: zero grad, forward, step
    /// - Returns: Mean loss from this step
    @discardableResult
    public func trainStep() -> Float {
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
}

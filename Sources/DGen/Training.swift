import Foundation

// MARK: - Parameter

/// A learnable parameter in the computation graph
/// Wraps a CellID and tracks its value, gradients, and optimizer state
public class Parameter {
    let cellId: CellID
    let nodeId: NodeID
    public var value: Float
    let name: String?

    // Internal: set by TrainingContext after compilation
    var gradId: GradID?

    // Gradient value (updated after each backward pass)
    public var grad: Float?

    public init(graph: Graph, value: Float, name: String? = nil) {
        self.cellId = graph.alloc()
        self.value = value
        self.name = name
        self.nodeId = graph.n(.param(cellId))
        self.grad = 0.0
    }

    public func node() -> NodeID {
        return nodeId
    }
}

// MARK: - TensorParameter

/// A learnable tensor parameter in the computation graph
/// Each element has its own gradient, allocated contiguously for efficient access
public class TensorParameter {
    let tensorId: TensorID
    let cellId: CellID
    let nodeId: NodeID
    public let shape: Shape
    public var data: [Float]
    let name: String?
    var baseGradId: GradID?

    /// Total number of elements in the tensor
    public var size: Int { shape.reduce(1, *) }

    /// Per-element gradients (updated after each backward pass)
    public var grads: [Float]

    public init(graph: Graph, shape: Shape, data: [Float]? = nil, name: String? = nil) {
        self.shape = shape
        self.name = name
        let totalSize = shape.reduce(1, *)

        // Initialize data with Xavier initialization if not provided
        if let providedData = data {
            self.data = providedData
            precondition(providedData.count == totalSize, "Data count must match shape size")
        } else {
            // Xavier initialization: N(0, sqrt(2 / (fan_in + fan_out)))
            // For simplicity, use sqrt(1/n) where n is size
            let stddev = sqrt(1.0 / Float(totalSize))
            self.data = (0..<totalSize).map { _ in
                // Box-Muller transform for normal distribution
                let u1 = Float.random(in: 0..<1)
                let u2 = Float.random(in: 0..<1)
                return stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            }
        }

        self.grads = [Float](repeating: 0.0, count: totalSize)

        // Allocate tensor in graph
        self.cellId = graph.alloc(vectorWidth: totalSize)
        self.tensorId = graph.nextTensorId
        graph.nextTensorId += 1
        graph.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId, data: self.data)
        graph.cellToTensor[cellId] = tensorId

        // Create tensorRef node
        self.nodeId = graph.n(.tensorRef(tensorId), [], shape: .tensor(shape))
        graph.nodeToTensor[nodeId] = tensorId
    }

    public func node() -> NodeID {
        return nodeId
    }
}

// MARK: - Optimizer Protocol

public protocol Optimizer {
    mutating func step(parameters: inout [Parameter], gradients: [Float])
    mutating func stepTensor(tensorParams: inout [TensorParameter], gradients: [[Float]])
    func zeroGrad()
}

// MARK: - SGD Optimizer

public struct SGD: Optimizer {
    public let learningRate: Float

    public init(lr: Float = 0.01) {
        self.learningRate = lr
    }

    public mutating func step(parameters: inout [Parameter], gradients: [Float]) {
        for i in 0..<parameters.count {
            parameters[i].value -= learningRate * gradients[i]
        }
    }

    public mutating func stepTensor(tensorParams: inout [TensorParameter], gradients: [[Float]]) {
        for i in 0..<tensorParams.count {
            for j in 0..<tensorParams[i].data.count {
                tensorParams[i].data[j] -= learningRate * gradients[i][j]
            }
        }
    }

    public func zeroGrad() {
        // No state to clear in SGD
    }
}

// MARK: - Adam Optimizer

public struct Adam: Optimizer {
    public let learningRate: Float
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float

    private var m: [Float] = []  // First moment
    private var v: [Float] = []  // Second moment
    private var t: Int = 0  // Timestep

    public init(lr: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8)
    {
        self.learningRate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    public mutating func step(parameters: inout [Parameter], gradients: [Float]) {
        // Initialize momentum buffers on first call
        if m.isEmpty {
            m = Array(repeating: 0.0, count: gradients.count)
            v = Array(repeating: 0.0, count: gradients.count)
        }

        t += 1

        for i in 0..<parameters.count {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i]

            // Compute bias-corrected first moment estimate
            let mHat = m[i] / (1 - pow(beta1, Float(t)))

            // Compute bias-corrected second raw moment estimate
            let vHat = v[i] / (1 - pow(beta2, Float(t)))

            // Update parameters
            parameters[i].value -= learningRate * mHat / (sqrt(vHat) + epsilon)
        }
    }

    // Separate momentum buffers for tensor parameters
    private var tensorM: [[Float]] = []
    private var tensorV: [[Float]] = []

    public mutating func stepTensor(tensorParams: inout [TensorParameter], gradients: [[Float]]) {
        // Initialize tensor momentum buffers on first call
        if tensorM.isEmpty {
            tensorM = tensorParams.map { [Float](repeating: 0.0, count: $0.size) }
            tensorV = tensorParams.map { [Float](repeating: 0.0, count: $0.size) }
        }

        t += 1

        for i in 0..<tensorParams.count {
            for j in 0..<tensorParams[i].data.count {
                let grad = gradients[i][j]

                // Update biased first moment estimate
                tensorM[i][j] = beta1 * tensorM[i][j] + (1 - beta1) * grad

                // Update biased second raw moment estimate
                tensorV[i][j] = beta2 * tensorV[i][j] + (1 - beta2) * grad * grad

                // Compute bias-corrected first moment estimate
                let mHat = tensorM[i][j] / (1 - pow(beta1, Float(t)))

                // Compute bias-corrected second raw moment estimate
                let vHat = tensorV[i][j] / (1 - pow(beta2, Float(t)))

                // Update parameters
                tensorParams[i].data[j] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }

    public func zeroGrad() {
        // Momentum is persistent across steps - don't clear
    }

    // GPU support methods
    public func getTimestep() -> Int {
        return t
    }

    public mutating func incrementTimestep() {
        t += 1
    }
}

// MARK: - Training Context

/// Manages learnable parameters, memory, and optimization during training
public class TrainingContext {
    // Debug toggle for gradient prints; can be enabled via env DGEN_DEBUG_GRADS=1
    private let debugGradients: Bool =
        (ProcessInfo.processInfo.environment["DGEN_DEBUG_GRADS"] == "1")
    public var parameters: [Parameter]
    public var tensorParameters: [TensorParameter]
    private var optimizer: Optimizer
    private var memory: UnsafeMutableRawPointer?
    private var cellAllocations: CellAllocations?
    private var runtime: MetalCompiledKernel?
    private var frameCount: Int = 0
    private var context: IRContext?
    private let lossNode: NodeID?
    // Cache physical cell indices for parameters to avoid recomputing
    private var paramPhysicalCells: [UInt32] = []
    // Cache physical cell indices for tensor parameters
    private var tensorParamPhysicalCells: [[UInt32]] = []
    // Profiling
    private let profile: Bool = (ProcessInfo.processInfo.environment["DGEN_PROFILE"] == "1")
    private let profileEvery: Int =
        Int(ProcessInfo.processInfo.environment["DGEN_PROFILE_EVERY"] ?? "1") ?? 1
    private var stepCounter: Int = 0
    private struct StepProfile {
        var zero: Double
        var fwdBwd: Double
        var reduce: Double
        var update: Double
        var readback: Double
        var syncHost: Double
        var total: Double
    }
    private var lastProfile: StepProfile?

    /// Initialize training context (simple version - requires manual initializeMemory() call)
    /// - Parameters:
    ///   - parameters: Learnable parameters to optimize
    ///   - tensorParameters: Learnable tensor parameters to optimize
    ///   - optimizer: Optimization algorithm (SGD, Adam, etc.)
    ///   - lossNode: The loss node to seed gradients from (optional, can be inferred if only one output)
    public init(parameters: [Parameter] = [], tensorParameters: [TensorParameter] = [], optimizer: Optimizer, lossNode: NodeID? = nil) {
        self.parameters = parameters
        self.tensorParameters = tensorParameters
        self.optimizer = optimizer
        self.lossNode = lossNode
    }

    /// Initialize training context with all dependencies (streamlined version)
    /// - Parameters:
    ///   - parameters: Learnable parameters to optimize
    ///   - tensorParameters: Learnable tensor parameters to optimize
    ///   - optimizer: Optimization algorithm (SGD, Adam, etc.)
    ///   - lossNode: The loss node to seed gradients from
    ///   - compilationResult: Result from CompilationPipeline.compile()
    ///   - frameCount: Number of audio frames per batch
    public convenience init(
        parameters: [Parameter] = [],
        tensorParameters: [TensorParameter] = [],
        optimizer: Optimizer,
        lossNode: NodeID,
        compilationResult: CompilationResult,
        frameCount: Int
    ) throws {
        self.init(parameters: parameters, tensorParameters: tensorParameters, optimizer: optimizer, lossNode: lossNode)

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: compilationResult.kernels,
            cellAllocations: compilationResult.cellAllocations,
            context: compilationResult.context
        )

        // Initialize memory automatically
        self.initializeMemory(
            runtime: runtime,
            cellAllocations: compilationResult.cellAllocations,
            context: compilationResult.context,
            frameCount: frameCount
        )
    }

    /// Initialize memory and set up parameter mappings after compilation
    /// Must be called before first forward pass
    public func initializeMemory(
        runtime: MetalCompiledKernel,
        cellAllocations: CellAllocations,
        context: IRContext,
        frameCount: Int
    ) {
        self.runtime = runtime
        self.cellAllocations = cellAllocations
        self.frameCount = frameCount
        self.context = context

        // Mark loss node as seed gradient if specified
        if let lossNode = lossNode {
            // Ensure the loss node is marked as a seed, even if a gradient ID
            // was already allocated during codegen.
            if let existing = context.gradients[lossNode] {
                if !context.seedGradients.contains(existing) {
                    context.seedGradients.append(existing)
                }
            } else {
                let _ = context.useGradient(src: lossNode, seed: true)
            }
        }

        // Verify we have seed gradients (loss nodes)
        guard !context.seedGradients.isEmpty else {
            fatalError(
                """
                No seed gradients found!

                You must either:
                1. Pass lossNode to TrainingContext init:
                   TrainingContext(parameters: [...], optimizer: Adam(), lossNode: loss)

                2. Or manually mark it before compilation:
                   let _ = context.useGradient(src: loss, seed: true)
                """)
        }

        print("🌱 Found \(context.seedGradients.count) seed gradient(s): \(context.seedGradients)")

        // Allocate persistent memory
        self.memory = runtime.allocateNodeMemory()
        guard let memory = self.memory else {
            fatalError("Failed to allocate node memory")
        }

        let memPtr = memory.assumingMemoryBound(to: Float.self)

        // Initialize parameter values in host memory and cache physical cells
        paramPhysicalCells.removeAll(keepingCapacity: true)
        for param in parameters {
            let physicalCell = cellAllocations.cellMappings[param.cellId] ?? param.cellId
            memPtr[physicalCell] = param.value
            paramPhysicalCells.append(UInt32(physicalCell))

            // Look up and store the GradID for this parameter
            param.gradId = context.gradients[param.nodeId]
        }

        // Verify all parameters have gradients
        for param in parameters {
            guard param.gradId != nil else {
                let name = param.name ?? "unnamed"
                fatalError(
                    "Parameter '\(name)' (node \(param.nodeId)) has no gradient. Did you set backwards: true?"
                )
            }
        }

        // Initialize tensor parameter values in host memory and cache physical cells
        tensorParamPhysicalCells.removeAll(keepingCapacity: true)
        for tensorParam in tensorParameters {
            let physicalCell = cellAllocations.cellMappings[tensorParam.cellId] ?? tensorParam.cellId
            var cells: [UInt32] = []

            // Inject tensor data into memory
            for i in 0..<tensorParam.size {
                memPtr[physicalCell + i] = tensorParam.data[i]
                cells.append(UInt32(physicalCell + i))
            }
            tensorParamPhysicalCells.append(cells)

            // Look up and store the base GradID for this tensor parameter
            tensorParam.baseGradId = context.tensorGradients[tensorParam.nodeId]
        }

        // Verify all tensor parameters have gradients if backwards was enabled
        for tensorParam in tensorParameters {
            if tensorParam.baseGradId == nil {
                let name = tensorParam.name ?? "unnamed"
                print("Warning: TensorParameter '\(name)' (node \(tensorParam.nodeId)) has no tensor gradient allocated yet")
            }
        }

        // Also write the initial parameter values directly into the device memory buffer
        // so that GPU runs don't need to copy the entire host memory every step.
        if let rt = self.runtime {
            let values: [Float] = parameters.map { $0.value }
            rt.writeParameters(physicalCells: paramPhysicalCells, values: values)
        }
    }

    /// Extract and reduce gradients from the gradients buffer
    /// Gradients are stored as: gradients[frameCount * gradId + frameIndex]
    /// We sum across all frames to get the total gradient
    public func extractGradients() -> [Float] {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        let gradPtr = runtime.getGradientsBuffer()

        // DEBUG: Check if seed gradient is still 1.0 after backward pass
        if let firstSeed = context?.seedGradients.first {
            let seedIdx = frameCount * firstSeed

            // Check all gradIds to see where gradients are
            for gradId in 0...7 {
                let idx = frameCount * gradId
                var sum: Float = 0.0
                for i in 0..<min(4, frameCount) {
                    sum += gradPtr[idx + i]
                }
            }
        }

        return parameters.map { param in
            guard let gradId = param.gradId else {
                fatalError("Parameter has no gradient ID")
            }

            // Sum gradients across all frames
            var totalGrad: Float = 0.0
            let baseIndex = frameCount * gradId

            for frameIndex in 0..<frameCount {
                totalGrad += gradPtr[baseIndex + frameIndex]
            }

            // Use mean instead of sum (average over frames)
            let meanGrad = totalGrad / Float(frameCount)

            return meanGrad
        }
    }

    /// Extract and reduce gradients from the gradients buffer for tensor parameters
    /// Gradients are stored as: gradients[frameCount * (baseGradId + elementIndex) + frameIndex]
    /// We sum across all frames to get the total gradient per element
    public func extractTensorGradients() -> [[Float]] {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        let gradPtr = runtime.getGradientsBuffer()

        return tensorParameters.map { tensorParam in
            guard let baseGradId = tensorParam.baseGradId else {
                // Return zeros if no gradient allocated
                return [Float](repeating: 0.0, count: tensorParam.size)
            }

            var grads = [Float](repeating: 0.0, count: tensorParam.size)
            for elementIndex in 0..<tensorParam.size {
                let gradId = baseGradId + elementIndex
                let baseIndex = frameCount * gradId

                // Sum gradients across all frames
                var totalGrad: Float = 0.0
                for frameIndex in 0..<frameCount {
                    totalGrad += gradPtr[baseIndex + frameIndex]
                }

                // Use mean instead of sum (average over frames)
                grads[elementIndex] = totalGrad / Float(frameCount)
            }

            return grads
        }
    }

    /// Zero out all gradients and reset memory before a backward pass
    /// - Parameter deviceMemory: when true, clears the device `memory` buffer and restores
    ///   parameter values without touching host-side memory. Defaults to false for CPU paths.
    public func zeroGrad(deviceMemory: Bool = false) {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        runtime.resetGradientBuffers(numFrames: frameCount)

        // Zero grad_memory buffer if it exists (used for certain back propagation)
        if let gradMemBuffer = runtime.getBuffer(name: "grad_memory") {
            let memorySize = runtime.getMemorySize()
            memset(gradMemBuffer.contents(), 0, memorySize * MemoryLayout<Float>.size)
        }

        // Zero parameter gradients
        for param in parameters {
            param.grad = 0.0
        }

        // Zero tensor parameter gradients
        for tensorParam in tensorParameters {
            for i in 0..<tensorParam.grads.count {
                tensorParam.grads[i] = 0.0
            }
        }

        optimizer.zeroGrad()

        // Reset compute memory for the next forward pass, but preserve parameter values.
        if deviceMemory {
            // Operate directly on the device `memory` buffer to avoid host copies.
            let values: [Float] = parameters.map { $0.value }
            runtime.clearMemoryPreservingParameters(
                physicalCells: paramPhysicalCells,
                values: values
            )
        } else if let mem = self.memory, let runtime = self.runtime,
            let cellAlloc = self.cellAllocations
        {
            // CPU path: zero host-side memory and restore parameter values; runWithMemory
            // will copy this into the device buffer on the next forward pass.
            let memorySize = runtime.getMemorySize()
            memset(mem, 0, memorySize * MemoryLayout<Float>.size)

            // Restore parameter values into their physical cells
            let memPtr = mem.assumingMemoryBound(to: Float.self)
            for param in parameters {
                let physicalCell = cellAlloc.cellMappings[param.cellId] ?? param.cellId
                memPtr[physicalCell] = param.value
            }

            // Restore tensor parameter values into their physical cells
            for tensorParam in tensorParameters {
                let physicalCell = cellAlloc.cellMappings[tensorParam.cellId] ?? tensorParam.cellId
                for i in 0..<tensorParam.size {
                    memPtr[physicalCell + i] = tensorParam.data[i]
                }
            }
        }
    }

    /// Update parameters using gradients and optimizer (CPU version)
    public func step() {
        // Extract gradients from gradient buffer
        let gradients = extractGradients()

        // Store gradients in parameters for access via param.grad
        for (i, param) in parameters.enumerated() {
            param.grad = gradients[i]
        }

        // Update parameters via optimizer
        optimizer.step(parameters: &parameters, gradients: gradients)

        // Extract and apply tensor gradients
        let tensorGradients = extractTensorGradients()
        for (i, tensorParam) in tensorParameters.enumerated() {
            tensorParam.grads = tensorGradients[i]
        }
        if !tensorParameters.isEmpty {
            optimizer.stepTensor(tensorParams: &tensorParameters, gradients: tensorGradients)
        }

        // Write updated values back to host memory that is passed to runWithMemory.
        // runWithMemory will copy this host memory into the Metal buffer at call time.
        guard let mem = self.memory, let cellAlloc = self.cellAllocations else {
            fatalError("Memory not initialized")
        }
        let memPtr = mem.assumingMemoryBound(to: Float.self)
        for param in parameters {
            let physicalCell = cellAlloc.cellMappings[param.cellId] ?? param.cellId
            memPtr[physicalCell] = param.value
        }

        // Write updated tensor parameter values to host memory
        for tensorParam in tensorParameters {
            let physicalCell = cellAlloc.cellMappings[tensorParam.cellId] ?? tensorParam.cellId
            for i in 0..<tensorParam.size {
                memPtr[physicalCell + i] = tensorParam.data[i]
            }
        }

        // Also mirror into the Metal buffer now so any debug reads of `memory` buffer
        // reflect the latest parameter values before the next runWithMemory call.
        if let rt = self.runtime, let memBuffer = rt.getBuffer(name: "memory") {
            let gpuMemPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
            for param in parameters {
                let physicalCell = cellAlloc.cellMappings[param.cellId] ?? param.cellId
                gpuMemPtr[physicalCell] = param.value
            }
            // Mirror tensor parameters too
            for tensorParam in tensorParameters {
                let physicalCell = cellAlloc.cellMappings[tensorParam.cellId] ?? tensorParam.cellId
                for i in 0..<tensorParam.size {
                    gpuMemPtr[physicalCell + i] = tensorParam.data[i]
                }
            }
        }
    }

    /// Update parameters using GPU kernels for gradient reduction and parameter updates
    public func stepGPU() {
        guard let runtime = runtime, let context = context, let cellAlloc = cellAllocations else {
            fatalError("Runtime not initialized")
        }
        let profEnabled = profile
        var tStart = CFAbsoluteTimeGetCurrent()
        var tReduce: Double = 0
        var tUpdate: Double = 0
        var tReadback: Double = 0
        var tSyncHost: Double = 0

        // Pre-reduction debug: inspect raw per-frame gradients for all gradIds
        if debugGradients, let gradientsBuffer = runtime.getBuffer(name: "gradients") {
            let gptr = gradientsBuffer.contents().assumingMemoryBound(to: Float.self)
            let fc = frameCount
            let numGradIds = context.maxGradId + 1
            print("   [DEBUG] Gradients pre-reduce: frameCount=\(fc), gradIds=0..\(numGradIds-1)")
            for gid in 0..<numGradIds {
                let base = gid * fc
                var sum: Float = 0
                var minv: Float = Float.greatestFiniteMagnitude
                var maxv: Float = -Float.greatestFiniteMagnitude
                var anyNaN = false
                var anyInf = false
                for i in 0..<fc {
                    let v = gptr[base + i]
                    if v.isNaN { anyNaN = true }
                    if !v.isFinite { anyInf = true }
                    if v < minv { minv = v }
                    if v > maxv { maxv = v }
                    sum += v
                }
                let previewCount = min(8, fc)
                var preview: [String] = []
                for i in 0..<previewCount { preview.append(String(format: "%.3e", gptr[base + i])) }
                let mean = sum / Float(fc)
                print(
                    "   [DEBUG] gid=\(gid) min=\(minv) max=\(maxv) mean=\(mean) NaN=\(anyNaN) Inf=\(anyInf) first=\(preview)"
                )
            }
        }

        // Step 1: Reduce gradients on GPU
        let numGradIds = context.maxGradId + 1
        guard runtime.reduceGradientsGPU(frameCount: frameCount, numGradIds: numGradIds) != nil
        else {
            print("⚠️ GPU gradient reduction failed, falling back to CPU")
            step()
            return
        }
        if profEnabled {
            tReduce = CFAbsoluteTimeGetCurrent() - tStart
            tStart = CFAbsoluteTimeGetCurrent()
        }

        // Debug: peek reduced gradient values for our params
        /*
        if let reducedGradsBuffer = runtime.getBuffer(name: "reducedGrads") {
            let reduced = reducedGradsBuffer.contents().assumingMemoryBound(to: Float.self)
            for (idx, p) in parameters.enumerated() {
                if let gid = p.gradId {
                    let val = reduced[gid]
                    print("   [DEBUG] param[\(idx)] gradId=\(gid) reducedGrad=\(val)")
                }
            }
        }

         */

        // Step 2: Build parameter mappings
        var gradIds: [UInt32] = []
        var physicalCells: [UInt32] = []

        for param in parameters {
            guard let gradId = param.gradId else {
                fatalError("Parameter has no gradient ID")
            }
            let physicalCell = cellAlloc.cellMappings[param.cellId] ?? param.cellId

            gradIds.append(UInt32(gradId))
            physicalCells.append(UInt32(physicalCell))
        }

        // Step 3: Dispatch optimizer-specific GPU kernel
        if var adamOpt = optimizer as? Adam {
            // Use Adam GPU kernel
            runtime.updateParametersAdamGPU(
                gradIds: gradIds,
                physicalCells: physicalCells,
                learningRate: adamOpt.learningRate,
                beta1: adamOpt.beta1,
                beta2: adamOpt.beta2,
                epsilon: adamOpt.epsilon,
                timestep: adamOpt.getTimestep() + 1
            )

            // Increment Adam timestep
            adamOpt.incrementTimestep()
            optimizer = adamOpt
        } else if let sgdOpt = optimizer as? SGD {
            // Use SGD GPU kernel
            runtime.updateParametersSGDGPU(
                gradIds: gradIds,
                physicalCells: physicalCells,
                learningRate: sgdOpt.learningRate
            )
        } else {
            print("⚠️ GPU optimization not supported for this optimizer, falling back to CPU")
            step()
            return
        }
        if profEnabled {
            tUpdate = CFAbsoluteTimeGetCurrent() - tStart
            tStart = CFAbsoluteTimeGetCurrent()
        }

        // Step 4: Read back parameter values from GPU memory buffer
        if let memBuffer = runtime.getBuffer(name: "memory") {
            let gpuMemPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
            for param in parameters {
                let physicalCell = cellAlloc.cellMappings[param.cellId] ?? param.cellId
                param.value = gpuMemPtr[physicalCell]
            }
        }
        if profEnabled {
            tReadback = CFAbsoluteTimeGetCurrent() - tStart
            tStart = CFAbsoluteTimeGetCurrent()
        }

        // Step 5: Sync to host memory
        guard let mem = self.memory else {
            fatalError("Memory not initialized")
        }
        let memPtr = mem.assumingMemoryBound(to: Float.self)
        for param in parameters {
            let physicalCell = cellAlloc.cellMappings[param.cellId] ?? param.cellId
            memPtr[physicalCell] = param.value
        }
        if profEnabled { tSyncHost = CFAbsoluteTimeGetCurrent() - tStart }

        if profEnabled {
            // Store last profile; zero and fwdBwd are captured in runStepGPU()
            let prev =
                lastProfile
                ?? StepProfile(
                    zero: 0, fwdBwd: 0, reduce: 0, update: 0, readback: 0, syncHost: 0, total: 0)
            lastProfile = StepProfile(
                zero: prev.zero,
                fwdBwd: prev.fwdBwd,
                reduce: tReduce,
                update: tUpdate,
                readback: tReadback,
                syncHost: tSyncHost,
                total: 0
            )
        }

        // Step 6: Update param.grad for inspection (read from reducedGrads buffer)
        if let reducedGradsBuffer = runtime.getBuffer(name: "reducedGrads") {
            let reducedGradsPtr = reducedGradsBuffer.contents().assumingMemoryBound(to: Float.self)
            for param in parameters {
                guard let gradId = param.gradId else { continue }
                param.grad = reducedGradsPtr[gradId]
                if debugGradients {
                    print("   [DEBUG] param gradId=\(gradId) reducedGrad=\(param.grad)")
                }
            }
        }
    }

    /// Get the memory pointer for passing to runWithMemory
    public func getMemory() -> UnsafeMutableRawPointer {
        guard let memory = memory else {
            fatalError("Memory not initialized. Call initializeMemory() first.")
        }
        return memory
    }

    /// Run a complete training step: zero gradients, forward+backward pass, optimizer step (CPU)
    /// - Returns: The loss value from this step
    public func runStep() -> Float {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        // Zero gradients
        zeroGrad()

        // Forward + backward pass - use simplified API!
        runtime.run(memory: getMemory(), frameCount: frameCount)

        // Update parameters
        step()

        // Return loss value (last frame)
        return runtime.getLastOutput() ?? 0.0
    }

    /// Run a complete training step using GPU kernels for optimization
    /// - Returns: The loss value from this step
    public func runStepGPU() -> Float {
        guard let runtime = runtime else {
            fatalError("Runtime not initialized")
        }

        // Zero gradients and reset device memory
        let profEnabled = profile
        var t0 = CFAbsoluteTimeGetCurrent()
        zeroGrad(deviceMemory: true)
        var zeroTime: Double = 0
        var fwdTime: Double = 0
        if profEnabled {
            zeroTime = CFAbsoluteTimeGetCurrent() - t0
            t0 = CFAbsoluteTimeGetCurrent()
        }

        // Forward + backward pass
        runtime.runNoCopy(frameCount: frameCount)
        if profEnabled { fwdTime = CFAbsoluteTimeGetCurrent() - t0 }

        // Update parameters using GPU
        stepGPU()

        if profEnabled {
            // Merge timings from stepGPU
            if var prof = lastProfile {
                prof.zero = zeroTime
                prof.fwdBwd = fwdTime
                prof.total =
                    zeroTime + fwdTime + prof.reduce + prof.update + prof.readback + prof.syncHost
                lastProfile = prof
                stepCounter += 1
                if stepCounter % max(profileEvery, 1) == 0 {
                    let ms = { (s: Double) -> String in String(format: "%.2f", s * 1000.0) }
                    print(
                        "[PROFILE] step=\(stepCounter) total=\(ms(prof.total)) ms | zero=\(ms(prof.zero)) fwd+bwd=\(ms(prof.fwdBwd)) reduce=\(ms(prof.reduce)) update=\(ms(prof.update)) readback=\(ms(prof.readback)) syncHost=\(ms(prof.syncHost))"
                    )
                }
            }
        }

        // Return loss value (last frame)
        return runtime.getLastOutput() ?? 0.0
    }

    deinit {
        if let mem = memory, let rt = runtime {
            rt.deallocateNodeMemory(mem)
        }
    }
}

// MARK: - Metal Backend Extensions

extension MetalCompiledKernel {
    /// Get direct access to the gradients buffer for parameter updates
    public func getGradientsBuffer() -> UnsafeMutablePointer<Float> {
        guard let buffer = getBuffer(name: "gradients") else {
            fatalError("No gradients buffer found. Ensure backwards: true in compilation options.")
        }
        return buffer.contents().assumingMemoryBound(to: Float.self)
    }
}

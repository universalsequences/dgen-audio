// Realize - Compilation and execution of lazy graphs
//
// Implements realize() which compiles the lazy graph and runs it on Metal/CPU,
// returning the computed values.

import Foundation
import DGen

// MARK: - LazyRuntime Protocol

/// Abstraction over Metal and C runtimes for realize() execution
public protocol LazyRuntime: AnyObject {
    var cellAllocations: CellAllocations { get }
    func zeroAllBuffers()
    func runNoCopy(frameCount: Int)
    func memoryPointer() -> UnsafeMutablePointer<Float>?
    func outputsPointer() -> UnsafeMutablePointer<Float>?
}

extension MetalCompiledKernel: LazyRuntime {
    public func memoryPointer() -> UnsafeMutablePointer<Float>? {
        getBuffer(name: "memory")?.contents().assumingMemoryBound(to: Float.self)
    }

    public func outputsPointer() -> UnsafeMutablePointer<Float>? {
        getBuffer(name: "outputs")?.contents().assumingMemoryBound(to: Float.self)
    }
}

/// C backend runtime wrapper for lazy execution
public class CLazyRuntime: LazyRuntime {
    public let cellAllocations: CellAllocations
    private let kernel: CCompiledKernel
    private let memory: UnsafeMutablePointer<Float>
    private let outputs: UnsafeMutablePointer<Float>
    private let memorySize: Int
    private let outputsSize: Int

    public init(kernels: [CompiledKernel], cellAllocations: CellAllocations, memorySize: Int, frameCount: Int) throws {
        self.cellAllocations = cellAllocations
        self.memorySize = max(memorySize, 1024)
        self.outputsSize = frameCount

        // Concatenate all kernel sources into a single C file
        let combinedSource = kernels.map { $0.source }.joined(separator: "\n\n")
        self.kernel = CCompiledKernel(
            source: combinedSource,
            cellAllocations: cellAllocations,
            memorySize: self.memorySize
        )

        self.memory = .allocate(capacity: self.memorySize)
        self.memory.initialize(repeating: 0, count: self.memorySize)
        self.outputs = .allocate(capacity: self.outputsSize)
        self.outputs.initialize(repeating: 0, count: self.outputsSize)

        try kernel.compileAndLoad()
    }

    public func zeroAllBuffers() {
        memory.initialize(repeating: 0, count: memorySize)
        outputs.initialize(repeating: 0, count: outputsSize)
    }

    public func runNoCopy(frameCount: Int) {
        let inputs = [Float](repeating: 0, count: frameCount)
        inputs.withUnsafeBufferPointer { inBuf in
            kernel.runWithMemory(
                outputs: outputs,
                inputs: inBuf.baseAddress!,
                memory: UnsafeMutableRawPointer(memory),
                frameCount: frameCount
            )
        }
    }

    public func memoryPointer() -> UnsafeMutablePointer<Float>? { memory }
    public func outputsPointer() -> UnsafeMutablePointer<Float>? { outputs }

    deinit {
        memory.deallocate()
        outputs.deallocate()
        kernel.cleanup()
    }
}

// MARK: - Execution Context

/// Stores compilation and runtime state for a realized graph
public class ExecutionContext {
    let compilationResult: CompilationResult
    let runtime: LazyRuntime
    let frameCount: Int

    init(compilationResult: CompilationResult, runtime: LazyRuntime, frameCount: Int) {
        self.compilationResult = compilationResult
        self.runtime = runtime
        self.frameCount = frameCount
    }
}

// MARK: - LazyGraph Compilation

extension LazyGraph {
    /// Compile and run the graph, returning the execution context
    func compile(frameCount: Int) throws -> ExecutionContext {
        // Use cached compilation if available and not dirty
        if !isDirty, let cached = compilationCache, let runtime = runtimeCache {
            return ExecutionContext(compilationResult: cached, runtime: runtime, frameCount: frameCount)
        }

        // Fast path: check full compilation cache using a lightweight graph fingerprint.
        // Graph is a class (reference type) — the cached CompilationResult.graph points to
        // the same live Graph object, so injectTensorData reads current tensor data.
        let fingerprint = "\(graph.nodes.count)|\(graph.tensors.count)|\(frameCount)"
        if let cached = fullCompilationCache, cached.fingerprint == fingerprint {
            compilationCache = cached.result
            runtimeCache = cached.runtime
            isDirty = false
            return ExecutionContext(compilationResult: cached.result, runtime: cached.runtime, frameCount: frameCount)
        }

        // Compile the graph
        let result = try CompilationPipeline.compile(
            graph: graph,
            backend: DGenConfig.backend,
            options: .init(frameCount: frameCount, debug: DGenConfig.debug, enableBufferReuse: DGenConfig.enableBufferReuse)
        )

        // Write kernels to disk if path is configured
        if let outputPath = DGenConfig.kernelOutputPath {
            writeKernelsToDisk(result, outputPath)
        }

        // Check if we have a cached runtime with matching kernel structure.
        // The graph topology is identical each epoch (same model + loss), so the
        // generated MSL is the same. We can reuse the MTLLibrary + pipeline states.
        var hasher = Hasher()
        for kernel in result.kernels {
            hasher.combine(kernel.source)
        }
        let structureKey = String(hasher.finalize())

        let runtime: LazyRuntime
        if let cachedRuntime = runtimeCacheByStructure[structureKey] {
            runtime = cachedRuntime
        } else {
            // Create runtime based on backend
            switch DGenConfig.backend {
            case .metal:
                runtime = try MetalCompiledKernel(
                    kernels: result.kernels,
                    cellAllocations: result.cellAllocations,
                    context: result.context,
                    frameCount: frameCount
                )
            case .c:
                runtime = try CLazyRuntime(
                    kernels: result.kernels,
                    cellAllocations: result.cellAllocations,
                    memorySize: result.totalMemorySlots,
                    frameCount: frameCount
                )
            }
            runtimeCacheByStructure[structureKey] = runtime
        }

        // Store in full compilation cache for subsequent epochs
        fullCompilationCache = (fingerprint: fingerprint, result: result, runtime: runtime)

        // Cache for reuse
        compilationCache = result
        runtimeCache = runtime
        isDirty = false

        return ExecutionContext(compilationResult: result, runtime: runtime, frameCount: frameCount)
    }

    /// Inject tensor data into memory buffer
    func injectTensorData(context: ExecutionContext) {
        if let memPtr = context.runtime.memoryPointer() {
            DGen.injectTensorData(result: context.compilationResult, memory: memPtr)
        }
    }

    /// Inject Signal param values into memory buffer
    func injectSignalParams(context: ExecutionContext) {
        guard let memPtr = context.runtime.memoryPointer() else { return }
        let cellMappings = context.compilationResult.cellAllocations.cellMappings

        for signal in parameterRegistry.signals {
            guard let cellId = signal.cellId,
                  let value = signal.data else { continue }

            // Map logical cell to physical cell
            let physicalCell = cellMappings[cellId] ?? cellId
            memPtr[physicalCell] = value
        }

        // Inject initial values for stateful cells (e.g., click cells)
        for (cellId, value) in cellInitialValues {
            let physicalCell = cellMappings[cellId] ?? cellId
            memPtr[physicalCell] = value
        }
    }

    /// Run the compiled graph
    func run(context: ExecutionContext, preserveState: Bool) {
        if !preserveState {
            context.runtime.zeroAllBuffers()
            injectTensorData(context: context)
            injectSignalParams(context: context)
        }
        context.runtime.runNoCopy(frameCount: context.frameCount)
    }

    /// Read output buffer
    func readOutputs(context: ExecutionContext) -> [Float] {
        var outputs = [Float](repeating: 0, count: context.frameCount)
        if let outPtr = context.runtime.outputsPointer() {
            for i in 0..<context.frameCount {
                outputs[i] = outPtr[i]
            }
        }
        return outputs
    }

    /// Read tensor data from memory buffer, walking the transform chain to map indices
    func readTensorData(context: ExecutionContext, tensorId: TensorID) -> [Float]? {
        guard let tensor = context.compilationResult.graph.tensors[tensorId] else { return nil }

        let physicalCellId = context.compilationResult.cellAllocations.cellMappings[tensor.cellId] ?? tensor.cellId

        guard let memPtr = context.runtime.memoryPointer() else { return nil }

        // Check if this is a frame-aware tensor (allocated as tensorSize * frameCount)
        // frameAwareCells uses the tensor's cellId (before physical remapping)
        let frameCount: Int
        if let frameAwareInfo = context.compilationResult.graph.frameAwareCells[tensor.cellId] {
            frameCount = frameAwareInfo.frameCount
        } else {
            frameCount = 1
        }

        let outputSize = tensor.size * frameCount
        var data = [Float](repeating: 0, count: outputSize)

        // Fast path: no transforms, contiguous base
        if tensor.transforms.isEmpty && tensor.baseStrides == DGen.Tensor.computeRowMajorStrides(tensor.baseShape) {
            for i in 0..<outputSize {
                data[i] = memPtr[physicalCellId + i]
            }
            return data
        }

        // Slow path: walk transform chain for each element
        let shape = tensor.shape
        let baseStrides = tensor.baseStrides
        let transforms = tensor.transforms

        for frame in 0..<frameCount {
            let frameOffset = frame * tensor.size
            let baseFrameOffset = frame * tensor.baseSize

            for flatIdx in 0..<tensor.size {
                // Convert flat index to multi-dimensional indices for output shape
                var indices = flatToMultiIndex(flatIdx, shape)
                var currentShape = shape
                var inBounds = true

                // Walk transforms BACKWARDS
                for transform in transforms.reversed() {
                    if !inBounds { break }

                    switch transform {
                    case .pad(let padding, let inputShape):
                        // Check bounds and adjust indices
                        for (i, pad) in padding.enumerated() {
                            let idx = indices[i]
                            let innerSize = currentShape[i] - pad.left - pad.right
                            if idx < pad.left || idx >= pad.left + innerSize {
                                inBounds = false
                                break
                            }
                            indices[i] = idx - pad.left
                        }
                        currentShape = inputShape

                    case .asStrided(_, let strides, let offset, let inputShape):
                        // Convert strided indices to flat offset, then to input indices
                        var flatOffset = offset
                        for (i, idx) in indices.enumerated() {
                            flatOffset += idx * strides[i]
                        }
                        indices = flatToMultiIndex(flatOffset, inputShape)
                        currentShape = inputShape

                    case .reshape(_, let inputShape):
                        // Linearize then un-linearize
                        let flat = multiIndexToFlat(indices, currentShape)
                        indices = flatToMultiIndex(flat, inputShape)
                        currentShape = inputShape

                    case .transpose(let axes, let inputShape):
                        // Inverse permutation
                        var inverse = [Int](repeating: 0, count: axes.count)
                        for (i, axis) in axes.enumerated() {
                            inverse[axis] = i
                        }
                        indices = inverse.map { indices[$0] }
                        currentShape = inputShape

                    case .shrink(let ranges, let inputShape):
                        // Add start offsets
                        for (i, range) in ranges.enumerated() {
                            if let r = range {
                                indices[i] += r.start
                            }
                        }
                        currentShape = inputShape

                    case .expand(_, let inputShape):
                        // Clamp broadcast dims to 0
                        for (i, dim) in inputShape.enumerated() {
                            if dim == 1 {
                                indices[i] = 0
                            }
                        }
                        currentShape = inputShape

                    case .repeatTile(let innerShape, _):
                        // Modular indexing
                        for (i, dim) in innerShape.enumerated() {
                            indices[i] = indices[i] % dim
                        }
                        currentShape = innerShape

                    case .slidingWindow(let windowSize, let inputShape, let positionNode):
                        let lastDim = indices.count - 1
                        if positionNode != nil {
                            // Circular buffer mode: reconstruct per-frame writePos from accum cell
                            // In single-shot DGenLazy execution, writePos == frame (accum starts at 0)
                            let bufSize = inputShape[lastDim]
                            let pos = frame  // In single-shot mode, position == frame index
                            let raw = pos - windowSize + 1 + indices[lastDim]
                            indices[lastDim] = ((raw % bufSize) + bufSize) % bufSize
                        } else {
                            // Original mode: index i at frame f → base[f - windowSize + 1 + i]
                            let baseIdx = frame - windowSize + 1 + indices[lastDim]
                            if baseIdx < 0 {
                                inBounds = false
                            } else {
                                indices[lastDim] = baseIdx
                            }
                        }
                        currentShape = inputShape
                    }
                }

                var value: Float = 0.0
                if inBounds {
                    // Compute base memory offset
                    var baseOffset = 0
                    for (i, idx) in indices.enumerated() {
                        baseOffset += idx * baseStrides[i]
                    }
                    value = memPtr[physicalCellId + baseFrameOffset + baseOffset]
                }

                data[frameOffset + flatIdx] = value
            }
        }

        return data
    }

    /// Convert flat index to multi-dimensional indices (row-major)
    private func flatToMultiIndex(_ flat: Int, _ shape: [Int]) -> [Int] {
        var indices = [Int](repeating: 0, count: shape.count)
        var remaining = flat
        for i in 0..<shape.count {
            let stride = shape[(i + 1)...].reduce(1, *)
            indices[i] = remaining / stride
            remaining = remaining % stride
        }
        return indices
    }

    /// Convert multi-dimensional indices to flat index (row-major)
    private func multiIndexToFlat(_ indices: [Int], _ shape: [Int]) -> Int {
        var flat = 0
        for i in 0..<shape.count {
            let stride = shape[(i + 1)...].reduce(1, *)
            flat += indices[i] * stride
        }
        return flat
    }
}

// MARK: - Tensor Realize

extension Tensor {
    /// Execute the computation graph and return the tensor data
    ///
    /// ```swift
    /// let t = Tensor([1, 2, 3]) * 2 + 1
    /// let values = try t.realize()  // [3, 5, 7]
    /// ```
    ///
    /// - Returns: Flat array of tensor values in row-major order
    public func realize() throws -> [Float] {
        // If we already have data (e.g., gradient tensors), return it directly
        if let data = getData() {
            return data
        }

        // For static tensors, we use frameCount=1
        let frameCount = 1

        // Add output to drive computation
        if isScalar {
            let _ = graph.node(.output(0), [nodeId])
        } else {
            // Sum to drive computation
            let sumNode = graph.node(.sum, [nodeId])
            let _ = graph.node(.output(0), [sumNode])
        }

        // Mark this node for materialization if it's a tensor
        // allocateTensorOutputs will create a tensor for this node during compilation
        // and materialize=true ensures allocateTensorMemory allocates memory for it
        if !isScalar {
            graph.graph.materializeNodes.insert(nodeId)
        }

        // Compile and run
        let context = try graph.compile(frameCount: frameCount)
        graph.run(context: context, preserveState: false)

        // Read tensor data from memory
        // Look up tensorId from nodeToTensor (created during compilation by allocateTensorOutputs)
        if let tensorId = self.tensorId ?? graph.graph.nodeToTensor[nodeId] {
            if let data = graph.readTensorData(context: context, tensorId: tensorId) {
                return data
            }
        }

        // Fallback for scalar tensors - read from output
        if isScalar {
            return graph.readOutputs(context: context)
        }

        // Should not reach here after the changes above
        throw DGenLazyError.cannotRealizeDerivedTensor
    }
}

// MARK: - Signal Realize

extension Signal {
    /// Execute the computation graph and return the signal samples
    ///
    /// ```swift
    /// let osc = Signal.phasor(440)
    /// let wave = sin(osc * 2 * .pi)
    /// let samples = try wave.realize(frames: 1024)
    /// ```
    ///
    /// - Parameters:
    ///   - frames: Number of frames to compute (default: DGenConfig.defaultFrameCount)
    ///   - preserveState: If true, maintain state across realize() calls (for inference)
    /// - Returns: Array of samples, one per frame
    public func realize(frames: Int = DGenConfig.defaultFrameCount, preserveState: Bool = false) throws -> [Float] {
        // Add output node for this signal
        let _ = graph.node(.output(0), [nodeId])

        // Compile and run
        let context = try graph.compile(frameCount: frames)
        graph.run(context: context, preserveState: preserveState)

        // Read outputs
        return graph.readOutputs(context: context)
    }
}

// MARK: - SignalTensor Realize

extension SignalTensor {
    /// Execute the computation graph and return the signal tensor data
    ///
    /// ```swift
    /// let freqs = Tensor([440, 880, 1320])
    /// let phases = Signal.phasor(freqs)
    /// let data = try phases.realize(frames: 64)  // [64 * 3] flattened
    /// ```
    ///
    /// - Parameters:
    ///   - frames: Number of frames to compute
    ///   - preserveState: If true, maintain state across realize() calls
    /// - Returns: Flat array of values: [frame0_elem0, frame0_elem1, ..., frame1_elem0, ...]
    public func realize(frames: Int = DGenConfig.defaultFrameCount, preserveState: Bool = false) throws -> [Float] {
        // Sum to scalar for output (drives computation)
        let sumNode = graph.node(.sum, [nodeId])
        let _ = graph.node(.output(0), [sumNode])

        // Mark this node for materialization so we can read the tensor values
        graph.graph.materializeNodes.insert(nodeId)

        // Compile and run
        let context = try graph.compile(frameCount: frames)
        graph.run(context: context, preserveState: preserveState)

        // Read tensor data from memory
        // Look up tensorId from nodeToTensor (created during compilation)
        if let tensorId = self.tensorId ?? graph.graph.nodeToTensor[nodeId] {
            if let data = graph.readTensorData(context: context, tensorId: tensorId) {
                return data
            }
        }

        // Fallback - return the summed output
        return graph.readOutputs(context: context)
    }
}

// MARK: - Errors

public enum DGenLazyError: Error, LocalizedError {
    case cannotRealizeDerivedTensor
    case compilationFailed(String)
    case runtimeError(String)

    public var errorDescription: String? {
        switch self {
        case .cannotRealizeDerivedTensor:
            return "Cannot realize derived tensor without tensorId. Store intermediate results in a tensor first."
        case .compilationFailed(let message):
            return "Compilation failed: \(message)"
        case .runtimeError(let message):
            return "Runtime error: \(message)"
        }
    }
}

// Realize - Compilation and execution of lazy graphs
//
// Implements realize() which compiles the lazy graph and runs it on Metal/CPU,
// returning the computed values.

import Foundation
import DGen

// MARK: - Execution Context

/// Stores compilation and runtime state for a realized graph
public class ExecutionContext {
    let compilationResult: CompilationResult
    let runtime: MetalCompiledKernel
    let frameCount: Int

    init(compilationResult: CompilationResult, runtime: MetalCompiledKernel, frameCount: Int) {
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

        // Compile the graph
        let result = try CompilationPipeline.compile(
            graph: graph,
            backend: DGenConfig.backend,
            options: .init(frameCount: frameCount, debug: DGenConfig.debug)
        )

        // Write kernels to disk if path is configured
        if let outputPath = DGenConfig.kernelOutputPath {
            writeKernelsToDisk(result, outputPath)
        }

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // Cache for reuse
        compilationCache = result
        runtimeCache = runtime
        isDirty = false

        return ExecutionContext(compilationResult: result, runtime: runtime, frameCount: frameCount)
    }

    /// Inject tensor data into Metal memory buffer
    func injectTensorData(context: ExecutionContext) {
        if let memBuffer = context.runtime.getBuffer(name: "memory") {
            let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
            DGen.injectTensorData(result: context.compilationResult, memory: memPtr)
        }
    }

    /// Inject Signal param values into Metal memory buffer
    func injectSignalParams(context: ExecutionContext) {
        guard let memBuffer = context.runtime.getBuffer(name: "memory") else { return }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
        let cellMappings = context.compilationResult.cellAllocations.cellMappings

        for signal in parameterRegistry.signals {
            guard let cellId = signal.cellId,
                  let value = signal.data else { continue }

            // Map logical cell to physical cell
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
        if let outBuffer = context.runtime.getBuffer(name: "outputs") {
            let outPtr = outBuffer.contents().assumingMemoryBound(to: Float.self)
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

        guard let memBuffer = context.runtime.getBuffer(name: "memory") else { return nil }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

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

                    case .slidingWindow(let windowSize, let inputShape):
                        // Sliding window: index i at frame f → base[f - windowSize + 1 + i]
                        let lastDim = indices.count - 1
                        let baseIdx = frame - windowSize + 1 + indices[lastDim]
                        if baseIdx < 0 {
                            inBounds = false
                        } else {
                            indices[lastDim] = baseIdx
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

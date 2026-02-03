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
            options: .init(frameCount: frameCount, debug: false)
        )

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
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

    /// Run the compiled graph
    func run(context: ExecutionContext, preserveState: Bool) {
        if !preserveState {
            context.runtime.zeroAllBuffers()
            injectTensorData(context: context)
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

    /// Read tensor data from memory buffer
    func readTensorData(context: ExecutionContext, tensorId: TensorID) -> [Float]? {
        guard let tensor = graph.tensors[tensorId] else { return nil }

        let physicalCellId = context.compilationResult.cellAllocations.cellMappings[tensor.cellId] ?? tensor.cellId

        guard let memBuffer = context.runtime.getBuffer(name: "memory") else { return nil }
        let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)

        var data = [Float](repeating: 0, count: tensor.size)
        for i in 0..<tensor.size {
            data[i] = memPtr[physicalCellId + i]
        }
        return data
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
        // For static tensors, we use frameCount=1
        let frameCount = 1

        // Add output node if this is a scalar result
        // For tensors, we read directly from memory
        if isScalar {
            // Add output node to drive computation
            let _ = graph.node(.output(0), [nodeId])
        } else {
            // For non-scalar tensors, we need a dummy output to drive computation
            // Sum the tensor to create a scalar output
            let sumNode = graph.node(.sum, [nodeId])
            let _ = graph.node(.output(0), [sumNode])
        }

        // Compile and run
        let context = try graph.compile(frameCount: frameCount)
        graph.run(context: context, preserveState: false)

        // Read tensor data from memory
        if let tensorId = self.tensorId {
            if let data = graph.readTensorData(context: context, tensorId: tensorId) {
                return data
            }
        }

        // Fallback: if no tensorId, this is a computed result
        // For scalar tensors, read from output
        if isScalar {
            return graph.readOutputs(context: context)
        }

        // For computed tensors without tensorId, we need to trace back
        // This is a limitation - computed tensors should have their results stored
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
        // For SignalTensor, we need to output each element
        // Sum to scalar for output (we'll read the tensor from memory)
        let sumNode = graph.node(.sum, [nodeId])
        let _ = graph.node(.output(0), [sumNode])

        // Compile and run
        let context = try graph.compile(frameCount: frames)
        graph.run(context: context, preserveState: preserveState)

        // Read tensor data from memory if we have a tensorId
        if let tensorId = self.tensorId {
            if let data = graph.readTensorData(context: context, tensorId: tensorId) {
                // For frame-varying tensors, data is stored per-frame
                // Return flattened: frames * tensorSize
                return data
            }
        }

        // Fallback for computed SignalTensors - return the summed output
        // This is a limitation for now
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

import AVFoundation
import Foundation
import Metal

public class MetalCompiledKernel: CompiledKernelRuntime {
  public let kernels: [CompiledKernel]
  public let cellAllocations: CellAllocations
  private let device: MTLDevice
  private let library: MTLLibrary
  private let commandQueue: MTLCommandQueue
  private var bufferPool: [String: MTLBuffer] = [:]  // shared source of buffers used by all kernels
  private var functions: [MTLFunction] = []  // one per kernel
  private var pipelineStates: [MTLComputePipelineState] = []  // cached PSOs matching `functions`
  private var context: IRContext
  private let maxFrameCount: Int
  private var firstDebug = true
  private let debugGradients: Bool =
    (ProcessInfo.processInfo.environment["DGEN_DEBUG_GRADS"] == "1")

  // Training kernel functions
  private var reduceGradientsFunction: MTLFunction?
  private var reduceGradientsSumFunction: MTLFunction?
  private var updateParametersSGDFunction: MTLFunction?
  private var updateParametersAdamFunction: MTLFunction?
  private var reduceGradientsPSO: MTLComputePipelineState?
  private var reduceGradientsSumPSO: MTLComputePipelineState?
  private var updateParametersSGDPSO: MTLComputePipelineState?
  private var updateParametersAdamPSO: MTLComputePipelineState?

  public init(
    kernels: [CompiledKernel], cellAllocations: CellAllocations, context: IRContext,
    frameCount: Int = 512 * 2
  )
    throws
  {
    self.maxFrameCount = frameCount * 2
    self.kernels = kernels
    self.cellAllocations = cellAllocations
    self.context = context

    guard let device = MTLCreateSystemDefaultDevice() else {
      throw NSError(
        domain: "MetalError", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal device"])
    }
    self.device = device

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "MetalError", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
    }
    self.commandQueue = commandQueue

    // Combine all kernel sources and compile
    var combinedSource = kernels.map { $0.source }.joined(separator: "\n\n")

    // Add training kernels
    let trainingKernelSource = """

      // MARK: - Training Kernels
      using namespace metal;

      kernel void reduceGradients(
          device const float* gradients [[buffer(0)]],
          device float* reducedGrads [[buffer(1)]],
          constant uint& frameCount [[buffer(2)]],
          constant uint& numGradIds [[buffer(3)]],
          uint gradId [[thread_position_in_grid]]
      ) {
          if (gradId >= numGradIds) return;

          float sum = 0.0;
          uint baseIdx = frameCount * gradId;
          for (uint i = 0; i < frameCount; i++) {
              float g = gradients[baseIdx + i];
              // Be robust to accidental non-finite per-frame gradients
              if (!isfinite(g)) { g = 0.0; }
              sum += g;
          }
          reducedGrads[gradId] = sum / float(frameCount);
      }

      kernel void reduceGradientsSum(
          device const float* gradients [[buffer(0)]],
          device float* reducedGradsSum [[buffer(1)]],
          constant uint& frameCount [[buffer(2)]],
          constant uint& numGradIds [[buffer(3)]],
          uint gradId [[thread_position_in_grid]]
      ) {
          if (gradId >= numGradIds) return;

          float sum = 0.0;
          uint baseIdx = frameCount * gradId;
          for (uint i = 0; i < frameCount; i++) {
              float g = gradients[baseIdx + i];
              if (!isfinite(g)) { g = 0.0; }
              sum += g;
          }
          reducedGradsSum[gradId] = sum;
      }

      kernel void updateParametersSGD(
          device float* memory [[buffer(0)]],
          device const float* reducedGrads [[buffer(1)]],
          constant uint* gradIds [[buffer(2)]],
          constant uint* physicalCells [[buffer(3)]],
          constant float& learningRate [[buffer(4)]],
          constant uint& paramCount [[buffer(5)]],
          uint paramIdx [[thread_position_in_grid]]
      ) {
          if (paramIdx >= paramCount) return;

          uint gradId = gradIds[paramIdx];
          uint physicalCell = physicalCells[paramIdx];
          float grad = reducedGrads[gradId];

          memory[physicalCell] -= learningRate * grad;
      }

      kernel void updateParametersAdam(
          device float* memory [[buffer(0)]],
          device const float* reducedGrads [[buffer(1)]],
          device float* m [[buffer(2)]],
          device float* v [[buffer(3)]],
          constant uint* gradIds [[buffer(4)]],
          constant uint* physicalCells [[buffer(5)]],
          constant float& learningRate [[buffer(6)]],
          constant float& beta1 [[buffer(7)]],
          constant float& beta2 [[buffer(8)]],
          constant float& epsilon [[buffer(9)]],
          constant uint& timestep [[buffer(10)]],
          constant uint& paramCount [[buffer(11)]],
          uint paramIdx [[thread_position_in_grid]]
      ) {
          if (paramIdx >= paramCount) return;

          uint gradId = gradIds[paramIdx];
          uint physicalCell = physicalCells[paramIdx];
          float grad = reducedGrads[gradId];

          m[paramIdx] = beta1 * m[paramIdx] + (1.0 - beta1) * grad;
          v[paramIdx] = beta2 * v[paramIdx] + (1.0 - beta2) * grad * grad;

          float m_hat = m[paramIdx] / (1.0 - pow(beta1, float(timestep)));
          float v_hat = v[paramIdx] / (1.0 - pow(beta2, float(timestep)));

          memory[physicalCell] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
      }
      """

    combinedSource += trainingKernelSource

    do {
      let options = MTLCompileOptions()
      options.fastMathEnabled = true
      // Don't treat warnings as errors - this is important for generated code
      // that may have unused variables in backward passes
      self.library = try device.makeLibrary(source: combinedSource, options: options)
    } catch {
      print("Metal compilation error:")
      print("=== Combined Metal Source ===")
      print(combinedSource)
      print("==============================")
      throw error
    }

    // Get function references for graph kernels
    for kernel in kernels {
      guard let function = library.makeFunction(name: kernel.name) else {
        throw NSError(
          domain: "MetalError", code: 3,
          userInfo: [NSLocalizedDescriptionKey: "Function \(kernel.name) not found"])
      }
      functions.append(function)
    }

    // Get training kernel function references
    reduceGradientsFunction = library.makeFunction(name: "reduceGradients")
    reduceGradientsSumFunction = library.makeFunction(name: "reduceGradientsSum")
    updateParametersSGDFunction = library.makeFunction(name: "updateParametersSGD")
    updateParametersAdamFunction = library.makeFunction(name: "updateParametersAdam")

    // Prebuild pipeline states for graph kernels
    pipelineStates.reserveCapacity(functions.count)
    for function in functions {
      let pso = try device.makeComputePipelineState(function: function)
      pipelineStates.append(pso)
    }

    // Prebuild pipeline states for training kernels
    if let f = reduceGradientsFunction {
      reduceGradientsPSO = try device.makeComputePipelineState(function: f)
    }
    if let f = reduceGradientsSumFunction {
      reduceGradientsSumPSO = try device.makeComputePipelineState(function: f)
    }
    if let f = updateParametersSGDFunction {
      updateParametersSGDPSO = try device.makeComputePipelineState(function: f)
    }
    if let f = updateParametersAdamFunction {
      updateParametersAdamPSO = try device.makeComputePipelineState(function: f)
    }

    try initializeBuffers()
  }

  private func getElementCount(_ bufferName: String) -> Int {
    switch bufferName {
    case "frameCount":
      return 1
    case "memory":
      return getMemorySize()
    case "t":
      return maxFrameCount * context.globals.count
    default:
      return maxFrameCount
    }
  }

  private func initializeBuffers() throws {
    // Collect all unique buffer names across kernels
    let allBufferNames = Set(kernels.flatMap { $0.buffers })

    // Create MTLBuffers for each unique buffer
    // Use larger buffer size to handle varying frameCount from AVAudioEngine
    for bufferName in allBufferNames {
      // Memory buffer size comes from getMemorySize(), others need maxFrameCount
      let elementCount = getElementCount(bufferName)

      let bufferSize = elementCount * MemoryLayout<Float>.size
      guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
      else {
        throw NSError(
          domain: "MetalError", code: 4,
          userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer \(bufferName)"])
      }

      // Initialize buffer contents to zero using memset (much faster than a loop)
      memset(buffer.contents(), 0, bufferSize)
      bufferPool[bufferName] = buffer
    }

    // Create frameCount buffer for scalar kernels
    guard
      let frameCountBuffer = device.makeBuffer(
        length: MemoryLayout<UInt32>.size, options: .storageModeShared)
    else {
      throw NSError(
        domain: "MetalError", code: 4,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create frameCount buffer"])
    }
    bufferPool["frameCount"] = frameCountBuffer
  }

  public func setParamValue(cellId: CellID, value: Float) {
    // TODO - implement
  }

  /// Get a buffer by name (used by training API)
  public func getBuffer(name: String) -> MTLBuffer? {
    return bufferPool[name]
  }

  /// Zero all GPU buffers (for deterministic training reset)
  public func zeroAllBuffers() {
    for (name, buffer) in bufferPool {
      // Skip frameCount - it's not a float buffer
      if name == "frameCount" { continue }
      memset(buffer.contents(), 0, buffer.length)
    }
  }

  public func run(
    outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int,
    volumeScale: Float = 1.0
  ) {
    precondition(
      frameCount <= maxFrameCount,
      "frameCount (\(frameCount)) exceeds maxFrameCount (\(maxFrameCount))"
    )
    // Update frameCount buffer with current frameCount value
    if let frameCountBuffer = bufferPool["frameCount"] {
      let frameCountPtr = frameCountBuffer.contents().assumingMemoryBound(to: UInt32.self)
      frameCountPtr[0] = UInt32(frameCount)
    }

    guard var commandBuffer = commandQueue.makeCommandBuffer() else { return }

    // Execute kernels in sequence
    // In non-debug mode, reuse a single encoder to reduce CPU overhead
    var sharedEncoder: MTLComputeCommandEncoder? = nil
    if !debugGradients {
      sharedEncoder = commandBuffer.makeComputeCommandEncoder()
    }
    for (index, kernel) in kernels.enumerated() {
      // Pick encoder: shared if available (non-debug), otherwise one per kernel
      let computeEncoder: MTLComputeCommandEncoder
      if let enc = sharedEncoder {
        computeEncoder = enc
      } else {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { continue }
        computeEncoder = enc
      }

      guard index < pipelineStates.count else { continue }
      _ = encodeKernel(
        kernel, pipelineState: pipelineStates[index], encoder: computeEncoder, frameCount: frameCount)

      if sharedEncoder == nil { computeEncoder.endEncoding() }
      if debugGradients {
        // Execute this kernel now so shared-memory buffers are visible for debug
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let cb = commandQueue.makeCommandBuffer() { commandBuffer = cb }
      }
    }

    // Close shared encoder if used
    if let enc = sharedEncoder {
      enc.endEncoding()
    }

    if !debugGradients {
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }
    firstDebug = false

    // Copy results back to outputs
    copyResultsToOutputs(outputs: outputs, frameCount: frameCount, volumeScale: volumeScale)
  }

  /// Bind buffer arguments and dispatch a single kernel on the given encoder.
  /// Returns a short description of the dispatch geometry for profiling.
  private func encodeKernel(
    _ kernel: CompiledKernel,
    pipelineState: MTLComputePipelineState,
    encoder: MTLComputeCommandEncoder,
    frameCount: Int
  ) -> String {
    encoder.setComputePipelineState(pipelineState)
    for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
      if let buffer = bufferPool[bufferName] {
        encoder.setBuffer(buffer, offset: 0, index: bufferIndex)
      }
    }

    if case .gemm(let tilesM, let tilesN, let fixedDepth) = kernel.dispatchMode {
      let depth = fixedDepth ?? (kernel.temporality == .static_ ? 1 : frameCount)
      encoder.dispatchThreadgroups(
        MTLSize(width: tilesN, height: tilesM, depth: max(1, depth)),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
      )
      return "gemm(\(tilesM)x\(tilesN)x\(depth))"
    }

    let totalThreads = kernel.dispatchMode.threadCount(frameCount: frameCount)
    let maxThreadsPerGroup = pipelineState.maxTotalThreadsPerThreadgroup
    let threadGroupWidth =
      totalThreads == 1
      ? 1
      : min(kernel.dispatchMode.threadGroupSize ?? 64, maxThreadsPerGroup, totalThreads)
    encoder.dispatchThreads(
      MTLSize(width: totalThreads, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: threadGroupWidth, height: 1, depth: 1)
    )
    return "\(kernel.dispatchMode)".components(separatedBy: "(").first ?? "?"
  }

  /// Run each kernel in isolation and record GPU timestamps.
  /// Returns per-kernel results sorted by index (not timed order).
  /// Note: kernels run on whatever data is currently in the buffers,
  /// so call this after a real backward pass to profile realistic workloads.
  public func profileKernels(frameCount: Int) -> [(index: Int, name: String, dispatchInfo: String, gpuMs: Double)] {
    if let frameCountBuffer = bufferPool["frameCount"] {
      frameCountBuffer.contents().assumingMemoryBound(to: UInt32.self)[0] = UInt32(frameCount)
    }

    var results: [(index: Int, name: String, dispatchInfo: String, gpuMs: Double)] = []

    for (index, kernel) in kernels.enumerated() {
      guard index < pipelineStates.count,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()
      else { continue }

      let dispatchInfo = encodeKernel(
        kernel, pipelineState: pipelineStates[index], encoder: computeEncoder, frameCount: frameCount)

      computeEncoder.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()

      let gpuMs = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000.0
      results.append((index: index, name: kernel.name, dispatchInfo: dispatchInfo, gpuMs: gpuMs))
    }

    return results
  }

  private func copyResultsToOutputs(
    outputs: UnsafeMutablePointer<Float>, frameCount: Int, volumeScale: Float
  ) {
    if let outputBuffer = bufferPool["outputs"] {
      let bufferContents = outputBuffer.contents().assumingMemoryBound(to: Float.self)
      for i in 0..<frameCount {
        outputs[i] = bufferContents[i] * volumeScale
      }
      return
    }

    // Fallback to other buffers if no dedicated outputs buffer
    for kernel in kernels {
      for bufferName in kernel.buffers {
        if bufferName.hasPrefix("t") || bufferName == "memory" {
          if let buffer = bufferPool[bufferName] {
            let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<frameCount {
              outputs[i] = bufferContents[i] * volumeScale
            }
            return
          }
        }
      }
    }
  }

  public func runAndPlay(
    engine: AVAudioEngine,
    sampleRate: Double,
    channels: Int,
    volumeScale: Float = 1.0,
    levelCallback: AudioLevelCallback? = nil
  ) throws -> AVAudioNode {
    //------------------------------------------------------------------
    // 1 ‧ Audio engine / format
    //------------------------------------------------------------------
    let format = AVAudioFormat(
      standardFormatWithSampleRate: sampleRate,
      channels: AVAudioChannelCount(channels))!

    //------------------------------------------------------------------
    // 2 ‧ One reusable 128-sample scratch buffer
    //------------------------------------------------------------------
    let blockSize = 128  // what the Metal kernel expects
    let silentInput = [Float](repeating: 0, count: blockSize)

    var interleavedScratch: UnsafeMutablePointer<Float>? = nil
    if channels > 1 {
      interleavedScratch = UnsafeMutablePointer<Float>
        .allocate(capacity: blockSize * channels)
    }

    //------------------------------------------------------------------
    // 3 ‧ Source node — handles ANY host frameCount, feeds kernel in 128s
    //------------------------------------------------------------------
    let sourceNode = AVAudioSourceNode(format: format) { _, _, frameCount, abl -> OSStatus in
      let buffers = UnsafeMutableAudioBufferListPointer(abl)

      var done = 0
      while done < Int(frameCount) {

        let n = min(blockSize, Int(frameCount) - done)  // ≤ 128

        // ----------------------------------------------------------
        // MONO
        // ----------------------------------------------------------
        if channels == 1 {
          let dst = buffers[0].mData!
            .assumingMemoryBound(to: Float.self)
            .advanced(by: done)

          silentInput.withUnsafeBufferPointer { zero in
            self.run(
              outputs: dst,
              inputs: zero.baseAddress!,
              frameCount: n,
              volumeScale: volumeScale)
          }

          // Call level callback after volume scaling if provided
          levelCallback?(dst, n)

          // ----------------------------------------------------------
          // MULTICHANNEL (Metal kernel returns interleaved)
          // ----------------------------------------------------------
        } else if let scratch = interleavedScratch {
          silentInput.withUnsafeBufferPointer { zero in
            self.run(
              outputs: scratch,
              inputs: zero.baseAddress!,
              frameCount: n,
              volumeScale: volumeScale)
          }

          // Call level callback with first channel after volume scaling if provided
          levelCallback?(scratch, n)

          // De-interleave into Core Audio’s non-interleaved buffers
          for ch in 0..<channels {
            let dst = buffers[ch].mData!
              .assumingMemoryBound(to: Float.self)
              .advanced(by: done)
            for i in 0..<n {
              dst[i] = scratch[i * channels + ch]
            }
          }
        }

        done += n
      }
      return noErr
    }

    engine.attach(sourceNode)
    engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
    try engine.start()
    return sourceNode
  }

  // Default implementation for backwards compatibility
  public func runAndPlay(
    engine: AVAudioEngine, sampleRate: Double, channels: Int, volumeScale: Float
  ) throws -> AVAudioNode {
    return try runAndPlay(
      engine: engine, sampleRate: sampleRate, channels: channels, volumeScale: volumeScale,
      levelCallback: nil)
  }

  // Helper method to read buffer contents for backpropagation
  public func readBuffer(named: String) -> [Float]? {
    guard let buffer = bufferPool[named] else { return nil }

    // Special handling for frameCount buffer which stores Int32
    if named == "frameCount" {
      let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
      let intValue = intContents[0]
      return [Float(intValue)]  // Return frameCount as a single-element Float array
    }

    let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
    let count = buffer.length / MemoryLayout<Float>.size  // Use actual buffer size
    return Array(UnsafeBufferPointer(start: bufferContents, count: count))
  }

  public func cleanup() {
    for (name, buffer) in bufferPool {
      if name == "frameCount" {
        let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
        intContents[0] = 0
      } else {
        memset(buffer.contents(), 0, buffer.length)
      }
    }

    functions.removeAll()

    if let commandBuffer = commandQueue.makeCommandBuffer() {
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }
  }

  // MARK: - New Memory Management Methods

  public func getMemorySize() -> Int {
    // Memory size comes from cell allocations computed during compilation
    return cellAllocations.totalMemorySlots
  }

  public func allocateNodeMemory() -> UnsafeMutableRawPointer? {
    let byteSize = getMemorySize() * MemoryLayout<Float>.size
    guard let ptr = malloc(byteSize) else { return nil }
    memset(ptr, 0, byteSize)
    // Inject precomputed tensor data (e.g. FFT twiddle factors, Hann window)
    let floatPtr = ptr.assumingMemoryBound(to: Float.self)
    for (offset, data) in cellAllocations.tensorInitData {
      for (i, value) in data.enumerated() {
        floatPtr[offset + i] = value
      }
    }
    return ptr
  }

  public func deallocateNodeMemory(_ memory: UnsafeMutableRawPointer) {
    free(memory)
  }

  public func runWithMemory(
    outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>,
    memory: UnsafeMutableRawPointer, frameCount: Int
  ) {
    // Copy the passed-in memory to Metal's internal buffer before running
    if let metalMemoryBuffer = bufferPool["memory"] {
      let memorySize = getMemorySize()
      let byteSize = memorySize * MemoryLayout<Float>.size
      let sourcePtr = memory.assumingMemoryBound(to: UInt8.self)
      let destPtr = metalMemoryBuffer.contents()
      memcpy(destPtr, sourcePtr, byteSize)
    }

    run(outputs: outputs, inputs: inputs, frameCount: frameCount, volumeScale: 1.0)
  }

  // MARK: - Convenience Methods with Internal Buffers

  private var internalInputBuffer: [Float]?
  private var internalOutputBuffer: [Float]?
  private var currentFrameCount: Int = 0

  /// Simplified run method that uses internal buffers
  /// - Parameters:
  ///   - memory: Memory pointer to copy into Metal buffer
  ///   - frameCount: Number of frames to process
  public func run(memory: UnsafeMutableRawPointer, frameCount: Int) {
    // Allocate or resize internal buffers if needed
    if internalInputBuffer == nil || currentFrameCount != frameCount {
      internalInputBuffer = [Float](repeating: 0.0, count: frameCount)
      internalOutputBuffer = [Float](repeating: 0.0, count: frameCount)
      currentFrameCount = frameCount
    }

    guard let inputBuf = internalInputBuffer else {
      fatalError("Failed to allocate internal input buffer")
    }

    inputBuf.withUnsafeBufferPointer { inPtr in
      internalOutputBuffer!.withUnsafeMutableBufferPointer { outPtr in
        runWithMemory(
          outputs: outPtr.baseAddress!,
          inputs: inPtr.baseAddress!,
          memory: memory,
          frameCount: frameCount
        )
      }
    }
  }

  /// Get a copy of the output buffer from the last run
  /// - Returns: Array of output samples
  public func getOutputBuffer() -> [Float] {
    guard let output = internalOutputBuffer else {
      return []
    }
    return output
  }

  /// Simplified run that uses existing device `memory` without copying from host
  /// - Parameter frameCount: number of frames to process
  public func runNoCopy(frameCount: Int) {
    // Allocate or resize internal buffers if needed
    if internalInputBuffer == nil || currentFrameCount != frameCount {
      internalInputBuffer = [Float](repeating: 0.0, count: frameCount)
      internalOutputBuffer = [Float](repeating: 0.0, count: frameCount)
      currentFrameCount = frameCount
    }

    guard let inputBuf = internalInputBuffer else {
      fatalError("Failed to allocate internal input buffer")
    }

    inputBuf.withUnsafeBufferPointer { inPtr in
      internalOutputBuffer!.withUnsafeMutableBufferPointer { outPtr in
        run(
          outputs: outPtr.baseAddress!,
          inputs: inPtr.baseAddress!,
          frameCount: frameCount,
          volumeScale: 1.0
        )
      }
    }
  }

  /// Get the last output value (useful for scalar loss values)
  /// - Returns: The last frame's output value, or nil if no output available
  public func getLastOutput() -> Float? {
    guard let output = internalOutputBuffer, !output.isEmpty else {
      return nil
    }
    return output.last
  }

  // MARK: - Parameter + Memory helpers

  /// Write a small set of parameter values into the device `memory` buffer at the given cells
  public func writeParameters(physicalCells: [UInt32], values: [Float]) {
    guard let memoryBuffer = bufferPool["memory"] else { return }
    let count = min(physicalCells.count, values.count)
    let memPtr = memoryBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<count {
      memPtr[Int(physicalCells[i])] = values[i]
    }
  }

  /// Clear the entire device `memory` buffer and then write back parameter values
  public func clearMemoryPreservingParameters(physicalCells: [UInt32], values: [Float]) {
    guard let memoryBuffer = bufferPool["memory"] else { return }
    let memorySize = getMemorySize()
    memset(memoryBuffer.contents(), 0, memorySize * MemoryLayout<Float>.size)
    writeParameters(physicalCells: physicalCells, values: values)
  }

  public func updateParametersSGDGPU(
    gradIds: [UInt32], physicalCells: [UInt32], learningRate: Float
  ) {
    guard updateParametersSGDFunction != nil else {
      print("updateParametersSGD function not found")
      return
    }

    guard let memoryBuffer = bufferPool["memory"],
      let reducedGradsBuffer = bufferPool["reducedGrads"]
    else {
      print("Required buffers not found for SGD update")
      return
    }

    let paramCount = gradIds.count
    guard paramCount > 0 else { return }

    ensureBuffer(named: "gradIds", minimumSize: paramCount * MemoryLayout<UInt32>.size)
    ensureBuffer(named: "physicalCells", minimumSize: paramCount * MemoryLayout<UInt32>.size)

    copyToBuffer(named: "gradIds", values: gradIds)
    copyToBuffer(named: "physicalCells", values: physicalCells)

    guard let gradIdsBuffer = bufferPool["gradIds"],
      let physicalCellsBuffer = bufferPool["physicalCells"]
    else { return }

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("Failed to create command buffer or encoder")
      return
    }

    guard let pipelineState = updateParametersSGDPSO else {
      print("updateParametersSGD pipeline state not found")
      return
    }

    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(memoryBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(reducedGradsBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(gradIdsBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(physicalCellsBuffer, offset: 0, index: 3)

    var lr = learningRate
    var pc = UInt32(paramCount)
    computeEncoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 4)
    computeEncoder.setBytes(&pc, length: MemoryLayout<UInt32>.size, index: 5)

    let threadsPerGrid = MTLSize(width: paramCount, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(
      width: min(paramCount, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  public func updateParametersAdamGPU(
    gradIds: [UInt32], physicalCells: [UInt32], learningRate: Float, beta1: Float, beta2: Float,
    epsilon: Float, timestep: Int
  ) {
    guard updateParametersAdamFunction != nil else {
      print("updateParametersAdam function not found")
      return
    }

    guard let memoryBuffer = bufferPool["memory"],
      let reducedGradsBuffer = bufferPool["reducedGrads"]
    else {
      print("Required buffers not found for Adam update")
      return
    }

    let paramCount = gradIds.count
    guard paramCount > 0 else { return }

    let momentumSize = paramCount * MemoryLayout<Float>.size
    ensureBuffer(named: "adam_m", minimumSize: momentumSize, zeroInitialize: true)
    ensureBuffer(named: "adam_v", minimumSize: momentumSize, zeroInitialize: true)
    ensureBuffer(named: "gradIds", minimumSize: paramCount * MemoryLayout<UInt32>.size)
    ensureBuffer(named: "physicalCells", minimumSize: paramCount * MemoryLayout<UInt32>.size)

    copyToBuffer(named: "gradIds", values: gradIds)
    copyToBuffer(named: "physicalCells", values: physicalCells)

    guard let gradIdsBuffer = bufferPool["gradIds"],
      let physicalCellsBuffer = bufferPool["physicalCells"],
      let mBuffer = bufferPool["adam_m"],
      let vBuffer = bufferPool["adam_v"]
    else { return }

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("Failed to create command buffer or encoder")
      return
    }

    guard let pipelineState = updateParametersAdamPSO else {
      print("updateParametersAdam pipeline state not found")
      return
    }

    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(memoryBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(reducedGradsBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(mBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(vBuffer, offset: 0, index: 3)
    computeEncoder.setBuffer(gradIdsBuffer, offset: 0, index: 4)
    computeEncoder.setBuffer(physicalCellsBuffer, offset: 0, index: 5)

    var lr = learningRate
    var b1 = beta1
    var b2 = beta2
    var eps = epsilon
    var t = UInt32(timestep)
    var pc = UInt32(paramCount)

    computeEncoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 6)
    computeEncoder.setBytes(&b1, length: MemoryLayout<Float>.size, index: 7)
    computeEncoder.setBytes(&b2, length: MemoryLayout<Float>.size, index: 8)
    computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 9)
    computeEncoder.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 10)
    computeEncoder.setBytes(&pc, length: MemoryLayout<UInt32>.size, index: 11)

    let threadsPerGrid = MTLSize(width: paramCount, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(
      width: min(paramCount, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  // MARK: - Buffer Helpers

  private func ensureBuffer(named name: String, minimumSize: Int, zeroInitialize: Bool = false) {
    if bufferPool[name] == nil || bufferPool[name]!.length < minimumSize {
      guard let buffer = device.makeBuffer(length: minimumSize, options: .storageModeShared) else {
        return
      }
      if zeroInitialize {
        memset(buffer.contents(), 0, minimumSize)
      }
      bufferPool[name] = buffer
    }
  }

  private func copyToBuffer(named name: String, values: [UInt32]) {
    guard let buffer = bufferPool[name] else { return }
    let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<values.count {
      ptr[i] = values[i]
    }
  }

  deinit {
    cleanup()
  }
}

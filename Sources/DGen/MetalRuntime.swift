import AVFoundation
import AudioToolbox  // kAudioUnitProperty_MaximumFramesPerSlice
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
    if bufferName == "frameCount" {
      return 1
    } else if bufferName == "memory" || bufferName == "grad_memory" {
      return getMemorySize()
    } else if bufferName == "t" {
      return maxFrameCount * context.globals.count
    } else if bufferName == "gradients" {
      return context.computeGradientBufferSize(frameCount: maxFrameCount)
    } else if bufferName == "reducedGradsSum" || bufferName == "reducedGrads" {
      return max(1, context.maxGradId + 1)
    } else {
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

  public func resetGradientBuffers(numFrames: Int) {
    // Reset gradients buffer
    guard let buffer = bufferPool["gradients"] else {
      print("   [DEBUG] No gradients buffer found!")
      return
    }

    let elementCount = getElementCount("gradients")
    let bufferSize = elementCount * MemoryLayout<Float>.size
    guard context.seedGradients.count > 0 else {
      print("   [DEBUG] No seed gradients to reset!")
      return
    }

    // Initialize buffer contents to zero
    let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
      bufferContents[i] = 0.0
    }

    // Set all seed gradients to 1.0 (typically loss nodes)
    // Each seed gets a region of numFrames values set to 1.0
    //print("   [DEBUG] Seeding gradients for \(context.seedGradients.count) seeds: \(context.seedGradients)")
    for seedId in context.seedGradients {
      let startIdx = numFrames * seedId
      let endIdx = numFrames * (seedId + 1)
      //   print("   [DEBUG] Setting gradId=\(seedId) range [\(startIdx)..<\(endIdx)] to 1.0")
      for i in startIdx..<endIdx {
        bufferContents[i] = 1.0
      }

      // Verify it was set
      let firstVal = bufferContents[startIdx]
      let lastVal = bufferContents[endIdx - 1]
      //print("   [DEBUG] Verification: gradients[\(startIdx)]=\(firstVal), gradients[\(endIdx-1)]=\(lastVal)")
    }

    // Reset grad_memory buffer (used for spectralLoss ring buffers and phasor gradient accumulation)
    if let gradMemBuffer = bufferPool["grad_memory"] {
      let gradMemSize = getElementCount("grad_memory")
      let gradMemBufferSize = gradMemSize * MemoryLayout<Float>.size
      let gradMemContents = gradMemBuffer.contents().assumingMemoryBound(to: Float.self)
      for i in 0..<(gradMemBufferSize / MemoryLayout<Float>.size) {
        gradMemContents[i] = 0.0
      }
    }
  }

  public func setParamValue(cellId: CellID, value: Float) {
    // TODO - implement
  }

  /// Get a buffer by name (used by training API)
  public func getBuffer(name: String) -> MTLBuffer? {
    return bufferPool[name]
  }

  public func run(
    outputs: UnsafeMutablePointer<Float>, inputs: UnsafePointer<Float>, frameCount: Int,
    volumeScale: Float = 1.0
  ) {
    // Update frameCount buffer with current frameCount value
    if let frameCountBuffer = bufferPool["frameCount"] {
      let frameCountPtr = frameCountBuffer.contents().assumingMemoryBound(to: UInt32.self)
      frameCountPtr[0] = UInt32(frameCount)
    }

    guard var commandBuffer = commandQueue.makeCommandBuffer() else { return }

    func debugPrintGradStats(label: String) {
      guard debugGradients, firstDebug, let gradientsBuf = bufferPool["gradients"] else { return }
      let gptr = gradientsBuf.contents().assumingMemoryBound(to: Float.self)
      let numGradIds = context.maxGradId + 1
      let fc = frameCount
      print("   [DEBUG] Grad stats after \(label):")
      for gid in 0..<numGradIds {
        let base = gid * fc
        var minv: Float = Float.greatestFiniteMagnitude
        var maxv: Float = -Float.greatestFiniteMagnitude
        var anyNaN = false
        var anyInf = false
        var sum: Float = 0
        for i in 0..<fc {
          let v = gptr[base + i]
          if v.isNaN { anyNaN = true }
          if !v.isFinite { anyInf = true }
          if v < minv { minv = v }
          if v > maxv { maxv = v }
          sum += v
        }
        let mean = sum / Float(fc)
        print("      gid=\(gid) min=\(minv) max=\(maxv) mean=\(mean) NaN=\(anyNaN) Inf=\(anyInf)")
      }
    }

    //resetGradientBuffers(numFrames: frameCount)

    // Execute kernels in sequence
    // In non-debug mode, reuse a single compute encoder across all kernels to reduce CPU overhead
    var sharedEncoder: MTLComputeCommandEncoder? = nil
    if !debugGradients {
      sharedEncoder = commandBuffer.makeComputeCommandEncoder()
    }
    for (index, kernel) in kernels.enumerated() {
      if firstDebug {
        print("   [DEBUG] Executing kernel \(index): \(kernel.name)  \(kernel.kind)")
      }
      if kernel.needsReducedGradsSum {
        if let enc = sharedEncoder {
          enc.endEncoding()
          sharedEncoder = nil
        }
        encodeReduceGradientsSum(commandBuffer: commandBuffer, frameCount: frameCount)
        if !debugGradients {
          sharedEncoder = commandBuffer.makeComputeCommandEncoder()
        }
      }
      // Pick encoder: shared if available (non-debug), otherwise one per kernel
      let computeEncoder: MTLComputeCommandEncoder
      if let enc = sharedEncoder {
        computeEncoder = enc
      } else {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { continue }
        computeEncoder = enc
      }

      guard index < pipelineStates.count else { continue }
      let pipelineState = pipelineStates[index]
      computeEncoder.setComputePipelineState(pipelineState)

      // Detect segmented kernel by presence of segmentLen buffer
      let isSegmented = kernel.buffers.contains("segmentLen")
      let segmentCapacity = MIN_FRAME_COUNT

      // isSegments is true when we use delay1 which requires dividing up a simd kernel into batches of 4
      if isSegmented {
        // We'll reuse the same encoder for multiple segment dispatches
        var base = 0
        while base < frameCount {
          let thisLen = min(segmentCapacity, frameCount - base)

          // Update segmentLen
          if let segBuf = bufferPool["segmentLen"] {
            let segPtr = segBuf.contents().assumingMemoryBound(to: UInt32.self)
            segPtr[0] = UInt32(thisLen)
          }
          // Update segmentBase (in frames)
          if let segBaseBuf = bufferPool["segmentBase"] {
            let basePtr = segBaseBuf.contents().assumingMemoryBound(to: UInt32.self)
            basePtr[0] = UInt32(base)
          }

          // Bind buffers with per-segment offsets
          for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
            if firstDebug {
              //print(
              //  "kernel[\(index) setting buffer=\(bufferName) index=\(bufferIndex)"
              // )
            }

            guard let buffer = bufferPool[bufferName] else { continue }
            computeEncoder.setBuffer(buffer, offset: 0, index: bufferIndex)

            /*
            // Offset per-frame buffers (t*, outputs); never offset memory or frameCount/segmentLen
            let needsOffset = (bufferName.hasPrefix("t") || bufferName == "outputs")
            let byteOffset = needsOffset ? base * MemoryLayout<Float>.size : 0
            computeEncoder.setBuffer(buffer, offset: byteOffset, index: bufferIndex)

             */
          }

          let threadsPerGroup = MTLSize(width: thisLen, height: 1, depth: 1)
          let numThreadGroups = MTLSize(width: 1, height: 1, depth: 1)
          computeEncoder.dispatchThreadgroups(
            numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
          base += thisLen
        }
        if sharedEncoder == nil { computeEncoder.endEncoding() }
        if debugGradients {
          // Execute this kernel now so shared-memory buffers are visible for debug
          commandBuffer.commit()
          commandBuffer.waitUntilCompleted()
          debugPrintGradStats(label: "kernel \(index)")
          // Start a new command buffer for next kernel
          if let cb = commandQueue.makeCommandBuffer() { commandBuffer = cb }
        }
      } else {
        // Non-segmented kernel: single dispatch as before
        for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
          if let buffer = bufferPool[bufferName] {
            if firstDebug {
              // print(
              //   "kernel[\(index) setting buffer=\(bufferName) index=\(bufferIndex)]"
              //)
            }
            computeEncoder.setBuffer(buffer, offset: 0, index: bufferIndex)
          }
        }

        let tgMax = pipelineState.maxTotalThreadsPerThreadgroup

        let totalThreads: Int
        if let overrideThreads = kernel.threadCount {
          totalThreads = max(1, overrideThreads)
        } else if kernel.kind == .scalar {
          totalThreads = 1
        } else {
          totalThreads = frameCount
        }

        if totalThreads == 1 {
          let threads = MTLSize(width: 1, height: 1, depth: 1)
          let tpg = MTLSize(width: 1, height: 1, depth: 1)
          computeEncoder.dispatchThreads(threads, threadsPerThreadgroup: tpg)
        } else {
          let tpgW = min(kernel.threadGroupSize ?? 64, tgMax, totalThreads)  // clamp to device cap
          let total = MTLSize(width: totalThreads, height: 1, depth: 1)
          let tpg = MTLSize(width: tpgW, height: 1, depth: 1)
          computeEncoder.dispatchThreads(total, threadsPerThreadgroup: tpg)
        }
        if sharedEncoder == nil { computeEncoder.endEncoding() }
        if debugGradients {
          // Execute this kernel now so shared-memory buffers are visible for debug
          commandBuffer.commit()
          commandBuffer.waitUntilCompleted()
          debugPrintGradStats(label: "kernel \(index)")
          if let cb = commandQueue.makeCommandBuffer() { commandBuffer = cb }
        }
        /*
        pending += 1
        if pending == 5 {
          print("FLUSHING at kernel=\(index)")
          // old flush path no longer used
          print("finished flush")
        }  // tune 4 to taste

         */
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

  private func copyResultsToOutputs(
    outputs: UnsafeMutablePointer<Float>, frameCount: Int, volumeScale: Float
  ) {
    // Find output buffers and copy their contents
    // First check for dedicated "outputs" buffer
    if let memory = bufferPool["grad_memory"] {
      let bufferContents = memory.contents().assumingMemoryBound(to: Float.self)
      // Copy all frameCount frames (not limited to MIN_FRAME_COUNT)
      for i in 0..<frameCount {
        if bufferContents[i] > 0 {
          //print("memory[\(i)] = \(bufferContents[i])")
        }
      }

    }
    if let outputBuffer = bufferPool["outputs"] {
      let bufferContents = outputBuffer.contents().assumingMemoryBound(to: Float.self)
      // Copy all frameCount frames (not limited to MIN_FRAME_COUNT)
      for i in 0..<frameCount {
        outputs[i] = bufferContents[i] * volumeScale
        //print("outputs[\(i)] = \(outputs[i])")
      }
      return
    }

    // Fallback to other buffers if no dedicated outputs buffer
    for kernel in kernels {
      for bufferName in kernel.buffers {
        // Check if this buffer contains output data
        if bufferName.hasPrefix("t") || bufferName == "memory" {
          if let buffer = bufferPool[bufferName] {
            let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
            // Copy all frameCount frames (not limited to MIN_FRAME_COUNT)
            for i in 0..<frameCount {
              outputs[i] = bufferContents[i] * volumeScale
            }
            break  // Take first matching buffer as output
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

    //------------------------------------------------------------------
    // 4 ‧ Spin the engine
    //------------------------------------------------------------------
    engine.attach(sourceNode)
    engine.connect(sourceNode, to: engine.mainMixerNode, format: format)
    try engine.start()
    return sourceNode

    //  print("🟢 Playing Metal kernels for \(durationSeconds) seconds …")
    //Thread.sleep(forTimeInterval: durationSeconds)
    //engine.stop()

    //     interleavedScratch?.deallocate()
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

  // Debug method to print intermediate buffer states
  public func debugBufferStates() {
    print("=== METAL BUFFER DEBUG ===")
    for (name, buffer) in bufferPool.sorted(by: { $0.key < $1.key }) {
      let values: [Float]
      let count: Int

      // Special handling for frameCount buffer which stores Int32
      if name == "frameCount" {
        let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
        let intValue = intContents[0]
        values = [Float(intValue)]
        count = 1
      } else {
        let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
        count = buffer.length / MemoryLayout<Float>.size  // Use actual buffer size
        values = Array(UnsafeBufferPointer(start: bufferContents, count: count))
      }

      // Show first 10 and last 10 values for large buffers
      if count > 20 {
        let first10 = Array(values.prefix(10))
        let last10 = Array(values.suffix(10))
        print(
          "\(name): [\(first10.map { String(format: "%.3f", $0) }.joined(separator: ", "))...] (length: \(count))"
        )
        if name == "memory" {
          // Show specific memory indices used in kernels
          print("  memory[0]: \(String(format: "%.6f", values[0]))")
          print("  memory[1]: \(String(format: "%.6f", values[1]))")
          print("  memory[2]: \(String(format: "%.6f", values[2]))")
        }
      } else {
        print(
          "\(name): [\(values.map { String(format: "%.3f", $0) }.joined(separator: ", "))]"
        )
      }
    }
    print("=========================")
  }

  public func cleanup() {
    // Reset all buffers to zero
    for (name, buffer) in bufferPool {
      if name == "frameCount" {
        // Special handling for frameCount buffer
        let intContents = buffer.contents().assumingMemoryBound(to: Int32.self)
        intContents[0] = 0
      } else {
        // Zero out float buffers
        let bufferSize = name == "memory" ? 512 : 2048
        memset(buffer.contents(), 0, bufferSize * MemoryLayout<Float>.size)
      }
    }

    // Clear function references
    functions.removeAll()

    // Ensure all GPU work is complete by creating and waiting for a command buffer
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
    let ptr = malloc(byteSize)
    if let ptr = ptr {
      // Zero initialize the memory
      memset(ptr, 0, byteSize)
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

    // Use internal buffers
    guard var inputBuf = internalInputBuffer, var outputBuf = internalOutputBuffer else {
      fatalError("Failed to allocate internal buffers")
    }

    inputBuf.withUnsafeBufferPointer { inPtr in
      outputBuf.withUnsafeMutableBufferPointer { outPtr in
        runWithMemory(
          outputs: outPtr.baseAddress!,
          inputs: inPtr.baseAddress!,
          memory: memory,
          frameCount: frameCount
        )
      }
    }

    // Store output for later retrieval
    internalOutputBuffer = outputBuf
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

    guard var inputBuf = internalInputBuffer, var outputBuf = internalOutputBuffer else {
      fatalError("Failed to allocate internal buffers")
    }

    inputBuf.withUnsafeBufferPointer { inPtr in
      outputBuf.withUnsafeMutableBufferPointer { outPtr in
        run(
          outputs: outPtr.baseAddress!,
          inputs: inPtr.baseAddress!,
          frameCount: frameCount,
          volumeScale: 1.0
        )
      }
    }

    internalOutputBuffer = outputBuf
  }

  /// Get the last output value (useful for scalar loss values)
  /// - Returns: The last frame's output value, or nil if no output available
  public func getLastOutput() -> Float? {
    guard let output = internalOutputBuffer, !output.isEmpty else {
      return nil
    }
    return output.last
  }

  // MARK: - Training Kernel Dispatch

  /// Reduce gradients across frames on GPU
  /// - Parameters:
  ///   - frameCount: Number of frames per gradient
  ///   - numGradIds: Total number of gradient IDs
  /// - Returns: Buffer containing reduced gradients
  public func reduceGradientsGPU(frameCount: Int, numGradIds: Int) -> MTLBuffer? {
    guard let function = reduceGradientsFunction else {
      print("⚠️ reduceGradients function not found")
      return nil
    }

    guard let gradientsBuffer = bufferPool["gradients"] else {
      print("⚠️ gradients buffer not found")
      return nil
    }

    // Create or get reducedGrads buffer
    let reducedGradsSize = numGradIds * MemoryLayout<Float>.size
    if bufferPool["reducedGrads"] == nil {
      guard let buffer = device.makeBuffer(length: reducedGradsSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create reducedGrads buffer")
        return nil
      }
      bufferPool["reducedGrads"] = buffer
    }

    guard let reducedGradsBuffer = bufferPool["reducedGrads"] else {
      return nil
    }

    // Create command buffer and encoder
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("⚠️ Failed to create command buffer or encoder")
      return nil
    }

    guard let pipelineState = reduceGradientsPSO else {
      print("⚠️ Failed to create reduceGradients pipeline state")
      return nil
    }

    computeEncoder.setComputePipelineState(pipelineState)

    // Set buffers
    computeEncoder.setBuffer(gradientsBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(reducedGradsBuffer, offset: 0, index: 1)

    // Set scalar parameters
    var frameCountUInt = UInt32(frameCount)
    var numGradIdsUInt = UInt32(numGradIds)
    computeEncoder.setBytes(&frameCountUInt, length: MemoryLayout<UInt32>.size, index: 2)
    computeEncoder.setBytes(&numGradIdsUInt, length: MemoryLayout<UInt32>.size, index: 3)

    // Dispatch one thread per gradient ID
    let threadsPerGrid = MTLSize(width: numGradIds, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(
      width: min(numGradIds, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    return reducedGradsBuffer
  }

  private func encodeReduceGradientsSum(commandBuffer: MTLCommandBuffer, frameCount: Int) {
    guard let pipelineState = reduceGradientsSumPSO else { return }
    guard let gradientsBuffer = bufferPool["gradients"] else { return }

    let numGradIds = max(0, context.maxGradId + 1)
    if numGradIds == 0 { return }

    // Create or get reducedGradsSum buffer
    let reducedGradsSize = numGradIds * MemoryLayout<Float>.size
    if bufferPool["reducedGradsSum"] == nil
      || bufferPool["reducedGradsSum"]!.length < reducedGradsSize
    {
      guard let buffer = device.makeBuffer(length: reducedGradsSize, options: .storageModeShared)
      else {
        return
      }
      bufferPool["reducedGradsSum"] = buffer
    }

    guard let reducedGradsSumBuffer = bufferPool["reducedGradsSum"] else { return }

    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(gradientsBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(reducedGradsSumBuffer, offset: 0, index: 1)

    var frameCountUInt = UInt32(frameCount)
    var numGradIdsUInt = UInt32(numGradIds)
    computeEncoder.setBytes(&frameCountUInt, length: MemoryLayout<UInt32>.size, index: 2)
    computeEncoder.setBytes(&numGradIdsUInt, length: MemoryLayout<UInt32>.size, index: 3)

    let threadsPerGrid = MTLSize(width: numGradIds, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(
      width: min(numGradIds, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    computeEncoder.endEncoding()
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

  /// Update parameters using SGD on GPU
  /// - Parameters:
  ///   - gradIds: Array mapping parameter index to gradient ID
  ///   - physicalCells: Array mapping parameter index to memory cell
  ///   - learningRate: Learning rate for SGD
  public func updateParametersSGDGPU(
    gradIds: [UInt32], physicalCells: [UInt32], learningRate: Float
  ) {
    guard let function = updateParametersSGDFunction else {
      print("⚠️ updateParametersSGD function not found")
      return
    }

    guard let memoryBuffer = bufferPool["memory"],
      let reducedGradsBuffer = bufferPool["reducedGrads"]
    else {
      print("⚠️ Required buffers not found")
      return
    }

    let paramCount = gradIds.count
    if paramCount == 0 {
      return
    }

    // Create or update gradIds buffer
    let gradIdsSize = paramCount * MemoryLayout<UInt32>.size
    if bufferPool["gradIds"] == nil || bufferPool["gradIds"]!.length < gradIdsSize {
      guard let buffer = device.makeBuffer(length: gradIdsSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create gradIds buffer")
        return
      }
      bufferPool["gradIds"] = buffer
    }

    // Create or update physicalCells buffer
    let physicalCellsSize = paramCount * MemoryLayout<UInt32>.size
    if bufferPool["physicalCells"] == nil
      || bufferPool["physicalCells"]!.length < physicalCellsSize
    {
      guard
        let buffer = device.makeBuffer(length: physicalCellsSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create physicalCells buffer")
        return
      }
      bufferPool["physicalCells"] = buffer
    }

    // Copy data to buffers
    if let gradIdsBuffer = bufferPool["gradIds"] {
      let ptr = gradIdsBuffer.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<paramCount {
        ptr[i] = gradIds[i]
      }
    }

    if let physicalCellsBuffer = bufferPool["physicalCells"] {
      let ptr = physicalCellsBuffer.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<paramCount {
        ptr[i] = physicalCells[i]
      }
    }

    guard let gradIdsBuffer = bufferPool["gradIds"],
      let physicalCellsBuffer = bufferPool["physicalCells"]
    else {
      return
    }

    // Create command buffer and encoder
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("⚠️ Failed to create command buffer or encoder")
      return
    }

    guard let pipelineState = updateParametersSGDPSO else {
      print("⚠️ Failed to create updateParametersSGD pipeline state")
      return
    }

    computeEncoder.setComputePipelineState(pipelineState)

    // Set buffers
    computeEncoder.setBuffer(memoryBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(reducedGradsBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(gradIdsBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(physicalCellsBuffer, offset: 0, index: 3)

    // Set scalar parameters
    var lr = learningRate
    var pc = UInt32(paramCount)
    computeEncoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 4)
    computeEncoder.setBytes(&pc, length: MemoryLayout<UInt32>.size, index: 5)

    // Dispatch one thread per parameter
    let threadsPerGrid = MTLSize(width: paramCount, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(
      width: min(paramCount, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  /// Update parameters using Adam on GPU
  /// - Parameters:
  ///   - gradIds: Array mapping parameter index to gradient ID
  ///   - physicalCells: Array mapping parameter index to memory cell
  ///   - learningRate: Learning rate for Adam
  ///   - beta1: Exponential decay rate for first moment estimates
  ///   - beta2: Exponential decay rate for second moment estimates
  ///   - epsilon: Small constant for numerical stability
  ///   - timestep: Current training timestep (for bias correction)
  public func updateParametersAdamGPU(
    gradIds: [UInt32], physicalCells: [UInt32], learningRate: Float, beta1: Float, beta2: Float,
    epsilon: Float, timestep: Int
  ) {
    guard let function = updateParametersAdamFunction else {
      print("⚠️ updateParametersAdam function not found")
      return
    }

    guard let memoryBuffer = bufferPool["memory"],
      let reducedGradsBuffer = bufferPool["reducedGrads"]
    else {
      print("⚠️ Required buffers not found")
      return
    }

    let paramCount = gradIds.count
    if paramCount == 0 {
      return
    }

    // Create or update momentum buffers (m and v)
    let momentumSize = paramCount * MemoryLayout<Float>.size
    if bufferPool["adam_m"] == nil || bufferPool["adam_m"]!.length < momentumSize {
      guard let buffer = device.makeBuffer(length: momentumSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create adam_m buffer")
        return
      }
      // Initialize to zero
      memset(buffer.contents(), 0, momentumSize)
      bufferPool["adam_m"] = buffer
    }

    if bufferPool["adam_v"] == nil || bufferPool["adam_v"]!.length < momentumSize {
      guard let buffer = device.makeBuffer(length: momentumSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create adam_v buffer")
        return
      }
      // Initialize to zero
      memset(buffer.contents(), 0, momentumSize)
      bufferPool["adam_v"] = buffer
    }

    // Create or update gradIds and physicalCells buffers (same as SGD)
    let gradIdsSize = paramCount * MemoryLayout<UInt32>.size
    if bufferPool["gradIds"] == nil || bufferPool["gradIds"]!.length < gradIdsSize {
      guard let buffer = device.makeBuffer(length: gradIdsSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create gradIds buffer")
        return
      }
      bufferPool["gradIds"] = buffer
    }

    let physicalCellsSize = paramCount * MemoryLayout<UInt32>.size
    if bufferPool["physicalCells"] == nil
      || bufferPool["physicalCells"]!.length < physicalCellsSize
    {
      guard
        let buffer = device.makeBuffer(length: physicalCellsSize, options: .storageModeShared)
      else {
        print("⚠️ Failed to create physicalCells buffer")
        return
      }
      bufferPool["physicalCells"] = buffer
    }

    // Copy data to buffers
    if let gradIdsBuffer = bufferPool["gradIds"] {
      let ptr = gradIdsBuffer.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<paramCount {
        ptr[i] = gradIds[i]
      }
    }

    if let physicalCellsBuffer = bufferPool["physicalCells"] {
      let ptr = physicalCellsBuffer.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<paramCount {
        ptr[i] = physicalCells[i]
      }
    }

    guard let gradIdsBuffer = bufferPool["gradIds"],
      let physicalCellsBuffer = bufferPool["physicalCells"],
      let mBuffer = bufferPool["adam_m"],
      let vBuffer = bufferPool["adam_v"]
    else {
      return
    }

    // Create command buffer and encoder
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("⚠️ Failed to create command buffer or encoder")
      return
    }

    guard let pipelineState = updateParametersAdamPSO else {
      print("⚠️ Failed to create updateParametersAdam pipeline state")
      return
    }

    computeEncoder.setComputePipelineState(pipelineState)

    // Set buffers
    computeEncoder.setBuffer(memoryBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(reducedGradsBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(mBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(vBuffer, offset: 0, index: 3)
    computeEncoder.setBuffer(gradIdsBuffer, offset: 0, index: 4)
    computeEncoder.setBuffer(physicalCellsBuffer, offset: 0, index: 5)

    // Set scalar parameters
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

    // Dispatch one thread per parameter
    let threadsPerGrid = MTLSize(width: paramCount, height: 1, depth: 1)
    let threadsPerThreadgroup = MTLSize(
      width: min(paramCount, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  deinit {
    cleanup()
  }
}

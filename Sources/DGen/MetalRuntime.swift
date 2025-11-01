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
  private var context: IRContext
  private let maxFrameCount: Int
  private var firstDebug = true

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
    let combinedSource = kernels.map { $0.source }.joined(separator: "\n\n")
    do {
      self.library = try device.makeLibrary(source: combinedSource, options: nil)
    } catch {
      print("Metal compilation error:")
      print("=== Combined Metal Source ===")
      print(combinedSource)
      print("==============================")
      throw error
    }

    // Get function references
    for kernel in kernels {
      guard let function = library.makeFunction(name: kernel.name) else {
        throw NSError(
          domain: "MetalError", code: 3,
          userInfo: [NSLocalizedDescriptionKey: "Function \(kernel.name) not found"])
      }
      functions.append(function)
    }

    try initializeBuffers()
  }

  private func getElementCount(_ bufferName: String) -> Int {
    return bufferName == "frameCount"
      ? 1
      : bufferName == "memory" || bufferName == "grad_memory"
        ? getMemorySize()
        : bufferName == "t"
          ? maxFrameCount * context.globals.count
          : bufferName == "gradients"
            // Use maxGradId + 1 to ensure we have enough slots (gradIds start at 1)
            ? 2 * maxFrameCount * (context.maxGradId + 1) : maxFrameCount

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
      print("name=\(bufferName) buffer size = \(bufferSize)")

      // Initialize buffer contents to zero
      let bufferContents = buffer.contents().assumingMemoryBound(to: Float.self)
      for i in 0..<(bufferSize / MemoryLayout<Float>.size) {
        bufferContents[i] = 0.0
      }
      bufferPool[bufferName] = buffer
    }

    // Create frameCount buffer for scalar kernels
    guard
      let frameCountBuffer = device.makeBuffer(
        length: MemoryLayout<Int32>.size, options: .storageModeShared)
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
      let frameCountPtr = frameCountBuffer.contents().assumingMemoryBound(to: Int32.self)
      frameCountPtr[0] = Int32(frameCount)
    }

    guard var commandBuffer = commandQueue.makeCommandBuffer() else { return }

    var pending = 0

    func flush() {
      commandBuffer.commit()
      commandBuffer.waitUntilScheduled()
      commandBuffer = commandQueue.makeCommandBuffer()!
      pending = 0
    }

    //resetGradientBuffers(numFrames: frameCount)

    // Execute kernels in sequence
    for (index, kernel) in kernels.enumerated() {
      if firstDebug {
        print("   [DEBUG] Executing kernel \(index): \(kernel.name)")
      }
      guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { continue }

      let function = functions[index]
      guard let pipelineState = try? device.makeComputePipelineState(function: function)
      else { continue }

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
            let segPtr = segBuf.contents().assumingMemoryBound(to: Int32.self)
            segPtr[0] = Int32(thisLen)
          }
          // Update segmentBase (in frames)
          if let segBaseBuf = bufferPool["segmentBase"] {
            let basePtr = segBaseBuf.contents().assumingMemoryBound(to: Int32.self)
            basePtr[0] = Int32(base)
          }

          // Bind buffers with per-segment offsets
          for (bufferIndex, bufferName) in kernel.buffers.enumerated() {
            if firstDebug {
              //print(
              //    "kernel[\(index) setting buffer=\(bufferName) index=\(bufferIndex)]"
              //)
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
        computeEncoder.endEncoding()
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

        if kernel.kind == .scalar {
          // exactly one thread
          let threads = MTLSize(width: 1, height: 1, depth: 1)
          let tpg = MTLSize(width: 1, height: 1, depth: 1)
          computeEncoder.dispatchThreads(threads, threadsPerThreadgroup: tpg)
        } else {
          // 1D over frames
          let tpgW = min(kernel.threadGroupSize ?? 64, tgMax)  // pick a sane default, clamp to device cap
          // Round up total threads to a multiple of tpgW
          let total = MTLSize(width: frameCount, height: 1, depth: 1)
          let tpg = MTLSize(width: tpgW, height: 1, depth: 1)
          computeEncoder.dispatchThreads(total, threadsPerThreadgroup: tpg)
        }
        computeEncoder.endEncoding()
        /*
        pending += 1
        if pending == 5 {
          print("FLUSHING at kernel=\(index)")
          flush()
          print("finished flush")
        }  // tune 4 to taste

         */
      }
    }

    firstDebug = false
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Copy results back to outputs
    copyResultsToOutputs(outputs: outputs, frameCount: frameCount, volumeScale: volumeScale)
  }

  private func copyResultsToOutputs(
    outputs: UnsafeMutablePointer<Float>, frameCount: Int, volumeScale: Float
  ) {
    // Find output buffers and copy their contents
    // First check for dedicated "outputs" buffer
    if let outputBuffer = bufferPool["outputs"] {
      let bufferContents = outputBuffer.contents().assumingMemoryBound(to: Float.self)
      // Copy all frameCount frames (not limited to MIN_FRAME_COUNT)
      for i in 0..<frameCount {
        outputs[i] = bufferContents[i] * volumeScale
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

  deinit {
    cleanup()
  }
}

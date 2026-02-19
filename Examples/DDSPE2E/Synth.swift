import DGenLazy
import Foundation

enum DDSPSynth {
  /// Pre-allocated tensors for synth rendering. Created once before the training loop.
  struct PreallocatedTensors {
    let f0: Tensor
    let uv: Tensor
    let harmonicIndices: Tensor
    let target: Tensor?
    let batchSize: Int

    init(featureFrames: Int, numHarmonics: Int, batchSize: Int = 1, frameCount: Int = 0) {
      self.batchSize = batchSize
      let K = numHarmonics
      let B = batchSize

      if B > 1 {
        // Batched mode: f0 [F, B], uv [F, B], harmonicIndices [B, K], target [frameCount, B]
        let zeroRow = [Float](repeating: 0, count: B)
        self.f0 = Tensor([[Float]](repeating: zeroRow, count: featureFrames))
        self.uv = Tensor([[Float]](repeating: zeroRow, count: featureFrames))
        let harmonicRow = (0..<K).map { Float($0 + 1) }
        self.harmonicIndices = Tensor([[Float]](repeating: harmonicRow, count: B))
        self.target = Tensor([[Float]](repeating: [Float](repeating: 0, count: B), count: frameCount))
      } else {
        self.f0 = Tensor([Float](repeating: 0, count: featureFrames))
        self.uv = Tensor([Float](repeating: 0, count: featureFrames))
        self.harmonicIndices = Tensor((0..<K).map { Float($0 + 1) })
        self.target = nil
      }
    }

    func updateChunkData(f0Frames: [Float], uvFrames: [Float]) {
      f0.updateDataLazily(f0Frames)
      uv.updateDataLazily(uvFrames)
    }

    /// Update tensors for batched mode. Data must be in time-major (interleaved) layout.
    /// f0Interleaved: [F*B] with f0[frame0_b0, frame0_b1, ..., frame1_b0, ...]
    /// uvInterleaved: same layout
    /// audioInterleaved: [frameCount*B] with audio[t0_b0, t0_b1, ..., t1_b0, ...]
    func updateBatchedData(
      f0Interleaved: [Float],
      uvInterleaved: [Float],
      audioInterleaved: [Float]
    ) {
      f0.updateDataLazily(f0Interleaved)
      uv.updateDataLazily(uvInterleaved)
      target?.updateDataLazily(audioInterleaved)
    }
  }

  /// Render batched audio from B chunks in a single pass. Returns [B]-shaped SignalTensor.
  static func renderBatchedSignal(
    controls: DecoderControls,
    tensors: PreallocatedTensors,
    batchSize: Int,
    featureFrames: Int,
    frameCount: Int,
    numHarmonics: Int
  ) -> SignalTensor {
    let B = batchSize
    let K = numHarmonics
    let F = featureFrames

    // Playhead: scalar accumulator stepping through feature frames
    let featureMaxIndex = Float(max(0, F - 1))
    let frameDenom = Float(max(1, frameCount - 1))
    let playheadStep = featureMaxIndex / frameDenom
    let playheadRaw = Signal.accum(
      Signal.constant(playheadStep),
      reset: 0.0,
      min: 0.0,
      max: featureMaxIndex
    )
    let playheadMaxSafe = max(0.0, featureMaxIndex - 1e-4)
    let playhead = playheadRaw.clip(0.0, Double(playheadMaxSafe))

    // Reshape model outputs from [B*F, ...] to time-major for sampling
    // harmonicAmps: [B*F, K] → [B, F, K] → [F, B, K]
    let amps3D = controls.harmonicAmps.reshape([B, F, K]).transpose([1, 0, 2])
    // harmonicGain: [B*F, 1] → [B, F] → [F, B]
    let gain2D = controls.harmonicGain.reshape([B, F]).transpose([1, 0])

    // Sample at playhead position: [F, B, K].sample(playhead) → [B, K]
    let ampsAtTime = amps3D.sample(playhead)
    // [F, B].sample(playhead) → [B]
    let gainAtTime = gain2D.sample(playhead)

    // Sample f0/uv from pre-allocated [F, B] tensors
    // f0/uv are already sanitized on the CPU side during data loading
    let f0AtTime = tensors.f0.sample(playhead)   // [B]
    let uvAtTime = tensors.uv.sample(playhead)   // [B]

    // Harmonic frequencies: harmonicIndices [B, K] * f0 [B] → [B, K]
    let f0Expanded = f0AtTime.reshape([B, 1]).expand([B, K])
    let harmonicFreqs = tensors.harmonicIndices * f0Expanded

    // Synthesize: phasor → sin → weighted sum
    let twoPi = Float.pi * 2.0
    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * twoPi)
    let harmonic = (harmonicSines * ampsAtTime).sum(axis: 1)  // [B, K] → [B]

    // Apply gain and UV mask
    let harmonicOut = harmonic * uvAtTime * gainAtTime * (1.0 / Float(max(1, K)))

    // Consume noiseGain to keep its matmul in the graph (matches unbatched path)
    let noiseGain2D = controls.noiseGain.reshape([B, F]).transpose([1, 0])
    _ = noiseGain2D.sample(playhead)

    return harmonicOut
  }

  static func renderSignal(
    controls: DecoderControls,
    tensors: PreallocatedTensors,
    featureFrames: Int,
    frameCount: Int,
    numHarmonics: Int
  ) -> Signal {
    let featureMaxIndex = Float(max(0, featureFrames - 1))
    let frameDenom = Float(max(1, frameCount - 1))
    let playheadStep = featureMaxIndex / frameDenom

    let playheadRaw = Signal.accum(
      Signal.constant(playheadStep),
      reset: 0.0,
      min: 0.0,
      max: featureMaxIndex
    )
    let playheadMaxSafe = max(0.0, featureMaxIndex - 1e-4)
    let playhead = playheadRaw.clip(0.0, Double(playheadMaxSafe))

    let f0 = min(max(tensors.f0.peek(playhead), 20.0), 500.0)
    let uv = tensors.uv.peek(playhead).clip(0.0, 1.0)

    let twoPi = Float.pi * 2.0
    let ampsAtTime = controls.harmonicAmps.peekRow(playhead)
    let harmonicFreqs = tensors.harmonicIndices * f0
    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * twoPi)
    let harmonic = (harmonicSines * ampsAtTime).sum() * uv

    let harmonicGain = controls.harmonicGain.peek(playhead, channel: Signal.constant(0.0))
    let harmonicOut = harmonic * harmonicGain * (1.0 / Float(max(1, numHarmonics)))

    guard let noiseFilter = controls.noiseFilter else {
      _ = controls.noiseGain.peek(playhead, channel: Signal.constant(0.0))
      return harmonicOut
    }

    let firSize = noiseFilter.shape[1]
    let noiseGain = controls.noiseGain.peek(playhead, channel: Signal.constant(0.0))
    let noiseExcitation = Signal.noise()
    let filterTaps = noiseFilter.peekRow(playhead)              // [firSize] learned per frame
    let noiseBuffer = noiseExcitation.buffer(size: firSize).reshape([firSize])
    let filteredNoise = (noiseBuffer * filterTaps).sum()
    let noiseOut = filteredNoise * noiseGain * (1.0 - uv)
    return harmonicOut + noiseOut
  }
}

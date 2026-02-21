import DGenLazy
import Foundation

enum DDSPSynth {
  // FIR mode applies a frame-domain kernel before time sampling.
  private enum ControlSmoothing {
    static let firTaps: [Float] = [0.1, 0.2, 0.4, 0.2, 0.1]
  }

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
    numHarmonics: Int,
    controlSmoothingMode: ControlSmoothingMode
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
    let amps3DRaw = controls.harmonicAmps.reshape([B, F, K]).transpose([1, 0, 2])
    // harmonicGain: [B*F, 1] → [B, F] → [F, B]
    let gain2DRaw = controls.harmonicGain.reshape([B, F]).transpose([1, 0])
    let noiseGain2DRaw = controls.noiseGain.reshape([B, F]).transpose([1, 0])

    let amps3D: Tensor
    let gain2D: Tensor
    let noiseGain2D: Tensor
    switch controlSmoothingMode {
    case .fir:
      amps3D = firSmoothFrames3D(amps3DRaw)
      gain2D = firSmoothFrames2D(gain2DRaw)
      noiseGain2D = firSmoothFrames2D(noiseGain2DRaw)
    case .off:
      amps3D = amps3DRaw
      gain2D = gain2DRaw
      noiseGain2D = noiseGain2DRaw
    }

    // Sample at playhead position: [F, B, K].sample(playhead) → [B, K]
    let ampsAtTimeRaw = amps3D.sample(playhead)
    // [F, B].sample(playhead) → [B]
    let gainAtTimeRaw = gain2D.sample(playhead)

    let smoothedAmps = ampsAtTimeRaw
    let smoothedGain = gainAtTimeRaw

    // Sample f0/uv from pre-allocated [F, B] tensors
    // f0/uv are already sanitized on the CPU side during data loading
    let f0AtTime = tensors.f0.sample(playhead)   // [B]
    let uvAtTime = tensors.uv.sample(playhead)   // [B]

    // Harmonic frequencies: harmonicIndices [B, K] * f0 [B] → [B, K]
    let f0Expanded = f0AtTime.reshape([B, 1]).expand([B, K])
    let harmonicFreqs = tensors.harmonicIndices * f0Expanded
    let ampsAtTime = nyquistNormalizedHarmonicsBatched(
      amps: smoothedAmps,
      harmonicFreqs: harmonicFreqs,
      batchSize: B,
      harmonicHeadMode: controls.harmonicHeadMode
    )

    // Synthesize: phasor → sin → weighted sum
    let twoPi = Float.pi * 2.0
    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * twoPi)
    let harmonicScale: Float =
      controls.harmonicHeadMode == .legacy ? (1.0 / Float(max(1, K))) : 1.0
    let harmonic = (harmonicSines * ampsAtTime).sum(axis: 1) * harmonicScale  // [B, K] → [B]

    // Apply gain and UV mask
    var harmonicOut = harmonic * uvAtTime * smoothedGain

    // --- Filtered Noise ---
    let noiseGainAtTimeRaw = noiseGain2D.sample(playhead)  // [B]
    let smoothedNoiseGain = noiseGainAtTimeRaw

    if let noiseFilter = controls.noiseFilter {
      let firK = noiseFilter.shape[1]  // noiseFilterSize

      // Reshape filter taps: [B*F, firK] → [B, F, firK] → [F, B, firK]
      let filter3D = noiseFilter.reshape([B, F, firK]).transpose([1, 0, 2])
      let filterTapsAtTime = filter3D.sample(playhead)  // [B, firK]

      // Shared noise → buffer → broadcast → per-batch FIR
      let noise = Signal.noise()
      let noiseBuffer = noise.buffer(size: firK)          // [1, firK]
      let noiseBatch = noiseBuffer.expand([B, firK])      // [B, firK]
      let filteredNoise = (noiseBatch * filterTapsAtTime).sum(axis: 1)  // [B]

      // UV masking + gain
      let noiseOut = filteredNoise * smoothedNoiseGain * (Tensor([1.0]) - uvAtTime)
      harmonicOut = harmonicOut + noiseOut
    }

    return harmonicOut
  }

  static func renderSignal(
    controls: DecoderControls,
    tensors: PreallocatedTensors,
    featureFrames: Int,
    frameCount: Int,
    numHarmonics: Int,
    controlSmoothingMode: ControlSmoothingMode
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

    let harmonicAmpsFrames: Tensor
    let harmonicGainFrames: Tensor
    let noiseGainFrames: Tensor
    if controlSmoothingMode == .fir {
      harmonicAmpsFrames = firSmoothFrames2D(controls.harmonicAmps)
      harmonicGainFrames = firSmoothFrames2D(controls.harmonicGain)
      noiseGainFrames = firSmoothFrames2D(controls.noiseGain)
    } else {
      harmonicAmpsFrames = controls.harmonicAmps
      harmonicGainFrames = controls.harmonicGain
      noiseGainFrames = controls.noiseGain
    }

    let twoPi = Float.pi * 2.0
    let ampsAtTimeRaw = harmonicAmpsFrames.peekRow(playhead)
    let smoothedAmps = ampsAtTimeRaw
    let harmonicFreqs = tensors.harmonicIndices * f0
    let ampsAtTime = nyquistNormalizedHarmonics(
      amps: smoothedAmps,
      harmonicFreqs: harmonicFreqs,
      harmonicHeadMode: controls.harmonicHeadMode
    )
    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * twoPi)
    let harmonicScale: Float =
      controls.harmonicHeadMode == .legacy ? (1.0 / Float(max(1, numHarmonics))) : 1.0
    let harmonic = (harmonicSines * ampsAtTime).sum() * harmonicScale * uv

    let harmonicGainRaw = harmonicGainFrames.peek(playhead, channel: Signal.constant(0.0))
    let harmonicGain = harmonicGainRaw
    let harmonicOut = harmonic * harmonicGain

    let noiseGainRaw = noiseGainFrames.peek(playhead, channel: Signal.constant(0.0))
    let noiseGain = noiseGainRaw

    guard let noiseFilter = controls.noiseFilter else {
      _ = noiseGain
      return harmonicOut
    }

    let firSize = noiseFilter.shape[1]
    let noiseExcitation = Signal.noise()
    let filterTaps = noiseFilter.peekRow(playhead)              // [firSize] learned per frame
    let noiseBuffer = noiseExcitation.buffer(size: firSize).reshape([firSize])
    let filteredNoise = (noiseBuffer * filterTaps).sum()
    let noiseOut = filteredNoise * noiseGain * (1.0 - uv)
    return harmonicOut + noiseOut
  }

  /// Applies Nyquist masking and (for distribution-like heads) renormalization.
  /// This mirrors DDSP's harmonic normalization behavior where above-Nyquist bins
  /// are removed before distribution renormalization.
  private static func nyquistNormalizedHarmonics(
    amps: SignalTensor,
    harmonicFreqs: SignalTensor,
    harmonicHeadMode: HarmonicHeadMode
  ) -> SignalTensor {
    let nyquistHz = Double(DGenConfig.sampleRate * 0.5)
    let nyquistMask = harmonicFreqs < nyquistHz
    let masked = amps * nyquistMask
    guard harmonicHeadMode != .legacy else {
      return masked
    }
    let denom = masked.sum(axis: 0) + 1e-8
    return masked / denom
  }

  private static func nyquistNormalizedHarmonicsBatched(
    amps: SignalTensor,
    harmonicFreqs: SignalTensor,
    batchSize: Int,
    harmonicHeadMode: HarmonicHeadMode
  ) -> SignalTensor {
    let nyquistHz = Double(DGenConfig.sampleRate * 0.5)
    let nyquistMask = harmonicFreqs < nyquistHz
    let masked = amps * nyquistMask
    guard harmonicHeadMode != .legacy else {
      return masked
    }
    let denom = masked.sum(axis: 1).reshape([batchSize, 1]) + 1e-8
    return masked / denom
  }

  private static func firSmoothFrames2D(_ x: Tensor) -> Tensor {
    guard x.shape.count == 2 else { return x }
    let kernel = Tensor(ControlSmoothing.firTaps).reshape([ControlSmoothing.firTaps.count, 1])
    let k = kernel.shape[0]
    guard k > 1 else { return x }

    let left = (k - 1) / 2
    let right = k - 1 - left
    let padded = replicatePadRows2D(x, left: left, right: right)
    return padded.conv2d(kernel)
  }

  private static func replicatePadRows2D(_ x: Tensor, left: Int, right: Int) -> Tensor {
    let frames = x.shape[0]
    let cols = x.shape[1]
    var padded = x.pad([(left, right), (0, 0)])

    if left > 0 {
      let leftEdge = x
        .shrink([(0, 1), nil])
        .expand([left, cols])
        .pad([(0, frames + right), (0, 0)])
      padded = padded + leftEdge
    }

    if right > 0 {
      let rightEdge = x
        .shrink([(max(0, frames - 1), frames), nil])
        .expand([right, cols])
        .pad([(left + frames, 0), (0, 0)])
      padded = padded + rightEdge
    }

    return padded
  }

  private static func firSmoothFrames3D(_ x: Tensor) -> Tensor {
    guard x.shape.count == 3 else { return x }
    let f = x.shape[0]
    let b = x.shape[1]
    let k = x.shape[2]
    let flat = x.reshape([f, b * k])
    let smoothed = firSmoothFrames2D(flat)
    return smoothed.reshape([f, b, k])
  }

}

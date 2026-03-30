import DGenLazy
import Foundation

enum HarmonicSynth {
  struct PreallocatedTensors {
    let f0: Tensor
    let uv: Tensor
    let harmonicIndices: Tensor
    let target: Tensor?
    let batchSize: Int

    init(featureFrames: Int, numHarmonics: Int, batchSize: Int = 1, frameCount: Int = 0) {
      self.batchSize = batchSize
      let B = batchSize
      let K = numHarmonics
      if B > 1 {
        let zeroRow = [Float](repeating: 0, count: B)
        self.f0 = Tensor([[Float]](repeating: zeroRow, count: featureFrames))
        self.uv = Tensor([[Float]](repeating: zeroRow, count: featureFrames))
        let harmonicRow = (0..<K).map { Float($0 + 1) }
        self.harmonicIndices = Tensor([[Float]](repeating: harmonicRow, count: B))
        self.target = Tensor([[Float]](repeating: [Float](repeating: 0, count: B), count: frameCount))
      } else {
        self.f0 = Tensor([Float](repeating: 0, count: featureFrames))
        self.uv = Tensor([Float](repeating: 0, count: featureFrames))
        self.harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
        self.target = nil
      }
    }

    func updateChunkData(f0Frames: [Float], uvFrames: [Float]) {
      f0.updateDataLazily(f0Frames)
      uv.updateDataLazily(uvFrames)
    }

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

  static func renderBatchedSignal(
    controls: HarmonicControls,
    tensors: PreallocatedTensors,
    batchSize: Int,
    featureFrames: Int,
    frameCount: Int,
    numHarmonics: Int,
    harmonicPathScale: Float = 1.0,
    noisePathScale: Float = 1.0
  ) -> SignalTensor {
    let B = batchSize
    let K = numHarmonics
    let F = featureFrames

    let featureMaxIndex = Float(max(0, F - 1))
    let frameDenom = Float(max(1, frameCount - 1))
    let playheadStep = featureMaxIndex / frameDenom
    let playheadRaw = Signal.accum(
      Signal.constant(playheadStep),
      reset: 0.0,
      min: 0.0,
      max: featureMaxIndex
    )
    let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

    let amps3D = controls.harmonicAmps.reshape([B, F, K]).transpose([1, 0, 2])
    let gain2D = controls.harmonicGain.reshape([B, F]).transpose([1, 0])
    let noiseGain2D = controls.noiseGain.reshape([B, F]).transpose([1, 0])
    let filter3D = controls.noiseFilter.reshape([B, F, controls.noiseFilter.shape[1]]).transpose([1, 0, 2])

    let ampsAtTimeRaw = amps3D.sample(playhead)   // [B, K]
    let gainAtTime = gain2D.sample(playhead)      // [B]
    let noiseGainAtTime = noiseGain2D.sample(playhead)
    let filterTapsAtTime = filter3D.sample(playhead)  // [B, FIR]

    let f0AtTime = tensors.f0.sample(playhead)
    let uvAtTime = tensors.uv.sample(playhead)
    let f0Expanded = f0AtTime.reshape([B, 1]).expand([B, K])
    let harmonicFreqs = tensors.harmonicIndices * f0Expanded

    let ampsAtTime = nyquistNormalizedHarmonicsBatched(
      amps: ampsAtTimeRaw,
      harmonicFreqs: harmonicFreqs,
      batchSize: B
    )

    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * (2.0 * Float.pi))
    let harmonicOut = (harmonicSines * ampsAtTime).sum(axis: 1) * gainAtTime * uvAtTime * harmonicPathScale

    let firSize = controls.noiseFilter.shape[1]
    let noiseBuffer = Signal.noise().buffer(size: firSize).expand([B, firSize])
    let filteredNoise = (noiseBuffer * filterTapsAtTime).sum(axis: 1)
    let noiseOut = filteredNoise * noiseGainAtTime * 0.12 * noisePathScale
    return harmonicOut + noiseOut
  }

  static func renderSignal(
    controls: HarmonicControls,
    tensors: PreallocatedTensors,
    featureFrames: Int,
    frameCount: Int,
    numHarmonics: Int,
    harmonicPathScale: Float = 1.0,
    noisePathScale: Float = 1.0
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
    let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

    let f0 = min(max(tensors.f0.peek(playhead), 20.0), Double(DGenConfig.sampleRate * 0.5))
    let uv = tensors.uv.peek(playhead).clip(0.0, 1.0)
    let harmonicFreqs = tensors.harmonicIndices * f0

    let ampsAtTimeRaw = controls.harmonicAmps.peekRow(playhead)
    let ampsAtTime = nyquistNormalizedHarmonics(amps: ampsAtTimeRaw, harmonicFreqs: harmonicFreqs)
    let gain = controls.harmonicGain.peek(playhead, channel: Signal.constant(0.0))
    let noiseGain = controls.noiseGain.peek(playhead, channel: Signal.constant(0.0))

    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * (2.0 * Float.pi))
    let harmonicOut = (harmonicSines * ampsAtTime).sum() * gain * uv * harmonicPathScale

    // Residual branch: frame-conditioned FIR noise shaping.
    // Signed, normalized taps keep the branch expressive without letting it explode.
    let firSize = controls.noiseFilter.shape[1]
    let filterTaps = controls.noiseFilter.peekRow(playhead)
    let noiseBuffer = Signal.noise().buffer(size: firSize).reshape([firSize])
    let filteredNoise = (noiseBuffer * filterTaps).sum()
    let noiseOut = filteredNoise * noiseGain * 0.12 * noisePathScale
    return harmonicOut + noiseOut
  }

  private static func nyquistNormalizedHarmonics(
    amps: SignalTensor,
    harmonicFreqs: SignalTensor
  ) -> SignalTensor {
    let nyquistHz = Double(DGenConfig.sampleRate * 0.5)
    let masked = amps * (harmonicFreqs < nyquistHz)
    let denom = masked.sum(axis: 0) + 1e-8
    return masked / denom
  }

  private static func nyquistNormalizedHarmonicsBatched(
    amps: SignalTensor,
    harmonicFreqs: SignalTensor,
    batchSize: Int
  ) -> SignalTensor {
    let nyquistHz = Double(DGenConfig.sampleRate * 0.5)
    let masked = amps * (harmonicFreqs < nyquistHz)
    let denom = masked.sum(axis: 1).reshape([batchSize, 1]) + 1e-8
    return masked / denom
  }
}

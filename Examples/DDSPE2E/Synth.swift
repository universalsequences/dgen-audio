import DGenLazy
import Foundation

enum DDSPSynth {
  /// Pre-allocated tensors for synth rendering. Created once before the training loop.
  struct PreallocatedTensors {
    let f0: Tensor
    let uv: Tensor
    let harmonicIndices: Tensor
    let firKernel: Tensor?

    init(featureFrames: Int, numHarmonics: Int, enableFIRNoise: Bool, firKernelSize: Int) {
      self.f0 = Tensor([Float](repeating: 0, count: featureFrames))
      self.uv = Tensor([Float](repeating: 0, count: featureFrames))
      self.harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })

      if enableFIRNoise {
        let firSize = max(2, firKernelSize)
        let firTap = 1.0 / Float(firSize)
        self.firKernel = Tensor([[Float](repeating: firTap, count: firSize)])
      } else {
        self.firKernel = nil
      }
    }

    func updateChunkData(f0Frames: [Float], uvFrames: [Float]) {
      f0.updateDataLazily(f0Frames)
      uv.updateDataLazily(uvFrames)
    }
  }

  static func renderSignal(
    controls: DecoderControls,
    tensors: PreallocatedTensors,
    featureFrames: Int,
    frameCount: Int,
    numHarmonics: Int,
    enableStaticFIRNoise: Bool,
    noiseFIRKernelSize: Int
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

    guard enableStaticFIRNoise, let firKernel = tensors.firKernel else {
      _ = controls.noiseGain.peek(playhead, channel: Signal.constant(0.0))
      return harmonicOut
    }

    let firSize = max(2, noiseFIRKernelSize)
    let noiseGain = controls.noiseGain.peek(playhead, channel: Signal.constant(0.0))
    let noiseExcitation = Signal.noise()
    let filteredNoise = noiseExcitation.buffer(size: firSize).conv2d(firKernel).sum()
    let noiseOut = filteredNoise * noiseGain * (1.0 - uv)
    return harmonicOut + noiseOut
  }
}

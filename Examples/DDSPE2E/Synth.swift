import DGenLazy
import Foundation

enum DDSPSynth {
  static func renderSignal(
    controls: DecoderControls,
    f0Frames: [Float],
    uvFrames: [Float],
    frameCount: Int,
    numHarmonics: Int
  ) -> Signal {
    let featureFrames = max(1, f0Frames.count)
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

    let f0Tensor = Tensor(f0Frames)
    let uvTensor = Tensor(uvFrames)

    let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
    let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

    let twoPi = Float.pi * 2.0
    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
    let ampsAtTime = controls.harmonicAmps.peekRow(playhead)
    let harmonicFreqs = harmonicIndices * f0
    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * twoPi)
    let harmonic = (harmonicSines * ampsAtTime).sum() * uv

    let harmonicGain = controls.harmonicGain.peek(playhead, channel: Signal.constant(0.0))
    let harmonicOut = harmonic * harmonicGain * (1.0 / Float(max(1, numHarmonics)))

    // Keep M2 baseline stable by using harmonic-only output first.
    // Noise branch stays in the model for future milestones.
    _ = controls.noiseGain.peek(playhead, channel: Signal.constant(0.0))
    return harmonicOut
  }
}

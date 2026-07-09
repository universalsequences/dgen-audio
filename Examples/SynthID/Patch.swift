import DGenLazy
import Foundation

enum KickVoice {
  static func build(params: KickVoiceSignals, config: SynthIDConfig) -> Signal {
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1) / config.sampleRate)

    // Closed-form phase of the exponential pitch sweep:
    //   ∫ (fEnd + (fStart - fEnd)·e^{pd·τ}) dτ = fEnd·t + (fStart - fEnd)/pd · (e^{pd·t} - 1)
    // Built from the accum time ramp instead of statefulPhasor because gradPhasor
    // applies the constant-frequency rule d(phase)/d(freq) = frameIdx/sr, which is
    // wrong for swept frequency input (fdcheck: fStart autograd 12x low, fEnd 25% high).
    let sweepPhase =
      params.fEnd * t
      + (params.fStart - params.fEnd) / params.pitchDecay
        * (DGenLazy.exp(params.pitchDecay * t) - 1.0)
    let body =
      DGenLazy.sin(sweepPhase * (2.0 * Float.pi)) * DGenLazy.exp(params.ampDecay * t)
      * params.bodyAmp

    let clickPhase = params.clickFreq * t * (2.0 * Float.pi)
    let click = DGenLazy.sin(clickPhase) * DGenLazy.exp(params.clickDecay * t) * params.clickAmp

    var noise = Signal.noise() * 2.0 - 1.0
    if config.enableNoiseFilter {
      noise = noise.biquad(
        cutoff: params.noiseCutoff,
        resonance: Signal.constant(0.707),
        gain: Signal.constant(1.0),
        mode: Signal.constant(0.0))
    }
    let noiseBurst = noise * DGenLazy.exp(params.noiseDecay * t) * params.noiseAmp

    let mixed = body + click + noiseBurst
    return DGenLazy.tanh(mixed * params.drive) * params.outGain
  }

  static func render(values: PatchValues, config: SynthIDConfig, parameterBacked: Bool = true) throws
    -> [Float]
  {
    config.applyRuntime()
    LazyGraphContext.reset()
    let params = TrainableKickParams(
      initial: values,
      trainable: parameterBacked,
      freezePitch: false)
    return try build(params: params.signals, config: config).realize(frames: config.frames)
  }

  static func renderToWav(values: PatchValues, config: SynthIDConfig, out: URL) throws {
    let samples = try render(values: values, config: config)
    try AudioFile.save(url: out, samples: samples, sampleRate: config.sampleRate)
  }
}

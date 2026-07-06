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

    let pitchEnv = params.fEnd + (params.fStart - params.fEnd) * DGenLazy.exp(params.pitchDecay * t)
    let bodyPhase = Signal.statefulPhasor(pitchEnv) * (2.0 * Float.pi)
    let body = DGenLazy.sin(bodyPhase) * DGenLazy.exp(params.ampDecay * t) * params.bodyAmp

    let clickPhase = Signal.statefulPhasor(params.clickFreq) * (2.0 * Float.pi)
    let click = DGenLazy.sin(clickPhase) * DGenLazy.exp(params.clickDecay * t) * params.clickAmp

    var noise = Signal.noise()
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

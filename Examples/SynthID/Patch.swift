import DGenLazy
import Foundation

enum KickVoice {
  static func build(params: TrainableKickParams, config: SynthIDConfig) -> Signal {
    if config.profile == "hoodie-bass" {
      return BassVoice.build(params: params.bassSignals, config: config)
    }
    return build(params: params.signals, config: config)
  }

  static func build(params: KickVoiceSignals, config: SynthIDConfig) -> Signal {
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      // Float32 accumulation drifts by more than one nominal sample over long
      // renders (notably 16,384 frames at 8 kHz). Keep the safety bound clear
      // of the render interval so the final sample cannot wrap to zero.
      max: Float(config.frames + 1) / config.sampleRate + 1.0)

    // Closed-form phase of the exponential pitch sweep:
    //   ∫ (fEnd + (fStart - fEnd)·e^{pd·τ}) dτ = fEnd·t + (fStart - fEnd)/pd · (e^{pd·t} - 1)
    // Closed form avoids phase-wrap discontinuities in the parameterization and
    // remains equivalent to the stateful phasor now that its temporal adjoint is
    // implemented as a reset-aware suffix scan.
    let sweepPhase =
      params.fEnd * t
      + (params.fStart - params.fEnd) / params.pitchDecay
        * (DGenLazy.exp(params.pitchDecay * t) - 1.0)
    // Shared body-family amplitude envelope. The TR-909 target's body decay
    // steepens over time (measured -3.3/s over 20-80ms, -12.4/s over
    // 150-450ms); the zero-default ampCurve term adds that log-quadratic
    // curvature without disturbing the exponential 808 envelope (ampCurve
    // pinned near 0 there).
    let bodyEnv = DGenLazy.exp(params.ampDecay * t + params.ampCurve * (t * t))
    let body =
      DGenLazy.sin(sweepPhase * (2.0 * Float.pi)) * bodyEnv
      * params.bodyAmp
    // The real target's even harmonic is attack-localized: a constant-ratio
    // second harmonic was correctly rejected because it polluted the tail.
    // This zero-default term decays 17/s faster than the fundamental and adds
    // one scalar in the compact base voice.
    let evenHarmonic =
      params.bodyAsymmetry
      * DGenLazy.sin(sweepPhase * (4.0 * Float.pi) - 0.62)
      * bodyEnv * params.bodyAmp
      * DGenLazy.exp(-17.0 * t)

    // The TR-909 measurement shows H3 persisting through ~200 ms (unlike the
    // attack-localized 808 H2 above), so this term uses the same amplitude
    // envelope as the body rather than an extra decay factor. 3rd and 5th
    // harmonics of the swept phase, triangle-series 1/n^2 weights.
    // Zero-default: mathematically inert when bodyHarmonic == 0.
    let oddHarmonics =
      params.bodyHarmonic
      * (DGenLazy.sin(sweepPhase * (6.0 * Float.pi)) * (1.0 / 9.0)
        + DGenLazy.sin(sweepPhase * (10.0 * Float.pi)) * (1.0 / 25.0))
      * bodyEnv * params.bodyAmp

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

    var harmonicCorrection = Signal.constant(0)
    for term in params.harmonicCorrections {
      let angle = sweepPhase * (2.0 * Float.pi * Float(term.spec.harmonic))
      let wave = term.spec.cosine ? DGenLazy.cos(angle) : DGenLazy.sin(angle)
      harmonicCorrection = harmonicCorrection
        + term.coefficient * wave * bodyEnv * params.bodyAmp
          * DGenLazy.exp(-term.spec.decay * t)
    }

    let mixed = body + evenHarmonic + oddHarmonics + harmonicCorrection + click + noiseBurst
    if config.profile == "909" {
      // The 909 VCA/output stage has a gentler knee than tanh. A biased
      // softsign also supplies the measured mild asymmetry without changing
      // the 808 signal path.
      let bias = Signal.constant(0.05)
      let shifted = mixed * params.drive + bias
      let shaped = shifted / (1.0 + DGenLazy.abs(shifted)) - bias / (1.0 + DGenLazy.abs(bias))
      return shaped * params.outGain
    }
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
    return try build(params: params, config: config).realize(frames: config.frames)
  }

  static func renderToWav(values: PatchValues, config: SynthIDConfig, out: URL) throws {
    let samples = try render(values: values, config: config)
    try AudioFile.save(url: out, samples: samples, sampleRate: config.sampleRate)
  }
}

enum BassVoice {
  static func build(params: BassVoiceSignals, config: SynthIDConfig) -> Signal {
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1) / config.sampleRate + 1.0)
    let phase = params.f0 * t * (2.0 * Float.pi)

    // Smooth, closed-form ADSR-like envelope. The logistic release avoids a
    // discontinuous note-off branch while making noteOff and releaseTime
    // differentiable. The measured two-second multisamples have a long
    // sustain followed by a capture-time release near 1.55 seconds.
    let attack = 1.0 - DGenLazy.exp(-t / params.attackTime)
    let decay = params.sustain
      + (1.0 - params.sustain) * DGenLazy.exp(-t / params.decayTime)
    let release = 1.0 / (1.0 + DGenLazy.exp((t - params.noteOff) / params.releaseTime))
    let amplitudeEnvelope = attack * decay * release

    var oscillator = Signal.constant(0)
    for partial in params.harmonics {
      let harmonic = Float(partial.spec.harmonic)
      let angle = phase * harmonic
      let wave = partial.spec.cosine ? DGenLazy.cos(angle) : DGenLazy.sin(angle)
      // The steady bank remains present through the sustain. Three zero-default
      // attack banks use 1x, 2x, and 4x one fitted brightness-decay rate. This
      // models the measured slow, medium, and fast filter-closing timescales without a
      // stateful time-varying filter or any target-derived lookup data.
      let brightnessEnvelope = DGenLazy.exp(
        -params.brightnessDecay * partial.spec.decay * t)
      oscillator = oscillator + partial.coefficient * wave * brightnessEnvelope
    }

    let driven = oscillator * amplitudeEnvelope * params.drive
    let shaped = driven / (1.0 + DGenLazy.abs(driven))
    return shaped * params.outGain
  }
}

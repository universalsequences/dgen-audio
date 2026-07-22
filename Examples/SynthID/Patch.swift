import DGenLazy
import Foundation

enum KickVoice {
  static func build(params: TrainableKickParams, config: SynthIDConfig) -> Signal {
    if config.profile == "hoodie-bass" {
      return BassVoice.build(params: params.bassSignals, config: config)
    }
    if config.profile == "subtractive-bass" {
      return SubtractiveBassVoice.build(params: params.subtractiveBassSignals, config: config)
    }
    if config.profile == "monologue-bass" {
      return MonologueVoice.build(params: params.monologueSignals, config: config)
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

enum SubtractiveBassVoice {
  private static func polyblep(_ phase: Signal, frequency: Signal, sampleRate: Signal) -> Signal {
    let dt = (frequency / sampleRate).clip(0.000001, 0.5)
    let leftX = phase / dt
    let left = 2.0 * leftX - leftX * leftX - 1.0
    let rightX = (phase - 1.0) / dt
    let right = rightX * rightX + 2.0 * rightX + 1.0
    return (phase < dt) * left + (phase > (1.0 - dt)) * right
  }

  private static func polyblepSaw(
    _ phase: Signal, frequency: Signal, sampleRate: Signal
  ) -> Signal {
    (phase * 2.0 - 1.0) - polyblep(phase, frequency: frequency, sampleRate: sampleRate)
  }

  private static func polyblepPulse(
    _ phase: Signal, width: Signal, frequency: Signal, sampleRate: Signal
  ) -> Signal {
    let clippedWidth = width.clip(0.01, 0.99)
    let fallingPhase = mod(phase - clippedWidth, 1.0)
    let rawPulse = (phase < clippedWidth) * 2.0 - 1.0
    return rawPulse
      + polyblep(phase, frequency: frequency, sampleRate: sampleRate)
      - polyblep(fallingPhase, frequency: frequency, sampleRate: sampleRate)
  }

  static func buildOscillator(
    params: SubtractiveBassVoiceSignals, config: SynthIDConfig
  ) -> Signal {
    let sr = Signal.constant(config.sampleRate)
    // Frozen fundamental from the CPU pitch fit (PatchValues.subF0; defaults
    // to the historical 110 Hz for legacy artifacts).
    let frequency = params.f0
    let phase = Signal.statefulPhasor(frequency)
    let saw = polyblepSaw(phase, frequency: frequency, sampleRate: sr)
    let pulse = polyblepPulse(
      phase, width: params.pw, frequency: frequency, sampleRate: sr)
    return (1.0 - params.shape) * saw + params.shape * pulse
  }

  static func renderOscillator(
    values: PatchValues, config: SynthIDConfig
  ) throws -> [Float] {
    config.applyRuntime()
    LazyGraphContext.reset()
    let params = TrainableKickParams(
      initial: values,
      trainable: false,
      freezePitch: false)
    return try buildOscillator(params: params.subtractiveBassSignals, config: config)
      .realize(frames: config.frames)
  }

  static func build(params: SubtractiveBassVoiceSignals, config: SynthIDConfig) -> Signal {
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1) / config.sampleRate + 1.0)

    // f0 and note-off remain fixed for the first subtractive topology; E1
    // trains the complete oscillator/filter/VCA/output patch sheet.
    let oscillator = buildOscillator(params: params, config: config)

    let cutoff = params.fBase + params.fAmt * DGenLazy.exp(-t / params.fDecay)
    // Reuse the existing diagnostic filter bypass so fdcheck failures can be
    // isolated to the oscillator/STFT path versus the biquad input adjoint.
    let filtered = config.enableNoiseFilter
      ? oscillator.biquad(
        cutoff: cutoff,
        resonance: params.res,
        gain: Signal.constant(1.0),
        mode: Signal.constant(0.0))
      : oscillator

    let attack = 1.0 - DGenLazy.exp(-t / params.attackTime)
    let decay = params.sustain
      + (1.0 - params.sustain) * DGenLazy.exp(-t / params.decayTime)
    let release = 1.0 / (1.0 + DGenLazy.exp((t - params.noteOff) / params.releaseTime))
    let driven = filtered * attack * decay * release * params.drive
    let shaped = driven / (1.0 + DGenLazy.abs(driven))
    return shaped * params.outGain
  }
}

/// Circuit-modeling subtractive voice: two detuned polyblep VCOs -> mixer ->
/// asymmetric polynomial saturator -> ZDF SVF lowpass with feedback-state
/// saturation -> VCA -> softsign drive. Every circuit stage is inert at its
/// default (vco2Level=0, satA*=0/satBias=0/satGain=1, filtSat=0 -> exact
/// linear Cytomic SVF), so the voice degrades to a plain subtractive patch.
enum MonologueVoice {
  private static func oscillator(
    phase: Signal, frequency: Signal, shape: Signal, pw: Signal, sampleRate: Signal
  ) -> Signal {
    let dt = (frequency / sampleRate).clip(0.000001, 0.5)
    let blep: (Signal) -> Signal = { p in
      let lX = p / dt
      let l = 2.0 * lX - lX * lX - 1.0
      let rX = (p - 1.0) / dt
      let r = rX * rX + 2.0 * rX + 1.0
      return (p < dt) * l + (p > (1.0 - dt)) * r
    }
    let saw = (phase * 2.0 - 1.0) - blep(phase)
    let clippedWidth = pw.clip(0.01, 0.99)
    let fallingPhase = mod(phase - clippedWidth, 1.0)
    let rawPulse = (phase < clippedWidth) * 2.0 - 1.0
    let pulse = rawPulse + blep(phase) - blep(fallingPhase)
    return (1.0 - shape) * saw + shape * pulse
  }

  /// Softsign with strength k: x / (1 + |k*x|). k = 0 is the identity.
  private static func stateSat(_ x: Signal, k: Signal) -> Signal {
    x / (1.0 + DGenLazy.abs(k * x))
  }

  static func build(params: MonologueVoiceSignals, config: SynthIDConfig) -> Signal {
    let sr = Signal.constant(config.sampleRate)
    let t = Signal.accum(
      Signal.constant(1.0) / sr,
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1) / config.sampleRate + 1.0)

    // VCO2's detuned phase is closed-form: wrap(phase1 - t*detune). A
    // trainable statefulPhasor FREQUENCY corrupts unrelated gradients (see
    // SVFBPTTScratchTests.testFullVoiceManyTargetsGradientsWithTrainableDetune);
    // the phase-offset form keeps vco2Detune differentiable through ordinary
    // arithmetic (d wrap/d in = 1 a.e.), per the playbook's closed-form rule.
    let phase1 = Signal.statefulPhasor(params.f0)
    let phase2 = mod(phase1 - t * params.vco2Detune, 1.0)
    let osc1 = oscillator(
      phase: phase1, frequency: params.f0, shape: params.shape, pw: params.pw,
      sampleRate: sr)
    let osc2 = oscillator(
      phase: phase2, frequency: params.f0, shape: params.shape, pw: params.pw,
      sampleRate: sr)
    let mixed = (osc1 + params.vco2Level * osc2) / (1.0 + params.vco2Level)

    // Asymmetric polynomial saturator; the bias operating point's DC is
    // subtracted so the stage is DC-free and exactly inert at a*=0.
    let y = params.satGain * mixed + params.satBias
    let b = params.satBias
    let shapedPre =
      y + params.satA2 * (y * y) + params.satA3 * (y * y * y)
      + params.satA5 * (y * y * y * y * y)
    let dcAtBias =
      b + params.satA2 * (b * b) + params.satA3 * (b * b * b)
      + params.satA5 * (b * b * b * b * b)
    let preSat = shapedPre - dcAtBias

    // ZDF SVF lowpass (Cytomic form; mirrors the eseq deployment `svf`
    // macro) with per-sample envelope-driven cutoff and softsign-saturated
    // integrator states (filtSat = 0 -> exact linear SVF).
    let cutoffHz = (params.fBase + params.fAmt * DGenLazy.exp(-t / params.fDecay))
      .clip(20.0, Double(config.sampleRate) * 0.49)
    let g = DGenLazy.tan(Signal.constant(Float.pi) * cutoffHz / sr)
    let kDamp = 1.0 / params.res
    let a1 = 1.0 / (1.0 + g * (g + kDamp))
    let a2 = g * a1

    // Pass-through history writes are REQUIRED for correct BPTT: dangling
    // writes (`_ = write(...)`) leave gradient carry cells unconsumed and the
    // temporal gradient silently truncates (docs/BIQUAD_BPTT_GRADIENT_BUG.md
    // Part B). Each write's output is routed back into the arithmetic with
    // identical forward values: write returns 2v-s, so v = (write+s)/2, and
    // v2 = s2 + g*v1 is the standard equivalent form of s2 + a2*s1 + a3*v3.
    let ic1 = Signal.history()
    let ic2 = Signal.history()
    let s1 = stateSat(ic1.read, k: params.filtSat)
    let s2 = stateSat(ic2.read, k: params.filtSat)
    let v3 = preSat - s2
    let v1 = a1 * s1 + a2 * v3
    let ic1New = ic1.write(2.0 * v1 - s1)
    let v1PassThrough = (ic1New + s1) * 0.5
    let v2 = s2 + g * v1PassThrough
    let ic2New = ic2.write(2.0 * v2 - s2)
    let lowpass = (ic2New + s2) * 0.5

    let attack = 1.0 - DGenLazy.exp(-t / params.attackTime)
    let decay = params.sustain
      + (1.0 - params.sustain) * DGenLazy.exp(-t / params.decayTime)
    let release = 1.0 / (1.0 + DGenLazy.exp((t - params.noteOff) / params.releaseTime))
    let driven = lowpass * attack * decay * release * params.drive
    let shaped = driven / (1.0 + DGenLazy.abs(driven))
    return shaped * params.outGain
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

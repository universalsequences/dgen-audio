import DGenLazy
import DGenTrainProtocol
import Foundation

enum Reparameterization: String, Codable {
  case raw
  case log
  case logOnePlus = "log(1+x)"
  case logNegative = "log(-x)"
}

struct ParameterSpec: Codable {
  var name: String
  var unit: String
  var min: Float
  var max: Float
  var reparam: Reparameterization
  var tolerance: Float

  func transform(_ natural: Float) -> Float {
    switch reparam {
    case .raw:
      return natural
    case .log:
      return Foundation.log(Swift.max(natural, 1e-20))
    case .logOnePlus:
      return Foundation.log1p(Swift.max(natural, 0))
    case .logNegative:
      return Foundation.log(Swift.max(-natural, 1e-20))
    }
  }

  func inverse(_ transformed: Float) -> Float {
    switch reparam {
    case .raw:
      return transformed
    case .log:
      return Foundation.exp(transformed)
    case .logOnePlus:
      return Foundation.expm1(transformed)
    case .logNegative:
      return -Foundation.exp(transformed)
    }
  }

  var transformedBounds: (min: Float, max: Float) {
    let a = transform(min)
    let b = transform(max)
    return (Swift.min(a, b), Swift.max(a, b))
  }

  var midpoint: Float {
    switch reparam {
    case .raw:
      return (min + max) * 0.5
    case .log:
      return Foundation.exp((Foundation.log(min) + Foundation.log(max)) * 0.5)
    case .logOnePlus:
      return Foundation.expm1((Foundation.log1p(min) + Foundation.log1p(max)) * 0.5)
    case .logNegative:
      let lo = Foundation.log(-min)
      let hi = Foundation.log(-max)
      return -Foundation.exp((lo + hi) * 0.5)
    }
  }
}

enum KickParamSpecs {
  struct HarmonicCorrection: Sendable {
    let name: String
    let harmonic: Int
    let decay: Float
    let cosine: Bool
  }

  // Compact result of pruning an overcomplete 909 Fourier/envelope basis.
  // These remain ordinary trainable scalars; no waveform samples or target
  // tables enter the patch. The 808 profile does not include them.
  static let tr909HarmonicCorrections: [HarmonicCorrection] = [
    ("d0h10s", 10, 0, false), ("d0h12s", 12, 0, false),
    ("d0h16c", 16, 0, true), ("d0h2c", 2, 0, true),
    ("d0h2s", 2, 0, false), ("d0h3c", 3, 0, true),
    ("d0h3s", 3, 0, false), ("d0h4c", 4, 0, true),
    ("d0h5c", 5, 0, true), ("d0h5s", 5, 0, false),
    ("d0h6s", 6, 0, false), ("d0h7c", 7, 0, true),
    ("d0h8c", 8, 0, true), ("d0h9c", 9, 0, true),
    ("d0h9s", 9, 0, false), ("d15h14s", 14, 15, false),
    ("d15h2c", 2, 15, true), ("d15h3c", 3, 15, true),
    ("d15h3s", 3, 15, false), ("d15h5c", 5, 15, true),
    ("d15h5s", 5, 15, false), ("d15h6c", 6, 15, true),
    ("d15h9s", 9, 15, false), ("d15h10s", 10, 15, false),
    ("d15h16s", 16, 15, false), ("d15h2s", 2, 15, false),
    ("d240h12s", 12, 240, false), ("d240h15c", 15, 240, true),
    ("d240h3s", 3, 240, false), ("d240h4s", 4, 240, false),
    ("d240h3c", 3, 240, true),
    ("d60h10s", 10, 60, false), ("d60h12s", 12, 60, false),
    ("d60h14c", 14, 60, true), ("d60h16c", 16, 60, true),
    ("d60h2c", 2, 60, true), ("d60h2s", 2, 60, false),
    ("d60h3c", 3, 60, true), ("d60h5s", 5, 60, false),
    ("d60h6s", 6, 60, false),
  ].map { HarmonicCorrection(name: $0.0, harmonic: $0.1, decay: Float($0.2), cosine: $0.3) }

  /// Which parameter table `all`/`byName` resolve to. Set once at the top of
  /// each command (via `SynthIDConfig.applyRuntime()`) so the whole program
  /// sees one consistent table for the duration of a run.
  static var activeProfile: String = "808"

  static let tr808: [ParameterSpec] = [
    .init(name: "fStart", unit: "Hz", min: 80, max: 180, reparam: .log, tolerance: 0.03),
    .init(name: "fEnd", unit: "Hz", min: 35, max: 60, reparam: .log, tolerance: 0.03),
    // logNegative, not raw: in the pitch group (small LR) a raw value at scale ~20
    // can only travel ~0.4% over a run — pitchDecay froze at init and fStart
    // equilibrated at a compensating wrong value (seed-7 plateau at fStart≈124).
    .init(name: "pitchDecay", unit: "1/s", min: -80, max: -15, reparam: .logNegative, tolerance: 0.10),
    .init(name: "bodyAmp", unit: "lin", min: 0.5, max: 1.0, reparam: .raw, tolerance: 0.10),
    .init(name: "ampDecay", unit: "1/s", min: -12, max: -3, reparam: .raw, tolerance: 0.10),
    .init(name: "clickFreq", unit: "Hz", min: 600, max: 3000, reparam: .log, tolerance: 0.10),
    .init(name: "clickAmp", unit: "lin", min: 0.05, max: 1.5, reparam: .raw, tolerance: 0.20),
    .init(name: "clickDecay", unit: "1/s", min: -1600, max: -200, reparam: .logNegative, tolerance: 0.20),
    .init(name: "noiseCutoff", unit: "Hz", min: 1000, max: 20000, reparam: .log, tolerance: 0.10),
    .init(name: "noiseAmp", unit: "lin", min: 0.0, max: 0.3, reparam: .raw, tolerance: 0.20),
    .init(name: "noiseDecay", unit: "1/s", min: -400, max: -0.001, reparam: .logNegative, tolerance: 0.20),
    .init(name: "drive", unit: "lin", min: 1.0, max: 3.0, reparam: .raw, tolerance: 0.10),
    .init(name: "outGain", unit: "lin", min: 0.4, max: 1.0, reparam: .raw, tolerance: 0.10),
    .init(name: "bodyAsymmetry", unit: "lin", min: -0.5, max: 0.5, reparam: .raw, tolerance: 0.20),
    // Zero-default, mathematically inert at 0 (see Patch.swift's oddHarmonics
    // term): does not change 808 numerics unless explicitly trained/set.
    .init(name: "bodyHarmonic", unit: "lin", min: -1.0, max: 1.0, reparam: .raw, tolerance: 0.20),
    // Zero-default log-quadratic envelope curvature term (see Patch.swift's
    // bodyEnv). Pinned to a near-zero range on the 808 table so it stays
    // mathematically inert there; the 909 target's steepening decay needs a
    // wide negative range (see tr909 below).
    .init(name: "ampCurve", unit: "1/s^2", min: -0.001, max: 0.001, reparam: .raw, tolerance: 0.2),
  ]

  // TR-909 kick bounds, derived from measurement of Assets/909kick.wav (see
  // SPEC.md rung-3 909 profile notes): fStart 150-400 Hz, fEnd 35-60 Hz,
  // pitchDecay -80..-20 1/s, wider click/noise/drive ranges than the 808 table.
  static let tr909Base: [ParameterSpec] = [
    .init(name: "fStart", unit: "Hz", min: 150, max: 400, reparam: .log, tolerance: 0.03),
    .init(name: "fEnd", unit: "Hz", min: 35, max: 60, reparam: .log, tolerance: 0.03),
    .init(name: "pitchDecay", unit: "1/s", min: -80, max: -20, reparam: .logNegative, tolerance: 0.10),
    .init(name: "bodyAmp", unit: "lin", min: 0.05, max: 1.0, reparam: .raw, tolerance: 0.10),
    .init(name: "ampDecay", unit: "1/s", min: -25, max: -3, reparam: .raw, tolerance: 0.10),
    .init(name: "clickFreq", unit: "Hz", min: 200, max: 1000, reparam: .log, tolerance: 0.10),
    .init(name: "clickAmp", unit: "lin", min: 0.0, max: 1.2, reparam: .raw, tolerance: 0.20),
    .init(name: "clickDecay", unit: "1/s", min: -800, max: -150, reparam: .logNegative, tolerance: 0.20),
    .init(name: "noiseCutoff", unit: "Hz", min: 1000, max: 18000, reparam: .log, tolerance: 0.10),
    .init(name: "noiseAmp", unit: "lin", min: 0.0, max: 0.05, reparam: .raw, tolerance: 0.20),
    .init(name: "noiseDecay", unit: "1/s", min: -150, max: -5, reparam: .logNegative, tolerance: 0.20),
    .init(name: "drive", unit: "lin", min: 1.0, max: 6.0, reparam: .raw, tolerance: 0.10),
    .init(name: "outGain", unit: "lin", min: 0.1, max: 1.0, reparam: .raw, tolerance: 0.10),
    .init(name: "bodyAsymmetry", unit: "lin", min: -0.5, max: 0.5, reparam: .raw, tolerance: 0.20),
    .init(name: "bodyHarmonic", unit: "lin", min: -1.0, max: 1.0, reparam: .raw, tolerance: 0.20),
    // The 909 target's body envelope decay steepens over time (measured
    // -3.3/s over 20-80ms, -12.4/s over 150-450ms); this adds a t^2 term to
    // the shared body-family envelope exp(ampDecay*t + ampCurve*t*t).
    .init(name: "ampCurve", unit: "1/s^2", min: -60.0, max: 0.0, reparam: .raw, tolerance: 0.2),
  ]
  static let tr909: [ParameterSpec] = tr909Base + tr909HarmonicCorrections.map {
    .init(name: $0.name, unit: "lin", min: -0.6, max: 0.6, reparam: .raw, tolerance: 0.2)
  }

  /// Additive, band-limited oscillator used for the Hoodie Bass Monologue
  /// profile. Sine and cosine coefficients make each partial's phase
  /// identifiable without storing a target waveform. Integer harmonics keep
  /// the result playable at pitches other than the fitted C sample.
  private static let hoodieBassSteadyHarmonics: [HarmonicCorrection] =
    (1...32).flatMap { harmonic in [
      HarmonicCorrection(name: "h\(harmonic)s", harmonic: harmonic, decay: 0, cosine: false),
      HarmonicCorrection(name: "h\(harmonic)c", harmonic: harmonic, decay: 0, cosine: true),
    ] }
  private static let hoodieBassSlowHarmonics: [HarmonicCorrection] =
    (2...32).flatMap { harmonic in [
      HarmonicCorrection(name: "bh\(harmonic)s", harmonic: harmonic, decay: 1, cosine: false),
      HarmonicCorrection(name: "bh\(harmonic)c", harmonic: harmonic, decay: 1, cosine: true),
    ] }
  private static let hoodieBassMediumHarmonics: [HarmonicCorrection] =
    (2...32).flatMap { harmonic in [
      HarmonicCorrection(name: "mh\(harmonic)s", harmonic: harmonic, decay: 2, cosine: false),
      HarmonicCorrection(name: "mh\(harmonic)c", harmonic: harmonic, decay: 2, cosine: true),
    ] }
  private static let hoodieBassFastHarmonics: [HarmonicCorrection] =
    (2...32).flatMap { harmonic in [
      HarmonicCorrection(name: "fh\(harmonic)s", harmonic: harmonic, decay: 4, cosine: false),
      HarmonicCorrection(name: "fh\(harmonic)c", harmonic: harmonic, decay: 4, cosine: true),
    ] }
  static let hoodieBassHarmonics: [HarmonicCorrection] =
    hoodieBassSteadyHarmonics + hoodieBassSlowHarmonics
    + hoodieBassMediumHarmonics + hoodieBassFastHarmonics

  static let hoodieBassBase: [ParameterSpec] = [
    .init(name: "f0", unit: "Hz", min: 25, max: 130, reparam: .log, tolerance: 0.01),
    .init(name: "attackTime", unit: "s", min: 0.003, max: 0.25, reparam: .log, tolerance: 0.15),
    .init(name: "decayTime", unit: "s", min: 0.03, max: 1.0, reparam: .log, tolerance: 0.20),
    .init(name: "sustain", unit: "lin", min: 0.05, max: 1.0, reparam: .raw, tolerance: 0.15),
    .init(name: "noteOff", unit: "s", min: 1.35, max: 1.75, reparam: .raw, tolerance: 0.05),
    .init(name: "releaseTime", unit: "s", min: 0.02, max: 0.30, reparam: .log, tolerance: 0.20),
    .init(name: "brightnessDecay", unit: "1/s", min: 0, max: 30, reparam: .raw, tolerance: 0.20),
    .init(name: "drive", unit: "lin", min: 0.25, max: 4.0, reparam: .log, tolerance: 0.15),
    .init(name: "outGain", unit: "lin", min: 0.05, max: 1.5, reparam: .log, tolerance: 0.15),
  ]
  static let hoodieBass: [ParameterSpec] = hoodieBassBase + hoodieBassHarmonics.map {
    .init(name: $0.name, unit: "lin", min: -2.0, max: 2.0, reparam: .raw, tolerance: 0.20)
  }

  // Subtractive voice surface. E0 exercised the first six new oscillator and
  // filter parameters; E1 adds the already-established smooth VCA/output
  // controls for self-inversion of the complete deployment topology.
  static let subtractiveBass: [ParameterSpec] = [
    .init(name: "shape", unit: "lin", min: 0, max: 1, reparam: .raw, tolerance: 0.10),
    .init(name: "pw", unit: "lin", min: 0.03, max: 0.97, reparam: .raw, tolerance: 0.10),
    .init(name: "fBase", unit: "Hz", min: 30, max: 8000, reparam: .log, tolerance: 0.10),
    .init(name: "fAmt", unit: "Hz", min: 0, max: 12000, reparam: .logOnePlus, tolerance: 0.10),
    .init(name: "fDecay", unit: "s", min: 0.005, max: 2, reparam: .log, tolerance: 0.10),
    .init(name: "res", unit: "Q", min: 0.5, max: 6, reparam: .log, tolerance: 0.10),
    .init(name: "attackTime", unit: "s", min: 0.001, max: 0.5, reparam: .log, tolerance: 0.10),
    .init(name: "decayTime", unit: "s", min: 0.01, max: 2, reparam: .log, tolerance: 0.10),
    .init(name: "sustain", unit: "lin", min: 0, max: 1, reparam: .raw, tolerance: 0.10),
    .init(name: "releaseTime", unit: "s", min: 0.01, max: 1, reparam: .log, tolerance: 0.10),
    .init(name: "drive", unit: "lin", min: 0.25, max: 8, reparam: .log, tolerance: 0.10),
    .init(name: "outGain", unit: "lin", min: 0.05, max: 2, reparam: .log, tolerance: 0.10),
  ]

  // Circuit-modeling voice (monologue-bass): the subtractive surface plus a
  // second detuned VCO, an asymmetric polynomial pre-filter saturator, and
  // feedback saturation inside the ZDF SVF. Every addition is inert at a
  // value inside its bounds (a*=0, satBias=0, filtSat=0 -> exact linear SVF,
  // vco2Level=0), so the voice degrades to the plain subtractive topology.
  static let monologueBass: [ParameterSpec] =
    subtractiveBass + [
      .init(name: "vco2Level", unit: "lin", min: 0, max: 1.5, reparam: .raw, tolerance: 0.10),
      .init(name: "vco2Detune", unit: "Hz", min: 0.05, max: 2.0, reparam: .log, tolerance: 0.10),
      .init(name: "satGain", unit: "lin", min: 0.25, max: 8, reparam: .log, tolerance: 0.10),
      .init(name: "satBias", unit: "lin", min: -0.4, max: 0.4, reparam: .raw, tolerance: 0.10),
      .init(name: "satA2", unit: "lin", min: -1, max: 1, reparam: .raw, tolerance: 0.10),
      .init(name: "satA3", unit: "lin", min: -1, max: 1, reparam: .raw, tolerance: 0.10),
      .init(name: "satA5", unit: "lin", min: -0.5, max: 0.5, reparam: .raw, tolerance: 0.10),
      .init(name: "filtSat", unit: "lin", min: 0, max: 4, reparam: .raw, tolerance: 0.10),
    ]

  static var all: [ParameterSpec] {
    switch activeProfile {
    case "909": return tr909
    case "hoodie-bass": return hoodieBass
    case "subtractive-bass": return subtractiveBass
    case "monologue-bass": return monologueBass
    default: return tr808
    }
  }

  static var byName: [String: ParameterSpec] {
    Dictionary(uniqueKeysWithValues: all.map { ($0.name, $0) })
  }
}

struct PatchValues: Codable, Equatable {
  var fStart: Float
  var fEnd: Float
  var pitchDecay: Float
  var bodyAmp: Float
  var ampDecay: Float
  var clickFreq: Float
  var clickAmp: Float
  var clickDecay: Float
  var noiseCutoff: Float
  var noiseAmp: Float
  var noiseDecay: Float
  var drive: Float
  var outGain: Float
  var bodyAsymmetry: Float
  var bodyHarmonic: Float
  var ampCurve: Float
  var harmonicCorrections: [String: Float]
  var f0: Float
  var attackTime: Float
  var decayTime: Float
  var sustain: Float
  var noteOff: Float
  var releaseTime: Float
  var brightnessDecay: Float
  var shape: Float
  var pw: Float
  var fBase: Float
  var fAmt: Float
  var fDecay: Float
  var res: Float
  // Frozen documented scalars for the subtractive-bass profile (playbook bass
  // section): oscillator fundamental from the CPU pitch fit and the measured
  // note-off time. Profile-scoped keys so legacy subtractive artifacts (which
  // carry unrelated hoodie-profile f0/noteOff defaults) keep rendering with
  // the historical 110 Hz / 0.6 s constants.
  var subF0: Float
  var subNoteOff: Float
  // monologue-bass circuit stages (inert defaults; absent in older JSONs)
  var vco2Level: Float
  var vco2Detune: Float
  var satGain: Float
  var satBias: Float
  var satA2: Float
  var satA3: Float
  var satA5: Float
  var filtSat: Float

  private enum CodingKeys: String, CodingKey {
    case fStart, fEnd, pitchDecay, bodyAmp, ampDecay
    case clickFreq, clickAmp, clickDecay
    case noiseCutoff, noiseAmp, noiseDecay
    case drive, outGain, bodyAsymmetry, bodyHarmonic, ampCurve, harmonicCorrections
    case f0, attackTime, decayTime, sustain, noteOff, releaseTime, brightnessDecay
    case shape, pw, fBase, fAmt, fDecay, res
    case subF0, subNoteOff
    case vco2Level, vco2Detune, satGain, satBias, satA2, satA3, satA5, filtSat
  }

  init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    self.init(
      fStart: try values.decode(Float.self, forKey: .fStart),
      fEnd: try values.decode(Float.self, forKey: .fEnd),
      pitchDecay: try values.decode(Float.self, forKey: .pitchDecay),
      bodyAmp: try values.decode(Float.self, forKey: .bodyAmp),
      ampDecay: try values.decode(Float.self, forKey: .ampDecay),
      clickFreq: try values.decode(Float.self, forKey: .clickFreq),
      clickAmp: try values.decode(Float.self, forKey: .clickAmp),
      clickDecay: try values.decode(Float.self, forKey: .clickDecay),
      noiseCutoff: try values.decode(Float.self, forKey: .noiseCutoff),
      noiseAmp: try values.decode(Float.self, forKey: .noiseAmp),
      noiseDecay: try values.decode(Float.self, forKey: .noiseDecay),
      drive: try values.decode(Float.self, forKey: .drive),
      outGain: try values.decode(Float.self, forKey: .outGain),
      bodyAsymmetry: try values.decodeIfPresent(Float.self, forKey: .bodyAsymmetry) ?? 0,
      bodyHarmonic: try values.decodeIfPresent(Float.self, forKey: .bodyHarmonic) ?? 0,
      ampCurve: try values.decodeIfPresent(Float.self, forKey: .ampCurve) ?? 0,
      harmonicCorrections: try values.decodeIfPresent([String: Float].self, forKey: .harmonicCorrections) ?? [:],
      f0: try values.decodeIfPresent(Float.self, forKey: .f0) ?? 32.7,
      attackTime: try values.decodeIfPresent(Float.self, forKey: .attackTime) ?? 0.05,
      decayTime: try values.decodeIfPresent(Float.self, forKey: .decayTime) ?? 0.2,
      sustain: try values.decodeIfPresent(Float.self, forKey: .sustain) ?? 0.8,
      noteOff: try values.decodeIfPresent(Float.self, forKey: .noteOff) ?? 1.55,
      releaseTime: try values.decodeIfPresent(Float.self, forKey: .releaseTime) ?? 0.08,
      brightnessDecay: try values.decodeIfPresent(Float.self, forKey: .brightnessDecay) ?? 1.0,
      shape: try values.decodeIfPresent(Float.self, forKey: .shape) ?? 0.5,
      pw: try values.decodeIfPresent(Float.self, forKey: .pw) ?? 0.5,
      fBase: try values.decodeIfPresent(Float.self, forKey: .fBase) ?? 490,
      fAmt: try values.decodeIfPresent(Float.self, forKey: .fAmt) ?? 108.55,
      fDecay: try values.decodeIfPresent(Float.self, forKey: .fDecay) ?? 0.1,
      res: try values.decodeIfPresent(Float.self, forKey: .res) ?? 1.73,
      subF0: try values.decodeIfPresent(Float.self, forKey: .subF0) ?? 110.0,
      subNoteOff: try values.decodeIfPresent(Float.self, forKey: .subNoteOff) ?? 0.6,
      vco2Level: try values.decodeIfPresent(Float.self, forKey: .vco2Level) ?? 0,
      vco2Detune: try values.decodeIfPresent(Float.self, forKey: .vco2Detune) ?? 0.316,
      satGain: try values.decodeIfPresent(Float.self, forKey: .satGain) ?? 1.0,
      satBias: try values.decodeIfPresent(Float.self, forKey: .satBias) ?? 0,
      satA2: try values.decodeIfPresent(Float.self, forKey: .satA2) ?? 0,
      satA3: try values.decodeIfPresent(Float.self, forKey: .satA3) ?? 0,
      satA5: try values.decodeIfPresent(Float.self, forKey: .satA5) ?? 0,
      filtSat: try values.decodeIfPresent(Float.self, forKey: .filtSat) ?? 0)
  }

  static var midpoint: PatchValues {
    var values = [String: Float]()
    for spec in KickParamSpecs.all {
      values[spec.name] = spec.midpoint
    }
    if KickParamSpecs.activeProfile == "hoodie-bass" {
      // A target-independent, square/triangle-like starting spectrum keeps
      // the baseline audible while leaving every coefficient free to move.
      values["h1s"] = 0.35
      values["h2s"] = 0.12
      values["h3s"] = 0.28
      values["h5s"] = 0.08
      values["h7s"] = 0.05
      return PatchValues(values)
    }
    if KickParamSpecs.activeProfile == "subtractive-bass" {
      return PatchValues(values)
    }
    if KickParamSpecs.activeProfile == "909" {
      // The 909 table's plain spec midpoints are already tuned to the
      // measured target; only the zero-default harmonic terms need an
      // explicit override (their midpoint is 0 anyway, but be explicit).
      values["bodyAsymmetry"] = 0
      values["bodyHarmonic"] = 0
      values["ampCurve"] = 0
      return PatchValues(values)
    }
    // Preserve the original Rung 1/2 generic initialization while allowing
    // wider real-target bounds during optimization.
    values["clickAmp"] = 0.325
    values["clickDecay"] = -Foundation.sqrt(900 * 200)
    values["noiseCutoff"] = Foundation.sqrt(1000 * 8000)
    values["noiseDecay"] = -Foundation.sqrt(400 * 60)
    values["bodyAsymmetry"] = 0
    values["bodyHarmonic"] = 0
    values["ampCurve"] = 0
    return PatchValues(values)
  }

  init(
    fStart: Float,
    fEnd: Float,
    pitchDecay: Float,
    bodyAmp: Float,
    ampDecay: Float,
    clickFreq: Float,
    clickAmp: Float,
    clickDecay: Float,
    noiseCutoff: Float,
    noiseAmp: Float,
    noiseDecay: Float,
    drive: Float,
    outGain: Float,
    bodyAsymmetry: Float = 0,
    bodyHarmonic: Float = 0,
    ampCurve: Float = 0,
    harmonicCorrections: [String: Float] = [:],
    f0: Float = 32.7,
    attackTime: Float = 0.05,
    decayTime: Float = 0.2,
    sustain: Float = 0.8,
    noteOff: Float = 1.55,
    releaseTime: Float = 0.08,
    brightnessDecay: Float = 1.0,
    shape: Float = 0.5,
    pw: Float = 0.5,
    fBase: Float = 490,
    fAmt: Float = 108.55,
    fDecay: Float = 0.1,
    res: Float = 1.73,
    subF0: Float = 110.0,
    subNoteOff: Float = 0.6,
    vco2Level: Float = 0,
    vco2Detune: Float = 0.316,
    satGain: Float = 1.0,
    satBias: Float = 0,
    satA2: Float = 0,
    satA3: Float = 0,
    satA5: Float = 0,
    filtSat: Float = 0
  ) {
    self.fStart = fStart
    self.fEnd = fEnd
    self.pitchDecay = pitchDecay
    self.bodyAmp = bodyAmp
    self.ampDecay = ampDecay
    self.clickFreq = clickFreq
    self.clickAmp = clickAmp
    self.clickDecay = clickDecay
    self.noiseCutoff = noiseCutoff
    self.noiseAmp = noiseAmp
    self.noiseDecay = noiseDecay
    self.drive = drive
    self.outGain = outGain
    self.bodyAsymmetry = bodyAsymmetry
    self.bodyHarmonic = bodyHarmonic
    self.ampCurve = ampCurve
    self.harmonicCorrections = harmonicCorrections
    self.f0 = f0
    self.attackTime = attackTime
    self.decayTime = decayTime
    self.sustain = sustain
    self.noteOff = noteOff
    self.releaseTime = releaseTime
    self.brightnessDecay = brightnessDecay
    self.shape = shape
    self.pw = pw
    self.fBase = fBase
    self.fAmt = fAmt
    self.fDecay = fDecay
    self.res = res
    self.subF0 = subF0
    self.subNoteOff = subNoteOff
    self.vco2Level = vco2Level
    self.vco2Detune = vco2Detune
    self.satGain = satGain
    self.satBias = satBias
    self.satA2 = satA2
    self.satA3 = satA3
    self.satA5 = satA5
    self.filtSat = filtSat
  }

  init(_ dictionary: [String: Float]) {
    self.init(
      fStart: dictionary["fStart"] ?? 120,
      fEnd: dictionary["fEnd"] ?? 45,
      pitchDecay: dictionary["pitchDecay"] ?? -35,
      bodyAmp: dictionary["bodyAmp"] ?? 0.75,
      ampDecay: dictionary["ampDecay"] ?? -7,
      clickFreq: dictionary["clickFreq"] ?? 1400,
      clickAmp: dictionary["clickAmp"] ?? 0.2,
      clickDecay: dictionary["clickDecay"] ?? -400,
      noiseCutoff: dictionary["noiseCutoff"] ?? 2800,
      noiseAmp: dictionary["noiseAmp"] ?? 0.08,
      noiseDecay: dictionary["noiseDecay"] ?? -140,
      drive: dictionary["drive"] ?? 1.5,
      outGain: dictionary["outGain"] ?? 0.7,
      bodyAsymmetry: dictionary["bodyAsymmetry"] ?? 0,
      bodyHarmonic: dictionary["bodyHarmonic"] ?? 0,
      ampCurve: dictionary["ampCurve"] ?? 0,
      harmonicCorrections: KickParamSpecs.activeProfile == "909" || KickParamSpecs.activeProfile == "hoodie-bass"
        ? Dictionary(uniqueKeysWithValues: (KickParamSpecs.activeProfile == "909"
          ? KickParamSpecs.tr909HarmonicCorrections : KickParamSpecs.hoodieBassHarmonics).map {
          ($0.name, dictionary[$0.name] ?? 0)
        }) : [:],
      f0: dictionary["f0"] ?? 32.7,
      attackTime: dictionary["attackTime"] ?? 0.05,
      decayTime: dictionary["decayTime"] ?? 0.2,
      sustain: dictionary["sustain"] ?? 0.8,
      noteOff: dictionary["noteOff"] ?? 1.55,
      releaseTime: dictionary["releaseTime"] ?? 0.08,
      brightnessDecay: dictionary["brightnessDecay"] ?? 1.0,
      shape: dictionary["shape"] ?? 0.5,
      pw: dictionary["pw"] ?? 0.5,
      fBase: dictionary["fBase"] ?? 490,
      fAmt: dictionary["fAmt"] ?? 108.55,
      fDecay: dictionary["fDecay"] ?? 0.1,
      res: dictionary["res"] ?? 1.73,
      subF0: dictionary["subF0"] ?? 110.0,
      subNoteOff: dictionary["subNoteOff"] ?? 0.6,
      vco2Level: dictionary["vco2Level"] ?? 0,
      vco2Detune: dictionary["vco2Detune"] ?? 0.316,
      satGain: dictionary["satGain"] ?? 1.0,
      satBias: dictionary["satBias"] ?? 0,
      satA2: dictionary["satA2"] ?? 0,
      satA3: dictionary["satA3"] ?? 0,
      satA5: dictionary["satA5"] ?? 0,
      filtSat: dictionary["filtSat"] ?? 0)
  }

  subscript(name: String) -> Float {
    get {
      switch name {
      case "fStart": return fStart
      case "fEnd": return fEnd
      case "pitchDecay": return pitchDecay
      case "bodyAmp": return bodyAmp
      case "ampDecay": return ampDecay
      case "clickFreq": return clickFreq
      case "clickAmp": return clickAmp
      case "clickDecay": return clickDecay
      case "noiseCutoff": return noiseCutoff
      case "noiseAmp": return noiseAmp
      case "noiseDecay": return noiseDecay
      case "drive": return drive
      case "outGain": return outGain
      case "bodyAsymmetry": return bodyAsymmetry
      case "bodyHarmonic": return bodyHarmonic
      case "ampCurve": return ampCurve
      case "f0": return f0
      case "attackTime": return attackTime
      case "decayTime": return decayTime
      case "sustain": return sustain
      case "noteOff": return noteOff
      case "releaseTime": return releaseTime
      case "brightnessDecay": return brightnessDecay
      case "shape": return shape
      case "pw": return pw
      case "fBase": return fBase
      case "fAmt": return fAmt
      case "fDecay": return fDecay
      case "res": return res
      case "subF0": return subF0
      case "subNoteOff": return subNoteOff
      case "vco2Level": return vco2Level
      case "vco2Detune": return vco2Detune
      case "satGain": return satGain
      case "satBias": return satBias
      case "satA2": return satA2
      case "satA3": return satA3
      case "satA5": return satA5
      case "filtSat": return filtSat
      default: return harmonicCorrections[name] ?? .nan
      }
    }
    set {
      switch name {
      case "fStart": fStart = newValue
      case "fEnd": fEnd = newValue
      case "pitchDecay": pitchDecay = newValue
      case "bodyAmp": bodyAmp = newValue
      case "ampDecay": ampDecay = newValue
      case "clickFreq": clickFreq = newValue
      case "clickAmp": clickAmp = newValue
      case "clickDecay": clickDecay = newValue
      case "noiseCutoff": noiseCutoff = newValue
      case "noiseAmp": noiseAmp = newValue
      case "noiseDecay": noiseDecay = newValue
      case "drive": drive = newValue
      case "outGain": outGain = newValue
      case "bodyAsymmetry": bodyAsymmetry = newValue
      case "bodyHarmonic": bodyHarmonic = newValue
      case "ampCurve": ampCurve = newValue
      case "f0": f0 = newValue
      case "attackTime": attackTime = newValue
      case "decayTime": decayTime = newValue
      case "sustain": sustain = newValue
      case "noteOff": noteOff = newValue
      case "releaseTime": releaseTime = newValue
      case "brightnessDecay": brightnessDecay = newValue
      case "shape": shape = newValue
      case "pw": pw = newValue
      case "fBase": fBase = newValue
      case "fAmt": fAmt = newValue
      case "fDecay": fDecay = newValue
      case "res": res = newValue
      case "subF0": subF0 = newValue
      case "subNoteOff": subNoteOff = newValue
      case "vco2Level": vco2Level = newValue
      case "vco2Detune": vco2Detune = newValue
      case "satGain": satGain = newValue
      case "satBias": satBias = newValue
      case "satA2": satA2 = newValue
      case "satA3": satA3 = newValue
      case "satA5": satA5 = newValue
      case "filtSat": filtSat = newValue
      default:
        if (KickParamSpecs.tr909HarmonicCorrections + KickParamSpecs.hoodieBassHarmonics)
          .contains(where: { $0.name == name }) {
          harmonicCorrections[name] = newValue
        }
      }
    }
  }

  var dictionary: [String: Float] {
    Dictionary(uniqueKeysWithValues: KickParamSpecs.all.map { ($0.name, self[$0.name]) })
  }

  func clamped() -> PatchValues {
    var copy = self
    for spec in KickParamSpecs.all {
      copy[spec.name] = Swift.min(Swift.max(copy[spec.name], spec.min), spec.max)
    }
    return copy
  }

  func withPitch(_ pitch: PitchFit) -> PatchValues {
    var copy = self
    if KickParamSpecs.activeProfile == "hoodie-bass" {
      copy.f0 = pitch.fEnd
      return copy.clamped()
    }
    if KickParamSpecs.activeProfile == "monologue-bass" {
      // subF0 is a frozen documented scalar, not a spec param — clamped()
      // must not touch it, so set it after the spec-table clamp.
      copy = copy.clamped()
      copy.subF0 = pitch.fEnd
      return copy
    }
    copy.fStart = pitch.fStart
    copy.fEnd = pitch.fEnd
    copy.pitchDecay = pitch.pitchDecay
    return copy.clamped()
  }

  static func sample(seed: UInt64) -> PatchValues {
    var rng = SplitMix64(seed: seed)
    var values = [String: Float]()
    if KickParamSpecs.activeProfile == "subtractive-bass" {
      // Keep synthetic truths strictly inside every transformed bound. The
      // shorter VCA ranges ensure the fixed 0.6 s note-off and its release are
      // observable in E1's 32,768-frame render without narrowing deployment
      // bounds or initializing an optimizer on a bound.
      for spec in KickParamSpecs.all {
        let bounds = spec.transformedBounds
        let u = rng.uniform(0.15, 0.85)
        values[spec.name] = spec.inverse(bounds.min + u * (bounds.max - bounds.min))
      }
      values["attackTime"] = Foundation.exp(rng.uniform(Foundation.log(0.003), Foundation.log(0.05)))
      values["decayTime"] = Foundation.exp(rng.uniform(Foundation.log(0.05), Foundation.log(0.5)))
      values["sustain"] = rng.uniform(0.3, 0.9)
      values["releaseTime"] = Foundation.exp(rng.uniform(Foundation.log(0.02), Foundation.log(0.15)))
      values["drive"] = Foundation.exp(rng.uniform(Foundation.log(0.7), Foundation.log(4.0)))
      values["outGain"] = rng.uniform(0.3, 0.8)
      return PatchValues(values)
    }
    let rung12Ranges: [String: (Float, Float)] = [
      "clickAmp": (0.05, 0.6),
      "clickDecay": (-900, -200),
      "noiseCutoff": (1000, 8000),
      "noiseDecay": (-400, -60),
      "bodyAsymmetry": (0, 0),
      "bodyHarmonic": (0, 0),
      "ampCurve": (0, 0),
    ]
    for spec in KickParamSpecs.all {
      let bounds = rung12Ranges[spec.name] ?? (spec.min, spec.max)
      values[spec.name] = rng.uniform(bounds.0, bounds.1)
    }
    if (values["noiseAmp"] ?? 0) < 0.02 {
      values["noiseAmp"] = 0.02
    }
    return PatchValues(values)
  }
}

struct KickVoiceSignals {
  var fStart: Signal
  var fEnd: Signal
  var pitchDecay: Signal
  var bodyAmp: Signal
  var ampDecay: Signal
  var clickFreq: Signal
  var clickAmp: Signal
  var clickDecay: Signal
  var noiseCutoff: Signal
  var noiseAmp: Signal
  var noiseDecay: Signal
  var drive: Signal
  var outGain: Signal
  var bodyAsymmetry: Signal
  var bodyHarmonic: Signal
  var ampCurve: Signal
  var harmonicCorrections: [(spec: KickParamSpecs.HarmonicCorrection, coefficient: Signal)]
}

struct BassVoiceSignals {
  var f0: Signal
  var attackTime: Signal
  var decayTime: Signal
  var sustain: Signal
  var noteOff: Signal
  var releaseTime: Signal
  var brightnessDecay: Signal
  var drive: Signal
  var outGain: Signal
  var harmonics: [(spec: KickParamSpecs.HarmonicCorrection, coefficient: Signal)]
}

struct MonologueVoiceSignals {
  // Frozen constants (not trainable): f0 from the CPU pitch fit, noteOff
  // from the Phase-1 measurement.
  var f0: Signal
  var noteOff: Signal
  // Subtractive surface
  var shape: Signal
  var pw: Signal
  var fBase: Signal
  var fAmt: Signal
  var fDecay: Signal
  var res: Signal
  var attackTime: Signal
  var decayTime: Signal
  var sustain: Signal
  var releaseTime: Signal
  var drive: Signal
  var outGain: Signal
  // Circuit stages
  var vco2Level: Signal
  var vco2Detune: Signal
  var satGain: Signal
  var satBias: Signal
  var satA2: Signal
  var satA3: Signal
  var satA5: Signal
  var filtSat: Signal
}

struct SubtractiveBassVoiceSignals {
  // Frozen constants (not trainable): f0 from the CPU pitch fit, noteOff
  // from the Phase-1 measurement. See PatchValues.subF0/subNoteOff.
  var f0: Signal
  var noteOff: Signal
  var shape: Signal
  var pw: Signal
  var fBase: Signal
  var fAmt: Signal
  var fDecay: Signal
  var res: Signal
  var attackTime: Signal
  var decayTime: Signal
  var sustain: Signal
  var releaseTime: Signal
  var drive: Signal
  var outGain: Signal
}

final class TrainableKickParams {
  private var storage: [String: Signal] = [:]
  private var trainableNames = Set<String>()
  private var frozenNaturalValues: PatchValues

  init(
    initial: PatchValues,
    trainable: Bool,
    freezePitch: Bool = false,
    freezeBodyAsymmetry: Bool = false,
    frozenNames: Set<String> = []
  ) {
    frozenNaturalValues = initial.clamped()
    for spec in KickParamSpecs.all {
      let shouldTrain =
        trainable && !(freezePitch && ["fStart", "fEnd", "pitchDecay", "f0"].contains(spec.name))
          && !(freezeBodyAsymmetry
            && ["bodyAsymmetry", "bodyHarmonic", "ampCurve"].contains(spec.name))
          && !frozenNames.contains(spec.name)
      let transformed = spec.transform(initial[spec.name])
      if shouldTrain {
        let bounds = spec.transformedBounds
        storage[spec.name] = Signal.param(transformed, min: bounds.min, max: bounds.max)
        trainableNames.insert(spec.name)
      } else {
        storage[spec.name] = Signal.constant(transformed)
      }
    }
  }

  var signals: KickVoiceSignals {
    KickVoiceSignals(
      fStart: naturalSignal("fStart"),
      fEnd: naturalSignal("fEnd"),
      pitchDecay: naturalSignal("pitchDecay"),
      bodyAmp: naturalSignal("bodyAmp"),
      ampDecay: naturalSignal("ampDecay"),
      clickFreq: naturalSignal("clickFreq"),
      clickAmp: naturalSignal("clickAmp"),
      clickDecay: naturalSignal("clickDecay"),
      noiseCutoff: naturalSignal("noiseCutoff"),
      noiseAmp: naturalSignal("noiseAmp"),
      noiseDecay: naturalSignal("noiseDecay"),
      drive: naturalSignal("drive"),
      outGain: naturalSignal("outGain"),
      bodyAsymmetry: naturalSignal("bodyAsymmetry"),
      bodyHarmonic: naturalSignal("bodyHarmonic"),
      ampCurve: naturalSignal("ampCurve"),
      harmonicCorrections: KickParamSpecs.tr909HarmonicCorrections.compactMap { spec in
        guard storage[spec.name] != nil else { return nil }
        return (spec, naturalSignal(spec.name))
      })
  }

  var bassSignals: BassVoiceSignals {
    BassVoiceSignals(
      f0: naturalSignal("f0"),
      attackTime: naturalSignal("attackTime"),
      decayTime: naturalSignal("decayTime"),
      sustain: naturalSignal("sustain"),
      noteOff: naturalSignal("noteOff"),
      releaseTime: naturalSignal("releaseTime"),
      brightnessDecay: naturalSignal("brightnessDecay"),
      drive: naturalSignal("drive"),
      outGain: naturalSignal("outGain"),
      harmonics: KickParamSpecs.hoodieBassHarmonics.map { ($0, naturalSignal($0.name)) })
  }

  var subtractiveBassSignals: SubtractiveBassVoiceSignals {
    SubtractiveBassVoiceSignals(
      f0: Signal.constant(frozenNaturalValues.subF0),
      noteOff: Signal.constant(frozenNaturalValues.subNoteOff),
      shape: naturalSignal("shape"),
      pw: naturalSignal("pw"),
      fBase: naturalSignal("fBase"),
      fAmt: naturalSignal("fAmt"),
      fDecay: naturalSignal("fDecay"),
      res: naturalSignal("res"),
      attackTime: naturalSignal("attackTime"),
      decayTime: naturalSignal("decayTime"),
      sustain: naturalSignal("sustain"),
      releaseTime: naturalSignal("releaseTime"),
      drive: naturalSignal("drive"),
      outGain: naturalSignal("outGain"))
  }

  var monologueSignals: MonologueVoiceSignals {
    MonologueVoiceSignals(
      f0: Signal.constant(frozenNaturalValues.subF0),
      noteOff: Signal.constant(frozenNaturalValues.subNoteOff),
      shape: naturalSignal("shape"),
      pw: naturalSignal("pw"),
      fBase: naturalSignal("fBase"),
      fAmt: naturalSignal("fAmt"),
      fDecay: naturalSignal("fDecay"),
      res: naturalSignal("res"),
      attackTime: naturalSignal("attackTime"),
      decayTime: naturalSignal("decayTime"),
      sustain: naturalSignal("sustain"),
      releaseTime: naturalSignal("releaseTime"),
      drive: naturalSignal("drive"),
      outGain: naturalSignal("outGain"),
      vco2Level: naturalSignal("vco2Level"),
      vco2Detune: naturalSignal("vco2Detune"),
      satGain: naturalSignal("satGain"),
      satBias: naturalSignal("satBias"),
      satA2: naturalSignal("satA2"),
      satA3: naturalSignal("satA3"),
      satA5: naturalSignal("satA5"),
      filtSat: naturalSignal("filtSat"))
  }

  var transformedParams: [String: Signal] { storage }

  func trainableStorage(names: [String]) -> [Signal] {
    names.compactMap { trainableNames.contains($0) ? storage[$0] : nil }
  }

  func naturalValues() -> PatchValues {
    // Start from the full frozen struct, NOT frozenNaturalValues.dictionary:
    // the dictionary carries spec-table params only, so non-spec frozen
    // scalars (subF0/subNoteOff) were silently reset to their defaults in
    // every checkpoint (a 35 Hz patch re-rendered at 110 Hz).
    var values = frozenNaturalValues
    for spec in KickParamSpecs.all {
      guard let raw = storage[spec.name]?.data else { continue }
      values[spec.name] = spec.inverse(raw)
    }
    return values.clamped()
  }

  func transformedValues() -> [String: Float] {
    Dictionary(uniqueKeysWithValues: KickParamSpecs.all.map { spec in
      (spec.name, storage[spec.name]?.data ?? spec.transform(frozenNaturalValues[spec.name]))
    })
  }

  func apply(natural values: PatchValues) {
    let clamped = values.clamped()
    frozenNaturalValues = clamped
    for spec in KickParamSpecs.all {
      storage[spec.name]?.updateDataLazily(spec.transform(clamped[spec.name]))
    }
  }

  func clipGradients(maxAbs: Float) {
    guard maxAbs > 0 else { return }
    for signal in storage.values where signal.requiresGrad {
      guard let grad = signal.grad?.data else { continue }
      signal.grad?.data = Swift.max(-maxAbs, Swift.min(maxAbs, grad))
    }
  }

  private func naturalSignal(_ name: String) -> Signal {
    guard let spec = KickParamSpecs.byName[name], let raw = storage[name] else {
      fatalError("missing parameter \(name)")
    }
    switch spec.reparam {
    case .raw:
      return raw
    case .log:
      return DGenLazy.exp(raw)
    case .logOnePlus:
      return DGenLazy.exp(raw) - 1.0
    case .logNegative:
      return -DGenLazy.exp(raw)
    }
  }
}

struct SplitMix64 {
  private var state: UInt64

  init(seed: UInt64) {
    state = seed
  }

  mutating func next() -> UInt64 {
    state &+= 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
  }

  mutating func unit() -> Float {
    let value = next() >> 40
    return Float(value) / Float(1 << 24)
  }

  mutating func uniform(_ min: Float, _ max: Float) -> Float {
    min + (max - min) * unit()
  }
}

func loadPatchValues(from url: URL) throws -> PatchValues {
  let data = try Data(contentsOf: url)
  let decoder = JSONDecoder()
  if let direct = try? decoder.decode(PatchValues.self, from: data) {
    return direct.clamped()
  }
  if let checkpoint = try? decoder.decode(SynthIDCheckpoint.self, from: data) {
    return checkpoint.params.clamped()
  }
  if let wrapped = try? decoder.decode(WrappedPatchValues.self, from: data) {
    return PatchValues(wrapped.params).clamped()
  }
  throw SynthIDError.message("could not decode patch parameters from \(url.path)")
}

private struct WrappedPatchValues: Codable {
  var params: [String: Float]
}

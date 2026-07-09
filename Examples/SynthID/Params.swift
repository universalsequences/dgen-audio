import DGenLazy
import Foundation

enum Reparameterization: String, Codable {
  case raw
  case log
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
    case .logNegative:
      let lo = Foundation.log(-min)
      let hi = Foundation.log(-max)
      return -Foundation.exp((lo + hi) * 0.5)
    }
  }
}

enum KickParamSpecs {
  static let all: [ParameterSpec] = [
    .init(name: "fStart", unit: "Hz", min: 80, max: 180, reparam: .log, tolerance: 0.03),
    .init(name: "fEnd", unit: "Hz", min: 35, max: 60, reparam: .log, tolerance: 0.03),
    // logNegative, not raw: in the pitch group (small LR) a raw value at scale ~20
    // can only travel ~0.4% over a run — pitchDecay froze at init and fStart
    // equilibrated at a compensating wrong value (seed-7 plateau at fStart≈124).
    .init(name: "pitchDecay", unit: "1/s", min: -80, max: -15, reparam: .logNegative, tolerance: 0.10),
    .init(name: "bodyAmp", unit: "lin", min: 0.5, max: 1.0, reparam: .raw, tolerance: 0.10),
    .init(name: "ampDecay", unit: "1/s", min: -12, max: -3, reparam: .raw, tolerance: 0.10),
    .init(name: "clickFreq", unit: "Hz", min: 600, max: 3000, reparam: .log, tolerance: 0.10),
    .init(name: "clickAmp", unit: "lin", min: 0.05, max: 0.6, reparam: .raw, tolerance: 0.20),
    .init(name: "clickDecay", unit: "1/s", min: -900, max: -200, reparam: .logNegative, tolerance: 0.20),
    .init(name: "noiseCutoff", unit: "Hz", min: 1000, max: 8000, reparam: .log, tolerance: 0.10),
    .init(name: "noiseAmp", unit: "lin", min: 0.0, max: 0.3, reparam: .raw, tolerance: 0.20),
    .init(name: "noiseDecay", unit: "1/s", min: -400, max: -60, reparam: .logNegative, tolerance: 0.20),
    .init(name: "drive", unit: "lin", min: 1.0, max: 3.0, reparam: .raw, tolerance: 0.10),
    .init(name: "outGain", unit: "lin", min: 0.4, max: 1.0, reparam: .raw, tolerance: 0.10),
  ]

  static let byName: [String: ParameterSpec] =
    Dictionary(uniqueKeysWithValues: all.map { ($0.name, $0) })
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

  static var midpoint: PatchValues {
    var values = [String: Float]()
    for spec in KickParamSpecs.all {
      values[spec.name] = spec.midpoint
    }
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
    outGain: Float
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
      outGain: dictionary["outGain"] ?? 0.7)
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
      default: return .nan
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
      default: break
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
    copy.fStart = pitch.fStart
    copy.fEnd = pitch.fEnd
    copy.pitchDecay = pitch.pitchDecay
    return copy.clamped()
  }

  static func sample(seed: UInt64) -> PatchValues {
    var rng = SplitMix64(seed: seed)
    var values = [String: Float]()
    for spec in KickParamSpecs.all {
      values[spec.name] = rng.uniform(spec.min, spec.max)
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
}

final class TrainableKickParams {
  private var storage: [String: Signal] = [:]
  private var trainableNames = Set<String>()
  private var frozenNaturalValues: PatchValues

  init(initial: PatchValues, trainable: Bool, freezePitch: Bool = false) {
    frozenNaturalValues = initial.clamped()
    for spec in KickParamSpecs.all {
      let shouldTrain =
        trainable && !(freezePitch && ["fStart", "fEnd", "pitchDecay"].contains(spec.name))
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
      outGain: naturalSignal("outGain"))
  }

  var transformedParams: [String: Signal] { storage }

  func trainableStorage(names: [String]) -> [Signal] {
    names.compactMap { trainableNames.contains($0) ? storage[$0] : nil }
  }

  func naturalValues() -> PatchValues {
    var values = frozenNaturalValues.dictionary
    for spec in KickParamSpecs.all {
      guard let raw = storage[spec.name]?.data else { continue }
      values[spec.name] = spec.inverse(raw)
    }
    return PatchValues(values).clamped()
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

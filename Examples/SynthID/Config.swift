import DGen
import DGenLazy
import Foundation

enum SynthIDError: Error, CustomStringConvertible {
  case message(String)

  var description: String {
    switch self {
    case .message(let text): return text
    }
  }
}

struct SynthIDConfig: Codable {
  var sampleRate: Float = 44_100.0
  var frames: Int = 32_768
  var peakNormalizeTo: Float = 0.9

  var spectralWindows: [Int] = [256, 512, 1024, 2048]
  var useHannWindow: Bool = true
  var includeLinearMagnitude: Bool = true
  var linearMagnitudeWeight: Float = 0.1
  // Spectral floor for log-magnitude terms: bins below this magnitude contribute ~0.
  // 1e-3 ~= -60 dBFS for peak-normalized signals. The library default (1e-8) makes
  // the loss compare float noise in empty bins, creating a loss floor ~20% of init
  // even at exact parameter match (measured on rung 1 seed 7). 1e-4 (-80 dB) still
  // log-amplifies inaudible sub-1e-2 bins enough that parts-per-million parameter
  // errors carried 0.15 of loss; -60 dB measures audible structure.
  var spectralLogEpsilon: Float = 1e-3
  var windowWeights: [Float] = []

  // Defaults are the configuration that passed the rung-1 acceptance gate
  // (4/5 seeds, 2026-07-07).
  var epochs: Int = 600
  var pitchRefineEpochs: Int = 300
  var restarts: Int = 5
  var logEvery: Int = 10
  var checkpointEvery: Int = 50
  var gradClip: Float = 1.0
  var fdEpsilon: Float = 1e-2
  var fdcheckLogMagnitudeL2: Bool? = nil
  var fdcheckTimeMSE: Bool? = nil
  var fdcheckDirectional: Bool? = nil
  var directionEpsilon: Float = 1e-4
  var useSmoothTrainingLoss: Bool = false
  var useSmoothBasinSearch: Bool = false

  // Spec §5 values. All pitch params are log-reparameterized, so pitchLR is a
  // relative step size; 1e-3 x 400 epochs allows up to ~40% travel when needed.
  var pitchLR: Float = 1e-3
  var ampLR: Float = 3e-2
  var decayLR: Float = 3e-2
  var toneLR: Float = 1e-2
  var noiseLR: Float = 1e-2

  var cosineLRDecay: Bool = true
  var freezePitch: Bool = false
  var frozenParams: [String] = []
  var enableNoiseFilter: Bool = true
  var seed: UInt64 = 1
  var rung: Int = 1
  var profile: String = "808"

  static var `default`: SynthIDConfig { SynthIDConfig() }

  mutating func applyCLI(_ options: [String: String]) throws {
    if let value = options["frames"] { frames = try parseInt(value, "--frames") }
    if let value = options["sample-rate"] { sampleRate = try parseFloat(value, "--sample-rate") }
    if let value = options["epochs"] { epochs = try parseInt(value, "--epochs") }
    if let value = options["pitch-refine-epochs"] {
      pitchRefineEpochs = try parseInt(value, "--pitch-refine-epochs")
    }
    if let value = options["restarts"] { restarts = try parseInt(value, "--restarts") }
    if let value = options["log-every"] { logEvery = try parseInt(value, "--log-every") }
    if let value = options["checkpoint-every"] {
      checkpointEvery = try parseInt(value, "--checkpoint-every")
    }
    if let value = options["seed"] { seed = UInt64(try parseInt(value, "--seed")) }
    if let value = options["rung"] { rung = try parseInt(value, "--rung") }
    if let value = options["grad-clip"] { gradClip = try parseFloat(value, "--grad-clip") }
    if let value = options["fd-eps"] { fdEpsilon = try parseFloat(value, "--fd-eps") }
    if let value = options["direction-eps"] {
      directionEpsilon = try parseFloat(value, "--direction-eps")
    }
    if let value = options["linear-mag-weight"] {
      linearMagnitudeWeight = try parseFloat(value, "--linear-mag-weight")
    }
    if let value = options["log-eps"] {
      spectralLogEpsilon = try parseFloat(value, "--log-eps")
    }
    if let value = options["pitch-lr"] { pitchLR = try parseFloat(value, "--pitch-lr") }
    if let value = options["amp-lr"] { ampLR = try parseFloat(value, "--amp-lr") }
    if let value = options["decay-lr"] { decayLR = try parseFloat(value, "--decay-lr") }
    if let value = options["tone-lr"] { toneLR = try parseFloat(value, "--tone-lr") }
    if let value = options["noise-lr"] { noiseLR = try parseFloat(value, "--noise-lr") }
    if let value = options["windows"] {
      spectralWindows = try parseIntList(value, "--windows")
    }
    if let value = options["window-weights"] {
      windowWeights = try parseFloatList(value, "--window-weights")
    }
    if options.keys.contains("freeze-pitch") { freezePitch = true }
    if let value = options["freeze-params"] {
      frozenParams = value.split(separator: ",").map(String.init).filter { !$0.isEmpty }
    }
    if options.keys.contains("no-lr-decay") { cosineLRDecay = false }
    if options.keys.contains("no-linear-mag") { includeLinearMagnitude = false }
    if options.keys.contains("no-noise-filter") { enableNoiseFilter = false }
    if options.keys.contains("fdcheck-log-l2") { fdcheckLogMagnitudeL2 = true }
    if options.keys.contains("fdcheck-time-mse") { fdcheckTimeMSE = true }
    if options.keys.contains("fdcheck-directional") { fdcheckDirectional = true }
    if options.keys.contains("smooth-training-loss") { useSmoothTrainingLoss = true }
    if options.keys.contains("smooth-basin-search") { useSmoothBasinSearch = true }
    if let value = options["backend"] {
      switch value {
      case "metal": DGenConfig.backend = .metal
      case "cpu", "c": DGenConfig.backend = .c
      default: throw SynthIDError.message("unknown --backend \(value); expected metal or cpu")
      }
    }
    if let value = options["profile"] {
      guard ["808", "909", "hoodie-bass", "subtractive-bass"].contains(value) else {
        throw SynthIDError.message(
          "unknown --profile \(value); expected 808, 909, hoodie-bass, or subtractive-bass")
      }
      profile = value
      if value == "hoodie-bass" {
        // The identified voice is band-limited to 32 partials. An 8 kHz
        // analysis rate preserves that basis through the tagged set's highest
        // measured 116.75 Hz note while giving a 2048-point STFT 3.9 Hz
        // resolution. 4096+ windows at 44.1 kHz exceed the Metal FFT threadgroup
        // memory limit and provide no additional audible information here.
        if options["sample-rate"] == nil { sampleRate = 8_000 }
        if options["frames"] == nil { frames = 16_384 }
        if options["windows"] == nil { spectralWindows = [256, 512, 1024, 2048] }
        // The steady-note CPU estimate is more reliable than the current
        // spectral-loss f0 adjoint (documented by the profile's fdcheck).
        // Freeze only f0; all timbre and envelope parameters remain trained.
        freezePitch = true
        if options["pitch-refine-epochs"] == nil { pitchRefineEpochs = 0 }
      }
      if value == "subtractive-bass" {
        // f0 is a fixed CPU estimate in this topology, so the kick pitch
        // refinement phase would perform hundreds of empty optimizer steps.
        freezePitch = true
        if options["pitch-refine-epochs"] == nil { pitchRefineEpochs = 0 }
      }
    }
  }

  func applyRuntime() {
    DGenConfig.sampleRate = sampleRate
    DGenConfig.defaultFrameCount = frames
    DGenConfig.maxFrameCount = max(DGenConfig.maxFrameCount, frames)
    DGenSpectralConfig.logMagnitudeEpsilon = spectralLogEpsilon
    KickParamSpecs.activeProfile = profile
  }
}

func loadConfig(url: URL?) throws -> SynthIDConfig {
  guard let url else { return .default }
  let data = try Data(contentsOf: url)
  return try JSONDecoder().decode(SynthIDConfig.self, from: data)
}

func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  let data = try encoder.encode(value)
  try data.write(to: url, options: .atomic)
}

func ensureDirectory(_ url: URL) throws {
  try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
}

func parseInt(_ raw: String, _ flag: String) throws -> Int {
  guard let value = Int(raw) else {
    throw SynthIDError.message("invalid integer for \(flag): \(raw)")
  }
  return value
}

func parseFloat(_ raw: String, _ flag: String) throws -> Float {
  guard let value = Float(raw) else {
    throw SynthIDError.message("invalid float for \(flag): \(raw)")
  }
  return value
}

func parseIntList(_ raw: String, _ flag: String) throws -> [Int] {
  try raw.split(separator: ",").map { try parseInt(String($0), flag) }
}

func parseFloatList(_ raw: String, _ flag: String) throws -> [Float] {
  try raw.split(separator: ",").map { try parseFloat(String($0), flag) }
}

func peakNormalized(_ samples: [Float], peak: Float) -> [Float] {
  let scale = peakNormalizationScale(samples, peak: peak)
  return samples.map { $0 * scale }
}

func peakNormalizationScale(_ samples: [Float], peak: Float) -> Float {
  let current = samples.map { abs($0) }.max() ?? 0
  guard current > 1e-12 else { return 1.0 }
  // Attenuate-only: never amplify a quiet target. Scaling up multiplies the
  // rung-1 compensated true outGain past its trainable bound (seed 5 hit
  // outGain=1.146 > 1.0), making the target unrepresentable by the student.
  // Scaling down is always safe: peak > peakTo requires outGain > peakTo, and
  // the compensated value stays in (peakTo, outGain] ⊂ bounds.
  return Swift.min(1.0, peak / current)
}

func fitOrPad(_ samples: [Float], frames: Int) -> [Float] {
  if samples.count == frames { return samples }
  if samples.count > frames { return Array(samples.prefix(frames)) }
  return samples + [Float](repeating: 0, count: frames - samples.count)
}

func timestampUTC() -> String {
  ISO8601DateFormatter().string(from: Date())
}

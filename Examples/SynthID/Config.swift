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
  var windowWeights: [Float] = []

  var epochs: Int = 400
  var restarts: Int = 3
  var logEvery: Int = 10
  var checkpointEvery: Int = 50
  var gradClip: Float = 1.0
  var fdEpsilon: Float = 1e-3

  var pitchLR: Float = 1e-3
  var ampLR: Float = 3e-2
  var decayLR: Float = 3e-2
  var toneLR: Float = 1e-2

  var freezePitch: Bool = false
  var enableNoiseFilter: Bool = true
  var seed: UInt64 = 1
  var rung: Int = 1

  static var `default`: SynthIDConfig { SynthIDConfig() }

  mutating func applyCLI(_ options: [String: String]) throws {
    if let value = options["frames"] { frames = try parseInt(value, "--frames") }
    if let value = options["sample-rate"] { sampleRate = try parseFloat(value, "--sample-rate") }
    if let value = options["epochs"] { epochs = try parseInt(value, "--epochs") }
    if let value = options["restarts"] { restarts = try parseInt(value, "--restarts") }
    if let value = options["log-every"] { logEvery = try parseInt(value, "--log-every") }
    if let value = options["checkpoint-every"] {
      checkpointEvery = try parseInt(value, "--checkpoint-every")
    }
    if let value = options["seed"] { seed = UInt64(try parseInt(value, "--seed")) }
    if let value = options["rung"] { rung = try parseInt(value, "--rung") }
    if let value = options["grad-clip"] { gradClip = try parseFloat(value, "--grad-clip") }
    if let value = options["fd-eps"] { fdEpsilon = try parseFloat(value, "--fd-eps") }
    if let value = options["pitch-lr"] { pitchLR = try parseFloat(value, "--pitch-lr") }
    if let value = options["amp-lr"] { ampLR = try parseFloat(value, "--amp-lr") }
    if let value = options["decay-lr"] { decayLR = try parseFloat(value, "--decay-lr") }
    if let value = options["tone-lr"] { toneLR = try parseFloat(value, "--tone-lr") }
    if let value = options["windows"] {
      spectralWindows = try parseIntList(value, "--windows")
    }
    if let value = options["window-weights"] {
      windowWeights = try parseFloatList(value, "--window-weights")
    }
    if options.keys.contains("freeze-pitch") { freezePitch = true }
    if options.keys.contains("no-linear-mag") { includeLinearMagnitude = false }
    if options.keys.contains("no-noise-filter") { enableNoiseFilter = false }
    if let value = options["backend"] {
      switch value {
      case "metal": DGenConfig.backend = .metal
      case "cpu", "c": DGenConfig.backend = .c
      default: throw SynthIDError.message("unknown --backend \(value); expected metal or cpu")
      }
    }
  }

  func applyRuntime() {
    DGenConfig.sampleRate = sampleRate
    DGenConfig.defaultFrameCount = frames
    DGenConfig.maxFrameCount = max(DGenConfig.maxFrameCount, frames)
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
  let current = samples.map { abs($0) }.max() ?? 0
  guard current > 1e-12 else { return samples }
  let scale = peak / current
  return samples.map { $0 * scale }
}

func fitOrPad(_ samples: [Float], frames: Int) -> [Float] {
  if samples.count == frames { return samples }
  if samples.count > frames { return Array(samples.prefix(frames)) }
  return samples + [Float](repeating: 0, count: frames - samples.count)
}

func timestampUTC() -> String {
  ISO8601DateFormatter().string(from: Date())
}

import Foundation

struct HarmonicE2EConfig: Codable {
  var sampleRate: Float = 24_000.0
  var chunkSize: Int = 16_384
  var chunkHop: Int = 8_192
  var frameSize: Int = 1_024
  var frameHop: Int = 256

  var minF0Hz: Float = 50.0
  var maxF0Hz: Float = 1_000.0
  var silenceRMS: Float = 0.0005
  var voicedThreshold: Float = 0.3
  var peakNormalizeTo: Float = 0.99

  var trainSplit: Float = 0.9
  var shuffleChunks: Bool = true
  var seed: UInt64 = 1337
  var maxFiles: Int?
  var maxChunksPerFile: Int?

  var modelHiddenSize: Int = 64
  var modelNumLayers: Int = 2
  var numHarmonics: Int = 32
  var noiseFilterSize: Int = 63
  var batchSize: Int = 1
  var harmonicPathScale: Float = 1.0
  var noisePathScale: Float = 1.0
  var learningRate: Float = 0.001

  var mseLossWeight: Float = 0.0
  var spectralWeight: Float = 1.0
  var spectralLogmagWeight: Float = 0.5
  var spectralWindowSizes: [Int] = [128, 256, 512]
  var spectralHopDivisor: Int = 4

  var logEvery: Int = 10
  var checkpointEvery: Int = 100

  static var `default`: HarmonicE2EConfig { HarmonicE2EConfig() }

  enum CodingKeys: String, CodingKey {
    case sampleRate
    case chunkSize
    case chunkHop
    case frameSize
    case frameHop
    case minF0Hz
    case maxF0Hz
    case silenceRMS
    case voicedThreshold
    case peakNormalizeTo
    case trainSplit
    case shuffleChunks
    case seed
    case maxFiles
    case maxChunksPerFile
    case modelHiddenSize
    case modelNumLayers
    case numHarmonics
    case noiseFilterSize
    case batchSize
    case harmonicPathScale
    case noisePathScale
    case learningRate
    case mseLossWeight
    case spectralWeight
    case spectralLogmagWeight
    case spectralWindowSizes
    case spectralHopDivisor
    case logEvery
    case checkpointEvery
  }

  init() {}

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self = .default

    sampleRate = try container.decodeIfPresent(Float.self, forKey: .sampleRate) ?? sampleRate
    chunkSize = try container.decodeIfPresent(Int.self, forKey: .chunkSize) ?? chunkSize
    chunkHop = try container.decodeIfPresent(Int.self, forKey: .chunkHop) ?? chunkHop
    frameSize = try container.decodeIfPresent(Int.self, forKey: .frameSize) ?? frameSize
    frameHop = try container.decodeIfPresent(Int.self, forKey: .frameHop) ?? frameHop
    minF0Hz = try container.decodeIfPresent(Float.self, forKey: .minF0Hz) ?? minF0Hz
    maxF0Hz = try container.decodeIfPresent(Float.self, forKey: .maxF0Hz) ?? maxF0Hz
    silenceRMS = try container.decodeIfPresent(Float.self, forKey: .silenceRMS) ?? silenceRMS
    voicedThreshold =
      try container.decodeIfPresent(Float.self, forKey: .voicedThreshold) ?? voicedThreshold
    peakNormalizeTo =
      try container.decodeIfPresent(Float.self, forKey: .peakNormalizeTo) ?? peakNormalizeTo
    trainSplit = try container.decodeIfPresent(Float.self, forKey: .trainSplit) ?? trainSplit
    shuffleChunks =
      try container.decodeIfPresent(Bool.self, forKey: .shuffleChunks) ?? shuffleChunks
    seed = try container.decodeIfPresent(UInt64.self, forKey: .seed) ?? seed
    maxFiles = try container.decodeIfPresent(Int.self, forKey: .maxFiles) ?? maxFiles
    maxChunksPerFile =
      try container.decodeIfPresent(Int.self, forKey: .maxChunksPerFile) ?? maxChunksPerFile
    modelHiddenSize =
      try container.decodeIfPresent(Int.self, forKey: .modelHiddenSize) ?? modelHiddenSize
    modelNumLayers =
      try container.decodeIfPresent(Int.self, forKey: .modelNumLayers) ?? modelNumLayers
    numHarmonics = try container.decodeIfPresent(Int.self, forKey: .numHarmonics) ?? numHarmonics
    noiseFilterSize =
      try container.decodeIfPresent(Int.self, forKey: .noiseFilterSize) ?? noiseFilterSize
    batchSize = try container.decodeIfPresent(Int.self, forKey: .batchSize) ?? batchSize
    harmonicPathScale =
      try container.decodeIfPresent(Float.self, forKey: .harmonicPathScale) ?? harmonicPathScale
    noisePathScale =
      try container.decodeIfPresent(Float.self, forKey: .noisePathScale) ?? noisePathScale
    learningRate = try container.decodeIfPresent(Float.self, forKey: .learningRate) ?? learningRate
    mseLossWeight =
      try container.decodeIfPresent(Float.self, forKey: .mseLossWeight) ?? mseLossWeight
    spectralWeight =
      try container.decodeIfPresent(Float.self, forKey: .spectralWeight) ?? spectralWeight
    spectralLogmagWeight =
      try container.decodeIfPresent(Float.self, forKey: .spectralLogmagWeight)
      ?? spectralLogmagWeight
    spectralWindowSizes =
      try container.decodeIfPresent([Int].self, forKey: .spectralWindowSizes)
      ?? spectralWindowSizes
    spectralHopDivisor =
      try container.decodeIfPresent(Int.self, forKey: .spectralHopDivisor) ?? spectralHopDivisor
    logEvery = try container.decodeIfPresent(Int.self, forKey: .logEvery) ?? logEvery
    checkpointEvery =
      try container.decodeIfPresent(Int.self, forKey: .checkpointEvery) ?? checkpointEvery
  }

  static func load(path: String?) throws -> HarmonicE2EConfig {
    guard let path else { return .default }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(HarmonicE2EConfig.self, from: data)
  }

  func write(to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(self)
    try data.write(to: url)
  }

  mutating func applyCLIOverrides(_ options: [String: String]) throws {
    if let value = options["sample-rate"] { sampleRate = try parseFloat(value, key: "sample-rate") }
    if let value = options["chunk-size"] { chunkSize = try parseInt(value, key: "chunk-size") }
    if let value = options["chunk-hop"] { chunkHop = try parseInt(value, key: "chunk-hop") }
    if let value = options["frame-size"] { frameSize = try parseInt(value, key: "frame-size") }
    if let value = options["frame-hop"] { frameHop = try parseInt(value, key: "frame-hop") }
    if let value = options["min-f0"] { minF0Hz = try parseFloat(value, key: "min-f0") }
    if let value = options["max-f0"] { maxF0Hz = try parseFloat(value, key: "max-f0") }
    if let value = options["silence-rms"] { silenceRMS = try parseFloat(value, key: "silence-rms") }
    if let value = options["voiced-threshold"] { voicedThreshold = try parseFloat(value, key: "voiced-threshold") }
    if let value = options["peak-normalize-to"] { peakNormalizeTo = try parseFloat(value, key: "peak-normalize-to") }
    if let value = options["train-split"] { trainSplit = try parseFloat(value, key: "train-split") }
    if let value = options["shuffle-chunks"] { shuffleChunks = parseBool(value, key: "shuffle-chunks") }
    if let value = options["seed"] { seed = try parseUInt64(value, key: "seed") }
    if let value = options["max-files"] { maxFiles = try parseInt(value, key: "max-files") }
    if let value = options["max-chunks-per-file"] { maxChunksPerFile = try parseInt(value, key: "max-chunks-per-file") }
    if let value = options["model-hidden"] { modelHiddenSize = try parseInt(value, key: "model-hidden") }
    if let value = options["model-layers"] { modelNumLayers = try parseInt(value, key: "model-layers") }
    if let value = options["num-harmonics"] { numHarmonics = try parseInt(value, key: "num-harmonics") }
    if let value = options["noise-filter-size"] {
      noiseFilterSize = try parseInt(value, key: "noise-filter-size")
    }
    if let value = options["batch-size"] { batchSize = try parseInt(value, key: "batch-size") }
    if let value = options["harmonic-path-scale"] {
      harmonicPathScale = try parseFloat(value, key: "harmonic-path-scale")
    }
    if let value = options["noise-path-scale"] {
      noisePathScale = try parseFloat(value, key: "noise-path-scale")
    }
    if let value = options["lr"] { learningRate = try parseFloat(value, key: "lr") }
    if let value = options["mse-weight"] { mseLossWeight = try parseFloat(value, key: "mse-weight") }
    if let value = options["spectral-weight"] { spectralWeight = try parseFloat(value, key: "spectral-weight") }
    if let value = options["spectral-logmag-weight"] {
      spectralLogmagWeight = try parseFloat(value, key: "spectral-logmag-weight")
    }
    if let value = options["spectral-windows"] { spectralWindowSizes = try parseIntList(value, key: "spectral-windows") }
    if let value = options["spectral-hop-divisor"] {
      spectralHopDivisor = try parseInt(value, key: "spectral-hop-divisor")
    }
    if let value = options["log-every"] { logEvery = try parseInt(value, key: "log-every") }
    if let value = options["checkpoint-every"] { checkpointEvery = try parseInt(value, key: "checkpoint-every") }
  }

  private func parseBool(_ value: String, key: String) -> Bool {
    switch value.lowercased() {
    case "1", "true", "yes", "y", "on":
      return true
    case "0", "false", "no", "n", "off":
      return false
    default:
      return key.isEmpty ? false : false
    }
  }

  private func parseInt(_ value: String, key: String) throws -> Int {
    guard let parsed = Int(value) else {
      throw ConfigError.invalid("Invalid integer for --\(key): \(value)")
    }
    return parsed
  }

  private func parseUInt64(_ value: String, key: String) throws -> UInt64 {
    guard let parsed = UInt64(value) else {
      throw ConfigError.invalid("Invalid UInt64 for --\(key): \(value)")
    }
    return parsed
  }

  private func parseFloat(_ value: String, key: String) throws -> Float {
    guard let parsed = Float(value) else {
      throw ConfigError.invalid("Invalid float for --\(key): \(value)")
    }
    return parsed
  }

  private func parseIntList(_ value: String, key: String) throws -> [Int] {
    let parts = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    guard !parts.isEmpty else {
      throw ConfigError.invalid("Invalid int list for --\(key): \(value)")
    }
    var out: [Int] = []
    for part in parts {
      guard let parsed = Int(part) else {
        throw ConfigError.invalid("Invalid integer '\(part)' in --\(key): \(value)")
      }
      out.append(parsed)
    }
    return out
  }
}

enum ConfigError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message):
      return "Config error: \(message)"
    }
  }
}

struct SeededGenerator: RandomNumberGenerator {
  private var state: UInt64

  init(seed: UInt64) {
    self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
  }

  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}

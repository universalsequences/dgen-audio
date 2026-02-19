import Foundation

enum GradientClipMode: String, Codable {
  case element
  case global
}

enum LRSchedule: String, Codable {
  case none
  case cosine
  case exp
}

struct DDSPE2EConfig: Codable {
  var sampleRate: Float = 16_000.0
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

  // M2 decoder-only model/training parameters
  var modelHiddenSize: Int = 32
  var modelNumLayers: Int = 1
  var numHarmonics: Int = 16
  var enableNoiseFilter: Bool = false
  var noiseFilterSize: Int = 15
  var learningRate: Float = 0.001
  var lrSchedule: LRSchedule = .cosine
  var lrMin: Float = 1e-5
  var lrHalfLife: Int = 50
  var lrWarmupSteps: Int = 0
  var batchSize: Int = 1
  var gradAccumSteps: Int = 1
  var gradClip: Float = 1.0
  var gradClipMode: GradientClipMode = .element
  var normalizeGradByFrames: Bool = true
  var spectralWindowSizes: [Int] = []
  var spectralWeight: Float = 0.0
  var spectralHopDivisor: Int = 4
  var spectralWarmupSteps: Int = 100
  var spectralRampSteps: Int = 200
  var mseLossWeight: Float = 1.0
  var logEvery: Int = 10
  var checkpointEvery: Int = 100

  static var `default`: DDSPE2EConfig { DDSPE2EConfig() }

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
    case enableNoiseFilter
    case noiseFilterSize
    case learningRate
    case lrSchedule
    case lrMin
    case lrHalfLife
    case lrWarmupSteps
    case batchSize
    case gradAccumSteps
    case gradClip
    case gradClipMode
    case normalizeGradByFrames
    case spectralWindowSizes
    case spectralWeight
    case spectralHopDivisor
    case spectralWarmupSteps
    case spectralRampSteps
    case mseLossWeight
    case logEvery
    case checkpointEvery
  }

  init() {}

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let d = DDSPE2EConfig.default

    sampleRate = try c.decodeIfPresent(Float.self, forKey: .sampleRate) ?? d.sampleRate
    chunkSize = try c.decodeIfPresent(Int.self, forKey: .chunkSize) ?? d.chunkSize
    chunkHop = try c.decodeIfPresent(Int.self, forKey: .chunkHop) ?? d.chunkHop
    frameSize = try c.decodeIfPresent(Int.self, forKey: .frameSize) ?? d.frameSize
    frameHop = try c.decodeIfPresent(Int.self, forKey: .frameHop) ?? d.frameHop
    minF0Hz = try c.decodeIfPresent(Float.self, forKey: .minF0Hz) ?? d.minF0Hz
    maxF0Hz = try c.decodeIfPresent(Float.self, forKey: .maxF0Hz) ?? d.maxF0Hz
    silenceRMS = try c.decodeIfPresent(Float.self, forKey: .silenceRMS) ?? d.silenceRMS
    voicedThreshold = try c.decodeIfPresent(Float.self, forKey: .voicedThreshold) ?? d.voicedThreshold
    peakNormalizeTo = try c.decodeIfPresent(Float.self, forKey: .peakNormalizeTo) ?? d.peakNormalizeTo
    trainSplit = try c.decodeIfPresent(Float.self, forKey: .trainSplit) ?? d.trainSplit
    shuffleChunks = try c.decodeIfPresent(Bool.self, forKey: .shuffleChunks) ?? d.shuffleChunks
    seed = try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? d.seed
    maxFiles = try c.decodeIfPresent(Int.self, forKey: .maxFiles)
    maxChunksPerFile = try c.decodeIfPresent(Int.self, forKey: .maxChunksPerFile)
    modelHiddenSize = try c.decodeIfPresent(Int.self, forKey: .modelHiddenSize) ?? d.modelHiddenSize
    modelNumLayers = try c.decodeIfPresent(Int.self, forKey: .modelNumLayers) ?? d.modelNumLayers
    numHarmonics = try c.decodeIfPresent(Int.self, forKey: .numHarmonics) ?? d.numHarmonics
    enableNoiseFilter =
      try c.decodeIfPresent(Bool.self, forKey: .enableNoiseFilter) ?? d.enableNoiseFilter
    noiseFilterSize =
      try c.decodeIfPresent(Int.self, forKey: .noiseFilterSize) ?? d.noiseFilterSize
    learningRate = try c.decodeIfPresent(Float.self, forKey: .learningRate) ?? d.learningRate
    lrSchedule = try c.decodeIfPresent(LRSchedule.self, forKey: .lrSchedule) ?? d.lrSchedule
    lrMin = try c.decodeIfPresent(Float.self, forKey: .lrMin) ?? d.lrMin
    lrHalfLife = try c.decodeIfPresent(Int.self, forKey: .lrHalfLife) ?? d.lrHalfLife
    lrWarmupSteps = try c.decodeIfPresent(Int.self, forKey: .lrWarmupSteps) ?? d.lrWarmupSteps
    batchSize = try c.decodeIfPresent(Int.self, forKey: .batchSize) ?? d.batchSize
    gradAccumSteps = try c.decodeIfPresent(Int.self, forKey: .gradAccumSteps) ?? d.gradAccumSteps
    gradClip = try c.decodeIfPresent(Float.self, forKey: .gradClip) ?? d.gradClip
    gradClipMode = try c.decodeIfPresent(GradientClipMode.self, forKey: .gradClipMode) ?? d.gradClipMode
    normalizeGradByFrames =
      try c.decodeIfPresent(Bool.self, forKey: .normalizeGradByFrames) ?? d.normalizeGradByFrames
    spectralWindowSizes = try c.decodeIfPresent([Int].self, forKey: .spectralWindowSizes) ?? d.spectralWindowSizes
    spectralWeight = try c.decodeIfPresent(Float.self, forKey: .spectralWeight) ?? d.spectralWeight
    spectralHopDivisor = try c.decodeIfPresent(Int.self, forKey: .spectralHopDivisor) ?? d.spectralHopDivisor
    spectralWarmupSteps = try c.decodeIfPresent(Int.self, forKey: .spectralWarmupSteps) ?? d.spectralWarmupSteps
    spectralRampSteps = try c.decodeIfPresent(Int.self, forKey: .spectralRampSteps) ?? d.spectralRampSteps
    mseLossWeight = try c.decodeIfPresent(Float.self, forKey: .mseLossWeight) ?? d.mseLossWeight
    logEvery = try c.decodeIfPresent(Int.self, forKey: .logEvery) ?? d.logEvery
    checkpointEvery = try c.decodeIfPresent(Int.self, forKey: .checkpointEvery) ?? d.checkpointEvery
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
    if let value = options["voiced-threshold"] {
      voicedThreshold = try parseFloat(value, key: "voiced-threshold")
    }
    if let value = options["normalize-to"] {
      peakNormalizeTo = try parseFloat(value, key: "normalize-to")
    }
    if let value = options["train-split"] { trainSplit = try parseFloat(value, key: "train-split") }
    if let value = options["seed"] { seed = try parseUInt64(value, key: "seed") }
    if let value = options["max-files"] { maxFiles = try parseInt(value, key: "max-files") }
    if let value = options["max-chunks-per-file"] {
      maxChunksPerFile = try parseInt(value, key: "max-chunks-per-file")
    }
    if let value = options["model-hidden"] {
      modelHiddenSize = try parseInt(value, key: "model-hidden")
    }
    if let value = options["model-layers"] {
      modelNumLayers = try parseInt(value, key: "model-layers")
    }
    if let value = options["harmonics"] {
      numHarmonics = try parseInt(value, key: "harmonics")
    }
    if let value = options["noise-filter"] {
      enableNoiseFilter = parseBool(value)
    }
    if let value = options["noise-filter-size"] {
      noiseFilterSize = try parseInt(value, key: "noise-filter-size")
    }
    if let value = options["lr"] {
      learningRate = try parseFloat(value, key: "lr")
    }
    if let value = options["lr-schedule"] {
      guard let schedule = LRSchedule(rawValue: value.lowercased()) else {
        throw ConfigError.invalid("Invalid LR schedule for --lr-schedule: \(value) (expected none|cosine)")
      }
      lrSchedule = schedule
    }
    if let value = options["lr-min"] {
      lrMin = try parseFloat(value, key: "lr-min")
    }
    if let value = options["lr-half-life"] {
      lrHalfLife = try parseInt(value, key: "lr-half-life")
    }
    if let value = options["lr-warmup-steps"] {
      lrWarmupSteps = try parseInt(value, key: "lr-warmup-steps")
    }
    if let value = options["batch-size"] {
      batchSize = try parseInt(value, key: "batch-size")
    }
    if let value = options["grad-accum-steps"] {
      gradAccumSteps = try parseInt(value, key: "grad-accum-steps")
    }
    if let value = options["grad-clip"] {
      gradClip = try parseFloat(value, key: "grad-clip")
    }
    if let value = options["clip-mode"] {
      guard let mode = GradientClipMode(rawValue: value.lowercased()) else {
        throw ConfigError.invalid("Invalid clip mode for --clip-mode: \(value) (expected element|global)")
      }
      gradClipMode = mode
    }
    if let value = options["normalize-grad-by-frames"] {
      normalizeGradByFrames = parseBool(value)
    }
    if let value = options["spectral-windows"] {
      spectralWindowSizes = try parseIntList(value, key: "spectral-windows")
    }
    if let value = options["spectral-weight"] {
      spectralWeight = try parseFloat(value, key: "spectral-weight")
    }
    if let value = options["spectral-hop-divisor"] {
      spectralHopDivisor = try parseInt(value, key: "spectral-hop-divisor")
    }
    if let value = options["spectral-warmup-steps"] {
      spectralWarmupSteps = try parseInt(value, key: "spectral-warmup-steps")
    }
    if let value = options["spectral-ramp-steps"] {
      spectralRampSteps = try parseInt(value, key: "spectral-ramp-steps")
    }
    if let value = options["mse-weight"] {
      mseLossWeight = try parseFloat(value, key: "mse-weight")
    }
    if let value = options["log-every"] {
      logEvery = try parseInt(value, key: "log-every")
    }
    if let value = options["checkpoint-every"] {
      checkpointEvery = try parseInt(value, key: "checkpoint-every")
    }
    if let value = options["shuffle"] {
      shuffleChunks = parseBool(value)
    }

    try validate()
  }

  func write(to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(self)
    try data.write(to: url)
  }

  static func load(path: String?) throws -> DDSPE2EConfig {
    guard let path else {
      var cfg = DDSPE2EConfig.default
      try cfg.validate()
      return cfg
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let decoder = JSONDecoder()
    var cfg = try decoder.decode(DDSPE2EConfig.self, from: data)
    try cfg.validate()
    return cfg
  }

  mutating func validate() throws {
    guard sampleRate > 0 else { throw ConfigError.invalid("sampleRate must be > 0") }
    guard chunkSize > 0 else { throw ConfigError.invalid("chunkSize must be > 0") }
    guard chunkHop > 0 else { throw ConfigError.invalid("chunkHop must be > 0") }
    guard frameSize > 0 else { throw ConfigError.invalid("frameSize must be > 0") }
    guard frameHop > 0 else { throw ConfigError.invalid("frameHop must be > 0") }
    guard frameSize <= chunkSize else {
      throw ConfigError.invalid("frameSize must be <= chunkSize")
    }
    guard minF0Hz > 0, maxF0Hz > minF0Hz else {
      throw ConfigError.invalid("Require 0 < minF0Hz < maxF0Hz")
    }
    guard trainSplit >= 0, trainSplit <= 1 else {
      throw ConfigError.invalid("trainSplit must be in [0, 1]")
    }
    guard silenceRMS >= 0 else {
      throw ConfigError.invalid("silenceRMS must be >= 0")
    }
    guard voicedThreshold >= 0, voicedThreshold <= 1 else {
      throw ConfigError.invalid("voicedThreshold must be in [0, 1]")
    }
    guard peakNormalizeTo > 0 else {
      throw ConfigError.invalid("peakNormalizeTo must be > 0")
    }
    guard modelHiddenSize > 0 else {
      throw ConfigError.invalid("modelHiddenSize must be > 0")
    }
    guard modelNumLayers >= 1 else {
      throw ConfigError.invalid("modelNumLayers must be >= 1")
    }
    guard numHarmonics > 0 else {
      throw ConfigError.invalid("numHarmonics must be > 0")
    }
    guard noiseFilterSize > 1 else {
      throw ConfigError.invalid("noiseFilterSize must be > 1")
    }
    guard learningRate > 0 else {
      throw ConfigError.invalid("learningRate must be > 0")
    }
    guard lrMin >= 0 else {
      throw ConfigError.invalid("lrMin must be >= 0")
    }
    guard lrHalfLife > 0 else {
      throw ConfigError.invalid("lrHalfLife must be > 0")
    }
    guard lrWarmupSteps >= 0 else {
      throw ConfigError.invalid("lrWarmupSteps must be >= 0")
    }
    guard batchSize >= 1 else {
      throw ConfigError.invalid("batchSize must be >= 1")
    }
    guard gradAccumSteps >= 1 else {
      throw ConfigError.invalid("gradAccumSteps must be >= 1")
    }
    guard gradClip > 0 else {
      throw ConfigError.invalid("gradClip must be > 0")
    }
    for w in spectralWindowSizes {
      guard w > 1 else {
        throw ConfigError.invalid("spectral window size must be > 1")
      }
    }
    guard spectralWeight >= 0 else {
      throw ConfigError.invalid("spectralWeight must be >= 0")
    }
    guard spectralHopDivisor > 0 else {
      throw ConfigError.invalid("spectralHopDivisor must be > 0")
    }
    guard spectralWarmupSteps >= 0 else {
      throw ConfigError.invalid("spectralWarmupSteps must be >= 0")
    }
    guard spectralRampSteps >= 0 else {
      throw ConfigError.invalid("spectralRampSteps must be >= 0")
    }
    guard mseLossWeight >= 0 else {
      throw ConfigError.invalid("mseLossWeight must be >= 0")
    }
    guard logEvery > 0 else {
      throw ConfigError.invalid("logEvery must be > 0")
    }
    guard checkpointEvery > 0 else {
      throw ConfigError.invalid("checkpointEvery must be > 0")
    }
  }

  private func parseBool(_ value: String) -> Bool {
    switch value.lowercased() {
    case "1", "true", "yes", "y", "on":
      return true
    default:
      return false
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
    let parts = value
      .split(separator: ",")
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
    if parts.isEmpty {
      throw ConfigError.invalid("Invalid int list for --\(key): \(value)")
    }
    var out = [Int]()
    out.reserveCapacity(parts.count)
    for p in parts {
      guard let parsed = Int(p) else {
        throw ConfigError.invalid("Invalid integer '\(p)' in --\(key): \(value)")
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

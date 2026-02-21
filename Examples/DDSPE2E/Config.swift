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

enum HarmonicHeadMode: String, Codable {
  case legacy
  case normalized
  case softmaxDB = "softmax-db"
  case expSigmoid = "exp-sigmoid"
}

enum ControlSmoothingMode: String, Codable {
  case fir
  case off
}

enum SpectralLossModeOption: String, Codable {
  case l2
  case l1
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
  var fixedBatch: Bool = false

  var seed: UInt64 = 1337
  var maxFiles: Int?
  var maxChunksPerFile: Int?

  // M2 decoder-only model/training parameters
  var modelHiddenSize: Int = 32
  var modelNumLayers: Int = 1
  var numHarmonics: Int = 16
  var harmonicHeadMode: HarmonicHeadMode = .legacy
  var controlSmoothingMode: ControlSmoothingMode = .fir
  // Backward-compatible alias. This is kept so older configs/CLI still decode.
  var normalizedHarmonicHead: Bool = false
  var softmaxTemperature: Float = 1.0
  var softmaxTemperatureEnd: Float?
  var softmaxTemperatureWarmupSteps: Int = 0
  var softmaxTemperatureRampSteps: Int = 0
  var softmaxAmpFloor: Float = 0.0
  var softmaxGainMinDB: Float = -50.0
  var softmaxGainMaxDB: Float = 6.0
  var harmonicEntropyWeight: Float = 0.0
  var harmonicEntropyWeightEnd: Float?
  var harmonicEntropyWarmupSteps: Int = 0
  var harmonicEntropyRampSteps: Int = 0
  var harmonicConcentrationWeight: Float = 0.0
  var harmonicConcentrationWeightEnd: Float?
  var harmonicConcentrationWarmupSteps: Int = 0
  var harmonicConcentrationRampSteps: Int = 0
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
  var earlyStopPatience: Int = 0
  var earlyStopMinDelta: Float = 0.0
  var spectralWindowSizes: [Int] = []
  var spectralWeight: Float = 0.0
  var spectralLogmagWeight: Float = 0.0
  var spectralLossMode: SpectralLossModeOption = .l2
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
    case fixedBatch
    case seed
    case maxFiles
    case maxChunksPerFile
    case modelHiddenSize
    case modelNumLayers
    case numHarmonics
    case harmonicHeadMode
    case controlSmoothingMode
    case normalizedHarmonicHead
    case softmaxTemperature
    case softmaxTemperatureEnd
    case softmaxTemperatureWarmupSteps
    case softmaxTemperatureRampSteps
    case softmaxAmpFloor
    case softmaxGainMinDB
    case softmaxGainMaxDB
    case harmonicEntropyWeight
    case harmonicEntropyWeightEnd
    case harmonicEntropyWarmupSteps
    case harmonicEntropyRampSteps
    case harmonicConcentrationWeight
    case harmonicConcentrationWeightEnd
    case harmonicConcentrationWarmupSteps
    case harmonicConcentrationRampSteps
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
    case earlyStopPatience
    case earlyStopMinDelta
    case spectralWindowSizes
    case spectralWeight
    case spectralLogmagWeight
    case spectralLossMode
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
    fixedBatch = try c.decodeIfPresent(Bool.self, forKey: .fixedBatch) ?? d.fixedBatch
    seed = try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? d.seed
    maxFiles = try c.decodeIfPresent(Int.self, forKey: .maxFiles)
    maxChunksPerFile = try c.decodeIfPresent(Int.self, forKey: .maxChunksPerFile)
    modelHiddenSize = try c.decodeIfPresent(Int.self, forKey: .modelHiddenSize) ?? d.modelHiddenSize
    modelNumLayers = try c.decodeIfPresent(Int.self, forKey: .modelNumLayers) ?? d.modelNumLayers
    numHarmonics = try c.decodeIfPresent(Int.self, forKey: .numHarmonics) ?? d.numHarmonics
    harmonicHeadMode = try c.decodeIfPresent(HarmonicHeadMode.self, forKey: .harmonicHeadMode)
      ?? d.harmonicHeadMode
    controlSmoothingMode =
      try c.decodeIfPresent(ControlSmoothingMode.self, forKey: .controlSmoothingMode)
      ?? d.controlSmoothingMode
    normalizedHarmonicHead =
      try c.decodeIfPresent(Bool.self, forKey: .normalizedHarmonicHead) ?? d.normalizedHarmonicHead
    if !c.contains(.harmonicHeadMode) {
      harmonicHeadMode = normalizedHarmonicHead ? .normalized : .legacy
    }
    normalizedHarmonicHead = harmonicHeadMode == .normalized
    softmaxTemperature =
      try c.decodeIfPresent(Float.self, forKey: .softmaxTemperature) ?? d.softmaxTemperature
    softmaxTemperatureEnd = try c.decodeIfPresent(Float.self, forKey: .softmaxTemperatureEnd)
    softmaxTemperatureWarmupSteps =
      try c.decodeIfPresent(Int.self, forKey: .softmaxTemperatureWarmupSteps)
      ?? d.softmaxTemperatureWarmupSteps
    softmaxTemperatureRampSteps =
      try c.decodeIfPresent(Int.self, forKey: .softmaxTemperatureRampSteps)
      ?? d.softmaxTemperatureRampSteps
    softmaxAmpFloor = try c.decodeIfPresent(Float.self, forKey: .softmaxAmpFloor) ?? d.softmaxAmpFloor
    softmaxGainMinDB = try c.decodeIfPresent(Float.self, forKey: .softmaxGainMinDB) ?? d.softmaxGainMinDB
    softmaxGainMaxDB = try c.decodeIfPresent(Float.self, forKey: .softmaxGainMaxDB) ?? d.softmaxGainMaxDB
    harmonicEntropyWeight =
      try c.decodeIfPresent(Float.self, forKey: .harmonicEntropyWeight) ?? d.harmonicEntropyWeight
    harmonicEntropyWeightEnd = try c.decodeIfPresent(Float.self, forKey: .harmonicEntropyWeightEnd)
    harmonicEntropyWarmupSteps =
      try c.decodeIfPresent(Int.self, forKey: .harmonicEntropyWarmupSteps) ?? d.harmonicEntropyWarmupSteps
    harmonicEntropyRampSteps =
      try c.decodeIfPresent(Int.self, forKey: .harmonicEntropyRampSteps) ?? d.harmonicEntropyRampSteps
    harmonicConcentrationWeight =
      try c.decodeIfPresent(Float.self, forKey: .harmonicConcentrationWeight) ?? d.harmonicConcentrationWeight
    harmonicConcentrationWeightEnd =
      try c.decodeIfPresent(Float.self, forKey: .harmonicConcentrationWeightEnd)
    harmonicConcentrationWarmupSteps =
      try c.decodeIfPresent(Int.self, forKey: .harmonicConcentrationWarmupSteps)
      ?? d.harmonicConcentrationWarmupSteps
    harmonicConcentrationRampSteps =
      try c.decodeIfPresent(Int.self, forKey: .harmonicConcentrationRampSteps)
      ?? d.harmonicConcentrationRampSteps
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
    earlyStopPatience = try c.decodeIfPresent(Int.self, forKey: .earlyStopPatience) ?? d.earlyStopPatience
    earlyStopMinDelta = try c.decodeIfPresent(Float.self, forKey: .earlyStopMinDelta) ?? d.earlyStopMinDelta
    spectralWindowSizes = try c.decodeIfPresent([Int].self, forKey: .spectralWindowSizes) ?? d.spectralWindowSizes
    spectralWeight = try c.decodeIfPresent(Float.self, forKey: .spectralWeight) ?? d.spectralWeight
    spectralLogmagWeight =
      try c.decodeIfPresent(Float.self, forKey: .spectralLogmagWeight) ?? d.spectralLogmagWeight
    spectralLossMode = try c.decodeIfPresent(SpectralLossModeOption.self, forKey: .spectralLossMode)
      ?? d.spectralLossMode
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
    if let value = options["normalized-harmonic-head"] {
      normalizedHarmonicHead = parseBool(value)
      harmonicHeadMode = normalizedHarmonicHead ? .normalized : .legacy
    }
    if let value = options["harmonic-head-mode"] {
      guard let mode = HarmonicHeadMode(rawValue: value.lowercased()) else {
        throw ConfigError.invalid(
          "Invalid harmonic head mode for --harmonic-head-mode: \(value) (expected legacy|normalized|softmax-db|exp-sigmoid)"
        )
      }
      harmonicHeadMode = mode
    }
    if let value = options["control-smoothing"] {
      guard let mode = ControlSmoothingMode(rawValue: value.lowercased()) else {
        throw ConfigError.invalid(
          "Invalid control smoothing mode for --control-smoothing: \(value) (expected fir|off)"
        )
      }
      controlSmoothingMode = mode
    }
    if let value = options["softmax-temp"] {
      softmaxTemperature = try parseFloat(value, key: "softmax-temp")
    }
    if let value = options["softmax-temp-end"] {
      softmaxTemperatureEnd = try parseFloat(value, key: "softmax-temp-end")
    }
    if let value = options["softmax-temp-warmup-steps"] {
      softmaxTemperatureWarmupSteps = try parseInt(value, key: "softmax-temp-warmup-steps")
    }
    if let value = options["softmax-temp-ramp-steps"] {
      softmaxTemperatureRampSteps = try parseInt(value, key: "softmax-temp-ramp-steps")
    }
    if let value = options["softmax-amp-floor"] {
      softmaxAmpFloor = try parseFloat(value, key: "softmax-amp-floor")
    }
    if let value = options["softmax-gain-min-db"] {
      softmaxGainMinDB = try parseFloat(value, key: "softmax-gain-min-db")
    }
    if let value = options["softmax-gain-max-db"] {
      softmaxGainMaxDB = try parseFloat(value, key: "softmax-gain-max-db")
    }
    if let value = options["harmonic-entropy-weight"] {
      harmonicEntropyWeight = try parseFloat(value, key: "harmonic-entropy-weight")
    }
    if let value = options["harmonic-entropy-weight-end"] {
      harmonicEntropyWeightEnd = try parseFloat(value, key: "harmonic-entropy-weight-end")
    }
    if let value = options["harmonic-entropy-warmup-steps"] {
      harmonicEntropyWarmupSteps = try parseInt(value, key: "harmonic-entropy-warmup-steps")
    }
    if let value = options["harmonic-entropy-ramp-steps"] {
      harmonicEntropyRampSteps = try parseInt(value, key: "harmonic-entropy-ramp-steps")
    }
    if let value = options["harmonic-concentration-weight"] {
      harmonicConcentrationWeight = try parseFloat(value, key: "harmonic-concentration-weight")
    }
    if let value = options["harmonic-concentration-weight-end"] {
      harmonicConcentrationWeightEnd = try parseFloat(value, key: "harmonic-concentration-weight-end")
    }
    if let value = options["harmonic-concentration-warmup-steps"] {
      harmonicConcentrationWarmupSteps =
        try parseInt(value, key: "harmonic-concentration-warmup-steps")
    }
    if let value = options["harmonic-concentration-ramp-steps"] {
      harmonicConcentrationRampSteps =
        try parseInt(value, key: "harmonic-concentration-ramp-steps")
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
        throw ConfigError.invalid("Invalid LR schedule for --lr-schedule: \(value) (expected none|cosine|exp)")
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
    if let value = options["early-stop-patience"] {
      earlyStopPatience = try parseInt(value, key: "early-stop-patience")
    }
    if let value = options["early-stop-min-delta"] {
      earlyStopMinDelta = try parseFloat(value, key: "early-stop-min-delta")
    }
    if let value = options["spectral-windows"] {
      spectralWindowSizes = try parseIntList(value, key: "spectral-windows")
    }
    if let value = options["spectral-weight"] {
      spectralWeight = try parseFloat(value, key: "spectral-weight")
    }
    if let value = options["spectral-logmag-weight"] {
      spectralLogmagWeight = try parseFloat(value, key: "spectral-logmag-weight")
    }
    if let value = options["spectral-loss-mode"] {
      guard let mode = SpectralLossModeOption(rawValue: value.lowercased()) else {
        throw ConfigError.invalid(
          "Invalid spectral loss mode for --spectral-loss-mode: \(value) (expected l2|l1)"
        )
      }
      spectralLossMode = mode
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
    if let value = options["fixed-batch"] {
      fixedBatch = parseBool(value)
    }

    normalizedHarmonicHead = harmonicHeadMode == .normalized
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
    guard softmaxTemperature > 0 else {
      throw ConfigError.invalid("softmaxTemperature must be > 0")
    }
    if let softmaxTemperatureEnd {
      guard softmaxTemperatureEnd > 0 else {
        throw ConfigError.invalid("softmaxTemperatureEnd must be > 0")
      }
    }
    guard softmaxTemperatureWarmupSteps >= 0 else {
      throw ConfigError.invalid("softmaxTemperatureWarmupSteps must be >= 0")
    }
    guard softmaxTemperatureRampSteps >= 0 else {
      throw ConfigError.invalid("softmaxTemperatureRampSteps must be >= 0")
    }
    guard softmaxAmpFloor >= 0, softmaxAmpFloor <= 1 else {
      throw ConfigError.invalid("softmaxAmpFloor must be in [0, 1]")
    }
    guard softmaxGainMaxDB > softmaxGainMinDB else {
      throw ConfigError.invalid("softmaxGainMaxDB must be > softmaxGainMinDB")
    }
    guard harmonicEntropyWeight >= 0 else {
      throw ConfigError.invalid("harmonicEntropyWeight must be >= 0")
    }
    if let harmonicEntropyWeightEnd {
      guard harmonicEntropyWeightEnd >= 0 else {
        throw ConfigError.invalid("harmonicEntropyWeightEnd must be >= 0")
      }
    }
    guard harmonicEntropyWarmupSteps >= 0 else {
      throw ConfigError.invalid("harmonicEntropyWarmupSteps must be >= 0")
    }
    guard harmonicEntropyRampSteps >= 0 else {
      throw ConfigError.invalid("harmonicEntropyRampSteps must be >= 0")
    }
    guard harmonicConcentrationWeight >= 0 else {
      throw ConfigError.invalid("harmonicConcentrationWeight must be >= 0")
    }
    if let harmonicConcentrationWeightEnd {
      guard harmonicConcentrationWeightEnd >= 0 else {
        throw ConfigError.invalid("harmonicConcentrationWeightEnd must be >= 0")
      }
    }
    guard harmonicConcentrationWarmupSteps >= 0 else {
      throw ConfigError.invalid("harmonicConcentrationWarmupSteps must be >= 0")
    }
    guard harmonicConcentrationRampSteps >= 0 else {
      throw ConfigError.invalid("harmonicConcentrationRampSteps must be >= 0")
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
    guard earlyStopPatience >= 0 else {
      throw ConfigError.invalid("earlyStopPatience must be >= 0")
    }
    guard earlyStopMinDelta >= 0 else {
      throw ConfigError.invalid("earlyStopMinDelta must be >= 0")
    }
    for w in spectralWindowSizes {
      guard w > 1 else {
        throw ConfigError.invalid("spectral window size must be > 1")
      }
    }
    guard spectralWeight >= 0 else {
      throw ConfigError.invalid("spectralWeight must be >= 0")
    }
    guard spectralLogmagWeight >= 0 else {
      throw ConfigError.invalid("spectralLogmagWeight must be >= 0")
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

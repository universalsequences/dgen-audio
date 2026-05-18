import DGen
import DGenLazy
import Foundation

struct StringCodingKey: CodingKey {
  var stringValue: String
  var intValue: Int?

  init(_ stringValue: String) {
    self.stringValue = stringValue
  }

  init?(stringValue: String) {
    self.stringValue = stringValue
  }

  init?(intValue: Int) {
    self.stringValue = "\(intValue)"
    self.intValue = intValue
  }
}

struct Config: Codable {
  var targetPath =
    "/Users/alecresende/code/learning/anthropic/eseq/crates/sequencer/samples/manufacturers/EMU/Kicky Fdz.wav"
  var frames = 4096
  var epochs = 40
  var windowSize = 512
  var outputDir = "/tmp/dgen-train-kick808"
  var learningRateScale: Float = 1.0
  var waveformWeight: Float = 35.0
  var transientWeight: Float = 90.0
  var slopeWeight: Float = 0.0
  var highBandWeight: Float = 0.0
  var highBandSpectralWeight: Float = 0.0
  var subBandWeight: Float = 0.0
  var presenceBandWeight: Float = 0.0
  var airBandWeight: Float = 0.0
  var envSize = 96
  var freqScalarLRScale: Float = 1.0
  var ampLRScale: Float = 1.0
  var decayLRScale: Float = 1.0
  var envLRScale: Float = 1.0
  var freqCurveLRScale: Float = 1.0
  var phaseWarpLRScale: Float = 1.0
  var bodyWaveLRScale: Float = 1.0
  var residualLRScale: Float = 1.0
  var noiseLRScale: Float = 1.0
  var sizzleLRScale: Float = 1.0
  var airSizzleLRScale: Float = 1.0
  var airNoiseLRScale: Float = 1.0
  var fmNoiseLRScale: Float = 1.0
  var crispNoiseLRScale: Float = 1.0
  var airModeLRScale: Float = 1.0
  var transientModeLRScale: Float = 1.0
  var subBodyLRScale: Float = 1.0
  var eqLRScale: Float = 1.0
  var residualFrames = 1536
  var residualPoints: Int?
  var phaseWarpFrames = 1536
  var phaseWarpPoints = 64
  var phaseWarpAmpInit: Float = 0.0
  var resetPhaseWarp = false
  var bodyWavePoints = 128
  var bodyWaveMixInit: Float = 0.0
  var noiseFilterSize = 31
  var sizzleFrames = 2048
  var sizzlePoints: Int?
  var sizzleInit = "zero"
  var sizzleAmpInit: Float = 1.0
  var sizzleLowHz: Float = 1_600.0
  var sizzleHighHz: Float = 0.0
  var sizzleDelayMs: Float = 3.0
  var sizzleFadeMs: Float = 1.0
  var sizzlePostDrive = false
  var resetSizzle = false
  var airSizzleFrames = 2048
  var airSizzlePoints: Int?
  var airSizzleInit = "zero"
  var airSizzleAmpInit: Float = 0.0
  var airSizzleLowHz: Float = 4_000.0
  var airSizzleHighHz: Float = 12_000.0
  var airSizzleDelayMs: Float = 3.0
  var airSizzleFadeMs: Float = 1.0
  var airSizzlePostDrive = true
  var resetAirSizzle = false
  var airNoiseAmpInit: Float = 0.0
  var airNoiseLowHz: Float = 4_000.0
  var airNoiseHighHz: Float = 12_000.0
  var airNoiseDelayMs: Float = 0.0
  var airNoiseFadeMs: Float = 2.0
  var fmNoiseFrames = 4096
  var fmNoisePoints: Int?
  var fmNoiseAmpInit: Float = 0.0
  var fmNoisePMInit: Float = 1.0
  var fmNoiseLowHz: Float = 2_000.0
  var fmNoiseHighHz: Float = 12_000.0
  var fmNoiseDelayMs: Float = 3.0
  var fmNoiseFadeMs: Float = 6.0
  var resetFMNoise = false
  var crispNoiseFrames = 2048
  var crispNoisePoints: Int?
  var crispNoiseInit = "noise"
  var crispNoiseAmpInit: Float = 0.0
  var crispNoiseSpacing: Float = 17.0
  var crispNoiseBurstDecay: Float = 0.88
  var crispNoiseLowHz: Float = 6_000.0
  var crispNoiseHighHz: Float = 16_000.0
  var crispNoiseHPStages = 1
  var crispNoiseDelayMs: Float = 0.5
  var crispNoiseFadeMs: Float = 1.5
  var presenceDuckInit: Float = 0.0
  var presenceDuckLowHz: Float = 1_000.0
  var presenceDuckHighHz: Float = 4_000.0
  var airModeAmpInit: Float = 0.0
  var airModeFreqInit: Float = 8_000.0
  var airModeDelayMs: Float = 5.0
  var airModeFadeMs: Float = 4.0
  var airModeHighpassHz: Float = 5_000.0
  var airModeLowpassHz: Float = 16_000.0
  var transientModeAmpInit: Float = 0.0
  var transientModeMs: Float = 30.0
  var transientModeFollowBody = false
  var transientModePostDrive = false
  var subBodyFrames = 4096
  var subBodyPoints: Int?
  var subBodyInit = "zero"
  var subBodyAmpInit: Float = 0.0
  var subBodyHz: Float = 180.0
  var subBodyPostDrive = true
  var cleanBodyPostDrive = false
  var cleanSubPostDrive = false
  var cleanBodyScale: Float = 1.0
  var eqFilterSize = 1
  var selectionEvery = 0
  var checkpointSelection = "loss"
  var perceptualCheckpointOut: String?
  var sustainedSubWeight: Float = 0.0
  var sustainedPresenceWeight: Float = 0.0
  var sustainedAirWeight: Float = 0.0
  var sustainedHFWeight: Float = 0.0
  var renderOnly = false
  var checkpointIn: String?
  var checkpointOut: String?

  init() {}

  init(from decoder: Decoder) throws {
    self.init()
    let c = try decoder.container(keyedBy: StringCodingKey.self)

    func decode<T: Decodable>(_ key: String, _ current: T) throws -> T {
      try c.decodeIfPresent(T.self, forKey: StringCodingKey(key)) ?? current
    }

    targetPath = try decode("targetPath", targetPath)
    frames = try decode("frames", frames)
    epochs = try decode("epochs", epochs)
    windowSize = try decode("windowSize", windowSize)
    outputDir = try decode("outputDir", outputDir)
    learningRateScale = try decode("learningRateScale", learningRateScale)
    waveformWeight = try decode("waveformWeight", waveformWeight)
    transientWeight = try decode("transientWeight", transientWeight)
    slopeWeight = try decode("slopeWeight", slopeWeight)
    highBandWeight = try decode("highBandWeight", highBandWeight)
    highBandSpectralWeight = try decode("highBandSpectralWeight", highBandSpectralWeight)
    subBandWeight = try decode("subBandWeight", subBandWeight)
    presenceBandWeight = try decode("presenceBandWeight", presenceBandWeight)
    airBandWeight = try decode("airBandWeight", airBandWeight)
    envSize = try decode("envSize", envSize)
    freqScalarLRScale = try decode("freqScalarLRScale", freqScalarLRScale)
    ampLRScale = try decode("ampLRScale", ampLRScale)
    decayLRScale = try decode("decayLRScale", decayLRScale)
    envLRScale = try decode("envLRScale", envLRScale)
    freqCurveLRScale = try decode("freqCurveLRScale", freqCurveLRScale)
    phaseWarpLRScale = try decode("phaseWarpLRScale", phaseWarpLRScale)
    bodyWaveLRScale = try decode("bodyWaveLRScale", bodyWaveLRScale)
    residualLRScale = try decode("residualLRScale", residualLRScale)
    noiseLRScale = try decode("noiseLRScale", noiseLRScale)
    sizzleLRScale = try decode("sizzleLRScale", sizzleLRScale)
    airSizzleLRScale = try decode("airSizzleLRScale", airSizzleLRScale)
    airNoiseLRScale = try decode("airNoiseLRScale", airNoiseLRScale)
    fmNoiseLRScale = try decode("fmNoiseLRScale", fmNoiseLRScale)
    crispNoiseLRScale = try decode("crispNoiseLRScale", crispNoiseLRScale)
    airModeLRScale = try decode("airModeLRScale", airModeLRScale)
    transientModeLRScale = try decode("transientModeLRScale", transientModeLRScale)
    subBodyLRScale = try decode("subBodyLRScale", subBodyLRScale)
    eqLRScale = try decode("eqLRScale", eqLRScale)
    residualFrames = try decode("residualFrames", residualFrames)
    residualPoints = try c.decodeIfPresent(Int.self, forKey: StringCodingKey("residualPoints")) ?? residualPoints
    phaseWarpFrames = try decode("phaseWarpFrames", phaseWarpFrames)
    phaseWarpPoints = try decode("phaseWarpPoints", phaseWarpPoints)
    phaseWarpAmpInit = try decode("phaseWarpAmpInit", phaseWarpAmpInit)
    resetPhaseWarp = try decode("resetPhaseWarp", resetPhaseWarp)
    bodyWavePoints = try decode("bodyWavePoints", bodyWavePoints)
    bodyWaveMixInit = try decode("bodyWaveMixInit", bodyWaveMixInit)
    noiseFilterSize = try decode("noiseFilterSize", noiseFilterSize)
    sizzleFrames = try decode("sizzleFrames", sizzleFrames)
    sizzlePoints = try c.decodeIfPresent(Int.self, forKey: StringCodingKey("sizzlePoints")) ?? sizzlePoints
    sizzleInit = try decode("sizzleInit", sizzleInit)
    sizzleAmpInit = try decode("sizzleAmpInit", sizzleAmpInit)
    sizzleLowHz = try decode("sizzleLowHz", sizzleLowHz)
    sizzleHighHz = try decode("sizzleHighHz", sizzleHighHz)
    sizzleDelayMs = try decode("sizzleDelayMs", sizzleDelayMs)
    sizzleFadeMs = try decode("sizzleFadeMs", sizzleFadeMs)
    sizzlePostDrive = try decode("sizzlePostDrive", sizzlePostDrive)
    resetSizzle = try decode("resetSizzle", resetSizzle)
    airSizzleFrames = try decode("airSizzleFrames", airSizzleFrames)
    airSizzlePoints = try c.decodeIfPresent(Int.self, forKey: StringCodingKey("airSizzlePoints")) ?? airSizzlePoints
    airSizzleInit = try decode("airSizzleInit", airSizzleInit)
    airSizzleAmpInit = try decode("airSizzleAmpInit", airSizzleAmpInit)
    airSizzleLowHz = try decode("airSizzleLowHz", airSizzleLowHz)
    airSizzleHighHz = try decode("airSizzleHighHz", airSizzleHighHz)
    airSizzleDelayMs = try decode("airSizzleDelayMs", airSizzleDelayMs)
    airSizzleFadeMs = try decode("airSizzleFadeMs", airSizzleFadeMs)
    airSizzlePostDrive = try decode("airSizzlePostDrive", airSizzlePostDrive)
    resetAirSizzle = try decode("resetAirSizzle", resetAirSizzle)
    airNoiseAmpInit = try decode("airNoiseAmpInit", airNoiseAmpInit)
    airNoiseLowHz = try decode("airNoiseLowHz", airNoiseLowHz)
    airNoiseHighHz = try decode("airNoiseHighHz", airNoiseHighHz)
    airNoiseDelayMs = try decode("airNoiseDelayMs", airNoiseDelayMs)
    airNoiseFadeMs = try decode("airNoiseFadeMs", airNoiseFadeMs)
    fmNoiseFrames = try decode("fmNoiseFrames", fmNoiseFrames)
    fmNoisePoints = try c.decodeIfPresent(Int.self, forKey: StringCodingKey("fmNoisePoints")) ?? fmNoisePoints
    fmNoiseAmpInit = try decode("fmNoiseAmpInit", fmNoiseAmpInit)
    fmNoisePMInit = try decode("fmNoisePMInit", fmNoisePMInit)
    fmNoiseLowHz = try decode("fmNoiseLowHz", fmNoiseLowHz)
    fmNoiseHighHz = try decode("fmNoiseHighHz", fmNoiseHighHz)
    fmNoiseDelayMs = try decode("fmNoiseDelayMs", fmNoiseDelayMs)
    fmNoiseFadeMs = try decode("fmNoiseFadeMs", fmNoiseFadeMs)
    resetFMNoise = try decode("resetFMNoise", resetFMNoise)
    crispNoiseFrames = try decode("crispNoiseFrames", crispNoiseFrames)
    crispNoisePoints = try c.decodeIfPresent(Int.self, forKey: StringCodingKey("crispNoisePoints")) ?? crispNoisePoints
    crispNoiseInit = try decode("crispNoiseInit", crispNoiseInit)
    crispNoiseAmpInit = try decode("crispNoiseAmpInit", crispNoiseAmpInit)
    crispNoiseSpacing = try decode("crispNoiseSpacing", crispNoiseSpacing)
    crispNoiseBurstDecay = try decode("crispNoiseBurstDecay", crispNoiseBurstDecay)
    crispNoiseLowHz = try decode("crispNoiseLowHz", crispNoiseLowHz)
    crispNoiseHighHz = try decode("crispNoiseHighHz", crispNoiseHighHz)
    crispNoiseHPStages = try decode("crispNoiseHPStages", crispNoiseHPStages)
    crispNoiseDelayMs = try decode("crispNoiseDelayMs", crispNoiseDelayMs)
    crispNoiseFadeMs = try decode("crispNoiseFadeMs", crispNoiseFadeMs)
    presenceDuckInit = try decode("presenceDuckInit", presenceDuckInit)
    presenceDuckLowHz = try decode("presenceDuckLowHz", presenceDuckLowHz)
    presenceDuckHighHz = try decode("presenceDuckHighHz", presenceDuckHighHz)
    airModeAmpInit = try decode("airModeAmpInit", airModeAmpInit)
    airModeFreqInit = try decode("airModeFreqInit", airModeFreqInit)
    airModeDelayMs = try decode("airModeDelayMs", airModeDelayMs)
    airModeFadeMs = try decode("airModeFadeMs", airModeFadeMs)
    airModeHighpassHz = try decode("airModeHighpassHz", airModeHighpassHz)
    airModeLowpassHz = try decode("airModeLowpassHz", airModeLowpassHz)
    transientModeAmpInit = try decode("transientModeAmpInit", transientModeAmpInit)
    transientModeMs = try decode("transientModeMs", transientModeMs)
    transientModeFollowBody = try decode("transientModeFollowBody", transientModeFollowBody)
    transientModePostDrive = try decode("transientModePostDrive", transientModePostDrive)
    subBodyFrames = try decode("subBodyFrames", subBodyFrames)
    subBodyPoints = try c.decodeIfPresent(Int.self, forKey: StringCodingKey("subBodyPoints")) ?? subBodyPoints
    subBodyInit = try decode("subBodyInit", subBodyInit)
    subBodyAmpInit = try decode("subBodyAmpInit", subBodyAmpInit)
    subBodyHz = try decode("subBodyHz", subBodyHz)
    subBodyPostDrive = try decode("subBodyPostDrive", subBodyPostDrive)
    cleanBodyPostDrive = try decode("cleanBodyPostDrive", cleanBodyPostDrive)
    cleanSubPostDrive = try decode("cleanSubPostDrive", cleanSubPostDrive)
    cleanBodyScale = try decode("cleanBodyScale", cleanBodyScale)
    eqFilterSize = try decode("eqFilterSize", eqFilterSize)
    selectionEvery = try decode("selectionEvery", selectionEvery)
    checkpointSelection = try decode("checkpointSelection", checkpointSelection)
    perceptualCheckpointOut = try c.decodeIfPresent(String.self, forKey: StringCodingKey("perceptualCheckpointOut")) ?? perceptualCheckpointOut
    sustainedSubWeight = try decode("sustainedSubWeight", sustainedSubWeight)
    sustainedPresenceWeight = try decode("sustainedPresenceWeight", sustainedPresenceWeight)
    sustainedAirWeight = try decode("sustainedAirWeight", sustainedAirWeight)
    sustainedHFWeight = try decode("sustainedHFWeight", sustainedHFWeight)
    renderOnly = try decode("renderOnly", renderOnly)
    checkpointIn = try c.decodeIfPresent(String.self, forKey: StringCodingKey("checkpointIn")) ?? checkpointIn
    checkpointOut = try c.decodeIfPresent(String.self, forKey: StringCodingKey("checkpointOut")) ?? checkpointOut
  }
}

struct TrainingCheckpoint: Codable {
  var config: Config?
  var params: [String: Float]
  var tensors: [String: [Float]]
}

func loadCheckpointConfig(path: String) throws -> Config? {
  let data = try Data(contentsOf: URL(fileURLWithPath: path))
  let checkpoint = try JSONDecoder().decode(TrainingCheckpoint.self, from: data)
  return checkpoint.config
}

func mergedCheckpointConfig(saved: Config, cli: Config) -> Config {
  var config = saved
  config.outputDir = cli.outputDir
  config.epochs = cli.epochs
  config.renderOnly = cli.renderOnly
  config.checkpointIn = cli.checkpointIn
  config.checkpointOut = cli.checkpointOut
  config.selectionEvery = cli.selectionEvery
  config.checkpointSelection = cli.checkpointSelection
  config.perceptualCheckpointOut = cli.perceptualCheckpointOut
  return config
}

func loadCheckpoint(
  path: String,
  params: [Signal],
  names: [String],
  tensors: [DGenLazy.Tensor],
  tensorNames: [String]
) throws {
  let data = try Data(contentsOf: URL(fileURLWithPath: path))
  let checkpoint = try JSONDecoder().decode(TrainingCheckpoint.self, from: data)
  var loadedParams = 0
  var loadedTensors = 0

  for (name, param) in zip(names, params) {
    if let value = checkpoint.params[name] {
      param.updateDataLazily(value)
      loadedParams += 1
    }
  }

  for (name, tensor) in zip(tensorNames, tensors) {
    guard let values = checkpoint.tensors[name] else { continue }
    let currentCount = tensor.getData()?.count ?? 0
    if values.count == currentCount {
      tensor.updateDataLazily(values)
      loadedTensors += 1
    } else {
      print("checkpoint skipped tensor=\(name) count=\(values.count) expected=\(currentCount)")
    }
  }

  print("checkpoint loaded=\(path) params=\(loadedParams) tensors=\(loadedTensors)")
}

func saveCheckpoint(
  path: String,
  config: Config,
  params: [Signal],
  names: [String],
  tensors: [DGenLazy.Tensor],
  tensorNames: [String]
) throws {
  var scalarValues: [String: Float] = [:]
  var tensorValues: [String: [Float]] = [:]

  for (name, param) in zip(names, params) {
    scalarValues[name] = param.data ?? 0.0
  }
  for (name, tensor) in zip(tensorNames, tensors) {
    tensorValues[name] = tensor.getData() ?? []
  }

  let checkpoint = TrainingCheckpoint(config: config, params: scalarValues, tensors: tensorValues)
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
  let data = try encoder.encode(checkpoint)
  let url = URL(fileURLWithPath: path)
  try FileManager.default.createDirectory(
    at: url.deletingLastPathComponent(),
    withIntermediateDirectories: true)
  try data.write(to: url)
  print("wrote=\(url.path)")
}

func parseArgs() -> Config {
  var config = Config()
  var i = 1
  while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    func next() -> String? {
      guard i + 1 < CommandLine.arguments.count else { return nil }
      i += 1
      return CommandLine.arguments[i]
    }

    switch arg {
    case "--target":
      if let value = next() { config.targetPath = value }
    case "--frames":
      if let value = next(), let n = Int(value) { config.frames = n }
    case "--epochs":
      if let value = next(), let n = Int(value) { config.epochs = n }
    case "--window":
      if let value = next(), let n = Int(value) { config.windowSize = n }
    case "--out":
      if let value = next() { config.outputDir = value }
    case "--lr-scale":
      if let value = next(), let n = Float(value) { config.learningRateScale = n }
    case "--waveform-weight":
      if let value = next(), let n = Float(value) { config.waveformWeight = n }
    case "--transient-weight":
      if let value = next(), let n = Float(value) { config.transientWeight = n }
    case "--slope-weight":
      if let value = next(), let n = Float(value) { config.slopeWeight = n }
    case "--high-band-weight":
      if let value = next(), let n = Float(value) { config.highBandWeight = n }
    case "--high-band-spectral-weight":
      if let value = next(), let n = Float(value) { config.highBandSpectralWeight = n }
    case "--sub-band-weight":
      if let value = next(), let n = Float(value) { config.subBandWeight = n }
    case "--presence-band-weight":
      if let value = next(), let n = Float(value) { config.presenceBandWeight = n }
    case "--air-band-weight":
      if let value = next(), let n = Float(value) { config.airBandWeight = n }
    case "--env-size":
      if let value = next(), let n = Int(value) { config.envSize = n }
    case "--freq-scalar-lr-scale":
      if let value = next(), let n = Float(value) { config.freqScalarLRScale = n }
    case "--amp-lr-scale":
      if let value = next(), let n = Float(value) { config.ampLRScale = n }
    case "--decay-lr-scale":
      if let value = next(), let n = Float(value) { config.decayLRScale = n }
    case "--env-lr-scale":
      if let value = next(), let n = Float(value) { config.envLRScale = n }
    case "--freq-curve-lr-scale":
      if let value = next(), let n = Float(value) { config.freqCurveLRScale = n }
    case "--phase-warp-lr-scale":
      if let value = next(), let n = Float(value) { config.phaseWarpLRScale = n }
    case "--body-wave-lr-scale":
      if let value = next(), let n = Float(value) { config.bodyWaveLRScale = n }
    case "--residual-lr-scale":
      if let value = next(), let n = Float(value) { config.residualLRScale = n }
    case "--noise-lr-scale":
      if let value = next(), let n = Float(value) { config.noiseLRScale = n }
    case "--sizzle-lr-scale":
      if let value = next(), let n = Float(value) { config.sizzleLRScale = n }
    case "--air-sizzle-lr-scale":
      if let value = next(), let n = Float(value) { config.airSizzleLRScale = n }
    case "--air-noise-lr-scale":
      if let value = next(), let n = Float(value) { config.airNoiseLRScale = n }
    case "--fm-noise-lr-scale":
      if let value = next(), let n = Float(value) { config.fmNoiseLRScale = n }
    case "--crisp-noise-lr-scale":
      if let value = next(), let n = Float(value) { config.crispNoiseLRScale = n }
    case "--air-mode-lr-scale":
      if let value = next(), let n = Float(value) { config.airModeLRScale = n }
    case "--transient-mode-lr-scale":
      if let value = next(), let n = Float(value) { config.transientModeLRScale = n }
    case "--subbody-lr-scale":
      if let value = next(), let n = Float(value) { config.subBodyLRScale = n }
    case "--eq-lr-scale":
      if let value = next(), let n = Float(value) { config.eqLRScale = n }
    case "--residual-frames":
      if let value = next(), let n = Int(value) { config.residualFrames = n }
    case "--residual-points":
      if let value = next(), let n = Int(value) { config.residualPoints = n }
    case "--phase-warp-frames":
      if let value = next(), let n = Int(value) { config.phaseWarpFrames = n }
    case "--phase-warp-points":
      if let value = next(), let n = Int(value) { config.phaseWarpPoints = n }
    case "--phase-warp-amp-init":
      if let value = next(), let n = Float(value) { config.phaseWarpAmpInit = n }
    case "--reset-phase-warp":
      config.resetPhaseWarp = true
    case "--body-wave-points":
      if let value = next(), let n = Int(value) { config.bodyWavePoints = n }
    case "--body-wave-mix-init":
      if let value = next(), let n = Float(value) { config.bodyWaveMixInit = n }
    case "--noise-filter-size":
      if let value = next(), let n = Int(value) { config.noiseFilterSize = n }
    case "--sizzle-frames":
      if let value = next(), let n = Int(value) { config.sizzleFrames = n }
    case "--sizzle-points":
      if let value = next(), let n = Int(value) { config.sizzlePoints = n }
    case "--sizzle-init":
      if let value = next() { config.sizzleInit = value }
    case "--sizzle-amp-init":
      if let value = next(), let n = Float(value) { config.sizzleAmpInit = n }
    case "--sizzle-low-hz":
      if let value = next(), let n = Float(value) { config.sizzleLowHz = n }
    case "--sizzle-high-hz":
      if let value = next(), let n = Float(value) { config.sizzleHighHz = n }
    case "--sizzle-delay-ms":
      if let value = next(), let n = Float(value) { config.sizzleDelayMs = n }
    case "--sizzle-fade-ms":
      if let value = next(), let n = Float(value) { config.sizzleFadeMs = n }
    case "--sizzle-post-drive":
      config.sizzlePostDrive = true
    case "--reset-sizzle":
      config.resetSizzle = true
    case "--air-sizzle-frames":
      if let value = next(), let n = Int(value) { config.airSizzleFrames = n }
    case "--air-sizzle-points":
      if let value = next(), let n = Int(value) { config.airSizzlePoints = n }
    case "--air-sizzle-init":
      if let value = next() { config.airSizzleInit = value }
    case "--air-sizzle-amp-init":
      if let value = next(), let n = Float(value) { config.airSizzleAmpInit = n }
    case "--air-sizzle-low-hz":
      if let value = next(), let n = Float(value) { config.airSizzleLowHz = n }
    case "--air-sizzle-high-hz":
      if let value = next(), let n = Float(value) { config.airSizzleHighHz = n }
    case "--air-sizzle-delay-ms":
      if let value = next(), let n = Float(value) { config.airSizzleDelayMs = n }
    case "--air-sizzle-fade-ms":
      if let value = next(), let n = Float(value) { config.airSizzleFadeMs = n }
    case "--air-sizzle-pre-drive":
      config.airSizzlePostDrive = false
    case "--reset-air-sizzle":
      config.resetAirSizzle = true
    case "--air-noise-amp-init":
      if let value = next(), let n = Float(value) { config.airNoiseAmpInit = n }
    case "--air-noise-low-hz":
      if let value = next(), let n = Float(value) { config.airNoiseLowHz = n }
    case "--air-noise-high-hz":
      if let value = next(), let n = Float(value) { config.airNoiseHighHz = n }
    case "--air-noise-delay-ms":
      if let value = next(), let n = Float(value) { config.airNoiseDelayMs = n }
    case "--air-noise-fade-ms":
      if let value = next(), let n = Float(value) { config.airNoiseFadeMs = n }
    case "--fm-noise-frames":
      if let value = next(), let n = Int(value) { config.fmNoiseFrames = n }
    case "--fm-noise-points":
      if let value = next(), let n = Int(value) { config.fmNoisePoints = n }
    case "--fm-noise-amp-init":
      if let value = next(), let n = Float(value) { config.fmNoiseAmpInit = n }
    case "--fm-noise-pm-init":
      if let value = next(), let n = Float(value) { config.fmNoisePMInit = n }
    case "--fm-noise-low-hz":
      if let value = next(), let n = Float(value) { config.fmNoiseLowHz = n }
    case "--fm-noise-high-hz":
      if let value = next(), let n = Float(value) { config.fmNoiseHighHz = n }
    case "--fm-noise-delay-ms":
      if let value = next(), let n = Float(value) { config.fmNoiseDelayMs = n }
    case "--fm-noise-fade-ms":
      if let value = next(), let n = Float(value) { config.fmNoiseFadeMs = n }
    case "--reset-fm-noise":
      config.resetFMNoise = true
    case "--crisp-noise-frames":
      if let value = next(), let n = Int(value) { config.crispNoiseFrames = n }
    case "--crisp-noise-points":
      if let value = next(), let n = Int(value) { config.crispNoisePoints = n }
    case "--crisp-noise-init":
      if let value = next() { config.crispNoiseInit = value }
    case "--crisp-noise-amp-init":
      if let value = next(), let n = Float(value) { config.crispNoiseAmpInit = n }
    case "--crisp-noise-spacing":
      if let value = next(), let n = Float(value) { config.crispNoiseSpacing = n }
    case "--crisp-noise-burst-decay":
      if let value = next(), let n = Float(value) { config.crispNoiseBurstDecay = n }
    case "--crisp-noise-low-hz":
      if let value = next(), let n = Float(value) { config.crispNoiseLowHz = n }
    case "--crisp-noise-high-hz":
      if let value = next(), let n = Float(value) { config.crispNoiseHighHz = n }
    case "--crisp-noise-hp-stages":
      if let value = next(), let n = Int(value) { config.crispNoiseHPStages = n }
    case "--crisp-noise-delay-ms":
      if let value = next(), let n = Float(value) { config.crispNoiseDelayMs = n }
    case "--crisp-noise-fade-ms":
      if let value = next(), let n = Float(value) { config.crispNoiseFadeMs = n }
    case "--presence-duck-init":
      if let value = next(), let n = Float(value) { config.presenceDuckInit = n }
    case "--presence-duck-low-hz":
      if let value = next(), let n = Float(value) { config.presenceDuckLowHz = n }
    case "--presence-duck-high-hz":
      if let value = next(), let n = Float(value) { config.presenceDuckHighHz = n }
    case "--air-mode-amp-init":
      if let value = next(), let n = Float(value) { config.airModeAmpInit = n }
    case "--air-mode-freq-init":
      if let value = next(), let n = Float(value) { config.airModeFreqInit = n }
    case "--air-mode-delay-ms":
      if let value = next(), let n = Float(value) { config.airModeDelayMs = n }
    case "--air-mode-fade-ms":
      if let value = next(), let n = Float(value) { config.airModeFadeMs = n }
    case "--air-mode-highpass-hz":
      if let value = next(), let n = Float(value) { config.airModeHighpassHz = n }
    case "--air-mode-lowpass-hz":
      if let value = next(), let n = Float(value) { config.airModeLowpassHz = n }
    case "--transient-mode-amp-init":
      if let value = next(), let n = Float(value) { config.transientModeAmpInit = n }
    case "--transient-mode-ms":
      if let value = next(), let n = Float(value) { config.transientModeMs = n }
    case "--transient-mode-follow-body":
      config.transientModeFollowBody = true
    case "--transient-mode-post-drive":
      config.transientModePostDrive = true
    case "--subbody-frames":
      if let value = next(), let n = Int(value) { config.subBodyFrames = n }
    case "--subbody-points":
      if let value = next(), let n = Int(value) { config.subBodyPoints = n }
    case "--subbody-init":
      if let value = next() { config.subBodyInit = value }
    case "--subbody-amp-init":
      if let value = next(), let n = Float(value) { config.subBodyAmpInit = n }
    case "--subbody-hz":
      if let value = next(), let n = Float(value) { config.subBodyHz = n }
    case "--subbody-pre-drive":
      config.subBodyPostDrive = false
    case "--clean-body-post-drive":
      config.cleanBodyPostDrive = true
      config.cleanSubPostDrive = true
    case "--clean-sub-post-drive":
      config.cleanSubPostDrive = true
    case "--clean-body-scale":
      if let value = next(), let n = Float(value) { config.cleanBodyScale = n }
    case "--eq-filter-size":
      if let value = next(), let n = Int(value) { config.eqFilterSize = n }
    case "--selection-every":
      if let value = next(), let n = Int(value) { config.selectionEvery = n }
    case "--checkpoint-selection":
      if let value = next() { config.checkpointSelection = value }
    case "--perceptual-checkpoint-out":
      if let value = next() { config.perceptualCheckpointOut = value }
    case "--sustained-sub-weight":
      if let value = next(), let n = Float(value) { config.sustainedSubWeight = n }
    case "--sustained-presence-weight":
      if let value = next(), let n = Float(value) { config.sustainedPresenceWeight = n }
    case "--sustained-air-weight":
      if let value = next(), let n = Float(value) { config.sustainedAirWeight = n }
    case "--sustained-hf-weight":
      if let value = next(), let n = Float(value) { config.sustainedHFWeight = n }
    case "--render-only":
      config.renderOnly = true
    case "--checkpoint-in":
      if let value = next() { config.checkpointIn = value }
    case "--checkpoint-out":
      if let value = next() { config.checkpointOut = value }
    case "--ignore-checkpoint-config":
      break
    case "--help", "-h":
      print(
        """
        Usage: swift run TrainKick808 [options]

          --target <wav>     Target kick WAV
          --frames <n>       Training frame count (default: 4096)
          --epochs <n>       Training epochs (default: 40)
          --window <n>       Spectral loss FFT window (default: 512)
          --out <dir>        Output directory
          --lr-scale <x>     Multiplier for optimizer learning rates
          --waveform-weight <x>
          --transient-weight <x>
          --slope-weight <x>
          --high-band-weight <x>
          --high-band-spectral-weight <x>
          --sub-band-weight <x>
          --presence-band-weight <x>
          --air-band-weight <x>
          --env-size <n>     Learnable amplitude envelope points
          --freq-scalar-lr-scale <x>
          --amp-lr-scale <x>
          --decay-lr-scale <x>
          --env-lr-scale <x>
          --freq-curve-lr-scale <x>
          --phase-warp-lr-scale <x>
          --body-wave-lr-scale <x>
          --residual-lr-scale <x>
          --noise-lr-scale <x>
          --sizzle-lr-scale <x>
          --air-sizzle-lr-scale <x>
          --air-noise-lr-scale <x>
          --subbody-lr-scale <x>
          --eq-lr-scale <x>
          --residual-frames <n>
          --residual-points <n>
          --phase-warp-frames <n>
          --phase-warp-points <n>
          --phase-warp-amp-init <x>
          --body-wave-points <n>
          --body-wave-mix-init <x>
          --noise-filter-size <n>
          --sizzle-frames <n>
          --sizzle-points <n>
          --sizzle-init zero|target
          --sizzle-amp-init <x>
          --sizzle-low-hz <hz>
          --sizzle-high-hz <hz>
          --sizzle-delay-ms <ms>
          --sizzle-fade-ms <ms>
          --sizzle-post-drive
          --reset-sizzle
          --air-sizzle-frames <n>
          --air-sizzle-points <n>
          --air-sizzle-init zero|target
          --air-sizzle-amp-init <x>
          --air-sizzle-low-hz <hz>
          --air-sizzle-high-hz <hz>
          --air-sizzle-delay-ms <ms>
          --air-sizzle-fade-ms <ms>
          --air-sizzle-pre-drive
          --reset-air-sizzle
          --air-noise-amp-init <x>
          --air-noise-low-hz <hz>
          --air-noise-high-hz <hz>
          --air-noise-delay-ms <ms>
          --air-noise-fade-ms <ms>
          --fm-noise-lr-scale <x>
          --air-mode-lr-scale <x>
          --transient-mode-lr-scale <x>
          --fm-noise-frames <n>
          --fm-noise-points <n>
          --fm-noise-amp-init <x>
          --fm-noise-pm-init <x>
          --fm-noise-low-hz <hz>
          --fm-noise-high-hz <hz>
          --fm-noise-delay-ms <ms>
          --fm-noise-fade-ms <ms>
          --reset-fm-noise
          --crisp-noise-lr-scale <x>
          --crisp-noise-frames <n>
          --crisp-noise-points <n>
          --crisp-noise-init noise|burst
          --crisp-noise-amp-init <x>
          --crisp-noise-spacing <frames>
          --crisp-noise-burst-decay <x>
          --crisp-noise-low-hz <hz>
          --crisp-noise-high-hz <hz>
          --crisp-noise-hp-stages <n>
          --crisp-noise-delay-ms <ms>
          --crisp-noise-fade-ms <ms>
          --presence-duck-init <x>
          --presence-duck-low-hz <hz>
          --presence-duck-high-hz <hz>
          --air-mode-amp-init <x>
          --air-mode-freq-init <hz>
          --air-mode-delay-ms <ms>
          --air-mode-fade-ms <ms>
          --air-mode-highpass-hz <hz>
          --air-mode-lowpass-hz <hz>
          --transient-mode-amp-init <x>
          --transient-mode-ms <ms>
          --transient-mode-follow-body
          --transient-mode-post-drive
          --subbody-frames <n>
          --subbody-points <n>
          --subbody-init zero|target
          --subbody-amp-init <x>
          --subbody-hz <hz>
          --subbody-pre-drive
          --clean-body-post-drive
          --clean-sub-post-drive
          --clean-body-scale <x>
          --eq-filter-size <n>
          --selection-every <n>
          --checkpoint-selection loss|perceptual
          --perceptual-checkpoint-out <json>
          --sustained-sub-weight <x>
          --sustained-presence-weight <x>
          --sustained-air-weight <x>
          --sustained-hf-weight <x>
          --render-only
          --checkpoint-in <json>
          --checkpoint-out <json>
          --ignore-checkpoint-config
        """)
      exit(0)
    default:
      if !arg.hasPrefix("-") {
        config.targetPath = arg
      } else {
        fputs("Unknown argument: \(arg)\n", stderr)
      }
    }
    i += 1
  }
  return config
}

func transientWindow(_ samples: [Float], frames: Int) -> [Float] {
  let searchCount = min(samples.count, max(frames * 2, frames))
  let threshold = (samples.prefix(searchCount).map { abs($0) }.max() ?? 0) * 0.08
  let onset = samples.prefix(searchCount).firstIndex { abs($0) >= threshold } ?? 0
  let paddedStart = max(0, onset - 16)
  var out = Array(samples.dropFirst(paddedStart).prefix(frames))
  if out.count < frames {
    out += [Float](repeating: 0, count: frames - out.count)
  }
  let peak = out.map { abs($0) }.max() ?? 1
  if peak > 0 {
    out = out.map { $0 / peak * 0.9 }
  }
  return out
}

func findOnset(_ samples: [Float], searchFrames: Int) -> Int {
  let searchCount = min(samples.count, searchFrames)
  let peak = samples.prefix(searchCount).map { abs($0) }.max() ?? 0
  let threshold = peak * 0.08
  return samples.prefix(searchCount).firstIndex { abs($0) >= threshold } ?? 0
}

func estimateFundamental(
  samples: [Float],
  sampleRate: Float,
  startFrame: Int,
  frameCount: Int,
  minHz: Float = 25,
  maxHz: Float = 220
) -> Float? {
  guard startFrame < samples.count else { return nil }
  let endFrame = min(samples.count, startFrame + frameCount)
  let window = Array(samples[startFrame..<endFrame])
  guard window.count >= Int(sampleRate / minHz) else { return nil }

  let mean = window.reduce(0, +) / Float(window.count)
  let centered = window.map { $0 - mean }
  let energy = centered.reduce(Float(0)) { $0 + $1 * $1 }
  guard energy > 1e-8 else { return nil }

  let minLag = max(1, Int(sampleRate / maxHz))
  let maxLag = min(centered.count - 1, Int(sampleRate / minHz))
  guard minLag < maxLag else { return nil }

  var bestLag = minLag
  var bestScore: Float = -.infinity
  for lag in minLag...maxLag {
    var sum: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    let limit = centered.count - lag
    for i in 0..<limit {
      let a = centered[i]
      let b = centered[i + lag]
      sum += a * b
      normA += a * a
      normB += b * b
    }
    let score = sum / max(sqrt(normA * normB), 1e-8)
    if score > bestScore {
      bestScore = score
      bestLag = lag
    }
  }

  guard bestScore > 0.15 else { return nil }
  return sampleRate / Float(bestLag)
}

func pitchTrack(samples: [Float], sampleRate: Float, onset: Int) -> [(ms: Float, hz: Float)] {
  let offsetsMs: [Float] = [6, 14, 24, 38, 56, 76]
  return offsetsMs.compactMap { ms in
    let start = onset + Int(sampleRate * ms / 1000.0)
    guard
      let hz = estimateFundamental(
        samples: samples,
        sampleRate: sampleRate,
        startFrame: start,
        frameCount: Int(sampleRate * 0.045),
        minHz: 25,
        maxHz: 220
      )
    else { return nil }
    return (ms: ms, hz: hz)
  }
}

func peakEnvelope(samples: [Float], size: Int) -> [Float] {
  let samplesPerPoint = max(1, samples.count / size)
  return (0..<size).map { i in
    let start = i * samplesPerPoint
    let end = min(samples.count, start + samplesPerPoint)
    guard start < end else { return Float(0.0) }
    return samples[start..<end].map { abs($0) }.max() ?? 0.0
  }
}

func highFrequencyEnvelope(samples: [Float], sampleRate: Float, size: Int) -> [Float] {
  let hp = highPassSamples(samples: samples, sampleRate: sampleRate, cutoff: 1_800.0)
  var env = peakEnvelope(samples: hp, size: size)
  let peak = env.max() ?? 0.0
  if peak > 1e-9 {
    env = env.map { min(1.0, max(0.015, $0 / peak)) }
  }

  return env.enumerated().map { i, value in
    let ms = Float(i) / Float(max(1, size - 1)) * Float(samples.count) / sampleRate * 1000.0
    if ms < 3.0 {
      return min(0.55, value * 0.7)
    }
    if ms <= 35.0 {
      return max(0.06, value)
    }
    return max(0.01, value * exp(-(ms - 35.0) / 18.0))
  }
}

func highPassSamples(samples: [Float], sampleRate: Float, cutoff: Float) -> [Float] {
  var hp = [Float](repeating: 0.0, count: samples.count)
  var low: Float = 0.0
  let alpha = exp(-2.0 * Float.pi * cutoff / sampleRate)
  for i in samples.indices {
    low = alpha * low + (1.0 - alpha) * samples[i]
    hp[i] = samples[i] - low
  }
  return hp
}

func lowPassSamples(samples: [Float], sampleRate: Float, cutoff: Float) -> [Float] {
  var out = [Float](repeating: 0.0, count: samples.count)
  var low: Float = 0.0
  let alpha = exp(-2.0 * Float.pi * cutoff / sampleRate)
  for i in samples.indices {
    low = alpha * low + (1.0 - alpha) * samples[i]
    out[i] = low
  }
  return out
}

func sinc(_ x: Float) -> Float {
  if abs(x) < 1e-6 { return 1.0 }
  return sin(Float.pi * x) / (Float.pi * x)
}

func firBandPassSamples(
  samples: [Float],
  sampleRate: Float,
  lowHz: Float,
  highHz: Float,
  taps: Int = 129
) -> [Float] {
  guard !samples.isEmpty, highHz > lowHz else { return samples }
  let tapCount = max(5, taps | 1)
  let center = tapCount / 2
  let low = max(0.0, min(0.49, lowHz / sampleRate))
  let high = max(low + 0.001, min(0.49, highHz / sampleRate))
  var kernel = [Float](repeating: 0.0, count: tapCount)

  for i in 0..<tapCount {
    let n = Float(i - center)
    let window = 0.54 - 0.46 * cos(2.0 * Float.pi * Float(i) / Float(tapCount - 1))
    let lowPassHigh = 2.0 * high * sinc(2.0 * high * n)
    let lowPassLow = 2.0 * low * sinc(2.0 * low * n)
    kernel[i] = (lowPassHigh - lowPassLow) * window
  }

  var out = [Float](repeating: 0.0, count: samples.count)
  for i in samples.indices {
    var sum: Float = 0.0
    for k in 0..<tapCount {
      let sampleIndex = i + k - center
      if sampleIndex >= 0 && sampleIndex < samples.count {
        sum += samples[sampleIndex] * kernel[k]
      }
    }
    out[i] = sum
  }
  return out
}

func bandLimitedBodyLayer(
  samples: [Float],
  sampleRate: Float,
  size: Int,
  lowHz: Float,
  highHz: Float
) -> [Float] {
  var band = highHz > lowHz
    ? firBandPassSamples(samples: samples, sampleRate: sampleRate, lowHz: lowHz, highHz: highHz)
    : highPassSamples(samples: samples, sampleRate: sampleRate, cutoff: lowHz)
  if highHz > lowHz {
    band = highPassSamples(samples: band, sampleRate: sampleRate, cutoff: lowHz)
  }
  var out = [Float](repeating: 0.0, count: size)
  guard !samples.isEmpty else { return out }

  for i in 0..<size {
    let frame = Float(i) / Float(max(1, size - 1)) * Float(samples.count - 1)
    let a = max(0, min(samples.count - 1, Int(frame)))
    let b = min(samples.count - 1, a + 1)
    let mix = frame - Float(a)
    let sample = band[a] * (1.0 - mix) + band[b] * mix
    let ms = frame / sampleRate * 1000.0
    let fadeIn = min(1.0, max(0.0, (ms - 3.0) / 4.0))
    let fadeOut = ms <= 42.0 ? 1.0 : exp(-(ms - 42.0) / 16.0)
    out[i] = sample * fadeIn * fadeOut
  }

  let peak = out.map { abs($0) }.max() ?? 0.0
  if peak > 1.0 {
    out = out.map { $0 / peak }
  }
  return out
}

func highFrequencyBodyLayer(samples: [Float], sampleRate: Float, size: Int) -> [Float] {
  bandLimitedBodyLayer(
    samples: samples,
    sampleRate: sampleRate,
    size: size,
    lowHz: 1_600.0,
    highHz: 0.0)
}

func deterministicNoiseTable(size: Int) -> [Float] {
  guard size > 0 else { return [] }
  return (0..<size).map { i in
    let a = sin(Float(i + 1) * 12.9898) * 43_758.547
    let b = sin(Float(i + 17) * 78.233) * 12_345.679
    let frac = (a + b) - floor(a + b)
    return frac * 2.0 - 1.0
  }
}

func deterministicBurstTable(size: Int, spacing: Float, decay: Float) -> [Float] {
  guard size > 0 else { return [] }
  let safeSpacing = max(1.0, spacing)
  let safeDecay = min(0.999, max(0.1, decay))
  return (0..<size).map { i in
    let impulseIndex = Int(Float(i) / safeSpacing)
    let nearest = Float(impulseIndex) * safeSpacing
    let distance = abs(Float(i) - nearest)
    let width = max(1.0, safeSpacing * 0.12)
    let pulse = max(0.0, 1.0 - distance / width)
    let sign: Float = impulseIndex % 2 == 0 ? 1.0 : -1.0
    let jitter = sin(Float(impulseIndex + 3) * 2.39996)
    return sign * pulse * pow(safeDecay, Float(impulseIndex)) * (0.75 + 0.25 * jitter)
  }
}

func crispExciterTable(config: Config, size: Int) -> [Float] {
  if config.crispNoiseInit == "burst" {
    return deterministicBurstTable(
      size: size,
      spacing: config.crispNoiseSpacing,
      decay: config.crispNoiseBurstDecay)
  }
  return deterministicNoiseTable(size: size)
}

func sineWaveTable(size: Int) -> [Float] {
  guard size > 0 else { return [] }
  return (0..<size).map { i in
    let phase = Float(i) / Float(max(1, size - 1)) * Float.pi * 2.0
    return sin(phase)
  }
}

func lowFrequencyBodyLayer(samples: [Float], sampleRate: Float, size: Int, cutoffHz: Float) -> [Float] {
  let low = lowPassSamples(samples: samples, sampleRate: sampleRate, cutoff: cutoffHz)
  var out = [Float](repeating: 0.0, count: size)
  guard !samples.isEmpty else { return out }

  for i in 0..<size {
    let frame = Float(i) / Float(max(1, size - 1)) * Float(samples.count - 1)
    let a = max(0, min(samples.count - 1, Int(frame)))
    let b = min(samples.count - 1, a + 1)
    let mix = frame - Float(a)
    let sample = low[a] * (1.0 - mix) + low[b] * mix
    let ms = frame / sampleRate * 1000.0
    let fadeIn = min(1.0, max(0.0, ms / 3.0))
    let fadeOut = ms <= 70.0 ? 1.0 : exp(-(ms - 70.0) / 45.0)
    out[i] = sample * fadeIn * fadeOut
  }

  let peak = out.map { abs($0) }.max() ?? 0.0
  if peak > 1.0 {
    out = out.map { $0 / peak }
  }
  return out
}

func zeroCrossingFrequencyCurve(samples: [Float], sampleRate: Float, size: Int, fallbackEndHz: Float) -> [Float] {
  var crossings: [Int] = []
  var previous = samples.first ?? 0
  for i in 1..<samples.count {
    let value = samples[i]
    if (previous < 0 && value >= 0) || (previous > 0 && value <= 0) {
      crossings.append(i)
    }
    if value != 0 {
      previous = value
    }
  }

  var points: [(frame: Float, hz: Float)] = []
  for i in 0..<(crossings.count - 2) {
    let a = crossings[i]
    let b = crossings[i + 2]
    guard b > a else { continue }
    let center = Float(a + b) * 0.5
    let hz = sampleRate / Float(b - a)
    if hz >= 25 && hz <= 900 {
      points.append((center, hz))
    }
  }

  guard !points.isEmpty else {
    return (0..<size).map { i -> Float in
      let x = Float(i) / Float(max(1, size - 1))
      let frames = x * Float(samples.count)
      return fallbackEndHz + (390.0 - fallbackEndHz) * exp(-frames / 650.0)
    }
  }

  var curve: [Float] = []
  for i in 0..<size {
    let frame = Float(i) / Float(max(1, size - 1)) * Float(samples.count - 1)
    if frame <= points[0].frame {
      curve.append(points[0].hz)
      continue
    }
    if frame >= points.last!.frame {
      let tailFrame = frame - points.last!.frame
      curve.append(fallbackEndHz + (points.last!.hz - fallbackEndHz) * exp(-tailFrame / 1800.0))
      continue
    }
    var j = 0
    while j + 1 < points.count && points[j + 1].frame < frame {
      j += 1
    }
    let a = points[j]
    let b = points[j + 1]
    let mix = (frame - a.frame) / max(1.0, b.frame - a.frame)
    curve.append(a.hz * (1 - mix) + b.hz * mix)
  }
  return curve.map { max(25.0, min(900.0, $0)) }
}

@discardableResult
func timed<T>(_ block: () throws -> T) rethrows -> (T, Double) {
  let start = DispatchTime.now().uptimeNanoseconds
  let value = try block()
  let end = DispatchTime.now().uptimeNanoseconds
  return (value, Double(end - start) / 1_000_000.0)
}

struct PerceptualStats {
  var score: Float
  var normMSE: Float
  var transientMSE: Float
  var centroidDelta: Float
  var subDelta: Float
  var bodyDelta: Float
  var presenceDelta: Float
  var airDelta: Float
  var hfDelta: Float
  var zeroCrossingDelta: Int
}

func normalizedSamples(_ samples: [Float]) -> [Float] {
  let peak = samples.map { abs($0) }.max() ?? 0.0
  guard peak > 1e-12 else { return samples }
  return samples.map { $0 / peak }
}

func sampleMSE(_ a: [Float], _ b: [Float]) -> Float {
  let n = min(a.count, b.count)
  guard n > 0 else { return 0.0 }
  var sum: Float = 0.0
  for i in 0..<n {
    let delta = a[i] - b[i]
    sum += delta * delta
  }
  return sum / Float(n)
}

func sampleZeroCrossings(_ samples: ArraySlice<Float>) -> Int {
  guard var previous = samples.first else { return 0 }
  var count = 0
  for value in samples.dropFirst() {
    if (previous < 0 && value >= 0) || (previous > 0 && value <= 0) {
      count += 1
    }
    if value != 0 {
      previous = value
    }
  }
  return count
}

func nextPowerOfTwo(_ n: Int) -> Int {
  var value = 1
  while value < n { value *= 2 }
  return value
}

func bandPowers(
  samples: ArraySlice<Float>,
  sampleRate: Float,
  bands: [(name: String, low: Float, high: Float)]
) -> (powers: [String: Float], centroid: Float) {
  guard !samples.isEmpty else {
    return (Dictionary(uniqueKeysWithValues: bands.map { ($0.name, 0.0) }), 0.0)
  }

  let source = Array(samples)
  let nFFT = max(256, nextPowerOfTwo(source.count))
  var windowed = [Float](repeating: 0.0, count: nFFT)
  for i in 0..<source.count {
    let window = source.count > 1
      ? 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(source.count - 1))
      : 1.0
    windowed[i] = source[i] * window
  }

  var powers = Dictionary(uniqueKeysWithValues: bands.map { ($0.name, Float(0.0)) })
  var centroidNum: Float = 0.0
  var centroidDen: Float = 0.0
  for k in 0...(nFFT / 2) {
    let freq = Float(k) * sampleRate / Float(nFFT)
    var real: Float = 0.0
    var imag: Float = 0.0
    for n in 0..<nFFT where windowed[n] != 0.0 {
      let angle = -2.0 * Float.pi * Float(k * n) / Float(nFFT)
      real += windowed[n] * cos(angle)
      imag += windowed[n] * sin(angle)
    }
    let power = (real * real + imag * imag) / Float(nFFT)
    centroidNum += freq * power
    centroidDen += power
    for band in bands where freq >= band.low && freq < band.high {
      powers[band.name, default: 0.0] += power
    }
  }

  return (powers, centroidDen > 1e-18 ? centroidNum / centroidDen : 0.0)
}

func dbRatio(_ learned: Float, _ target: Float) -> Float {
  10.0 * log10(max(learned, 1e-18) / max(target, 1e-18))
}

func deficit(_ value: Float) -> Float {
  max(0.0, -value)
}

func excess(_ value: Float) -> Float {
  max(0.0, value)
}

func timeSlicePenalty(learned: [Float], target: [Float], sampleRate: Float) -> Float {
  let bands = [
    (name: "low_0_1000", low: Float(0.0), high: Float(1_000.0)),
    (name: "air_4000_12000", low: Float(4_000.0), high: Float(12_000.0)),
    (name: "hf_2000_16000", low: Float(2_000.0), high: Float(16_000.0)),
  ]
  let slices: [(Float, Float)] = [(0.005, 0.010), (0.010, 0.020), (0.020, 0.030)]
  var score: Float = 0.0
  for (startS, endS) in slices {
    let start = min(learned.count, Int(sampleRate * startS))
    let end = min(learned.count, target.count, Int(sampleRate * endS))
    guard end > start else { continue }
    let learnedBands = bandPowers(samples: learned[start..<end], sampleRate: sampleRate, bands: bands).powers
    let targetBands = bandPowers(samples: target[start..<end], sampleRate: sampleRate, bands: bands).powers
    let airDelta = dbRatio(learnedBands["air_4000_12000"] ?? 0, targetBands["air_4000_12000"] ?? 0)
    let hfDelta = dbRatio(learnedBands["hf_2000_16000"] ?? 0, targetBands["hf_2000_16000"] ?? 0)
    let learnedHFRatio = dbRatio(learnedBands["hf_2000_16000"] ?? 0, learnedBands["low_0_1000"] ?? 0)
    let targetHFRatio = dbRatio(targetBands["hf_2000_16000"] ?? 0, targetBands["low_0_1000"] ?? 0)
    score += deficit(airDelta) / 4.0
    score += deficit(hfDelta) / 5.0
    score += abs(learnedHFRatio - targetHFRatio) / 10.0
  }
  return score
}

func perceptualStats(learned rawLearned: [Float], target rawTarget: [Float], sampleRate: Float) -> PerceptualStats {
  let n = min(rawLearned.count, rawTarget.count)
  let learned = normalizedSamples(Array(rawLearned.prefix(n)))
  let target = normalizedSamples(Array(rawTarget.prefix(n)))
  let transientN = min(n, Int(sampleRate * 0.030))
  let bands = [
    (name: "sub_0_200", low: Float(0.0), high: Float(200.0)),
    (name: "body_200_1000", low: Float(200.0), high: Float(1_000.0)),
    (name: "presence_1000_4000", low: Float(1_000.0), high: Float(4_000.0)),
    (name: "air_4000_12000", low: Float(4_000.0), high: Float(12_000.0)),
    (name: "hf_2000_16000", low: Float(2_000.0), high: Float(16_000.0)),
  ]
  let learnedBands = bandPowers(samples: learned[0..<transientN], sampleRate: sampleRate, bands: bands)
  let targetBands = bandPowers(samples: target[0..<transientN], sampleRate: sampleRate, bands: bands)
  let subDelta = dbRatio(learnedBands.powers["sub_0_200"] ?? 0, targetBands.powers["sub_0_200"] ?? 0)
  let bodyDelta = dbRatio(learnedBands.powers["body_200_1000"] ?? 0, targetBands.powers["body_200_1000"] ?? 0)
  let presenceDelta = dbRatio(learnedBands.powers["presence_1000_4000"] ?? 0, targetBands.powers["presence_1000_4000"] ?? 0)
  let airDelta = dbRatio(learnedBands.powers["air_4000_12000"] ?? 0, targetBands.powers["air_4000_12000"] ?? 0)
  let hfDelta = dbRatio(learnedBands.powers["hf_2000_16000"] ?? 0, targetBands.powers["hf_2000_16000"] ?? 0)
  let normMSE = sampleMSE(learned, target)
  let transientMSE = sampleMSE(Array(learned.prefix(transientN)), Array(target.prefix(transientN)))
  let zeroDelta = sampleZeroCrossings(learned[0..<transientN])
    - sampleZeroCrossings(target[0..<transientN])
  let centroidDelta = learnedBands.centroid - targetBands.centroid
  let score = normMSE * 120.0
    + transientMSE * 80.0
    + abs(centroidDelta) / 90.0
    + abs(subDelta) / 4.0
    + abs(bodyDelta) / 10.0
    + excess(presenceDelta) / 12.0
    + deficit(airDelta) / 3.0
    + deficit(hfDelta) / 4.0
    + timeSlicePenalty(learned: learned, target: target, sampleRate: sampleRate)
    + Float(abs(zeroDelta)) / 8.0
  return PerceptualStats(
    score: score,
    normMSE: normMSE,
    transientMSE: transientMSE,
    centroidDelta: centroidDelta,
    subDelta: subDelta,
    bodyDelta: bodyDelta,
    presenceDelta: presenceDelta,
    airDelta: airDelta,
    hfDelta: hfDelta,
    zeroCrossingDelta: zeroDelta)
}

func run() throws {
  var config = parseArgs()
  let ignoreCheckpointConfig = CommandLine.arguments.contains("--ignore-checkpoint-config")
  if !ignoreCheckpointConfig,
     let checkpointIn = config.checkpointIn,
     let savedConfig = try loadCheckpointConfig(path: checkpointIn) {
    config = mergedCheckpointConfig(saved: savedConfig, cli: config)
    print("checkpoint config loaded=\(checkpointIn)")
  }
  if config.windowSize > config.frames {
    config.windowSize = config.frames
  }

  let targetURL = URL(fileURLWithPath: config.targetPath)
  let (rawSamples, sampleRate) = try AudioFile.load(url: targetURL, mono: true)
  let onset = findOnset(rawSamples, searchFrames: max(config.frames * 2, config.frames))
  let targetSamples = transientWindow(rawSamples, frames: config.frames)
  let targetEnv = peakEnvelope(samples: targetSamples, size: config.envSize)
  let targetNoiseEnv = highFrequencyEnvelope(samples: targetSamples, sampleRate: sampleRate, size: config.envSize)
  let analyzedPitches = pitchTrack(samples: rawSamples, sampleRate: sampleRate, onset: onset)
  let endFreqGuess = estimateFundamental(
    samples: rawSamples,
    sampleRate: sampleRate,
    startFrame: onset + Int(sampleRate * 0.045),
    frameCount: Int(sampleRate * 0.080),
    minHz: 25,
    maxHz: 140
  ) ?? analyzedPitches.suffix(2).map(\.hz).min() ?? 52.0
  let earlyMax = analyzedPitches.filter { $0.ms <= 38 }.map(\.hz).max()
  let startFreqGuess = min(max(earlyMax ?? endFreqGuess * 3.0, endFreqGuess * 2.6), 180.0)

  try FileManager.default.createDirectory(
    atPath: config.outputDir, withIntermediateDirectories: true)

  DGenConfig.sampleRate = sampleRate
  let previousMaxFrames = DGenConfig.maxFrameCount
  DGenConfig.maxFrameCount = config.frames
  defer { DGenConfig.maxFrameCount = previousMaxFrames }
  LazyGraphContext.reset()

  let firstTarget = targetSamples.first ?? 0.0
  let initialPhase = asin(max(-0.95, min(0.95, firstTarget / 0.9)))
  let targetTensor = Tensor(targetSamples)
  let bodyEnvTensor = Tensor.param([config.envSize], data: targetEnv.map { max($0, 0.03) })
  bodyEnvTensor.minBound = 0.0
  bodyEnvTensor.maxBound = 1.5
  let subEnvTensor = Tensor.param([config.envSize], data: targetEnv.map { max($0 * 0.25, 0.01) })
  subEnvTensor.minBound = 0.0
  subEnvTensor.maxBound = 1.2
  let harmEnvTensor = Tensor.param([config.envSize], data: targetEnv.map { max($0 * 0.18, 0.01) })
  harmEnvTensor.minBound = 0.0
  harmEnvTensor.maxBound = 1.2
  let clickFrontPoints = max(4, config.envSize / 8)
  let clickEnvTensor = Tensor.param([config.envSize], data: targetEnv.enumerated().map { i, v in
    i < clickFrontPoints ? max(v * 0.5, 0.02) : 0.0
  })
  clickEnvTensor.minBound = 0.0
  clickEnvTensor.maxBound = 1.5
  let freqCurveInit = zeroCrossingFrequencyCurve(
    samples: targetSamples,
    sampleRate: sampleRate,
    size: config.envSize,
    fallbackEndHz: endFreqGuess)
  let freqCurveTensor = Tensor.param([config.envSize], data: freqCurveInit)
  freqCurveTensor.minBound = 25.0
  freqCurveTensor.maxBound = 900.0
  let residualFrames = max(1, min(config.residualFrames, config.frames))
  let residualPointCount = max(2, config.residualPoints ?? residualFrames)
  let residualTensor = Tensor.param(
    [residualPointCount], data: [Float](repeating: 0.0, count: residualPointCount))
  residualTensor.minBound = -1.5
  residualTensor.maxBound = 1.5
  let sizzleFrames = max(1, min(config.sizzleFrames, config.frames))
  let sizzlePointCount = max(2, config.sizzlePoints ?? sizzleFrames)
  let sizzleInit: [Float]
  if config.sizzleInit == "target" {
    sizzleInit = bandLimitedBodyLayer(
      samples: targetSamples,
      sampleRate: sampleRate,
      size: sizzlePointCount,
      lowHz: config.sizzleLowHz,
      highHz: config.sizzleHighHz)
  } else {
    sizzleInit = [Float](repeating: 0.0, count: sizzlePointCount)
  }
  let sizzleTensor = Tensor.param([sizzlePointCount], data: sizzleInit)
  sizzleTensor.minBound = -1.5
  sizzleTensor.maxBound = 1.5
  let airSizzleFrames = max(1, min(config.airSizzleFrames, config.frames))
  let airSizzlePointCount = max(2, config.airSizzlePoints ?? airSizzleFrames)
  let airSizzleInit: [Float]
  if config.airSizzleInit == "target" {
    airSizzleInit = bandLimitedBodyLayer(
      samples: targetSamples,
      sampleRate: sampleRate,
      size: airSizzlePointCount,
      lowHz: config.airSizzleLowHz,
      highHz: config.airSizzleHighHz)
  } else {
    airSizzleInit = [Float](repeating: 0.0, count: airSizzlePointCount)
  }
  let airSizzleTensor = Tensor.param([airSizzlePointCount], data: airSizzleInit)
  airSizzleTensor.minBound = -1.5
  airSizzleTensor.maxBound = 1.5
  let subBodyFrames = max(1, min(config.subBodyFrames, config.frames))
  let subBodyPointCount = max(2, config.subBodyPoints ?? subBodyFrames)
  let subBodyInit: [Float]
  if config.subBodyInit == "target" {
    subBodyInit = lowFrequencyBodyLayer(
      samples: targetSamples,
      sampleRate: sampleRate,
      size: subBodyPointCount,
      cutoffHz: config.subBodyHz)
  } else {
    subBodyInit = [Float](repeating: 0.0, count: subBodyPointCount)
  }
  let subBodyTensor = Tensor.param([subBodyPointCount], data: subBodyInit)
  subBodyTensor.minBound = -1.5
  subBodyTensor.maxBound = 1.5
  let phaseWarpFrames = max(1, min(config.phaseWarpFrames, config.frames))
  let phaseWarpPointCount = max(2, config.phaseWarpPoints)
  let phaseWarpTensor = Tensor.param([phaseWarpPointCount], data: [Float](repeating: 0.0, count: phaseWarpPointCount))
  phaseWarpTensor.minBound = -1.0
  phaseWarpTensor.maxBound = 1.0
  let bodyWavePointCount = max(8, config.bodyWavePoints)
  let bodyWaveTensor = Tensor.param([bodyWavePointCount], data: sineWaveTable(size: bodyWavePointCount))
  bodyWaveTensor.minBound = -2.0
  bodyWaveTensor.maxBound = 2.0
  let noiseFilterSize = max(3, config.noiseFilterSize | 1)
  let noiseFilterInit = (0..<noiseFilterSize).map { i -> Float in
    let center = noiseFilterSize / 2
    if i == center { return 0.35 }
    let sign: Float = i % 2 == 0 ? 1.0 : -1.0
    let distance = Float(abs(i - center) + 1)
    return sign * 0.04 / distance
  }
  let noiseFilterTensor = Tensor.param([1, noiseFilterSize], data: noiseFilterInit)
  noiseFilterTensor.minBound = -1.5
  noiseFilterTensor.maxBound = 1.5
  let noiseEnvTensor = Tensor.param([config.envSize], data: targetNoiseEnv)
  noiseEnvTensor.minBound = 0.0
  noiseEnvTensor.maxBound = 2.0
  let airNoiseEnvTensor = Tensor.param([config.envSize], data: targetNoiseEnv)
  airNoiseEnvTensor.minBound = 0.0
  airNoiseEnvTensor.maxBound = 4.0
  let fmNoiseFrames = max(1, min(config.fmNoiseFrames, config.frames))
  let fmNoisePointCount = max(2, config.fmNoisePoints ?? fmNoiseFrames)
  let fmNoiseTensor = Tensor.param([fmNoisePointCount], data: deterministicNoiseTable(size: fmNoisePointCount))
  fmNoiseTensor.minBound = -2.0
  fmNoiseTensor.maxBound = 2.0
  let crispNoiseFrames = max(1, min(config.crispNoiseFrames, config.frames))
  let crispNoisePointCount = max(2, config.crispNoisePoints ?? crispNoiseFrames)
  let crispNoiseInit = crispExciterTable(config: config, size: crispNoisePointCount)
  let crispNoiseTensor = Tensor.param([crispNoisePointCount], data: crispNoiseInit)
  crispNoiseTensor.minBound = -2.0
  crispNoiseTensor.maxBound = 2.0
  let eqFilterSize = max(1, config.eqFilterSize | 1)
  let eqFilterInit = (0..<eqFilterSize).map { i -> Float in
    i == eqFilterSize - 1 ? 1.0 : 0.0
  }
  let eqFilterTensor = Tensor.param([1, eqFilterSize], data: eqFilterInit)
  eqFilterTensor.minBound = -3.0
  eqFilterTensor.maxBound = 3.0

  let startFreq = Signal.param(startFreqGuess, min: 25.0, max: 220.0)
  let endFreq = Signal.param(endFreqGuess, min: 25.0, max: 120.0)
  let freqLogDecay = Signal.param(log(0.9965), min: -0.03, max: -1e-6)
  let phase = Signal.param(initialPhase, min: -Float.pi, max: Float.pi)
  let phaseWarpAmp = Signal.param(config.phaseWarpAmpInit, min: 0.0, max: Float.pi)
  let bodyWaveMix = Signal.param(config.bodyWaveMixInit, min: 0.0, max: 1.0)

  let bodyAmp = Signal.param(0.85, min: 0.0, max: 2.0)
  let bodyLogDecay = Signal.param(log(0.99985), min: -0.01, max: -1e-6)
  let subAmp = Signal.param(0.2, min: 0.0, max: 1.5)
  let subLogDecay = Signal.param(log(0.9999), min: -0.01, max: -1e-6)

  let harmAmp = Signal.param(0.12, min: 0.0, max: 1.0)
  let harmLogDecay = Signal.param(log(0.996), min: -0.05, max: -1e-6)
  let clickAmp = Signal.param(0.25, min: 0.0, max: 1.8)
  let clickFreq = Signal.param(380.0, min: 80.0, max: 6000.0)
  let clickLogDecay = Signal.param(log(0.997), min: -0.2, max: -1e-6)
  let impulseAmp = Signal.param(0.12, min: -1.5, max: 1.5)
  let ringAmp = Signal.param(0.18, min: -1.5, max: 1.5)
  let ringFreq = Signal.param(max(startFreqGuess * 2.0, 110.0), min: 40.0, max: 1800.0)
  let noiseAmp = Signal.param(0.10, min: 0.0, max: 2.0)
  let noiseLogDecay = Signal.param(log(0.994), min: -0.4, max: -1e-6)
  let sizzleAmp = Signal.param(config.sizzleAmpInit, min: 0.0, max: 2.0)
  let airSizzleAmp = Signal.param(config.airSizzleAmpInit, min: 0.0, max: 2.0)
  let airNoiseAmp = Signal.param(config.airNoiseAmpInit, min: 0.0, max: 3.0)
  let airNoiseLogDecay = Signal.param(log(0.9985), min: -0.2, max: -1e-6)
  let fmNoiseAmp = Signal.param(config.fmNoiseAmpInit, min: 0.0, max: 3.0)
  let fmNoisePM = Signal.param(config.fmNoisePMInit, min: 0.0, max: 24.0)
  let fmNoiseLogDecay = Signal.param(log(0.9985), min: -0.2, max: -1e-6)
  let crispNoiseAmp = Signal.param(config.crispNoiseAmpInit, min: 0.0, max: 3.0)
  let crispNoiseLogDecay = Signal.param(log(0.996), min: -0.2, max: -1e-6)
  let presenceDuck = Signal.param(config.presenceDuckInit, min: 0.0, max: 1.5)
  let airModeAmp = Signal.param(config.airModeAmpInit, min: 0.0, max: 1.5)
  let airModeFreq = Signal.param(config.airModeFreqInit, min: 3_000.0, max: 16_000.0)
  let airModeLogDecay = Signal.param(log(0.9975), min: -0.2, max: -1e-6)
  let mode1Amp = Signal.param(config.transientModeAmpInit, min: -1.5, max: 1.5)
  let mode1Freq = Signal.param(260.0, min: 80.0, max: 900.0)
  let mode1LogDecay = Signal.param(log(0.9965), min: -0.2, max: -1e-6)
  let mode1Phase = Signal.param(0.0, min: -Float.pi, max: Float.pi)
  let mode2Amp = Signal.param(config.transientModeAmpInit, min: -1.5, max: 1.5)
  let mode2Freq = Signal.param(420.0, min: 100.0, max: 1_400.0)
  let mode2LogDecay = Signal.param(log(0.9955), min: -0.2, max: -1e-6)
  let mode2Phase = Signal.param(0.0, min: -Float.pi, max: Float.pi)
  let mode3Amp = Signal.param(config.transientModeAmpInit, min: -1.5, max: 1.5)
  let mode3Freq = Signal.param(760.0, min: 150.0, max: 2_400.0)
  let mode3LogDecay = Signal.param(log(0.994), min: -0.2, max: -1e-6)
  let mode3Phase = Signal.param(0.0, min: -Float.pi, max: Float.pi)
  let mode4Amp = Signal.param(config.transientModeAmpInit, min: -1.5, max: 1.5)
  let mode4Freq = Signal.param(1_250.0, min: 250.0, max: 4_000.0)
  let mode4LogDecay = Signal.param(log(0.992), min: -0.2, max: -1e-6)
  let mode4Phase = Signal.param(0.0, min: -Float.pi, max: Float.pi)
  let subBodyAmp = Signal.param(config.subBodyAmpInit, min: -2.0, max: 2.0)
  let drive = Signal.param(1.2, min: 0.3, max: 8.0)

  let params: [Signal] = [
    startFreq, endFreq, freqLogDecay, phase, phaseWarpAmp, bodyWaveMix,
    bodyAmp, bodyLogDecay,
    subAmp, subLogDecay,
    harmAmp, harmLogDecay,
    clickAmp, clickFreq, clickLogDecay, impulseAmp, ringAmp, ringFreq,
    noiseAmp, noiseLogDecay, sizzleAmp, airSizzleAmp, airNoiseAmp, airNoiseLogDecay,
    fmNoiseAmp, fmNoisePM, fmNoiseLogDecay, crispNoiseAmp, crispNoiseLogDecay, presenceDuck,
    airModeAmp, airModeFreq, airModeLogDecay, subBodyAmp,
    mode1Amp, mode1Freq, mode1LogDecay, mode1Phase,
    mode2Amp, mode2Freq, mode2LogDecay, mode2Phase,
    mode3Amp, mode3Freq, mode3LogDecay, mode3Phase,
    mode4Amp, mode4Freq, mode4LogDecay, mode4Phase,
    drive,
  ]
  let names = [
    "startFreq", "endFreq", "freqLogDecay", "phase", "phaseWarpAmp", "bodyWaveMix",
    "bodyAmp", "bodyLogDecay",
    "subAmp", "subLogDecay",
    "harmAmp", "harmLogDecay",
    "clickAmp", "clickFreq", "clickLogDecay", "impulseAmp", "ringAmp", "ringFreq",
    "noiseAmp", "noiseLogDecay", "sizzleAmp", "airSizzleAmp", "airNoiseAmp", "airNoiseLogDecay",
    "fmNoiseAmp", "fmNoisePM", "fmNoiseLogDecay", "crispNoiseAmp", "crispNoiseLogDecay", "presenceDuck",
    "airModeAmp", "airModeFreq", "airModeLogDecay", "subBodyAmp",
    "mode1Amp", "mode1Freq", "mode1LogDecay", "mode1Phase",
    "mode2Amp", "mode2Freq", "mode2LogDecay", "mode2Phase",
    "mode3Amp", "mode3Freq", "mode3LogDecay", "mode3Phase",
    "mode4Amp", "mode4Freq", "mode4LogDecay", "mode4Phase",
    "drive",
  ]
  let tensorParams = [
    bodyEnvTensor, subEnvTensor, harmEnvTensor, clickEnvTensor, freqCurveTensor, phaseWarpTensor, bodyWaveTensor,
    residualTensor,
    noiseFilterTensor, noiseEnvTensor, airNoiseEnvTensor, fmNoiseTensor, crispNoiseTensor,
    sizzleTensor, airSizzleTensor, subBodyTensor, eqFilterTensor,
  ]
  let tensorNames = [
    "bodyEnv", "subEnv", "harmEnv", "clickEnv", "freqCurve", "phaseWarp", "bodyWave", "residual", "noiseFilter",
    "noiseEnv", "airNoiseEnv", "fmNoise", "crispNoise", "sizzle", "airSizzle", "subBody", "eqFilter",
  ]

  if let checkpointIn = config.checkpointIn {
    try loadCheckpoint(
      path: checkpointIn,
      params: params,
      names: names,
      tensors: tensorParams,
      tensorNames: tensorNames)
    if config.resetSizzle {
      sizzleAmp.updateDataLazily(config.sizzleAmpInit)
      sizzleTensor.updateDataLazily(sizzleInit)
      print("checkpoint reset=sizzle")
    }
    if config.resetAirSizzle {
      airSizzleAmp.updateDataLazily(config.airSizzleAmpInit)
      airSizzleTensor.updateDataLazily(airSizzleInit)
      print("checkpoint reset=airSizzle")
    }
    if config.resetFMNoise {
      fmNoiseAmp.updateDataLazily(config.fmNoiseAmpInit)
      fmNoisePM.updateDataLazily(config.fmNoisePMInit)
      fmNoiseLogDecay.updateDataLazily(log(0.9985))
      fmNoiseTensor.updateDataLazily(deterministicNoiseTable(size: fmNoisePointCount))
      print("checkpoint reset=fmNoise")
    }
    if config.resetPhaseWarp {
      phaseWarpAmp.updateDataLazily(config.phaseWarpAmpInit)
      phaseWarpTensor.updateDataLazily([Float](repeating: 0.0, count: phaseWarpPointCount))
      print("checkpoint reset=phaseWarp")
    }
  }

  let freqOpt = Adam(
    params: [startFreq, endFreq, clickFreq, ringFreq],
    lr: 0.75 * config.learningRateScale * config.freqScalarLRScale)
  let ampOpt = Adam(
    params: [phase, bodyAmp, subAmp, harmAmp, clickAmp, impulseAmp, ringAmp, drive],
    lr: 0.015 * config.learningRateScale * config.ampLRScale)
  let decayOpt = Adam(
    params: [freqLogDecay, bodyLogDecay, subLogDecay, harmLogDecay, clickLogDecay],
    lr: 0.00002 * config.learningRateScale * config.decayLRScale)
  let envOpt = Adam(
    params: [bodyEnvTensor, subEnvTensor, harmEnvTensor, clickEnvTensor],
    lr: 0.002 * config.learningRateScale * config.envLRScale)
  let freqCurveOpt = Adam(
    params: [freqCurveTensor],
    lr: 0.7 * config.learningRateScale * config.freqCurveLRScale)
  let phaseWarpOpt = Adam(
    params: [phaseWarpAmp, phaseWarpTensor],
    lr: 0.02 * config.learningRateScale * config.phaseWarpLRScale)
  let bodyWaveOpt = Adam(
    params: [bodyWaveMix, bodyWaveTensor],
    lr: 0.015 * config.learningRateScale * config.bodyWaveLRScale)
  let residualOpt = Adam(
    params: [residualTensor],
    lr: 0.08 * config.learningRateScale * config.residualLRScale)
  let noiseOpt = Adam(
    params: [noiseAmp, noiseLogDecay, noiseFilterTensor, noiseEnvTensor],
    lr: 0.02 * config.learningRateScale * config.noiseLRScale)
  let airNoiseOpt = Adam(
    params: [airNoiseAmp, airNoiseLogDecay, airNoiseEnvTensor],
    lr: 0.02 * config.learningRateScale * config.airNoiseLRScale)
  let fmNoiseOpt = Adam(
    params: [fmNoiseAmp, fmNoisePM, fmNoiseLogDecay, fmNoiseTensor],
    lr: 0.02 * config.learningRateScale * config.fmNoiseLRScale)
  let crispNoiseOpt = Adam(
    params: [crispNoiseAmp, crispNoiseLogDecay, presenceDuck, crispNoiseTensor],
    lr: 0.02 * config.learningRateScale * config.crispNoiseLRScale)
  let airModeOpt = Adam(
    params: [airModeAmp, airModeFreq, airModeLogDecay],
    lr: 0.02 * config.learningRateScale * config.airModeLRScale)
  let transientModeOpt = Adam(
    params: [
      mode1Amp, mode1Freq, mode1LogDecay, mode1Phase,
      mode2Amp, mode2Freq, mode2LogDecay, mode2Phase,
      mode3Amp, mode3Freq, mode3LogDecay, mode3Phase,
      mode4Amp, mode4Freq, mode4LogDecay, mode4Phase,
    ],
    lr: 0.02 * config.learningRateScale * config.transientModeLRScale)
  let sizzleOpt = Adam(
    params: [sizzleAmp, sizzleTensor],
    lr: 0.05 * config.learningRateScale * config.sizzleLRScale)
  let airSizzleOpt = Adam(
    params: [airSizzleAmp, airSizzleTensor],
    lr: 0.05 * config.learningRateScale * config.airSizzleLRScale)
  let subBodyOpt = Adam(
    params: [subBodyAmp, subBodyTensor],
    lr: 0.04 * config.learningRateScale * config.subBodyLRScale)
  let eqOpt = Adam(
    params: [eqFilterTensor],
    lr: 0.02 * config.learningRateScale * config.eqLRScale)

  func buildSynth() -> Signal {
    let t = Signal.accum(
      Signal.constant(1.0),
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1)
    )
    let playhead = Signal.accum(
      Signal.constant(Float(config.envSize - 1) / Float(config.frames)),
      reset: 0.0,
      min: 0.0,
      max: Float(config.envSize - 1)
    )

    let freq = freqCurveTensor.peek(playhead)

    let bodyEnv = exp(bodyLogDecay * t)
    let phaseWarpIndex = (t * (Float(phaseWarpPointCount - 1) / Float(max(1, phaseWarpFrames - 1))))
      .clip(0.0, Double(phaseWarpPointCount - 1))
    let phaseWarpGate = t < Double(phaseWarpFrames - 1)
    let phaseWarp = gswitch(phaseWarpGate, phaseWarpTensor.peek(phaseWarpIndex) * phaseWarpAmp, 0.0)
    let bodyPhase01 = Signal.phasor(freq)
      + Signal.constant(8.0)
      + phase * (1.0 / (Float.pi * 2.0))
      + phaseWarp * (1.0 / (Float.pi * 2.0))
    let bodyPhaseWrapped = mod(bodyPhase01, 1.0)
    let bodyPhase = bodyPhaseWrapped * Float.pi * 2.0
    let learnedBodyWave = bodyWaveTensor.peek(bodyPhaseWrapped * Float(bodyWavePointCount - 1))
    let bodyOsc = sin(bodyPhase) * (1.0 - bodyWaveMix) + learnedBodyWave * bodyWaveMix
    let body = bodyOsc * bodyEnv * bodyEnvTensor.peek(playhead) * bodyAmp

    let subEnv = exp(subLogDecay * t)
    let sub = sin(Signal.phasor(freq * 0.5) * Float.pi * 2.0 + phase * 0.5)
      * subEnv * subEnvTensor.peek(playhead) * subAmp

    let harmEnv = exp(harmLogDecay * t)
    let harmonic = sin(Signal.phasor(freq * 2.0) * Float.pi * 2.0 + phase * 2.0)
      * harmEnv * harmEnvTensor.peek(playhead) * harmAmp

    let clickEnv = exp(clickLogDecay * t)
    let click = sin(Signal.phasor(clickFreq) * Float.pi * 2.0)
      * clickEnv * clickEnvTensor.peek(playhead) * clickAmp
    let impulse = Signal.click()
    let ring = impulse.biquad(cutoff: ringFreq, resonance: Signal.constant(1.8), gain: Signal.constant(1.0), mode: Signal.constant(2.0)) * ringAmp
    let noiseExcitation = Signal.noise()
    let filteredNoise = noiseExcitation.buffer(size: noiseFilterSize).conv2d(noiseFilterTensor).sum()
    let noiseEnv = exp(noiseLogDecay * t)
    let attackNoise = filteredNoise * noiseEnv * noiseEnvTensor.peek(playhead) * noiseAmp
    let airNoiseBand = Signal.noise()
      .biquad(cutoff: config.airNoiseLowHz, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: config.airNoiseHighHz, resonance: 0.707, gain: 1.0, mode: 0)
    let airNoiseEnv = exp(airNoiseLogDecay * t)
    let airNoiseDelayFrames = config.airNoiseDelayMs / 1000.0 * sampleRate
    let airNoiseFadeFrames = max(1.0, config.airNoiseFadeMs / 1000.0 * sampleRate)
    let airNoiseFade = ((t - airNoiseDelayFrames) / airNoiseFadeFrames).clip(0.0, 1.0)
    let airNoiseGate = t > Double(airNoiseDelayFrames)
    let airNoise = gswitch(
      airNoiseGate,
      airNoiseBand * airNoiseEnv * airNoiseEnvTensor.peek(playhead) * airNoiseAmp * airNoiseFade,
      0.0)
    let fmNoiseIndex = (t * (Float(fmNoisePointCount - 1) / Float(max(1, fmNoiseFrames - 1))))
      .clip(0.0, Double(fmNoisePointCount - 1))
    let fmNoiseDelayFrames = config.fmNoiseDelayMs / 1000.0 * sampleRate
    let fmNoiseFadeFrames = max(1.0, config.fmNoiseFadeMs / 1000.0 * sampleRate)
    let fmNoiseFade = ((t - fmNoiseDelayFrames) / fmNoiseFadeFrames).clip(0.0, 1.0)
    let fmNoiseGate = (t > Double(fmNoiseDelayFrames)) * (t < Double(fmNoiseFrames - 1))
    let fmNoiseEnv = exp(fmNoiseLogDecay * t)
    let fmNoiseValue = fmNoiseTensor.peek(fmNoiseIndex)
    let fmCarrier = sin(bodyPhase)
    let fmDirtyCarrier = sin(bodyPhase + fmNoiseValue * fmNoisePM)
    let fmNoiseRaw = (fmDirtyCarrier - fmCarrier) * fmNoiseEnv * fmNoiseAmp * fmNoiseFade
    let fmNoiseAir = gswitch(
      fmNoiseGate,
      fmNoiseRaw
        .biquad(cutoff: config.fmNoiseLowHz, resonance: 0.707, gain: 1.0, mode: 1)
        .biquad(cutoff: config.fmNoiseHighHz, resonance: 0.707, gain: 1.0, mode: 0),
      0.0)
    let crispNoiseIndex = (t * (Float(crispNoisePointCount - 1) / Float(max(1, crispNoiseFrames - 1))))
      .clip(0.0, Double(crispNoisePointCount - 1))
    let crispNoiseDelayFrames = config.crispNoiseDelayMs / 1000.0 * sampleRate
    let crispNoiseFadeFrames = max(1.0, config.crispNoiseFadeMs / 1000.0 * sampleRate)
    let crispNoiseFade = ((t - crispNoiseDelayFrames) / crispNoiseFadeFrames).clip(0.0, 1.0)
    let crispNoiseGate = (t > Double(crispNoiseDelayFrames)) * (t < Double(crispNoiseFrames - 1))
    let crispNoiseEnv = exp(crispNoiseLogDecay * t)
    let crispNoiseRaw = crispNoiseTensor.peek(crispNoiseIndex) * crispNoiseEnv * crispNoiseAmp * crispNoiseFade
    var crispFiltered = crispNoiseRaw
    for _ in 0..<max(1, config.crispNoiseHPStages) {
      crispFiltered = crispFiltered.biquad(cutoff: config.crispNoiseLowHz, resonance: 0.707, gain: 1.0, mode: 1)
    }
    crispFiltered = crispFiltered.biquad(cutoff: config.crispNoiseHighHz, resonance: 0.707, gain: 1.0, mode: 0)
    let crispNoise = gswitch(
      crispNoiseGate,
      crispFiltered,
      0.0)
    let airModeDelayFrames = config.airModeDelayMs / 1000.0 * sampleRate
    let airModeFadeFrames = max(1.0, config.airModeFadeMs / 1000.0 * sampleRate)
    let airModeFade = ((t - airModeDelayFrames) / airModeFadeFrames).clip(0.0, 1.0)
    let airModeGate = t > Double(airModeDelayFrames)
    let airModeAge = (t - airModeDelayFrames).clip(0.0, Double(config.frames))
    let airModeEnv = exp(airModeLogDecay * airModeAge) * airModeFade
    let airModePhase = Signal.phasor(airModeFreq) * Float.pi * 2.0
    let airModeRaw = (
      sin(airModePhase)
        + sin(airModePhase * 1.47 + phase * 0.25) * 0.5
        + sin(airModePhase * 2.11 + phase * 0.5) * 0.25
    ) * airModeEnv * airModeAmp
    let airMode = gswitch(
      airModeGate,
      airModeRaw
        .biquad(cutoff: config.airModeHighpassHz, resonance: 0.707, gain: 1.0, mode: 1)
        .biquad(cutoff: config.airModeLowpassHz, resonance: 0.707, gain: 1.0, mode: 0),
      0.0)
    let transientModeFrames = config.transientModeMs / 1000.0 * sampleRate
    let transientModeGate = t < Double(transientModeFrames)
    let transientModeFade = (1.0 - (t / Swift.max(1.0, transientModeFrames))).clip(0.0, 1.0)
    let transientMode: Signal
    if config.transientModeFollowBody {
      transientMode =
        (
          sin(bodyPhase + mode1Phase) * exp(mode1LogDecay * t) * mode1Amp
            + sin(bodyPhase * 1.5 + mode2Phase) * exp(mode2LogDecay * t) * mode2Amp
            + sin(bodyPhase * 2.25 + mode3Phase) * exp(mode3LogDecay * t) * mode3Amp
            + sin(bodyPhase * 3.5 + mode4Phase) * exp(mode4LogDecay * t) * mode4Amp
        ) * transientModeFade
    } else {
      transientMode =
        (
          sin(Signal.phasor(mode1Freq) * Float.pi * 2.0 + mode1Phase) * exp(mode1LogDecay * t) * mode1Amp
            + sin(Signal.phasor(mode2Freq) * Float.pi * 2.0 + mode2Phase) * exp(mode2LogDecay * t) * mode2Amp
            + sin(Signal.phasor(mode3Freq) * Float.pi * 2.0 + mode3Phase) * exp(mode3LogDecay * t) * mode3Amp
            + sin(Signal.phasor(mode4Freq) * Float.pi * 2.0 + mode4Phase) * exp(mode4LogDecay * t) * mode4Amp
        ) * transientModeFade
    }
    let transientModeLayer = gswitch(transientModeGate, transientMode, 0.0)
    let residualIndex = (t * (Float(residualPointCount - 1) / Float(max(1, residualFrames - 1))))
      .clip(0.0, Double(residualPointCount - 1))
    let residualGate = t < Double(residualFrames - 1)
    let residual = gswitch(residualGate, residualTensor.peek(residualIndex), 0.0)
    let sizzleIndex = (t * (Float(sizzlePointCount - 1) / Float(max(1, sizzleFrames - 1))))
      .clip(0.0, Double(sizzlePointCount - 1))
    let sizzleDelayFrames = config.sizzleDelayMs / 1000.0 * sampleRate
    let sizzleFadeFrames = max(1.0, config.sizzleFadeMs / 1000.0 * sampleRate)
    let sizzleFade = ((t - sizzleDelayFrames) / sizzleFadeFrames).clip(0.0, 1.0)
    let sizzleGate = (t > Double(sizzleDelayFrames)) * (t < Double(sizzleFrames - 1))
    let sizzle = gswitch(sizzleGate, sizzleTensor.peek(sizzleIndex) * sizzleAmp * sizzleFade, 0.0)
    let airSizzleIndex = (t * (Float(airSizzlePointCount - 1) / Float(max(1, airSizzleFrames - 1))))
      .clip(0.0, Double(airSizzlePointCount - 1))
    let airSizzleDelayFrames = config.airSizzleDelayMs / 1000.0 * sampleRate
    let airSizzleFadeFrames = max(1.0, config.airSizzleFadeMs / 1000.0 * sampleRate)
    let airSizzleFade = ((t - airSizzleDelayFrames) / airSizzleFadeFrames).clip(0.0, 1.0)
    let airSizzleGate = (t > Double(airSizzleDelayFrames)) * (t < Double(airSizzleFrames - 1))
    let airSizzle = gswitch(
      airSizzleGate,
      airSizzleTensor.peek(airSizzleIndex) * airSizzleAmp * airSizzleFade,
      0.0)
    let subBodyIndex = (t * (Float(subBodyPointCount - 1) / Float(max(1, subBodyFrames - 1))))
      .clip(0.0, Double(subBodyPointCount - 1))
    let subBodyGate = t < Double(subBodyFrames - 1)
    let subBodyLayer = gswitch(
      subBodyGate,
      subBodyTensor.peek(subBodyIndex) * subBodyAmp,
      0.0)

    let postCleanBody = config.cleanBodyPostDrive ? body * config.cleanBodyScale : Signal.constant(0.0)
    let postCleanSub = config.cleanSubPostDrive ? sub * config.cleanBodyScale : Signal.constant(0.0)
    let drivenBody = config.cleanBodyPostDrive ? Signal.constant(0.0) : body
    let drivenSub = config.cleanSubPostDrive ? Signal.constant(0.0) : sub
    let preDriveSub = config.subBodyPostDrive ? Signal.constant(0.0) : subBodyLayer
    let preDriveAir = config.airSizzlePostDrive ? Signal.constant(0.0) : airSizzle
    let preDriveMode = config.transientModePostDrive ? Signal.constant(0.0) : transientModeLayer
    let driven = tanh((drivenBody + drivenSub + harmonic + click + impulse * impulseAmp + ring + attackNoise + residual + preDriveSub + preDriveAir + preDriveMode) * drive)
    let postDriveBody = postCleanBody + postCleanSub
    let postDriveSub = config.subBodyPostDrive ? subBodyLayer : Signal.constant(0.0)
    let postDriveAir = config.airSizzlePostDrive ? airSizzle : Signal.constant(0.0)
    let postDriveMode = config.transientModePostDrive ? transientModeLayer : Signal.constant(0.0)
    let mixed: Signal
    if config.sizzlePostDrive {
      mixed = driven + postDriveBody + sizzle + postDriveSub + postDriveAir + postDriveMode + airNoise + fmNoiseAir + crispNoise + airMode
    } else {
      mixed = tanh((drivenBody + drivenSub + harmonic + click + impulse * impulseAmp + ring + attackNoise + residual + sizzle + preDriveSub + preDriveAir + preDriveMode) * drive) + postDriveBody + postDriveSub + postDriveAir + postDriveMode + airNoise + fmNoiseAir + crispNoise + airMode
    }
    let presenceBand = mixed
      .biquad(cutoff: config.presenceDuckLowHz, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: config.presenceDuckHighHz, resonance: 0.707, gain: 1.0, mode: 0)
    let tuned = mixed - presenceBand * presenceDuck
    if eqFilterSize <= 1 {
      return tuned
    }
    return tuned.buffer(size: eqFilterSize).conv2d(eqFilterTensor).sum()
  }

  func buildTarget() -> Signal {
    targetTensor.toSignal(maxFrames: config.frames)
  }

  let targetOut = URL(fileURLWithPath: config.outputDir).appendingPathComponent("target.wav")
  try AudioFile.save(url: targetOut, samples: targetSamples, sampleRate: sampleRate)

  print("target=\(targetURL.path)")
  print(
    "frames=\(config.frames) sampleRate=\(sampleRate) durationMs=\(String(format: "%.1f", Float(config.frames) / sampleRate * 1000)) window=\(config.windowSize) epochs=\(config.epochs) waveformWeight=\(config.waveformWeight) transientWeight=\(config.transientWeight) slopeWeight=\(config.slopeWeight) highBandWeight=\(config.highBandWeight) highBandSpectralWeight=\(config.highBandSpectralWeight) subBandWeight=\(config.subBandWeight) presenceBandWeight=\(config.presenceBandWeight) airBandWeight=\(config.airBandWeight) envSize=\(config.envSize) residualFrames=\(residualFrames) residualPoints=\(residualPointCount) bodyWavePoints=\(bodyWavePointCount) bodyWaveMixInit=\(config.bodyWaveMixInit) noiseFilterSize=\(noiseFilterSize) sizzleFrames=\(sizzleFrames) sizzlePoints=\(sizzlePointCount) sizzleInit=\(config.sizzleInit) sizzleBand=\(config.sizzleLowHz)-\(config.sizzleHighHz)Hz sizzlePostDrive=\(config.sizzlePostDrive) airSizzleFrames=\(airSizzleFrames) airSizzlePoints=\(airSizzlePointCount) airSizzleInit=\(config.airSizzleInit) airSizzleBand=\(config.airSizzleLowHz)-\(config.airSizzleHighHz)Hz airSizzlePostDrive=\(config.airSizzlePostDrive) airNoiseAmpInit=\(config.airNoiseAmpInit) airNoiseBand=\(config.airNoiseLowHz)-\(config.airNoiseHighHz)Hz airNoiseDelayMs=\(config.airNoiseDelayMs) airNoiseFadeMs=\(config.airNoiseFadeMs) fmNoiseFrames=\(fmNoiseFrames) fmNoisePoints=\(fmNoisePointCount) fmNoiseAmpInit=\(config.fmNoiseAmpInit) fmNoisePMInit=\(config.fmNoisePMInit) fmNoiseBand=\(config.fmNoiseLowHz)-\(config.fmNoiseHighHz)Hz fmNoiseDelayMs=\(config.fmNoiseDelayMs) fmNoiseFadeMs=\(config.fmNoiseFadeMs) subBodyFrames=\(subBodyFrames) subBodyPoints=\(subBodyPointCount) subBodyInit=\(config.subBodyInit) subBodyHz=\(config.subBodyHz) subBodyPostDrive=\(config.subBodyPostDrive) cleanBodyPostDrive=\(config.cleanBodyPostDrive) cleanSubPostDrive=\(config.cleanSubPostDrive) cleanBodyScale=\(config.cleanBodyScale) eqFilterSize=\(eqFilterSize)"
  )
  print(
    "phaseWarp frames=\(phaseWarpFrames) points=\(phaseWarpPointCount) ampInit=\(config.phaseWarpAmpInit) lrScale=\(config.phaseWarpLRScale)"
  )
  print(
    "lrScales freqScalar=\(config.freqScalarLRScale) amp=\(config.ampLRScale) decay=\(config.decayLRScale) env=\(config.envLRScale) freqCurve=\(config.freqCurveLRScale) phaseWarp=\(config.phaseWarpLRScale) bodyWave=\(config.bodyWaveLRScale) residual=\(config.residualLRScale) noise=\(config.noiseLRScale) airNoise=\(config.airNoiseLRScale) fmNoise=\(config.fmNoiseLRScale) transientMode=\(config.transientModeLRScale) sizzle=\(config.sizzleLRScale) airSizzle=\(config.airSizzleLRScale) subBody=\(config.subBodyLRScale) eq=\(config.eqLRScale)"
  )
  print(
    "crispNoise frames=\(crispNoiseFrames) points=\(crispNoisePointCount) init=\(config.crispNoiseInit) ampInit=\(config.crispNoiseAmpInit) spacing=\(config.crispNoiseSpacing) burstDecay=\(config.crispNoiseBurstDecay) band=\(config.crispNoiseLowHz)-\(config.crispNoiseHighHz)Hz hpStages=\(config.crispNoiseHPStages) delayMs=\(config.crispNoiseDelayMs) fadeMs=\(config.crispNoiseFadeMs)"
  )
  print(
    "airMode ampInit=\(config.airModeAmpInit) freqInit=\(config.airModeFreqInit) delayMs=\(config.airModeDelayMs) fadeMs=\(config.airModeFadeMs) band=\(config.airModeHighpassHz)-\(config.airModeLowpassHz)Hz lrScale=\(config.airModeLRScale)"
  )
  print(
    "transientMode ampInit=\(config.transientModeAmpInit) ms=\(config.transientModeMs) followBody=\(config.transientModeFollowBody) postDrive=\(config.transientModePostDrive) lrScale=\(config.transientModeLRScale)"
  )
  print(
    "selection every=\(config.selectionEvery) checkpointSelection=\(config.checkpointSelection) sustainedWeights sub=\(config.sustainedSubWeight) presence=\(config.sustainedPresenceWeight) air=\(config.sustainedAirWeight) hf=\(config.sustainedHFWeight)"
  )
  print(
    "analysis onsetFrame=\(onset) startFreqGuess=\(String(format: "%.2f", startFreqGuess))Hz endFreqGuess=\(String(format: "%.2f", endFreqGuess))Hz"
  )
  if !analyzedPitches.isEmpty {
    let formatted = analyzedPitches
      .map { "\(String(format: "%.0f", $0.ms))ms:\(String(format: "%.1f", $0.hz))Hz" }
      .joined(separator: " ")
    print("pitchTrack \(formatted)")
  }

  if config.renderOnly {
    let learned = try buildSynth().realize(frames: config.frames)
    let learnedOut = URL(fileURLWithPath: config.outputDir).appendingPathComponent("learned.wav")
    try AudioFile.save(url: learnedOut, samples: learned, sampleRate: sampleRate)
    print("renderOnly=true")
    print("wrote=\(targetOut.path)")
    print("wrote=\(learnedOut.path)")
    return
  }

  let (_, warmupMs) = try timed {
    let synth = buildSynth()
    let target = buildTarget()
    let loss = combinedLoss(synth: synth, target: target)
    _ = try loss.backward(frames: config.frames)
    freqOpt.zeroGrad()
    ampOpt.zeroGrad()
    decayOpt.zeroGrad()
    envOpt.zeroGrad()
    freqCurveOpt.zeroGrad()
    phaseWarpOpt.zeroGrad()
    bodyWaveOpt.zeroGrad()
    residualOpt.zeroGrad()
    noiseOpt.zeroGrad()
    airNoiseOpt.zeroGrad()
    fmNoiseOpt.zeroGrad()
    crispNoiseOpt.zeroGrad()
    airModeOpt.zeroGrad()
    transientModeOpt.zeroGrad()
    sizzleOpt.zeroGrad()
    airSizzleOpt.zeroGrad()
    subBodyOpt.zeroGrad()
    eqOpt.zeroGrad()
  }
  print("warmupMs=\(String(format: "%.2f", warmupMs))")

  func combinedLoss(synth: Signal, target: Signal) -> Signal {
    let spectral = spectralLossFFT(
      synth, target,
      windowSize: config.windowSize,
      useHannWindow: true,
      hop: max(1, config.windowSize / 4),
      normalize: true)
    let perFrameMSE = mse(synth, target)
    let frameScale = 1.0 / Float(config.frames)
    let t = Signal.accum(
      Signal.constant(1.0),
      reset: 0.0,
      min: 0.0,
      max: Float(config.frames + 1)
    )
    let transientWeight = exp(Signal.constant(log(0.996)) * t)
    let synthHistory = Signal.history()
    let targetHistory = Signal.history()
    let synthDelta = synthHistory.write(synth) - synthHistory.read
    let targetDelta = targetHistory.write(target) - targetHistory.read
    let slopeMSE = mse(synthDelta, targetDelta)
    let slopeWindow = exp(Signal.constant(log(0.992)) * t)
    let hpSynth = synth.biquad(cutoff: 2_500.0, resonance: 0.707, gain: 1.0, mode: 1)
    let hpTarget = target.biquad(cutoff: 2_500.0, resonance: 0.707, gain: 1.0, mode: 1)
    let synthHighBandEnv = (hpSynth * hpSynth).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0)
    let targetHighBandEnv = (hpTarget * hpTarget).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0)
    let highBandEnergyMSE = mse(synthHighBandEnv, targetHighBandEnv)
    let subSynth = synth.biquad(cutoff: 200.0, resonance: 0.707, gain: 1.0, mode: 0)
    let subTarget = target.biquad(cutoff: 200.0, resonance: 0.707, gain: 1.0, mode: 0)
    let subMSE = mse(
      (subSynth * subSynth).biquad(cutoff: 400.0, resonance: 0.707, gain: 1.0, mode: 0),
      (subTarget * subTarget).biquad(cutoff: 400.0, resonance: 0.707, gain: 1.0, mode: 0))
    let presenceSynth = synth.biquad(cutoff: 1_000.0, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: 4_000.0, resonance: 0.707, gain: 1.0, mode: 0)
    let presenceTarget = target.biquad(cutoff: 1_000.0, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: 4_000.0, resonance: 0.707, gain: 1.0, mode: 0)
    let presenceMSE = mse(
      (presenceSynth * presenceSynth).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0),
      (presenceTarget * presenceTarget).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0))
    let airSynth = synth.biquad(cutoff: 4_000.0, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: 12_000.0, resonance: 0.707, gain: 1.0, mode: 0)
    let airTarget = target.biquad(cutoff: 4_000.0, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: 12_000.0, resonance: 0.707, gain: 1.0, mode: 0)
    let airMSE = mse(
      (airSynth * airSynth).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0),
      (airTarget * airTarget).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0))
    let highBandStart = Float(sampleRate * 0.005)
    let highBandEnd = Float(sampleRate * 0.032)
    let highBandGate = (t > Double(highBandStart)) * (t < Double(highBandEnd))
    let highBandAge = (t - highBandStart).clip(0.0, Double(config.frames))
    let highBandWindow = highBandGate * exp(Signal.constant(log(0.995)) * highBandAge)
    let sustainedStart = Float(sampleRate * 0.005)
    let sustainedEnd = Float(sampleRate * 0.030)
    let sustainedGate = (t > Double(sustainedStart)) * (t < Double(sustainedEnd))
    let sustainedAge = (t - sustainedStart).clip(0.0, Double(config.frames))
    let sustainedWindow = sustainedGate * exp(Signal.constant(log(0.998)) * sustainedAge)
    let hfSynth = synth.biquad(cutoff: 2_000.0, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: 16_000.0, resonance: 0.707, gain: 1.0, mode: 0)
    let hfTarget = target.biquad(cutoff: 2_000.0, resonance: 0.707, gain: 1.0, mode: 1)
      .biquad(cutoff: 16_000.0, resonance: 0.707, gain: 1.0, mode: 0)
    let hfMSE = mse(
      (hfSynth * hfSynth).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0),
      (hfTarget * hfTarget).biquad(cutoff: 1_200.0, resonance: 0.707, gain: 1.0, mode: 0))
    let highBandSpectral = spectralLossFFT(
      hpSynth, hpTarget,
      windowSize: min(256, config.windowSize),
      useHannWindow: true,
      hop: 64,
      normalize: true)
    return spectral
      + perFrameMSE * (config.waveformWeight * frameScale)
      + perFrameMSE * transientWeight * (config.transientWeight * frameScale)
      + slopeMSE * slopeWindow * (config.slopeWeight * frameScale)
      + highBandEnergyMSE * highBandWindow * (config.highBandWeight * frameScale)
      + highBandSpectral * config.highBandSpectralWeight
      + subMSE * highBandWindow * (config.subBandWeight * frameScale)
      + presenceMSE * highBandWindow * (config.presenceBandWeight * frameScale)
      + airMSE * highBandWindow * (config.airBandWeight * frameScale)
      + subMSE * sustainedWindow * (config.sustainedSubWeight * frameScale)
      + presenceMSE * sustainedWindow * (config.sustainedPresenceWeight * frameScale)
      + airMSE * sustainedWindow * (config.sustainedAirWeight * frameScale)
      + hfMSE * sustainedWindow * (config.sustainedHFWeight * frameScale)
  }

  var firstLoss: Float = 0
  var lastLoss: Float = 0
  var bestLoss: Float = .infinity
  var bestEpoch = 0
  var bestValues: [Float] = params.map { $0.data ?? 0.0 }
  var bestTensorValues: [[Float]] = tensorParams.map { $0.getData() ?? [] }
  var epochTimes: [Double] = []

  for epoch in 0..<config.epochs {
    let ((lossValues), epochMs) = try timed {
      let loss = combinedLoss(synth: buildSynth(), target: buildTarget())
      return try loss.backward(frames: config.frames)
    }
    epochTimes.append(epochMs)
    let epochLoss = lossValues.reduce(0, +)
    if epoch == 0 { firstLoss = epochLoss }
    lastLoss = epochLoss
    if epochLoss < bestLoss {
      bestLoss = epochLoss
      bestEpoch = epoch
      bestValues = params.map { $0.data ?? 0.0 }
      bestTensorValues = tensorParams.map { $0.getData() ?? [] }
    }
    let shouldScore = config.selectionEvery > 0
      && (epoch % config.selectionEvery == 0 || epoch == config.epochs - 1)
    if shouldScore {
      let candidatePath = URL(fileURLWithPath: config.outputDir)
        .appendingPathComponent("candidates")
        .appendingPathComponent(String(format: "epoch-%04d.json", epoch))
        .path
      try saveCheckpoint(
        path: candidatePath,
        config: config,
        params: params,
        names: names,
        tensors: tensorParams,
        tensorNames: tensorNames)
      print("perceptualCandidate epoch=\(epoch) checkpoint=\(candidatePath)")
    }

    if epoch % 5 == 0 || epoch == config.epochs - 1 {
      print(
        "epoch=\(epoch) loss=\(String(format: "%.6f", epochLoss)) stepMs=\(String(format: "%.2f", epochMs))"
      )
      for (name, param) in zip(names, params) {
        let value = param.data ?? .nan
        let grad = param.grad?.data ?? .nan
        print(
          "  \(name)=\(String(format: "%12.6f", value)) grad=\(String(format: "%12.4e", grad))"
        )
      }
      for (name, tensor) in zip(tensorNames, tensorParams) {
        let data = tensor.getData() ?? []
        let grad = tensor.grad?.getData() ?? []
        let gMax = grad.map { abs($0) }.max() ?? 0
        print(
          "  \(name): max=\(String(format: "%8.4f", data.max() ?? 0)) gradMax=\(String(format: "%10.4e", gMax))"
        )
      }
    }

    if epochLoss.isNaN || epochLoss.isInfinite {
      print("stopping: non-finite loss")
      break
    }

    freqOpt.step()
    ampOpt.step()
    decayOpt.step()
    envOpt.step()
    freqCurveOpt.step()
    phaseWarpOpt.step()
    bodyWaveOpt.step()
    residualOpt.step()
    noiseOpt.step()
    airNoiseOpt.step()
    fmNoiseOpt.step()
    crispNoiseOpt.step()
    airModeOpt.step()
    transientModeOpt.step()
    sizzleOpt.step()
    airSizzleOpt.step()
    subBodyOpt.step()
    eqOpt.step()
    freqOpt.zeroGrad()
    ampOpt.zeroGrad()
    decayOpt.zeroGrad()
    envOpt.zeroGrad()
    freqCurveOpt.zeroGrad()
    phaseWarpOpt.zeroGrad()
    bodyWaveOpt.zeroGrad()
    residualOpt.zeroGrad()
    noiseOpt.zeroGrad()
    airNoiseOpt.zeroGrad()
    fmNoiseOpt.zeroGrad()
    crispNoiseOpt.zeroGrad()
    airModeOpt.zeroGrad()
    transientModeOpt.zeroGrad()
    sizzleOpt.zeroGrad()
    airSizzleOpt.zeroGrad()
    subBodyOpt.zeroGrad()
    eqOpt.zeroGrad()
  }

  if config.selectionEvery > 0 {
    print(
      "perceptualSelection note=candidate checkpoints written; render candidates in separate TrainKick808 invocations and rank with waveform_compare.py selection_score"
    )
  }

  if config.checkpointSelection == "perceptual" {
    print(
      "checkpointSelection warning=perceptual requires external ranking of candidate checkpoints; writing best loss checkpoint"
    )
  }

  for (param, value) in zip(params, bestValues) {
    param.updateDataLazily(value)
  }
  for (tensor, values) in zip(tensorParams, bestTensorValues) {
    tensor.updateDataLazily(values)
  }
  let checkpointOut = config.checkpointOut
    ?? URL(fileURLWithPath: config.outputDir).appendingPathComponent("checkpoint.json").path
  try saveCheckpoint(
    path: checkpointOut,
    config: config,
    params: params,
    names: names,
    tensors: tensorParams,
    tensorNames: tensorNames)

  let learned = try buildSynth().realize(frames: config.frames)
  let learnedOut = URL(fileURLWithPath: config.outputDir).appendingPathComponent("learned.wav")
  try AudioFile.save(url: learnedOut, samples: learned, sampleRate: sampleRate)

  let avgMs = epochTimes.reduce(0, +) / Double(max(1, epochTimes.count))
  print("firstLoss=\(firstLoss) finalLoss=\(lastLoss)")
  print("bestLoss=\(bestLoss) bestEpoch=\(bestEpoch)")
  print("selectedCheckpoint=loss")
  print("bestReduction=\(String(format: "%.2fx", firstLoss / max(bestLoss, 1e-12)))")
  print("avgStepMs=\(String(format: "%.2f", avgMs))")
  print("wrote=\(targetOut.path)")
  print("wrote=\(learnedOut.path)")
}

do {
  try run()
} catch {
  fputs("TrainKick808 error: \(error)\n", stderr)
  exit(1)
}

import Foundation

do {
  try DDSPE2EMain.run()
} catch {
  fputs("error: \(error)\n", stderr)
  exit(1)
}

struct DDSPE2EMain {
  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard let command = args.first else {
      printHelp()
      return
    }

    if command == "--help" || command == "-h" || command == "help" {
      printHelp()
      return
    }

    let options = parseOptions(Array(args.dropFirst()))

    switch command {
    case "dump-config":
      try handleDumpConfig(options)
    case "preprocess":
      try handlePreprocess(options)
    case "inspect-cache":
      try handleInspectCache(options)
    case "train":
      try handleTrain(options)
    case "probe-smoothing":
      try handleProbeSmoothing(options)
    default:
      throw CLIError.invalid("Unknown command: \(command)")
    }
  }

  private static func handleDumpConfig(_ options: [String: String]) throws {
    let output = options["output"] ?? "ddsp_config.json"
    try DDSPE2EConfig.default.write(to: URL(fileURLWithPath: output))
    print("Wrote default config to \(output)")
  }

  private static func handlePreprocess(_ options: [String: String]) throws {
    guard let input = options["input"] else {
      throw CLIError.invalid("preprocess requires --input <wav-dir>")
    }
    guard let cache = options["cache"] else {
      throw CLIError.invalid("preprocess requires --cache <cache-dir>")
    }

    var config = try DDSPE2EConfig.load(path: options["config"])
    try config.applyCLIOverrides(options)

    let inputURL = URL(fileURLWithPath: input)
    let cacheURL = URL(fileURLWithPath: cache)

    let manifest = try DatasetPreprocessor.preprocess(
      inputRoot: inputURL,
      cacheRoot: cacheURL,
      config: config,
      logger: log
    )

    print(
      "Done. total=\(manifest.chunkCount) train=\(manifest.trainCount) val=\(manifest.valCount)"
    )
  }

  private static func handleInspectCache(_ options: [String: String]) throws {
    guard let cache = options["cache"] else {
      throw CLIError.invalid("inspect-cache requires --cache <cache-dir>")
    }

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))
    let split = parseSplit(options["split"])
    let limit = Int(options["limit"] ?? "5") ?? 5

    print("cache: \(cache)")
    print("chunks: total=\(dataset.manifest.chunkCount) train=\(dataset.manifest.trainCount) val=\(dataset.manifest.valCount)")

    let entries = dataset.entries(for: split)
    print("showing \(min(limit, entries.count)) / \(entries.count) entries (split=\(split?.rawValue ?? "all"))")

    for entry in entries.prefix(limit) {
      let chunk = try dataset.loadChunk(entry)
      let voiced = zip(chunk.f0Hz, chunk.uvMask).filter { $0.1 > 0.5 }.map { $0.0 }
      let f0Mean = voiced.isEmpty ? 0 : voiced.reduce(0, +) / Float(voiced.count)
      let rms = sqrt(chunk.audio.map { $0 * $0 }.reduce(0, +) / Float(max(1, chunk.audio.count)))
      print(
        "- \(entry.id) split=\(entry.split.rawValue) samples=\(chunk.audio.count) frames=\(chunk.f0Hz.count) rms=\(fmt(rms)) voicedF0Mean=\(fmt(f0Mean)) source=\(entry.sourceFile)"
      )
    }
  }

  private static func handleTrain(_ options: [String: String]) throws {
    guard let cache = options["cache"] else {
      throw CLIError.invalid("train requires --cache <cache-dir>")
    }

    var config = try DDSPE2EConfig.load(path: options["config"])
    try config.applyCLIOverrides(options)

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))

    let runsBase = URL(fileURLWithPath: options["runs-dir"] ?? "runs")
    let runName = options["run-name"]
    let steps = Int(options["steps"] ?? "200") ?? 200
    let split = parseSplit(options["split"]) ?? .train
    let mode = TrainMode(rawValue: (options["mode"] ?? "m2").lowercased()) ?? .m2
    let profileStep = Int(options["profile-step"] ?? "-1") ?? -1
    let renderEvery = Int(options["render-every"] ?? "0") ?? 0
    let renderWavPath = options["render-wav"]
    let dumpControlsEvery = Int(options["dump-controls-every"] ?? "0") ?? 0
    let initCheckpointPath = options["init-checkpoint"]
    let rawKernelDump = options["kernel-dump"]

    if parseBoolOption(options["auto-abc"], defaultValue: false) {
      try runAutoABCTraining(
        dataset: dataset,
        baseConfig: config,
        runsBase: runsBase,
        runName: runName,
        split: split,
        mode: mode,
        defaultSteps: steps,
        profileStep: profileStep,
        renderEvery: renderEvery,
        renderWavPath: renderWavPath,
        dumpControlsEvery: dumpControlsEvery,
        rawKernelDumpValue: rawKernelDump,
        initialCheckpointPath: initCheckpointPath,
        options: options
      )
      return
    }

    let runDirs = try RunDirectories.create(base: runsBase, runName: runName)
    let kernelDumpPath = resolveKernelDumpPath(rawValue: rawKernelDump, runDir: runDirs.root)

    try DDSPE2ETrainer.run(
      dataset: dataset,
      config: config,
      runDirs: runDirs,
      options: TrainerOptions(
        steps: steps,
        split: split,
        mode: mode,
        kernelDumpPath: kernelDumpPath,
        initCheckpointPath: initCheckpointPath,
        profileKernelsStep: profileStep,
        renderEvery: renderEvery,
        renderWavPath: renderWavPath,
        dumpControlsEvery: dumpControlsEvery
      ),
      logger: log
    )
  }

  private struct TrainSummaryValues {
    let steps: Int
    let minStep: Int
    let minLoss: Float
    let finalLoss: Float
  }

  private struct AutoABCStageResult: Codable {
    let stage: String
    let runName: String
    let runDir: String
    let bestCheckpoint: String
    let steps: Int
    let minStep: Int
    let minLoss: Float
    let finalLoss: Float
  }

  private struct AutoABCSummary: Codable {
    let createdAtUTC: String
    let runPrefix: String
    let initialCheckpoint: String?
    let split: String
    let stages: [AutoABCStageResult]
    let bestCheckpoint: String
    let bestLoss: Float
  }

  private enum AutoABCPreset: String {
    case baseline
    case bestLowLoss = "best-low-loss"
  }

  private static func runAutoABCTraining(
    dataset: CachedDataset,
    baseConfig: DDSPE2EConfig,
    runsBase: URL,
    runName: String?,
    split: DatasetSplit,
    mode: TrainMode,
    defaultSteps: Int,
    profileStep: Int,
    renderEvery: Int,
    renderWavPath: String?,
    dumpControlsEvery: Int,
    rawKernelDumpValue: String?,
    initialCheckpointPath: String?,
    options: [String: String]
  ) throws {
    guard mode == .m2 else {
      throw CLIError.invalid("--auto-abc currently supports only --mode m2")
    }

    let preset = try parseAutoABCPreset(options["auto-abc-preset"])
    let presetBaseConfig = applyAutoABCPreset(preset, to: baseConfig)

    let prefix = runName ?? "autoabc_\(timestampString())"
    let stageARun = "\(prefix)_stageA"
    let stageBRun = "\(prefix)_stageB"
    let stageCRun = "\(prefix)_stageC"

    let stepsA = try parsePositiveIntOption(
      options["auto-abc-steps-a"],
      key: "auto-abc-steps-a",
      defaultValue: max(1, defaultSteps)
    )
    let stepsB = try parsePositiveIntOption(
      options["auto-abc-steps-b"],
      key: "auto-abc-steps-b",
      defaultValue: 300
    )
    let stepsC = try parsePositiveIntOption(
      options["auto-abc-steps-c"],
      key: "auto-abc-steps-c",
      defaultValue: 300
    )
    let patienceA = try parseNonNegativeIntOption(
      options["auto-abc-patience-a"],
      key: "auto-abc-patience-a",
      defaultValue: 40
    )
    let patienceB = try parseNonNegativeIntOption(
      options["auto-abc-patience-b"],
      key: "auto-abc-patience-b",
      defaultValue: 40
    )
    let patienceC = try parseNonNegativeIntOption(
      options["auto-abc-patience-c"],
      key: "auto-abc-patience-c",
      defaultValue: 40
    )
    let minDelta = try parseNonNegativeFloatOption(
      options["auto-abc-min-delta"],
      key: "auto-abc-min-delta",
      defaultValue: max(1e-7, presetBaseConfig.earlyStopMinDelta)
    )

    log(
      "Auto A/B/C enabled: preset=\(preset.rawValue) steps=(\(stepsA),\(stepsB),\(stepsC)) "
        + "patience=(\(patienceA),\(patienceB),\(patienceC)) minDelta=\(fmt(minDelta))"
    )

    var stageAConfig = presetBaseConfig
    stageAConfig.lrSchedule = .exp
    stageAConfig.learningRate = 3e-4
    stageAConfig.lrMin = 1e-4
    stageAConfig.lrHalfLife = 2000
    stageAConfig.lrWarmupSteps = 0
    stageAConfig.loudnessLossMode = .dbL1
    stageAConfig.loudnessLossWeight = 0.0
    stageAConfig.loudnessLossWeightEnd = 0.05
    stageAConfig.loudnessLossWarmupSteps = 10
    stageAConfig.loudnessLossRampSteps = 40
    stageAConfig.earlyStopPatience = patienceA
    stageAConfig.earlyStopMinDelta = minDelta
    try stageAConfig.validate()

    let stageARunDirs = try RunDirectories.create(base: runsBase, runName: stageARun)
    let stageAKernelDumpPath = resolveKernelDumpPath(rawValue: rawKernelDumpValue, runDir: stageARunDirs.root)
    try DDSPE2ETrainer.run(
      dataset: dataset,
      config: stageAConfig,
      runDirs: stageARunDirs,
      options: TrainerOptions(
        steps: stepsA,
        split: split,
        mode: .m2,
        kernelDumpPath: stageAKernelDumpPath,
        initCheckpointPath: initialCheckpointPath,
        profileKernelsStep: profileStep,
        renderEvery: renderEvery,
        renderWavPath: renderWavPath,
        dumpControlsEvery: dumpControlsEvery
      ),
      logger: log
    )
    let stageASummary = try loadTrainSummary(
      from: stageARunDirs.logs.appendingPathComponent("train_summary.json")
    )
    let stageABestCheckpoint = stageARunDirs.checkpoints.appendingPathComponent("model_best.json")
    guard FileManager.default.fileExists(atPath: stageABestCheckpoint.path) else {
      throw CLIError.invalid("Auto A/B/C Stage A did not produce model_best.json")
    }
    log(
      "Auto A/B/C Stage A done: minLoss=\(fmt(stageASummary.minLoss)) "
        + "minStep=\(stageASummary.minStep) best=\(stageABestCheckpoint.path)"
    )

    var stageBConfig = presetBaseConfig
    stageBConfig.lrSchedule = .exp
    stageBConfig.learningRate = 3e-5
    stageBConfig.lrMin = 3e-6
    stageBConfig.lrHalfLife = 120
    stageBConfig.lrWarmupSteps = 0
    stageBConfig.loudnessLossMode = .dbL1
    stageBConfig.loudnessLossWeight = 0.02
    stageBConfig.loudnessLossWeightEnd = nil
    stageBConfig.loudnessLossWarmupSteps = 0
    stageBConfig.loudnessLossRampSteps = 0
    stageBConfig.earlyStopPatience = patienceB
    stageBConfig.earlyStopMinDelta = minDelta
    try stageBConfig.validate()

    let stageBRunDirs = try RunDirectories.create(base: runsBase, runName: stageBRun)
    let stageBKernelDumpPath = resolveKernelDumpPath(rawValue: rawKernelDumpValue, runDir: stageBRunDirs.root)
    try DDSPE2ETrainer.run(
      dataset: dataset,
      config: stageBConfig,
      runDirs: stageBRunDirs,
      options: TrainerOptions(
        steps: stepsB,
        split: split,
        mode: .m2,
        kernelDumpPath: stageBKernelDumpPath,
        initCheckpointPath: stageABestCheckpoint.path,
        profileKernelsStep: profileStep,
        renderEvery: renderEvery,
        renderWavPath: renderWavPath,
        dumpControlsEvery: dumpControlsEvery
      ),
      logger: log
    )
    let stageBSummary = try loadTrainSummary(
      from: stageBRunDirs.logs.appendingPathComponent("train_summary.json")
    )
    let stageBBestCheckpoint = stageBRunDirs.checkpoints.appendingPathComponent("model_best.json")
    guard FileManager.default.fileExists(atPath: stageBBestCheckpoint.path) else {
      throw CLIError.invalid("Auto A/B/C Stage B did not produce model_best.json")
    }
    log(
      "Auto A/B/C Stage B done: minLoss=\(fmt(stageBSummary.minLoss)) "
        + "minStep=\(stageBSummary.minStep) best=\(stageBBestCheckpoint.path)"
    )

    var stageCConfig = presetBaseConfig
    stageCConfig.lrSchedule = .exp
    stageCConfig.learningRate = 1e-5
    stageCConfig.lrMin = 1e-6
    stageCConfig.lrHalfLife = 80
    stageCConfig.lrWarmupSteps = 0
    stageCConfig.loudnessLossMode = .dbL1
    stageCConfig.loudnessLossWeight = 0.0
    stageCConfig.loudnessLossWeightEnd = nil
    stageCConfig.loudnessLossWarmupSteps = 0
    stageCConfig.loudnessLossRampSteps = 0
    stageCConfig.earlyStopPatience = patienceC
    stageCConfig.earlyStopMinDelta = minDelta
    try stageCConfig.validate()

    let stageCRunDirs = try RunDirectories.create(base: runsBase, runName: stageCRun)
    let stageCKernelDumpPath = resolveKernelDumpPath(rawValue: rawKernelDumpValue, runDir: stageCRunDirs.root)
    try DDSPE2ETrainer.run(
      dataset: dataset,
      config: stageCConfig,
      runDirs: stageCRunDirs,
      options: TrainerOptions(
        steps: stepsC,
        split: split,
        mode: .m2,
        kernelDumpPath: stageCKernelDumpPath,
        initCheckpointPath: stageBBestCheckpoint.path,
        profileKernelsStep: profileStep,
        renderEvery: renderEvery,
        renderWavPath: renderWavPath,
        dumpControlsEvery: dumpControlsEvery
      ),
      logger: log
    )
    let stageCSummary = try loadTrainSummary(
      from: stageCRunDirs.logs.appendingPathComponent("train_summary.json")
    )
    let stageCBestCheckpoint = stageCRunDirs.checkpoints.appendingPathComponent("model_best.json")
    guard FileManager.default.fileExists(atPath: stageCBestCheckpoint.path) else {
      throw CLIError.invalid("Auto A/B/C Stage C did not produce model_best.json")
    }
    log(
      "Auto A/B/C Stage C done: minLoss=\(fmt(stageCSummary.minLoss)) "
        + "minStep=\(stageCSummary.minStep) best=\(stageCBestCheckpoint.path)"
    )

    let stageResults = [
      AutoABCStageResult(
        stage: "A",
        runName: stageARun,
        runDir: stageARunDirs.root.path,
        bestCheckpoint: stageABestCheckpoint.path,
        steps: stageASummary.steps,
        minStep: stageASummary.minStep,
        minLoss: stageASummary.minLoss,
        finalLoss: stageASummary.finalLoss
      ),
      AutoABCStageResult(
        stage: "B",
        runName: stageBRun,
        runDir: stageBRunDirs.root.path,
        bestCheckpoint: stageBBestCheckpoint.path,
        steps: stageBSummary.steps,
        minStep: stageBSummary.minStep,
        minLoss: stageBSummary.minLoss,
        finalLoss: stageBSummary.finalLoss
      ),
      AutoABCStageResult(
        stage: "C",
        runName: stageCRun,
        runDir: stageCRunDirs.root.path,
        bestCheckpoint: stageCBestCheckpoint.path,
        steps: stageCSummary.steps,
        minStep: stageCSummary.minStep,
        minLoss: stageCSummary.minLoss,
        finalLoss: stageCSummary.finalLoss
      ),
    ]
    let recommendedStage = stageResults.min { $0.minLoss < $1.minLoss } ?? stageResults[stageResults.count - 1]

    let summary = AutoABCSummary(
      createdAtUTC: ISO8601DateFormatter().string(from: Date()),
      runPrefix: prefix,
      initialCheckpoint: initialCheckpointPath,
      split: split.rawValue,
      stages: stageResults,
      bestCheckpoint: recommendedStage.bestCheckpoint,
      bestLoss: recommendedStage.minLoss
    )
    let summaryPath = runsBase.appendingPathComponent("\(prefix)_auto_abc_summary.json")
    try writeJSON(summary, to: summaryPath)

    log("Auto A/B/C summary written: \(summaryPath.path)")
    log(
      "Auto A/B/C recommended checkpoint: \(recommendedStage.bestCheckpoint) "
        + "(stage=\(recommendedStage.stage), minLoss=\(fmt(recommendedStage.minLoss)))"
    )
  }

  private static func loadTrainSummary(from path: URL) throws -> TrainSummaryValues {
    let data = try Data(contentsOf: path)
    let dict = try JSONDecoder().decode([String: String].self, from: data)

    guard let stepsRaw = dict["steps"], let steps = Int(stepsRaw) else {
      throw CLIError.invalid("Missing or invalid 'steps' in \(path.path)")
    }
    guard let minStepRaw = dict["minStep"], let minStep = Int(minStepRaw) else {
      throw CLIError.invalid("Missing or invalid 'minStep' in \(path.path)")
    }
    guard let minLossRaw = dict["minLoss"], let minLoss = Float(minLossRaw) else {
      throw CLIError.invalid("Missing or invalid 'minLoss' in \(path.path)")
    }
    guard let finalLossRaw = dict["finalLoss"], let finalLoss = Float(finalLossRaw) else {
      throw CLIError.invalid("Missing or invalid 'finalLoss' in \(path.path)")
    }

    return TrainSummaryValues(steps: steps, minStep: minStep, minLoss: minLoss, finalLoss: finalLoss)
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }

  private static func parseBoolOption(_ raw: String?, defaultValue: Bool) -> Bool {
    guard let raw else { return defaultValue }
    let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    switch normalized {
    case "1", "true", "yes", "y", "on":
      return true
    case "0", "false", "no", "n", "off":
      return false
    default:
      return defaultValue
    }
  }

  private static func parsePositiveIntOption(_ raw: String?, key: String, defaultValue: Int) throws -> Int {
    guard let raw else { return defaultValue }
    guard let parsed = Int(raw), parsed > 0 else {
      throw CLIError.invalid("Invalid positive integer for --\(key): \(raw)")
    }
    return parsed
  }

  private static func parseNonNegativeIntOption(_ raw: String?, key: String, defaultValue: Int) throws -> Int {
    guard let raw else { return defaultValue }
    guard let parsed = Int(raw), parsed >= 0 else {
      throw CLIError.invalid("Invalid non-negative integer for --\(key): \(raw)")
    }
    return parsed
  }

  private static func parseNonNegativeFloatOption(
    _ raw: String?,
    key: String,
    defaultValue: Float
  ) throws -> Float {
    guard let raw else { return defaultValue }
    guard let parsed = Float(raw), parsed >= 0 else {
      throw CLIError.invalid("Invalid non-negative float for --\(key): \(raw)")
    }
    return parsed
  }

  private static func parseAutoABCPreset(_ raw: String?) throws -> AutoABCPreset {
    guard let raw else { return .baseline }
    let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard let preset = AutoABCPreset(rawValue: normalized) else {
      throw CLIError.invalid(
        "Invalid value for --auto-abc-preset: \(raw) (expected baseline|best-low-loss)"
      )
    }
    return preset
  }

  private static func applyAutoABCPreset(
    _ preset: AutoABCPreset,
    to config: DDSPE2EConfig
  ) -> DDSPE2EConfig {
    var cfg = config
    switch preset {
    case .baseline:
      return cfg
    case .bestLowLoss:
      // Mirrors the best-known A/B/C chain baseline so auto mode is apples-to-apples.
      cfg.shuffleChunks = false
      cfg.fixedBatch = true
      cfg.seed = 1
      cfg.batchSize = 1
      cfg.gradAccumSteps = 1
      cfg.gradClip = 1.0
      cfg.gradClipMode = .element
      cfg.normalizeGradByFrames = false

      cfg.mseLossWeight = 0.0
      cfg.spectralWeight = 1.0
      cfg.spectralLogmagWeight = 1.0
      cfg.spectralLossMode = .l1
      cfg.spectralWindowSizes = [64, 128, 256, 512, 1024]
      cfg.spectralHopDivisor = 4
      cfg.spectralWarmupSteps = 0
      cfg.spectralRampSteps = 0

      cfg.modelHiddenSize = 128
      cfg.modelNumLayers = 2
      cfg.numHarmonics = 64
      cfg.harmonicHeadMode = .expSigmoid
      cfg.enableNoiseFilter = true
      cfg.decoderBackbone = .transformer
      cfg.transformerDModel = 64
      cfg.transformerLayers = 2
      cfg.transformerFFMultiplier = 2
      cfg.transformerCausal = true
      cfg.transformerUsePositionalEncoding = true
      cfg.controlSmoothingMode = .off
      cfg.loudnessLossMode = .dbL1
      return cfg
    }
  }

  private static func timestampString() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HHmmss"
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    return formatter.string(from: Date())
  }

  private static func handleProbeSmoothing(_ options: [String: String]) throws {
    try DDSPE2ESmoothingProbe.run(options: options, logger: log)
  }

  private static func resolveKernelDumpPath(rawValue: String?, runDir: URL) -> String? {
    guard let rawValue else { return nil }
    if rawValue == "true" {
      return runDir.appendingPathComponent("kernels.metal").path
    }
    return URL(fileURLWithPath: rawValue).path
  }

  private static func parseSplit(_ raw: String?) -> DatasetSplit? {
    guard let raw else { return nil }
    return DatasetSplit(rawValue: raw.lowercased())
  }

  private static func parseOptions(_ args: [String]) -> [String: String] {
    var options: [String: String] = [:]
    var i = 0

    while i < args.count {
      let token = args[i]
      if token.hasPrefix("--") {
        let key = String(token.dropFirst(2))
        if i + 1 < args.count, !args[i + 1].hasPrefix("--") {
          options[key] = args[i + 1]
          i += 2
        } else {
          options[key] = "true"
          i += 1
        }
      } else {
        i += 1
      }
    }

    return options
  }

  private static func fmt(_ value: Float) -> String {
    String(format: "%.5f", value)
  }

  private static func log(_ message: String) {
    print("[DDSPE2E] \(message)")
  }

  private static func printHelp() {
    let text = """
    DDSPE2E - DDSP end-to-end scaffold

    Commands:
      dump-config --output <path>
      preprocess --input <wav-dir> --cache <cache-dir> [--config <json>] [overrides]
      inspect-cache --cache <cache-dir> [--split train|val] [--limit N]
      train --cache <cache-dir> [--runs-dir <dir>] [--run-name <name>] [--steps N] [--split train|val] [--mode dry|m2] [--config <json>] [overrides]
      probe-smoothing --cache <cache-dir> [--split train|val] [--index N] [--output <dir>] [--config <json>] [--init-checkpoint <path>] [overrides]

    Common overrides:
      --sample-rate <float>
      --chunk-size <int>
      --chunk-hop <int>
      --frame-size <int>
      --frame-hop <int>
      --min-f0 <float>
      --max-f0 <float>
      --silence-rms <float>
      --voiced-threshold <float>
      --normalize-to <float>
      --train-split <float>
      --seed <uint64>
      --max-files <int>
      --max-chunks-per-file <int>
      --shuffle <true|false>
      --fixed-batch <true|false>
      --model-hidden <int>
      --model-layers <int>
      --decoder-backbone <mlp|transformer>
      --transformer-d-model <int>
      --transformer-layers <int>
      --transformer-ff-multiplier <int>
      --transformer-causal <true|false>
      --transformer-positional-encoding <true|false>
      --harmonics <int>
      --harmonic-head-mode <legacy|normalized|softmax-db|exp-sigmoid>
      --control-smoothing <fir|off>
      --normalized-harmonic-head <true|false>
      --softmax-temp <float>
      --softmax-temp-end <float>
      --softmax-temp-warmup-steps <int>
      --softmax-temp-ramp-steps <int>
      --softmax-amp-floor <float>
      --softmax-gain-min-db <float>
      --softmax-gain-max-db <float>
      --harmonic-entropy-weight <float>
      --harmonic-entropy-weight-end <float>
      --harmonic-entropy-warmup-steps <int>
      --harmonic-entropy-ramp-steps <int>
      --harmonic-concentration-weight <float>
      --harmonic-concentration-weight-end <float>
      --harmonic-concentration-warmup-steps <int>
      --harmonic-concentration-ramp-steps <int>
      --noise-filter <true|false>
      --noise-filter-size <int>
      --lr <float>
      --batch-size <int>
      --grad-accum-steps <int>
      --grad-clip <float>
      --clip-mode <element|global>
      --normalize-grad-by-frames <true|false>
      --early-stop-patience <int>
      --early-stop-min-delta <float>
      --spectral-windows <csv-int-list>
      --spectral-weight <float>
      --spectral-logmag-weight <float>
      --spectral-loss-mode <l2|l1>
      --spectral-hop-divisor <int>
      --spectral-warmup-steps <int>
      --spectral-ramp-steps <int>
      --loudness-weight <float>
      --loudness-loss-mode <linear-l2|db-l1>
      --loudness-weight-end <float>
      --loudness-warmup-steps <int>
      --loudness-ramp-steps <int>
      --mse-weight <float>
      --log-every <int>
      --checkpoint-every <int>
      --kernel-dump [path]
      --init-checkpoint <model-checkpoint-json>
      --render-every <int>
      --render-wav <path>
      --dump-controls-every <int>
      --auto-abc <true|false> (default: false; runs staged A->B->C sequence in one command)
      --auto-abc-steps-a <int> (default: --steps value)
      --auto-abc-steps-b <int> (default: 300)
      --auto-abc-steps-c <int> (default: 300)
      --auto-abc-patience-a <int> (default: 40)
      --auto-abc-patience-b <int> (default: 40)
      --auto-abc-patience-c <int> (default: 40)
      --auto-abc-min-delta <float> (default: max(1e-7, --early-stop-min-delta))
      --auto-abc-preset <baseline|best-low-loss> (default: baseline)

    Examples:
      swift run DDSPE2E dump-config --output ddsp_config.json
      swift run DDSPE2E preprocess --input Assets --cache .ddsp_cache --max-files 1
      swift run DDSPE2E inspect-cache --cache .ddsp_cache --limit 3
      swift run DDSPE2E train --cache .ddsp_cache --steps 50 --mode m2
      swift run DDSPE2E train --cache .ddsp_cache --steps 1 --kernel-dump
      swift run DDSPE2E probe-smoothing --cache .ddsp_cache --split train --index 0 --output /tmp/ddsp_smoothing_probe
    """
    print(text)
  }
}

enum CLIError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message):
      return "CLI error: \(message)"
    }
  }
}

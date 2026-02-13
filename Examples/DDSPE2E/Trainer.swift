import DGenLazy
import Foundation

struct RunDirectories {
  let root: URL
  let checkpoints: URL
  let renders: URL
  let logs: URL

  static func create(base: URL, runName: String?) throws -> RunDirectories {
    let fm = FileManager.default
    let stamp = timestampString()
    let name = runName ?? "run_\(stamp)"

    let root = base.appendingPathComponent(name, isDirectory: true)
    let checkpoints = root.appendingPathComponent("checkpoints", isDirectory: true)
    let renders = root.appendingPathComponent("renders", isDirectory: true)
    let logs = root.appendingPathComponent("logs", isDirectory: true)

    try fm.createDirectory(at: checkpoints, withIntermediateDirectories: true)
    try fm.createDirectory(at: renders, withIntermediateDirectories: true)
    try fm.createDirectory(at: logs, withIntermediateDirectories: true)

    return RunDirectories(root: root, checkpoints: checkpoints, renders: renders, logs: logs)
  }

  private static func timestampString() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HHmmss"
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    return formatter.string(from: Date())
  }
}

enum TrainMode: String {
  case dry
  case m2
}

struct TrainerOptions {
  var steps: Int
  var split: DatasetSplit
  var mode: TrainMode
  var kernelDumpPath: String?
}

enum DDSPE2ETrainer {
  static func run(
    dataset: CachedDataset,
    config: DDSPE2EConfig,
    runDirs: RunDirectories,
    options: TrainerOptions,
    logger: (String) -> Void
  ) throws {
    switch options.mode {
    case .dry:
      try runDryStart(
        dataset: dataset,
        config: config,
        runDirs: runDirs,
        options: options,
        logger: logger
      )
    case .m2:
      try runDecoderOnlyTraining(
        dataset: dataset,
        config: config,
        runDirs: runDirs,
        options: options,
        logger: logger
      )
    }
  }

  static func runDryStart(
    dataset: CachedDataset,
    config: DDSPE2EConfig,
    runDirs: RunDirectories,
    options: TrainerOptions,
    logger: (String) -> Void
  ) throws {
    let splitEntries = dataset.entries(for: options.split)
    guard !splitEntries.isEmpty else {
      throw DatasetError.invalid("No entries for split \(options.split.rawValue)")
    }

    let resolvedConfigURL = runDirs.root.appendingPathComponent("resolved_config.json")
    try config.write(to: resolvedConfigURL)

    let runMeta: [String: String] = [
      "createdAtUTC": ISO8601DateFormatter().string(from: Date()),
      "split": options.split.rawValue,
      "mode": options.mode.rawValue,
      "datasetRoot": dataset.root.path,
      "manifestChunks": "\(dataset.manifest.chunkCount)",
      "requestedSteps": "\(options.steps)",
    ]
    try writeJSON(runMeta, to: runDirs.root.appendingPathComponent("run_meta.json"))

    logger("Run directory: \(runDirs.root.path)")
    logger("Starting dry training loop (M0 scaffold)")

    let steps = max(1, options.steps)
    var runningAudioMean: Float = 0
    var runningF0Mean: Float = 0
    var totalSamples = 0
    var seen = 0

    for step in 0..<steps {
      let entry = splitEntries[step % splitEntries.count]
      let chunk = try dataset.loadChunk(entry)
      seen += 1

      let audioMean = chunk.audio.reduce(0, +) / Float(max(1, chunk.audio.count))
      let voicedF0 = zip(chunk.f0Hz, chunk.uvMask).compactMap { f0, uv in uv > 0.5 ? f0 : nil }
      let f0Mean = voicedF0.isEmpty ? 0 : voicedF0.reduce(0, +) / Float(voicedF0.count)

      totalSamples += chunk.audio.count
      runningAudioMean += audioMean
      runningF0Mean += f0Mean

      if step == 0 || step == steps - 1 || step % 25 == 0 {
        logger(
          "step=\(step) chunk=\(entry.id) split=\(entry.split.rawValue) "
            + "audioMean=\(format(audioMean)) f0Mean=\(format(f0Mean))"
        )
      }
    }

    let denom = Float(max(1, seen))
    let summary = [
      "steps": "\(steps)",
      "seenChunks": "\(seen)",
      "totalSamplesRead": "\(totalSamples)",
      "avgAudioMean": format(runningAudioMean / denom),
      "avgVoicedF0Mean": format(runningF0Mean / denom),
    ]

    try writeJSON(summary, to: runDirs.logs.appendingPathComponent("dry_train_summary.json"))
    try CheckpointStore.writePlaceholder(
      checkpointsDir: runDirs.checkpoints,
      step: steps,
      seenChunks: seen,
      note: "M0 dry-start checkpoint (no model weights yet)"
    )

    logger("Dry training scaffold complete")
    logger("Wrote summary and placeholder checkpoint")
  }

  static func runDecoderOnlyTraining(
    dataset: CachedDataset,
    config: DDSPE2EConfig,
    runDirs: RunDirectories,
    options: TrainerOptions,
    logger: (String) -> Void
  ) throws {
    let splitEntries = dataset.entries(for: options.split)
    guard !splitEntries.isEmpty else {
      throw DatasetError.invalid("No entries for split \(options.split.rawValue)")
    }

    DGenConfig.backend = .metal
    DGenConfig.sampleRate = config.sampleRate
    DGenConfig.maxFrameCount = max(config.chunkSize, 1)
    DGenConfig.kernelOutputPath = options.kernelDumpPath
    DGenConfig.debug = false
    LazyGraphContext.reset()

    let model = DDSPDecoderModel(config: config)
    let optimizer = Adam(params: model.parameters, lr: config.learningRate)

    let resolvedConfigURL = runDirs.root.appendingPathComponent("resolved_config.json")
    try config.write(to: resolvedConfigURL)

    let runMeta: [String: String] = [
      "createdAtUTC": ISO8601DateFormatter().string(from: Date()),
      "split": options.split.rawValue,
      "mode": options.mode.rawValue,
      "datasetRoot": dataset.root.path,
      "manifestChunks": "\(dataset.manifest.chunkCount)",
      "requestedSteps": "\(options.steps)",
      "numHarmonics": "\(config.numHarmonics)",
      "hiddenSize": "\(config.modelHiddenSize)",
      "lr": "\(config.learningRate)",
      "mseWeight": "\(config.mseLossWeight)",
      "spectralWeightTarget": "\(config.spectralWeight)",
      "spectralHopDivisor": "\(config.spectralHopDivisor)",
      "spectralWarmupSteps": "\(config.spectralWarmupSteps)",
      "spectralRampSteps": "\(config.spectralRampSteps)",
    ]
    try writeJSON(runMeta, to: runDirs.root.appendingPathComponent("run_meta.json"))

    logger("Run directory: \(runDirs.root.path)")
    logger("Starting M2 decoder-only training")

    var order = Array(splitEntries.indices)
    var rng = SeededGenerator(seed: config.seed)
    if config.shuffleChunks {
      order.shuffle(using: &rng)
    }

    let steps = max(1, options.steps)
    var firstLoss: Float?
    var lastFiniteLoss: Float = 0
    var minLoss = Float.greatestFiniteMagnitude
    var minStep = 0
    var validUpdates = 0

    var logLines = [String]()
    logLines.append("step,loss,chunk_id")

    for step in 0..<steps {
      if step > 0, step % order.count == 0, config.shuffleChunks {
        order.shuffle(using: &rng)
      }

      let entry = splitEntries[order[step % order.count]]
      let chunk = try dataset.loadChunk(entry)
      let frameCount = chunk.audio.count

      // Conditioning features [F, 3] -> (f0Norm, loudNorm, uv)
      let features = makeConditioningTensor(
        f0Hz: chunk.f0Hz,
        loudnessDB: chunk.loudnessDB,
        uvMask: chunk.uvMask
      )

      let controls = model.forward(features: features)
      let prediction = DDSPSynth.renderSignal(
        controls: controls,
        f0Frames: chunk.f0Hz,
        uvFrames: chunk.uvMask,
        frameCount: frameCount,
        numHarmonics: config.numHarmonics
      )

      let target = Tensor(chunk.audio).toSignal(maxFrames: frameCount)
      let spectralWeight = spectralWeightForStep(
        step: step,
        targetWeight: config.spectralWeight,
        warmupSteps: config.spectralWarmupSteps,
        rampSteps: config.spectralRampSteps
      )
      let loss = DDSPTrainingLosses.fullLoss(
        prediction: prediction,
        target: target,
        spectralWindowSizes: config.spectralWindowSizes,
        spectralHopDivisor: config.spectralHopDivisor,
        frameCount: frameCount,
        mseWeight: config.mseLossWeight,
        spectralWeight: spectralWeight
      )

      let lossValues = try loss.backward(frames: frameCount)
      let stepLoss = lossValues.reduce(0, +) / Float(max(1, lossValues.count))

      logLines.append("\(step),\(stepLoss),\(entry.id)")

      if !stepLoss.isFinite || stepLoss > 1e6 {
        logger("step=\(step) unstable loss=\(stepLoss); skipping update")
        continue
      }

      if firstLoss == nil { firstLoss = stepLoss }
      lastFiniteLoss = stepLoss
      if stepLoss < minLoss {
        minLoss = stepLoss
        minStep = step
      }

      sanitizeAndClipGradients(params: model.parameters, clip: config.gradClip)
      optimizer.step()
      optimizer.zeroGrad()
      validUpdates += 1

      if step == 0 || step == steps - 1 || step % config.logEvery == 0 {
        logger(
          "step=\(step) loss=\(format(stepLoss)) specW=\(format(spectralWeight)) chunk=\(entry.id)")
      }

      if step > 0, step % config.checkpointEvery == 0 {
        try CheckpointStore.writeModelState(
          checkpointsDir: runDirs.checkpoints,
          step: step,
          params: model.snapshots()
        )
      }
    }

    if validUpdates == 0 {
      throw DatasetError.invalid("No valid training updates were performed (all steps unstable)")
    }

    let first = firstLoss ?? lastFiniteLoss
    let summary: [String: String] = [
      "steps": "\(steps)",
      "validUpdates": "\(validUpdates)",
      "firstLoss": "\(first)",
      "finalLoss": "\(lastFiniteLoss)",
      "minLoss": "\(minLoss)",
      "minStep": "\(minStep)",
      "reduction": "\(first / max(lastFiniteLoss, 1e-12))",
    ]

    try CheckpointStore.writeModelState(
      checkpointsDir: runDirs.checkpoints,
      step: steps,
      params: model.snapshots()
    )
    try writeJSON(summary, to: runDirs.logs.appendingPathComponent("train_summary.json"))

    let csv = logLines.joined(separator: "\n") + "\n"
    try csv.write(to: runDirs.logs.appendingPathComponent("train_log.csv"), atomically: true, encoding: .utf8)

    logger("M2 training complete")
    logger(
      "firstLoss=\(format(first)) finalLoss=\(format(lastFiniteLoss)) reduction=\(format(first / max(lastFiniteLoss, 1e-12))) validUpdates=\(validUpdates)"
    )
  }

  private static func sanitizeAndClipGradients(params: [any LazyValue], clip: Float) {
    for param in params {
      if let tensor = param as? Tensor, let gradTensor = tensor.grad, let gradData = gradTensor.getData() {
        let cleaned = gradData.map { g -> Float in
          if !g.isFinite { return 0 }
          if g > clip { return clip }
          if g < -clip { return -clip }
          return g
        }
        gradTensor.updateDataLazily(cleaned)
      } else if let signal = param as? Signal, let gradSignal = signal.grad, let g = gradSignal.data {
        let cleaned: Float
        if !g.isFinite {
          cleaned = 0
        } else if g > clip {
          cleaned = clip
        } else if g < -clip {
          cleaned = -clip
        } else {
          cleaned = g
        }
        gradSignal.updateDataLazily(cleaned)
      }
    }
  }

  private static func spectralWeightForStep(
    step: Int,
    targetWeight: Float,
    warmupSteps: Int,
    rampSteps: Int
  ) -> Float {
    if targetWeight <= 0 { return 0 }
    if step < warmupSteps { return 0 }
    if rampSteps <= 0 { return targetWeight }
    let rampProgress = Float(step - warmupSteps) / Float(rampSteps)
    let alpha = min(1.0, max(0.0, rampProgress))
    return targetWeight * alpha
  }

  private static func makeConditioningTensor(
    f0Hz: [Float],
    loudnessDB: [Float],
    uvMask: [Float]
  ) -> Tensor {
    let n = min(f0Hz.count, min(loudnessDB.count, uvMask.count))
    if n == 0 {
      return Tensor([[0.0, 0.0, 0.0]])
    }

    var rows = [[Float]]()
    rows.reserveCapacity(n)

    for i in 0..<n {
      let uv = min(1.0, max(0.0, uvMask[i]))
      let safeF0 = max(1.0, f0Hz[i])
      let f0Norm = log2(safeF0 / 440.0)
      let loudNorm = min(1.0, max(0.0, (loudnessDB[i] + 80.0) / 80.0))
      rows.append([f0Norm, loudNorm, uv])
    }

    return Tensor(rows)
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }

  private static func format(_ value: Float) -> String {
    String(format: "%.6f", value)
  }
}

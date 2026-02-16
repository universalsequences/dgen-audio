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
  var initCheckpointPath: String?
}

enum DDSPE2ETrainer {
  private struct GradientStats {
    var paramCount: Int = 0
    var paramsWithGrad: Int = 0
    var totalCount: Int = 0
    var finiteCount: Int = 0
    var nonZeroCount: Int = 0
    var sumSquares: Double = 0
    var maxAbs: Float = 0

    var l2Norm: Float {
      Float(Foundation.sqrt(sumSquares))
    }
  }

  private struct ClipStats {
    var totalCount: Int = 0
    var finiteCount: Int = 0
    var clippedCount: Int = 0
    var nonFiniteCount: Int = 0

    var clippedFraction: Double {
      guard finiteCount > 0 else { return 0 }
      return Double(clippedCount) / Double(finiteCount)
    }
  }

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
    if let checkpointPath = options.initCheckpointPath {
      let checkpoint = try CheckpointStore.readModelState(from: URL(fileURLWithPath: checkpointPath))
      model.loadSnapshots(checkpoint.params)
      logger("Loaded model checkpoint: \(checkpointPath) (step=\(checkpoint.step))")
    }
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
      "lrSchedule": config.lrSchedule.rawValue,
      "lrMin": "\(config.lrMin)",
      "lrWarmupSteps": "\(config.lrWarmupSteps)",
      "mseWeight": "\(config.mseLossWeight)",
      "gradClipMode": config.gradClipMode.rawValue,
      "gradClip": "\(config.gradClip)",
      "normalizeGradByFrames": "\(config.normalizeGradByFrames)",
      "spectralWeightTarget": "\(config.spectralWeight)",
      "spectralHopDivisor": "\(config.spectralHopDivisor)",
      "spectralWarmupSteps": "\(config.spectralWarmupSteps)",
      "spectralRampSteps": "\(config.spectralRampSteps)",
    ]
    try writeJSON(runMeta, to: runDirs.root.appendingPathComponent("run_meta.json"))

    logger("Run directory: \(runDirs.root.path)")
    logger("Starting M2 decoder-only training")

    // Pre-allocate data tensors ONCE before the training loop.
    // This matches the test pattern: define tensors ahead of time,
    // then use updateDataLazily to inject new chunk data each iteration.
    let firstEntry = splitEntries[0]
    let firstChunkFeatureFrames = firstEntry.featureFrames
    let frameCount = max(config.chunkSize, 1)

    // Pad feature frames to next multiple of 8 for GEMM eligibility.
    // Set to firstChunkFeatureFrames to disable padding for debugging.
    let paddedFeatureFrames = ((firstChunkFeatureFrames + 7) / 8) * 8
    logger("Feature frames: \(firstChunkFeatureFrames) → padded: \(paddedFeatureFrames)")

    let featuresTensor = Tensor(
      [[Float]](repeating: [Float](repeating: 0, count: 3), count: paddedFeatureFrames)
    )
    let targetTensor = Tensor([Float](repeating: 0, count: frameCount))
    let synthTensors = DDSPSynth.PreallocatedTensors(
      featureFrames: paddedFeatureFrames,
      numHarmonics: config.numHarmonics,
      enableFIRNoise: config.enableStaticFIRNoise,
      firKernelSize: config.noiseFIRKernelSize
    )

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
    var totalStepMs: Double = 0
    var totalLoadMs: Double = 0
    var totalGraphMs: Double = 0
    var totalBackwardMs: Double = 0
    var totalOptMs: Double = 0
    var maxStepMs: Double = 0
    var maxStep = 0
    var emaStepMs: Double = 0
    let emaAlpha: Double = 0.1

    var logLines = [String]()
    logLines.append("step,loss,chunk_id,step_ms,load_ms,graph_ms,backward_ms,opt_ms")

    for step in 0..<steps {
      let tStepStart = CFAbsoluteTimeGetCurrent()

      // Update learning rate per schedule
      let currentLR = computeLR(
        step: step,
        totalSteps: steps,
        maxLR: config.learningRate,
        minLR: config.lrMin,
        schedule: config.lrSchedule,
        warmupSteps: config.lrWarmupSteps,
        halfLife: config.lrHalfLife
      )
      optimizer.lr = currentLR

      if step > 0, step % order.count == 0, config.shuffleChunks {
        order.shuffle(using: &rng)
      }

      let entry = splitEntries[order[step % order.count]]
      let chunk = try dataset.loadChunk(entry)
      let tAfterLoad = CFAbsoluteTimeGetCurrent()

      // Inject new chunk data into pre-allocated tensors (padded to GEMM-aligned size)
      var conditioningData = makeConditioningData(
        f0Hz: chunk.f0Hz,
        loudnessDB: chunk.loudnessDB,
        uvMask: chunk.uvMask
      )
      let paddingRows = paddedFeatureFrames * 3 - conditioningData.count
      if paddingRows > 0 {
        conditioningData.append(contentsOf: [Float](repeating: 0, count: paddingRows))
      }
      featuresTensor.updateDataLazily(conditioningData)
      targetTensor.updateDataLazily(chunk.audio)

      var paddedF0 = chunk.f0Hz
      var paddedUV = chunk.uvMask
      let framePadding = paddedFeatureFrames - paddedF0.count
      if framePadding > 0 {
        paddedF0.append(contentsOf: [Float](repeating: 0, count: framePadding))
        paddedUV.append(contentsOf: [Float](repeating: 0, count: framePadding))
      }
      synthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)

      // Build graph using pre-allocated tensors
      let controls = model.forward(features: featuresTensor)
      let prediction = DDSPSynth.renderSignal(
        controls: controls,
        tensors: synthTensors,
        featureFrames: chunk.f0Hz.count,
        frameCount: frameCount,
        numHarmonics: config.numHarmonics,
        enableStaticFIRNoise: config.enableStaticFIRNoise,
        noiseFIRKernelSize: config.noiseFIRKernelSize
      )

      let target = targetTensor.toSignal(maxFrames: frameCount)
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
      let tAfterGraph = CFAbsoluteTimeGetCurrent()

      let lossValues = try loss.backward(frames: frameCount)
      let tAfterBackward = CFAbsoluteTimeGetCurrent()
      let stepLoss = lossValues.reduce(0, +) / Float(max(1, lossValues.count))

      let shouldLog = step == 0 || step == steps - 1 || step % config.logEvery == 0
      let loadMs = (tAfterLoad - tStepStart) * 1000.0
      let graphMs = (tAfterGraph - tAfterLoad) * 1000.0
      let backwardMs = (tAfterBackward - tAfterGraph) * 1000.0

      if !stepLoss.isFinite || stepLoss > 1e6 {
        let stepMs = (CFAbsoluteTimeGetCurrent() - tStepStart) * 1000.0
        totalStepMs += stepMs
        totalLoadMs += loadMs
        totalGraphMs += graphMs
        totalBackwardMs += backwardMs
        if stepMs > maxStepMs {
          maxStepMs = stepMs
          maxStep = step
        }
        emaStepMs = step == 0 ? stepMs : (emaStepMs * (1.0 - emaAlpha) + stepMs * emaAlpha)
        logLines.append("\(step),\(stepLoss),\(entry.id),\(stepMs),\(loadMs),\(graphMs),\(backwardMs),0")
        logger(
          "step=\(step) unstable loss=\(stepLoss); skipping update "
            + "tStepMs=\(format(Double(stepMs))) tLoadMs=\(format(Double(loadMs))) "
            + "tGraphMs=\(format(Double(graphMs))) tBackwardMs=\(format(Double(backwardMs)))"
        )
        continue
      }

      if firstLoss == nil { firstLoss = stepLoss }
      lastFiniteLoss = stepLoss
      if stepLoss < minLoss {
        minLoss = stepLoss
        minStep = step
      }

      let tOptStart = CFAbsoluteTimeGetCurrent()
      let preClipGradStats = shouldLog ? summarizeGradients(params: model.parameters) : nil
      let gradScale = config.normalizeGradByFrames ? (1.0 / Float(max(1, frameCount))) : 1.0
      let clipStats = sanitizeAndClipGradients(
        params: model.parameters,
        clip: config.gradClip,
        mode: config.gradClipMode,
        gradScale: gradScale
      )
      let postClipGradStats = shouldLog ? summarizeGradients(params: model.parameters) : nil
      optimizer.step()
      optimizer.zeroGrad()
      let tAfterOpt = CFAbsoluteTimeGetCurrent()
      let optMs = (tAfterOpt - tOptStart) * 1000.0
      let stepMs = (tAfterOpt - tStepStart) * 1000.0
      totalStepMs += stepMs
      totalLoadMs += loadMs
      totalGraphMs += graphMs
      totalBackwardMs += backwardMs
      totalOptMs += optMs
      if stepMs > maxStepMs {
        maxStepMs = stepMs
        maxStep = step
      }
      emaStepMs = step == 0 ? stepMs : (emaStepMs * (1.0 - emaAlpha) + stepMs * emaAlpha)
      logLines.append("\(step),\(stepLoss),\(entry.id),\(stepMs),\(loadMs),\(graphMs),\(backwardMs),\(optMs)")
      validUpdates += 1

      if shouldLog {
        let gradInfo: String
        if let pre = preClipGradStats, let post = postClipGradStats {
          let clipPct = clipStats.clippedFraction * 100.0
          gradInfo =
            " gL2=\(format(pre.l2Norm)) gMax=\(format(pre.maxAbs)) "
            + "gNZ=\(pre.nonZeroCount)/\(pre.finiteCount) "
            + "gFinite=\(pre.finiteCount)/\(pre.totalCount) "
            + "gParams=\(pre.paramsWithGrad)/\(pre.paramCount) "
            + "gMaxPostClip=\(format(post.maxAbs)) "
            + "gScale=\(format(gradScale)) "
            + "gClipMode=\(config.gradClipMode.rawValue) "
            + "gClip=\(clipStats.clippedCount)/\(clipStats.finiteCount) (\(format(clipPct))%) "
            + "gNonFinite=\(clipStats.nonFiniteCount)"
        } else {
          gradInfo = ""
        }
        logger(
          "step=\(step) loss=\(formatLoss(stepLoss)) lr=\(formatLR(currentLR)) specW=\(format(spectralWeight)) "
            + "chunk=\(entry.id)\(gradInfo) "
            + "tStepMs=\(format(Double(stepMs))) tEMAms=\(format(Double(emaStepMs))) "
            + "tLoadMs=\(format(Double(loadMs))) tGraphMs=\(format(Double(graphMs))) "
            + "tBackwardMs=\(format(Double(backwardMs))) tOptMs=\(format(Double(optMs)))")
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
    let denomSteps = Double(max(1, steps))
    let summary: [String: String] = [
      "steps": "\(steps)",
      "validUpdates": "\(validUpdates)",
      "firstLoss": "\(first)",
      "finalLoss": "\(lastFiniteLoss)",
      "minLoss": "\(minLoss)",
      "minStep": "\(minStep)",
      "reduction": "\(first / max(lastFiniteLoss, 1e-12))",
      "avgStepMs": "\(totalStepMs / denomSteps)",
      "avgLoadMs": "\(totalLoadMs / denomSteps)",
      "avgGraphMs": "\(totalGraphMs / denomSteps)",
      "avgBackwardMs": "\(totalBackwardMs / denomSteps)",
      "avgOptMs": "\(totalOptMs / denomSteps)",
      "maxStepMs": "\(maxStepMs)",
      "maxStep": "\(maxStep)",
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
      "firstLoss=\(formatLoss(first)) finalLoss=\(formatLoss(lastFiniteLoss)) reduction=\(format(first / max(lastFiniteLoss, 1e-12))) "
        + "validUpdates=\(validUpdates) avgStepMs=\(format(totalStepMs / denomSteps)) "
        + "avgBackwardMs=\(format(totalBackwardMs / denomSteps)) maxStepMs=\(format(maxStepMs))@\(maxStep)"
    )
  }

  private static func sanitizeAndClipGradients(
    params: [any LazyValue],
    clip: Float,
    mode: GradientClipMode,
    gradScale: Float
  ) -> ClipStats {
    switch mode {
    case .element:
      return sanitizeAndClipGradientsElementwise(params: params, clip: clip, gradScale: gradScale)
    case .global:
      return sanitizeAndClipGradientsGlobalNorm(params: params, clip: clip, gradScale: gradScale)
    }
  }

  private static func sanitizeAndClipGradientsElementwise(
    params: [any LazyValue],
    clip: Float,
    gradScale: Float
  ) -> ClipStats {
    var stats = ClipStats()
    for param in params {
      if let tensor = param as? Tensor, let gradTensor = tensor.grad, let gradData = gradTensor.getData() {
        let cleaned = gradData.map { g -> Float in
          stats.totalCount += 1
          if !g.isFinite {
            stats.nonFiniteCount += 1
            return 0
          }
          stats.finiteCount += 1
          let scaled = g * gradScale
          if scaled > clip {
            stats.clippedCount += 1
            return clip
          }
          if scaled < -clip {
            stats.clippedCount += 1
            return -clip
          }
          return scaled
        }
        gradTensor.updateDataLazily(cleaned)
      } else if let signal = param as? Signal, let gradSignal = signal.grad, let g = gradSignal.data {
        stats.totalCount += 1
        let cleaned: Float
        if !g.isFinite {
          stats.nonFiniteCount += 1
          cleaned = 0
        } else if g * gradScale > clip {
          stats.finiteCount += 1
          stats.clippedCount += 1
          cleaned = clip
        } else if g * gradScale < -clip {
          stats.finiteCount += 1
          stats.clippedCount += 1
          cleaned = -clip
        } else {
          stats.finiteCount += 1
          cleaned = g * gradScale
        }
        gradSignal.updateDataLazily(cleaned)
      }
    }
    return stats
  }

  private static func sanitizeAndClipGradientsGlobalNorm(
    params: [any LazyValue],
    clip: Float,
    gradScale: Float
  ) -> ClipStats {
    var stats = ClipStats()
    var sumSquares: Double = 0.0

    // Pass 1: sanitize non-finite gradients and compute global norm over finite values.
    for param in params {
      if let tensor = param as? Tensor, let gradTensor = tensor.grad, let gradData = gradTensor.getData() {
        let cleaned = gradData.map { g -> Float in
          stats.totalCount += 1
          if !g.isFinite {
            stats.nonFiniteCount += 1
            return 0
          }
          let scaled = g * gradScale
          stats.finiteCount += 1
          sumSquares += Double(scaled) * Double(scaled)
          return scaled
        }
        gradTensor.updateDataLazily(cleaned)
      } else if let signal = param as? Signal, let gradSignal = signal.grad, let g = gradSignal.data {
        stats.totalCount += 1
        if !g.isFinite {
          stats.nonFiniteCount += 1
          gradSignal.updateDataLazily(0)
        } else {
          let scaled = g * gradScale
          stats.finiteCount += 1
          sumSquares += Double(scaled) * Double(scaled)
          gradSignal.updateDataLazily(scaled)
        }
      }
    }

    let globalNorm = Foundation.sqrt(sumSquares)
    guard globalNorm.isFinite, globalNorm > 0 else {
      return stats
    }
    guard globalNorm > Double(clip) else {
      return stats
    }

    let scale = Float(Double(clip) / globalNorm)
    stats.clippedCount = stats.finiteCount

    // Pass 2: apply global scaling.
    for param in params {
      if let tensor = param as? Tensor, let gradTensor = tensor.grad, let gradData = gradTensor.getData() {
        let scaled = gradData.map { $0.isFinite ? ($0 * scale) : 0 }
        gradTensor.updateDataLazily(scaled)
      } else if let signal = param as? Signal, let gradSignal = signal.grad, let g = gradSignal.data {
        gradSignal.updateDataLazily(g.isFinite ? (g * scale) : 0)
      }
    }

    return stats
  }

  private static func summarizeGradients(params: [any LazyValue]) -> GradientStats {
    var stats = GradientStats()
    stats.paramCount = params.count

    for param in params {
      if let tensor = param as? Tensor, let gradTensor = tensor.grad, let gradData = gradTensor.getData() {
        stats.paramsWithGrad += 1
        for g in gradData {
          stats.totalCount += 1
          if g.isFinite {
            stats.finiteCount += 1
            let absG = Swift.abs(g)
            if absG > 0 {
              stats.nonZeroCount += 1
            }
            if absG > stats.maxAbs {
              stats.maxAbs = absG
            }
            stats.sumSquares += Double(g) * Double(g)
          }
        }
      } else if let signal = param as? Signal, let gradSignal = signal.grad, let g = gradSignal.data {
        stats.paramsWithGrad += 1
        stats.totalCount += 1
        if g.isFinite {
          stats.finiteCount += 1
          let absG = Swift.abs(g)
          if absG > 0 {
            stats.nonZeroCount += 1
          }
          if absG > stats.maxAbs {
            stats.maxAbs = absG
          }
          stats.sumSquares += Double(g) * Double(g)
        }
      }
    }

    return stats
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

  /// Compute conditioning features as flat [Float] data (row-major [N, 3]).
  /// Used with pre-allocated tensor via updateDataLazily.
  private static func makeConditioningData(
    f0Hz: [Float],
    loudnessDB: [Float],
    uvMask: [Float]
  ) -> [Float] {
    let n = min(f0Hz.count, min(loudnessDB.count, uvMask.count))
    if n == 0 { return [0.0, 0.0, 0.0] }

    var flat = [Float]()
    flat.reserveCapacity(n * 3)

    for i in 0..<n {
      let uv = min(1.0, max(0.0, uvMask[i]))
      let safeF0 = max(1.0, f0Hz[i])
      let f0Norm = log2(safeF0 / 440.0)
      let loudNorm = min(1.0, max(0.0, (loudnessDB[i] + 80.0) / 80.0))
      flat.append(f0Norm)
      flat.append(loudNorm)
      flat.append(uv)
    }

    return flat
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }

  private static func computeLR(
    step: Int,
    totalSteps: Int,
    maxLR: Float,
    minLR: Float,
    schedule: LRSchedule,
    warmupSteps: Int,
    halfLife: Int = 50
  ) -> Float {
    switch schedule {
    case .none:
      return maxLR
    case .cosine:
      // Linear warmup from minLR to maxLR
      if warmupSteps > 0, step < warmupSteps {
        let t = Float(step) / Float(warmupSteps)
        return minLR + (maxLR - minLR) * t
      }
      // Cosine decay from maxLR to minLR
      let decaySteps = totalSteps - warmupSteps
      guard decaySteps > 0 else { return maxLR }
      let progress = Float(step - warmupSteps) / Float(decaySteps)
      let cosineDecay = 0.5 * (1.0 + cos(Float.pi * progress))
      return minLR + (maxLR - minLR) * cosineDecay
    case .exp:
      // Linear warmup from minLR to maxLR
      if warmupSteps > 0, step < warmupSteps {
        let t = Float(step) / Float(warmupSteps)
        return minLR + (maxLR - minLR) * t
      }
      // Exponential decay: LR halves every `halfLife` steps
      let decayStep = Float(step - warmupSteps)
      let decay = pow(0.5, decayStep / Float(halfLife))
      return max(minLR, minLR + (maxLR - minLR) * decay)
    }
  }

  private static func format(_ value: Float) -> String {
    String(format: "%.6f", value)
  }

  private static func formatLoss(_ value: Float) -> String {
    String(format: "%.4e", value)
  }

  private static func formatLR(_ value: Float) -> String {
    String(format: "%.2e", value)
  }

  private static func format(_ value: Double) -> String {
    String(format: "%.3f", value)
  }
}

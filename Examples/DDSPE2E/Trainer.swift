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
  var profileKernelsStep: Int = -1  // step at which to profile GPU kernels (-1 = disabled)
  var renderEvery: Int = 0          // render audio snapshot every N steps (0 = disabled)
  var renderWavPath: String? = nil  // path to write rendered WAV (overwritten each time)
  var dumpControlsEvery: Int = 0    // dump decoder controls every N steps (0 = disabled)
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
      "harmonicHeadMode": config.harmonicHeadMode.rawValue,
      "normalizedHarmonicHead": "\(config.normalizedHarmonicHead)",
      "softmaxTemperature": "\(config.softmaxTemperature)",
      "softmaxTemperatureEnd": "\(config.softmaxTemperatureEnd ?? config.softmaxTemperature)",
      "softmaxTemperatureWarmupSteps": "\(config.softmaxTemperatureWarmupSteps)",
      "softmaxTemperatureRampSteps": "\(config.softmaxTemperatureRampSteps)",
      "softmaxAmpFloor": "\(config.softmaxAmpFloor)",
      "softmaxGainMinDB": "\(config.softmaxGainMinDB)",
      "softmaxGainMaxDB": "\(config.softmaxGainMaxDB)",
      "harmonicEntropyWeight": "\(config.harmonicEntropyWeight)",
      "harmonicEntropyWeightEnd": "\(config.harmonicEntropyWeightEnd ?? config.harmonicEntropyWeight)",
      "harmonicEntropyWarmupSteps": "\(config.harmonicEntropyWarmupSteps)",
      "harmonicEntropyRampSteps": "\(config.harmonicEntropyRampSteps)",
      "harmonicConcentrationWeight": "\(config.harmonicConcentrationWeight)",
      "harmonicConcentrationWeightEnd":
        "\(config.harmonicConcentrationWeightEnd ?? config.harmonicConcentrationWeight)",
      "harmonicConcentrationWarmupSteps": "\(config.harmonicConcentrationWarmupSteps)",
      "harmonicConcentrationRampSteps": "\(config.harmonicConcentrationRampSteps)",
      "hiddenSize": "\(config.modelHiddenSize)",
      "fixedBatch": "\(config.fixedBatch)",
      "lr": "\(config.learningRate)",
      "lrSchedule": config.lrSchedule.rawValue,
      "lrMin": "\(config.lrMin)",
      "lrWarmupSteps": "\(config.lrWarmupSteps)",
      "mseWeight": "\(config.mseLossWeight)",
      "gradClipMode": config.gradClipMode.rawValue,
      "gradClip": "\(config.gradClip)",
      "normalizeGradByFrames": "\(config.normalizeGradByFrames)",
      "earlyStopPatience": "\(config.earlyStopPatience)",
      "earlyStopMinDelta": "\(config.earlyStopMinDelta)",
      "spectralWeightTarget": "\(config.spectralWeight)",
      "spectralLogmagWeight": "\(config.spectralLogmagWeight)",
      "spectralHopDivisor": "\(config.spectralHopDivisor)",
      "spectralWarmupSteps": "\(config.spectralWarmupSteps)",
      "spectralRampSteps": "\(config.spectralRampSteps)",
      "dumpControlsEvery": "\(options.dumpControlsEvery)",
    ]
    try writeJSON(runMeta, to: runDirs.root.appendingPathComponent("run_meta.json"))

    logger("Run directory: \(runDirs.root.path)")
    logger("Starting M2 decoder-only training")

    if config.batchSize > 1 {
      try runBatchedDecoderTraining(
        dataset: dataset, config: config, runDirs: runDirs, options: options,
        splitEntries: splitEntries, model: model, optimizer: optimizer, logger: logger
      )
      return
    }

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
      numHarmonics: config.numHarmonics
    )

    var order = Array(splitEntries.indices)
    var rng = SeededGenerator(seed: config.seed)
    if config.shuffleChunks {
      order.shuffle(using: &rng)
    }

    let gradAccum = max(1, config.gradAccumSteps)
    let fixedGradAccumEntries: [CachedChunkEntry]
    if config.fixedBatch {
      fixedGradAccumEntries =
        (0..<gradAccum).map { splitEntries[order[$0 % order.count]] }
      logger(
        "Fixed batch enabled (single path): reusing chunk(s) every step: "
          + fixedGradAccumEntries.map(\.id).joined(separator: "+")
      )
    } else {
      fixedGradAccumEntries = []
    }

    let steps = max(1, options.steps)
    var firstLoss: Float?
    var lastFiniteLoss: Float = 0
    var minLoss = Float.greatestFiniteMagnitude
    var minStep = 0
    var bestModelSnapshots: [NamedTensorSnapshot]?
    var bestModelStep = 0
    var noImproveSteps = 0
    var completedSteps = 0
    let earlyStopPatience = config.earlyStopPatience
    let earlyStopMinDelta = config.earlyStopMinDelta
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

    var chunkOffset = 0

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

      // --- Gradient accumulation inner loop ---
      // Run gradAccum backward passes, sum gradients, then do one optimizer step.
      var accumGrads: [Int: [Float]] = [:]
      var totalAccumLoss: Float = 0
      var anyUnstable = false
      var lastEntry = splitEntries[order[0]]
      var lastChunkFeatureFrames = lastEntry.featureFrames
      var stepLoadMs: Double = 0
      var stepGraphMs: Double = 0
      var stepBackwardMs: Double = 0
      let spectralWeight = spectralWeightForStep(
        step: step,
        targetWeight: config.spectralWeight,
        warmupSteps: config.spectralWarmupSteps,
        rampSteps: config.spectralRampSteps
      )
      let harmonicEntropyWeight = harmonicEntropyWeightForStep(
        step: step,
        startWeight: config.harmonicEntropyWeight,
        endWeight: config.harmonicEntropyWeightEnd ?? config.harmonicEntropyWeight,
        warmupSteps: config.harmonicEntropyWarmupSteps,
        rampSteps: config.harmonicEntropyRampSteps
      )
      let harmonicConcentrationWeight = harmonicConcentrationWeightForStep(
        step: step,
        startWeight: config.harmonicConcentrationWeight,
        endWeight: config.harmonicConcentrationWeightEnd ?? config.harmonicConcentrationWeight,
        warmupSteps: config.harmonicConcentrationWarmupSteps,
        rampSteps: config.harmonicConcentrationRampSteps
      )
      let softmaxTemperature = softmaxTemperatureForStep(
        step: step,
        startTemperature: config.softmaxTemperature,
        endTemperature: config.softmaxTemperatureEnd ?? config.softmaxTemperature,
        warmupSteps: config.softmaxTemperatureWarmupSteps,
        rampSteps: config.softmaxTemperatureRampSteps
      )
      model.softmaxTemperature = softmaxTemperature

      for accumIdx in 0..<gradAccum {
        if config.fixedBatch {
          lastEntry = fixedGradAccumEntries[accumIdx % fixedGradAccumEntries.count]
        } else {
          if chunkOffset > 0, chunkOffset % order.count == 0, config.shuffleChunks {
            order.shuffle(using: &rng)
          }
          lastEntry = splitEntries[order[chunkOffset % order.count]]
          chunkOffset += 1
        }

        let chunk = try dataset.loadChunk(lastEntry)
        lastChunkFeatureFrames = chunk.f0Hz.count
        let tAfterLoad = CFAbsoluteTimeGetCurrent()

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

        let controls = model.forward(features: featuresTensor)
        if shouldDumpControls(step: step, every: options.dumpControlsEvery) {
          try dumpDecoderControls(
            step: step,
            model: model,
            conditioningData: conditioningData,
            featureFrames: paddedFeatureFrames,
            batchSize: 1,
            runDirs: runDirs,
            logger: logger
          )
        }
        let prediction = DDSPSynth.renderSignal(
          controls: controls,
          tensors: synthTensors,
          featureFrames: chunk.f0Hz.count,
          frameCount: frameCount,
          numHarmonics: config.numHarmonics
        )
        let target = targetTensor.toSignal(maxFrames: frameCount)
        var loss = DDSPTrainingLosses.fullLoss(
          prediction: prediction,
          target: target,
          spectralWindowSizes: config.spectralWindowSizes,
          spectralHopDivisor: config.spectralHopDivisor,
          frameCount: frameCount,
          mseWeight: config.mseLossWeight,
          spectralWeight: spectralWeight,
          spectralLogmagWeight: config.spectralLogmagWeight
        )
        loss = addHarmonicEntropyRegularization(
          baseLoss: loss,
          controls: controls,
          numHarmonics: config.numHarmonics,
          weight: harmonicEntropyWeight
        )
        loss = addHarmonicConcentrationRegularization(
          baseLoss: loss,
          controls: controls,
          numHarmonics: config.numHarmonics,
          weight: harmonicConcentrationWeight
        )
        let tAfterGraph = CFAbsoluteTimeGetCurrent()

        let lossValues = try loss.backward(frames: frameCount)
        let tAfterBackward = CFAbsoluteTimeGetCurrent()

        stepLoadMs += (tAfterLoad - tStepStart) * 1000.0
        stepGraphMs += (tAfterGraph - tAfterLoad) * 1000.0
        stepBackwardMs += (tAfterBackward - tAfterGraph) * 1000.0

        let accumLoss = lossValues.reduce(0, +) / Float(max(1, lossValues.count))
        if !accumLoss.isFinite || accumLoss > 1e6 {
          anyUnstable = true
          break
        }
        totalAccumLoss += accumLoss

        // Sum gradients from this pass
        for (i, param) in model.parameters.enumerated() {
          if let tensor = param as? Tensor,
            let gradData = tensor.grad?.getData()
          {
            if accumGrads[i] == nil {
              accumGrads[i] = gradData
            } else {
              for j in 0..<min(gradData.count, accumGrads[i]!.count) {
                accumGrads[i]![j] += gradData[j]
              }
            }
          }
        }
      }

      let loadMs = stepLoadMs
      let graphMs = stepGraphMs
      let backwardMs = stepBackwardMs
      let stepLoss = anyUnstable ? Float.nan : totalAccumLoss / Float(gradAccum)
      let shouldLog = step == 0 || step == steps - 1 || step % config.logEvery == 0

      if anyUnstable || !stepLoss.isFinite || stepLoss > 1e6 {
        let stepMs = (CFAbsoluteTimeGetCurrent() - tStepStart) * 1000.0
        totalStepMs += stepMs
        totalLoadMs += loadMs
        totalGraphMs += graphMs
        totalBackwardMs += backwardMs
        if stepMs > maxStepMs { maxStepMs = stepMs; maxStep = step }
        emaStepMs = step == 0 ? stepMs : (emaStepMs * (1.0 - emaAlpha) + stepMs * emaAlpha)
        logLines.append("\(step),\(stepLoss),\(lastEntry.id),\(stepMs),\(loadMs),\(graphMs),\(backwardMs),0")
        logger(
          "step=\(step) unstable loss=\(stepLoss); skipping update "
            + "tStepMs=\(format(Double(stepMs))) tLoadMs=\(format(Double(loadMs))) "
            + "tGraphMs=\(format(Double(graphMs))) tBackwardMs=\(format(Double(backwardMs)))"
        )
        completedSteps = step + 1
        if earlyStopPatience > 0 {
          noImproveSteps += 1
          if noImproveSteps >= earlyStopPatience {
            logger(
              "early-stop triggered after \(completedSteps) steps (patience=\(earlyStopPatience), minDelta=\(earlyStopMinDelta))"
            )
            break
          }
        }
        continue
      }

      if firstLoss == nil { firstLoss = stepLoss }
      lastFiniteLoss = stepLoss
      let improved = stepLoss < (minLoss - earlyStopMinDelta)
      if improved {
        minLoss = stepLoss
        minStep = step
        noImproveSteps = 0
      } else {
        noImproveSteps += 1
      }

      // Average accumulated gradients and write back via the last backward's .grad tensors
      let invAccum = 1.0 / Float(gradAccum)
      for (i, param) in model.parameters.enumerated() {
        if let tensor = param as? Tensor, let gradTensor = tensor.grad,
          var averaged = accumGrads[i]
        {
          for j in 0..<averaged.count { averaged[j] *= invAccum }
          gradTensor.updateDataLazily(averaged)
        }
      }

      if step == options.profileKernelsStep {
        LazyGraphContext.current.profileGPU(frames: frameCount)
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
      if improved {
        bestModelSnapshots = model.snapshots()
        bestModelStep = step
      }

      // Render audio snapshot after the optimizer update so we hear the current model state.
      // backward() already cleared the graph, so we rebuild a fresh forward-only pass,
      // realize it, then clear again so the next training iteration starts clean.
      if let wavPath = options.renderWavPath, options.renderEvery > 0,
        step % options.renderEvery == 0
      {
        do {
          let renderControls = model.forward(features: featuresTensor)
          let renderPrediction = DDSPSynth.renderSignal(
            controls: renderControls,
            tensors: synthTensors,
            featureFrames: lastChunkFeatureFrames,
            frameCount: frameCount,
            numHarmonics: config.numHarmonics
          )
          let samples = try renderPrediction.realize(frames: frameCount)
          LazyGraphContext.current.clearComputationGraph()
          try AudioFile.save(
            url: URL(fileURLWithPath: wavPath),
            samples: samples,
            sampleRate: config.sampleRate)
          logger("step=\(step) rendered audio → \(wavPath)")
        } catch {
          LazyGraphContext.current.clearComputationGraph()
          logger("step=\(step) render warning: \(error)")
        }
      }

      let tAfterOpt = CFAbsoluteTimeGetCurrent()
      let optMs = (tAfterOpt - tOptStart) * 1000.0
      let stepMs = (tAfterOpt - tStepStart) * 1000.0
      totalStepMs += stepMs
      totalLoadMs += loadMs
      totalGraphMs += graphMs
      totalBackwardMs += backwardMs
      totalOptMs += optMs
      if stepMs > maxStepMs { maxStepMs = stepMs; maxStep = step }
      emaStepMs = step == 0 ? stepMs : (emaStepMs * (1.0 - emaAlpha) + stepMs * emaAlpha)
      logLines.append("\(step),\(stepLoss),\(lastEntry.id),\(stepMs),\(loadMs),\(graphMs),\(backwardMs),\(optMs)")
      validUpdates += 1
      completedSteps = step + 1

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
          "step=\(step) loss=\(formatLoss(stepLoss)) lr=\(formatLR(currentLR)) specW=\(format(spectralWeight)) specLogW=\(format(config.spectralLogmagWeight)) entW=\(format(harmonicEntropyWeight)) concW=\(format(harmonicConcentrationWeight)) temp=\(format(softmaxTemperature)) "
            + "chunk=\(lastEntry.id) accum=\(gradAccum)\(gradInfo) "
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
      if earlyStopPatience > 0 && noImproveSteps >= earlyStopPatience {
        logger(
          "early-stop triggered after \(completedSteps) steps (bestStep=\(minStep), patience=\(earlyStopPatience), minDelta=\(earlyStopMinDelta))"
        )
        break
      }
    }

    if validUpdates == 0 {
      throw DatasetError.invalid("No valid training updates were performed (all steps unstable)")
    }

    let first = firstLoss ?? lastFiniteLoss
    let executedSteps = max(1, completedSteps)
    let denomSteps = Double(executedSteps)
    let summary: [String: String] = [
      "steps": "\(executedSteps)",
      "requestedSteps": "\(steps)",
      "earlyStopPatience": "\(earlyStopPatience)",
      "earlyStopMinDelta": "\(earlyStopMinDelta)",
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
      step: executedSteps,
      params: model.snapshots()
    )
    try CheckpointStore.writeBestModelState(
      checkpointsDir: runDirs.checkpoints,
      step: bestModelStep,
      params: bestModelSnapshots ?? model.snapshots()
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

  // MARK: - Batched Training Path

  private static func runBatchedDecoderTraining(
    dataset: CachedDataset,
    config: DDSPE2EConfig,
    runDirs: RunDirectories,
    options: TrainerOptions,
    splitEntries: [CachedChunkEntry],
    model: DDSPDecoderModel,
    optimizer: Adam,
    logger: (String) -> Void
  ) throws {
    let B = config.batchSize
    let frameCount = max(config.chunkSize, 1)
    let firstChunkFeatureFrames = splitEntries[0].featureFrames
    let paddedFeatureFrames = ((firstChunkFeatureFrames + 7) / 8) * 8

    logger("Batched training: batchSize=\(B) featureFrames=\(firstChunkFeatureFrames) → padded=\(paddedFeatureFrames)")

    // Pre-allocate batched tensors
    let featuresTensor = Tensor(
      [[Float]](repeating: [Float](repeating: 0, count: 3), count: paddedFeatureFrames * B)
    )
    let synthTensors = DDSPSynth.PreallocatedTensors(
      featureFrames: paddedFeatureFrames,
      numHarmonics: config.numHarmonics,
      batchSize: B,
      frameCount: frameCount
    )

    var order = Array(splitEntries.indices)
    var rng = SeededGenerator(seed: config.seed)
    if config.shuffleChunks {
      order.shuffle(using: &rng)
    }
    let fixedBatchEntries: [CachedChunkEntry]
    if config.fixedBatch {
      fixedBatchEntries =
        (0..<B).map { splitEntries[order[$0 % order.count]] }
      logger(
        "Fixed batch enabled (batched path): reusing chunk(s) every step: "
          + fixedBatchEntries.map(\.id).joined(separator: "+")
      )
    } else {
      fixedBatchEntries = []
    }

    let steps = max(1, options.steps)
    var firstLoss: Float?
    var lastFiniteLoss: Float = 0
    var minLoss = Float.greatestFiniteMagnitude
    var minStep = 0
    var bestModelSnapshots: [NamedTensorSnapshot]?
    var bestModelStep = 0
    var noImproveSteps = 0
    var completedSteps = 0
    let earlyStopPatience = config.earlyStopPatience
    let earlyStopMinDelta = config.earlyStopMinDelta
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
    logLines.append("step,loss,chunk_ids,step_ms,load_ms,graph_ms,backward_ms,opt_ms")

    var chunkOffset = 0
    let F = paddedFeatureFrames
    let K = config.numHarmonics

    for step in 0..<steps {
      let tStepStart = CFAbsoluteTimeGetCurrent()

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

      let spectralWeight = spectralWeightForStep(
        step: step,
        targetWeight: config.spectralWeight,
        warmupSteps: config.spectralWarmupSteps,
        rampSteps: config.spectralRampSteps
      )
      let harmonicEntropyWeight = harmonicEntropyWeightForStep(
        step: step,
        startWeight: config.harmonicEntropyWeight,
        endWeight: config.harmonicEntropyWeightEnd ?? config.harmonicEntropyWeight,
        warmupSteps: config.harmonicEntropyWarmupSteps,
        rampSteps: config.harmonicEntropyRampSteps
      )
      let harmonicConcentrationWeight = harmonicConcentrationWeightForStep(
        step: step,
        startWeight: config.harmonicConcentrationWeight,
        endWeight: config.harmonicConcentrationWeightEnd ?? config.harmonicConcentrationWeight,
        warmupSteps: config.harmonicConcentrationWarmupSteps,
        rampSteps: config.harmonicConcentrationRampSteps
      )
      let softmaxTemperature = softmaxTemperatureForStep(
        step: step,
        startTemperature: config.softmaxTemperature,
        endTemperature: config.softmaxTemperatureEnd ?? config.softmaxTemperature,
        warmupSteps: config.softmaxTemperatureWarmupSteps,
        rampSteps: config.softmaxTemperatureRampSteps
      )
      model.softmaxTemperature = softmaxTemperature

      // Load B chunks
      var chunks = [CachedChunk]()
      chunks.reserveCapacity(B)
      var chunkIds = [String]()
      for batchIdx in 0..<B {
        let entry: CachedChunkEntry
        if config.fixedBatch {
          entry = fixedBatchEntries[batchIdx % fixedBatchEntries.count]
        } else {
          if chunkOffset > 0, chunkOffset % order.count == 0, config.shuffleChunks {
            order.shuffle(using: &rng)
          }
          entry = splitEntries[order[chunkOffset % order.count]]
          chunkOffset += 1
        }
        chunks.append(try dataset.loadChunk(entry))
        chunkIds.append(entry.id)
      }
      let tAfterLoad = CFAbsoluteTimeGetCurrent()

      // Stack features as [B*F, 3] (batch-major: all frames of chunk0, then chunk1, etc.)
      var conditioningData = [Float]()
      conditioningData.reserveCapacity(B * F * 3)
      for chunk in chunks {
        var chunkCond = makeConditioningData(
          f0Hz: chunk.f0Hz,
          loudnessDB: chunk.loudnessDB,
          uvMask: chunk.uvMask
        )
        let paddingRows = F * 3 - chunkCond.count
        if paddingRows > 0 {
          chunkCond.append(contentsOf: [Float](repeating: 0, count: paddingRows))
        }
        conditioningData.append(contentsOf: chunkCond)
      }
      featuresTensor.updateDataLazily(conditioningData)

      // Stack f0/uv as [F, B] (time-major interleaved)
      var f0Interleaved = [Float](repeating: 0, count: F * B)
      var uvInterleaved = [Float](repeating: 0, count: F * B)
      for frame in 0..<F {
        for b in 0..<B {
          let srcFrame = min(frame, chunks[b].f0Hz.count - 1)
          f0Interleaved[frame * B + b] = srcFrame >= 0 ? chunks[b].f0Hz[srcFrame] : 0
          uvInterleaved[frame * B + b] = srcFrame >= 0 ? chunks[b].uvMask[srcFrame] : 0
        }
      }

      // Stack audio as [frameCount, B] (time-major interleaved)
      var audioInterleaved = [Float](repeating: 0, count: frameCount * B)
      for t in 0..<frameCount {
        for b in 0..<B {
          if t < chunks[b].audio.count {
            audioInterleaved[t * B + b] = chunks[b].audio[t]
          }
        }
      }

      synthTensors.updateBatchedData(
        f0Interleaved: f0Interleaved,
        uvInterleaved: uvInterleaved,
        audioInterleaved: audioInterleaved
      )

      // Forward pass
      let controls = model.forward(features: featuresTensor)
      if shouldDumpControls(step: step, every: options.dumpControlsEvery) {
        try dumpDecoderControls(
          step: step,
          model: model,
          conditioningData: conditioningData,
          featureFrames: F,
          batchSize: B,
          runDirs: runDirs,
          logger: logger
        )
      }
      let prediction = DDSPSynth.renderBatchedSignal(
        controls: controls,
        tensors: synthTensors,
        batchSize: B,
        featureFrames: F,
        frameCount: frameCount,
        numHarmonics: K
      )

      // Target playhead: steps 0, 1, 2, ..., frameCount-1 (one audio sample per frame)
      let targetPlayhead = Signal.accum(
        Signal.constant(1.0),
        reset: 0.0,
        min: 0.0,
        max: Float(frameCount)
      )
      let targetBatched = synthTensors.target!.sample(targetPlayhead)  // [B]

      var loss = DDSPTrainingLosses.fullBatchedLoss(
        prediction: prediction,
        target: targetBatched,
        batchSize: B,
        spectralWindowSizes: config.spectralWindowSizes,
        spectralHopDivisor: config.spectralHopDivisor,
        frameCount: frameCount,
        mseWeight: config.mseLossWeight,
        spectralWeight: spectralWeight,
        spectralLogmagWeight: config.spectralLogmagWeight
      )
      loss = addHarmonicEntropyRegularization(
        baseLoss: loss,
        controls: controls,
        numHarmonics: config.numHarmonics,
        weight: harmonicEntropyWeight
      )
      loss = addHarmonicConcentrationRegularization(
        baseLoss: loss,
        controls: controls,
        numHarmonics: config.numHarmonics,
        weight: harmonicConcentrationWeight
      )
      let tAfterGraph = CFAbsoluteTimeGetCurrent()

      let lossValues = try loss.backward(frames: frameCount)
      let tAfterBackward = CFAbsoluteTimeGetCurrent()

      let loadMs = (tAfterLoad - tStepStart) * 1000.0
      let graphMs = (tAfterGraph - tAfterLoad) * 1000.0
      let backwardMs = (tAfterBackward - tAfterGraph) * 1000.0

      let stepLoss = lossValues.reduce(0, +) / Float(max(1, lossValues.count))
      let shouldLog = step == 0 || step == steps - 1 || step % config.logEvery == 0

      if !stepLoss.isFinite || stepLoss > 1e6 {
        let stepMs = (CFAbsoluteTimeGetCurrent() - tStepStart) * 1000.0
        totalStepMs += stepMs
        totalLoadMs += loadMs
        totalGraphMs += graphMs
        totalBackwardMs += backwardMs
        if stepMs > maxStepMs { maxStepMs = stepMs; maxStep = step }
        emaStepMs = step == 0 ? stepMs : (emaStepMs * (1.0 - emaAlpha) + stepMs * emaAlpha)
        logLines.append("\(step),\(stepLoss),\(chunkIds.joined(separator: "+")),\(stepMs),\(loadMs),\(graphMs),\(backwardMs),0")
        logger(
          "step=\(step) unstable loss=\(stepLoss); skipping update "
            + "tStepMs=\(format(Double(stepMs)))")
        completedSteps = step + 1
        if earlyStopPatience > 0 {
          noImproveSteps += 1
          if noImproveSteps >= earlyStopPatience {
            logger(
              "early-stop triggered after \(completedSteps) steps (patience=\(earlyStopPatience), minDelta=\(earlyStopMinDelta))"
            )
            break
          }
        }
        continue
      }

      if firstLoss == nil { firstLoss = stepLoss }
      lastFiniteLoss = stepLoss
      let improved = stepLoss < (minLoss - earlyStopMinDelta)
      if improved {
        minLoss = stepLoss
        minStep = step
        noImproveSteps = 0
      } else {
        noImproveSteps += 1
      }

      if step == options.profileKernelsStep {
        LazyGraphContext.current.profileGPU(frames: frameCount)
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
      if improved {
        bestModelSnapshots = model.snapshots()
        bestModelStep = step
      }

      // Render audio snapshot
      if let wavPath = options.renderWavPath, options.renderEvery > 0,
        step % options.renderEvery == 0
      {
        do {
          // For rendering, use the first chunk's signal (non-batched path)
          let renderSynthTensors = DDSPSynth.PreallocatedTensors(
            featureFrames: F,
            numHarmonics: K
          )
          var paddedF0 = chunks[0].f0Hz
          var paddedUV = chunks[0].uvMask
          let framePadding = F - paddedF0.count
          if framePadding > 0 {
            paddedF0.append(contentsOf: [Float](repeating: 0, count: framePadding))
            paddedUV.append(contentsOf: [Float](repeating: 0, count: framePadding))
          }
          renderSynthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)

          // Use only first chunk's features for render: extract [F, 3] from [B*F, 3]
          let renderFeatures = Tensor(
            [[Float]](repeating: [Float](repeating: 0, count: 3), count: F)
          )
          let firstChunkCond = makeConditioningData(
            f0Hz: chunks[0].f0Hz,
            loudnessDB: chunks[0].loudnessDB,
            uvMask: chunks[0].uvMask
          )
          var paddedCond = firstChunkCond
          let condPadding = F * 3 - paddedCond.count
          if condPadding > 0 {
            paddedCond.append(contentsOf: [Float](repeating: 0, count: condPadding))
          }
          renderFeatures.updateDataLazily(paddedCond)

          let singleControls = model.forward(features: renderFeatures)
          let renderPrediction = DDSPSynth.renderSignal(
            controls: singleControls,
            tensors: renderSynthTensors,
            featureFrames: chunks[0].f0Hz.count,
            frameCount: frameCount,
            numHarmonics: K
          )
          let samples = try renderPrediction.realize(frames: frameCount)
          LazyGraphContext.current.clearComputationGraph()
          try AudioFile.save(
            url: URL(fileURLWithPath: wavPath),
            samples: samples,
            sampleRate: config.sampleRate)
          logger("step=\(step) rendered audio → \(wavPath)")
        } catch {
          LazyGraphContext.current.clearComputationGraph()
          logger("step=\(step) render warning: \(error)")
        }
      }

      let tAfterOpt = CFAbsoluteTimeGetCurrent()
      let optMs = (tAfterOpt - tOptStart) * 1000.0
      let stepMs = (tAfterOpt - tStepStart) * 1000.0
      totalStepMs += stepMs
      totalLoadMs += loadMs
      totalGraphMs += graphMs
      totalBackwardMs += backwardMs
      totalOptMs += optMs
      if stepMs > maxStepMs { maxStepMs = stepMs; maxStep = step }
      emaStepMs = step == 0 ? stepMs : (emaStepMs * (1.0 - emaAlpha) + stepMs * emaAlpha)
      logLines.append("\(step),\(stepLoss),\(chunkIds.joined(separator: "+")),\(stepMs),\(loadMs),\(graphMs),\(backwardMs),\(optMs)")
      validUpdates += 1
      completedSteps = step + 1

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
          "step=\(step) loss=\(formatLoss(stepLoss)) lr=\(formatLR(currentLR)) specW=\(format(spectralWeight)) specLogW=\(format(config.spectralLogmagWeight)) entW=\(format(harmonicEntropyWeight)) concW=\(format(harmonicConcentrationWeight)) temp=\(format(softmaxTemperature)) "
            + "batch=\(B)\(gradInfo) "
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
      if earlyStopPatience > 0 && noImproveSteps >= earlyStopPatience {
        logger(
          "early-stop triggered after \(completedSteps) steps (bestStep=\(minStep), patience=\(earlyStopPatience), minDelta=\(earlyStopMinDelta))"
        )
        break
      }
    }

    if validUpdates == 0 {
      throw DatasetError.invalid("No valid training updates were performed (all steps unstable)")
    }

    let first = firstLoss ?? lastFiniteLoss
    let executedSteps = max(1, completedSteps)
    let denomSteps = Double(executedSteps)
    let summary: [String: String] = [
      "steps": "\(executedSteps)",
      "requestedSteps": "\(steps)",
      "earlyStopPatience": "\(earlyStopPatience)",
      "earlyStopMinDelta": "\(earlyStopMinDelta)",
      "validUpdates": "\(validUpdates)",
      "batchSize": "\(B)",
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
      step: executedSteps,
      params: model.snapshots()
    )
    try CheckpointStore.writeBestModelState(
      checkpointsDir: runDirs.checkpoints,
      step: bestModelStep,
      params: bestModelSnapshots ?? model.snapshots()
    )
    try writeJSON(summary, to: runDirs.logs.appendingPathComponent("train_summary.json"))

    let csv = logLines.joined(separator: "\n") + "\n"
    try csv.write(to: runDirs.logs.appendingPathComponent("train_log.csv"), atomically: true, encoding: .utf8)

    logger("M2 batched training complete")
    logger(
      "firstLoss=\(formatLoss(first)) finalLoss=\(formatLoss(lastFiniteLoss)) reduction=\(format(first / max(lastFiniteLoss, 1e-12))) "
        + "validUpdates=\(validUpdates) batchSize=\(B) avgStepMs=\(format(totalStepMs / denomSteps)) "
        + "avgBackwardMs=\(format(totalBackwardMs / denomSteps)) maxStepMs=\(format(maxStepMs))@\(maxStep)")
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

  private static func harmonicEntropyWeightForStep(
    step: Int,
    startWeight: Float,
    endWeight: Float,
    warmupSteps: Int,
    rampSteps: Int
  ) -> Float {
    let start = max(0, startWeight)
    let end = max(0, endWeight)
    if step < warmupSteps {
      return start
    }
    if rampSteps <= 0 {
      return end
    }
    let rampProgress = Float(step - warmupSteps) / Float(rampSteps)
    let alpha = min(1.0, max(0.0, rampProgress))
    return start + (end - start) * alpha
  }

  private static func harmonicConcentrationWeightForStep(
    step: Int,
    startWeight: Float,
    endWeight: Float,
    warmupSteps: Int,
    rampSteps: Int
  ) -> Float {
    let start = max(0, startWeight)
    let end = max(0, endWeight)
    if step < warmupSteps {
      return start
    }
    if rampSteps <= 0 {
      return end
    }
    let rampProgress = Float(step - warmupSteps) / Float(rampSteps)
    let alpha = min(1.0, max(0.0, rampProgress))
    return start + (end - start) * alpha
  }

  private static func softmaxTemperatureForStep(
    step: Int,
    startTemperature: Float,
    endTemperature: Float,
    warmupSteps: Int,
    rampSteps: Int
  ) -> Float {
    let start = max(1e-4, startTemperature)
    let end = max(1e-4, endTemperature)
    if step < warmupSteps {
      return start
    }
    if rampSteps <= 0 {
      return end
    }
    let rampProgress = Float(step - warmupSteps) / Float(rampSteps)
    let alpha = min(1.0, max(0.0, rampProgress))
    return start + (end - start) * alpha
  }

  /// Entropy regularizer for softmax harmonic distributions.
  /// Adds weight * (log(K) - mean_entropy) so minimizing loss encourages broader spectra.
  private static func addHarmonicEntropyRegularization(
    baseLoss: Signal,
    controls: DecoderControls,
    numHarmonics: Int,
    weight: Float
  ) -> Signal {
    guard weight > 0, controls.harmonicHeadMode == .softmaxDB else {
      return baseLoss
    }
    let eps: Float = 1e-8
    let amps = controls.harmonicAmps
    let entropyPerRow = -((amps * (amps + eps).log()).sum(axis: 1))  // [rows]
    let entropyMean = entropyPerRow.mean()  // [1]
    let maxEntropy = Float(Foundation.log(Double(max(1, numHarmonics))))
    let entropyGap = Tensor([maxEntropy]) - entropyMean  // [1], >= 0 when entropy below max
    let entropyPenalty = entropyGap.peek(Signal.constant(0.0))
    return baseLoss + entropyPenalty * weight
  }

  /// Concentration regularizer for softmax harmonic distributions.
  /// Adds weight * max(mean(sum(p^2)) - 1/K, 0) so minimizing loss discourages one-bin collapse.
  private static func addHarmonicConcentrationRegularization(
    baseLoss: Signal,
    controls: DecoderControls,
    numHarmonics: Int,
    weight: Float
  ) -> Signal {
    guard weight > 0, controls.harmonicHeadMode == .softmaxDB else {
      return baseLoss
    }
    let amps = controls.harmonicAmps
    let concentrationPerRow = (amps * amps).sum(axis: 1)  // [rows]
    let concentrationMean = concentrationPerRow.mean()  // [1]
    let minConcentration = 1.0 / Float(max(1, numHarmonics))
    let concentrationGap = max(concentrationMean - Tensor([minConcentration]), 0.0)
    let concentrationPenalty = concentrationGap.peek(Signal.constant(0.0))
    return baseLoss + concentrationPenalty * weight
  }

  private struct ControlSnapshot {
    var rows: Int
    var harmonics: Int
    var noiseFilterSize: Int
    var harmonicHeadMode: HarmonicHeadMode
    var harmonicAmps: [Float]
    var harmonicGain: [Float]
    var noiseGain: [Float]
    var noiseFilter: [Float]?
  }

  private static func shouldDumpControls(step: Int, every: Int) -> Bool {
    return every > 0 && step % every == 0
  }

  private static func dumpDecoderControls(
    step: Int,
    model: DDSPDecoderModel,
    conditioningData: [Float],
    featureFrames: Int,
    batchSize: Int,
    runDirs: RunDirectories,
    logger: (String) -> Void
  ) throws {
    let rows = featureFrames * batchSize
    guard rows > 0, conditioningData.count == rows * 3 else { return }
    guard let snapshot = computeDecoderControlsCPU(model: model, conditioningData: conditioningData, rows: rows)
    else {
      logger("step=\(step) control dump skipped (unable to read model parameter data)")
      return
    }

    let controlsDir = runDirs.logs.appendingPathComponent("controls", isDirectory: true)
    try FileManager.default.createDirectory(at: controlsDir, withIntermediateDirectories: true)
    let tag = String(format: "step_%06d", step)

    // 1) Per-frame control summary.
    var summary = "batch,frame,f0_norm,loudness_norm,uv,harmonic_gain,noise_gain,amp_sum,amp_max,amp_argmax\n"
    summary.reserveCapacity(rows * 96)
    for row in 0..<rows {
      let batch = row / featureFrames
      let frame = row % featureFrames
      let base = row * 3
      let f0Norm = conditioningData[base]
      let loudNorm = conditioningData[base + 1]
      let uv = conditioningData[base + 2]

      var ampSum: Float = 0
      var ampMax: Float = -Float.greatestFiniteMagnitude
      var ampArgmax = 0
      let harmBase = row * snapshot.harmonics
      for h in 0..<snapshot.harmonics {
        let a = snapshot.harmonicAmps[harmBase + h]
        ampSum += a
        if a > ampMax {
          ampMax = a
          ampArgmax = h + 1
        }
      }

      summary += "\(batch),\(frame),\(f0Norm),\(loudNorm),\(uv),\(snapshot.harmonicGain[row]),\(snapshot.noiseGain[row]),\(ampSum),\(ampMax),\(ampArgmax)\n"
    }
    try summary.write(
      to: controlsDir.appendingPathComponent("\(tag)_control_summary.csv"),
      atomically: true,
      encoding: .utf8
    )

    // 2) Harmonic amplitudes for batch 0 at start/mid/end frame.
    let selectedFrames = uniqueSortedIndices([0, max(0, featureFrames / 2), max(0, featureFrames - 1)])
    for frame in selectedFrames {
      let row = frame  // batch 0 only
      let harmBase = row * snapshot.harmonics
      var harmCSV = "harmonic,amp\n"
      for h in 0..<snapshot.harmonics {
        harmCSV += "\(h + 1),\(snapshot.harmonicAmps[harmBase + h])\n"
      }
      try harmCSV.write(
        to: controlsDir.appendingPathComponent("\(tag)_b0_f\(frame)_harmonics.csv"),
        atomically: true,
        encoding: .utf8
      )

      // 3) Wavetable synthesized from harmonic amplitudes (one cycle, zero-phase sine basis).
      let tableSize = 512
      let gain = snapshot.harmonicGain[row]
      let harmonicScale: Float =
        snapshot.harmonicHeadMode == .legacy ? (1.0 / Float(max(1, snapshot.harmonics))) : 1.0
      var waveCSV = "sample,value\n"
      waveCSV.reserveCapacity(tableSize * 20)
      for n in 0..<tableSize {
        let phase = 2.0 * Float.pi * Float(n) / Float(tableSize)
        var value: Float = 0
        for h in 0..<snapshot.harmonics {
          let amp = snapshot.harmonicAmps[harmBase + h]
          value += amp * sin(Float(h + 1) * phase)
        }
        value = value * harmonicScale * gain
        waveCSV += "\(n),\(value)\n"
      }
      try waveCSV.write(
        to: controlsDir.appendingPathComponent("\(tag)_b0_f\(frame)_wavetable.csv"),
        atomically: true,
        encoding: .utf8
      )

      if let filter = snapshot.noiseFilter, snapshot.noiseFilterSize > 0 {
        let filterBase = row * snapshot.noiseFilterSize
        var filterCSV = "tap,value\n"
        for i in 0..<snapshot.noiseFilterSize {
          filterCSV += "\(i),\(filter[filterBase + i])\n"
        }
        try filterCSV.write(
          to: controlsDir.appendingPathComponent("\(tag)_b0_f\(frame)_noise_filter.csv"),
          atomically: true,
          encoding: .utf8
        )
      }
    }

    logger("step=\(step) dumped decoder controls → \(controlsDir.path)")
  }

  private static func computeDecoderControlsCPU(
    model: DDSPDecoderModel,
    conditioningData: [Float],
    rows: Int
  ) -> ControlSnapshot? {
    guard rows > 0 else { return nil }

    var hidden = conditioningData  // [rows, 3]
    var inSize = model.inputSize

    for i in 0..<model.numLayers {
      guard let w = model.trunkWeights[i].getData(),
        let b = model.trunkBiases[i].getData(),
        b.count == model.hiddenSize
      else {
        return nil
      }
      let mm = matmul(
        lhs: hidden,
        lhsRows: rows,
        lhsCols: inSize,
        rhs: w,
        rhsCols: model.hiddenSize
      )
      var next = [Float](repeating: 0, count: rows * model.hiddenSize)
      for r in 0..<rows {
        let rowBase = r * model.hiddenSize
        for c in 0..<model.hiddenSize {
          next[rowBase + c] = tanh(mm[rowBase + c] + b[c])
        }
      }
      hidden = next
      inSize = model.hiddenSize
    }

    guard
      let wHarm = model.W_harm.getData(),
      let bHarm = model.b_harm.getData(),
      bHarm.count == model.numHarmonics,
      let wHGain = model.W_hgain.getData(),
      let bHGain = model.b_hgain.getData(),
      bHGain.count == 1,
      let wNoise = model.W_noise.getData(),
      let bNoise = model.b_noise.getData(),
      bNoise.count == 1
    else {
      return nil
    }

    let harmLogits = addRowBias(
      matmul(lhs: hidden, lhsRows: rows, lhsCols: model.hiddenSize, rhs: wHarm, rhsCols: model.numHarmonics),
      rows: rows,
      cols: model.numHarmonics,
      bias: bHarm
    )
    let hGainLogits = addRowBias(
      matmul(lhs: hidden, lhsRows: rows, lhsCols: model.hiddenSize, rhs: wHGain, rhsCols: 1),
      rows: rows,
      cols: 1,
      bias: bHGain
    )
    let noiseLogits = addRowBias(
      matmul(lhs: hidden, lhsRows: rows, lhsCols: model.hiddenSize, rhs: wNoise, rhsCols: 1),
      rows: rows,
      cols: 1,
      bias: bNoise
    )

    let harm: [Float]
    let hGain: [Float]
    switch model.harmonicHeadMode {
    case .legacy:
      harm = harmLogits.map(sigmoidCPU)
      hGain = hGainLogits.map(sigmoidCPU)
    case .normalized:
      let harmPositive = harmLogits.map(softplusCPU)
      var normalized = [Float](repeating: 0, count: rows * model.numHarmonics)
      for r in 0..<rows {
        let rowBase = r * model.numHarmonics
        var sum: Float = 0
        for c in 0..<model.numHarmonics {
          sum += harmPositive[rowBase + c]
        }
        let denom = sum + 1e-6
        for c in 0..<model.numHarmonics {
          normalized[rowBase + c] = harmPositive[rowBase + c] / denom
        }
      }
      harm = normalized
      hGain = hGainLogits.map(softplusCPU)
    case .softmaxDB:
      let temperature = max(1e-4, model.softmaxTemperature)
      let scaledLogits = harmLogits.map { $0 / temperature }
      let softmaxDist = softmaxRowsCPU(scaledLogits, rows: rows, cols: model.numHarmonics)
      if model.softmaxAmpFloor > 0 {
        let floorMix = min(max(model.softmaxAmpFloor, 0), 1)
        let uniform = floorMix / Float(max(1, model.numHarmonics))
        harm = softmaxDist.map { $0 * (1.0 - floorMix) + uniform }
      } else {
        harm = softmaxDist
      }
      let ln10Over20: Float = 0.11512925464970229
      let gainDB = hGainLogits.map {
        model.softmaxGainMinDB + sigmoidCPU($0) * (model.softmaxGainMaxDB - model.softmaxGainMinDB)
      }
      hGain = gainDB.map { Foundation.exp($0 * ln10Over20) }
    case .expSigmoid:
      harm = harmLogits.map { expSigmoidCPU($0) }
      hGain = hGainLogits.map { expSigmoidCPU($0) }
    }
    let nGain = model.harmonicHeadMode == .expSigmoid ? noiseLogits.map { expSigmoidCPU($0) } : noiseLogits.map(sigmoidCPU)

    var filterOut: [Float]? = nil
    var filterSize = 0
    if model.enableNoiseFilter,
      let wf = model.W_filter?.getData(),
      let bf = model.b_filter?.getData()
    {
      filterSize = model.noiseFilterSize
      guard bf.count == filterSize else { return nil }
      let filterLogits = addRowBias(
        matmul(lhs: hidden, lhsRows: rows, lhsCols: model.hiddenSize, rhs: wf, rhsCols: filterSize),
        rows: rows,
        cols: filterSize,
        bias: bf
      )
      filterOut = filterLogits.map(sigmoidCPU)
    }

    return ControlSnapshot(
      rows: rows,
      harmonics: model.numHarmonics,
      noiseFilterSize: filterSize,
      harmonicHeadMode: model.harmonicHeadMode,
      harmonicAmps: harm,
      harmonicGain: hGain,
      noiseGain: nGain,
      noiseFilter: filterOut
    )
  }

  private static func matmul(
    lhs: [Float],
    lhsRows: Int,
    lhsCols: Int,
    rhs: [Float],
    rhsCols: Int
  ) -> [Float] {
    var out = [Float](repeating: 0, count: lhsRows * rhsCols)
    for r in 0..<lhsRows {
      let lhsRow = r * lhsCols
      let outRow = r * rhsCols
      for c in 0..<rhsCols {
        var sum: Float = 0
        for k in 0..<lhsCols {
          sum += lhs[lhsRow + k] * rhs[k * rhsCols + c]
        }
        out[outRow + c] = sum
      }
    }
    return out
  }

  private static func addRowBias(_ x: [Float], rows: Int, cols: Int, bias: [Float]) -> [Float] {
    var out = x
    for r in 0..<rows {
      let rowBase = r * cols
      for c in 0..<cols {
        out[rowBase + c] += bias[c]
      }
    }
    return out
  }

  private static func sigmoidCPU(_ x: Float) -> Float {
    1.0 / (1.0 + Foundation.exp(-x))
  }

  private static func softplusCPU(_ x: Float) -> Float {
    let positive: Float = max(0, x)
    let correction = Foundation.log(1.0 + Foundation.exp(-Double(abs(x))))
    return positive + Float(correction)
  }

  private static func softmaxRowsCPU(_ x: [Float], rows: Int, cols: Int) -> [Float] {
    guard rows > 0, cols > 0 else { return x }
    var out = [Float](repeating: 0, count: x.count)
    for r in 0..<rows {
      let base = r * cols
      var rowMax = -Float.greatestFiniteMagnitude
      for c in 0..<cols {
        rowMax = max(rowMax, x[base + c])
      }
      var sumExp: Float = 0
      for c in 0..<cols {
        let e = Float(Foundation.exp(Double(x[base + c] - rowMax)))
        out[base + c] = e
        sumExp += e
      }
      let denom = max(sumExp, 1e-12)
      for c in 0..<cols {
        out[base + c] /= denom
      }
    }
    return out
  }

  private static func expSigmoidCPU(
    _ x: Float,
    exponent: Float = 10.0,
    maxValue: Float = 2.0,
    threshold: Float = 1e-7
  ) -> Float {
    let sig = sigmoidCPU(x)
    let shaped = pow(sig, Float(Foundation.log(Double(exponent))))
    return maxValue * shaped + threshold
  }

  private static func uniqueSortedIndices(_ values: [Int]) -> [Int] {
    Array(Set(values)).sorted()
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

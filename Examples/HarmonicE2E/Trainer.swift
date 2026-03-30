import DGen
import DGenLazy
import Foundation

struct RunDirectories {
  let root: URL
  let checkpoints: URL
  let renders: URL
  let logs: URL

  static func create(base: URL, runName: String?) throws -> RunDirectories {
    let fm = FileManager.default
    let name = runName ?? "run_\(timestampString())"
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

enum HarmonicE2ETrainer {
  private static let conditioningFeatureCount = 3

  static func run(
    dataset: CachedDataset,
    config: HarmonicE2EConfig,
    runDirs: RunDirectories,
    steps: Int,
    split: DatasetSplit,
    renderEvery: Int,
    kernelDumpPath: String?,
    logGraphStats: Bool,
    logger: (String) -> Void
  ) throws {
    let splitEntries = dataset.entries(for: split)
    guard !splitEntries.isEmpty else {
      throw DatasetError.invalid("No entries for split \(split.rawValue)")
    }

    DGenConfig.backend = .metal
    DGenConfig.sampleRate = config.sampleRate
    DGenConfig.maxFrameCount = max(config.chunkSize, 1)
    DGenConfig.kernelOutputPath = kernelDumpPath
    DGenConfig.debug = false
    LazyGraphContext.reset()

    let model = HarmonicDecoderModel(config: config)
    let optimizer = Adam(params: model.parameters, lr: config.learningRate)
    let frameCount = max(config.chunkSize, 1)
    let paddedFeatureFrames = ((splitEntries[0].featureFrames + 7) / 8) * 8
    let batchSize = max(1, config.batchSize)

    if batchSize > 1 {
      try runBatched(
        dataset: dataset,
        splitEntries: splitEntries,
        config: config,
        runDirs: runDirs,
        steps: steps,
        renderEvery: renderEvery,
        logger: logger,
        model: model,
        optimizer: optimizer,
        frameCount: frameCount,
        paddedFeatureFrames: paddedFeatureFrames
      )
      return
    }

    let featuresTensor = Tensor(
      [[Float]](repeating: [Float](repeating: 0, count: conditioningFeatureCount), count: paddedFeatureFrames)
    )
    let targetTensor = Tensor([Float](repeating: 0, count: frameCount))
    let synthTensors = HarmonicSynth.PreallocatedTensors(
      featureFrames: paddedFeatureFrames,
      numHarmonics: config.numHarmonics
    )

    try config.write(to: runDirs.root.appendingPathComponent("resolved_config.json"))
    try writeJSON(
      [
        "createdAtUTC": ISO8601DateFormatter().string(from: Date()),
        "split": split.rawValue,
        "requestedSteps": "\(steps)",
        "datasetRoot": dataset.root.path,
      ],
      to: runDirs.root.appendingPathComponent("run_meta.json")
    )

    var order = Array(splitEntries.indices)
    var rng = SeededGenerator(seed: config.seed)
    if config.shuffleChunks {
      order.shuffle(using: &rng)
    }

    var firstLoss: Float?
    var finalLoss: Float = 0
    var minLoss = Float.greatestFiniteMagnitude
    var minStep = 0
    var bestSnapshots: [NamedTensorSnapshot] = model.snapshots()
    var chunkOffset = 0
    var logLines = ["step,loss,chunk_id"]

    for step in 0..<max(1, steps) {
      if chunkOffset > 0, chunkOffset % order.count == 0, config.shuffleChunks {
        order.shuffle(using: &rng)
      }
      let entry = splitEntries[order[chunkOffset % order.count]]
      chunkOffset += 1
      let chunk = try dataset.loadChunk(entry)

      var conditioningData = makeConditioningData(
        chunk: chunk,
        config: config,
        paddedFrames: paddedFeatureFrames
      )
      let expectedCount = paddedFeatureFrames * conditioningFeatureCount
      if conditioningData.count < expectedCount {
        conditioningData.append(contentsOf: [Float](repeating: 0, count: expectedCount - conditioningData.count))
      }
      featuresTensor.updateDataLazily(conditioningData)
      targetTensor.updateDataLazily(chunk.audio)

      var paddedF0 = chunk.f0Hz
      var paddedUV = chunk.uvMask
      if paddedF0.count < paddedFeatureFrames {
        paddedF0.append(contentsOf: [Float](repeating: 0, count: paddedFeatureFrames - paddedF0.count))
        paddedUV.append(contentsOf: [Float](repeating: 0, count: paddedFeatureFrames - paddedUV.count))
      }
      synthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)

      let controls = model.forward(features: featuresTensor)
      let prediction = HarmonicSynth.renderSignal(
        controls: controls,
        tensors: synthTensors,
        featureFrames: chunk.f0Hz.count,
        frameCount: frameCount,
        numHarmonics: config.numHarmonics,
        harmonicPathScale: config.harmonicPathScale,
        noisePathScale: config.noisePathScale
      )
      let target = targetTensor.toSignal(maxFrames: frameCount)
      let loss = fullLoss(
        prediction: prediction,
        target: target,
        config: config,
        frameCount: frameCount
      )
      if logGraphStats, step == 0 {
        let lazyGraph = LazyGraphContext.current
        logger(
          "graphStats nodes=\(lazyGraph.debugNodeCount) tensors=\(lazyGraph.debugTensorCount) "
            + "memoryCells=\(lazyGraph.debugMemoryCellCount)"
        )
      }
      let lossValues = try loss.backward(frames: frameCount)
      let stepLoss = lossValues.reduce(0, +) / Float(max(1, lossValues.count))
      optimizer.step()
      optimizer.zeroGrad()

      if firstLoss == nil {
        firstLoss = stepLoss
      }
      finalLoss = stepLoss
      if stepLoss.isFinite, stepLoss < minLoss {
        minLoss = stepLoss
        minStep = step
        bestSnapshots = model.snapshots()
      }

      logLines.append("\(step),\(stepLoss),\(entry.id)")
      if step == 0 || step == steps - 1 || step % max(1, config.logEvery) == 0 {
        logger("step=\(step) loss=\(format(stepLoss)) chunk=\(entry.id)")
      }

      if renderEvery > 0, (step == 0 || (step + 1) % renderEvery == 0 || step == steps - 1) {
        LazyGraphContext.current.clearComputationGraph()
        featuresTensor.updateDataLazily(conditioningData)
        synthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)
        let renderControls = model.forward(features: featuresTensor)
        let renderPrediction = HarmonicSynth.renderSignal(
          controls: renderControls,
          tensors: synthTensors,
          featureFrames: chunk.f0Hz.count,
          frameCount: frameCount,
          numHarmonics: config.numHarmonics,
          harmonicPathScale: config.harmonicPathScale,
          noisePathScale: config.noisePathScale
        )
        let samples = try renderPrediction.realize(frames: frameCount)
        let wavURL = runDirs.renders.appendingPathComponent(String(format: "step_%08d.wav", step))
        try AudioFile.save(url: wavURL, samples: samples, sampleRate: config.sampleRate)
        try exportNoiseFilterControls(
          noiseFilter: renderControls.noiseFilter,
          step: step,
          runDirs: runDirs,
          sampleRate: config.sampleRate
        )
        LazyGraphContext.current.clearComputationGraph()
      }

      if (step + 1) % max(1, config.checkpointEvery) == 0 || step == steps - 1 {
        try CheckpointStore.writeModelState(
          checkpointsDir: runDirs.checkpoints,
          step: step,
          params: model.snapshots()
        )
      }
    }

    try CheckpointStore.writeBestModelState(
      checkpointsDir: runDirs.checkpoints,
      step: minStep,
      params: bestSnapshots
    )
    try writeJSON(
      [
        "steps": "\(steps)",
        "minStep": "\(minStep)",
        "minLoss": "\(minLoss)",
        "firstLoss": "\(firstLoss ?? finalLoss)",
        "finalLoss": "\(finalLoss)",
      ],
      to: runDirs.logs.appendingPathComponent("train_summary.json")
    )
    try logLines.joined(separator: "\n").write(
      to: runDirs.logs.appendingPathComponent("train_log.csv"),
      atomically: true,
      encoding: .utf8
    )
    logger(
      "Training complete: firstLoss=\(format(firstLoss ?? finalLoss)) finalLoss=\(format(finalLoss)) minLoss=\(format(minLoss))@\(minStep)"
    )
  }

  private static func runBatched(
    dataset: CachedDataset,
    splitEntries: [CachedChunkEntry],
    config: HarmonicE2EConfig,
    runDirs: RunDirectories,
    steps: Int,
    renderEvery: Int,
    logger: (String) -> Void,
    model: HarmonicDecoderModel,
    optimizer: Adam,
    frameCount: Int,
    paddedFeatureFrames: Int
  ) throws {
    let B = max(1, config.batchSize)
    let F = paddedFeatureFrames
    logger("Batched training: batchSize=\(B) featureFrames=\(splitEntries[0].featureFrames) -> padded=\(F)")

    let featuresTensor = Tensor(
      [[Float]](
        repeating: [Float](repeating: 0, count: conditioningFeatureCount),
        count: F * B
      )
    )
    let synthTensors = HarmonicSynth.PreallocatedTensors(
      featureFrames: F,
      numHarmonics: config.numHarmonics,
      batchSize: B,
      frameCount: frameCount
    )

    try config.write(to: runDirs.root.appendingPathComponent("resolved_config.json"))
    try writeJSON(
      [
        "createdAtUTC": ISO8601DateFormatter().string(from: Date()),
        "split": "train",
        "requestedSteps": "\(steps)",
        "datasetRoot": dataset.root.path,
        "batchSize": "\(B)",
      ],
      to: runDirs.root.appendingPathComponent("run_meta.json")
    )

    var order = Array(splitEntries.indices)
    var rng = SeededGenerator(seed: config.seed)
    if config.shuffleChunks {
      order.shuffle(using: &rng)
    }

    var firstLoss: Float?
    var finalLoss: Float = 0
    var minLoss = Float.greatestFiniteMagnitude
    var minStep = 0
    var bestSnapshots: [NamedTensorSnapshot] = model.snapshots()
    var chunkOffset = 0
    var logLines = ["step,loss,chunk_ids"]

    for step in 0..<max(1, steps) {
      var chunks = [CachedChunk]()
      var entries = [CachedChunkEntry]()
      chunks.reserveCapacity(B)
      entries.reserveCapacity(B)

      for _ in 0..<B {
        if chunkOffset > 0, chunkOffset % order.count == 0, config.shuffleChunks {
          order.shuffle(using: &rng)
        }
        let entry = splitEntries[order[chunkOffset % order.count]]
        chunkOffset += 1
        entries.append(entry)
        chunks.append(try dataset.loadChunk(entry))
      }

      var conditioningData = [Float]()
      conditioningData.reserveCapacity(B * F * conditioningFeatureCount)
      for chunk in chunks {
        var chunkCond = makeConditioningData(
          chunk: chunk,
          config: config,
          paddedFrames: F
        )
        let expectedCount = F * conditioningFeatureCount
        if chunkCond.count < expectedCount {
          chunkCond.append(contentsOf: [Float](repeating: 0, count: expectedCount - chunkCond.count))
        }
        conditioningData.append(contentsOf: chunkCond)
      }
      featuresTensor.updateDataLazily(conditioningData)

      var f0Interleaved = [Float](repeating: 0, count: F * B)
      var uvInterleaved = [Float](repeating: 0, count: F * B)
      for frame in 0..<F {
        for b in 0..<B {
          let srcFrame = min(frame, max(0, chunks[b].f0Hz.count - 1))
          f0Interleaved[frame * B + b] = chunks[b].f0Hz.isEmpty ? 0 : chunks[b].f0Hz[srcFrame]
          uvInterleaved[frame * B + b] = chunks[b].uvMask.isEmpty ? 0 : chunks[b].uvMask[srcFrame]
        }
      }

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

      let controls = model.forward(features: featuresTensor)
      let prediction = HarmonicSynth.renderBatchedSignal(
        controls: controls,
        tensors: synthTensors,
        batchSize: B,
        featureFrames: F,
        frameCount: frameCount,
        numHarmonics: config.numHarmonics,
        harmonicPathScale: config.harmonicPathScale,
        noisePathScale: config.noisePathScale
      )
      let targetPlayhead = Signal.accum(
        Signal.constant(1.0),
        reset: 0.0,
        min: 0.0,
        max: Float(frameCount)
      )
      let target = synthTensors.target!.sample(targetPlayhead)
      let loss = fullBatchedLoss(
        prediction: prediction,
        target: target,
        batchSize: B,
        config: config,
        frameCount: frameCount
      )
      let lossValues = try loss.backward(frames: frameCount)
      let stepLoss = lossValues.reduce(0, +) / Float(max(1, lossValues.count))
      optimizer.step()
      optimizer.zeroGrad()

      if firstLoss == nil {
        firstLoss = stepLoss
      }
      finalLoss = stepLoss
      if stepLoss.isFinite, stepLoss < minLoss {
        minLoss = stepLoss
        minStep = step
        bestSnapshots = model.snapshots()
      }

      let chunkIds = entries.map(\.id).joined(separator: "+")
      logLines.append("\(step),\(stepLoss),\(chunkIds)")
      if step == 0 || step == steps - 1 || step % max(1, config.logEvery) == 0 {
        logger("step=\(step) loss=\(format(stepLoss)) chunks=\(chunkIds)")
      }

      if renderEvery > 0, (step == 0 || (step + 1) % renderEvery == 0 || step == steps - 1) {
        LazyGraphContext.current.clearComputationGraph()
        let renderFeatures = Tensor(
          [[Float]](repeating: [Float](repeating: 0, count: conditioningFeatureCount), count: F)
        )
        var renderCond = makeConditioningData(
          chunk: chunks[0],
          config: config,
          paddedFrames: F
        )
        let expectedCount = F * conditioningFeatureCount
        if renderCond.count < expectedCount {
          renderCond.append(contentsOf: [Float](repeating: 0, count: expectedCount - renderCond.count))
        }
        renderFeatures.updateDataLazily(renderCond)

        let renderSynthTensors = HarmonicSynth.PreallocatedTensors(
          featureFrames: F,
          numHarmonics: config.numHarmonics
        )
        var paddedF0 = chunks[0].f0Hz
        var paddedUV = chunks[0].uvMask
        if paddedF0.count < F {
          paddedF0.append(contentsOf: [Float](repeating: 0, count: F - paddedF0.count))
          paddedUV.append(contentsOf: [Float](repeating: 0, count: F - paddedUV.count))
        }
        renderSynthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)
        let renderControls = model.forward(features: renderFeatures)
        let renderPrediction = HarmonicSynth.renderSignal(
          controls: renderControls,
          tensors: renderSynthTensors,
          featureFrames: chunks[0].f0Hz.count,
          frameCount: frameCount,
          numHarmonics: config.numHarmonics,
          harmonicPathScale: config.harmonicPathScale,
          noisePathScale: config.noisePathScale
        )
        let samples = try renderPrediction.realize(frames: frameCount)
        let wavURL = runDirs.renders.appendingPathComponent(String(format: "step_%08d.wav", step))
        try AudioFile.save(url: wavURL, samples: samples, sampleRate: config.sampleRate)
        try exportNoiseFilterControls(
          noiseFilter: renderControls.noiseFilter,
          step: step,
          runDirs: runDirs,
          sampleRate: config.sampleRate
        )
        LazyGraphContext.current.clearComputationGraph()
      }

      if (step + 1) % max(1, config.checkpointEvery) == 0 || step == steps - 1 {
        try CheckpointStore.writeModelState(
          checkpointsDir: runDirs.checkpoints,
          step: step,
          params: model.snapshots()
        )
      }
    }

    try CheckpointStore.writeBestModelState(
      checkpointsDir: runDirs.checkpoints,
      step: minStep,
      params: bestSnapshots
    )
    try writeJSON(
      [
        "steps": "\(steps)",
        "minStep": "\(minStep)",
        "minLoss": "\(minLoss)",
        "firstLoss": "\(firstLoss ?? finalLoss)",
        "finalLoss": "\(finalLoss)",
        "batchSize": "\(B)",
      ],
      to: runDirs.logs.appendingPathComponent("train_summary.json")
    )
    try logLines.joined(separator: "\n").write(
      to: runDirs.logs.appendingPathComponent("train_log.csv"),
      atomically: true,
      encoding: .utf8
    )
    logger(
      "Training complete: firstLoss=\(format(firstLoss ?? finalLoss)) finalLoss=\(format(finalLoss)) minLoss=\(format(minLoss))@\(minStep) batchSize=\(B)"
    )
  }

  private static func fullLoss(
    prediction: Signal,
    target: Signal,
    config: HarmonicE2EConfig,
    frameCount: Int
  ) -> Signal {
    var total = Signal.constant(0.0)

    if config.mseLossWeight > 0 {
      total = total + mse(prediction, target) * config.mseLossWeight
    }

    let usableWindows = config.spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    if config.spectralWeight > 0, !usableWindows.isEmpty {
      var spectral = Signal.constant(0.0)
      for window in usableWindows {
        let hop = max(1, window / max(1, config.spectralHopDivisor))
        spectral = spectral + spectralLossFFT(
          prediction,
          target,
          windowSize: window,
          lossMode: .l1,
          hop: hop,
          normalize: true
        )
      }
      total = total + spectral * (config.spectralWeight / Float(usableWindows.count))
    }

    if config.spectralLogmagWeight > 0, !usableWindows.isEmpty {
      var spectral = Signal.constant(0.0)
      for window in usableWindows {
        let hop = max(1, window / max(1, config.spectralHopDivisor))
        spectral = spectral + spectralLossFFT(
          prediction,
          target,
          windowSize: window,
          useLogMagnitude: true,
          lossMode: .l1,
          hop: hop,
          normalize: true
        )
      }
      total = total + spectral * (config.spectralLogmagWeight / Float(usableWindows.count))
    }

    return total
  }

  private static func fullBatchedLoss(
    prediction: SignalTensor,
    target: SignalTensor,
    batchSize: Int,
    config: HarmonicE2EConfig,
    frameCount: Int
  ) -> Signal {
    var total = Signal.constant(0.0)

    if config.mseLossWeight > 0 {
      let diff = prediction - target
      let batchMSE = (diff * diff).sum() * (1.0 / Float(batchSize))
      total = total + batchMSE * config.mseLossWeight
    }

    let usableWindows = config.spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    if config.spectralWeight > 0, !usableWindows.isEmpty {
      var spectral = Signal.constant(0.0)
      for window in usableWindows {
        let hop = max(1, window / max(1, config.spectralHopDivisor))
        spectral = spectral + spectralLossFFT(
          prediction,
          target,
          windowSize: window,
          lossMode: .l1,
          hop: hop,
          normalize: true
        )
      }
      total = total + spectral * (config.spectralWeight / Float(usableWindows.count))
    }

    if config.spectralLogmagWeight > 0, !usableWindows.isEmpty {
      var spectral = Signal.constant(0.0)
      for window in usableWindows {
        let hop = max(1, window / max(1, config.spectralHopDivisor))
        spectral = spectral + spectralLossFFT(
          prediction,
          target,
          windowSize: window,
          useLogMagnitude: true,
          lossMode: .l1,
          hop: hop,
          normalize: true
        )
      }
      total = total + spectral * (config.spectralLogmagWeight / Float(usableWindows.count))
    }

    return total
  }

  private static func makeConditioningData(
    chunk: CachedChunk,
    config: HarmonicE2EConfig,
    paddedFrames: Int
  ) -> [Float] {
    let minLogF0 = Foundation.log(max(config.minF0Hz, 1.0))
    let maxLogF0 = Foundation.log(max(config.maxF0Hz, config.minF0Hz + 1.0))
    let logSpan = max(1e-6, maxLogF0 - minLogF0)

    var rows: [Float] = []
    rows.reserveCapacity(paddedFrames * conditioningFeatureCount)
    for i in 0..<paddedFrames {
      if i < chunk.f0Hz.count {
        let uv = chunk.uvMask[i]
        let f0 = max(chunk.f0Hz[i], config.minF0Hz)
        let f0Norm = uv > 0.5 ? (Foundation.log(f0) - minLogF0) / logSpan : 0.0
        let loudNorm = min(max((chunk.loudnessDB[i] + 80.0) / 80.0, 0.0), 1.0)
        rows.append(Float(f0Norm))
        rows.append(loudNorm)
        rows.append(uv)
      } else {
        rows.append(0.0)
        rows.append(0.0)
        rows.append(0.0)
      }
    }
    return rows
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }

  private static func exportNoiseFilterControls(
    noiseFilter: DGenLazy.Tensor,
    step: Int,
    runDirs: RunDirectories,
    sampleRate: Float
  ) throws {
    let data = try noiseFilter.realize()
    guard noiseFilter.shape.count == 2 else { return }
    let frames = noiseFilter.shape[0]
    let taps = noiseFilter.shape[1]
    guard data.count == frames * taps else { return }

    var tapsCSV = "frame,tap,value\n"
    tapsCSV.reserveCapacity(max(64, data.count * 12))
    for frame in 0..<frames {
      let base = frame * taps
      for tap in 0..<taps {
        tapsCSV += "\(frame),\(tap),\(data[base + tap])\n"
      }
    }
    let tapsURL = runDirs.logs.appendingPathComponent(
      String(format: "noise_filter_step_%08d.csv", step)
    )
    try tapsCSV.write(to: tapsURL, atomically: true, encoding: .utf8)

    var summaryCSV = "frame,dc_gain,energy,centroid_hz,rolloff85_hz\n"
    let nyquistHz = Double(sampleRate) * 0.5
    let bins = 128
    for frame in 0..<frames {
      let base = frame * taps
      let frameTaps = Array(data[base..<(base + taps)])
      let response = frequencyResponseMagnitudes(taps: frameTaps, bins: bins)
      let energy = response.reduce(0.0, +)
      let dcGain = frameTaps.reduce(0.0, +)
      let centroidHz = spectralCentroidHz(response: response, nyquistHz: nyquistHz)
      let rolloffHz = spectralRolloffHz(response: response, nyquistHz: nyquistHz, threshold: 0.85)
      summaryCSV += "\(frame),\(dcGain),\(energy),\(centroidHz),\(rolloffHz)\n"
    }
    let summaryURL = runDirs.logs.appendingPathComponent(
      String(format: "noise_filter_summary_step_%08d.csv", step)
    )
    try summaryCSV.write(to: summaryURL, atomically: true, encoding: .utf8)
  }

  private static func frequencyResponseMagnitudes(taps: [Float], bins: Int) -> [Double] {
    guard bins > 0 else { return [] }
    var mags = [Double](repeating: 0.0, count: bins)
    let tapCount = taps.count
    for bin in 0..<bins {
      let omega = Double(bin) / Double(max(1, bins - 1)) * Double.pi
      var real = 0.0
      var imag = 0.0
      for n in 0..<tapCount {
        let x = Double(taps[n])
        let angle = -omega * Double(n)
        real += x * cos(angle)
        imag += x * sin(angle)
      }
      mags[bin] = sqrt(real * real + imag * imag)
    }
    return mags
  }

  private static func spectralCentroidHz(response: [Double], nyquistHz: Double) -> Double {
    let total = response.reduce(0.0, +)
    guard total > 0 else { return 0.0 }
    var weighted = 0.0
    for (i, mag) in response.enumerated() {
      let hz = Double(i) / Double(max(1, response.count - 1)) * nyquistHz
      weighted += hz * mag
    }
    return weighted / total
  }

  private static func spectralRolloffHz(
    response: [Double],
    nyquistHz: Double,
    threshold: Double
  ) -> Double {
    let total = response.reduce(0.0, +)
    guard total > 0 else { return 0.0 }
    let target = total * threshold
    var accum = 0.0
    for (i, mag) in response.enumerated() {
      accum += mag
      if accum >= target {
        return Double(i) / Double(max(1, response.count - 1)) * nyquistHz
      }
    }
    return nyquistHz
  }

  private static func format(_ value: Float) -> String {
    String(format: "%.6g", value)
  }
}

import DGenLazy
import Foundation

enum DDSPE2EBatchRenderer {
  private struct RenderManifestEntry: Codable {
    let batchIndex: Int
    let chunkID: String
    let sourceFile: String
    let wavPath: String
    let referenceChunkID: String?
    let referenceSourceFile: String?
    let referenceInstrumentIndex: Int?
  }

  private struct RenderManifest: Codable {
    let createdAtUTC: String
    let cache: String
    let checkpoint: String
    let split: String
    let batchSize: Int
    let configPath: String?
    let entries: [RenderManifestEntry]
  }

  static func run(options: [String: String], logger: (String) -> Void) throws {
    guard let cachePath = options["cache"] else {
      throw CLIError.invalid("render-checkpoint-batch requires --cache <cache-dir>")
    }
    guard let checkpointPath = options["init-checkpoint"] else {
      throw CLIError.invalid("render-checkpoint-batch requires --init-checkpoint <model-checkpoint-json>")
    }

    // If no config path is provided and checkpoint sits under runs/<name>/checkpoints/,
    // auto-resolve runs/<name>/resolved_config.json to avoid shape mismatches.
    let configPath = resolveConfigPath(rawConfigPath: options["config"], checkpointPath: checkpointPath)
    var config = try DDSPE2EConfig.load(path: configPath)
    try config.applyCLIOverrides(options)

    let split = DatasetSplit(rawValue: (options["split"] ?? "train").lowercased()) ?? .train
    let requestedBatchSize = max(1, Int(options["batch-size"] ?? "\(max(1, config.batchSize))") ?? max(1, config.batchSize))
    let instrumentOverride = options["instrument-index"].flatMap(Int.init)
    let referenceInstrumentOverride = options["reference-instrument-index"].flatMap(Int.init)
    let outputPath = options["output"] ?? "runs/batch_render_\(timestampString())"
    let outputDir = URL(fileURLWithPath: outputPath, isDirectory: true)
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cachePath))
    let splitEntries = dataset.entries(for: split)
    guard !splitEntries.isEmpty else {
      throw CLIError.invalid("No entries for split \(split.rawValue)")
    }

    var order = Array(splitEntries.indices)
    if config.shuffleChunks {
      var rng = SeededGenerator(seed: config.seed)
      order.shuffle(using: &rng)
    }

    let selectedEntries = (0..<requestedBatchSize).map { splitEntries[order[$0 % order.count]] }

    DGenConfig.backend = .metal
    DGenConfig.sampleRate = config.sampleRate
    DGenConfig.maxFrameCount = max(config.chunkSize, 1)
    DGenConfig.debug = false
    LazyGraphContext.reset()

    let model = DDSPDecoderModel(config: config)
    let conditioningFeatureCount = model.inputSize
    let checkpoint = try CheckpointStore.readModelState(from: URL(fileURLWithPath: checkpointPath))
    model.loadSnapshots(checkpoint.params)
    logger("Loaded model checkpoint: \(checkpointPath) (step=\(checkpoint.step))")

    var manifestEntries: [RenderManifestEntry] = []
    manifestEntries.reserveCapacity(selectedEntries.count)

    for (batchIndex, entry) in selectedEntries.enumerated() {
      let chunk = try dataset.loadChunk(entry)
      let selectedInstrument = instrumentOverride ?? chunk.instrumentIndex ?? 0
      logger(
        "Rendering \(entry.id) with instrumentIndex=\(selectedInstrument) "
          + "(source instrumentIndex=\(chunk.instrumentIndex ?? 0))")
      let frameCount = max(config.chunkSize, 1)
      let rawFeatureFrames = chunk.f0Hz.count
      let paddedFeatureFrames = ((rawFeatureFrames + 7) / 8) * 8

      let featuresTensor = Tensor(
        [[Float]](
          repeating: [Float](repeating: 0, count: conditioningFeatureCount),
          count: paddedFeatureFrames
        )
      )
      let modelFeatures = config.referenceConditioning
        ? FeatureExtractor.canonicalReferenceControls(
            features: ChunkFeatures(
              f0Hz: chunk.f0Hz, loudnessDB: chunk.loudnessDB, uvMask: chunk.uvMask),
            sourceFile: chunk.sourceFile)
        : ChunkFeatures(f0Hz: chunk.f0Hz, loudnessDB: chunk.loudnessDB, uvMask: chunk.uvMask)
      var conditioning = makeConditioningData(
        f0Hz: modelFeatures.f0Hz,
        loudnessDB: modelFeatures.loudnessDB,
        uvMask: modelFeatures.uvMask,
        instrumentIndex: selectedInstrument,
        numInstruments: config.referenceConditioning ? 1 : config.numInstruments
      )
      let condPad = paddedFeatureFrames * conditioningFeatureCount - conditioning.count
      if condPad > 0 {
        conditioning.append(contentsOf: [Float](repeating: 0, count: condPad))
      }
      featuresTensor.updateDataLazily(conditioning)

      var referenceTensor: Tensor? = nil
      var selectedReferenceEntry: CachedChunkEntry? = nil
      if config.referenceConditioning {
        let referenceEntry = splitEntries.first {
          ($0.instrumentIndex ?? 0) == (referenceInstrumentOverride ?? entry.instrumentIndex ?? 0)
            && $0.sourceFile != entry.sourceFile
        } ?? splitEntries.first {
          ($0.instrumentIndex ?? 0) == (referenceInstrumentOverride ?? entry.instrumentIndex ?? 0)
        }
        guard let referenceEntry else {
          throw CLIError.invalid("No matching reference chunk for \(entry.id)")
        }
        selectedReferenceEntry = referenceEntry
        let referenceChunk = try dataset.loadChunk(referenceEntry)
        var row = Array((referenceChunk.timbreFeatures ?? []).prefix(config.referenceFeatureSize))
        if row.count < config.referenceFeatureSize {
          row += [Float](repeating: 0, count: config.referenceFeatureSize - row.count)
        }
        referenceTensor = Tensor(Array(repeating: row, count: paddedFeatureFrames))
        logger(
          "  reference=\(referenceEntry.id) instrumentIndex=\(referenceEntry.instrumentIndex ?? 0)")
      }

      let synthTensors = DDSPSynth.PreallocatedTensors(
        featureFrames: paddedFeatureFrames,
        numHarmonics: config.numHarmonics
      )
      var paddedF0 = chunk.f0Hz
      var paddedUV = chunk.uvMask
      let framePad = paddedFeatureFrames - paddedF0.count
      if framePad > 0 {
        paddedF0.append(contentsOf: [Float](repeating: 0, count: framePad))
        paddedUV.append(contentsOf: [Float](repeating: 0, count: framePad))
      }
      synthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)

      let controls = model.forward(features: featuresTensor, reference: referenceTensor)
      let prediction = DDSPSynth.renderSignal(
        controls: controls,
        tensors: synthTensors,
        featureFrames: rawFeatureFrames,
        frameCount: frameCount,
        numHarmonics: config.numHarmonics,
        controlSmoothingMode: config.controlSmoothingMode,
          noiseSettings: config.noiseFilterSettings
      )
      let samples = try prediction.realize(frames: frameCount)
      LazyGraphContext.current.clearComputationGraph()

      let wavName = String(format: "batch_%02d_%@.wav", batchIndex, entry.id)
      let wavURL = outputDir.appendingPathComponent(wavName)
      try AudioFile.save(url: wavURL, samples: samples, sampleRate: config.sampleRate)
      logger("Rendered batch[\(batchIndex)] \(entry.id) -> \(wavURL.path)")

      manifestEntries.append(
        RenderManifestEntry(
          batchIndex: batchIndex,
          chunkID: entry.id,
          sourceFile: entry.sourceFile,
          wavPath: wavURL.path,
          referenceChunkID: selectedReferenceEntry?.id,
          referenceSourceFile: selectedReferenceEntry?.sourceFile,
          referenceInstrumentIndex: selectedReferenceEntry?.instrumentIndex
        )
      )
    }

    let manifest = RenderManifest(
      createdAtUTC: ISO8601DateFormatter().string(from: Date()),
      cache: cachePath,
      checkpoint: checkpointPath,
      split: split.rawValue,
      batchSize: requestedBatchSize,
      configPath: configPath,
      entries: manifestEntries
    )
    try writeJSON(manifest, to: outputDir.appendingPathComponent("render_manifest.json"))
    logger("render-checkpoint-batch complete -> \(outputDir.path)")
  }

  private static func resolveConfigPath(rawConfigPath: String?, checkpointPath: String) -> String? {
    if let rawConfigPath {
      return rawConfigPath
    }
    let checkpointURL = URL(fileURLWithPath: checkpointPath)
    let checkpointsDir = checkpointURL.deletingLastPathComponent()
    if checkpointsDir.lastPathComponent != "checkpoints" {
      return nil
    }
    let runRoot = checkpointsDir.deletingLastPathComponent()
    let resolved = runRoot.appendingPathComponent("resolved_config.json")
    if FileManager.default.fileExists(atPath: resolved.path) {
      return resolved.path
    }
    return nil
  }

  private static func makeConditioningData(
    f0Hz: [Float],
    loudnessDB: [Float],
    uvMask: [Float],
    instrumentIndex: Int?,
    numInstruments: Int
  ) -> [Float] {
    let n = min(f0Hz.count, min(loudnessDB.count, uvMask.count))
    let width = 5 + (numInstruments > 1 ? numInstruments : 0)
    if n == 0 { return [Float](repeating: 0, count: width) }
    var flat = [Float]()
    flat.reserveCapacity(n * width)
    var prevF0Norm: Float = 0
    var prevLoudNorm: Float = 0
    for i in 0..<n {
      let uv = min(1.0, max(0.0, uvMask[i]))
      let safeF0 = max(1.0, f0Hz[i])
      let f0Norm = log2(safeF0 / 440.0)
      let loudNorm = min(1.0, max(0.0, (loudnessDB[i] + 80.0) / 80.0))
      let deltaF0 = i == 0 ? 0 : (f0Norm - prevF0Norm)
      let deltaLoud = i == 0 ? 0 : (loudNorm - prevLoudNorm)
      flat.append(f0Norm)
      flat.append(loudNorm)
      flat.append(uv)
      flat.append(deltaF0)
      flat.append(deltaLoud)
      if numInstruments > 1 {
        let selected = min(max(0, instrumentIndex ?? 0), numInstruments - 1)
        for index in 0..<numInstruments { flat.append(index == selected ? 1 : 0) }
      }
      prevF0Norm = f0Norm
      prevLoudNorm = loudNorm
    }
    return flat
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }

  private static func timestampString() -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HHmmss"
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    return formatter.string(from: Date())
  }
}

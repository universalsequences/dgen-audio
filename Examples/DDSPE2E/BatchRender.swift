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
        switch config.referenceEncoderMode {
        case .averaged:
          var row = Array((referenceChunk.timbreFeatures ?? []).prefix(config.referenceFeatureSize))
          if row.count < config.referenceFeatureSize {
            row += [Float](repeating: 0, count: config.referenceFeatureSize - row.count)
          }
          referenceTensor = Tensor(Array(repeating: row, count: paddedFeatureFrames))
        case .temporal:
          let frames = referenceTimbreFrames(chunk: referenceChunk, config: config)
          referenceTensor = Tensor(frames).reshape(
            [config.referenceTimeFrames, config.referenceMelBins])
        }
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

  private struct TripletRenderEntry: Codable {
    let caseIndex: Int
    let role: String
    let wavPath: String
    let targetChunkID: String
    let targetSourceFile: String
    let targetInstrumentIndex: Int?
    let referenceChunkID: String?
    let referenceSourceFile: String?
    let referenceInstrumentIndex: Int?
    let referenceInstrumentName: String?
  }

  private struct TripletRenderManifest: Codable {
    let createdAtUTC: String
    let cache: String
    let checkpoint: String
    let split: String
    let configPath: String?
    let referenceEncoderMode: String
    let instrumentNames: [String]
    let entries: [TripletRenderEntry]
  }

  /// The R8 listening-gate artifact: for each held-out target, write the
  /// TARGET audio plus one prediction per instrument reference, rendered from
  /// identical f0/loudness controls. File names state exactly what each WAV
  /// is — predictions are named PREDICTED_USING_<INSTRUMENT>_REFERENCE, never
  /// "REFERENCE_<INSTRUMENT>".
  static func runReferenceTriplets(options: [String: String], logger: (String) -> Void) throws {
    guard let cachePath = options["cache"] else {
      throw CLIError.invalid("render-reference-triplets requires --cache <cache-dir>")
    }
    guard let checkpointPath = options["init-checkpoint"] else {
      throw CLIError.invalid(
        "render-reference-triplets requires --init-checkpoint <model-checkpoint-json>")
    }

    let configPath = resolveConfigPath(
      rawConfigPath: options["config"], checkpointPath: checkpointPath)
    var config = try DDSPE2EConfig.load(path: configPath)
    try config.applyCLIOverrides(options)
    guard config.referenceConditioning else {
      throw CLIError.invalid("render-reference-triplets requires a reference-conditioned config")
    }

    let split = DatasetSplit(rawValue: (options["split"] ?? "val").lowercased()) ?? .val
    let caseCount = max(1, Int(options["count"] ?? "6") ?? 6)
    let outputPath = options["output"] ?? "runs/reference_triplets_\(timestampString())"
    let outputDir = URL(fileURLWithPath: outputPath, isDirectory: true)
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cachePath))
    let instrumentNames = dataset.manifest.instrumentNames ?? ["default"]
    let splitEntries = dataset.entries(for: split)
    guard !splitEntries.isEmpty else {
      throw CLIError.invalid("No entries for split \(split.rawValue)")
    }

    // Source-distinct targets, interleaved across instruments so both timbres
    // appear even for small case counts.
    var seenSources = Set<String>()
    let sourceDistinct = splitEntries.filter { seenSources.insert($0.sourceFile).inserted }
    let byInstrument = Dictionary(grouping: sourceDistinct) { $0.instrumentIndex ?? 0 }
      .sorted { $0.key < $1.key }
    var targets = [CachedChunkEntry]()
    var cursor = 0
    while targets.count < caseCount {
      var appended = false
      for (_, entries) in byInstrument where cursor < entries.count && targets.count < caseCount {
        targets.append(entries[cursor])
        appended = true
      }
      if !appended { break }
      cursor += 1
    }

    // One reference chunk per source recording: the temporally middle chunk.
    let sourceReferenceEntries = Dictionary(grouping: splitEntries, by: \.sourceFile).values.map {
      entries -> CachedChunkEntry in
      let sorted = entries.sorted { $0.startSample < $1.startSample }
      return sorted[sorted.count / 2]
    }
    let referencesByInstrument = Dictionary(grouping: sourceReferenceEntries) {
      $0.instrumentIndex ?? 0
    }

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

    func sanitizedName(_ name: String) -> String {
      String(name.uppercased().map { $0.isLetter || $0.isNumber ? $0 : "_" })
    }

    var manifestEntries = [TripletRenderEntry]()

    for (caseIndex, entry) in targets.enumerated() {
      let chunk = try dataset.loadChunk(entry)
      let frameCount = max(config.chunkSize, 1)
      let rawFeatureFrames = chunk.f0Hz.count
      let paddedFeatureFrames = ((rawFeatureFrames + 7) / 8) * 8

      let targetName = String(format: "case%02d_%@_TARGET.wav", caseIndex, entry.id)
      let targetURL = outputDir.appendingPathComponent(targetName)
      try AudioFile.save(url: targetURL, samples: chunk.audio, sampleRate: config.sampleRate)
      manifestEntries.append(TripletRenderEntry(
        caseIndex: caseIndex, role: "TARGET", wavPath: targetURL.path,
        targetChunkID: entry.id, targetSourceFile: entry.sourceFile,
        targetInstrumentIndex: entry.instrumentIndex,
        referenceChunkID: nil, referenceSourceFile: nil,
        referenceInstrumentIndex: nil, referenceInstrumentName: nil))

      let modelFeatures = FeatureExtractor.canonicalReferenceControls(
        features: ChunkFeatures(
          f0Hz: chunk.f0Hz, loudnessDB: chunk.loudnessDB, uvMask: chunk.uvMask),
        sourceFile: chunk.sourceFile)
      var conditioning = makeConditioningData(
        f0Hz: modelFeatures.f0Hz,
        loudnessDB: modelFeatures.loudnessDB,
        uvMask: modelFeatures.uvMask,
        instrumentIndex: 0,
        numInstruments: 1)
      let condPad = paddedFeatureFrames * conditioningFeatureCount - conditioning.count
      if condPad > 0 {
        conditioning.append(contentsOf: [Float](repeating: 0, count: condPad))
      }

      var paddedF0 = chunk.f0Hz
      var paddedUV = chunk.uvMask
      let framePad = paddedFeatureFrames - paddedF0.count
      if framePad > 0 {
        paddedF0.append(contentsOf: [Float](repeating: 0, count: framePad))
        paddedUV.append(contentsOf: [Float](repeating: 0, count: framePad))
      }

      for instrumentIndex in 0..<max(1, instrumentNames.count) {
        let candidates = (referencesByInstrument[instrumentIndex] ?? [])
          .filter { $0.sourceFile != entry.sourceFile }
          .sorted { $0.id < $1.id }
        guard let referenceEntry = candidates.first ?? referencesByInstrument[instrumentIndex]?.first
        else {
          logger("case \(caseIndex): no reference for instrument \(instrumentIndex); skipping")
          continue
        }
        let referenceChunk = try dataset.loadChunk(referenceEntry)

        let featuresTensor = Tensor(
          [[Float]](
            repeating: [Float](repeating: 0, count: conditioningFeatureCount),
            count: paddedFeatureFrames))
        featuresTensor.updateDataLazily(conditioning)
        let referenceTensor: Tensor
        switch config.referenceEncoderMode {
        case .averaged:
          var row = Array(
            (referenceChunk.timbreFeatures ?? []).prefix(config.referenceFeatureSize))
          if row.count < config.referenceFeatureSize {
            row += [Float](repeating: 0, count: config.referenceFeatureSize - row.count)
          }
          referenceTensor = Tensor(Array(repeating: row, count: paddedFeatureFrames))
        case .temporal:
          referenceTensor = Tensor(referenceTimbreFrames(chunk: referenceChunk, config: config))
            .reshape([config.referenceTimeFrames, config.referenceMelBins])
        }

        let synthTensors = DDSPSynth.PreallocatedTensors(
          featureFrames: paddedFeatureFrames,
          numHarmonics: config.numHarmonics)
        synthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)

        let controls = model.forward(features: featuresTensor, reference: referenceTensor)
        let prediction = DDSPSynth.renderSignal(
          controls: controls,
          tensors: synthTensors,
          featureFrames: rawFeatureFrames,
          frameCount: frameCount,
          numHarmonics: config.numHarmonics,
          controlSmoothingMode: config.controlSmoothingMode,
          noiseSettings: config.noiseFilterSettings,
          reverbSettings: config.reverbSettings,
          reverbIR: model.reverbIR)
        let samples = try prediction.realize(frames: frameCount)
        LazyGraphContext.current.clearComputationGraph()

        let instrumentLabel = sanitizedName(
          instrumentNames.indices.contains(instrumentIndex)
            ? instrumentNames[instrumentIndex] : "inst\(instrumentIndex)")
        let role = "PREDICTED_USING_\(instrumentLabel)_REFERENCE"
        let wavName = String(format: "case%02d_%@_%@.wav", caseIndex, entry.id, role)
        let wavURL = outputDir.appendingPathComponent(wavName)
        try AudioFile.save(url: wavURL, samples: samples, sampleRate: config.sampleRate)
        logger(
          "case \(caseIndex) target=\(entry.id) (\(entry.sourceFile)) "
            + "reference=\(referenceEntry.id) (\(referenceEntry.sourceFile)) → \(wavName)")

        manifestEntries.append(TripletRenderEntry(
          caseIndex: caseIndex, role: role, wavPath: wavURL.path,
          targetChunkID: entry.id, targetSourceFile: entry.sourceFile,
          targetInstrumentIndex: entry.instrumentIndex,
          referenceChunkID: referenceEntry.id,
          referenceSourceFile: referenceEntry.sourceFile,
          referenceInstrumentIndex: referenceEntry.instrumentIndex,
          referenceInstrumentName: instrumentNames.indices.contains(instrumentIndex)
            ? instrumentNames[instrumentIndex] : nil))
      }
    }

    let manifest = TripletRenderManifest(
      createdAtUTC: ISO8601DateFormatter().string(from: Date()),
      cache: cachePath,
      checkpoint: checkpointPath,
      split: split.rawValue,
      configPath: configPath,
      referenceEncoderMode: config.referenceEncoderMode.rawValue,
      instrumentNames: instrumentNames,
      entries: manifestEntries)
    try writeJSON(manifest, to: outputDir.appendingPathComponent("triplet_manifest.json"))
    logger("render-reference-triplets complete -> \(outputDir.path)")
  }

  /// Diagnostic for reference-conditioning collapse: encodes references from
  /// the chosen split, then reports per-instrument latent separation and the
  /// auxiliary classifier's accuracy. Distinguishes "encoder collapsed" (z not
  /// instrument-separable) from "decoder ignores a good z".
  static func runReferenceZDebug(options: [String: String], logger: (String) -> Void) throws {
    guard let cachePath = options["cache"] else {
      throw CLIError.invalid("debug-reference-z requires --cache <cache-dir>")
    }
    guard let checkpointPath = options["init-checkpoint"] else {
      throw CLIError.invalid("debug-reference-z requires --init-checkpoint <model-checkpoint-json>")
    }
    let configPath = resolveConfigPath(
      rawConfigPath: options["config"], checkpointPath: checkpointPath)
    var config = try DDSPE2EConfig.load(path: configPath)
    try config.applyCLIOverrides(options)
    guard config.referenceConditioning, config.referenceEncoderMode == .temporal else {
      throw CLIError.invalid("debug-reference-z requires --reference-encoder temporal config")
    }
    let split = DatasetSplit(rawValue: (options["split"] ?? "val").lowercased()) ?? .val
    let perInstrument = max(2, Int(options["limit"] ?? "20") ?? 20)

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cachePath))
    let splitEntries = dataset.entries(for: split)
    let byInstrument = Dictionary(grouping: splitEntries) { $0.instrumentIndex ?? 0 }
      .sorted { $0.key < $1.key }

    // The encoder is a handful of tiny matmuls; the C backend realizes derived
    // tensors directly (the Metal materialize path cannot).
    DGenConfig.backend = .c
    DGenConfig.sampleRate = config.sampleRate
    DGenConfig.maxFrameCount = 1
    DGenConfig.debug = false
    LazyGraphContext.reset()

    let model = DDSPDecoderModel(config: config)
    let checkpoint = try CheckpointStore.readModelState(from: URL(fileURLWithPath: checkpointPath))
    model.loadSnapshots(checkpoint.params)
    logger("Loaded model checkpoint: \(checkpointPath) (step=\(checkpoint.step))")

    let classifierSnapshot = model.snapshots().first { $0.name == "ref_classifier_W" }
    let classifierBias = model.snapshots().first { $0.name == "ref_classifier_b" }

    var latents: [Int: [[Float]]] = [:]
    var rawFeatures: [Int: [[Float]]] = [:]
    var correctClass = 0
    var totalClass = 0
    for (instrument, entries) in byInstrument {
      var seenSources = Set<String>()
      let picks = entries.filter { seenSources.insert($0.sourceFile).inserted }
        .prefix(perInstrument)
      for entry in picks {
        let chunk = try dataset.loadChunk(entry)
        let frames = referenceTimbreFrames(chunk: chunk, config: config)
        rawFeatures[instrument, default: []].append(frames)
        if (options["raw-only"].map { ["1", "true", "yes"].contains($0.lowercased()) }) == true {
          latents[instrument, default: []].append([])
          continue
        }
        var rows = [[Float]]()
        for t in 0..<config.referenceTimeFrames {
          rows.append(
            Array(frames[(t * config.referenceMelBins)..<((t + 1) * config.referenceMelBins)]))
        }
        let referenceTensor = Tensor(rows)
        guard let z = model.encodeReferenceLatent(referenceTensor) else {
          throw CLIError.invalid("model has no temporal encoder")
        }
        // No graph clear between realizes: encoder graphs are tiny, and
        // clearing invalidates the realize path's tensor materialization.
        let zData = try z.realize()
        latents[instrument, default: []].append(zData)
        if let cw = classifierSnapshot, let cb = classifierBias,
          cw.shape.count == 2, cw.shape[0] == zData.count
        {
          let classes = cw.shape[1]
          var logits = [Float](repeating: 0, count: classes)
          for c in 0..<classes {
            var v: Float = cb.data.count > c ? cb.data[c] : 0
            for k in 0..<zData.count { v += zData[k] * cw.data[k * classes + c] }
            logits[c] = v
          }
          let predicted = logits.indices.max { logits[$0] < logits[$1] } ?? 0
          if predicted == instrument { correctClass += 1 }
          totalClass += 1
        }
      }
    }

    func mean(_ vectors: [[Float]]) -> [Float] {
      guard let first = vectors.first else { return [] }
      var m = [Float](repeating: 0, count: first.count)
      for v in vectors { for i in v.indices { m[i] += v[i] } }
      return m.map { $0 / Float(vectors.count) }
    }
    func distance(_ a: [Float], _ b: [Float]) -> Float {
      var s: Float = 0
      for i in a.indices { s += (a[i] - b[i]) * (a[i] - b[i]) }
      return Foundation.sqrt(s)
    }
    func meanPairwise(_ vectors: [[Float]]) -> Float {
      guard vectors.count > 1 else { return 0 }
      var total: Float = 0
      var pairs = 0
      for i in 0..<vectors.count {
        for j in (i + 1)..<vectors.count {
          total += distance(vectors[i], vectors[j])
          pairs += 1
        }
      }
      return total / Float(max(1, pairs))
    }

    let instruments = latents.keys.sorted()
    for instrument in instruments {
      let vectors = latents[instrument] ?? []
      let m = mean(vectors)
      let norm = Foundation.sqrt(m.reduce(Float(0)) { $0 + $1 * $1 })
      logger(
        "instrument \(instrument): n=\(vectors.count) |meanZ|=\(String(format: "%.4f", norm)) "
          + "withinDist=\(String(format: "%.4f", meanPairwise(vectors)))")
    }
    for i in 0..<instruments.count {
      for j in (i + 1)..<instruments.count {
        for (label, source) in [("z", latents), ("rawLogMel", rawFeatures)] {
          let a = mean(source[instruments[i]] ?? [])
          let b = mean(source[instruments[j]] ?? [])
          let between = distance(a, b)
          let within = (meanPairwise(source[instruments[i]] ?? [])
            + meanPairwise(source[instruments[j]] ?? [])) / 2
          logger(
            "\(label) instruments \(instruments[i]) vs \(instruments[j]): betweenMeans="
              + String(format: "%.4f", between)
              + " meanWithin=\(String(format: "%.4f", within))"
              + " separationRatio=\(String(format: "%.3f", between / max(within, 1e-6)))")
        }
      }
    }
    if totalClass > 0 {
      logger("classifier accuracy on z: \(correctClass)/\(totalClass)")
    }

    if let dumpPath = options["dump"] {
      struct Dump: Codable {
        let instrument: Int
        let rawLogMel: [Float]
        let z: [Float]
      }
      var rows = [Dump]()
      for instrument in instruments {
        let raws = rawFeatures[instrument] ?? []
        let zs = latents[instrument] ?? []
        for i in 0..<min(raws.count, zs.count) {
          rows.append(Dump(instrument: instrument, rawLogMel: raws[i], z: zs[i]))
        }
      }
      let data = try JSONEncoder().encode(rows)
      try data.write(to: URL(fileURLWithPath: dumpPath))
      logger("dumped \(rows.count) reference feature rows -> \(dumpPath)")
    }
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

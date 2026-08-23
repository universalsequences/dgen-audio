import DGenLazy
import Foundation

enum DatasetSplit: String, Codable {
  case train
  case val
}

struct CachedChunkEntry: Codable {
  var id: String
  var split: DatasetSplit
  var sourceFile: String
  var instrumentIndex: Int?
  var sourceSampleRate: Float
  var startSample: Int
  var chunkSamples: Int
  var featureFrames: Int
  var relativeChunkPath: String
}

struct CachedDatasetManifest: Codable {
  var version: Int
  var createdAtUTC: String
  var sourceRoot: String
  var config: DDSPE2EConfig
  var chunkCount: Int
  var trainCount: Int
  var valCount: Int
  var instrumentNames: [String]?
  var entries: [CachedChunkEntry]
}

struct CachedChunk: Codable {
  var id: String
  var sourceFile: String
  var instrumentIndex: Int?
  var sourceSampleRate: Float
  var sampleRate: Float
  var startSample: Int
  var audio: [Float]
  var f0Hz: [Float]
  var loudnessDB: [Float]
  var uvMask: [Float]
  /// Pitch-reduced audio descriptor for reference-conditioned timbre encoding.
  /// Optional so caches created before R8 remain readable.
  var timbreFeatures: [Float]?
  /// Time-varying log-mel reference frames, row-major
  /// [timbreFrameCount, timbreMelBins]. Optional so older caches remain
  /// readable; loaders fall back to computing frames from `audio`.
  var timbreFrames: [Float]?
  var timbreFrameCount: Int?
  var timbreMelBins: Int?
}

/// Returns [config.referenceTimeFrames × config.referenceMelBins] log-mel
/// frames for a reference chunk, preferring cached frames with matching
/// dimensions and computing them from the chunk audio otherwise (which keeps
/// pre-temporal caches usable without re-preprocessing).
func referenceTimbreFrames(chunk: CachedChunk, config: DDSPE2EConfig) -> [Float] {
  if let frames = chunk.timbreFrames,
    chunk.timbreFrameCount == config.referenceTimeFrames,
    chunk.timbreMelBins == config.referenceMelBins,
    frames.count == config.referenceTimeFrames * config.referenceMelBins
  {
    return frames
  }
  return FeatureExtractor.timbreLogMelFrames(
    samples: chunk.audio,
    sampleRate: chunk.sampleRate,
    frameSize: config.frameSize,
    frameHop: config.frameHop,
    timeFrames: config.referenceTimeFrames,
    melBins: config.referenceMelBins
  )
}

enum DatasetError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message):
      return "Dataset error: \(message)"
    }
  }
}

enum DatasetPreprocessor {
  static func preprocess(
    inputRoot: URL,
    cacheRoot: URL,
    config: DDSPE2EConfig,
    logger: (String) -> Void
  ) throws -> CachedDatasetManifest {
    let fm = FileManager.default

    let wavFiles = try findWavFiles(
      in: inputRoot,
      maxFiles: config.maxFiles,
      shuffle: config.shuffleChunks,
      seed: config.seed
    )
    if wavFiles.isEmpty {
      throw DatasetError.invalid("No .wav files found in \(inputRoot.path)")
    }

    try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    let chunksDir = cacheRoot.appendingPathComponent("chunks", isDirectory: true)
    try fm.createDirectory(at: chunksDir, withIntermediateDirectories: true)

    logger("Found \(wavFiles.count) wav files")

    let instrumentNames: [String]
    if config.labelByTopLevelDirectory {
      instrumentNames = Array(Set(wavFiles.map {
        topLevelComponent(of: relativePath(of: $0, root: inputRoot))
      })).sorted()
    } else {
      instrumentNames = ["default"]
    }
    let instrumentIndices = Dictionary(
      uniqueKeysWithValues: instrumentNames.enumerated().map { ($0.element, $0.offset) })
    if config.labelByTopLevelDirectory {
      let labels = instrumentNames.enumerated().map { "\($0.offset)=\($0.element)" }.joined(separator: ", ")
      logger("Instrument labels: \(labels)")
    }

    var datasetGain: Float?
    if config.normalizeAcrossDataset, config.peakNormalizeTo > 0 {
      var datasetPeak: Float = 0
      for fileURL in wavFiles {
        let (samples, _) = try AudioFile.load(url: fileURL, mono: true)
        for sample in samples { datasetPeak = max(datasetPeak, abs(sample)) }
      }
      if datasetPeak > 0 {
        datasetGain = config.peakNormalizeTo / datasetPeak
        logger("Global peak normalization: peak=\(datasetPeak) gain=\(datasetGain!)")
      }
    }

    var entries = [CachedChunkEntry]()
    entries.reserveCapacity(wavFiles.count * 8)

    var chunkIndex = 0

    for (fileIndex, fileURL) in wavFiles.enumerated() {
      let relative = relativePath(of: fileURL, root: inputRoot)
      let instrumentName = config.labelByTopLevelDirectory
        ? topLevelComponent(of: relative) : instrumentNames[0]
      let instrumentIndex = instrumentIndices[instrumentName] ?? 0
      logger("[\(fileIndex + 1)/\(wavFiles.count)] \(relative)")

      let (rawSamples, sourceRate) = try AudioFile.load(url: fileURL, mono: true)
      var samples = resampleLinear(
        samples: rawSamples,
        sourceRate: sourceRate,
        targetRate: config.sampleRate
      )
      if let datasetGain {
        samples = samples.map { $0 * datasetGain }
      } else {
        samples = normalizePeak(samples, peakTarget: config.peakNormalizeTo)
      }

      let starts = chunkStartIndices(
        sampleCount: samples.count,
        chunkSize: config.chunkSize,
        chunkHop: config.chunkHop
      )

      var chunksWrittenForFile = 0
      for start in starts {
        if let maxChunks = config.maxChunksPerFile, chunksWrittenForFile >= maxChunks {
          break
        }

        let end = min(samples.count, start + config.chunkSize)
        var chunk = Array(samples[start..<end])
        if chunk.count < config.chunkSize {
          chunk += [Float](repeating: 0, count: config.chunkSize - chunk.count)
        }

        let features = FeatureExtractor.extract(
          samples: chunk,
          sampleRate: config.sampleRate,
          config: config
        )

        let timbreFeatures = FeatureExtractor.timbreDescriptor(
          samples: chunk,
          sampleRate: config.sampleRate,
          features: features,
          frameSize: config.frameSize,
          frameHop: config.frameHop,
          count: config.referenceFeatureSize
        )

        let timbreFrames = FeatureExtractor.timbreLogMelFrames(
          samples: chunk,
          sampleRate: config.sampleRate,
          frameSize: config.frameSize,
          frameHop: config.frameHop,
          timeFrames: config.referenceTimeFrames,
          melBins: config.referenceMelBins
        )

        let id = String(format: "chunk_%08d", chunkIndex)
        let chunkFileName = "\(id).json"
        let chunkFileURL = chunksDir.appendingPathComponent(chunkFileName)
        let relativeChunkPath = "chunks/\(chunkFileName)"

        let chunkRecord = CachedChunk(
          id: id,
          sourceFile: relative,
          instrumentIndex: instrumentIndex,
          sourceSampleRate: sourceRate,
          sampleRate: config.sampleRate,
          startSample: start,
          audio: chunk,
          f0Hz: features.f0Hz,
          loudnessDB: features.loudnessDB,
          uvMask: features.uvMask,
          timbreFeatures: timbreFeatures,
          timbreFrames: timbreFrames,
          timbreFrameCount: config.referenceTimeFrames,
          timbreMelBins: config.referenceMelBins
        )

        try writeJSON(chunkRecord, to: chunkFileURL)

        let entry = CachedChunkEntry(
          id: id,
          split: .train,
          sourceFile: relative,
          instrumentIndex: instrumentIndex,
          sourceSampleRate: sourceRate,
          startSample: start,
          chunkSamples: chunk.count,
          featureFrames: features.f0Hz.count,
          relativeChunkPath: relativeChunkPath
        )
        entries.append(entry)

        chunkIndex += 1
        chunksWrittenForFile += 1
      }
    }

    if entries.isEmpty {
      throw DatasetError.invalid("No chunks were generated. Check chunk size/hop and audio length")
    }

    assignSplits(
      entries: &entries,
      trainSplit: config.trainSplit,
      seed: config.seed,
      shuffle: config.shuffleChunks,
      groupBySourceFile: config.splitBySourceFile
    )

    let trainCount = entries.filter { $0.split == .train }.count
    let valCount = entries.count - trainCount

    let manifest = CachedDatasetManifest(
      version: 1,
      createdAtUTC: iso8601Now(),
      sourceRoot: inputRoot.path,
      config: config,
      chunkCount: entries.count,
      trainCount: trainCount,
      valCount: valCount,
      instrumentNames: instrumentNames,
      entries: entries
    )

    let manifestURL = cacheRoot.appendingPathComponent("manifest.json")
    try writeJSON(manifest, to: manifestURL)

    logger("Wrote manifest: \(manifestURL.path)")
    logger("Chunks: total=\(entries.count) train=\(trainCount) val=\(valCount)")
    if config.splitBySourceFile {
      let trainSources = Set(entries.filter { $0.split == .train }.map(\.sourceFile)).count
      let valSources = Set(entries.filter { $0.split == .val }.map(\.sourceFile)).count
      logger("Source-grouped split: trainFiles=\(trainSources) valFiles=\(valSources)")
    }

    return manifest
  }

  private static func findWavFiles(
    in root: URL,
    maxFiles: Int?,
    shuffle: Bool,
    seed: UInt64
  ) throws -> [URL] {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(
      at: root,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    ) else {
      throw DatasetError.invalid("Could not enumerate \(root.path)")
    }

    var files = [URL]()
    for case let url as URL in enumerator {
      if url.pathExtension.lowercased() == "wav" {
        files.append(url)
      }
    }

    files.sort { $0.path < $1.path }

    if shuffle {
      var rng = SeededGenerator(seed: seed)
      files.shuffle(using: &rng)
    }

    if let maxFiles, files.count > maxFiles {
      files = Array(files.prefix(maxFiles))
    }

    return files
  }

  private static func chunkStartIndices(sampleCount: Int, chunkSize: Int, chunkHop: Int) -> [Int] {
    if sampleCount <= 0 {
      return []
    }

    if sampleCount <= chunkSize {
      return [0]
    }

    var starts = [Int]()
    var start = 0
    while start + chunkSize <= sampleCount {
      starts.append(start)
      start += chunkHop
    }

    if let last = starts.last, last + chunkSize < sampleCount {
      starts.append(max(0, sampleCount - chunkSize))
    }

    return starts
  }

  private static func assignSplits(
    entries: inout [CachedChunkEntry],
    trainSplit: Float,
    seed: UInt64,
    shuffle: Bool,
    groupBySourceFile: Bool
  ) {
    guard !entries.isEmpty else { return }

    if groupBySourceFile {
      var sources = Array(Set(entries.map(\.sourceFile)).sorted())
      if shuffle {
        var rng = SeededGenerator(seed: seed)
        sources.shuffle(using: &rng)
      }

      var trainSourceCount = Int((Float(sources.count) * trainSplit).rounded(.down))
      trainSourceCount = max(1, min(sources.count - 1, trainSourceCount))
      if sources.count == 1 { trainSourceCount = 1 }
      let trainSources = Set(sources.prefix(trainSourceCount))

      for index in entries.indices {
        entries[index].split = trainSources.contains(entries[index].sourceFile) ? .train : .val
      }
      return
    }

    var indices = Array(entries.indices)
    if shuffle {
      var rng = SeededGenerator(seed: seed)
      indices.shuffle(using: &rng)
    }

    var trainCount = Int((Float(entries.count) * trainSplit).rounded(.down))
    trainCount = max(1, min(entries.count - 1, trainCount))
    if entries.count == 1 { trainCount = 1 }

    for (position, index) in indices.enumerated() {
      entries[index].split = position < trainCount ? .train : .val
    }
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }

  private static func topLevelComponent(of relativePath: String) -> String {
    relativePath.split(separator: "/").first.map(String.init) ?? "default"
  }

  private static func relativePath(of url: URL, root: URL) -> String {
    let rootPath = root.standardizedFileURL.path
    let fullPath = url.standardizedFileURL.path
    if fullPath.hasPrefix(rootPath) {
      let start = fullPath.index(fullPath.startIndex, offsetBy: rootPath.count)
      let sliced = fullPath[start...]
      return sliced.hasPrefix("/") ? String(sliced.dropFirst()) : String(sliced)
    }
    return url.lastPathComponent
  }

  private static func resampleLinear(samples: [Float], sourceRate: Float, targetRate: Float) -> [Float] {
    if samples.isEmpty { return [] }
    if abs(sourceRate - targetRate) < 0.5 { return samples }

    let ratio = targetRate / sourceRate
    let outputCount = max(1, Int((Float(samples.count) * ratio).rounded()))
    var out = [Float](repeating: 0, count: outputCount)

    for i in 0..<outputCount {
      let sourcePos = Float(i) / ratio
      let i0 = Int(floor(sourcePos))
      let i1 = min(i0 + 1, samples.count - 1)
      let frac = sourcePos - Float(i0)
      let a = samples[min(max(i0, 0), samples.count - 1)]
      let b = samples[i1]
      out[i] = a + (b - a) * frac
    }

    return out
  }

  private static func normalizePeak(_ samples: [Float], peakTarget: Float) -> [Float] {
    guard peakTarget > 0 else { return samples }
    guard let peak = samples.map({ abs($0) }).max(), peak > 0 else {
      return samples
    }
    let gain = peakTarget / peak
    return samples.map { $0 * gain }
  }

  private static func iso8601Now() -> String {
    ISO8601DateFormatter().string(from: Date())
  }
}

struct CachedDataset {
  let root: URL
  let manifest: CachedDatasetManifest

  static func load(from cacheRoot: URL) throws -> CachedDataset {
    let manifestURL = cacheRoot.appendingPathComponent("manifest.json")
    let data = try Data(contentsOf: manifestURL)
    let manifest = try JSONDecoder().decode(CachedDatasetManifest.self, from: data)
    return CachedDataset(root: cacheRoot, manifest: manifest)
  }

  var trainEntries: [CachedChunkEntry] {
    manifest.entries.filter { $0.split == .train }
  }

  var valEntries: [CachedChunkEntry] {
    manifest.entries.filter { $0.split == .val }
  }

  func entries(for split: DatasetSplit?) -> [CachedChunkEntry] {
    guard let split else { return manifest.entries }
    return manifest.entries.filter { $0.split == split }
  }

  func loadChunk(_ entry: CachedChunkEntry) throws -> CachedChunk {
    let url = root.appendingPathComponent(entry.relativeChunkPath)
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(CachedChunk.self, from: data)
  }
}

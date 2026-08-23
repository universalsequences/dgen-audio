import DGenLazy
import Foundation

enum DDSPE2ETransfer {
  private struct TransferManifest: Codable {
    let createdAtUTC: String
    let input: String
    let checkpoint: String
    let configPath: String?
    let output: String
    let sampleRate: Float
    let inputSamples: Int
    let chunks: Int
    let transposeSemitones: Float
    let loudnessOffsetDB: Float
    let instrumentIndex: Int
    let stabilizeF0: Bool
    let voicedFrameRatio: Float
    let medianInputF0Hz: Float
    let medianOutputF0Hz: Float
  }

  static func run(options: [String: String], logger: (String) -> Void) throws {
    guard let inputPath = options["input"] else {
      throw CLIError.invalid("transfer requires --input <voice.wav>")
    }
    guard let checkpointPath = options["init-checkpoint"] else {
      throw CLIError.invalid("transfer requires --init-checkpoint <model-checkpoint-json>")
    }

    let configPath = resolveConfigPath(
      rawConfigPath: options["config"], checkpointPath: checkpointPath)
    var config = try DDSPE2EConfig.load(path: configPath)
    try config.applyCLIOverrides(options)

    let transpose = try parseFloat(options["transpose"] ?? "12", key: "transpose")
    let loudnessOffset = try parseFloat(
      options["loudness-offset-db"] ?? "0", key: "loudness-offset-db")
    let stabilizePitch = parseBool(options["stabilize-f0"] ?? "true")
    let instrumentIndex = min(
      max(0, Int(options["instrument-index"] ?? "0") ?? 0), config.numInstruments - 1)
    let outputDir = URL(
      fileURLWithPath: options["output"] ?? "runs/voice_to_flute", isDirectory: true)
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let inputURL = URL(fileURLWithPath: inputPath)
    let (rawAudio, sourceRate) = try AudioFile.load(url: inputURL, mono: true)
    let audio = resampleLinear(samples: rawAudio, sourceRate: sourceRate, targetRate: config.sampleRate)
    guard !audio.isEmpty else { throw CLIError.invalid("Input audio is empty") }

    DGenConfig.backend = .metal
    DGenConfig.sampleRate = config.sampleRate
    DGenConfig.maxFrameCount = max(config.chunkSize, 1)
    DGenConfig.debug = false
    LazyGraphContext.reset()

    let model = DDSPDecoderModel(config: config)
    let conditioningFeatureCount = model.inputSize
    let checkpoint = try CheckpointStore.readModelState(from: URL(fileURLWithPath: checkpointPath))
    model.loadSnapshots(checkpoint.params)
    logger("Loaded flute checkpoint: \(checkpointPath) (step=\(checkpoint.step))")

    let starts = chunkStarts(
      sampleCount: audio.count, chunkSize: config.chunkSize, hop: config.chunkHop)
    var mixed = [Float](repeating: 0, count: max(audio.count, starts.last! + config.chunkSize))
    var weights = [Float](repeating: 0, count: mixed.count)
    var inputPitches = [Float]()
    var outputPitches = [Float]()
    var voicedFrames = 0
    var totalFrames = 0
    let pitchScale = pow(2.0 as Float, transpose / 12.0)

    for (chunkIndex, start) in starts.enumerated() {
      var chunkAudio = [Float](repeating: 0, count: config.chunkSize)
      let available = min(config.chunkSize, max(0, audio.count - start))
      if available > 0 {
        chunkAudio.replaceSubrange(0..<available, with: audio[start..<(start + available)])
      }

      var features = FeatureExtractor.extract(
        samples: chunkAudio, sampleRate: config.sampleRate, config: config)
      if stabilizePitch {
        features.f0Hz = stabilizeF0(features.f0Hz, uvMask: features.uvMask)
      }
      totalFrames += features.uvMask.count
      for i in features.f0Hz.indices {
        features.loudnessDB[i] += loudnessOffset
        if features.uvMask[i] > 0.5, features.f0Hz[i] > 0 {
          inputPitches.append(features.f0Hz[i])
          features.f0Hz[i] = min(config.sampleRate * 0.49, features.f0Hz[i] * pitchScale)
          outputPitches.append(features.f0Hz[i])
          voicedFrames += 1
        }
      }

      let rawFeatureFrames = features.f0Hz.count
      let paddedFeatureFrames = ((rawFeatureFrames + 7) / 8) * 8
      let featuresTensor = Tensor(
        [[Float]](
          repeating: [Float](repeating: 0, count: conditioningFeatureCount),
          count: paddedFeatureFrames
        ))
      var conditioning = makeConditioningData(
        f0Hz: features.f0Hz,
        loudnessDB: features.loudnessDB,
        uvMask: features.uvMask,
        instrumentIndex: instrumentIndex,
        numInstruments: config.numInstruments)
      let conditioningPadding = paddedFeatureFrames * conditioningFeatureCount - conditioning.count
      if conditioningPadding > 0 {
        conditioning.append(contentsOf: [Float](repeating: 0, count: conditioningPadding))
      }
      featuresTensor.updateDataLazily(conditioning)

      let synthTensors = DDSPSynth.PreallocatedTensors(
        featureFrames: paddedFeatureFrames, numHarmonics: config.numHarmonics)
      var paddedF0 = features.f0Hz
      var paddedUV = features.uvMask
      let framePadding = paddedFeatureFrames - rawFeatureFrames
      if framePadding > 0 {
        paddedF0.append(contentsOf: [Float](repeating: 0, count: framePadding))
        paddedUV.append(contentsOf: [Float](repeating: 0, count: framePadding))
      }
      synthTensors.updateChunkData(f0Frames: paddedF0, uvFrames: paddedUV)

      let controls = model.forward(features: featuresTensor)
      let prediction = DDSPSynth.renderSignal(
        controls: controls,
        tensors: synthTensors,
        featureFrames: rawFeatureFrames,
        frameCount: config.chunkSize,
        numHarmonics: config.numHarmonics,
        controlSmoothingMode: config.controlSmoothingMode,
        noiseSettings: config.noiseFilterSettings,
        reverbSettings: config.reverbSettings,
        reverbIR: model.reverbIR
      )
      let samples = try prediction.realize(frames: config.chunkSize)
      LazyGraphContext.current.clearComputationGraph()

      for i in 0..<min(config.chunkSize, samples.count) {
        let position = start + i
        let weight = overlapWindow(
          index: i,
          count: config.chunkSize,
          isFirst: chunkIndex == 0,
          isLast: chunkIndex == starts.count - 1
        )
        mixed[position] += samples[i] * weight
        weights[position] += weight
      }
      logger("Rendered transfer chunk \(chunkIndex + 1)/\(starts.count)")
    }

    var output = Array(mixed.prefix(audio.count))
    for i in output.indices where weights[i] > 1e-6 { output[i] /= weights[i] }

    let base = inputURL.deletingPathExtension().lastPathComponent
    let outputURL = outputDir.appendingPathComponent("\(base)_flute.wav")
    let inputCopyURL = outputDir.appendingPathComponent("\(base)_input_16k.wav")
    try AudioFile.save(url: outputURL, samples: output, sampleRate: config.sampleRate)
    try AudioFile.save(url: inputCopyURL, samples: audio, sampleRate: config.sampleRate)

    let manifest = TransferManifest(
      createdAtUTC: ISO8601DateFormatter().string(from: Date()),
      input: inputURL.path,
      checkpoint: checkpointPath,
      configPath: configPath,
      output: outputURL.path,
      sampleRate: config.sampleRate,
      inputSamples: audio.count,
      chunks: starts.count,
      transposeSemitones: transpose,
      loudnessOffsetDB: loudnessOffset,
      instrumentIndex: instrumentIndex,
      stabilizeF0: stabilizePitch,
      voicedFrameRatio: Float(voicedFrames) / Float(max(1, totalFrames)),
      medianInputF0Hz: median(inputPitches),
      medianOutputF0Hz: median(outputPitches)
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(manifest).write(to: outputDir.appendingPathComponent("transfer_manifest.json"))
    logger("Voice-to-flute render complete: \(outputURL.path)")
  }

  private static func resolveConfigPath(rawConfigPath: String?, checkpointPath: String) -> String? {
    if let rawConfigPath { return rawConfigPath }
    let checkpointURL = URL(fileURLWithPath: checkpointPath)
    let checkpointsDir = checkpointURL.deletingLastPathComponent()
    guard checkpointsDir.lastPathComponent == "checkpoints" else { return nil }
    let resolved = checkpointsDir.deletingLastPathComponent().appendingPathComponent("resolved_config.json")
    return FileManager.default.fileExists(atPath: resolved.path) ? resolved.path : nil
  }

  private static func chunkStarts(sampleCount: Int, chunkSize: Int, hop: Int) -> [Int] {
    guard sampleCount > chunkSize else { return [0] }
    var starts = Array(stride(from: 0, through: sampleCount - chunkSize, by: hop))
    let finalStart = sampleCount - chunkSize
    if starts.last != finalStart { starts.append(finalStart) }
    return starts
  }

  private static func overlapWindow(index: Int, count: Int, isFirst: Bool, isLast: Bool) -> Float {
    let half = max(1, count / 2)
    if index < half {
      if isFirst { return 1 }
      let phase = Float(index) / Float(half)
      return 0.5 - 0.5 * cos(Float.pi * phase)
    }
    if isLast { return 1 }
    let phase = Float(index - half) / Float(max(1, count - half))
    return 0.5 + 0.5 * cos(Float.pi * phase)
  }

  /// Remove isolated autocorrelation jumps without low-pass smoothing, then
  /// fill unvoiced f0 from the nearest voiced frame. The latter is important:
  /// the synth linearly samples frame controls, so a literal 0 Hz beside a
  /// voiced frame creates an audible 0→f0 chirp while the UV mask fades in.
  private static func stabilizeF0(_ f0Hz: [Float], uvMask: [Float]) -> [Float] {
    guard !f0Hz.isEmpty else { return f0Hz }
    var result = f0Hz

    // A centered median only replaces substantial one-frame pitch outliers;
    // unlike a one-pole filter it preserves genuine note-change edges.
    if f0Hz.count >= 3 {
      for i in 1..<(f0Hz.count - 1) where uvMask[i] > 0.5 {
        let voicedNeighbors = (i - 1...i + 1).compactMap { j -> Float? in
          uvMask[j] > 0.5 && f0Hz[j] > 0 ? log2(f0Hz[j]) : nil
        }
        guard voicedNeighbors.count == 3 else { continue }
        let medianLog = voicedNeighbors.sorted()[1]
        let centsFromMedian = abs((log2(f0Hz[i]) - medianLog) * 1200)
        if centsFromMedian > 250 { result[i] = pow(2, medianLog) }
      }
    }

    let voicedIndices = result.indices.filter { uvMask[$0] > 0.5 && result[$0] > 0 }
    guard !voicedIndices.isEmpty else { return result }
    var previous = [Int?](repeating: nil, count: result.count)
    var next = [Int?](repeating: nil, count: result.count)
    var last: Int?
    for i in result.indices {
      if uvMask[i] > 0.5, result[i] > 0 { last = i }
      previous[i] = last
    }
    last = nil
    for i in result.indices.reversed() {
      if uvMask[i] > 0.5, result[i] > 0 { last = i }
      next[i] = last
    }
    for i in result.indices where uvMask[i] <= 0.5 || result[i] <= 0 {
      switch (previous[i], next[i]) {
      case let (left?, right?): result[i] = i - left <= right - i ? result[left] : result[right]
      case let (left?, nil): result[i] = result[left]
      case let (nil, right?): result[i] = result[right]
      case (nil, nil): break
      }
    }
    return result
  }

  private static func makeConditioningData(
    f0Hz: [Float],
    loudnessDB: [Float],
    uvMask: [Float],
    instrumentIndex: Int,
    numInstruments: Int
  ) -> [Float] {
    let n = min(f0Hz.count, min(loudnessDB.count, uvMask.count))
    let width = 5 + (numInstruments > 1 ? numInstruments : 0)
    var flat = [Float]()
    flat.reserveCapacity(n * width)
    var previousF0: Float = 0
    var previousLoudness: Float = 0
    for i in 0..<n {
      let uv = min(1, max(0, uvMask[i]))
      let f0 = log2(max(1, f0Hz[i]) / 440)
      let loudness = min(1, max(0, (loudnessDB[i] + 80) / 80))
      flat.append(f0)
      flat.append(loudness)
      flat.append(uv)
      flat.append(i == 0 ? 0 : f0 - previousF0)
      flat.append(i == 0 ? 0 : loudness - previousLoudness)
      if numInstruments > 1 {
        for index in 0..<numInstruments { flat.append(index == instrumentIndex ? 1 : 0) }
      }
      previousF0 = f0
      previousLoudness = loudness
    }
    return flat
  }

  private static func resampleLinear(
    samples: [Float], sourceRate: Float, targetRate: Float
  ) -> [Float] {
    if samples.isEmpty || abs(sourceRate - targetRate) < 0.5 { return samples }
    let ratio = targetRate / sourceRate
    let count = max(1, Int((Float(samples.count) * ratio).rounded()))
    return (0..<count).map { i in
      let position = Float(i) / ratio
      let lower = min(samples.count - 1, max(0, Int(position)))
      let upper = min(samples.count - 1, lower + 1)
      let fraction = position - Float(lower)
      return samples[lower] + (samples[upper] - samples[lower]) * fraction
    }
  }

  private static func median(_ values: [Float]) -> Float {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
  }

  private static func parseBool(_ value: String) -> Bool {
    ["1", "true", "yes", "on"].contains(value.lowercased())
  }

  private static func parseFloat(_ value: String, key: String) throws -> Float {
    guard let parsed = Float(value) else {
      throw CLIError.invalid("Invalid float for --\(key): \(value)")
    }
    return parsed
  }
}

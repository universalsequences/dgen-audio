import DGenLazy
import Foundation

enum DDSPE2ESmoothingProbe {
  private static let firTaps: [Float] = [0.1, 0.2, 0.4, 0.2, 0.1]
  private static let conditioningFeatureCount = 5

  static func run(options: [String: String], logger: (String) -> Void) throws {
    guard let cache = options["cache"] else {
      throw CLIError.invalid("probe-smoothing requires --cache <cache-dir>")
    }

    var config = try DDSPE2EConfig.load(path: options["config"])
    try config.applyCLIOverrides(options)

    let split = DatasetSplit(rawValue: (options["split"] ?? "train").lowercased()) ?? .train
    let index = max(0, Int(options["index"] ?? "0") ?? 0)
    let maxFrames = max(0, Int(options["max-frames"] ?? "0") ?? 0)
    let outputPath = options["output"] ?? "/tmp/ddsp_smoothing_probe"
    let outputDir = URL(fileURLWithPath: outputPath, isDirectory: true)

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))
    let entries = dataset.entries(for: split)
    guard !entries.isEmpty else {
      throw CLIError.invalid("No entries for split \(split.rawValue)")
    }

    let entry = entries[index % entries.count]
    let chunk = try dataset.loadChunk(entry)

    let allFrames = min(chunk.f0Hz.count, min(chunk.loudnessDB.count, chunk.uvMask.count))
    let frameCount = maxFrames > 0 ? min(maxFrames, allFrames) : allFrames
    guard frameCount > 0 else {
      throw CLIError.invalid("Selected chunk has no valid frames")
    }

    let conditioning = makeConditioningData(
      f0Hz: Array(chunk.f0Hz.prefix(frameCount)),
      loudnessDB: Array(chunk.loudnessDB.prefix(frameCount)),
      uvMask: Array(chunk.uvMask.prefix(frameCount))
    )
    let features = Tensor(
      [[Float]](repeating: [Float](repeating: 0, count: conditioningFeatureCount), count: frameCount)
    )
    features.updateDataLazily(conditioning)

    let model = DDSPDecoderModel(config: config)
    if let checkpointPath = options["init-checkpoint"] {
      let checkpoint = try CheckpointStore.readModelState(from: URL(fileURLWithPath: checkpointPath))
      model.loadSnapshots(checkpoint.params)
      logger("Loaded model checkpoint for probe: \(checkpointPath) (step=\(checkpoint.step))")
    }

    let controls = model.forward(features: features)
    let K = config.numHarmonics

    let ampsRaw = controls.harmonicAmps.reshape([frameCount, K])
    let hGainRaw = controls.harmonicGain.reshape([frameCount, 1])
    let nGainRaw = controls.noiseGain.reshape([frameCount, 1])

    let ampsFir = firSmoothFrames2D(ampsRaw)
    let hGainFir = firSmoothFrames2D(hGainRaw)
    let nGainFir = firSmoothFrames2D(nGainRaw)

    let ampsRawData = try ampsRaw.realize()
    let hGainRawData = try hGainRaw.realize()
    let nGainRawData = try nGainRaw.realize()
    let ampsFirData = try ampsFir.realize()
    let hGainFirData = try hGainFir.realize()
    let nGainFirData = try nGainFir.realize()

    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    try writeGainCSV(
      path: outputDir.appendingPathComponent("harmonic_gain_compare.csv"),
      raw: hGainRawData,
      fir: hGainFirData,
      frames: frameCount
    )
    try writeGainCSV(
      path: outputDir.appendingPathComponent("noise_gain_compare.csv"),
      raw: nGainRawData,
      fir: nGainFirData,
      frames: frameCount
    )
    try writeAmpSumCSV(
      path: outputDir.appendingPathComponent("amp_sum_compare.csv"),
      raw: ampsRawData,
      fir: ampsFirData,
      frames: frameCount,
      harmonics: K
    )
    try writeAmpsLongCSV(
      path: outputDir.appendingPathComponent("harmonic_amps_compare.csv"),
      raw: ampsRawData,
      fir: ampsFirData,
      frames: frameCount,
      harmonics: K
    )

    let analysis = SmoothingAnalysis(
      chunkID: entry.id,
      sourceFile: entry.sourceFile,
      split: split.rawValue,
      frames: frameCount,
      harmonics: K,
      harmonicGainRawTV: meanAbsFrameDiff(hGainRawData, rows: frameCount, cols: 1),
      harmonicGainFirTV: meanAbsFrameDiff(hGainFirData, rows: frameCount, cols: 1),
      noiseGainRawTV: meanAbsFrameDiff(nGainRawData, rows: frameCount, cols: 1),
      noiseGainFirTV: meanAbsFrameDiff(nGainFirData, rows: frameCount, cols: 1),
      ampSumRawTV: meanAbsFrameDiff(ampSum(ampsRawData, rows: frameCount, cols: K), rows: frameCount, cols: 1),
      ampSumFirTV: meanAbsFrameDiff(ampSum(ampsFirData, rows: frameCount, cols: K), rows: frameCount, cols: 1)
    )
    try writeJSON(analysis, to: outputDir.appendingPathComponent("analysis.json"))

    logger("probe-smoothing wrote realized controls to \(outputDir.path)")
    logger(
      "TV raw->fir (harmonicGain): \(fmt(analysis.harmonicGainRawTV)) -> \(fmt(analysis.harmonicGainFirTV)); "
        + "noiseGain: \(fmt(analysis.noiseGainRawTV)) -> \(fmt(analysis.noiseGainFirTV)); "
        + "ampSum: \(fmt(analysis.ampSumRawTV)) -> \(fmt(analysis.ampSumFirTV))"
    )
  }

  private static func firSmoothFrames2D(_ x: Tensor) -> Tensor {
    guard x.shape.count == 2 else { return x }
    let kernel = Tensor(firTaps).reshape([firTaps.count, 1])
    let k = kernel.shape[0]
    guard k > 1 else { return x }
    let left = (k - 1) / 2
    let right = k - 1 - left
    let padded = replicatePadRows2D(x, left: left, right: right)
    return padded.conv2d(kernel)
  }

  private static func replicatePadRows2D(_ x: Tensor, left: Int, right: Int) -> Tensor {
    let frames = x.shape[0]
    let cols = x.shape[1]
    var padded = x.pad([(left, right), (0, 0)])

    if left > 0 {
      let leftEdge = x
        .shrink([(0, 1), nil])
        .expand([left, cols])
        .pad([(0, frames + right), (0, 0)])
      padded = padded + leftEdge
    }

    if right > 0 {
      let rightEdge = x
        .shrink([(max(0, frames - 1), frames), nil])
        .expand([right, cols])
        .pad([(left + frames, 0), (0, 0)])
      padded = padded + rightEdge
    }

    return padded
  }

  private static func makeConditioningData(
    f0Hz: [Float],
    loudnessDB: [Float],
    uvMask: [Float]
  ) -> [Float] {
    let n = min(f0Hz.count, min(loudnessDB.count, uvMask.count))
    if n == 0 { return [Float](repeating: 0, count: conditioningFeatureCount) }
    var flat = [Float]()
    flat.reserveCapacity(n * conditioningFeatureCount)
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
      prevF0Norm = f0Norm
      prevLoudNorm = loudNorm
    }
    return flat
  }

  private static func writeGainCSV(path: URL, raw: [Float], fir: [Float], frames: Int) throws {
    var text = "frame,raw,fir,delta\n"
    for i in 0..<frames {
      let r = raw[i]
      let f = fir[i]
      text += "\(i),\(r),\(f),\(f - r)\n"
    }
    try text.write(to: path, atomically: true, encoding: .utf8)
  }

  private static func writeAmpSumCSV(
    path: URL,
    raw: [Float],
    fir: [Float],
    frames: Int,
    harmonics: Int
  ) throws {
    let rawSum = ampSum(raw, rows: frames, cols: harmonics)
    let firSum = ampSum(fir, rows: frames, cols: harmonics)
    var text = "frame,raw_sum,fir_sum,delta\n"
    for i in 0..<frames {
      text += "\(i),\(rawSum[i]),\(firSum[i]),\(firSum[i] - rawSum[i])\n"
    }
    try text.write(to: path, atomically: true, encoding: .utf8)
  }

  private static func writeAmpsLongCSV(
    path: URL,
    raw: [Float],
    fir: [Float],
    frames: Int,
    harmonics: Int
  ) throws {
    var text = "frame,harmonic,raw,fir,delta\n"
    for f in 0..<frames {
      for h in 0..<harmonics {
        let idx = f * harmonics + h
        let rv = raw[idx]
        let fv = fir[idx]
        text += "\(f),\(h + 1),\(rv),\(fv),\(fv - rv)\n"
      }
    }
    try text.write(to: path, atomically: true, encoding: .utf8)
  }

  private static func ampSum(_ data: [Float], rows: Int, cols: Int) -> [Float] {
    var out = [Float](repeating: 0, count: rows)
    for r in 0..<rows {
      var s: Float = 0
      let base = r * cols
      for c in 0..<cols {
        s += data[base + c]
      }
      out[r] = s
    }
    return out
  }

  private static func meanAbsFrameDiff(_ data: [Float], rows: Int, cols: Int) -> Double {
    guard rows > 1 else { return 0 }
    var sum: Double = 0
    var count = 0
    for r in 0..<(rows - 1) {
      let a = r * cols
      let b = (r + 1) * cols
      for c in 0..<cols {
        sum += Double(abs(data[b + c] - data[a + c]))
        count += 1
      }
    }
    return count > 0 ? sum / Double(count) : 0
  }

  private static func fmt(_ v: Double) -> String {
    String(format: "%.6e", v)
  }

  private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: url)
  }
}

private struct SmoothingAnalysis: Codable {
  var chunkID: String
  var sourceFile: String
  var split: String
  var frames: Int
  var harmonics: Int
  var harmonicGainRawTV: Double
  var harmonicGainFirTV: Double
  var noiseGainRawTV: Double
  var noiseGainFirTV: Double
  var ampSumRawTV: Double
  var ampSumFirTV: Double
}

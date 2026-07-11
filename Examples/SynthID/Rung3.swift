import Foundation

struct Rung3PreparationReport: Codable {
  var sourcePath: String
  var sourceSampleRate: Float
  var targetSampleRate: Float
  var sourceFrames: Int
  var sourceDurationSeconds: Float
  var onsetFrame: Int
  var onsetSeconds: Float
  var onsetThresholdDB: Float
  var resampledFrames: Int
  var outputFrames: Int
  var cropped: Bool
  var padded: Bool
  var normalizationScale: Float
}

struct Rung3Comparison: Codable {
  var initialDistance: Float
  var learnedDistance: Float
  var improvement: Float
  var requiredImprovement: Float
  var logEpsilon: Float?
  var magnitudeNormalization: String?
  var windows: [Int]?
  var highpassHz: Float?
  var pass: Bool
}

enum Rung3TargetPreprocessor {
  static func prepare(
    samples: [Float],
    sourceRate: Float,
    sourcePath: String,
    config: SynthIDConfig,
    onsetThresholdDB: Float = -40
  ) throws -> (samples: [Float], report: Rung3PreparationReport) {
    guard !samples.isEmpty else {
      throw SynthIDError.message("rung3 target is empty")
    }
    guard sourceRate.isFinite, sourceRate > 0 else {
      throw SynthIDError.message("rung3 target has invalid sample rate \(sourceRate)")
    }

    let peak = samples.lazy.map { abs($0) }.max() ?? 0
    guard peak.isFinite, peak > 1e-8 else {
      throw SynthIDError.message("rung3 target is silent")
    }
    let relativeThreshold = Foundation.pow(10.0, onsetThresholdDB / 20.0)
    let threshold = max(1e-8, peak * relativeThreshold)
    let onsetFrame = samples.firstIndex { abs($0) >= threshold } ?? 0
    let onsetAligned = Array(samples[onsetFrame...])
    let resampled = windowedSincResample(
      onsetAligned,
      sourceRate: sourceRate,
      targetRate: config.sampleRate)
    let fitted = fitOrPad(resampled, frames: config.frames)
    let scale = peakNormalizationScale(fitted, peak: config.peakNormalizeTo)
    let normalized = fitted.map { $0 * scale }

    return (
      normalized,
      Rung3PreparationReport(
        sourcePath: sourcePath,
        sourceSampleRate: sourceRate,
        targetSampleRate: config.sampleRate,
        sourceFrames: samples.count,
        sourceDurationSeconds: Float(samples.count) / sourceRate,
        onsetFrame: onsetFrame,
        onsetSeconds: Float(onsetFrame) / sourceRate,
        onsetThresholdDB: onsetThresholdDB,
        resampledFrames: resampled.count,
        outputFrames: normalized.count,
        cropped: resampled.count > config.frames,
        padded: resampled.count < config.frames,
        normalizationScale: scale))
  }

  /// Windowed-sinc conversion preserves pitch when real targets do not match
  /// the training sample rate. The cutoff also makes downsampling safe.
  private static func windowedSincResample(
    _ input: [Float],
    sourceRate: Float,
    targetRate: Float,
    radius: Int = 16
  ) -> [Float] {
    guard abs(sourceRate - targetRate) > 0.5 else { return input }
    let ratio = Double(targetRate / sourceRate)
    let outputCount = max(1, Int((Double(input.count) * ratio).rounded()))
    let cutoff = min(1.0, ratio)
    var output = [Float](repeating: 0, count: outputCount)

    for outputIndex in 0..<outputCount {
      let sourcePosition = Double(outputIndex) / ratio
      let center = Int(Foundation.floor(sourcePosition))
      var weightedSum = 0.0
      var weightSum = 0.0
      for sampleIndex in (center - radius + 1)...(center + radius) {
        guard sampleIndex >= 0, sampleIndex < input.count else { continue }
        let distance = sourcePosition - Double(sampleIndex)
        let normalizedDistance = distance / Double(radius)
        guard abs(normalizedDistance) < 1 else { continue }
        let window = 0.5 + 0.5 * Foundation.cos(Double.pi * normalizedDistance)
        let x = cutoff * distance
        let sinc = abs(x) < 1e-12 ? 1.0 : Foundation.sin(Double.pi * x) / (Double.pi * x)
        let weight = cutoff * sinc * window
        weightedSum += Double(input[sampleIndex]) * weight
        weightSum += weight
      }
      output[outputIndex] = Float(weightedSum / max(abs(weightSum), 1e-12))
    }
    return output
  }
}

enum Rung3Comparator {
  static var defaultScriptURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/compare.py")
  }

  static func run(
    targetURL: URL,
    initialURL: URL,
    learnedURL: URL,
    outputImageURL: URL,
    outputJSONURL: URL,
    scriptURL: URL? = nil,
    python: String = "python3"
  ) throws -> Rung3Comparison {
    let script = scriptURL ?? defaultScriptURL
    guard FileManager.default.fileExists(atPath: script.path) else {
      throw SynthIDError.message("missing rung3 comparator at \(script.path)")
    }

    let arguments = [
      script.path,
      "--target", targetURL.path,
      "--initial", initialURL.path,
      "--learned", learnedURL.path,
      "--out", outputImageURL.path,
      "--json", outputJSONURL.path,
    ]
    let process = Process()
    if python.contains("/") {
      process.executableURL = URL(fileURLWithPath: python)
      process.arguments = arguments
    } else {
      process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
      process.arguments = [python] + arguments
    }
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    process.environment = environment
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr

    do {
      try process.run()
    } catch {
      throw SynthIDError.message("could not launch rung3 comparator with \(python): \(error)")
    }
    process.waitUntilExit()
    let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
    let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
    let output = String(data: outputData, encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard process.terminationStatus == 0 else {
      let detail = String(data: errorData, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
      throw SynthIDError.message(
        "rung3 comparator failed (exit \(process.terminationStatus))"
          + ((detail?.isEmpty == false) ? ": \(detail!)" : ""))
    }
    if output?.isEmpty == false { print(output!) }

    let comparison = try JSONDecoder().decode(
      Rung3Comparison.self,
      from: Data(contentsOf: outputJSONURL))
    guard comparison.initialDistance.isFinite,
      comparison.learnedDistance.isFinite,
      comparison.improvement.isFinite
    else {
      throw SynthIDError.message("rung3 comparator produced non-finite metrics")
    }
    return comparison
  }
}

enum Rung3Refiner {
  static var defaultScriptURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/refine_rung3.py")
  }

  static func run(
    targetURL: URL,
    initialURL: URL,
    paramsURL: URL,
    outputParamsURL: URL,
    outputJSONURL: URL,
    scriptURL: URL? = nil,
    python: String = "python3"
  ) throws {
    let script = scriptURL ?? defaultScriptURL
    guard FileManager.default.fileExists(atPath: script.path) else {
      throw SynthIDError.message("missing rung3 refiner at \(script.path)")
    }
    let arguments = [
      script.path,
      "--target", targetURL.path,
      "--initial", initialURL.path,
      "--params", paramsURL.path,
      "--out-params", outputParamsURL.path,
      "--json", outputJSONURL.path,
    ]
    let process = Process()
    if python.contains("/") {
      process.executableURL = URL(fileURLWithPath: python)
      process.arguments = arguments
    } else {
      process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
      process.arguments = [python] + arguments
    }
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process.environment = environment
    let output = Pipe()
    let error = Pipe()
    process.standardOutput = output
    process.standardError = error
    try process.run()
    process.waitUntilExit()
    let stdout = String(
      data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines)
    let stderr = String(
      data: error.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard process.terminationStatus == 0 else {
      throw SynthIDError.message(
        "rung3 refiner failed (exit \(process.terminationStatus))"
          + ((stderr?.isEmpty == false) ? ": \(stderr!)" : ""))
    }
    if stdout?.isEmpty == false { print(stdout!) }
  }
}

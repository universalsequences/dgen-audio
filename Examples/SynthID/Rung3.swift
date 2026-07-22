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
  var profile: String = "808"
  // The training frame count actually used for this run. Equal to the
  // configured `frames` unless the onset-cropped, resampled target is
  // shorter, in which case it shrinks to the target's real length (rounded
  // up to a multiple of the largest spectral window) so the model is never
  // asked to fit digital silence past the target's end. See FIX 2 in the
  // rung-3 909 diagnosis: zero-padding the tail cost ~4.66pp of measured
  // improvement and regressed training in the padded region.
  var fittedFrames: Int = 0
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
    // Only ever shrink `frames`, never grow it: if the onset-cropped,
    // resampled target is shorter than the configured frame count, fit at
    // its real length (rounded up to a multiple of the largest spectral
    // window) instead of zero-padding a silent tail the model can never
    // reach exactly. Longer targets (e.g. the 808 asset) are cropped exactly
    // as before, so this is a no-op for the existing 808 path.
    let fittedFrames = fittedFrameCount(
      naturalLength: resampled.count,
      configFrames: config.frames,
      windows: config.spectralWindows)
    let fitted = fitOrPad(resampled, frames: fittedFrames)
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
        cropped: resampled.count > fittedFrames,
        padded: resampled.count < fittedFrames,
        normalizationScale: scale,
        profile: config.profile,
        fittedFrames: fittedFrames))
  }

  /// The frame count to fit the target at: `configFrames` unless the natural
  /// (onset-cropped, resampled) target is shorter, in which case shrink to
  /// its length rounded up to a multiple of the largest spectral window
  /// (never below `naturalLength`, never above `configFrames`).
  static func fittedFrameCount(naturalLength: Int, configFrames: Int, windows: [Int]) -> Int {
    guard naturalLength < configFrames else { return configFrames }
    let alignment = max(1, windows.max() ?? 1)
    let rounded = ((naturalLength + alignment - 1) / alignment) * alignment
    return rounded < configFrames ? rounded : configFrames
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

/// Rescores a single rung-3 restart's learned params with the independent
/// CPU MR-STFT metric (compare.py's `mrstft` + `capture_highpass`, applied to
/// a render_reference.py render), reusing the same subprocess plumbing as
/// `Rung3Comparator`/`Rung3Refiner`. Used to select the restart winner by the
/// independent metric instead of GPU training loss (FIX 2).
enum Rung3IndependentScorer {
  static var defaultScriptURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/score_params.py")
  }

  private struct ScoreResult: Codable {
    var distance: Float
  }

  static func score(
    targetURL: URL,
    paramsURL: URL,
    frames: Int,
    sampleRate: Float,
    profile: String = "808",
    scriptURL: URL? = nil,
    python: String = "python3"
  ) throws -> Float {
    let script = scriptURL ?? defaultScriptURL
    guard FileManager.default.fileExists(atPath: script.path) else {
      throw SynthIDError.message("missing rung3 independent scorer at \(script.path)")
    }
    let arguments = [
      script.path,
      "--target", targetURL.path,
      "--params", paramsURL.path,
      "--frames", String(frames),
      "--sample-rate", String(Int(sampleRate.rounded())),
      "--profile", profile,
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
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr

    do {
      try process.run()
    } catch {
      throw SynthIDError.message("could not launch rung3 independent scorer with \(python): \(error)")
    }
    process.waitUntilExit()
    let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
    let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
    guard process.terminationStatus == 0 else {
      let detail = String(data: errorData, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
      throw SynthIDError.message(
        "rung3 independent scorer failed (exit \(process.terminationStatus))"
          + ((detail?.isEmpty == false) ? ": \(detail!)" : ""))
    }
    let output = String(data: outputData, encoding: .utf8)?
      .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard let lastLine = output.split(separator: "\n").last,
      let jsonData = lastLine.data(using: .utf8)
    else {
      throw SynthIDError.message("rung3 independent scorer produced no output")
    }
    let result = try JSONDecoder().decode(ScoreResult.self, from: jsonData)
    guard result.distance.isFinite else {
      throw SynthIDError.message("rung3 independent scorer produced non-finite distance")
    }
    return result.distance
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
    profile: String = "808",
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
      "--profile", profile,
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

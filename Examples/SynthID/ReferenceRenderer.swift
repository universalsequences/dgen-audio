import DGenLazy
import Foundation

struct RendererEquivalenceReport: Codable {
  var frames: Int
  var threshold: Float
  var maxAbsoluteError: Float
  var meanAbsoluteError: Float
  var rootMeanSquareError: Float
  var firstFailingFrame: Int?
  var pass: Bool
}

struct PolyblepSpectralReport: Codable {
  var sampleRate: Int
  var frames: Int
  var windows: [Int]
  var logEpsilon: Double
  var threshold: Double
  var distance: Double
  var pass: Bool
}

enum ReferenceRenderer {
  static let equivalenceThreshold: Float = 1e-3
  static let polyblepEquivalenceThreshold: Double = 0.00308

  static var defaultScriptURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/render_reference.py")
  }

  static var defaultPolyblepComparatorURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/compare_polyblep.py")
  }

  static func render(
    paramsURL: URL,
    outputURL: URL,
    config: SynthIDConfig,
    scriptURL: URL? = nil,
    python: String = "python3",
    oscillatorOnly: Bool = false
  ) throws {
    let script = scriptURL ?? defaultScriptURL
    guard FileManager.default.fileExists(atPath: script.path) else {
      throw SynthIDError.message("missing NumPy reference renderer at \(script.path)")
    }

    let process = Process()
    if python.contains("/") {
      process.executableURL = URL(fileURLWithPath: python)
      process.arguments = rendererArguments(
        script: script, paramsURL: paramsURL, outputURL: outputURL, config: config,
        oscillatorOnly: oscillatorOnly)
    } else {
      process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
      process.arguments = [python]
        + rendererArguments(
          script: script, paramsURL: paramsURL, outputURL: outputURL, config: config,
          oscillatorOnly: oscillatorOnly)
    }
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process.environment = environment

    let stderr = Pipe()
    process.standardError = stderr
    do {
      try process.run()
    } catch {
      throw SynthIDError.message(
        "could not launch NumPy reference renderer with \(python): \(error)")
    }
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
      let data = stderr.fileHandleForReading.readDataToEndOfFile()
      let detail = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
      throw SynthIDError.message(
        "NumPy reference renderer failed (exit \(process.terminationStatus))"
          + ((detail?.isEmpty == false) ? ": \(detail!)" : ""))
    }
  }

  static func comparePolyblep(
    trainingURL: URL,
    deploymentURL: URL,
    reportURL: URL,
    scriptURL: URL? = nil,
    python: String = "python3",
    threshold: Double = polyblepEquivalenceThreshold
  ) throws -> PolyblepSpectralReport {
    let script = scriptURL ?? defaultPolyblepComparatorURL
    guard FileManager.default.fileExists(atPath: script.path) else {
      throw SynthIDError.message("missing PolyBLEP comparator at \(script.path)")
    }
    let arguments = [
      script.path,
      "--training", trainingURL.path,
      "--deployment", deploymentURL.path,
      "--out", reportURL.path,
      "--threshold", String(threshold),
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
    let stderr = Pipe()
    process.standardError = stderr
    do {
      try process.run()
    } catch {
      throw SynthIDError.message(
        "could not launch PolyBLEP comparator with \(python): \(error)")
    }
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
      let data = stderr.fileHandleForReading.readDataToEndOfFile()
      let detail = String(data: data, encoding: .utf8)?.trimmingCharacters(
        in: .whitespacesAndNewlines)
      throw SynthIDError.message(
        "PolyBLEP comparator failed (exit \(process.terminationStatus))"
          + ((detail?.isEmpty == false) ? ": \(detail!)" : ""))
    }
    let report = try JSONDecoder().decode(
      PolyblepSpectralReport.self, from: Data(contentsOf: reportURL))
    guard abs(report.threshold - threshold) <= 1e-12 else {
      throw SynthIDError.message(
        "PolyBLEP comparator reported threshold \(report.threshold); expected \(threshold)")
    }
    guard report.pass == (report.distance < threshold) else {
      throw SynthIDError.message(
        "PolyBLEP comparator pass field disagrees with distance \(report.distance)")
    }
    return report
  }

  static func verify(
    params: PatchValues,
    referenceSamples: [Float],
    config: SynthIDConfig,
    threshold: Float = equivalenceThreshold
  ) throws -> RendererEquivalenceReport {
    guard referenceSamples.count == config.frames else {
      throw SynthIDError.message(
        "reference renderer produced \(referenceSamples.count) frames; expected \(config.frames)")
    }
    let dgenSamples = try KickVoice.render(
      values: params,
      config: config,
      parameterBacked: false)
    return try compareSamples(
      dgenSamples, referenceSamples: referenceSamples, threshold: threshold)
  }

  static func verifyOscillator(
    trainingSamples: [Float],
    referenceSamples: [Float],
    config: SynthIDConfig,
    threshold: Float = equivalenceThreshold
  ) throws -> RendererEquivalenceReport {
    guard referenceSamples.count == config.frames else {
      throw SynthIDError.message(
        "reference oscillator produced \(referenceSamples.count) frames; expected \(config.frames)")
    }
    guard trainingSamples.count == config.frames else {
      throw SynthIDError.message(
        "training oscillator produced \(trainingSamples.count) frames; expected \(config.frames)")
    }
    return try compareSamples(
      trainingSamples, referenceSamples: referenceSamples, threshold: threshold)
  }

  private static func compareSamples(
    _ dgenSamples: [Float],
    referenceSamples: [Float],
    threshold: Float
  ) throws -> RendererEquivalenceReport {
    guard dgenSamples.count == referenceSamples.count else {
      throw SynthIDError.message(
        "renderer length mismatch: DGen=\(dgenSamples.count) NumPy=\(referenceSamples.count)")
    }

    var maxAbsoluteError: Float = 0
    var absoluteErrorSum: Double = 0
    var squaredErrorSum: Double = 0
    var firstFailingFrame: Int?
    for (index, pair) in zip(dgenSamples, referenceSamples).enumerated() {
      let error = abs(pair.0 - pair.1)
      guard error.isFinite else {
        throw SynthIDError.message("renderer equivalence produced non-finite error at frame \(index)")
      }
      maxAbsoluteError = max(maxAbsoluteError, error)
      absoluteErrorSum += Double(error)
      squaredErrorSum += Double(error) * Double(error)
      if firstFailingFrame == nil && error >= threshold {
        firstFailingFrame = index
      }
    }

    let count = Double(max(1, referenceSamples.count))
    return RendererEquivalenceReport(
      frames: referenceSamples.count,
      threshold: threshold,
      maxAbsoluteError: maxAbsoluteError,
      meanAbsoluteError: Float(absoluteErrorSum / count),
      rootMeanSquareError: Float(Foundation.sqrt(squaredErrorSum / count)),
      firstFailingFrame: firstFailingFrame,
      pass: maxAbsoluteError < threshold)
  }

  private static func rendererArguments(
    script: URL,
    paramsURL: URL,
    outputURL: URL,
    config: SynthIDConfig,
    oscillatorOnly: Bool = false
  ) -> [String] {
    var arguments = [
      script.path,
      "--params", paramsURL.path,
      "--out", outputURL.path,
      "--frames", String(config.frames),
      "--sample-rate", String(Int(config.sampleRate.rounded())),
      "--profile", config.profile,
    ]
    if !config.enableNoiseFilter {
      arguments.append("--no-noise-filter")
    }
    if oscillatorOnly {
      arguments.append("--oscillator-only")
    }
    return arguments
  }
}

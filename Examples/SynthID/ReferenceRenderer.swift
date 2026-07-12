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

enum ReferenceRenderer {
  static let equivalenceThreshold: Float = 1e-3

  static var defaultScriptURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/render_reference.py")
  }

  static func render(
    paramsURL: URL,
    outputURL: URL,
    config: SynthIDConfig,
    scriptURL: URL? = nil,
    python: String = "python3"
  ) throws {
    let script = scriptURL ?? defaultScriptURL
    guard FileManager.default.fileExists(atPath: script.path) else {
      throw SynthIDError.message("missing NumPy reference renderer at \(script.path)")
    }

    let process = Process()
    if python.contains("/") {
      process.executableURL = URL(fileURLWithPath: python)
      process.arguments = rendererArguments(
        script: script, paramsURL: paramsURL, outputURL: outputURL, config: config)
    } else {
      process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
      process.arguments = [python]
        + rendererArguments(
          script: script, paramsURL: paramsURL, outputURL: outputURL, config: config)
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
    config: SynthIDConfig
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
    return arguments
  }
}

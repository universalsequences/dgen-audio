import DGenLazy
import Darwin
import Foundation

enum SynthIDCLI {
  static func run() throws {
    var args = Array(CommandLine.arguments.dropFirst())
    guard !args.isEmpty else {
      printUsage()
      return
    }

    let command = args.removeFirst()
    let parsed = try parseOptions(args)
    switch command {
    case "render":
      try render(options: parsed)
    case "train":
      try train(options: parsed)
    case "rung1":
      try rung1(options: parsed)
    case "rung2":
      try rung2(options: parsed)
    case "rung3":
      try rung3(options: parsed)
    case "help", "--help", "-h":
      printUsage()
    default:
      throw SynthIDError.message("unknown command \(command)")
    }
  }

  private static func render(options: [String: String]) throws {
    guard let paramsPath = options["params"], let outPath = options["out"] else {
      throw SynthIDError.message("render requires --params <json> and --out <wav>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.applyRuntime()

    let params = try loadPatchValues(from: URL(fileURLWithPath: paramsPath))
    let out = URL(fileURLWithPath: outPath)
    try ensureDirectory(out.deletingLastPathComponent())
    try KickVoice.renderToWav(values: params, config: config, out: out)
    print("wrote=\(out.path)")
  }

  private static func train(options: [String: String]) throws {
    guard let targetPath = options["target"], let outPath = options["out"] else {
      throw SynthIDError.message("train requires --target <wav> and --out <dir>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.applyRuntime()

    let outDir = URL(fileURLWithPath: outPath)
    try ensureDirectory(outDir)
    let targetURL = URL(fileURLWithPath: targetPath)
    let (samples, sampleRate) = try AudioFile.load(url: targetURL)
    if abs(sampleRate - config.sampleRate) > 0.5 {
      print("warning: target sampleRate=\(sampleRate) config sampleRate=\(config.sampleRate); no resampling is applied")
    }
    let trueParams = try options["params"].map { try loadPatchValues(from: URL(fileURLWithPath: $0)) }
    try AudioFile.save(
      url: outDir.appendingPathComponent("target.wav"),
      samples: fitOrPad(peakNormalized(samples, peak: config.peakNormalizeTo), frames: config.frames),
      sampleRate: config.sampleRate)

    if let paramName = options["fdcheck"] {
      let result = try SynthIDTrainer(config: config).fdcheck(
        paramName: paramName,
        targetSamples: samples,
        initial: trueParams,
        outDir: outDir)
      print(
        "fdcheck param=\(result.paramName) fd=\(String(format: "%.6e", result.finiteDifferenceGrad)) autograd=\(String(format: "%.6e", result.autogradGrad)) relErr=\(String(format: "%.6e", result.relativeError))"
      )
      return
    }

    let result = try SynthIDTrainer(config: config).train(
      targetSamples: samples,
      outDir: outDir,
      trueParams: trueParams)
    try renderLearnedAndReport(
      result: result,
      trueParams: trueParams,
      targetSamples: samples,
      config: config,
      outDir: outDir)
  }

  private static func rung1(options: [String: String]) throws {
    guard let outPath = options["out"] else {
      throw SynthIDError.message("rung1 requires --out <dir>")
    }
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    try config.applyCLI(options)
    config.rung = 1
    config.applyRuntime()

    let seeds = try options["seeds"].map { try parseIntList($0, "--seeds") }
      ?? [Int(config.seed)]
    let root = URL(fileURLWithPath: outPath)
    try ensureDirectory(root)

    var passed = 0
    for seed in seeds {
      let seedDir = seeds.count == 1 ? root : root.appendingPathComponent("seed-\(seed)")
      try ensureDirectory(seedDir)
      var seedConfig = config
      seedConfig.seed = UInt64(seed)
      let trueParams = PatchValues.sample(seed: UInt64(seed))
      try writeJSON(trueParams, to: seedDir.appendingPathComponent("true_params.json"))

      let target = peakNormalized(
        try KickVoice.render(values: trueParams, config: seedConfig, parameterBacked: true),
        peak: seedConfig.peakNormalizeTo)
      try AudioFile.save(
        url: seedDir.appendingPathComponent("target.wav"),
        samples: target,
        sampleRate: seedConfig.sampleRate)

      var bestResult: TrainingRunResult?
      var bestRestartDir: URL?
      for restart in 0..<max(1, seedConfig.restarts) {
        let restartDir = seedDir.appendingPathComponent("restart-\(restart)")
        let result = try SynthIDTrainer(config: seedConfig).train(
          targetSamples: target,
          outDir: restartDir,
          trueParams: trueParams,
          restartIndex: restart)
        if bestResult == nil || result.bestLoss < bestResult!.bestLoss {
          bestResult = result
          bestRestartDir = restartDir
        }
      }
      guard let result = bestResult else {
        throw SynthIDError.message("no rung1 result for seed \(seed)")
      }
      if let bestRestartDir {
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("checkpoint.json"),
          to: seedDir.appendingPathComponent("checkpoint.json"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("loss_curve.csv"),
          to: seedDir.appendingPathComponent("loss_curve.csv"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("initial_params.json"),
          to: seedDir.appendingPathComponent("initial_params.json"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("pitch_fit.json"),
          to: seedDir.appendingPathComponent("pitch_fit.json"))
        try copyIfPresent(
          from: bestRestartDir.appendingPathComponent("pitch_points.json"),
          to: seedDir.appendingPathComponent("pitch_points.json"))
      }
      try writeJSON(result.recovered, to: seedDir.appendingPathComponent("recovered_params.json"))
      try renderLearnedAndReport(
        result: result,
        trueParams: trueParams,
        targetSamples: target,
        config: seedConfig,
        outDir: seedDir)
      let report = ReportWriter.make(
        rung: 1,
        trueParams: trueParams,
        recovered: result.recovered,
        initLoss: result.initLoss,
        finalLoss: result.bestLoss)
      if report.pass { passed += 1 }
      print("seed=\(seed) pass=\(report.pass) bestLoss=\(String(format: "%.6f", result.bestLoss))")
    }

    let requiredPasses = seeds.count >= 5 ? 4 : seeds.count
    if passed < requiredPasses && !options.keys.contains("allow-fail") {
      throw SynthIDError.message(
        "rung1 failed: \(passed)/\(seeds.count) seeds passed; required \(requiredPasses)")
    }
  }

  private static func rung2(options: [String: String]) throws {
    guard options["params"] != nil else {
      throw SynthIDError.message("rung2 requires --params <json> with external renderer truth")
    }
    var withRung = options
    withRung["rung"] = "2"
    try train(options: withRung)
  }

  private static func rung3(options: [String: String]) throws {
    var withRung = options
    withRung["rung"] = "3"
    try train(options: withRung)
  }

  private static func renderLearnedAndReport(
    result: TrainingRunResult,
    trueParams: PatchValues?,
    targetSamples: [Float],
    config: SynthIDConfig,
    outDir: URL
  ) throws {
    let learned = peakNormalized(
      try KickVoice.render(values: result.recovered, config: config, parameterBacked: true),
      peak: config.peakNormalizeTo)
    try AudioFile.save(
      url: outDir.appendingPathComponent("learned.wav"),
      samples: learned,
      sampleRate: config.sampleRate)

    let normalizedTarget = fitOrPad(
      peakNormalized(targetSamples, peak: config.peakNormalizeTo),
      frames: config.frames)
    let silence = [Float](repeating: 0, count: Int(config.sampleRate * 0.5))
    let ab = normalizedTarget + silence + fitOrPad(learned, frames: config.frames)
    try AudioFile.save(url: outDir.appendingPathComponent("ab.wav"), samples: ab, sampleRate: config.sampleRate)

    let report = ReportWriter.make(
      rung: config.rung,
      trueParams: trueParams,
      recovered: result.recovered,
      initLoss: result.initLoss,
      finalLoss: result.bestLoss)
    try ReportWriter.write(report: report, to: outDir)
  }

  private static func parseOptions(_ args: [String]) throws -> [String: String] {
    var options: [String: String] = [:]
    var index = 0
    while index < args.count {
      let arg = args[index]
      guard arg.hasPrefix("--") else {
        throw SynthIDError.message("unexpected positional argument \(arg)")
      }
      let key = String(arg.dropFirst(2))
      let flags: Set<String> = ["freeze-pitch", "no-linear-mag", "no-noise-filter", "allow-fail"]
      if flags.contains(key) {
        options[key] = "true"
        index += 1
        continue
      }
      guard index + 1 < args.count else {
        throw SynthIDError.message("missing value for \(arg)")
      }
      options[key] = args[index + 1]
      index += 2
    }
    return options
  }

  private static func copyIfPresent(from source: URL, to destination: URL) throws {
    guard FileManager.default.fileExists(atPath: source.path) else { return }
    if FileManager.default.fileExists(atPath: destination.path) {
      try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.copyItem(at: source, to: destination)
  }

  private static func printUsage() {
    print(
      """
      SynthID

      swift run SynthID render --params <json> --out <wav> [--frames N]
      swift run SynthID train  --target <wav> --out <dir> [--rung 1|2|3] [--params <truth.json>]
      swift run SynthID train  --target <wav> --out <dir> --fdcheck <param> [--params <point.json>]
      swift run SynthID rung1  --seed <N> --out <dir> [--epochs N] [--restarts N]
      swift run SynthID rung2  --target <wav-from-numpy> --params <json> --out <dir>
      swift run SynthID rung3  --target <real-808-wav> --out <dir>
      """
    )
  }
}

do {
  try SynthIDCLI.run()
} catch {
  fputs("error: \(error)\n", stderr)
  exit(1)
}

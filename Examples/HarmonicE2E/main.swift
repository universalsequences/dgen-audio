import Foundation

do {
  try HarmonicE2EMain.run()
} catch {
  fputs("error: \(error)\n", stderr)
  exit(1)
}

struct HarmonicE2EMain {
  static func run() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard let command = args.first else {
      printHelp()
      return
    }

    if command == "--help" || command == "-h" || command == "help" {
      printHelp()
      return
    }

    let options = parseOptions(Array(args.dropFirst()))
    switch command {
    case "dump-config":
      let output = options["output"] ?? "harmonic_e2e_config.json"
      try HarmonicE2EConfig.default.write(to: URL(fileURLWithPath: output))
      log("Wrote default config to \(output)")
    case "preprocess":
      guard let input = options["input"] else {
        throw CLIError.invalid("preprocess requires --input <wav-dir>")
      }
      guard let cache = options["cache"] else {
        throw CLIError.invalid("preprocess requires --cache <cache-dir>")
      }
      var config = try HarmonicE2EConfig.load(path: options["config"])
      try config.applyCLIOverrides(options)
      let manifest = try DatasetPreprocessor.preprocess(
        inputRoot: URL(fileURLWithPath: input),
        cacheRoot: URL(fileURLWithPath: cache),
        config: config,
        logger: log
      )
      log("Done. total=\(manifest.chunkCount) train=\(manifest.trainCount) val=\(manifest.valCount)")
    case "inspect-cache":
      guard let cache = options["cache"] else {
        throw CLIError.invalid("inspect-cache requires --cache <cache-dir>")
      }
      let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))
      let split = parseSplit(options["split"])
      let limit = Int(options["limit"] ?? "5") ?? 5
      let entries = dataset.entries(for: split)
      log("cache: \(cache)")
      log("showing \(min(limit, entries.count)) / \(entries.count) entries")
      for entry in entries.prefix(limit) {
        let chunk = try dataset.loadChunk(entry)
        let voiced = zip(chunk.f0Hz, chunk.uvMask).filter { $0.1 > 0.5 }.map(\.0)
        let voicedMean = voiced.isEmpty ? 0 : voiced.reduce(0, +) / Float(voiced.count)
        log("- \(entry.id) split=\(entry.split.rawValue) frames=\(chunk.f0Hz.count) voicedF0Mean=\(fmt(voicedMean)) source=\(entry.sourceFile)")
      }
    case "train":
      guard let cache = options["cache"] else {
        throw CLIError.invalid("train requires --cache <cache-dir>")
      }
      var config = try HarmonicE2EConfig.load(path: options["config"])
      try config.applyCLIOverrides(options)
      let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))
      let runDirs = try RunDirectories.create(
        base: URL(fileURLWithPath: options["runs-dir"] ?? "runs"),
        runName: options["run-name"]
      )
      let steps = Int(options["steps"] ?? "200") ?? 200
      let split = parseSplit(options["split"]) ?? .train
      let renderEvery = Int(options["render-every"] ?? "50") ?? 50
      let kernelDumpPath = resolveKernelDumpPath(rawValue: options["kernel-dump"], runDir: runDirs.root)
      let logGraphStats = parseBoolOption(options["graph-stats"], defaultValue: false)
      try HarmonicE2ETrainer.run(
        dataset: dataset,
        config: config,
        runDirs: runDirs,
        steps: steps,
        split: split,
        renderEvery: renderEvery,
        kernelDumpPath: kernelDumpPath,
        logGraphStats: logGraphStats,
        logger: log
      )
    default:
      throw CLIError.invalid("Unknown command: \(command)")
    }
  }

  private static func parseSplit(_ raw: String?) -> DatasetSplit? {
    guard let raw else { return nil }
    return DatasetSplit(rawValue: raw.lowercased())
  }

  private static func parseOptions(_ args: [String]) -> [String: String] {
    var options: [String: String] = [:]
    var i = 0
    while i < args.count {
      let arg = args[i]
      if arg.hasPrefix("--") {
        let key = String(arg.dropFirst(2))
        if i + 1 < args.count, !args[i + 1].hasPrefix("--") {
          options[key] = args[i + 1]
          i += 2
        } else {
          options[key] = "true"
          i += 1
        }
      } else {
        i += 1
      }
    }
    return options
  }

  private static func printHelp() {
    print(
      """
      HarmonicE2E

      Commands:
        dump-config --output <file>
        preprocess --input <wav-dir> --cache <cache-dir> [--config <json>]
        inspect-cache --cache <cache-dir> [--split train|val] [--limit 5]
        train --cache <cache-dir> [--steps 200] [--run-name <name>] [--split train|val] [--batch-size N] [--kernel-dump [path]]

      Goal:
        Small batched end-to-end monophonic resynthesis example.

      Known-good dataset:
        datasets/tinysol/Keyboards/Accordion/ordinario

      Known-good train command:
        swift run HarmonicE2E train --cache .harmonic_cache_accordion_24k --steps 150 --batch-size 4 --run-name harmonic_accordion_demo
      """
    )
  }

  private static func log(_ message: String) {
    print("[HarmonicE2E] \(message)")
  }

  private static func fmt(_ value: Float) -> String {
    String(format: "%.6g", value)
  }

  private static func parseBoolOption(_ raw: String?, defaultValue: Bool) -> Bool {
    guard let raw else { return defaultValue }
    switch raw.lowercased() {
    case "1", "true", "yes", "y", "on":
      return true
    case "0", "false", "no", "n", "off":
      return false
    default:
      return defaultValue
    }
  }

  private static func resolveKernelDumpPath(rawValue: String?, runDir: URL) -> String? {
    guard let rawValue else { return nil }
    if rawValue == "true" || rawValue.isEmpty {
      return runDir.appendingPathComponent("kernels.metal").path
    }
    return URL(fileURLWithPath: rawValue, relativeTo: runDir).path
  }
}

enum CLIError: Error, CustomStringConvertible {
  case invalid(String)

  var description: String {
    switch self {
    case .invalid(let message):
      return "CLI error: \(message)"
    }
  }
}

import Foundation

do {
  try DDSPE2EMain.run()
} catch {
  fputs("error: \(error)\n", stderr)
  exit(1)
}

struct DDSPE2EMain {
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
      try handleDumpConfig(options)
    case "preprocess":
      try handlePreprocess(options)
    case "inspect-cache":
      try handleInspectCache(options)
    case "train":
      try handleTrain(options)
    default:
      throw CLIError.invalid("Unknown command: \(command)")
    }
  }

  private static func handleDumpConfig(_ options: [String: String]) throws {
    let output = options["output"] ?? "ddsp_config.json"
    try DDSPE2EConfig.default.write(to: URL(fileURLWithPath: output))
    print("Wrote default config to \(output)")
  }

  private static func handlePreprocess(_ options: [String: String]) throws {
    guard let input = options["input"] else {
      throw CLIError.invalid("preprocess requires --input <wav-dir>")
    }
    guard let cache = options["cache"] else {
      throw CLIError.invalid("preprocess requires --cache <cache-dir>")
    }

    var config = try DDSPE2EConfig.load(path: options["config"])
    try config.applyCLIOverrides(options)

    let inputURL = URL(fileURLWithPath: input)
    let cacheURL = URL(fileURLWithPath: cache)

    let manifest = try DatasetPreprocessor.preprocess(
      inputRoot: inputURL,
      cacheRoot: cacheURL,
      config: config,
      logger: log
    )

    print(
      "Done. total=\(manifest.chunkCount) train=\(manifest.trainCount) val=\(manifest.valCount)"
    )
  }

  private static func handleInspectCache(_ options: [String: String]) throws {
    guard let cache = options["cache"] else {
      throw CLIError.invalid("inspect-cache requires --cache <cache-dir>")
    }

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))
    let split = parseSplit(options["split"])
    let limit = Int(options["limit"] ?? "5") ?? 5

    print("cache: \(cache)")
    print("chunks: total=\(dataset.manifest.chunkCount) train=\(dataset.manifest.trainCount) val=\(dataset.manifest.valCount)")

    let entries = dataset.entries(for: split)
    print("showing \(min(limit, entries.count)) / \(entries.count) entries (split=\(split?.rawValue ?? "all"))")

    for entry in entries.prefix(limit) {
      let chunk = try dataset.loadChunk(entry)
      let voiced = zip(chunk.f0Hz, chunk.uvMask).filter { $0.1 > 0.5 }.map { $0.0 }
      let f0Mean = voiced.isEmpty ? 0 : voiced.reduce(0, +) / Float(voiced.count)
      let rms = sqrt(chunk.audio.map { $0 * $0 }.reduce(0, +) / Float(max(1, chunk.audio.count)))
      print(
        "- \(entry.id) split=\(entry.split.rawValue) samples=\(chunk.audio.count) frames=\(chunk.f0Hz.count) rms=\(fmt(rms)) voicedF0Mean=\(fmt(f0Mean)) source=\(entry.sourceFile)"
      )
    }
  }

  private static func handleTrain(_ options: [String: String]) throws {
    guard let cache = options["cache"] else {
      throw CLIError.invalid("train requires --cache <cache-dir>")
    }

    var config = try DDSPE2EConfig.load(path: options["config"])
    try config.applyCLIOverrides(options)

    let dataset = try CachedDataset.load(from: URL(fileURLWithPath: cache))

    let runsBase = URL(fileURLWithPath: options["runs-dir"] ?? "runs")
    let runName = options["run-name"]
    let runDirs = try RunDirectories.create(base: runsBase, runName: runName)
    let kernelDumpPath = resolveKernelDumpPath(
      rawValue: options["kernel-dump"],
      runDir: runDirs.root
    )
    let initCheckpointPath = options["init-checkpoint"]

    let steps = Int(options["steps"] ?? "200") ?? 200
    let split = parseSplit(options["split"]) ?? .train
    let mode = TrainMode(rawValue: (options["mode"] ?? "m2").lowercased()) ?? .m2
    let profileStep = Int(options["profile-step"] ?? "-1") ?? -1
    let renderEvery = Int(options["render-every"] ?? "0") ?? 0
    let renderWavPath = options["render-wav"]

    try DDSPE2ETrainer.run(
      dataset: dataset,
      config: config,
      runDirs: runDirs,
      options: TrainerOptions(
        steps: steps,
        split: split,
        mode: mode,
        kernelDumpPath: kernelDumpPath,
        initCheckpointPath: initCheckpointPath,
        profileKernelsStep: profileStep,
        renderEvery: renderEvery,
        renderWavPath: renderWavPath
      ),
      logger: log
    )
  }

  private static func resolveKernelDumpPath(rawValue: String?, runDir: URL) -> String? {
    guard let rawValue else { return nil }
    if rawValue == "true" {
      return runDir.appendingPathComponent("kernels.metal").path
    }
    return URL(fileURLWithPath: rawValue).path
  }

  private static func parseSplit(_ raw: String?) -> DatasetSplit? {
    guard let raw else { return nil }
    return DatasetSplit(rawValue: raw.lowercased())
  }

  private static func parseOptions(_ args: [String]) -> [String: String] {
    var options: [String: String] = [:]
    var i = 0

    while i < args.count {
      let token = args[i]
      if token.hasPrefix("--") {
        let key = String(token.dropFirst(2))
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

  private static func fmt(_ value: Float) -> String {
    String(format: "%.5f", value)
  }

  private static func log(_ message: String) {
    print("[DDSPE2E] \(message)")
  }

  private static func printHelp() {
    let text = """
    DDSPE2E - DDSP end-to-end scaffold

    Commands:
      dump-config --output <path>
      preprocess --input <wav-dir> --cache <cache-dir> [--config <json>] [overrides]
      inspect-cache --cache <cache-dir> [--split train|val] [--limit N]
      train --cache <cache-dir> [--runs-dir <dir>] [--run-name <name>] [--steps N] [--split train|val] [--mode dry|m2] [--config <json>] [overrides]

    Common overrides:
      --sample-rate <float>
      --chunk-size <int>
      --chunk-hop <int>
      --frame-size <int>
      --frame-hop <int>
      --min-f0 <float>
      --max-f0 <float>
      --silence-rms <float>
      --voiced-threshold <float>
      --normalize-to <float>
      --train-split <float>
      --seed <uint64>
      --max-files <int>
      --max-chunks-per-file <int>
      --shuffle <true|false>
      --model-hidden <int>
      --harmonics <int>
      --noise-filter <true|false>
      --noise-filter-size <int>
      --lr <float>
      --batch-size <int>
      --grad-accum-steps <int>
      --grad-clip <float>
      --clip-mode <element|global>
      --normalize-grad-by-frames <true|false>
      --spectral-windows <csv-int-list>
      --spectral-weight <float>
      --spectral-hop-divisor <int>
      --spectral-warmup-steps <int>
      --spectral-ramp-steps <int>
      --mse-weight <float>
      --log-every <int>
      --checkpoint-every <int>
      --kernel-dump [path]
      --init-checkpoint <model-checkpoint-json>
      --render-every <int>
      --render-wav <path>

    Examples:
      swift run DDSPE2E dump-config --output ddsp_config.json
      swift run DDSPE2E preprocess --input Assets --cache .ddsp_cache --max-files 1
      swift run DDSPE2E inspect-cache --cache .ddsp_cache --limit 3
      swift run DDSPE2E train --cache .ddsp_cache --steps 50 --mode m2
      swift run DDSPE2E train --cache .ddsp_cache --steps 1 --kernel-dump
    """
    print(text)
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

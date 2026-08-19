import DGenLazy
import Foundation

private enum CLIError: Error, CustomStringConvertible {
  case message(String)
  var description: String {
    if case .message(let text) = self { return text }
    return "error"
  }
}

private func options(_ args: ArraySlice<String>) -> [String: String] {
  var result: [String: String] = [:]
  var index = args.startIndex
  while index < args.endIndex {
    let key = args[index]
    guard key.hasPrefix("--") else {
      index = args.index(after: index)
      continue
    }
    let next = args.index(after: index)
    if next < args.endIndex, !args[next].hasPrefix("--") {
      result[String(key.dropFirst(2))] = args[next]
      index = args.index(after: next)
    } else {
      result[String(key.dropFirst(2))] = "true"
      index = next
    }
  }
  return result
}

private func configured(_ opts: [String: String]) -> ModalDrumConfig {
  var c = ModalDrumConfig()
  if let v = opts["frames"].flatMap(Int.init) { c.frames = v }
  if let v = opts["modes"].flatMap(Int.init) { c.modes = v }
  if let v = opts["steps"].flatMap(Int.init) { c.steps = v }
  if let v = opts["sample-rate"].flatMap(Float.init) { c.sampleRate = v }
  if let v = opts["lr"].flatMap(Float.init) { c.learningRate = v }
  if let v = opts["seed"].flatMap(UInt64.init) { c.seed = v }
  if let v = opts["render-every"].flatMap(Int.init) { c.renderEvery = v }
  if opts["no-loudness"] == "true" { c.loudnessWeight = 0 }
  return c
}

private func usage() {
  print(
    """
    ModalDrum M0 synthetic parameter recovery

      swift run ModalDrum fdcheck [--frames N] [--out runs/modal_m0_fd]
      swift run ModalDrum train [--steps N] [--frames N] [--modes K] [--out runs/modal_m0]
      swift run ModalDrum render-target [--frames N] [--modes K] [--out target.wav]

    Training uses MR-STFT windows 64...2048 (hop=window/4), linear+log magnitude,
    a small frame-RMS L1 auxiliary loss, fixed frequencies, and deterministic noise.
    """)
}

@main
enum ModalDrumMain {
  static func main() throws {
    let args = CommandLine.arguments.dropFirst()
    guard let command = args.first else {
      usage()
      return
    }
    let opts = options(args.dropFirst())
    let config = configured(opts)
    switch command {
    case "fdcheck":
      let root = URL(fileURLWithPath: opts["out"] ?? "runs/modal_m0_fd", isDirectory: true)
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      let report = try ModalFDChecker.run(
        config: config, outputURL: root.appendingPathComponent("fdcheck.json"))
      for group in report.groups {
        print("\(group.group): cosine=\(String(format: "%.7f", group.cosine))")
      }
      print("fdcheck passed=\(report.passed)")
      if !report.passed { throw CLIError.message("fdcheck gate failed") }
    case "train":
      let root = URL(fileURLWithPath: opts["out"] ?? "runs/modal_m0", isDirectory: true)
      let summary = try ModalDrumTrainer.train(
        config: config, runDirectory: root, kernelDumpPath: opts["kernel-dump"])
      print(
        "loss reduction=\(summary.lossReduction)x gain cosine=\(summary.gainCosine) max tau error=\(summary.maxAudibleTauRelativeError)"
      )
      print("recovery gate passed=\(summary.passedRecoveryGate)")
    case "render-target":
      let patch = knownModalPatch(modes: config.modes)
      let samples = try ModalDrumTrainer.render(patch: patch, config: config)
      let url = URL(fileURLWithPath: opts["out"] ?? "modal_target.wav")
      try AudioFile.save(url: url, samples: samples, sampleRate: config.sampleRate)
      print(url.path)
    case "help", "--help", "-h": usage()
    default:
      usage()
      throw CLIError.message("unknown command: \(command)")
    }
  }
}

import DGenLazy
import Foundation

struct CLIError: Error, CustomStringConvertible {
  var description: String
}

func parseOptions(_ args: ArraySlice<String>) -> [String: String] {
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

func idList(_ s: String?, default d: [String]) -> [String] {
  guard let s else { return d }
  if s == "fits" { return VillainData.snareIDs() }
  if s == "all" { return VillainData.allIDs() }
  return s.split(separator: ",").map { String($0) }
}

@main
enum ModalSnareADMain {
  static func main() throws {
    setvbuf(stdout, nil, _IOLBF, 0)
    let args = CommandLine.arguments.dropFirst()
    guard let command = args.first else {
      print("usage: ModalSnareAD parity|fdcheck|fit|calibrate|physfit|latent-pretrain|latent-train|latent-fitz|latent-render [options]")
      return
    }
    let opts = parseOptions(args.dropFirst())
    let frames = opts["frames"].flatMap(Int.init) ?? 48_000
    switch command {
    case "parity":
      let ids = idList(
        opts["ids"],
        default: ["snare_00_v100", "snare_05_v100", "snare_12_v100", "snare_20_v040", "blend_03", "pc_07"])
      let out = opts["out"].map { URL(fileURLWithPath: $0, isDirectory: true) }
      let results = try HeadParity.run(ids: ids, frames: frames, outDir: out)
      var worst = 0.0
      for (id, e) in results {
        print(String(format: "%@ relrms=%.3e", id, e))
        worst = max(worst, e)
      }
      print(String(format: "worst relrms=%.3e gate(<1e-3)=%@", worst, worst < 1e-3 ? "PASS" : "FAIL"))
    case "fit":
      var cfg = FitConfig()
      cfg.frames = frames
      cfg.ids = idList(opts["ids"], default: VillainData.snareIDs())
      if let v = opts["steps"].flatMap(Int.init) { cfg.steps = v }
      if let v = opts["lr"].flatMap(Float.init) { cfg.lr = v }
      if let v = opts["eval-every"].flatMap(Int.init) { cfg.evalEvery = v }
      if let v = opts["unfreeze-head"].flatMap(Int.init) { cfg.unfreezeHeadAt = v }
      if let v = opts["head-l2"].flatMap(Float.init) { cfg.headL2 = v }
      if let v = opts["windows"] { cfg.windows = v.split(separator: ",").compactMap { Int($0) } }
      if let v = opts["target"] { cfg.targetKind = v }
      if opts["loss"] == "l2" { cfg.lossMode = .l2 }
      if let v = opts["lin-weight"].flatMap(Float.init) { cfg.linearWeight = v }
      if let v = opts["log-weight"].flatMap(Float.init) { cfg.logWeight = v }
      if opts["no-coupling"] == "true" { cfg.includeCoupling = false }
      if opts["no-noise"] == "true" { cfg.includeNoise = false }
      if let v = opts["freeze"] { cfg.frozenGroups = Set(v.split(separator: ",").map(String.init)) }
      if let v = opts["freeze-output"] { cfg.freezeOutputUntil = v == "always" ? Int.max : (Int(v) ?? Int.max) }
      if let v = opts["init"] { cfg.initFrom = URL(fileURLWithPath: v) }
      if let v = opts["out"] { cfg.out = URL(fileURLWithPath: v, isDirectory: true) }
      let s = try SnareFitter.run(config: cfg)
      print(String(format: "mean fit %.4f (init %.4f, head-only %.4f, wrong-min %.4f, silence %.4f), %.2f s/step",
        s["mean_fit"] as! Float, s["mean_init"] as! Float, s["mean_head_only_stage"] as! Float,
        s["mean_wrong_min"] as! Float, s["mean_silence"] as! Float, s["s_per_step_median"] as! Double))
    case "fdcheck":
      let kind = SnareFDCheck.LossKind(rawValue: opts["loss"] ?? "mse") ?? .mse
      let out = opts["out"].map { URL(fileURLWithPath: $0) }
      let res = try SnareFDCheck.run(
        ids: idList(opts["ids"], default: ["snare_03_v100", "snare_12_v100"]),
        frames: opts["frames"].flatMap(Int.init) ?? 4096, loss: kind,
        epsilon: opts["eps"].flatMap(Float.init) ?? 1e-2,
        perGroup: opts["per-group"].flatMap(Int.init) ?? 6,
        threshold: opts["threshold"].flatMap(Float.init) ?? 0.99,
        trainHead: opts["head"] == "true", outURL: out)
      print("fdcheck passed=\(res.allSatisfy { $0.passed })")
    case "physfit":
      let out = URL(fileURLWithPath: opts["out"] ?? Paths.e1.appendingPathComponent("physical/run").path, isDirectory: true)
      let s = try PhysHeadFitter.run(
        ids: idList(opts["ids"], default: VillainData.snareIDs()), frames: frames,
        steps: opts["steps"].flatMap(Int.init) ?? 300, lr: opts["lr"].flatMap(Float.init) ?? 0.02,
        lambda: opts["lambda"].flatMap(Float.init) ?? 0, K: opts["k"].flatMap(Int.init) ?? 24,
        pure: opts["pure"] == "true", keepTableTemporal: opts["table-temporal"] == "true", out: out)
      print(String(format: "phys lambda=%g: mean head score %.4f (init %.4f, wrong-min %.4f); residual f %.1f cents rms, dec %.3f, amp %.3f",
        s["lambda"] as! Float, s["mean_fit"] as! Float, s["mean_init"] as! Float, s["mean_wrong_min_head"] as! Float,
        s["res_f_cents_rms"] as! Float, s["res_dec_log_rms"] as! Float, s["res_amp_log_rms"] as! Float))
    case "latent-pretrain": try LatentCommands.pretrain(opts)
    case "latent-train": try LatentCommands.train(opts)
    case "latent-fitz": try LatentCommands.fitz(opts)
    case "latent-render": try LatentCommands.render(opts)
    case "latent-path": try LatentCommands.path(opts)
    case "latent-surrogate": try LatentCommands.surrogateDump(opts)
    default:
      throw CLIError(description: "unknown command \(command)")
    }
  }
}

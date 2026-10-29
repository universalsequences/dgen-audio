import DGen
import DGenLazy
import Foundation

/// Finite-difference validation of every parameter group of the snare model.
///
/// Each evaluation rebuilds the graph after LazyGraphContext.reset(), so FD
/// and autograd see identical fresh compiles. Scoring: a group passes when
/// cosine(FD, AD) >= threshold, or when it is inert (both vectors ~0 relative
/// to the largest group gradient) — ModalDrum's harness scored 0-vs-0 as
/// cosine 0 and failed inert groups.
enum SnareFDCheck {
  enum LossKind: String { case mse, spectral }

  struct GroupResult {
    var name: String
    var indices: [Int]
    var fd: [Float]
    var ad: [Float]
    var cosine: Float
    var relErr: Float
    var inert: Bool
    var passed: Bool
  }

  static func run(
    ids: [String], frames: Int, loss kind: LossKind, epsilon: Float, perGroup: Int,
    threshold: Float, trainHead: Bool, outURL: URL?
  ) throws -> [GroupResult] {
    configureRuntime(frames: frames)
    LazyGraphContext.preserveCompilationCaches = false
    let renders = try VillainData.loadRenders(ids: ids)
    let B = renders.count
    let constants = SnareConstants(renders: renders)
    let bank = try NoiseBank.load()
    let targets = try ids.map { try loadTarget($0, kind: "out", frames: frames) }
    let options = RenderOptions(includeHead: true, includeNoise: true, includeCoupling: true, trainHead: trainHead)

    // A non-trivial, non-symmetric operating point: perturb raw params off init.
    LazyGraphContext.reset()
    let base = SnareParams(constants: constants, bands: bank.bands)
    var rng = SeededRNG(seed: 7)
    var point: [String: [Float]] = [:]
    for (name, t) in base.allGroups {
      let v = t.getData() ?? []
      let jitter: Float = name.hasPrefix("head") ? 0.3 : 0.6
      point[name] = v.map { $0 + Float.random(in: -jitter...jitter, using: &rng) }
      if name == "noise_amp" { point[name] = point[name]!.map { $0 + 3 } }  // audible noise
      if name == "noise_couple" { point[name] = point[name]!.map { $0 + 2 } }
    }

    func build(_ values: [String: [Float]]) -> (SnareParams, Signal) {
      LazyGraphContext.reset()
      let targetTable = Tensor(interleave(targets, frames: frames)).reshape([frames, B])
      let noiseTable = Tensor(bank.timeMajor(frames: frames)).reshape([frames, bank.bands])
      let p = SnareParams(constants: constants, bands: bank.bands, initial: values)
      let r = SnareRenderer.render(
        params: p, constants: constants, noiseTable: noiseTable, frames: frames,
        sampleRate: VillainData.sampleRate, options: options)
      let playhead = floor(Clock.seconds(frames: frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
      let target = targetTable.sampleRow(playhead)
      var loss: Signal
      switch kind {
      case .mse:
        let d = r.output - target
        loss = (d * d).sum() * (1 / Float(B))
      case .spectral:
        loss = Signal.constant(0)
        let mode: SpectralLossMode = ProcessInfo.processInfo.environment["MSAD_FD_L1"] != nil ? .l1 : .l2
        let ws = (ProcessInfo.processInfo.environment["MSAD_FD_WINDOWS"] ?? "256,1024").split(separator: ",").compactMap { Int($0) }
        let terms = ProcessInfo.processInfo.environment["MSAD_FD_TERMS"] ?? "lin,log"
        let hopDiv = ProcessInfo.processInfo.environment["MSAD_FD_HOPDIV"].flatMap(Int.init) ?? 4
        for w in ws where w <= frames {
          if terms.contains("lin") {
            loss = loss + spectralLossFFT(r.output, target, windowSize: w, lossMode: mode, hop: w / hopDiv, normalize: true)
          }
          if terms.contains("log") {
            loss = loss + spectralLossFFT(r.output, target, windowSize: w, useLogMagnitude: true, lossMode: mode, hop: w / hopDiv, normalize: true)
          }
        }
      }
      return (p, loss)
    }

    func lossValue(_ values: [String: [Float]]) throws -> Double {
      let (_, loss) = build(values)
      return try loss.realize(frames: frames).reduce(0.0) { $0 + Double($1) }
    }

    if let path = ProcessInfo.processInfo.environment["MSAD_FD_FROM"],
      let j = try? JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any],
      let raw = j["raw"] as? [String: [NSNumber]]
    {
      for (k, v) in raw where point[k] != nil { point[k] = v.map { $0.floatValue } }
    } else if ProcessInfo.processInfo.environment["MSAD_FD_INIT"] != nil {
      point = Dictionary(uniqueKeysWithValues: base.allGroups.map { ($0.0, $0.1.getData() ?? []) })
    }
    print("loss at point:", try lossValue(point), " at init:", try lossValue([:]))
    if ProcessInfo.processInfo.environment["MSAD_BISECT"] != nil {
      let initV = Dictionary(uniqueKeysWithValues: base.allGroups.map { ($0.0, $0.1.getData() ?? []) })
      for (name, _) in base.allGroups {
        var v = initV; v[name] = point[name]
        print("only \(name):", try lossValue(v))
      }
    }
    let (pAD, lossAD) = build(point)
    _ = try lossAD.backward(frames: frames)
    var adAll: [String: [Float]] = [:]
    for (name, t) in pAD.allGroups { adAll[name] = t.grad?.getData() ?? [] }

    let groups = (trainHead ? pAD.allGroups : pAD.noiseGroups + pAD.outputGroups).map { $0.0 }
    var maxAbs: Float = 0
    for g in groups { maxAbs = max(maxAbs, (adAll[g] ?? []).map { Swift.abs($0) }.max() ?? 0) }

    var results: [GroupResult] = []
    for g in groups {
      let ad = adAll[g] ?? []
      let n = point[g]!.count
      // Pick the largest-|AD| coordinates plus a few random ones.
      var candidates = Array(0..<n)
      if let lane = ProcessInfo.processInfo.environment["MSAD_FD_LANE"].flatMap(Int.init) {
        let per = n / B
        candidates = Array((lane * per)..<((lane + 1) * per))
      }
      let order = candidates.sorted { Swift.abs(ad[safe: $0] ?? 0) > Swift.abs(ad[safe: $1] ?? 0) }
      var idx = Array(order.prefix(max(1, perGroup / 2)))
      while idx.count < min(perGroup, candidates.count) {
        let r = candidates[Int.random(in: 0..<candidates.count, using: &rng)]
        if !idx.contains(r) { idx.append(r) }
      }
      var fd: [Float] = []
      for i in idx {
        var plus = point, minus = point
        plus[g]![i] += epsilon
        minus[g]![i] -= epsilon
        fd.append(Float((try lossValue(plus) - (try lossValue(minus))) / Double(2 * epsilon)))
      }
      let adSel = idx.map { ad[safe: $0] ?? 0 }
      let dot = zip(fd, adSel).reduce(Float(0)) { $0 + $1.0 * $1.1 }
      let nf = fd.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
      let na = adSel.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
      let inert = max(nf, na) < 1e-6 * max(maxAbs, 1e-30)
      let cosine = inert ? 1 : dot / max(nf * na, 1e-30)
      let diffN = zip(fd, adSel).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }.squareRoot()
      let rel = diffN / max(nf, na, 1e-30)
      let passed = inert || cosine >= threshold
      results.append(GroupResult(name: g, indices: idx, fd: fd, ad: adSel, cosine: cosine, relErr: rel, inert: inert, passed: passed))
      print(String(format: "%-14@ cos=%.6f rel=%.2e %@ %@", g as NSString, cosine, rel, inert ? "inert" : "", passed ? "PASS" : "FAIL"))
    }
    if let outURL {
      try writeJSON(
        [
          "ids": ids, "frames": frames, "loss": kind.rawValue, "epsilon": epsilon, "threshold": threshold,
          "groups": results.map {
            ["group": $0.name, "indices": $0.indices, "fd": $0.fd, "ad": $0.ad, "cosine": $0.cosine,
             "rel_err": $0.relErr, "inert": $0.inert, "passed": $0.passed] as [String: Any]
          },
          "passed": results.allSatisfy { $0.passed },
        ], to: outURL)
    }
    return results
  }
}

extension Array {
  subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}

struct SeededRNG: RandomNumberGenerator {
  var state: UInt64
  init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
  mutating func next() -> UInt64 {
    state &+= 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
  }
}

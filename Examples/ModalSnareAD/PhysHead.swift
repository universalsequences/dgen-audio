import DGen
import DGenLazy
import Foundation

/// Physically structured head (E1 step 5): circular-membrane mode ratios,
/// a two-term damping law and strike-position amplitudes, each with a bounded
/// per-mode residual whose size an L2 penalty controls.
///
///   f_k    = f0 r_k exp(sF tanh(pf_k))                 r_k = j_mn / j_01
///   1/dec  = (a + b f_k^2) exp(sD tanh(pd_k))
///   amp_k  = A |J_m(j_mn rs)| exp(sA tanh(pa_k))        (J via degree-12 polynomial in rs)
///   rise, glide g (sh_k = g f_k) and glide tau shared per snare; phase 0, t0 0.
struct BesselTable {
  var ratios: [Float]
  var coeffs: [[Float]]  // ascending

  static func load() throws -> BesselTable {
    let url = Paths.e1.appendingPathComponent("physical/bessel_poly.json")
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    let modes = j["modes"] as! [[String: Any]]
    return BesselTable(
      ratios: modes.map { ($0["ratio"] as! NSNumber).floatValue },
      coeffs: modes.map { ($0["coeffs"] as! [NSNumber]).map { $0.floatValue } })
  }
}

enum PhysRanges {
  static let f0: (Float, Float) = (25, 2500)
  static let a: (Float, Float) = (0.3, 3000)
  static let b: (Float, Float) = (1e-10, 1e-2)
  static let A: (Float, Float) = (1e-4, 50)
  static let rise: (Float, Float) = (1e-5, 0.05)
  static let tau: (Float, Float) = (2e-4, 0.3)
  static let sF: Float = 0.3466  // ln 2 / 2: +-600 cents
  static let sD: Float = 2.3026  // ln 10
  static let sA: Float = 3.0
  static let rsMax: Float = 0.95
}

final class PhysHeadParams {
  let B: Int, K: Int
  let rF0, rA, rB, rAmp, rRs, rRise, rG, rTau: DGenLazy.Tensor  // [B,1]
  let pF, pD, pA: DGenLazy.Tensor  // [B,K]

  var groups: [(String, DGenLazy.Tensor)] {
    [("f0", rF0), ("damp_a", rA), ("damp_b", rB), ("amp", rAmp), ("strike", rRs), ("rise", rRise),
     ("glide_g", rG), ("glide_tau", rTau), ("res_f", pF), ("res_dec", pD), ("res_amp", pA)]
  }

  init(B: Int, K: Int, initial: [String: [Float]]) {
    self.B = B
    self.K = K
    func p(_ n: String, _ shape: [Int]) -> DGenLazy.Tensor {
      Tensor.param(shape, data: initial[n] ?? [Float](repeating: 0, count: shape.reduce(1, *)))
    }
    rF0 = p("f0", [B, 1]); rA = p("damp_a", [B, 1]); rB = p("damp_b", [B, 1]); rAmp = p("amp", [B, 1])
    rRs = p("strike", [B, 1]); rRise = p("rise", [B, 1]); rG = p("glide_g", [B, 1]); rTau = p("glide_tau", [B, 1])
    pF = p("res_f", [B, K]); pD = p("res_dec", [B, K]); pA = p("res_amp", [B, K])
  }

  func snapshot() -> [String: [Float]] {
    Dictionary(uniqueKeysWithValues: groups.map { ($0.0, $0.1.getData() ?? []) })
  }

  /// Head tensor overrides for HeadRenderer (shapes [B,K]).
  func overrides(table: BesselTable) -> [String: DGenLazy.Tensor] {
    let ratios = Tensor((0..<B).flatMap { _ in Array(table.ratios.prefix(K)) }).reshape([B, K])
    func ex(_ t: DGenLazy.Tensor) -> DGenLazy.Tensor { t.expand([B, K]) }
    let f0 = ex(logMap(rF0, PhysRanges.f0.0, PhysRanges.f0.1))
    let f = f0 * ratios * (tanh(pF) * PhysRanges.sF).exp()
    let a = ex(logMap(rA, PhysRanges.a.0, PhysRanges.a.1))
    let b = ex(logMap(rB, PhysRanges.b.0, PhysRanges.b.1))
    let idec = (a + b * f * f) * (tanh(pD) * PhysRanges.sD).exp()
    // Strike-position mode shape |J_m(j_mn rs)| by Horner on per-mode coefficients.
    let rs = ex(rRs.sigmoid() * PhysRanges.rsMax)
    let degree = table.coeffs[0].count - 1
    var poly = Tensor((0..<B).flatMap { _ in (0..<K).map { table.coeffs[$0][degree] } }).reshape([B, K])
    for d in stride(from: degree - 1, through: 0, by: -1) {
      poly = poly * rs + Tensor((0..<B).flatMap { _ in (0..<K).map { table.coeffs[$0][d] } }).reshape([B, K])
    }
    let A = ex(logMap(rAmp, PhysRanges.A.0, PhysRanges.A.1))
    let amp = A * abs(poly) * (tanh(pA) * PhysRanges.sA).exp()
    let irise = ex(1.0 / logMap(rRise, PhysRanges.rise.0, PhysRanges.rise.1))
    let g = ex(tanh(rG) * 1.2 + 0.3)  // (-0.9, 1.5): VILLAIN clamps sh >= -0.9 f
    let tau = ex(logMap(rTau, PhysRanges.tau.0, PhysRanges.tau.1))
    let zeros = Tensor([Float](repeating: 0, count: B * K)).reshape([B, K])
    let ones = Tensor([Float](repeating: 1, count: B * K)).reshape([B, K])
    return [
      "f": f, "amp": amp, "idec": idec, "irise": irise, "t0": zeros, "ph": zeros, "sh": g * f,
      "tau": tau, "mh": ones, "md": ones,
    ]
  }

  func penalty() -> DGenLazy.Tensor {
    let f = tanh(pF) * PhysRanges.sF, d = tanh(pD) * PhysRanges.sD, a = tanh(pA) * PhysRanges.sA
    return ((f * f).sum() + (d * d).sum() * 0.1 + (a * a).sum() * 0.1) * (1 / Float(B * K))
  }
}

enum PhysHeadFitter {
  /// CPU init from the VILLAIN table: f0 from e1/physical/table_vs_bessel.json,
  /// greedy energy-ordered assignment of table modes to membrane slots, residuals
  /// set to reproduce the assigned table modes (clamped to the residual span),
  /// damping law from the table fit, strike 0.3.
  static func initialFromTable(renders: [VillainRender], K: Int, table: BesselTable, pureStructure: Bool) throws -> [String: [Float]] {
    let rows = try JSONSerialization.jsonObject(
      with: Data(contentsOf: Paths.e1.appendingPathComponent("physical/table_vs_bessel.json"))) as! [[String: Any]]
    let B = renders.count
    var out: [String: [Float]] = [:]
    func put(_ n: String, _ b: Int, _ k: Int, _ width: Int, _ v: Float) {
      if out[n] == nil { out[n] = [Float](repeating: 0, count: B * width) }
      out[n]![b * width + k] = v
    }
    func atanhClamped(_ x: Float) -> Float { let y = min(max(x, -0.97), 0.97); return 0.5 * Foundation.log((1 + y) / (1 - y)) }
    let tuning = Float(VillainData.tuning)
    for (b, r) in renders.enumerated() {
      let idx = Int(r.id.dropFirst(6).prefix(2)) ?? b
      let row = rows[min(idx, rows.count - 1)]
      let f0 = (row["f0"] as! NSNumber).floatValue / tuning  // table f is at tuning 1
      let a = max((row["decay_law_a"] as! NSNumber).floatValue, 1)
      let bb = max((row["decay_law_b"] as! NSNumber).floatValue, 1e-9)
      put("f0", b, 0, 1, logMapInverse(f0, PhysRanges.f0.0, PhysRanges.f0.1))
      put("damp_a", b, 0, 1, logMapInverse(a, PhysRanges.a.0, PhysRanges.a.1))
      put("damp_b", b, 0, 1, logMapInverse(bb * tuning * tuning, PhysRanges.b.0, PhysRanges.b.1))
      put("strike", b, 0, 1, logitInverse(0.3 / PhysRanges.rsMax))
      // Energy-weighted mean rise / glide from the table.
      let slots = r.slots
      let energy = slots.map { $0.amp * $0.amp * $0.dec0 }
      let tot = max(energy.reduce(0, +), 1e-30)
      let rise = zip(slots, energy).reduce(0.0) { $0 + $1.0.rise0 * $1.1 } / tot
      let g = zip(slots, energy).reduce(0.0) { $0 + $1.0.g0 * $1.1 } / tot
      let tau = zip(slots, energy).reduce(0.0) { $0 + $1.0.tau * $1.1 } / tot
      put("rise", b, 0, 1, logMapInverse(Float(rise), PhysRanges.rise.0, PhysRanges.rise.1))
      put("glide_g", b, 0, 1, atanhClamped((Float(g) - 0.3) / 1.2))
      put("glide_tau", b, 0, 1, logMapInverse(Float(tau), PhysRanges.tau.0, PhysRanges.tau.1))
      // Amplitude scale: strongest table mode / strongest structural amplitude.
      let poly = (0..<K).map { k -> Float in
        var y: Float = 0
        for c in table.coeffs[k].reversed() { y = y * 0.3 + c }
        return Swift.abs(y)
      }
      let maxAmp = Float(slots.map { $0.amp }.max() ?? 1)
      let A = maxAmp / max(poly.max() ?? 1, 1e-3)
      put("amp", b, 0, 1, logMapInverse(A, PhysRanges.A.0, PhysRanges.A.1))
      for k in 0..<K { put("res_f", b, k, K, 0); put("res_dec", b, k, K, 0); put("res_amp", b, k, K, atanhClamped(-0.5)) }
      if pureStructure { continue }
      // Greedy assignment, strongest table modes first.
      var free = Set(0..<K)
      for s in slots.indices.sorted(by: { energy[$0] > energy[$1] }) where energy[s] > 0.002 * tot {
        let fs = Float(slots[s].f)
        guard fs > 0.9 * f0 else { continue }
        guard let k = free.min(by: { Swift.abs(Foundation.log(fs / (f0 * table.ratios[$0]))) < Swift.abs(Foundation.log(fs / (f0 * table.ratios[$1]))) }) else { break }
        free.remove(k)
        let dlog = Foundation.log(fs / (f0 * table.ratios[k]))
        put("res_f", b, k, K, atanhClamped(dlog / PhysRanges.sF))
        let fk = f0 * table.ratios[k] * Foundation.exp(PhysRanges.sF * Foundation.tanh(atanhClamped(dlog / PhysRanges.sF)))
        let lawIdec = a + bb * fk * fk
        put("res_dec", b, k, K, atanhClamped(Foundation.log(Float(1 / slots[s].dec0) / lawIdec) / PhysRanges.sD))
        put("res_amp", b, k, K, atanhClamped(Foundation.log(max(Float(slots[s].amp), 1e-6) / max(A * poly[k], 1e-9)) / PhysRanges.sA))
      }
    }
    return out
  }

  static func run(ids: [String], frames: Int, steps: Int, lr: Float, lambda: Float, K: Int, pure: Bool, out: URL) throws -> [String: Any] {
    configureRuntime(frames: frames)
    LazyGraphContext.preserveCompilationCaches = true
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let renders = try VillainData.loadRenders(ids: ids)
    let B = renders.count
    let table = try BesselTable.load()
    let targets = try ids.map { try loadTarget($0, kind: "head", frames: frames) }
    let windows = [128, 256, 512, 1024, 2048]
    let metricWindows = [64, 128, 256, 512, 1024, 2048]
    let headArrays = HeadArrays(renders: renders, K: K)
    let init0 = try initialFromTable(renders: renders, K: K, table: table, pureStructure: pure)

    LazyGraphContext.reset()
    let targetTable = Tensor(interleave(targets, frames: frames)).reshape([frames, B])
    let params = PhysHeadParams(B: B, K: K, initial: init0)
    let opt = Adam(params: params.groups.map { $0.1 }, lr: lr)

    func render() -> SignalTensor {
      let h = headArrays.tensors(overrides: params.overrides(table: table))
      return HeadRenderer.render(h, time: Clock.seconds(frames: frames, sampleRate: VillainData.sampleRate), sampleRate: VillainData.sampleRate)
    }
    func evaluate() throws -> ([[Float]], [Float]) {
      LazyGraphContext.current.clearComputationGraph()
      let per = deinterleave(try render().realize(frames: frames), batch: B)
      return (per, zip(per, targets).map { Scores.score($0, $1, windows: metricWindows) })
    }
    let (_, initScores) = try evaluate()
    var bestMean = initScores.reduce(0, +) / Float(B)
    var bestSnap = params.snapshot()
    var times: [Double] = []
    for step in 0..<steps {
      let ts = Date()
      LazyGraphContext.current.clearComputationGraph()
      let y = render()
      let ph = floor(Clock.seconds(frames: frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
      let tgt = targetTable.sampleRow(ph)
      var loss = Signal.constant(0)
      for w in windows {
        loss = loss + spectralLossFFT(y, tgt, windowSize: w, lossMode: .l1, hop: w / 4, normalize: true) * (1 / Float(windows.count))
        loss = loss + spectralLossFFT(y, tgt, windowSize: w, useLogMagnitude: true, lossMode: .l1, hop: w / 4, normalize: true) * (1 / Float(windows.count))
      }
      if lambda > 0 {
        loss = loss + (Tensor([0.0]) + params.penalty() * (lambda / Float(frames))).peek(Signal.constant(0))
      }
      let v = try loss.backward(frames: frames)
      let progress = Float(step) / Float(max(1, steps - 1))
      opt.lr = lr * max(0.02, 0.5 * (1 + Foundation.cos(Float.pi * progress)))
      opt.step(); opt.zeroGrad()
      times.append(Date().timeIntervalSince(ts))
      if (step + 1) % 50 == 0 || step == steps - 1 {
        let (_, sc) = try evaluate()
        let m = sc.reduce(0, +) / Float(B)
        if m < bestMean { bestMean = m; bestSnap = params.snapshot() }
        print(String(format: "phys step %d loss %.4f mean head score %.4f", step, v.reduce(0, +), m))
        fflush(stdout)
      }
    }
    // Final with best params.
    for (n, t) in params.groups { if let v = bestSnap[n] { t.updateDataLazily(v) } }
    let (per, finalScores) = try evaluate()
    // Residual statistics (energy-weighted by structural amplitude is overkill; plain RMS + percentiles).
    let resF = (bestSnap["res_f"] ?? []).map { 1200 / Float(M_LN2) * PhysRanges.sF * Foundation.tanh($0) }
    let resD = (bestSnap["res_dec"] ?? []).map { PhysRanges.sD * Foundation.tanh($0) }
    let resA = (bestSnap["res_amp"] ?? []).map { PhysRanges.sA * Foundation.tanh($0) }
    func rms(_ x: [Float]) -> Float { (x.reduce(0) { $0 + $1 * $1 } / Float(max(1, x.count))).squareRoot() }
    for b in 0..<B {
      try WAV.write(out.appendingPathComponent("\(ids[b]).phys_head.wav"), per[b])
    }
    let wrong = (0..<B).map { b -> Float in
      (0..<B).filter { $0 != b }.map { Scores.score(targets[$0], targets[b], windows: metricWindows) }.min() ?? .nan
    }
    let summary: [String: Any] = [
      "lambda": lambda, "K": K, "steps": steps, "pure_init": pure,
      "mean_init": initScores.reduce(0, +) / Float(B), "mean_fit": finalScores.reduce(0, +) / Float(B),
      "per_snare_fit": finalScores, "per_snare_init": initScores, "wrong_min_head": wrong,
      "mean_wrong_min_head": wrong.reduce(0, +) / Float(B),
      "res_f_cents_rms": rms(resF), "res_dec_log_rms": rms(resD), "res_amp_log_rms": rms(resA),
      "s_per_step_median": times.sorted()[times.count / 2],
      "raw": bestSnap,
    ]
    try writeJSON(summary, to: out.appendingPathComponent("phys_summary.json"))
    return summary
  }
}

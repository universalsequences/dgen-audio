import DGenLazy
import Foundation

// E6: path-regularised fine-tune of the E2 latent decoder (`latent-path`).
//
// Straight lines z(t) = (1-t) z_A + t z_B between snare codes should sound like even steps.
// Every step samples random pairs (A, B), decodes S+1 points along each segment and scores them
// with an ANALYTIC band-envelope surrogate: log band-power envelopes on a coarse time grid computed
// in closed form from the decoded parameters (head modes as Gaussian bumps in log f with their
// glide, band noises spread over 32 log bands by their filter response, gates/coupling/fade exact,
// drive + clipper as an envelope-level gain). No audio is rendered, so the whole graph is static.
// The numpy twin and its validation against the real band-envelope metric are in
// research e6/common.py (surrogate) and e6/validate_surrogate.py.
//
// Losses:
//   rec   : weighted squared error of sigmoid positions at the 96 fit rows against the E2 decoder's
//           own positions (self-distillation, pretrain weights; mode_freq x30 in there already)
//   even  : per pair, mean_i (d_i / mean(d) - 1)^2 over the S step distances d_i
//   len   : per pair, sum(d) / d(A, B) - 1   (path length vs chord: no detours)
//   hot   : relu(peak level along the path - max(endpoint peak levels) - 1 dB)^2
//   box   : E2's range hinge, on the path points
//   lip   : relu(d(z_c, z_c + delta) - cap)^2 for random knob-space offsets around codes
//   logf  : optional slot-wise log-f linearity along the path (energy-share weighted)

struct PathData {
  var names: [String]
  var snare: [Int]
  var vel: [Float]
  var P: Int
  var sAnchor: [Float]
  var w: [Float]
  var boxLo: [Float], boxHi: [Float]
  var eshare: [Float]  // [32, K]
  var T: [Float], LC: [Float], SIG: Float, NORM: Float, invWidth: [Float], FLOOR: Float, S: [Float], NBM: Int, SR: Float
  var wireCodes: [[Float]]?  // [32][5] sigmoid positions for wire columns (E7), optional
  var null: [Float]?  // [H+1, k] orthonormal null space of the anchor readout features
  var nullK: Int

  static func load(_ url: URL) throws -> PathData {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    func fl(_ d: [String: Any], _ k: String) -> [Float] { (d[k] as! [NSNumber]).map { $0.floatValue } }
    let s = j["sur"] as! [String: Any]
    var wire: [[Float]]? = nil
    if let wt = j["wire_targets"] as? [String: Any], let pos = wt["positions"] as? [[NSNumber]] {
      wire = pos.map { $0.map { $0.floatValue } }
    }
    return PathData(
      names: j["names"] as! [String], snare: (j["snare"] as! [NSNumber]).map { $0.intValue }, vel: fl(j, "vel"),
      P: (j["P"] as! NSNumber).intValue, sAnchor: fl(j, "s_anchor"), w: fl(j, "w"),
      boxLo: fl(j, "box_lo"), boxHi: fl(j, "box_hi"), eshare: fl(j, "mode_eshare"),
      T: fl(s, "T"), LC: fl(s, "LC"), SIG: (s["SIG"] as! NSNumber).floatValue, NORM: (s["NORM"] as! NSNumber).floatValue,
      invWidth: fl(s, "INV_WIDTH"), FLOOR: (s["FLOOR"] as! NSNumber).floatValue, S: fl(s, "S"),
      NBM: (s["NBM"] as! NSNumber).intValue, SR: (s["SR"] as! NSNumber).floatValue, wireCodes: wire,
      null: (j["null"] as? [NSNumber])?.map { $0.floatValue }, nullK: (j["null_k"] as? NSNumber)?.intValue ?? 0)
  }
}

func smooth01T(_ x: Tensor) -> Tensor {
  let c = max(min(x, 1.0), 0.0)
  return c * c * (c * -2.0 + 3.0)
}

enum BandSurrogate {
  /// Decoded natural params (each [R, width]) -> (D [R, Tg, NBM] dB band density, level [R, Tg] dB).
  /// Same math as research e6/common.py `surrogate`.
  static func eval(_ p: [String: Tensor], rows R: Int, layout L: LatentLayout, data c: PathData) -> (Tensor, Tensor) {
    let Tg = c.T.count, K = L.K, NN = L.NB, NBM = c.NBM
    let tt = Tensor(c.T).reshape([1, Tg, 1]).expand([R, Tg, 1])
    func g1(_ n: String) -> Tensor { p[n]!.reshape([R, 1, 1]).expand([R, Tg, 1]) }  // [R,Tg,1]
    func g2(_ n: String) -> Tensor { p[n]!.reshape([R, 1]).expand([R, Tg]) }  // [R,Tg]
    func gk(_ n: String) -> Tensor { p[n]!.reshape([R, 1, K]).expand([R, Tg, K]) }
    func gn(_ n: String) -> Tensor { p[n]!.reshape([R, 1, NN]).expand([R, Tg, NN]) }
    // head
    let T = relu(tt - g1("pre"))
    let GH = smooth01T((T - g1("gh_time")) * (1.0 / g1("gh_fall"))) * g1("gh_depth") * -1.0 + 1.0
    let GD = smooth01T((T - g1("gd_time")) * (1.0 / g1("gd_fall"))) * g1("gd_depth") * -1.0 + 1.0
    let t = relu(T.expand([R, Tg, K]) - gk("mode_t0"))
    let f = gk("mode_freq"), sh = f * gk("mode_glide")
    let nyq = LatentRenderer.clampT((f + relu(sh)) * (-1 / (0.08 * c.SR)) + (0.48 / 0.08), 0, 1)
    let env = gk("mode_amp") * nyq * (GH * GD).expand([R, Tg, K]) * exp(t * (1.0 / gk("mode_decay")) * -1.0)
      * (exp(t * (1.0 / gk("mode_rise")) * -1.0) * -1.0 + 1.0)
    let fk = f + sh * exp(t * (1.0 / gk("mode_gtau")) * -1.0)
    let lc = Tensor(c.LC).reshape([1, 1, 1, NBM]).expand([R, Tg, K, NBM])
    let x = (log(fk).reshape([R, Tg, K, 1]).expand([R, Tg, K, NBM]) - lc) * (1 / c.SIG)
    let e2 = env * env
    let Hb = (exp(x * x * -0.5) * (e2 * (0.5 / c.NORM)).reshape([R, Tg, K, 1]).expand([R, Tg, K, NBM])).sum(axis: 2)
    let Esum = e2.sum(axis: 2)  // [R,Tg]
    let E = sqrt(Esum + 1e-10).reshape([R, Tg, 1]).expand([R, Tg, NN])
    // noise
    let nt = relu(tt - g1("noise_onset")).expand([R, Tg, NN])
    let att = exp(nt * (1.0 / g1("noise_rise").expand([R, Tg, NN])) * -1.0) * -1.0 + 1.0
    let A = gn("band_amp"), Lt = gn("band_tail")
    let benv = att * (A * (Lt * -1.0 + 1.0) * exp(nt * (1.0 / gn("band_d1")) * -1.0) + A * Lt * exp(nt * (1.0 / gn("band_d2")) * -1.0))
      + gn("band_couple") * E
    let ng = ((GD * -1.0 + 1.0) * g1("gate_noise") * -1.0 + 1.0).expand([R, Tg, NN])
    let nb = ng * benv
    let Np = nb * nb
    let Nb = Np.reshape([R * Tg, NN]).matmul(Tensor(c.S).reshape([NN, NBM])).reshape([R, Tg, NBM])
    // output stage as an envelope-level gain
    let Ptot = Esum * 0.5 + Np.sum(axis: 2)  // [R,Tg]
    let Aa = sqrt(Ptot + 1e-12) * Float(2).squareRoot()
    let amt = g2("drive_amt")
    let ge = amt * -1.0 + 1.0 + amt * g2("drive_norm") * tanh(g2("drive") * Aa) / (Aa + 1e-6)
    let v = ge * Aa / g2("clip_ceil")
    let v2 = v * v
    let cmix = g2("clip_mix")
    let gc = cmix * -1.0 + 1.0 + cmix / sqrt(sqrt(v2 * v2 + 1.0))
    let tt2 = Tensor(c.T).reshape([1, Tg]).expand([R, Tg])
    let fd = g2("fade")
    let f1 = smooth01T((tt2 - (g2("len") - fd)) * (1.0 / fd)) * -1.0 + 1.0
    let f2 = f1 * f1
    let curve = g2("fade_curve")
    let fade = f1 * (curve * -1.0 + 1.0) + f2 * f2 * curve
    let G = g2("gain") * fade * ge * gc
    let G2 = G * G
    let iw = Tensor(c.invWidth).reshape([1, 1, NBM]).expand([R, Tg, NBM])
    let dens = G2.reshape([R, Tg, 1]).expand([R, Tg, NBM]) * (Hb + Nb) * iw + c.FLOOR
    let db = Float(10 / Foundation.log(10.0))
    let D = log(dens) * db
    let level = log(G2 * Ptot + 1e-12) * db
    return (D, level)
  }
}

/// Exact anchor projection: with h2 = the decoder's last hidden layer at the anchor rows, the minimum-norm
/// change [dW3; db3] = A^T (A A^T + eps I)^{-1} (U_target - U_now), A = [h2 1], makes the decoder output
/// exactly U_target at every anchor row. Applied after every optimizer step, the path losses only move
/// the decoder inside the set of decoders that reproduce the anchors (reconstruction is kept exactly).
func projectAnchors(_ dec: LatentDecoder, rows: [(Int, Float)], target: [Double]) -> Double {
  let s = dec.snapshot(), D = dec.D, H = dec.H, P = dec.P, n = rows.count
  let Z = s["Z"]!, W1 = s["W1z"]!, w1v = s["w1v"]!, b1 = s["b1"]!, W2 = s["W2"]!, b2 = s["b2"]!
  var W3 = s["W3"]!, b3 = s["b3"]!
  var A = [Double](repeating: 0, count: n * (H + 1))
  for (r, (c, v)) in rows.enumerated() {
    var h1 = [Double](repeating: 0, count: H)
    for j in 0..<H {
      var acc = Double(b1[j]) + Double(min(max(v, 0), 1)) * Double(w1v[j])
      for d in 0..<D { acc += Double(Z[c * D + d]) * Double(W1[d * H + j]) }
      h1[j] = Foundation.tanh(acc)
    }
    for j in 0..<H {
      var acc = Double(b2[j])
      for i in 0..<H { acc += h1[i] * Double(W2[i * H + j]) }
      A[r * (H + 1) + j] = Foundation.tanh(acc)
    }
    A[r * (H + 1) + H] = 1
  }
  // residual R = U_target - A [W3; b3]
  var Rm = [Double](repeating: 0, count: n * P)
  var maxErr = 0.0
  for r in 0..<n { for k in 0..<P {
    var u = Double(b3[k])
    for j in 0..<H { u += A[r * (H + 1) + j] * Double(W3[j * P + k]) }
    let e = target[r * P + k] - u
    Rm[r * P + k] = e; maxErr = max(maxErr, abs(e))
  } }
  // G = A A^T + eps I (n x n), Cholesky, Y = G^{-1} R
  var Gm = [Double](repeating: 0, count: n * n)
  for i in 0..<n { for j in 0...i {
    var acc = 0.0
    for k in 0...H { acc += A[i * (H + 1) + k] * A[j * (H + 1) + k] }
    Gm[i * n + j] = acc + (i == j ? 1e-9 : 0); Gm[j * n + i] = Gm[i * n + j]
  } }
  var Lc = [Double](repeating: 0, count: n * n)
  for i in 0..<n { for j in 0...i {
    var sum = Gm[i * n + j]
    for k in 0..<j { sum -= Lc[i * n + k] * Lc[j * n + k] }
    Lc[i * n + j] = i == j ? sum.squareRoot() : sum / Lc[j * n + j]
  } }
  var Y = Rm
  for k in 0..<P {  // forward / back substitution per output column
    for i in 0..<n { var sum = Y[i * P + k]; for j in 0..<i { sum -= Lc[i * n + j] * Y[j * P + k] }; Y[i * P + k] = sum / Lc[i * n + i] }
    for i in stride(from: n - 1, through: 0, by: -1) { var sum = Y[i * P + k]; for j in (i + 1)..<n { sum -= Lc[j * n + i] * Y[j * P + k] }; Y[i * P + k] = sum / Lc[i * n + i] }
  }
  for j in 0...H { for k in 0..<P {
    var acc = 0.0
    for r in 0..<n { acc += A[r * (H + 1) + j] * Y[r * P + k] }
    if j < H { W3[j * P + k] += Float(acc) } else { b3[k] += Float(acc) }
  } }
  dec.W3.updateDataLazily(W3); dec.b3.updateDataLazily(b3)
  return maxErr
}

extension LatentCommands {
  static func path(_ o: [String: String]) throws {
    let steps = Int(o["steps"] ?? "2000")!, lr = Float(o["lr"] ?? "1e-4")!
    let NP = Int(o["pairs"] ?? "16")!, S = Int(o["segs"] ?? "10")!
    let wRec = Float(o["rec"] ?? "10")!, wEven = Float(o["even"] ?? "1")!, wLen = Float(o["len"] ?? "1")!
    let wHot = Float(o["hot"] ?? "0.01")!, wBox = Float(o["box"] ?? "3")!
    let wLip = Float(o["lip"] ?? "0")!, lipCap = Float(o["lip-cap"] ?? "1.0")!, lipDelta = Float(o["lip-delta"] ?? "0.5")!
    let wLogf = Float(o["logf"] ?? "0")!, wWire = Float(o["wire"] ?? "10")!
    let trainZ = o["train-z"] == "true"
    let project = o["project"] == "true"
    let saveEvery = Int(o["save-every"] ?? "500")!
    let out = url(o["out"]!)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let L = try LatentLayout.load(o["layout"].map(url) ?? LatentLayout.defaultURL)
    let data = try PathData.load(url(o["data"]!))
    let ck = try LatentCheckpoint.load(url(o["init"]!))
    let frames = 64
    configureRuntime(frames: frames)
    LazyGraphContext.preserveCompilationCaches = true
    LazyGraphContext.reset()
    let dec = LatentDecoder(D: ck.D, H: ck.H, P: ck.P, codes: ck.codes, zInit: nil, b3Init: nil, initial: ck.tensors)
    let D = ck.D, codes = ck.codes, K = L.K
    // --readout-null: W1/W2/Z frozen; W3 = W3_0 + N_H C, b3 = b3_0 + n_b C with N the null space of the
    // anchor rows' [h2 1] features, so every anchor output is EXACTLY unchanged by construction.
    let readoutNull = o["readout-null"] == "true"
    let kN = data.nullK, Hd = ck.H
    let Cn = Tensor.param([max(kN, 1), ck.P], data: [Float](repeating: 0, count: max(kN, 1) * ck.P))
    let W30 = dec.W3.getData()!, b30 = dec.b3.getData()!
    let NH = Tensor(Array((data.null ?? [0]).prefix(Hd * max(kN, 1)))), nb = Tensor(Array((data.null ?? [0]).suffix(max(kN, 1))))
    let W30T = Tensor(W30), b30T = Tensor(b30)
    if readoutNull { precondition(kN > 0 && data.null != nil, "path data has no null space") }
    let opt = Adam(params: readoutNull ? [Cn] : dec.decoderParams, lr: lr)
    func forwardU(z: Tensor, vel: Tensor) -> Tensor {
      guard readoutNull else { return dec.forward(z: z, vel: vel) }
      let B = z.shape[0]
      let h1 = tanh(z.matmul(dec.W1z) + vel.matmul(dec.w1v) + dec.b1.expand([B, Hd]))
      let h2 = tanh(h1.matmul(dec.W2) + dec.b2.expand([B, Hd]))
      let W3e = W30T.reshape([Hd, ck.P]) + NH.reshape([Hd, kN]).matmul(Cn)
      let b3e = b30T.reshape([1, ck.P]) + nb.reshape([1, kN]).matmul(Cn)
      return h2.matmul(W3e) + b3e.expand([B, ck.P])
    }
    func commitNull() {
      guard readoutNull, let C = Cn.getData(), let N = data.null else { return }
      var W3 = W30, b3 = b30
      for j in 0...Hd { for k in 0..<ck.P {
        var acc: Float = 0
        for q in 0..<kN { acc += N[j * kN + q] * C[q * ck.P + k] }
        if j < Hd { W3[j * ck.P + k] += acc } else { b3[k] += acc }
      } }
      dec.W3.updateDataLazily(W3); dec.b3.updateDataLazily(b3)
    }
    let lrZ = Float(o["lr-z"] ?? "\(lr)")!
    let optZ: Adam? = trainZ ? Adam(params: [dec.Z], lr: lrZ) : nil
    // knob-space offsets for the Lipschitz term: z_raw = z_c + delta * (dir @ M), M from the PCA of the codes
    let Z0 = dec.Z.getData()!
    var M = [Float](repeating: 0, count: D * D)
    do {  // M = diag(s) V^T as in export_instrument.py (without the final shrink): sample cov Cholesky is fine here
      var mu = [Double](repeating: 0, count: D)
      for c in 0..<codes { for d in 0..<D { mu[d] += Double(Z0[c * D + d]) / Double(codes) } }
      var Cv = [Double](repeating: 0, count: D * D)
      for c in 0..<codes { for i in 0..<D { for j in 0..<D {
        Cv[i * D + j] += (Double(Z0[c * D + i]) - mu[i]) * (Double(Z0[c * D + j]) - mu[j]) / Double(codes) } } }
      for i in 0..<D { for j in 0...i {
        var s = Cv[i * D + j]
        for k in 0..<j { s -= Double(M[i * D + k]) * Double(M[j * D + k]) }
        M[i * D + j] = Float(i == j ? Foundation.sqrt(max(s, 1e-12)) : s / Double(M[j * D + j]))
      } }
    }
    // rows: [anchor 96][path NP*(S+1)][lip codes NL][lip perturbed NL]
    let NA = data.snare.count, NR = NP * (S + 1), NL = Int(o["lip-rows"] ?? "16")!
    let RS = NR + 2 * NL  // surrogate rows
    var rng = SplitMix64(seed: UInt64(o["seed"] ?? "6")!)
    let selA = Tensor(data.snare.flatMap { c -> [Float] in var r = [Float](repeating: 0, count: codes); r[c] = 1; return r })
    let velA = Tensor(data.vel)
    let sA = Tensor(data.sAnchor), wA = Tensor(data.w)
    let selP = Tensor([Float](repeating: 0, count: NR * codes))
    let zOff = Tensor([Float](repeating: 0, count: RS * D))  // added to sel@Z (lip perturbations)
    let selS = Tensor([Float](repeating: 0, count: RS * codes))
    let logfW = Tensor([Float](repeating: 0, count: NR * K))
    let wireB = data.wireCodes != nil && wWire > 0
    var wireT = [Float](), wireW = [Float]()
    if let wc = data.wireCodes {
      let wseg = L.byName["wire_thr"]!.offset
      for (r, c) in data.snare.enumerated() {
        _ = r
        for k in 0..<L.P {
          if k >= wseg && k < wseg + 5 { wireT.append(wc[c][k - wseg]); wireW.append(1) } else { wireT.append(0); wireW.append(0) }
        }
      }
    }
    let wireTT = Tensor(wireT.isEmpty ? [0] : wireT), wireWT = Tensor(wireW.isEmpty ? [0] : wireW)
    var rec = data.w
    if wireB {  // E7 supervises the wire columns: drop the self-distillation weight there
      let wseg = L.byName["wire_thr"]!.offset
      for r in 0..<NA { for k in wseg..<(wseg + 5) { rec[r * L.P + k] = 0 } }
      wA.updateDataLazily(rec)
    }
    let pathVel = Tensor([Float](repeating: 1, count: RS))
    func sample() {
      var sp = [Float](repeating: 0, count: NR * codes), ss = [Float](repeating: 0, count: RS * codes)
      var zo = [Float](repeating: 0, count: RS * D), lw = [Float](repeating: 0, count: NR * K)
      for q in 0..<NP {
        let a = Int(rng.next() % UInt64(codes))
        var b = Int(rng.next() % UInt64(codes - 1)); if b >= a { b += 1 }
        for i in 0...S {
          let t = Float(i) / Float(S), r = q * (S + 1) + i
          sp[r * codes + a] += 1 - t; sp[r * codes + b] += t
          ss[r * codes + a] += 1 - t; ss[r * codes + b] += t
          for k in 0..<K { lw[r * K + k] = (1 - t) * data.eshare[a * K + k] + t * data.eshare[b * K + k] }
        }
      }
      for l in 0..<NL {
        let c = Int(rng.next() % UInt64(codes))
        ss[(NR + l) * codes + c] = 1; ss[(NR + NL + l) * codes + c] = 1
        var dir = (0..<D).map { _ in rng.gaussian() }
        let n = dir.reduce(0) { $0 + $1 * $1 }.squareRoot(); dir = dir.map { $0 / max(n, 1e-6) * lipDelta }
        for j in 0..<D { var acc: Float = 0; for i in 0..<D { acc += dir[i] * M[j * D + i] }; zo[(NR + NL + l) * D + j] = acc }
      }
      selP.updateDataLazily(sp); selS.updateDataLazily(ss); zOff.updateDataLazily(zo); logfW.updateDataLazily(lw)
    }
    struct Parts { var total: Tensor; var named: [(String, Tensor)] }
    func lossGraph() -> Parts {
      // anchors
      let uA = forwardU(z: dec.codes(select: selA.reshape([NA, codes])), vel: velA.reshape([NA, 1]))
      var named: [(String, Tensor)] = []
      let lrec = supervisedLossT(u: uA, layout: L, sTarget: sA.reshape([NA, L.P]), weights: wA.reshape([NA, L.P]), batch: NA)
      named.append(("rec", lrec))
      var total = lrec * wRec
      if wireB {
        var t: Tensor? = nil
        for s in L.segments where s.name.hasPrefix("wire_") {
          let r: [(Int, Int)?] = [(0, NA), (s.offset, s.offset + s.width)]
          let d = uA.shrink(r).sigmoid() - wireTT.reshape([NA, L.P]).shrink(r)
          let term = (d * d * wireWT.reshape([NA, L.P]).shrink(r)).sum()
          t = t.map { $0 + term } ?? term
        }
        let lw = t! * (1 / Float(NA * 5))
        named.append(("wire", lw)); total = total + lw * wWire
      }
      // path + lip rows through the surrogate
      let zS = dec.codes(select: selS.reshape([RS, codes])) + zOff.reshape([RS, D])
      let uS = forwardU(z: zS, vel: pathVel.reshape([RS, 1]))
      let pS = L.squash(uS, batch: RS)
      let (Ds, level) = BandSurrogate.eval(pS, rows: RS, layout: L, data: data)
      let F = data.T.count * data.NBM
      let Dp = Ds.shrink([(0, NR), nil, nil]).reshape([NP, S + 1, F])
      func sabs(_ x: Tensor) -> Tensor { sqrt(x * x + 0.01) }
      let dstep = sabs(Dp.shrink([nil, (1, S + 1), nil]) - Dp.shrink([nil, (0, S), nil])).mean(axis: 2)  // [NP,S]
      let chord = sabs(Dp.shrink([nil, (S, S + 1), nil]).reshape([NP, F]) - Dp.shrink([nil, (0, 1), nil]).reshape([NP, F])).mean(axis: 1)  // [NP]
      let m = dstep.mean(axis: 1)  // [NP]
      let ratio = dstep / m.reshape([NP, 1]).expand([NP, S]) - 1.0
      let leven = (ratio * ratio).sum() * (1 / Float(NP * S))
      let llen = (dstep.sum(axis: 1) / chord - 1.0).sum() * (1 / Float(NP))
      named.append(("even", leven)); named.append(("len", llen))
      total = total + leven * wEven + llen * wLen
      let pk = level.shrink([(0, NR), nil]).max(axis: 1).reshape([NP, S + 1])  // [NP,S+1]
      let endMax = max(pk.shrink([nil, (0, 1)]), pk.shrink([nil, (S, S + 1)])).expand([NP, S + 1])
      let hotx = relu(pk - endMax - 1.0)
      // and no holes: not more than 3 dB below the quieter endpoint
      let endMin = min(pk.shrink([nil, (0, 1)]), pk.shrink([nil, (S, S + 1)])).expand([NP, S + 1])
      let holex = relu(endMin - pk - 3.0)
      let lhot = (hotx * hotx + holex * holex).sum() * (1 / Float(NR))
      named.append(("hot", lhot)); total = total + lhot * wHot
      if wBox > 0 {
        let lbox = boxLoss(u: uS.shrink([(0, NR), nil]), layout: L, lo: data.boxLo, hi: data.boxHi, batch: NR)
        named.append(("box", lbox)); total = total + lbox * wBox
      }
      if wLip > 0 && NL > 0 {
        let F2 = F
        let d0 = Ds.shrink([(NR, NR + NL), nil, nil]).reshape([NL, F2])
        let d1 = Ds.shrink([(NR + NL, NR + 2 * NL), nil, nil]).reshape([NL, F2])
        let dl = sabs(d1 - d0).mean(axis: 1)
        let ex = relu(dl - lipCap)
        let llip = (ex * ex).sum() * (1 / Float(NL))
        named.append(("lip", llip)); total = total + llip * wLip
      }
      if wLogf > 0 {
        let lf = log(pS["mode_freq"]!.shrink([(0, NR), nil])).reshape([NP, S + 1, K])
        let la = lf.shrink([nil, (0, 1), nil]).expand([NP, S + 1, K]), lb = lf.shrink([nil, (S, S + 1), nil]).expand([NP, S + 1, K])
        let tcol = Tensor((0...S).map { Float($0) / Float(S) }).reshape([1, S + 1, 1]).expand([NP, S + 1, K])
        let dev = lf - (la + (lb - la) * tcol)
        let llogf = (dev * dev * logfW.reshape([NP, S + 1, K])).sum() * (1 / Float(NP * (S + 1))) * Float(1200 / Foundation.log(2.0) / 100)
        named.append(("logf", llogf)); total = total + llogf * wLogf
      }
      return Parts(total: total, named: named)
    }
    if o["dircheck"] == "true" {
      // directional FD of the total loss along -g/|g| on one fixed sample (checks the surrogate's gradients)
      sample()
      let params: [Tensor] = readoutNull ? [Cn] : dec.decoderParams
      let v0 = try staticToSignal(lossGraph().total, frames: frames).backward(frames: frames).reduce(0, +)
      let base = params.map { $0.getData()! }, grads = params.map { $0.grad?.getData() ?? [] }
      for t in params { t.grad = nil }
      let gn = grads.flatMap { $0 }.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
      for alpha in [Float(1e-4), 1e-3, 1e-2] {
        var lv: [Float] = []
        for sg: Float in [1, -1] {
          for (i, t) in params.enumerated() { t.updateDataLazily(zip(base[i], grads[i]).map { $0 - sg * alpha * $1 / gn }) }
          LazyGraphContext.current.clearComputationGraph()
          let viaBackward = ProcessInfo.processInfo.environment["E6_DC_BACKWARD"] != nil
          let sig = staticToSignal(lossGraph().total, frames: frames)
          lv.append(try (viaBackward ? sig.backward(frames: frames) : sig.realize(frames: frames)).reduce(0, +))
          for t in params { t.grad = nil }
        }
        print(String(format: "dircheck alpha %.0e: FD %.6g vs AD %.6g (ratio %.4f) L0 %.6g", alpha, (lv[0] - lv[1]) / (2 * alpha), -gn, (lv[0] - lv[1]) / (2 * alpha) / -gn, v0))
      }
      return
    }
    let anchorRows = zip(data.snare, data.vel).map { ($0, $1) }
    let anchorU: [Double] = dec.cpuForward(z: data.snare.flatMap { c in Array(Z0[(c * D)..<((c + 1) * D)]) }, vel: data.vel).map { Double($0) }
    var csv = "step,loss,sec\n"
    let t0 = Date()
    for step in 0..<steps {
      if step > 0 { LazyGraphContext.current.clearComputationGraph() }
      sample()
      let parts = lossGraph()
      let v = try retryingGPU("path step \(step)") { try staticToSignal(parts.total, frames: frames).backward(frames: frames) }
      let Lv = v.reduce(0, +)
      if step == 3 && ProcessInfo.processInfo.environment["E6_PROFILE"] != nil { LazyGraphContext.current.profileGPU(frames: frames); if let d = ProcessInfo.processInfo.environment["E6_DUMP_KERNELS"] { LazyGraphContext.current.dumpKernelSources(to: d) } }
      opt.lr = cosineLR(lr, step: step, steps: steps, warmup: 50)
      opt.step(); opt.zeroGrad()
      if let optZ { optZ.lr = cosineLR(lrZ, step: step, steps: steps, warmup: 50); optZ.step(); optZ.zeroGrad() } else { dec.Z.grad = nil }
      if project {
        let e = projectAnchors(dec, rows: anchorRows, target: anchorU)
        if step % 100 == 0 { print(String(format: "  projection: max |u - u_base| before %.3e", e)) }
      }
      csv += "\(step),\(Lv),\(Date().timeIntervalSince(t0))\n"
      if step % 10 == 0 && step % 100 != 0 { print(String(format: "path step %d loss %.5f (%.1f s)", step, Lv, Date().timeIntervalSince(t0))) }
      if step % 100 == 0 || step == steps - 1 {
        // component values (forward only, same sample), all in one graph
        LazyGraphContext.current.clearComputationGraph()
        let parts2 = lossGraph()
        let n = parts2.named.count
        var vec: Tensor? = nil
        for (i, c) in parts2.named.enumerated() {
          var e = [Float](repeating: 0, count: n); e[i] = 1
          let term = Tensor(e) * c.1
          vec = vec.map { $0 + term } ?? term
        }
        let vals = try vec!.realize()
        LazyGraphContext.current.clearComputationGraph()
        var s = ""
        for (i, c) in parts2.named.enumerated() { s += String(format: " %@ %.5f", c.0, vals[i]) }
        print(String(format: "path step %d loss %.5f |%@ (%.1f s)", step, Lv, s, Date().timeIntervalSince(t0)))
      }
      if (step + 1) % saveEvery == 0 || step == steps - 1 {
        commitNull()
        try LatentCheckpoint.save(dec, names: data.names, to: out,
          extra: ["stage": "path", "steps": step + 1, "lr": lr, "pairs": NP, "segs": S,
                  "weights": ["rec": wRec, "even": wEven, "len": wLen, "hot": wHot, "box": wBox, "lip": wLip, "logf": wLogf, "wire": wireB ? wWire : 0],
                  "train_z": trainZ, "readout_null": readoutNull, "null_k": kN])
        try csv.write(to: out.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
      }
    }
    print("saved \(out.path)")
  }

  // MARK: latent-surrogate (parity of the Swift surrogate against research e6/common.py)
  static func surrogateDump(_ o: [String: String]) throws {
    let L = try LatentLayout.load(o["layout"].map(url) ?? LatentLayout.defaultURL)
    let data = try PathData.load(url(o["data"]!))
    let ck = try LatentCheckpoint.load(url(o["init"]!))
    let spec = try JSONSerialization.jsonObject(with: Data(contentsOf: url(o["spec"]!))) as! [String: Any]
    let zs = (spec["z"] as! [[NSNumber]]).flatMap { $0.map { $0.floatValue } }
    let R = zs.count / ck.D
    configureRuntime(frames: 64)
    LazyGraphContext.reset()
    let dec = LatentDecoder(D: ck.D, H: ck.H, P: ck.P, codes: ck.codes, zInit: nil, b3Init: nil, initial: ck.tensors)
    let u = dec.forward(z: Tensor(zs).reshape([R, ck.D]), vel: Tensor([Float](repeating: 1, count: R)).reshape([R, 1]))
    let (Ds, _) = BandSurrogate.eval(L.squash(u, batch: R), rows: R, layout: L, data: data)
    let flat = Ds.reshape([R * data.T.count * data.NBM])
    let vals = try (flat * 1.0).realize()
    try writeJSON(["R": R, "D": vals], to: url(o["out"]!))
    print("surrogate dump \(R) rows -> \(o["out"]!)")
  }
}

import DGen
import DGenLazy
import Foundation

// Latent Kick audio stages (E2 two-pass recipe, readout-only):
//   pass 1 (Metal): audio loss of the KickVoice render from a persistent leaf u [G, P] -> dL/du
//   pass 2 (CPU):   W3/b3 gradients = h2^T dL/du (h2 is fixed: W1/W2/Z frozen), masked to the
//                   stage's columns, + the param-space anchor of all 33 rows; Adam on CPU.
// stage modal: waveform (energy-normalised) + MR on VILLAIN's modal stem; trains mode_* / tr_*
//              columns except mode_f (frequencies stay bit-identical to the pretrain).
// stage noise: MR on VILLAIN's texture+grit stem (exact, deterministic noise); trains band_* /
//              noise_* / hum_* / couple_ga.
// Reads <run>/weights.json, codes.json, anchor.json (scripts/train_np.py, scripts/prep_audio.py);
// writes <out>/weights.json (+ codes.json copy, loss.csv).

struct Mat { var r: Int; var c: Int; var d: [Float] }

func loadMat(_ j: [String: Any]) -> Mat {
  let shp = (j["shape"] as! [NSNumber]).map { $0.intValue }
  return Mat(r: shp[0], c: shp.count > 1 ? shp[1] : 1, d: (j["data"] as! [NSNumber]).map { $0.floatValue })
}

func runTrainAudio(_ a: Args) throws {
  let run = URL(fileURLWithPath: a.s("run", ""))
  let out = URL(fileURLWithPath: a.s("out", ""))
  try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
  let stage = a.s("stage", "modal")
  let steps = a.i("steps", 150), frames = a.i("frames", 48_000), G = a.i("group", 11)
  let lr = a.f("lr", 3e-4), lam0 = a.f("anchor", 10), lam1 = a.f("anchor-end", 3)
  let wave = a.f("wave", 1.0)
  configure("metal", frames: frames)
  DGenSpectralConfig.logMagnitudeEpsilon = 1e-3
  LazyGraphContext.preserveCompilationCaches = true
  let layout = try KickLayout.load()
  let P = layout.P
  let wj = try JSONSerialization.jsonObject(with: Data(contentsOf: run.appendingPathComponent("weights.json"))) as! [String: Any]
  func layer(_ k: String) -> (Mat, Mat) {
    let l = wj[k] as! [String: Any]
    return (loadMat(l["W"] as! [String: Any]), loadMat(l["b"] as! [String: Any]))
  }
  let (W1, b1) = layer("l1"), (W2, b2) = layer("l2")
  var (W3, b3) = layer("l3")
  let H = W3.r
  let cj = try JSONSerialization.jsonObject(with: Data(contentsOf: run.appendingPathComponent("codes.json"))) as! [String: Any]
  let order = cj["order"] as! [String]
  let codes = cj["codes"] as! [String: [NSNumber]]
  let aj = try JSONSerialization.jsonObject(with: Data(contentsOf: run.appendingPathComponent("anchor.json"))) as! [String: Any]
  let wcol = (aj["w"] as! [NSNumber]).map { $0.floatValue }
  let base = (aj["base"] as! NSNumber).floatValue
  let Starg = (aj["S"] as! [NSNumber]).map { $0.floatValue }
  let N = order.count
  // fixed h2 per row
  func mlpH2(_ z: [Float]) -> [Float] {
    var h1 = [Float](repeating: 0, count: W1.c)
    for j in 0..<W1.c { var s = b1.d[j]; for i in 0..<W1.r { s += z[i] * W1.d[i * W1.c + j] }; h1[j] = Foundation.tanh(s) }
    var h2 = [Float](repeating: 0, count: W2.c)
    for j in 0..<W2.c { var s = b2.d[j]; for i in 0..<W2.r { s += h1[i] * W2.d[i * W2.c + j] }; h2[j] = Foundation.tanh(s) }
    return h2
  }
  let H2 = order.map { mlpH2(codes[$0]!.map { $0.floatValue }) }
  func uRow(_ h2: [Float]) -> [Float] {
    var u = b3.d
    for i in 0..<H { let h = h2[i]; if h == 0 { continue }; for j in 0..<P { u[j] += h * W3.d[i * P + j] } }
    return u
  }
  // trainable column mask
  var mask = [Bool](repeating: false, count: P)
  for sg in layout.segments {
    let n = sg.name
    let on: Bool
    if stage == "modal" { on = (n.hasPrefix("mode_") && n != "mode_f") || n.hasPrefix("tr_") }
    else { on = n.hasPrefix("band_") || n.hasPrefix("noise_") || n.hasPrefix("hum_") || n == "couple_ga" }
    if on { for k in sg.offset..<(sg.offset + sg.width) { mask[k] = true } }
  }
  // targets
  func target(_ id: String) throws -> [Float] {
    var y: [Float]
    if stage == "modal" {
      y = try readWavF32(KickPaths.data.appendingPathComponent("villain/stems/\(id).modal.wav"))
    } else {
      let d = try Data(contentsOf: KickPaths.data.appendingPathComponent("tex_targets/\(id).f32"))
      y = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    if y.count < frames { y += [Float](repeating: 0, count: frames - y.count) }
    return Array(y.prefix(frames))
  }
  let targets = try order.map(target)
  let nChunks = (N + G - 1) / G
  precondition(N % G == 0, "--group must divide \(N)")
  let chunks: [[Int]] = (0..<nChunks).map { c in stride(from: c, to: N, by: nChunks).map { $0 } }
  let tables = try KickTables.load(bands: layout.NB)
  let hits = [KickHit](repeating: KickHit(), count: G)

  LazyGraphContext.reset()
  let uLeaf = LTensor.param([G, P], data: [Float](repeating: 0, count: G * P))
  func interleave(_ rows: [Int]) -> [Float] {
    var t = [Float](repeating: 0, count: frames * G)
    for (b, r) in rows.enumerated() { for f in 0..<frames { t[f * G + b] = targets[r][f] } }
    return t
  }
  let tabB = LTensor(interleave(chunks[0]))
  let invEB = LTensor(chunks[0].map { r in Float(frames) / max(targets[r].reduce(0) { $0 + $1 * $1 }, 1e-9) })
  let bandT = LTensor(Array(tables.bandsTimeMajor.prefix(frames * tables.bands)))
  let humT = LTensor(Array(tables.hum.prefix(frames)))
  let windows = stage == "modal" ? [256, 1024, 2048] : [128, 256, 512, 1024, 2048]
  func lossGraph() -> Signal {
    let nat = layout.squash(uLeaf, batch: G)
    let r = KickVoice.render(nat: nat, layout: layout, hits: hits, tables: bandT.reshape([frames, tables.bands]),
                             hum: humT.reshape([frames, 1]), frames: frames, sampleRate: 48_000)
    let y = stage == "modal" ? r.modal : r.noise
    let ph = floor(KickClock.seconds(frames: frames, sampleRate: 48_000) * 48_000 + 0.5)
    let tgt = tabB.reshape([frames, G]).sampleRow(ph)
    var loss = Signal.constant(0)
    for w in windows where w <= frames {
      loss = loss + spectralLossFFT(y, tgt, windowSize: w, lossMode: .l1, hop: w / 4, normalize: true)
      loss = loss + spectralLossFFT(y, tgt, windowSize: w, useLogMagnitude: true, lossMode: .l1, hop: w / 4, normalize: true)
    }
    if stage == "modal" && wave > 0 {
      let d = y - tgt
      loss = loss + (d * d * invEB).sum() * (wave / Float(G * frames))
    }
    return loss
  }
  // anchor (param space) value + gradient wrt u for all rows
  func anchor(_ U: [[Float]]) -> (Float, [[Float]]) {
    var L: Float = 0
    var g = [[Float]](repeating: [Float](repeating: 0, count: P), count: N)
    for k in 0..<N {
      for j in 0..<P {
        let s = 1 / (1 + Foundation.exp(-U[k][j]))
        let r = (s - Starg[k * P + j]) * wcol[j]
        L += r * r
        g[k][j] = 2 * r * wcol[j] * s * (1 - s)
      }
    }
    let scale = 1 / (Float(N * P) * base)
    return (L * scale, g.map { $0.map { $0 * scale } })
  }
  // Adam on W3, b3
  var mW = [Float](repeating: 0, count: W3.d.count), vW = mW, mB = [Float](repeating: 0, count: P), vB = mB
  var csv = "step,chunk,audio,anchor,lambda,sec\n"
  var audio0: [Float] = [Float](repeating: -1, count: nChunks)
  let t0 = Date()
  for step in 0..<steps {
    let ci = step % nChunks
    let rows = chunks[ci]
    if step >= nChunks || ci != 0 || step > 0 {
      tabB.updateDataLazily(interleave(rows))
      invEB.updateDataLazily(rows.map { r in Float(frames) / max(targets[r].reduce(0) { $0 + $1 * $1 }, 1e-9) })
    }
    let U = H2.map(uRow)
    uLeaf.updateDataLazily(rows.flatMap { U[$0] })
    if step > 0 { LazyGraphContext.current.clearComputationGraph() }
    var la: Float = 0
    var attempt = 0
    while true {
      do {
        let v = try lossGraph().backward(frames: frames)
        la = v.reduce(0, +) / Float(v.count)
        break
      } catch let e as NSError where e.domain == "MetalError" && attempt < 3 {
        attempt += 1; print("MetalError \(e.code), retry"); LazyGraphContext.current.clearComputationGraph()
      }
    }
    guard let gU = uLeaf.grad?.getData() else { throw NSError(domain: "kick", code: 3, userInfo: [NSLocalizedDescriptionKey: "no u grad"]) }
    if audio0[ci] < 0 { audio0[ci] = la }
    let (lAnc, gA) = anchor(U)
    let prog = Float(step) / Float(max(steps - 1, 1))
    let lam = lam0 + (lam1 - lam0) * prog
    // dL/dW3 = sum_rows h2^T g ; audio term normalised by its chunk's initial value
    var gW = [Float](repeating: 0, count: W3.d.count), gB = [Float](repeating: 0, count: P)
    let aScale = 1 / max(audio0[ci], 1e-12)
    for (b, r) in rows.enumerated() {
      let h2 = H2[r]
      for j in 0..<P where mask[j] {
        let g = gU[b * P + j] * aScale
        if g == 0 { continue }
        gB[j] += g
        for i in 0..<H { gW[i * P + j] += h2[i] * g }
      }
    }
    for r in 0..<N {
      let h2 = H2[r]
      for j in 0..<P where mask[j] {
        let g = gA[r][j] * lam / 1e-3
        gB[j] += g
        for i in 0..<H { gW[i * P + j] += h2[i] * g }
      }
    }
    let t = Float(step + 1), b1c = 1 - Foundation.pow(0.9, t), b2c = 1 - Foundation.pow(0.999, t)
    for i in 0..<gW.count where gW[i] != 0 || mW[i] != 0 {
      mW[i] = 0.9 * mW[i] + 0.1 * gW[i]; vW[i] = 0.999 * vW[i] + 0.001 * gW[i] * gW[i]
      W3.d[i] -= lr * (mW[i] / b1c) / ((vW[i] / b2c).squareRoot() + 1e-8)
    }
    for j in 0..<P where mask[j] {
      mB[j] = 0.9 * mB[j] + 0.1 * gB[j]; vB[j] = 0.999 * vB[j] + 0.001 * gB[j] * gB[j]
      b3.d[j] -= lr * (mB[j] / b1c) / ((vB[j] / b2c).squareRoot() + 1e-8)
    }
    let sec = Date().timeIntervalSince(t0)
    csv += String(format: "%d,%d,%.6g,%.6g,%.3g,%.1f\n", step, ci, la, lAnc, lam, sec)
    if step % 10 == 0 || step == steps - 1 {
      print(String(format: "step %d chunk %d audio %.5f (x%.3f of start) anchor %.3g  %.1f s", step, ci, la, la * aScale, lAnc, sec))
      fflush(stdout)
    }
  }
  // write weights in train_np format (l3 updated)
  func mj(_ m: Mat) -> [String: Any] { ["shape": m.r == 1 && m.c > 0 && m.d.count == m.c ? [1, m.c] : [m.r, m.c], "data": m.d] }
  let outJ: [String: Any] = ["l1": ["W": mj(W1), "b": mj(b1)], "l2": ["W": mj(W2), "b": mj(b2)], "l3": ["W": mj(W3), "b": mj(b3)]]
  try JSONSerialization.data(withJSONObject: outJ).write(to: out.appendingPathComponent("weights.json"))
  try? FileManager.default.removeItem(at: out.appendingPathComponent("codes.json"))
  try? FileManager.default.removeItem(at: out.appendingPathComponent("anchor.json"))
  try? FileManager.default.copyItem(at: run.appendingPathComponent("codes.json"), to: out.appendingPathComponent("codes.json"))
  try? FileManager.default.copyItem(at: run.appendingPathComponent("anchor.json"), to: out.appendingPathComponent("anchor.json"))
  try csv.write(to: out.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
  print("wrote \(out.path)")
}

func readWavF32(_ url: URL) throws -> [Float] {
  let d = try Data(contentsOf: url)
  var off = 12
  while off + 8 <= d.count {
    let id = String(data: d.subdata(in: off..<off + 4), encoding: .ascii) ?? ""
    let size = Int(d.subdata(in: off + 4..<off + 8).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
    if id == "data" {
      return d.subdata(in: off + 8..<off + 8 + size).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    off += 8 + size
  }
  throw NSError(domain: "kick", code: 4, userInfo: [NSLocalizedDescriptionKey: "no data chunk in \(url.path)"])
}

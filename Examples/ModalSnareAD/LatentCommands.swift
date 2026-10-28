import DGenLazy
import Foundation

// E2 subcommands: latent-pretrain, latent-train, latent-fitz, latent-render.
// Data prep (targets, weights, PCA z init) is e2/targets.py + e2/prep_pretrain.py;
// evaluation with the band metric and floors is e2/eval_e2.py.

/// pretrain.json (e2/prep_pretrain.py): the 96 fit renders, s* targets and loss weights.
struct LatentData {
  var ids: [String]
  var snare: [Int]
  var vel: [Float]
  var names: [String]
  var P: Int
  var s: [Float]  // [R, P]
  var w: [Float]  // [R, P]
  var zInit: [String: [Float]]
  // Blend-line rows (prep_pretrain.py --blend): z = (1-t) Z[a] + t Z[b]; fit rows have a = b.
  var a: [Int], b: [Int], t: [Float]
  var audioRows: [Int]
  var nfit: Int

  /// Selector weights over the 32 codes for row i.
  func selRow(_ i: Int) -> [Float] {
    var w = [Float](repeating: 0, count: 32)
    w[a[i]] += 1 - t[i]; w[b[i]] += t[i]
    return w
  }

  static func load(_ url: URL = Paths.research.appendingPathComponent("e2/pretrain.json")) throws -> LatentData {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    func fl(_ k: String) -> [Float] { (j[k] as! [NSNumber]).map { $0.floatValue } }
    var zi: [String: [Float]] = [:]
    for (k, v) in j["z_init"] as! [String: [NSNumber]] { zi[k] = v.map { $0.floatValue } }
    let snare = (j["snare"] as! [NSNumber]).map { $0.intValue }
    let n = snare.count
    func ints(_ k: String) -> [Int]? { (j[k] as? [NSNumber])?.map { $0.intValue } }
    return LatentData(
      ids: j["ids"] as! [String], snare: snare, vel: fl("vel"),
      names: j["names"] as! [String], P: (j["P"] as! NSNumber).intValue, s: fl("s"), w: fl("w"), zInit: zi,
      a: ints("a") ?? snare, b: ints("b") ?? snare,
      t: (j["t"] as? [NSNumber])?.map { $0.floatValue } ?? [Float](repeating: 0, count: n),
      audioRows: ints("audio_rows") ?? Array(0..<n), nfit: (j["nfit"] as? NSNumber)?.intValue ?? n)
  }

  func rows(_ idx: [Int], _ a: [Float]) -> [Float] { idx.flatMap { Array(a[($0 * P)..<(($0 + 1) * P)]) } }
}

struct LatentCheckpoint {
  var D: Int, H: Int, P: Int, codes: Int
  var tensors: [String: [Float]]

  static func load(_ url: URL) throws -> LatentCheckpoint {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    var t: [String: [Float]] = [:]
    for (k, v) in j["tensors"] as! [String: [NSNumber]] { t[k] = v.map { $0.floatValue } }
    func i(_ k: String) -> Int { (j[k] as! NSNumber).intValue }
    return LatentCheckpoint(D: i("D"), H: i("H"), P: i("P"), codes: i("codes"), tensors: t)
  }

  /// checkpoint.json (resume), weights.json (export_weights.py input), codes.json (named z).
  static func save(_ dec: LatentDecoder, names: [String], to dir: URL, extra: [String: Any] = [:]) throws {
    let snap = dec.snapshot()
    try writeJSON(["D": dec.D, "H": dec.H, "P": dec.P, "codes": dec.codes, "tensors": snap],
      to: dir.appendingPathComponent("checkpoint.json"))
    try writeJSON(LatentDecoder.exportJSON(snap, D: dec.D, H: dec.H, P: dec.P),
      to: dir.appendingPathComponent("weights.json"))
    let Z = snap["Z"] ?? []
    var codes: [String: Any] = [:]
    for (c, n) in names.enumerated() where c < dec.codes {
      codes[n] = Array(Z[(c * dec.D)..<((c + 1) * dec.D)])
    }
    var j: [String: Any] = [
      "D": dec.D, "codes": codes, "order": names,
      "velocity": "decoder input 5 (after z1..zD) = note velocity in [0,1] (eseq `velocity` input, clipped); trained at 0.4/0.7/1.0",
    ]
    for (k, v) in extra { j[k] = v }
    try writeJSON(j, to: dir.appendingPathComponent("codes.json"))
  }
}

func oneHot(_ idx: [Int], classes: Int) -> Tensor {
  var d = [Float](repeating: 0, count: idx.count * classes)
  for (r, c) in idx.enumerated() { d[r * classes + c] = 1 }
  return Tensor(d).reshape([idx.count, classes])
}

/// Weighted squared error of the decoder's sigmoid positions against s* (static scalar [1]).
func supervisedLoss(u: Tensor, layout L: LatentLayout, sTarget: [Float], weights: [Float], batch B: Int,
  only: Set<String>? = nil) -> Tensor {
  var total: Tensor? = nil
  let P = L.P
  for s in L.segments where only?.contains(s.name) ?? true {
    let x = u.shrink([(0, B), (s.offset, s.offset + s.width)])
    let pos = (s.squash == "pi" ? x * 2.0 : x).sigmoid()
    var tg = [Float](), wt = [Float]()
    for b in 0..<B {
      tg += sTarget[(b * P + s.offset)..<(b * P + s.offset + s.width)]
      wt += weights[(b * P + s.offset)..<(b * P + s.offset + s.width)]
    }
    let d = pos - Tensor(tg).reshape([B, s.width])
    let term = (d * d * Tensor(wt).reshape([B, s.width])).sum()
    total = total.map { $0 + term } ?? term
  }
  return total! * (1 / Float(B * P))
}

/// Same as supervisedLoss, with s* and weights as [B, P] tensors (updated in place per chunk,
/// so every chunk shares one compiled graph).
func supervisedLossT(u: Tensor, layout L: LatentLayout, sTarget: Tensor, weights: Tensor, batch B: Int,
  only: Set<String>? = nil) -> Tensor {
  var total: Tensor? = nil
  for s in L.segments where only?.contains(s.name) ?? true {
    let r: [(Int, Int)?] = [(0, B), (s.offset, s.offset + s.width)]
    let x = u.shrink(r)
    let pos = (s.squash == "pi" ? x * 2.0 : x).sigmoid()
    let d = pos - sTarget.shrink(r)
    let term = (d * d * weights.shrink(r)).sum()
    total = total.map { $0 + term } ?? term
  }
  return total! * (1 / Float(B * L.P))
}

/// Range hinge on decoded sigmoid positions: sum relu(lo - p)^2 + relu(p - hi)^2, with lo/hi
/// per output column ([P] each; phase columns get [0, 1]). Applied to random latent samples
/// it keeps off-code decodes inside the corpus parameter box.
func boxLoss(u: Tensor, layout L: LatentLayout, lo: [Float], hi: [Float], batch B: Int) -> Tensor {
  var total: Tensor? = nil
  for s in L.segments where s.squash != "pi" {
    let r: [(Int, Int)?] = [(0, B), (s.offset, s.offset + s.width)]
    let pos = u.shrink(r).sigmoid()
    let l = Tensor(Array(lo[s.offset..<(s.offset + s.width)])).reshape([1, s.width]).expand([B, s.width])
    let h = Tensor(Array(hi[s.offset..<(s.offset + s.width)])).reshape([1, s.width]).expand([B, s.width])
    let below = relu(l - pos), above = relu(pos - h)
    let term = (below * below + above * above).sum()
    total = total.map { $0 + term } ?? term
  }
  return total! * (1 / Float(B * L.P))
}

/// Per-column [min, max] of s* over the fit rows, widened by `margin`.
func boxBounds(_ data: LatentData, margin: Float = 0.02) -> ([Float], [Float]) {
  var lo = [Float](repeating: 1, count: data.P), hi = [Float](repeating: 0, count: data.P)
  for r in 0..<data.nfit {
    for k in 0..<data.P { let v = data.s[r * data.P + k]; lo[k] = min(lo[k], v); hi[k] = max(hi[k], v) }
  }
  return (lo.map { max(0, $0 - margin) }, hi.map { min(1, $0 + margin) })
}

/// z samples from N(mean, cov) of the current codes (Cholesky on CPU), as [n * D].
func sampleCodePrior(_ Z: [Float], codes: Int, D: Int, n: Int, rng: inout SplitMix64, spread: Float = 1) -> [Float] {
  var mu = [Double](repeating: 0, count: D)
  for c in 0..<codes { for d in 0..<D { mu[d] += Double(Z[c * D + d]) / Double(codes) } }
  var C = [Double](repeating: 0, count: D * D)
  for c in 0..<codes { for i in 0..<D { for j in 0..<D {
    C[i * D + j] += (Double(Z[c * D + i]) - mu[i]) * (Double(Z[c * D + j]) - mu[j]) / Double(codes - 1) } } }
  var Lc = [Double](repeating: 0, count: D * D)
  for i in 0..<D { for j in 0...i {
    var sum = C[i * D + j] + (i == j ? 1e-6 : 0)
    for k in 0..<j { sum -= Lc[i * D + k] * Lc[j * D + k] }
    Lc[i * D + j] = i == j ? Foundation.sqrt(max(sum, 1e-12)) : sum / Lc[j * D + j]
  } }
  var out = [Float](repeating: 0, count: n * D)
  for r in 0..<n {
    let e = (0..<D).map { _ in Double(rng.gaussian()) }
    for i in 0..<D {
      var acc = 0.0
      for k in 0...i { acc += Lc[i * D + k] * e[k] }
      out[r * D + i] = Float(mu[i] + Double(spread) * acc)
    }
  }
  return out
}

func staticToSignal(_ t: Tensor, frames: Int) -> Signal {
  (Tensor([0.0]) + t).peek(Signal.constant(0)) * (1 / Float(frames))
}

func mrstft(_ y: SignalTensor, _ target: SignalTensor, windows: [Int], frames: Int, lin: Float = 1, log: Float = 1) -> Signal {
  var loss = Signal.constant(0)
  let wScale = 1 / Float(windows.count)
  for w in windows where w <= frames {
    let hop = max(1, w / 4)
    if lin > 0 {
      loss = loss + spectralLossFFT(y, target, windowSize: w, lossMode: .l1, hop: hop, normalize: true) * (lin * wScale)
    }
    if log > 0 {
      loss = loss + spectralLossFFT(y, target, windowSize: w, useLogMagnitude: true, lossMode: .l1, hop: hop, normalize: true) * (log * wScale)
    }
  }
  return loss
}

/// Retry a GPU run that the macOS watchdog killed (MetalError 6, e.g. "Impacting
/// Interactivity" when a long command buffer shares the GPU with the display). Nothing is
/// stepped before the run returns, so a retry is exact.
func retryingGPU<T>(_ what: String, attempts: Int = 4, _ body: () throws -> T) throws -> T {
  var n = 0
  while true {
    do { return try body() } catch let e as NSError where e.domain == "MetalError" && e.code == 6 && n < attempts - 1 {
      n += 1
      print("[gpu] \(what): \(e.localizedDescription); retry \(n)")
      LazyGraphContext.current.clearComputationGraph()
    }
  }
}

func cosineLR(_ base: Float, step: Int, steps: Int, floor: Float = 0.02, warmup: Int = 0) -> Float {
  if step < warmup { return base * Float(step + 1) / Float(warmup) }
  let progress = Float(step) / Float(max(1, steps - 1))
  return base * max(floor, 0.5 * (1 + Foundation.cos(Float.pi * progress)))
}

enum LatentCommands {
  static func url(_ s: String) -> URL { URL(fileURLWithPath: s, isDirectory: s.hasSuffix("/")) }

  // MARK: latent-pretrain (stage a: param-space supervision, no audio)

  static func pretrain(_ o: [String: String]) throws {
    let D = Int(o["d"] ?? "4")!, H = Int(o["h"] ?? "128")!
    let steps = Int(o["steps"] ?? "4000")!, lr = Float(o["lr"] ?? "0.003")!
    let zl2 = Float(o["zl2"] ?? "0.01")!
    let out = url(o["out"] ?? Paths.research.appendingPathComponent("e2/runs/pre_d\(D)").path)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let L = try LatentLayout.load(o["layout"].map(url) ?? LatentLayout.defaultURL)
    let data = try LatentData.load(o["data"].map(url) ?? Paths.research.appendingPathComponent("e2/pretrain.json"))
    precondition(data.P == L.P, "pretrain.json P \(data.P) != layout P \(L.P): rerun prep_pretrain.py")
    let frames = 64
    configureRuntime(frames: frames)
    LazyGraphContext.preserveCompilationCaches = true
    let R = data.ids.count, codes = 32
    // b3 init: mean target logit (the decoder starts at the average snare).
    var b3 = [Float](repeating: 0, count: L.P)
    for r in 0..<data.nfit { for k in 0..<L.P { b3[k] += data.s[r * L.P + k] / Float(data.nfit) } }
    for s in L.segments {
      for k in s.offset..<(s.offset + s.width) {
        let q = min(max(b3[k], 1e-3), 1 - 1e-3)
        b3[k] = Foundation.log(q / (1 - q)) * (s.squash == "pi" ? 0.5 : 1)
      }
    }
    LazyGraphContext.reset()
    var initial: [String: [Float]] = [:]
    if let ck = o["init"] { initial = try LatentCheckpoint.load(url(ck)).tensors }
    let dec = LatentDecoder(D: D, H: H, P: L.P, codes: codes, zInit: data.zInit["\(D)"], b3Init: b3,
      initial: initial, seed: UInt64(o["seed"] ?? "7")!)
    let opt = Adam(params: dec.decoderParams + [dec.Z], lr: lr)
    // Batch = every fit row + (if any) a fresh random sample of blend-line rows each step,
    // held in persistent 1-D tensors updated in place (one compiled graph for all steps).
    let nBlendRows = R - data.nfit
    let nb = min(Int(o["batch-blend"] ?? "288")!, nBlendRows)
    let BR = data.nfit + nb
    var rows = Array(0..<BR)
    var rng = SplitMix64(seed: 99)
    func gather(_ rs: [Int]) -> (sel: [Float], vel: [Float], s: [Float], w: [Float]) {
      (rs.flatMap { data.selRow($0) }, rs.map { data.vel[$0] }, data.rows(rs, data.s), data.rows(rs, data.w))
    }
    var g0 = gather(rows)
    let selB = Tensor(g0.sel), velB = Tensor(g0.vel), sB = Tensor(g0.s), wB = Tensor(g0.w)
    let boxW = Float(o["box"] ?? "0")!, nBox = 64
    let (boxLo, boxHi) = boxBounds(data)
    let zbB = Tensor([Float](repeating: 0, count: nBox * D))
    let vbB = Tensor((0..<nBox).map { [Float(0.4), 0.7, 1.0][$0 % 3] })
    var csv = "step,loss,sec\n"
    let t0 = Date()
    for step in 0..<steps {
      if step > 0 { LazyGraphContext.current.clearComputationGraph() }
      if boxW > 0 {
        zbB.updateDataLazily(sampleCodePrior(dec.Z.getData() ?? [], codes: codes, D: D, n: nBox, rng: &rng,
          spread: [Float(1), 1.5, 2][step % 3]))
      }
      if nb > 0 && step > 0 {
        rows = Array(0..<data.nfit) + (0..<nb).map { _ in data.nfit + Int(rng.next() % UInt64(nBlendRows)) }
        g0 = gather(rows)
        selB.updateDataLazily(g0.sel); velB.updateDataLazily(g0.vel); sB.updateDataLazily(g0.s); wB.updateDataLazily(g0.w)
      }
      let u = dec.forward(z: dec.codes(select: selB.reshape([BR, codes])), vel: velB.reshape([BR, 1]))
      var loss = supervisedLossT(u: u, layout: L, sTarget: sB.reshape([BR, L.P]), weights: wB.reshape([BR, L.P]), batch: BR)
      loss = loss + (dec.Z * dec.Z).sum() * (zl2 / Float(codes * D))
      if boxW > 0 {
        // samples are constants: the box shapes the decoder, not the codes
        let ub = dec.forward(z: zbB.reshape([nBox, D]), vel: vbB.reshape([nBox, 1]))
        loss = loss + boxLoss(u: ub, layout: L, lo: boxLo, hi: boxHi, batch: nBox) * boxW
      }
      let v = try staticToSignal(loss, frames: frames).backward(frames: frames)
      let Lv = v.reduce(0, +)
      opt.lr = cosineLR(lr, step: step, steps: steps, warmup: 50)
      opt.step(); opt.zeroGrad()
      csv += "\(step),\(Lv),\(Date().timeIntervalSince(t0))\n"
      if step % 250 == 0 || step == steps - 1 { print(String(format: "pretrain step %d loss %.6f (%.1f s)", step, Lv, Date().timeIntervalSince(t0))) }
    }
    try csv.write(to: out.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
    try LatentCheckpoint.save(dec, names: (0..<32).map { data.names[$0 * 3] }, to: out,
      extra: ["stage": "pretrain", "steps": steps, "zl2": zl2])
    print("saved \(out.path)")
  }

  // MARK: latent-train (stage b: head stems; stage c: out.wav)

  static func train(_ o: [String: String]) throws {
    let stage = o["stage"] ?? "out"  // "head" | "out"
    let steps = Int(o["steps"] ?? "300")!, lr = Float(o["lr"] ?? "0.001")!
    let zl2 = Float(o["zl2"] ?? "0.01")!, anchor = Float(o["anchor"] ?? "0.1")!
    let anchorEnd = Float(o["anchor-end"] ?? o["anchor"] ?? "0.1")!
    let wave = Float(o["wave"] ?? (stage == "head" ? "1" : "0"))!
    let frames = Int(o["frames"] ?? "48000")!
    let windows = (o["windows"] ?? "128,256,512,1024,2048").split(separator: ",").compactMap { Int($0) }
    let evalEvery = Int(o["eval-every"] ?? "50")!
    let freqAnchor = Float(o["freq-anchor"] ?? "0")!
    let freezeDecoder = o["freeze-decoder"] == "true"
    let freezeZ = o["freeze-z"] == "true"
    let out = url(o["out"]!)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let L = try LatentLayout.load(o["layout"].map(url) ?? LatentLayout.defaultURL)
    let data = try LatentData.load(o["data"].map(url) ?? Paths.research.appendingPathComponent("e2/pretrain.json"))
    let ck = try LatentCheckpoint.load(url(o["init"]!))
    // Render subset: "all" = the 96 fit renders, "v100" = 32. Trained in rotating chunks of
    // --group renders (interleaved, so each chunk mixes snares and velocities).
    // --rows fits (default): the 96 fit renders; audio: + the rendered blend-line rows
    var sel = o["rows"] == "audio" ? data.audioRows : Array(0..<data.nfit)
    if o["ids"] == "v100" { sel = sel.filter { data.ids[$0].hasSuffix("v100") } }
    let group = min(Int(o["group"] ?? "32")!, sel.count)
    let nChunks = (sel.count + group - 1) / group
    let chunkSel: [[Int]] = (0..<nChunks).map { c in stride(from: c, to: sel.count, by: nChunks).map { sel[$0] } }
    configureRuntime(frames: frames)
    LazyGraphContext.preserveCompilationCaches = ProcessInfo.processInfo.environment["MSAD_NO_PRESERVE"] == nil
    struct Chunk {
      var ids: [String]; var targets: [[Float]]; var invE: [Float]; var sT: [Float]; var wT: [Float]
      var snareIdx: [Int]; var vels: [Float]; var table: Tensor? = nil
      // Persistent leaves (updated in place each step): a fresh data tensor per step
      // changes the compiled kernels, so every step recompiled.
      var oneHot: [Float] = []
      var B: Int { ids.count }
    }
    var chunks: [Chunk] = []
    for cs in chunkSel {
      let ids = cs.map { data.ids[$0] }
      let renders = stage == "head" ? try VillainData.loadRenders(ids: ids) : []
      var targets: [[Float]] = []
      for (b, id) in ids.enumerated() {
        if stage == "head" {
          // template head absorbs gin = vel^-c_vin into the mode amplitudes
          let r = renders[b]
          let gin = Float(r.vel > 0 ? Foundation.exp(-r.c("c_vin") * Foundation.log(max(r.vel, 1e-4))) : 0)
          targets.append(try loadTarget(id, kind: "head", frames: frames).map { $0 * gin })
        } else if id.hasPrefix("bl_") {
          var t = try WAV.read(Paths.research.appendingPathComponent("e2/blend_targets/\(id).wav"))
          if t.count < frames { t += [Float](repeating: 0, count: frames - t.count) }
          targets.append(Array(t.prefix(frames)))
        } else {
          targets.append(try loadTarget(id, kind: "out", frames: frames))
        }
      }
      chunks.append(Chunk(ids: ids, targets: targets,
        invE: targets.map { t in Float(t.count) / max(t.reduce(0) { $0 + $1 * $1 }, 1e-9) },
        sT: data.rows(cs, data.s), wT: data.rows(cs, data.w), snareIdx: cs, vels: cs.map { data.vel[$0] }))
    }
    let bank = try NoiseBank.load()

    LazyGraphContext.reset()
    // One shared set of batch tensors; each step loads its chunk's data in place.
    precondition(chunks.allSatisfy { $0.B == chunks[0].B }, "--group must divide the render count")
    let G = chunks[0].B
    for c in chunks.indices {
      chunks[c].oneHot = chunks[c].snareIdx.flatMap { data.selRow($0) }   // snareIdx holds data row indices
    }
    // 1-D leaf bases (data updated in place), reshaped at each use inside the step graph.
    let tableB = Tensor(interleave(chunks[0].targets, frames: frames))
    let invET = Tensor(chunks[0].invE)
    let sTB = Tensor(chunks[0].sT), wTB = Tensor(chunks[0].wT)
    let selectB = Tensor(chunks[0].oneHot), velB = Tensor(chunks[0].vels)
    let uLeaf = Tensor.param([G, L.P], data: [Float](repeating: 0, count: G * L.P))
    let gUB = Tensor([Float](repeating: 0, count: G * L.P))
    var tableT: Tensor { tableB.reshape([frames, G]) }
    var sTT: Tensor { sTB.reshape([G, L.P]) }
    var wTT: Tensor { wTB.reshape([G, L.P]) }
    var selectT: Tensor { selectB.reshape([G, 32]) }
    var velT: Tensor { velB.reshape([G, 1]) }
    var loaded = -1
    func load(_ ci: Int) {
      guard ci != loaded else { return }
      let c = chunks[ci]
      tableB.updateDataLazily(interleave(c.targets, frames: frames)); invET.updateDataLazily(c.invE)
      sTB.updateDataLazily(c.sT); wTB.updateDataLazily(c.wT)
      selectB.updateDataLazily(c.oneHot); velB.updateDataLazily(c.vels)
      loaded = ci
    }
    let bankData = o["zero-noise"] == "true" ? [Float](repeating: 0, count: frames * bank.bands) : bank.timeMajor(frames: frames)
    let noiseTable = Tensor(bankData).reshape([frames, bank.bands])
    let dec = LatentDecoder(D: ck.D, H: ck.H, P: ck.P, codes: ck.codes, zInit: nil, b3Init: nil, initial: ck.tensors)
    var trainable: [Tensor] = []
    // --train readout (default): only the output layer W3/b3 moves; W1/W2/Z (the latent
    // geometry) stay at the pretrained values. Adam moves every weight by ~lr per step,
    // which in W1/W2 shifts every output, including mode frequencies that must stay within
    // ~1 cent (E5). --mask zeroes the gradient of whole segments' columns of W3/b3.
    let trainMode = o["train"] ?? "readout"
    if trainMode == "readout" { trainable = [dec.W3, dec.b3] } else {
      if !freezeDecoder { trainable += dec.decoderParams }
      if !freezeZ { trainable.append(dec.Z) }
    }
    let maskSegs = Set((o["mask"] ?? "mode_freq").split(separator: ",").map(String.init))
    var maskCols = [Bool](repeating: false, count: L.P)
    for sg in L.segments where maskSegs.contains(sg.name) { for k in sg.offset..<(sg.offset + sg.width) { maskCols[k] = true } }
    func applyMask() {
      guard maskCols.contains(true) else { return }
      for t in [dec.W3, dec.b3] {
        guard var g = t.grad?.getData() else { continue }
        for i in 0..<g.count where maskCols[i % L.P] { g[i] = 0 }
        t.grad = Tensor(g)
      }
    }
    let opt = Adam(params: trainable, lr: lr)
    let allIds = chunks.flatMap { $0.ids }, allTargets = chunks.flatMap { $0.targets }
    let NB = allIds.count

    let staticFrames = 64
    // Range box on random latent samples (see boxLoss), kept through the audio stages.
    let boxW = Float(o["box"] ?? "0")!, nBox = 64
    let (boxLo, boxHi) = boxBounds(data)
    let zbB = Tensor([Float](repeating: 0, count: nBox * dec.D))
    let vbB = Tensor((0..<nBox).map { [Float(0.4), 0.7, 1.0][$0 % 3] })
    var boxRng = SplitMix64(seed: 1234)
    var boxStep = 0
    func zRows(_ c: Chunk) -> [Float] {
      let Z = dec.Z.getData() ?? []
      return c.snareIdx.flatMap { row -> [Float] in
        let w = data.selRow(row)
        return (0..<dec.D).map { d in (0..<32).reduce(Float(0)) { $0 + w[$1] * Z[$1 * dec.D + d] } }
      }
    }
    /// Render from decoder outputs u (a leaf [B,P] tensor, or the live decoder graph).
    func renderFrom(_ u: Tensor, _ c: Chunk) -> LatentRender {
      LatentRenderer.render(L.squash(u, batch: c.B), batch: c.B, layout: L, noiseTable: stage == "out" ? noiseTable : nil,
        frames: frames, sampleRate: VillainData.sampleRate, includeNoise: stage == "out")
    }
    func evaluate() throws -> ([[Float]], [Float]) {
      var per: [[Float]] = []
      for (ci, c) in chunks.enumerated() {
        load(ci)
        LazyGraphContext.current.clearComputationGraph()
        uLeaf.updateDataLazily(dec.cpuForward(z: zRows(c), vel: c.vels))
        let r = renderFrom(uLeaf, c)
        per += deinterleave(try (stage == "head" ? r.head : r.output).realize(frames: frames), batch: c.B)
      }
      return (per, zip(per, allTargets).map { Scores.score($0, $1, windows: [64, 128, 256, 512, 1024, 2048]) })
    }
    /// Audio loss of chunk c as a function of a leaf u [B,P].
    func audioLoss(_ uLeaf: Tensor, _ c: Chunk) -> Signal {
      let r = renderFrom(uLeaf, c)
      let y = stage == "head" ? r.head : r.output
      let playhead = floor(Clock.seconds(frames: frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
      let target = tableT.sampleRow(playhead)
      var loss = mrstft(y, target, windows: windows, frames: frames)
      if wave > 0 {
        let d = y - target
        loss = loss + (d * d * invET).sum() * (wave / Float(c.B * frames))
      }
      return loss
    }
    /// Static part: sum(u * dL/du) (exact chain rule for the audio term) + anchors + z prior.
    func staticLoss(_ c: Chunk, gU: [Float]?, progress: Float) -> Tensor {
      let u = dec.forward(z: dec.codes(select: selectT), vel: velT)
      var reg = (dec.Z * dec.Z).sum() * (zl2 / Float(dec.codes * dec.D))
      if let gU { gUB.updateDataLazily(gU); reg = reg + (u * gUB.reshape([c.B, L.P])).sum() }
      let a = anchor + (anchorEnd - anchor) * progress
      if a > 0 { reg = reg + supervisedLossT(u: u, layout: L, sTarget: sTT, weights: wTT, batch: c.B) * a }
      if freqAnchor > 0 {
        // E5: head mode frequencies must stay within ~1 cent of the table through every stage.
        reg = reg + supervisedLossT(u: u, layout: L, sTarget: sTT, weights: wTT, batch: c.B, only: ["mode_freq"]) * freqAnchor
      }
      if boxW > 0 {
        zbB.updateDataLazily(sampleCodePrior(dec.Z.getData() ?? [], codes: dec.codes, D: dec.D, n: nBox, rng: &boxRng,
          spread: [Float(1), 1.5, 2][boxStep % 3]))
        boxStep += 1
        let ub = dec.forward(z: zbB.reshape([nBox, dec.D]), vel: vbB.reshape([nBox, 1]))
        reg = reg + boxLoss(u: ub, layout: L, lo: boxLo, hi: boxHi, batch: nBox) * boxW
      }
      return reg
    }
    /// One optimizer gradient: pass 1 audio backward to the u leaf (frame-summed dL/du),
    /// pass 2 static backward through the decoder. Returns (audio loss, static loss).
    /// (A single graph would backprop the per-frame adjoint through the static MLP per
    /// frame: [frames, H, P] tapes, > 2^31 floats even at B=12.)
    func twoPass(_ c: Chunk, progress: Float) throws -> (Float, Float) {
      LazyGraphContext.current.clearComputationGraph()
      uLeaf.updateDataLazily(dec.cpuForward(z: zRows(c), vel: c.vels))
      uLeaf.grad = nil
      let noAudio = ProcessInfo.processInfo.environment["E2_NO_AUDIO"] != nil
      let la = noAudio ? 0 : try audioLoss(uLeaf, c).backward(frames: frames).reduce(0, +)
      let gU = noAudio ? [Float](repeating: 0, count: c.B * L.P) : (uLeaf.grad?.getData() ?? [Float](repeating: 0, count: c.B * L.P))
      if ProcessInfo.processInfo.environment["E2_DEBUG_GU"] != nil {
        print("gU norm", gU.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot(), "count", gU.count, "grad nil?", uLeaf.grad == nil)
      }
      LazyGraphContext.current.clearComputationGraph()
      let ls = try staticToSignal(staticLoss(c, gU: gU, progress: progress), frames: staticFrames)
        .backward(frames: staticFrames).reduce(0, +)
      return (la, ls)
    }
    if o["dircheck"] == "true" {
      // Two-pass gradient vs central FD of (audio + static) along -g/|g| (chunk 0).
      let c = chunks[0]
      load(0)
      func total() throws -> Float {
        LazyGraphContext.current.clearComputationGraph()
        uLeaf.updateDataLazily(dec.cpuForward(z: zRows(c), vel: c.vels))
        let u = uLeaf
        let la = ProcessInfo.processInfo.environment["E2_NO_AUDIO"] != nil ? 0 : try audioLoss(u, c).realize(frames: frames).reduce(0, +)
        LazyGraphContext.current.clearComputationGraph()
        let ls = try staticToSignal(staticLoss(c, gU: nil, progress: 0), frames: staticFrames).realize(frames: staticFrames).reduce(0, +)
        return la + ls
      }
      opt.zeroGrad()
      let (la, ls) = try twoPass(c, progress: 0)
      let base = trainable.map { $0.getData() ?? [] }
      let grads = trainable.map { $0.grad?.getData() ?? [] }
      opt.zeroGrad()
      let gn = grads.flatMap { $0 }.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
      for alpha in (o["alphas"] ?? "1e-5,3e-5,1e-4,3e-4,1e-3").split(separator: ",").compactMap({ Float($0) }) {
        var lv: [Float] = []
        for sgn: Float in [1, -1] {
          for (i, t) in trainable.enumerated() { t.updateDataLazily(zip(base[i], grads[i]).map { $0 - sgn * alpha * $1 / gn }) }
          lv.append(try total())
        }
        let fd = (lv[0] - lv[1]) / (2 * alpha)
        print(String(format: "dircheck alpha %.0e: FD dL/ds %.6g vs AD -|g| %.6g (ratio %.4f), L0 %.6g", alpha, fd, -gn, fd / -gn, la + ls))
      }
      for (i, t) in trainable.enumerated() { t.updateDataLazily(base[i]) }
      return
    }
    let (_, initScores) = try evaluate()
    print(String(format: "[%@] init mean score %.4f (%d renders, %d chunks)", stage, initScores.reduce(0, +) / Float(NB), NB, nChunks))
    var bestMean = initScores.reduce(0, +) / Float(NB)
    var bestSnap = dec.snapshot(), bestStep = -1
    var csv = "step,loss,mean_score,median_score,sec\n"
    let t0 = Date()
    for step in 0..<steps {
      let ts = Date()
      let progress = Float(step) / Float(max(1, steps - 1))
      load(step % nChunks)
      let (la, ls) = try retryingGPU("step \(step)") { try twoPass(chunks[step % nChunks], progress: progress) }
      applyMask()
      let stepLoss = la  // audio loss (ls includes the u.dL/du chain-rule term)
      _ = ls
      opt.lr = cosineLR(lr, step: step, steps: steps, warmup: 10)
      opt.step(); opt.zeroGrad()
      let dt = Date().timeIntervalSince(ts)
      var sc = ","
      if (step + 1) % evalEvery == 0 || step == steps - 1 {
        let (_, scores) = try retryingGPU("eval") { try evaluate() }
        let mean = scores.reduce(0, +) / Float(NB), med = scores.sorted()[NB / 2]
        sc = String(format: "%.5f,%.5f", mean, med)
        if mean < bestMean { bestMean = mean; bestSnap = dec.snapshot(); bestStep = step }
        print(String(format: "[%@] step %d loss %.5f mean %.4f median %.4f (%.2f s/step)", stage, step, stepLoss, mean, med, dt))
      } else if step % 10 == 0 {
        print(String(format: "[%@] step %d loss %.5f (%.2f s/step)", stage, step, stepLoss, dt))
      }
      csv += "\(step),\(stepLoss),\(sc),\(dt)\n"
    }
    try csv.write(to: out.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
    dec.load(bestSnap)
    let (finalRender, finalScores) = try evaluate()
    for b in 0..<NB {
      try WAV.write(out.appendingPathComponent("\(allIds[b]).fit.wav"), finalRender[b])
      try WAV.write(out.appendingPathComponent("\(allIds[b]).target.wav"), allTargets[b])
    }
    try LatentCheckpoint.save(dec, names: (0..<32).map { data.names[$0 * 3] }, to: out,
      extra: ["stage": stage, "steps": steps, "best_step": bestStep, "lr": lr, "anchor": [anchor, anchorEnd],
              "freq_anchor": freqAnchor, "zl2": zl2, "group": group])
    try writeJSON(["ids": allIds, "init": initScores, "fit": finalScores, "mean_fit": finalScores.reduce(0, +) / Float(NB),
      "mean_init": initScores.reduce(0, +) / Float(NB), "best_step": bestStep, "wall_s": Date().timeIntervalSince(t0)],
      to: out.appendingPathComponent("summary.json"))
    print(String(format: "[%@] done: mean %.4f (init %.4f), best step %d, %.0f s", stage,
      finalScores.reduce(0, +) / Float(NB), initScores.reduce(0, +) / Float(NB), bestStep, Date().timeIntervalSince(t0)))
  }

  // MARK: latent-fitz (held-out: fresh z only, decoder frozen)

  static func fitz(_ o: [String: String]) throws {
    let steps = Int(o["steps"] ?? "150")!, lr = Float(o["lr"] ?? "0.05")!
    let zl2 = Float(o["zl2"] ?? "1e-3")!
    let frames = Int(o["frames"] ?? "48000")!
    let group = Int(o["group"] ?? "20")!
    let windows = (o["windows"] ?? "128,256,512,1024,2048").split(separator: ",").compactMap { Int($0) }
    let out = url(o["out"]!)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let L = try LatentLayout.load(o["layout"].map(url) ?? LatentLayout.defaultURL)
    let ck = try LatentCheckpoint.load(url(o["ckpt"]!))
    // targets + initial z rows: {"ids": [...], "vel": [...], "z0": [[...]...]}
    let spec = try JSONSerialization.jsonObject(with: Data(contentsOf: url(o["spec"]!))) as! [String: Any]
    let allIds = spec["ids"] as! [String]
    let allVels = (spec["vel"] as! [NSNumber]).map { $0.floatValue }
    let allZ0 = (spec["z0"] as! [[NSNumber]]).map { $0.map { $0.floatValue } }
    let D = ck.D, staticFrames = 64
    configureRuntime(frames: frames)
    LazyGraphContext.preserveCompilationCaches = true
    let bank = try NoiseBank.load()
    var rows: [String: Any] = [:]
    var csv = "chunk,step,audio_loss,mean_score\n"
    for c0 in stride(from: 0, to: allIds.count, by: group) {
      let idx = Array(c0..<min(c0 + group, allIds.count))
      let ids = idx.map { allIds[$0] }, vels = idx.map { allVels[$0] }, B = ids.count
      let targets = try ids.map { try loadTarget($0, kind: "out", frames: frames) }
      LazyGraphContext.reset()
      let targetTable = Tensor(interleave(targets, frames: frames)).reshape([frames, B])
      let noiseTable = Tensor(bank.timeMajor(frames: frames)).reshape([frames, bank.bands])
      let dec = LatentDecoder(D: D, H: ck.H, P: ck.P, codes: ck.codes, zInit: nil, b3Init: nil, initial: ck.tensors)
      let z = Tensor.param([B, D], data: idx.flatMap { allZ0[$0] })
      // --free-u: optimise the decoder OUTPUTS u directly per target (template capacity
      // bound), from spec "u0" rows; mode_freq columns stay fixed (no in-bin moves).
      let freeU = o["free-u"] == "true"
      let u0 = (spec["u0"] as? [[NSNumber]])?.map { $0.map { $0.floatValue } }
      let uFree = Tensor.param([B, L.P], data: freeU ? idx.flatMap { u0![$0] } : [Float](repeating: 0, count: B * L.P))
      let opt = Adam(params: [freeU ? uFree : z], lr: lr)
      var freqCols = [Bool](repeating: false, count: L.P)
      for sg in L.segments where sg.name == "mode_freq" { for k in sg.offset..<(sg.offset + sg.width) { freqCols[k] = true } }
      let uLeaf = Tensor.param([B, L.P], data: [Float](repeating: 0, count: B * L.P))
      let gUT = Tensor([Float](repeating: 0, count: B * L.P)).reshape([B, L.P])
      let velT = Tensor(vels).reshape([B, 1])
      func renderU(_ u: Tensor) -> LatentRender {
        LatentRenderer.render(L.squash(u, batch: B), batch: B, layout: L, noiseTable: noiseTable, frames: frames,
          sampleRate: VillainData.sampleRate)
      }
      func currentU() -> [Float] { freeU ? (uFree.getData() ?? []) : dec.cpuForward(z: z.getData() ?? [], vel: vels) }
      func scores() throws -> ([[Float]], [Float]) {
        LazyGraphContext.current.clearComputationGraph()
        uLeaf.updateDataLazily(currentU())
        let per = deinterleave(try renderU(uLeaf).output.realize(frames: frames), batch: B)
        return (per, zip(per, targets).map { Scores.score($0, $1, windows: [64, 128, 256, 512, 1024, 2048]) })
      }
      let (initPer, initScores) = try scores()
      var best = initScores, bestPer = initPer
      var bestZ = z.getData() ?? []
      for step in 0..<steps {
        LazyGraphContext.current.clearComputationGraph()
        uLeaf.updateDataLazily(currentU())
        uLeaf.grad = nil
        let playhead = floor(Clock.seconds(frames: frames, sampleRate: VillainData.sampleRate) * VillainData.sampleRate + 0.5)
        let la = try mrstft(renderU(uLeaf).output, targetTable.sampleRow(playhead), windows: windows, frames: frames)
          .backward(frames: frames).reduce(0, +)
        var gU = uLeaf.grad?.getData() ?? [Float](repeating: 0, count: B * L.P)
        if freeU {
          for i in 0..<gU.count where freqCols[i % L.P] { gU[i] = 0 }
          uFree.grad = Tensor(gU)
        } else {
          LazyGraphContext.current.clearComputationGraph()
          let u = dec.forward(z: z, vel: velT)
          gUT.updateDataLazily(gU)
          let st = (u * gUT).sum() + (z * z).sum() * (zl2 / Float(B * D))
          _ = try staticToSignal(st, frames: staticFrames).backward(frames: staticFrames)
        }
        // the decoder weights are params too (loaded from the checkpoint); only z steps
        opt.lr = cosineLR(lr, step: step, steps: steps)
        opt.step(); opt.zeroGrad()
        for t in dec.decoderParams + [dec.Z] { t.grad = nil }
        var sc = ""
        if (step + 1) % 25 == 0 || step == steps - 1 {
          let (per, s) = try scores()
          let zc = z.getData() ?? []
          for b in 0..<B where s[b] < best[b] {
            best[b] = s[b]; bestPer[b] = per[b]
            for d in 0..<D { bestZ[b * D + d] = zc[b * D + d] }
          }
          sc = String(format: "%.5f", s.reduce(0, +) / Float(B))
          print(String(format: "fitz chunk %d step %d audio %.4f mean %.4f (best-per-lane %.4f)", c0 / group, step, la,
            s.reduce(0, +) / Float(B), best.reduce(0, +) / Float(B)))
        }
        csv += "\(c0 / group),\(step),\(la),\(sc)\n"
      }
      for b in 0..<B {
        try WAV.write(out.appendingPathComponent("\(ids[b]).fit.wav"), bestPer[b])
        try WAV.write(out.appendingPathComponent("\(ids[b]).target.wav"), targets[b])
        rows[ids[b]] = ["init": initScores[b], "fit": best[b], "z0": allZ0[idx[b]], "z": Array(bestZ[(b * D)..<((b + 1) * D)])]
      }
    }
    try csv.write(to: out.appendingPathComponent("loss.csv"), atomically: true, encoding: .utf8)
    try writeJSON(rows, to: out.appendingPathComponent("summary.json"))
    let fits = rows.values.compactMap { ($0 as? [String: Any])?["fit"] as? Float }
    let inits = rows.values.compactMap { ($0 as? [String: Any])?["init"] as? Float }
    print(String(format: "fitz done: mean %.4f (init %.4f) over %d", fits.reduce(0, +) / Float(fits.count),
      inits.reduce(0, +) / Float(inits.count), fits.count))
  }

  // MARK: latent-render (arbitrary z rows; trainer side of the parity check)

  static func render(_ o: [String: String]) throws {
    let frames = Int(o["frames"] ?? "48000")!
    let out = url(o["out"]!)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let L = try LatentLayout.load(o["layout"].map(url) ?? LatentLayout.defaultURL)
    let ck = try LatentCheckpoint.load(url(o["ckpt"]!))
    let spec = try JSONSerialization.jsonObject(with: Data(contentsOf: url(o["spec"]!))) as! [String: Any]
    let names = spec["names"] as! [String]
    let vels = (spec["vel"] as! [NSNumber]).map { $0.floatValue }
    let zs = (spec["z"] as! [[NSNumber]]).flatMap { $0.map { $0.floatValue } }
    let B = names.count
    configureRuntime(frames: frames)
    let bank = try NoiseBank.load()
    LazyGraphContext.reset()
    let noiseTable = Tensor(bank.timeMajor(frames: frames)).reshape([frames, bank.bands])
    let dec = LatentDecoder(D: ck.D, H: ck.H, P: ck.P, codes: ck.codes, zInit: nil, b3Init: nil, initial: ck.tensors)
    let p = L.squash(dec.forward(z: Tensor(zs).reshape([B, ck.D]), vel: Tensor(vels).reshape([B, 1])), batch: B)
    let r = LatentRenderer.render(p, batch: B, layout: L, noiseTable: noiseTable, frames: frames, sampleRate: VillainData.sampleRate)
    let y = deinterleave(try r.output.realize(frames: frames), batch: B)
    LazyGraphContext.current.clearComputationGraph()
    let p2 = L.squash(dec.forward(z: Tensor(zs).reshape([B, ck.D]), vel: Tensor(vels).reshape([B, 1])), batch: B)
    let r2 = LatentRenderer.render(p2, batch: B, layout: L, noiseTable: nil, frames: frames, sampleRate: VillainData.sampleRate, includeNoise: false)
    let h = deinterleave(try r2.head.realize(frames: frames), batch: B)
    for b in 0..<B {
      try WAV.write(out.appendingPathComponent("\(names[b]).swift.wav"), y[b])
      try WAV.write(out.appendingPathComponent("\(names[b]).swift_head.wav"), h[b])
    }
    print("rendered \(B) -> \(out.path)")
  }
}

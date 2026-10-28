import DGen
import DGenLazy

typealias LTensor = DGenLazy.Tensor
import Foundation

// ModalKickAD: the Latent Kick trainer-side renderer (K1). Commands:
//   parity  --ids kick_00_v100,sweep_11_tune_0.5,... [--backend c|metal] [--frames N]
//           renders each id's teacher row (kick/data/teacher.json, kick_KK_v100) with that
//           id's velocity / pitch / knobs; writes <tap>.f32 per id to kick/data/swift_parity/
//           (kick/scripts/swift_parity.py compares with kick_np trainer mode + VILLAIN stems)
//   fdcheck [--backend c|metal] [--frames N] [--ids a,b] [--per-group N] [--eps E]
//           MSE(out) + MSE(modal) + MSE(noise) at a perturbed teacher point, autograd wrt the
//           decoder-output leaf u [B,P] vs central differences, per layout segment.

struct Args {
  var cmd = "parity"
  var kv: [String: String] = [:]
  init() {
    let a = CommandLine.arguments
    if a.count > 1 { cmd = a[1] }
    var i = 2
    while i < a.count {
      if a[i].hasPrefix("--"), i + 1 < a.count { kv[String(a[i].dropFirst(2))] = a[i + 1]; i += 2 } else { i += 1 }
    }
  }
  func s(_ k: String, _ d: String) -> String { kv[k] ?? d }
  func i(_ k: String, _ d: Int) -> Int { kv[k].flatMap(Int.init) ?? d }
  func f(_ k: String, _ d: Float) -> Float { kv[k].flatMap(Float.init) ?? d }
}

struct ManifestItem {
  var id: String
  var vel: Double
  var pitch: Double
  var params: [String: Double]
}

func loadManifest() throws -> [String: ManifestItem] {
  let url = KickPaths.data.appendingPathComponent("villain/manifest.json")
  let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
  var out: [String: ManifestItem] = [:]
  for it in j["items"] as! [[String: Any]] {
    let id = it["id"] as! String
    var p: [String: Double] = [:]
    for (k, v) in it["params"] as! [String: Any] { p[k] = (v as! NSNumber).doubleValue }
    out[id] = ManifestItem(
      id: id, vel: (it["vel"] as! NSNumber).doubleValue, pitch: (it["pitch_hz"] as! NSNumber).doubleValue,
      params: p)
  }
  return out
}

/// teacher row of an id: kick_KK_* and sweep_KK_* use kick_KK_v100; blends/pc their own row
func teacherKey(_ id: String) -> String {
  if id.hasPrefix("kick_") || id.hasPrefix("sweep_") {
    let kk = id.split(separator: "_")[1]
    return "kick_\(kk)_v100"
  }
  return id
}

func hitFor(_ m: ManifestItem) -> KickHit {
  var knobs: [String: Double] = [:]
  for (k, v) in m.params where KickHit.defaults[k] != nil { knobs[k] = v }
  return KickHit(vel: m.vel, pitch: m.pitch, knobs: knobs)
}

func configure(_ backend: String, frames: Int) {
  DGenConfig.backend = backend == "metal" ? .metal : .c
  DGenConfig.sampleRate = 48_000
  DGenConfig.maxFrameCount = frames
  DGenConfig.debug = false
}

func deinterleave(_ flat: [Float], batch B: Int) -> [[Float]] {
  let n = flat.count / max(1, B)
  return (0..<B).map { b in (0..<n).map { flat[$0 * B + b] } }
}

struct Graph {
  var u: LTensor
  var render: KickRender
}

func buildGraph(layout: KickLayout, tables: KickTables, U: [Float], hits: [KickHit], frames: Int, param: Bool) -> Graph {
  let B = hits.count
  let bandT = LTensor(Array(tables.bandsTimeMajor.prefix(frames * tables.bands))).reshape([frames, tables.bands])
  let humT = LTensor(Array(tables.hum.prefix(frames))).reshape([frames, 1])
  let u = param ? LTensor.param([B, layout.P], data: U) : LTensor(U).reshape([B, layout.P])
  let nat = layout.squash(u, batch: B)
  let r = KickVoice.render(nat: nat, layout: layout, hits: hits, tables: bandT, hum: humT, frames: frames, sampleRate: 48_000)
  return Graph(u: u, render: r)
}

func runParity(_ a: Args) throws {
  let frames = a.i("frames", 48_000)
  configure(a.s("backend", "c"), frames: frames)
  let layout = try KickLayout.load()
  let teacher = try KickTeacher.load()
  let man = try loadManifest()
  let tables = try KickTables.load(bands: layout.NB)
  let ids = a.s("ids", "kick_00_v100,kick_11_v100,kick_29_v100").split(separator: ",").map(String.init)
  var U: [Float] = []
  var hits: [KickHit] = []
  for id in ids {
    guard let m = man[id], let u = teacher[teacherKey(id)] else { throw NSError(domain: "kick", code: 1, userInfo: [NSLocalizedDescriptionKey: "unknown id \(id)"]) }
    U += u
    hits.append(hitFor(m))
  }
  LazyGraphContext.reset()
  let g = buildGraph(layout: layout, tables: tables, U: U, hits: hits, frames: frames, param: false)
  let outDir = KickPaths.data.appendingPathComponent("swift_parity")
  try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
  let t0 = Date()
  for (name, sig) in [("out", g.render.out), ("modal", g.render.modal), ("noise", g.render.noise)] {
    let flat = try sig.realize(frames: frames)
    for (b, y) in deinterleave(flat, batch: ids.count).enumerated() {
      let d = y.withUnsafeBufferPointer { Data(buffer: $0) }
      try d.write(to: outDir.appendingPathComponent("\(ids[b]).\(name).f32"))
    }
  }
  print(String(format: "parity B=%d frames=%d backend=%@: %.2f s (3 realizes incl. compile)", ids.count, frames,
               a.s("backend", "c"), Date().timeIntervalSince(t0)))
}

struct SplitMix {
  var s: UInt64
  mutating func next() -> Double {
    s &+= 0x9E37_79B9_7F4A_7C15
    var z = s
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return Double(z ^ (z >> 31)) / Double(UInt64.max)
  }
}

func runFDCheck(_ a: Args) throws {
  let frames = a.i("frames", 4096)
  configure(a.s("backend", "c"), frames: frames)
  LazyGraphContext.preserveCompilationCaches = false
  let layout = try KickLayout.load()
  let teacher = try KickTeacher.load()
  let man = try loadManifest()
  let tables = try KickTables.load(bands: layout.NB)
  let ids = a.s("ids", "kick_03_v100,kick_29_v070").split(separator: ",").map(String.init)
  let B = ids.count, P = layout.P
  var U0: [Float] = [], hits: [KickHit] = []
  for id in ids { U0 += teacher[teacherKey(id)]!; hits.append(hitFor(man[id]!)) }
  // targets = the teacher render; evaluate at a jittered point so every group has a gradient
  LazyGraphContext.reset()
  let tg = buildGraph(layout: layout, tables: tables, U: U0, hits: hits, frames: frames, param: false)
  let tOut = try tg.render.out.realize(frames: frames)
  let tModal = try tg.render.modal.realize(frames: frames)
  let tNoise = try tg.render.noise.realize(frames: frames)
  var rng = SplitMix(s: 11)
  let jitter = a.f("jitter", 0.15)
  let U = U0.map { $0 + Float((rng.next() * 2 - 1)) * jitter }
  func loss(_ Uv: [Float], param: Bool) -> (Graph, Signal) {
    LazyGraphContext.reset()
    let g = buildGraph(layout: layout, tables: tables, U: Uv, hits: hits, frames: frames, param: param)
    let ph = floor(KickClock.seconds(frames: frames, sampleRate: 48_000) * 48_000 + 0.5)
    func tab(_ x: [Float]) -> SignalTensor { LTensor(x).reshape([frames, B]).sampleRow(ph) }
    let d1 = g.render.out - tab(tOut), d2 = g.render.modal - tab(tModal), d3 = g.render.noise - tab(tNoise)
    let l = (d1 * d1).sum() + (d2 * d2).sum() + (d3 * d3).sum() * 100.0
    return (g, l * (1 / Float(B)))
  }
  func value(_ Uv: [Float]) throws -> Double {
    try loss(Uv, param: false).1.realize(frames: frames).reduce(0.0) { $0 + Double($1) }
  }
  let t0 = Date()
  let (gAD, lAD) = loss(U, param: true)
  _ = try lAD.backward(frames: frames)
  guard let ad = gAD.u.grad?.getData() else { throw NSError(domain: "kick", code: 2, userInfo: [NSLocalizedDescriptionKey: "no gradient"]) }
  let L0 = try value(U)
  print(String(format: "loss at point %.6g (backward+forward %.1f s)", L0, Date().timeIntervalSince(t0)))
  let eps = a.f("eps", 1e-3)
  let perGroup = a.i("per-group", 3)
  let maxAll = ad.map { Swift.abs($0) }.max() ?? 0
  var report: [[String: Any]] = []
  var pass = 0, total = 0
  for seg in layout.segments {
    var idx: [Int] = []
    for b in 0..<B { for k in 0..<seg.width { idx.append(b * P + seg.offset + k) } }
    let sorted = idx.sorted { Swift.abs(ad[$0]) > Swift.abs(ad[$1]) }
    var pick = Array(sorted.prefix(perGroup))
    if let r = idx.randomElement(using: &rngG) { if !pick.contains(r) { pick.append(r) } }
    var fd: [Float] = [], adv: [Float] = []
    for j in pick {
      var up = U, dn = U
      up[j] += eps; dn[j] -= eps
      fd.append(Float((try value(up) - (try value(dn))) / Double(2 * eps)))
      adv.append(ad[j])
    }
    let dot: Double = zip(fd, adv).map { Double($0) * Double($1) }.reduce(0, +)
    let nf: Double = fd.map { Double($0 * $0) }.reduce(0, +).squareRoot()
    let na: Double = adv.map { Double($0 * $0) }.reduce(0, +).squareRoot()
    let inert = nf < 1e-6 * Double(maxAll) && na < 1e-6 * Double(maxAll)
    let cos: Double = inert ? 1 : dot / Swift.max(nf * na, 1e-30)
    let errSum: Double = zip(fd, adv).map { Swift.abs(Double($0) - Double($1)) }.reduce(0, +)
    let fdSum: Double = fd.map { Swift.abs(Double($0)) }.reduce(0, +)
    let rel: Double = errSum / Swift.max(fdSum, 1e-30)
    let ok = inert || cos > 0.999
    total += 1; if ok { pass += 1 }
    print(String(format: "%-14@ cos %.6f rel %.2e %@ |ad| %.3g", seg.name, cos, rel, inert ? "(inert)" : (ok ? "" : "FAIL"), na))
    report.append(["segment": seg.name, "cos": cos, "rel": rel, "inert": inert, "pass": ok, "fd": fd.map(Double.init), "ad": adv.map(Double.init), "idx": pick])
  }
  print("fdcheck: \(pass)/\(total) segments pass (cos > 0.999 or inert)")
  let out = KickPaths.data.appendingPathComponent("fdcheck_\(a.s("backend", "c")).json")
  try JSONSerialization.data(withJSONObject: ["ids": ids, "frames": frames, "eps": eps, "loss": L0, "pass": pass, "total": total, "segments": report], options: [.prettyPrinted]).write(to: out)
}

/// realise a few static post-decoder tensors per lane (debugging the knob paths)
func runProbe(_ a: Args) throws {
  configure(a.s("backend", "c"), frames: 64)
  let layout = try KickLayout.load()
  let teacher = try KickTeacher.load()
  let man = try loadManifest()
  let ids = a.s("ids", "kick_11_v100,sweep_11_tilt_1.5,sweep_11_spread_1.5").split(separator: ",").map(String.init)
  var U: [Float] = [], hits: [KickHit] = []
  for id in ids { U += teacher[teacherKey(id)]!; hits.append(hitFor(man[id]!)) }
  let B = ids.count, K = layout.K
  LazyGraphContext.reset()
  let u = LTensor(U).reshape([B, layout.P])
  let p = layout.squash(u, batch: B)
  let lf0 = log(p["mode_f"]!)
  let lfsub = lf0.shrink([(0, B), (0, 1)]).expand([B, K])
  var kmulData = [Float](repeating: 0, count: B * K)
  for (b, h) in hits.enumerated() {
    kmulData[b * K] = 1; kmulData[b * K + 1] = Float(h.k("beat"))
    for i in 2..<K { kmulData[b * K + i] = Float(h.k("spread")) }
  }
  let kmul = LTensor(kmulData).reshape([B, K])
  let lf = lfsub + kmul * (lf0 - lfsub)
  let tilt = LTensor(hits.map { Float($0.k("tilt")) }).reshape([B, 1]).expand([B, K])
  for (name, t) in [("lf0", lf0), ("lfsub", lfsub), ("kmul", kmul), ("lf", lf), ("tilt", tilt), ("tiltlf", tilt * (lf - lfsub))] {
    let v = try t.realize()
    for b in 0..<B { print(name, ids[b], v[(b * K)..<(b * K + 4)].map { String(format: "%.4f", $0) }.joined(separator: " ")) }
  }
}

/// minimal repro: static [B,K] tensors derived from per-lane constants, read by a frame-varying kernel
func runProbe2(_ a: Args) throws {
  let frames = 16
  configure(a.s("backend", "c"), frames: frames)
  let B = 2, K = 4
  let base: [Float] = [1, 2, 3, 4, 10, 20, 30, 40]
  func check(_ name: String, _ build: (Signal) -> SignalTensor, _ expect: (Int, Float) -> Float) throws {
    LazyGraphContext.reset()
    let time = KickClock.seconds(frames: frames, sampleRate: 48_000) * 48_000  // frame index
    let y = build(time)
    let flat = try y.realize(frames: frames)
    let t = 5
    let got = (0..<B).map { flat[t * B + $0] }
    let exp = (0..<B).map { expect($0, Float(t)) }
    print(name, "got", got, "expected", exp, zip(got, exp).allSatisfy { Swift.abs($0 - $1) < 1e-3 * Swift.max(1, Swift.abs($1)) } ? "OK" : "WRONG")
  }
  let lane: [Float] = [1, 1.5]
  // A: per-lane constant expanded along K, times a [B,K] constant, consumed per frame
  try check("A expand*const", { time in
    let c = LTensor(lane).reshape([B, 1]).expand([B, K])
    let d = c * LTensor(base).reshape([B, K])
    return (time * d).sum(axis: 1)
  }, { b, t in t * lane[b] * base[(b * K)..<(b * K + K)].reduce(0, +) })
  try check("D shrink.expand direct", { time in
    let x = LTensor(base).reshape([B, K])
    return (time * x.shrink([(0, B), (0, 1)]).expand([B, K])).sum(axis: 1)
  }, { b, t in t * Float(K) * base[b * K] })
  try check("E log->shrink.expand", { time in
    let x = log(LTensor(base).reshape([B, K]))
    return (time * x.shrink([(0, B), (0, 1)]).expand([B, K])).sum(axis: 1)
  }, { b, t in t * Float(K) * Foundation.log(base[b * K]) })
  try check("F x - shrink.expand", { time in
    let x = LTensor(base).reshape([B, K])
    return (time * (x - x.shrink([(0, B), (0, 1)]).expand([B, K]))).sum(axis: 1)
  }, { b, t in var s: Float = 0; for k in 0..<K { s += base[b * K + k] - base[b * K] }; return t * s })
  try check("G shrink col1.expand", { time in
    let x = LTensor(base).reshape([B, K])
    return (time * x.shrink([(0, B), (1, 2)]).expand([B, K])).sum(axis: 1)
  }, { b, t in t * Float(K) * base[b * K + 1] })
  // B: non-uniform-along-K data tensor times derived tensor
  let km: [Float] = [1, 1, 1, 1, 1, 1, 1.5, 1.5]
  try check("B kmul*derived", { time in
    let x = log(LTensor(base).reshape([B, K]))
    let sub = x.shrink([(0, B), (0, 1)]).expand([B, K])
    let lf = sub + LTensor(km).reshape([B, K]) * (x - sub)
    return (time * lf.exp()).sum(axis: 1)
  }, { b, t in
    var s: Float = 0
    for k in 0..<K { let x = Foundation.log(base[b * K + k]), sub = Foundation.log(base[b * K]); s += Foundation.exp(sub + km[b * K + k] * (x - sub)) }
    return t * s })
  // C: same as B, consumed through exp(-t * v) (the idec path)
  try check("C exp(-t*derived)", { time in
    let x = log(LTensor(base).reshape([B, K]))
    let sub = x.shrink([(0, B), (0, 1)]).expand([B, K])
    let til = LTensor(lane).reshape([B, 1]).expand([B, K])
    let v = (til * (x - sub) * -1.0).exp()
    return exp((time * 0.01) * v * -1.0).sum(axis: 1)
  }, { b, t in
    var s: Float = 0
    for k in 0..<K { let x = Foundation.log(base[b * K + k]), sub = Foundation.log(base[b * K]); s += Foundation.exp(-t * 0.01 * Foundation.exp(-lane[b] * (x - sub))) }
    return s })
}

var rngG = SystemRandomNumberGenerator()
let args = Args()
do {
  switch args.cmd {
  case "parity": try runParity(args)
  case "fdcheck": try runFDCheck(args)
  case "probe": try runProbe(args)
  case "probe2": try runProbe2(args)
  default: print("usage: ModalKickAD parity|fdcheck [--key value ...]")
  }
} catch {
  print("error: \(error)")
  exit(1)
}

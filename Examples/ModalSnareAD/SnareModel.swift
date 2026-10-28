import DGenLazy
import Foundation

// MARK: - Bounded parameterizations

/// x = exp(log lo + sigmoid(raw) (log hi - log lo)): bounded, log-uniform in raw.
func logMap(_ raw: Tensor, _ lo: Float, _ hi: Float) -> Tensor {
  let a = Foundation.log(lo)
  let s = Foundation.log(hi) - a
  return (raw.sigmoid() * s + a).exp()
}

func logMapInverse(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
  let a = Foundation.log(lo), s = Foundation.log(hi) - a
  let p = min(max((Foundation.log(max(x, 1e-30)) - a) / s, 1e-4), 1 - 1e-4)
  return Foundation.log(p / (1 - p))
}

func logitInverse(_ p: Float) -> Float {
  let q = min(max(p, 1e-4), 1 - 1e-4)
  return Foundation.log(q / (1 - q))
}

func sigmoidF(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }

func logMapCPU(_ raw: Float, _ lo: Float, _ hi: Float) -> Float {
  Foundation.exp(Foundation.log(lo) + sigmoidF(raw) * (Foundation.log(hi) - Foundation.log(lo)))
}

// MARK: - Band noise bank (precomputed constants, see e1/make_noise_bands.py)

struct NoiseBank {
  var bands: Int
  var samples: Int
  var centers: [Float]
  /// Row-major [bands, samples].
  var data: [Float]

  static func load(dir: URL = Paths.e1.appendingPathComponent("noise_bands")) throws -> NoiseBank {
    let spec = try JSONSerialization.jsonObject(
      with: Data(contentsOf: dir.appendingPathComponent("bands.json"))) as! [String: Any]
    let bandSpecs = spec["bands"] as! [[String: Any]]
    let samples = (spec["samples"] as! NSNumber).intValue
    let raw = try Data(contentsOf: dir.appendingPathComponent("bands.f32"))
    let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    precondition(floats.count == bandSpecs.count * samples, "bands.f32 size mismatch")
    return NoiseBank(
      bands: bandSpecs.count, samples: samples,
      centers: bandSpecs.map { ($0["fc"] as! NSNumber).floatValue }, data: floats)
  }

  /// Time-major [frames, bands] for sampleRow(playhead).
  func timeMajor(frames: Int) -> [Float] {
    var out = [Float](repeating: 0, count: frames * bands)
    for t in 0..<min(frames, samples) {
      for b in 0..<bands { out[t * bands + b] = data[b * samples + t] }
    }
    return out
  }
}

// MARK: - Ranges

enum Ranges {
  static let noiseAmp: (Float, Float) = (1e-5, 2.0)
  static let noiseD1: (Float, Float) = (0.002, 0.4)
  static let noiseD2: (Float, Float) = (0.02, 2.0)
  static let noiseCouple: (Float, Float) = (1e-5, 20.0)
  static let noiseRise: (Float, Float) = (0.00005, 0.02)
  static let noiseOnsetMax: Float = 0.05
  static let drive: (Float, Float) = (1.0, 6.0)
  static let outGain: (Float, Float) = (0.05, 8.0)
  // Head residuals when unfrozen: multiplicative, bounded.
  static let headFreqSpan: Float = 0.03  // f * exp(0.03 tanh(r))  (+-3%)
  static let headAmpSpan: Float = 2.0  // amp * exp(2 tanh(r))
  static let headDecSpan: Float = 1.0  // idec * exp(tanh(r))
}

// MARK: - Per-render constants from the VILLAIN table

struct SnareConstants {
  var B: Int
  var head: HeadArrays
  var gin: [Float]  // vel^-c_vin
  var lenS: [Float], fadeS: [Float], fadeCurve: [Float]
  var dv: [Float], dAmt: [Float], dNorm: [Float]
  var outGain: [Float]  // lvl * vel^c_vout
  var pre: [Float]

  init(renders: [VillainRender], K: Int = 32) {
    B = renders.count
    head = HeadArrays(renders: renders, K: K)
    func vp(_ r: VillainRender, _ e: Double) -> Float {
      Float(r.vel > 0 ? Foundation.exp(e * Foundation.log(max(r.vel, 1e-4))) : 0)
    }
    gin = renders.map { vp($0, -$0.c("c_vin")) }
    lenS = renders.map { Float($0.global["len_s"] ?? 1) }
    fadeS = renders.map { Float(max($0.global["fade_s"] ?? 0.02, 1e-4)) }
    fadeCurve = renders.map { Float($0.c("c_fade_curve")) }
    dv = renders.map { Float($0.global["dv"] ?? 1) }
    dAmt = renders.map { Float($0.global["d_amt"] ?? 0) }
    dNorm = renders.map { Float($0.global["d_norm"] ?? 1) }
    outGain = renders.map { Float($0.global["lvl"] ?? 1) * vp($0, $0.c("c_vout")) }
    pre = renders.map { Float($0.c("c_pre")) }
  }
}

// MARK: - Trainable parameters (raw, unbounded; mapped through bounded transforms)

final class SnareParams {
  let B: Int, K: Int, NB: Int
  // Noise [B, NB]
  let rawA: Tensor, rawD1: Tensor, rawL: Tensor, rawD2: Tensor, rawC: Tensor
  // Noise [B, 1]
  let rawRise: Tensor, rawOnset: Tensor
  // Output [B]
  let rawDrive: Tensor, rawDAmt: Tensor, rawGain: Tensor
  // Head residuals [B, K] (zero = table); optimized only when unfrozen
  let rawHF: Tensor, rawHA: Tensor, rawHD: Tensor

  var noiseGroups: [(String, Tensor)] {
    [("noise_amp", rawA), ("noise_d1", rawD1), ("noise_tail", rawL), ("noise_d2", rawD2),
     ("noise_couple", rawC), ("noise_rise", rawRise), ("noise_onset", rawOnset)]
  }
  var outputGroups: [(String, Tensor)] {
    [("drive", rawDrive), ("drive_amt", rawDAmt), ("out_gain", rawGain)]
  }
  var headGroups: [(String, Tensor)] {
    [("head_freq", rawHF), ("head_amp", rawHA), ("head_dec", rawHD)]
  }
  var allGroups: [(String, Tensor)] { noiseGroups + outputGroups + headGroups }

  /// Must be created after LazyGraphContext.reset().
  init(constants c: SnareConstants, bands NB: Int, initial: [String: [Float]] = [:]) {
    B = c.B
    K = c.head.K
    self.NB = NB
    func p(_ name: String, _ shape: [Int], _ fallback: @autoclosure () -> [Float]) -> Tensor {
      let n = shape.reduce(1, *)
      if let v = initial[name], v.count == n { return Tensor.param(shape, data: v) }
      return Tensor.param(shape, data: fallback())
    }
    let bn = B * NB
    rawA = p("noise_amp", [B, NB], [Float](repeating: logMapInverse(0.01, Ranges.noiseAmp.0, Ranges.noiseAmp.1), count: bn))
    rawD1 = p("noise_d1", [B, NB], [Float](repeating: logMapInverse(0.03, Ranges.noiseD1.0, Ranges.noiseD1.1), count: bn))
    rawL = p("noise_tail", [B, NB], [Float](repeating: logitInverse(0.15), count: bn))
    rawD2 = p("noise_d2", [B, NB], [Float](repeating: logMapInverse(0.2, Ranges.noiseD2.0, Ranges.noiseD2.1), count: bn))
    rawC = p("noise_couple", [B, NB], [Float](repeating: logMapInverse(0.01, Ranges.noiseCouple.0, Ranges.noiseCouple.1), count: bn))
    rawRise = p("noise_rise", [B, 1], [Float](repeating: logMapInverse(0.0005, Ranges.noiseRise.0, Ranges.noiseRise.1), count: B))
    rawOnset = p("noise_onset", [B, 1], c.pre.map { logitInverse(max($0, 0.0005) / Ranges.noiseOnsetMax) })
    rawDrive = p("drive", [B], c.dv.map { logMapInverse(max($0, 1.0005), Ranges.drive.0, Ranges.drive.1) })
    rawDAmt = p("drive_amt", [B], c.dAmt.map { logitInverse(max($0, 0.001)) })
    rawGain = p("out_gain", [B], c.outGain.map { logMapInverse($0, Ranges.outGain.0, Ranges.outGain.1) })
    let z = [Float](repeating: 0, count: B * K)
    rawHF = p("head_freq", [B, K], z)
    rawHA = p("head_amp", [B, K], z)
    rawHD = p("head_dec", [B, K], z)
  }

  func snapshot() -> [String: [Float]] {
    var out: [String: [Float]] = [:]
    for (n, t) in allGroups { out[n] = t.getData() ?? [] }
    return out
  }

  func load(_ values: [String: [Float]]) {
    for (n, t) in allGroups { if let v = values[n] { t.updateDataLazily(v) } }
  }
}

struct RenderOptions {
  var includeHead = true
  var includeNoise = true
  var includeCoupling = true
  var trainHead = false
}

struct SnareRender {
  var output: SignalTensor  // [B]
  var head: SignalTensor  // [B]
  var noise: SignalTensor  // [B]
}

enum SnareRenderer {
  /// Full closed-form, frame-parallel render. Every op is elementwise in a
  /// time-varying `t`, so the graph has no state cells.
  static func render(
    params p: SnareParams, constants c: SnareConstants, noiseTable: Tensor, frames: Int,
    sampleRate: Float, options o: RenderOptions = RenderOptions()
  ) -> SnareRender {
    let B = c.B, K = c.head.K, NB = p.NB
    let time = Clock.seconds(frames: frames, sampleRate: sampleRate)

    // ---- head ----
    var overrides: [String: Tensor] = [:]
    if o.trainHead {
      func base(_ k: String) -> Tensor { Tensor(c.head.fields[k]!).reshape([B, K]) }
      overrides["f"] = base("f") * (tanh(p.rawHF) * Ranges.headFreqSpan).exp()
      overrides["amp"] = base("amp") * (tanh(p.rawHA) * Ranges.headAmpSpan).exp()
      overrides["idec"] = base("idec") * tanh(p.rawHD).exp()
      // Keep the glide clamp sh >= -0.9 f consistent with the moved f: scale sh with f.
      overrides["sh"] = base("sh") * (tanh(p.rawHF) * Ranges.headFreqSpan).exp()
    }
    let ht = c.head.tensors(overrides: overrides)
    let parts = HeadRenderer.parts(ht, time: time, sampleRate: sampleRate)
    let head = parts.modes.sum(axis: 1)  // [B]

    // ---- noise ----
    var noise: SignalTensor
    if o.includeNoise {
      let onset = (p.rawOnset.sigmoid() * Ranges.noiseOnsetMax).expand([B, NB])
      let t = relu(time - onset)  // [B, NB]
      let A = logMap(p.rawA, Ranges.noiseAmp.0, Ranges.noiseAmp.1)
      let iD1 = 1.0 / logMap(p.rawD1, Ranges.noiseD1.0, Ranges.noiseD1.1)
      let iD2 = 1.0 / logMap(p.rawD2, Ranges.noiseD2.0, Ranges.noiseD2.1)
      let L = p.rawL.sigmoid()
      let iRise = (1.0 / logMap(p.rawRise, Ranges.noiseRise.0, Ranges.noiseRise.1)).expand([B, NB])
      let attack: SignalTensor = exp(t * (iRise * -1.0)) * -1.0 + 1.0
      let fast: SignalTensor = exp(t * (iD1 * -1.0)) * (L * -1.0 + 1.0)
      let slow: SignalTensor = exp(t * (iD2 * -1.0)) * L
      var env: SignalTensor = attack * (fast + slow) * A
      if o.includeCoupling {
        // Head-driven wire term: the head's own RMS amplitude envelope
        // E(t) = sqrt(sum_k env_k(t)^2) modulates every band with gain C_b.
        let e2 = (parts.envelope * parts.envelope).sum(axis: 1)  // [B]
        let E = sqrt(e2 + 1e-10).reshape([B, 1]).expand([B, NB])
        let C = logMap(p.rawC, Ranges.noiseCouple.0, Ranges.noiseCouple.1)
        env = env + E * C
      }
      let playhead = floor(time * sampleRate + 0.5)
      let bandsNow = noiseTable.sampleRow(playhead).reshape([1, NB]).expand([B, NB])
      noise = (env * bandsNow).sum(axis: 1)
    } else {
      noise = head * 0.0
    }

    // ---- output stage ----
    var x: SignalTensor
    if o.includeHead { x = head + noise } else { x = noise }
    x = x * Tensor(c.gin)
    let dv = logMap(p.rawDrive, Ranges.drive.0, Ranges.drive.1)
    let dAmt = p.rawDAmt.sigmoid()
    let driven: SignalTensor = x + (tanh(x * dv) * Tensor(c.dNorm) - x) * dAmt
    // Length fade (fixed from the table): fu = smooth01((age - (len - fade)) / fade).
    let lenStart = Tensor(zip(c.lenS, c.fadeS).map { $0 - $1 })
    let invFade = Tensor(c.fadeS.map { 1 / $0 })
    let f1: SignalTensor = smooth01((time - lenStart) * invFade) * -1.0 + 1.0
    let f2 = f1 * f1
    let curve = Tensor(c.fadeCurve)
    let fade: SignalTensor = f1 * (curve * -1.0 + 1.0) + f2 * f2 * curve
    let G = logMap(p.rawGain, Ranges.outGain.0, Ranges.outGain.1)
    let y = driven * fade * G
    return SnareRender(output: y, head: head, noise: noise)
  }
}

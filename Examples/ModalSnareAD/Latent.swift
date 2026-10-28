import DGenLazy
import Foundation

// E2: latent snare autodecoder.
//
// A per-snare latent code z (learned, dim D) plus velocity goes through a tanh MLP
// (D+1 -> H -> H -> P) into the segments of the realtime Latent Snare template
// (eseq research e3proto/gen_dsp.py, layout "e2"), and the decoded parameters drive
// the SAME closed-form voice the instrument runs per sample (e2/template.py is the
// numpy twin). Everything is static per render except the frame-parallel clock, so
// the [B,K] render is frame-parallel and batched exactly like E1's SnareRenderer.

// MARK: - Layout (decoder_layout.json written by gen_dsp.py)

struct LatentSegment {
  var name: String
  var offset: Int
  var width: Int
  var squash: String
  var lo: Float
  var hi: Float
}

struct LatentLayout {
  var P: Int
  var K: Int
  var NB: Int
  var hidden: [Int]
  var segments: [LatentSegment]
  var byName: [String: LatentSegment]

  static var defaultURL: URL { Paths.research.appendingPathComponent("e2/decoder_layout.json") }

  static func load(_ url: URL = defaultURL) throws -> LatentLayout {
    let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
    let segs = (j["segments"] as! [[String: Any]]).map { s in
      LatentSegment(
        name: s["name"] as! String, offset: (s["offset"] as! NSNumber).intValue,
        width: (s["width"] as! NSNumber).intValue, squash: s["squash"] as! String,
        lo: (s["lo"] as! NSNumber).floatValue, hi: (s["hi"] as! NSNumber).floatValue)
    }
    var by: [String: LatentSegment] = [:]
    for s in segs { by[s.name] = s }
    return LatentLayout(
      P: (j["P"] as! NSNumber).intValue, K: (j["K"] as! NSNumber).intValue,
      NB: (j["B"] as! NSNumber).intValue, hidden: (j["hidden"] as! [NSNumber]).map { $0.intValue },
      segments: segs, byName: by)
  }

  /// u [B, P] -> squashed natural values per segment, each [B, width].
  func squash(_ u: Tensor, batch B: Int) -> [String: Tensor] {
    var out: [String: Tensor] = [:]
    for s in segments {
      let x = u.shrink([(0, B), (s.offset, s.offset + s.width)])
      switch s.squash {
      case "log":
        let a = Foundation.log(s.lo), span = Foundation.log(s.hi) - a
        out[s.name] = (x.sigmoid() * span + a).exp()
      case "lin":
        out[s.name] = x.sigmoid() * (s.hi - s.lo) + s.lo
      default:  // "pi"
        out[s.name] = tanh(x) * Float.pi
      }
    }
    return out
  }
}

// MARK: - Decoder

final class LatentDecoder {
  let D: Int, H: Int, P: Int, codes: Int
  // z table [codes, D]; layer 1 split into z rows and the velocity row (export stacks them).
  let Z: Tensor
  let W1z: Tensor, w1v: Tensor, b1: Tensor
  let W2: Tensor, b2: Tensor
  let W3: Tensor, b3: Tensor

  var decoderParams: [Tensor] { [W1z, w1v, b1, W2, b2, W3, b3] }
  var named: [(String, Tensor)] {
    [("Z", Z), ("W1z", W1z), ("w1v", w1v), ("b1", b1), ("W2", W2), ("b2", b2), ("W3", W3), ("b3", b3)]
  }

  /// Must be created after LazyGraphContext.reset(). `initial` overrides any tensor by name.
  init(D: Int, H: Int, P: Int, codes: Int, zInit: [Float]?, b3Init: [Float]?, initial: [String: [Float]] = [:], seed: UInt64 = 7) {
    self.D = D; self.H = H; self.P = P; self.codes = codes
    var rng = SplitMix64(seed: seed)
    func gauss(_ n: Int, _ sd: Float) -> [Float] { (0..<n).map { _ in rng.gaussian() * sd } }
    func p(_ name: String, _ shape: [Int], _ fallback: @autoclosure () -> [Float]) -> Tensor {
      let n = shape.reduce(1, *)
      if let v = initial[name], v.count == n { return Tensor.param(shape, data: v) }
      return Tensor.param(shape, data: fallback())
    }
    Z = p("Z", [codes, D], zInit ?? gauss(codes * D, 0.5))
    W1z = p("W1z", [D, H], gauss(D * H, 1.0 / Float(D + 1).squareRoot()))
    w1v = p("w1v", [1, H], gauss(H, 1.0))
    b1 = p("b1", [1, H], gauss(H, 0.1))
    W2 = p("W2", [H, H], gauss(H * H, 1.0 / Float(H).squareRoot()))
    b2 = p("b2", [1, H], gauss(H, 0.1))
    W3 = p("W3", [H, P], gauss(H * P, 0.1 / Float(H).squareRoot()))
    b3 = p("b3", [1, P], b3Init ?? [Float](repeating: 0, count: P))
  }

  /// z rows [B, D], velocity [B, 1] -> u [B, P].
  func forward(z: Tensor, vel: Tensor) -> Tensor {
    let B = z.shape[0]
    let h1 = tanh(z.matmul(W1z) + vel.matmul(w1v) + b1.expand([B, H]))
    let h2 = tanh(h1.matmul(W2) + b2.expand([B, H]))
    return h2.matmul(W3) + b3.expand([B, P])
  }

  /// One-hot selector [B, codes] -> z rows [B, D] (differentiable gather).
  func codes(select: Tensor) -> Tensor { select.matmul(Z) }

  /// CPU forward of the current weights: z rows [B*D], vel [B] -> u [B*P] (row-major).
  /// Used to make u a leaf for the audio backward (see LatentCommands.twoPassStep).
  func cpuForward(z: [Float], vel: [Float]) -> [Float] {
    let s = snapshot()
    let B = vel.count
    func layer(_ x: [Float], _ n: Int, _ W: [Float], _ b: [Float], _ m: Int, act: Bool) -> [Float] {
      var y = [Float](repeating: 0, count: B * m)
      for r in 0..<B {
        for j in 0..<m {
          var acc = Double(b[j])
          for i in 0..<n { acc += Double(x[r * n + i]) * Double(W[i * m + j]) }
          y[r * m + j] = act ? Float(Foundation.tanh(acc)) : Float(acc)
        }
      }
      return y
    }
    var x = [Float](repeating: 0, count: B * (D + 1))
    for r in 0..<B {
      for d in 0..<D { x[r * (D + 1) + d] = z[r * D + d] }
      x[r * (D + 1) + D] = min(max(vel[r], 0), 1)
    }
    let W1 = s["W1z"]! + s["w1v"]!
    let h1 = layer(x, D + 1, W1, s["b1"]!, H, act: true)
    let h2 = layer(h1, H, s["W2"]!, s["b2"]!, H, act: true)
    return layer(h2, H, s["W3"]!, s["b3"]!, P, act: false)
  }

  func snapshot() -> [String: [Float]] {
    var out: [String: [Float]] = [:]
    for (n, t) in named { out[n] = t.getData() ?? [] }
    return out
  }

  func load(_ values: [String: [Float]]) {
    for (n, t) in named { if let v = values[n], v.count == t.shape.reduce(1, *) { t.updateDataLazily(v) } }
  }

  /// export_weights.py input: {l1,l2,l3:{W:{shape,data},b:{shape,data}}}, W [in,out] row-major;
  /// layer-1 rows are z1..zD then velocity.
  static func exportJSON(_ s: [String: [Float]], D: Int, H: Int, P: Int) -> [String: Any] {
    func t(_ shape: [Int], _ d: [Float]) -> [String: Any] { ["shape": shape, "data": d] }
    let W1 = (s["W1z"] ?? []) + (s["w1v"] ?? [])
    return [
      "l1": ["W": t([D + 1, H], W1), "b": t([1, H], s["b1"] ?? [])],
      "l2": ["W": t([H, H], s["W2"] ?? []), "b": t([1, H], s["b2"] ?? [])],
      "l3": ["W": t([H, P], s["W3"] ?? []), "b": t([1, P], s["b3"] ?? [])],
    ]
  }
}

struct SplitMix64 {
  var state: UInt64
  init(seed: UInt64) { state = seed }
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
  mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
  mutating func gaussian() -> Float {
    let u1 = max(uniform(), 1e-7), u2 = uniform()
    return (-2 * Foundation.log(u1)).squareRoot() * Foundation.cos(2 * Float.pi * u2)
  }
}

// MARK: - Template renderer (same math as the realtime instrument)

struct LatentRender {
  var output: SignalTensor  // [B]
  var head: SignalTensor  // [B] gated head (pre drive)
  var noise: SignalTensor  // [B]
}

enum LatentRenderer {
  /// Decoded natural params p (each [B, width]) -> one hit at frame 0 per lane.
  ///   modes: sh = g f; nyq = clip((0.48 sr - f - max(sh,0)) / (0.08 sr)); T = max(0, age - pre)
  ///          G = (1 - ghd s01((T-ght)/ghf)) (1 - gdd s01((T-gdt)/gdf)); t = max(0, T - t0)
  ///          env = amp nyq G e^{-t/dec} (1 - e^{-t/rise}); head = sum env sin 2pi(f t + sh tau (1-e^{-t/tau}) + ph/2pi)
  ///   noise: nt = max(0, age - onset); benv = (1-e^{-nt/rise})(A(1-L)e^{-nt/d1} + A L e^{-nt/d2}) + C E
  ///          noise = (1 - gate_noise (1 - GD)) sum benv N_b,  E = sqrt(sum env^2 + 1e-10)
  ///   out:   x = head + noise; d0 = x + amt (tanh(dv x) norm - x); driven = d0 + cmix (ceil u/(1+u^4)^.25 - d0), u = d0/ceil
  ///          y = driven fade gain
  static func render(
    _ p: [String: Tensor], batch B: Int, layout L: LatentLayout, noiseTable: Tensor?, frames: Int,
    sampleRate sr: Float, includeNoise: Bool = true, includeHead: Bool = true
  ) -> LatentRender {
    let K = L.K, NB = L.NB
    let time = Clock.seconds(frames: frames, sampleRate: sr)
    func g(_ n: String) -> Tensor { p[n]! }  // [B, 1]
    func gb(_ n: String) -> Tensor { p[n]!.reshape([B]) }

    // ---- head ----
    let f = g("mode_freq")
    let sh = f * g("mode_glide")
    let nyq = clampT((f + relu(sh)) * (-1 / (0.08 * sr)) + (0.48 / 0.08), 0, 1)
    let amp = g("mode_amp") * nyq
    // Per-render clock and gates are computed on [B] and broadcast to [B,K]: every
    // [B,K] intermediate is a per-frame tape in the backward pass.
    let Tb = relu(time - gb("pre"))  // [B]
    let gateHb = smooth01((Tb - gb("gh_time")) * (1.0 / gb("gh_fall"))) * gb("gh_depth") * -1.0 + 1.0
    let gdB = smooth01((Tb - gb("gd_time")) * (1.0 / gb("gd_fall"))) * gb("gd_depth") * -1.0 + 1.0
    let gate = (gateHb * gdB).reshape([B, 1]).expand([B, K])
    let T = Tb.reshape([B, 1]).expand([B, K])
    let t = relu(T - g("mode_t0"))
    let idec = 1.0 / g("mode_decay"), irise = 1.0 / g("mode_rise"), itau = 1.0 / g("mode_gtau")
    let env: SignalTensor = gate * exp(t * (idec * -1.0)) * (exp(t * (irise * -1.0)) * -1.0 + 1.0) * amp
    let cyc: SignalTensor = t * f + (exp(t * (itau * -1.0)) * -1.0 + 1.0) * (sh * g("mode_gtau"))
      + g("mode_phase") * (1 / (2 * Float.pi))
    let head = (env * sin(cyc * (2 * Float.pi))).sum(axis: 1)  // [B]

    // ---- noise ----
    var noise: SignalTensor
    if includeNoise, let noiseTable {
      func gn(_ n: String) -> Tensor { p[n]!.expand([B, NB]) }
      let nt = relu(time - gn("noise_onset"))
      let att: SignalTensor = exp(nt * ((1.0 / gn("noise_rise")) * -1.0)) * -1.0 + 1.0
      let A = g("band_amp"), Lt = g("band_tail")
      let aFast: Tensor = A * (Lt * -1.0 + 1.0)
      let aSlow: Tensor = A * Lt
      let nid1: Tensor = (1.0 / g("band_d1")) * -1.0
      let nid2: Tensor = (1.0 / g("band_d2")) * -1.0
      let fast: SignalTensor = exp(nt * nid1) * aFast
      let slow: SignalTensor = exp(nt * nid2) * aSlow
      let E = sqrt((env * env).sum(axis: 1) + 1e-10).reshape([B, 1]).expand([B, NB])
      let benv: SignalTensor = att * (fast + slow) + E * g("band_couple")
      let playhead = floor(time * sr + 0.5)
      let bandsNow = noiseTable.sampleRow(playhead).reshape([1, NB]).expand([B, NB])
      let ngate: SignalTensor = (gdB * -1.0 + 1.0) * gb("gate_noise") * -1.0 + 1.0
      noise = (benv * bandsNow).sum(axis: 1) * ngate
    } else {
      noise = head * 0.0
    }

    // ---- output stage ----
    let x: SignalTensor = includeHead ? head + noise : noise
    let driven0: SignalTensor = x + (tanh(x * gb("drive")) * gb("drive_norm") - x) * gb("drive_amt")
    // VILLAIN-style soft clipper, k=4: x += mix (ceil u / (1+u^4)^(1/4) - x), u = x / ceil
    let ceil = gb("clip_ceil")
    let cu: SignalTensor = driven0 * (1.0 / ceil)
    let cu2 = cu * cu
    let clipped: SignalTensor = cu / sqrt(sqrt(cu2 * cu2 + 1.0)) * ceil
    let driven: SignalTensor = driven0 + (clipped - driven0) * gb("clip_mix")
    let fadeS = gb("fade")
    let f1: SignalTensor = smooth01((time - (gb("len") - fadeS)) * (1.0 / fadeS)) * -1.0 + 1.0
    let f2 = f1 * f1
    let curve = gb("fade_curve")
    let fade: SignalTensor = f1 * (curve * -1.0 + 1.0) + f2 * f2 * curve
    let gain: Tensor = gb("gain")
    let shaped: SignalTensor = driven * fade
    let y: SignalTensor = shaped * gain
    return LatentRender(output: y, head: head, noise: noise)
  }

  static func clampT(_ x: Tensor, _ lo: Float, _ hi: Float) -> Tensor {
    let n = x.shape.reduce(1, *)
    return max(min(x, Tensor([Float](repeating: hi, count: n)).reshape(x.shape)),
               Tensor([Float](repeating: lo, count: n)).reshape(x.shape))
  }
}

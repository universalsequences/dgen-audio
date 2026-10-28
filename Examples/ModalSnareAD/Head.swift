import DGenLazy
import Foundation

/// Natural-domain head bank inputs, every tensor shaped [B, K]. Per-render
/// scalars (predelay, gates, head level) are replicated across K so every op
/// is a same-shape elementwise op (no implicit broadcasts on the AD path).
struct HeadTensors {
  var f, amp, idec, irise, t0, ph, sh, tau, mh, md: Tensor
  var pre, tuning: Tensor
  var gdDepth, gdMs, gdFall, gdSag, gdSagMs, gdSagSpan: Tensor
  var ghDepth, ghMs, ghFall: Tensor
  var headV0, ihr: Tensor
}

/// CPU-side [B, K] arrays (row-major b*K + k) for a batch of renders.
struct HeadArrays {
  var B: Int
  var K: Int
  var fields: [String: [Float]]

  static let modeKeys = ["f", "amp", "idec", "irise", "t0", "ph", "sh", "tau", "mh", "md"]
  static let renderKeys = [
    "pre", "tuning", "gdDepth", "gdMs", "gdFall", "gdSag", "gdSagMs", "gdSagSpan",
    "ghDepth", "ghMs", "ghFall", "headV0", "ihr",
  ]

  init(renders: [VillainRender], K: Int = 32, tuning: Double = VillainData.tuning) {
    B = renders.count
    self.K = K
    var f: [String: [Float]] = [:]
    for key in Self.modeKeys + Self.renderKeys { f[key] = [Float](repeating: 0, count: B * K) }
    for (b, r) in renders.enumerated() {
      let velE = max(r.vel, 1e-4)
      let vpow = r.vel > 0 ? Foundation.exp(r.c("c_hv") * Foundation.log(velE)) : 0
      let perRender: [String: Double] = [
        "pre": r.c("c_pre"), "tuning": tuning,
        "gdDepth": r.c("c_gd_depth"), "gdMs": r.c("c_gd_ms"), "gdFall": r.c("c_gd_fall"),
        "gdSag": r.c("c_gd_sag"), "gdSagMs": r.c("c_gd_sag_ms"),
        "gdSagSpan": max(0.001, r.c("c_gd_ms") - r.c("c_gd_sag_ms")),
        "ghDepth": r.c("c_gh_depth"), "ghMs": r.c("c_gh_ms"), "ghFall": r.c("c_gh_fall"),
        "headV0": r.c("c_head") * vpow, "ihr": 1 / max(r.c("c_hr"), 1e-9),
      ]
      for k in 0..<K {
        let i = b * K + k
        for (key, v) in perRender { f[key]![i] = Float(v) }
        guard k < r.slots.count else { continue }
        let s = r.slots[k]
        f["f"]![i] = Float(s.f)
        f["amp"]![i] = Float(s.amp)
        f["idec"]![i] = Float(1 / s.dec0)
        f["irise"]![i] = Float(1 / s.rise0)
        f["t0"]![i] = Float(s.t0)
        f["ph"]![i] = Float(s.ph)
        f["sh"]![i] = Float(max(-0.9 * s.f, s.g0 * s.f))
        f["tau"]![i] = Float(s.tau)
        f["mh"]![i] = Float(s.mh)
        f["md"]![i] = Float(s.md)
      }
    }
    fields = f
  }

  /// Build constant tensors. Must run after LazyGraphContext.reset().
  func tensors(overrides: [String: Tensor] = [:]) -> HeadTensors {
    func t(_ k: String) -> Tensor {
      if let o = overrides[k] { return o }
      return Tensor(fields[k]!).reshape([B, K])
    }
    return HeadTensors(
      f: t("f"), amp: t("amp"), idec: t("idec"), irise: t("irise"), t0: t("t0"), ph: t("ph"),
      sh: t("sh"), tau: t("tau"), mh: t("mh"), md: t("md"), pre: t("pre"), tuning: t("tuning"),
      gdDepth: t("gdDepth"), gdMs: t("gdMs"), gdFall: t("gdFall"), gdSag: t("gdSag"),
      gdSagMs: t("gdSagMs"), gdSagSpan: t("gdSagSpan"), ghDepth: t("ghDepth"), ghMs: t("ghMs"),
      ghFall: t("ghFall"), headV0: t("headV0"), ihr: t("ihr"))
  }
}

enum Clock {
  /// Seconds since the trigger as a frame-parallel Signal. The phasor runs at
  /// sr/(2N) so its ramp spans [0, 0.5) and never wraps inside the render.
  static func seconds(frames: Int, sampleRate: Float) -> Signal {
    let rampHz = sampleRate / (2 * Float(frames))
    return Signal.phasor(Tensor([rampHz])).sum() * (2 * Float(frames) / sampleRate)
  }
}

func clip01(_ x: SignalTensor) -> SignalTensor { max(min(x, 1.0), 0.0) }
func smooth01(_ x: SignalTensor) -> SignalTensor {
  let c = clip01(x)
  return c * c * (c * -2.0 + 3.0)
}

enum HeadRenderer {
  /// VILLAIN head_g, closed form and frame-parallel:
  ///   m_k = amp e^{-t idec} (1-e^{-t irise}) nyq(hz) sin(2π tuning (f t + sh tau (1-e^{-t/tau})) + ph)
  ///   t = max(0, T - t0), T = max(0, age - pre), hz = tuning (f + sh e^{-t/tau})
  /// gated per mode by (1-mh(1-GH))(1-md(1-GD)), summed over K and scaled by
  /// head_v = headV0 (1 - e^{-T ihr}). Returns the per-mode [B,K] contribution
  /// (already gated and head_v-scaled) so callers can reduce or reuse it.
  struct Parts {
    var modes: SignalTensor  // [B,K] signed contributions
    var envelope: SignalTensor  // [B,K] amplitude envelopes (no carrier)
  }

  static func modes(_ h: HeadTensors, time: Signal, sampleRate: Float) -> SignalTensor {
    parts(h, time: time, sampleRate: sampleRate).modes
  }

  static func parts(_ h: HeadTensors, time: Signal, sampleRate: Float) -> Parts {
    let T = relu(time - h.pre)  // [B,K]
    let t = relu(T - h.t0)
    let se = exp(t * (-1 / h.tau))
    let glide: SignalTensor = (se * -1.0 + 1.0) * (h.sh * h.tau)
    let cycles: SignalTensor = (t * h.f + glide) * h.tuning
    let carrier = sin(cycles * (2 * Float.pi) + h.ph)
    let hz = (se * h.sh + h.f) * h.tuning
    let nyqTop: Float = 0.48 * sampleRate
    let nyqScale: Float = 1 / (0.08 * sampleRate)
    let nyq = clip01((hz * -1.0 + nyqTop) * nyqScale)
    let decay = exp(t * (h.idec * -1.0))
    let attack: SignalTensor = exp(t * (h.irise * -1.0)) * -1.0 + 1.0
    let env = decay * attack * h.amp
    // Gates.
    let sagRamp = clip01((T - h.gdSagMs) * (1 / h.gdSagSpan))
    let gdSagg = exp(sagRamp * h.gdSag * 0.115129255)
    let gdFall = smooth01((T - h.gdMs) * (1 / h.gdFall)) * -1.0 + 1.0
    let GD: SignalTensor = gdSagg * gdFall * h.gdDepth + (h.gdDepth * -1.0 + 1.0)
    let GH = smooth01((T - h.ghMs) * (1 / h.ghFall)) * h.ghDepth * -1.0 + 1.0
    let gateH: SignalTensor = (GH * -1.0 + 1.0) * h.mh * -1.0 + 1.0
    let gateD: SignalTensor = (GD * -1.0 + 1.0) * h.md * -1.0 + 1.0
    let gate = gateH * gateD
    let headV: SignalTensor = (exp(T * (h.ihr * -1.0)) * -1.0 + 1.0) * h.headV0
    let amplitude = env * nyq * gate * headV
    return Parts(modes: amplitude * carrier, envelope: amplitude)
  }

  /// [B] head signal.
  static func render(_ h: HeadTensors, time: Signal, sampleRate: Float) -> SignalTensor {
    modes(h, time: time, sampleRate: sampleRate).sum(axis: 1)
  }
}

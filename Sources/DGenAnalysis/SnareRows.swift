// SnareRows.swift — rows for the snare voice (latent-snare-from-samples voice/snare1.lisp,
// `row-layout snare`) from an `analyze-drums --kind snare` result.
//
//   modes      slot 0 (fundamental) + K-1 roles from KickSlotTemplate / KickSlots.assignV3 (the same
//              correspondence as the kicks); every mode shares slot 0's glide law (one head
//              tension glide), so the row carries one (g, td); slot 0's `fundamental-tail` mode
//              becomes its slow tail (tamp, tdec);
//   strikes    as KickRowsV3 (2 roles per StrikeFit region);
//   wires      the decaying part of the noise bands (L, m, d1, d2), shared rise; onset delay 0
//              (the refine moves it);
//   hiss       the stationary floor of the noise bands (H), one slow decay;
//   record     crackle + 2 events as KickRowsV3;
//   chain      shaper / clipper / record low-pass at neutral values (the refine learns them).

import Foundation

public enum SnareRows {
  public static let silent = 1e-5
  public static let boxMargin = 0.01

  /// (name, width, lo, hi) of snare1.lisp's `row-layout snare`, in order (raw segments: nil range).
  public static func layout(slots k: Int) -> [(String, Int, Double?, Double?)] {
    [
      ("f", k, 30, 16000), ("amp", k, 0.000001, 20), ("dec", k, 0.001, 4), ("rise", k, 0.00001, 0.05),
      ("ph", k, nil, nil),
      ("g", 1, 0, 1), ("td", 1, 0.0005, 0.2), ("tamp", 1, 0.000001, 4), ("tdec", 1, 0.005, 4),
      ("xf", 4, 100, 16000), ("xg", 4, 0, 4), ("xtd", 4, 0.0003, 0.05), ("xamp", 4, 0.000001, 4),
      ("xdec", 4, 0.0002, 0.05), ("xph", 4, nil, nil),
      ("sh_in", 1, 0.05, 20), ("sh_odd", 1, 0, 1), ("sh_even", 1, -1, 1),
      ("c_pos", 1, 0.05, 4), ("c_neg", 1, 0.3, 1.5), ("c_off", 1, -0.3, 0.3), ("c_mix", 1, 0, 1),
      ("rl_hz", 1, 300, 20000), ("rl_mix", 1, 0, 1),
      ("t0", 1, -0.002, 0.008), ("end", 1, 0.02, 1.6), ("fade", 1, 0.0002, 0.2), ("gain", 1, 0.05, 20),
      ("wl", 16, 0.0000001, 4), ("wm", 16, 0, 1), ("wd1", 16, 0.0002, 0.1), ("wd2", 16, 0.002, 2),
      ("w_t0", 1, 0, 0.02), ("wrise", 1, 0.00001, 0.05),
      ("hl", 16, 0.0000001, 1), ("hdec", 1, 0.1, 30),
      ("cr_lvl", 1, 0.000001, 4), ("cr_rate", 1, 1, 4000), ("cr_col", 4, 0.001, 10), ("cr_dec", 1, 0.05, 20),
      ("ev_t", 2, 0, 1.2), ("ev_lvl", 2, 0.000001, 4), ("ev_dec", 2, 0.002, 1), ("ev_rise", 2, 0.0001, 0.02),
      ("ev_col", 32, 0.0001, 1),
    ]
  }

  static let logSegments: Set<String> = [
    "f", "amp", "dec", "rise", "td", "tamp", "tdec", "xf", "xtd", "xamp", "xdec", "sh_in", "c_pos", "rl_hz",
    "fade", "gain", "wl", "wd1", "wd2", "wrise", "hl", "hdec", "cr_lvl", "cr_rate", "cr_col", "cr_dec", "ev_lvl",
    "ev_dec", "ev_rise", "ev_col",
  ]

  /// The effective [lo, hi] the trainer renders for a segment (the sigmoid box less its margin).
  public static func effectiveBox(_ name: String, slots: Int) -> (Double, Double)? {
    guard let seg = layout(slots: slots).first(where: { $0.0 == name }), let lo = seg.2, let hi = seg.3 else {
      return nil
    }
    if logSegments.contains(name) {
      let r = log(hi / lo)
      return (lo * exp(boxMargin * r), lo * exp((1 - boxMargin) * r))
    }
    return (lo + boxMargin * (hi - lo), hi - boxMargin * (hi - lo))
  }

  /// Character knobs the analysis cannot measure: neutral starting values (the refine learns them).
  public static let neutral: [String: [Double]] = [
    "sh_in": [1], "sh_odd": [0.02], "sh_even": [0],
    "c_pos": [4], "c_neg": [1], "c_off": [0], "c_mix": [0.5], "rl_hz": [18000], "rl_mix": [0.05],
    "gain": [1], "t0": [0], "w_t0": [0.0003], "hdec": [20],
  ]

  public static func row(_ s: KickAnalysis, slots k: Int, template: KickSlotTemplate, assignment a: KickSlotAssignment,
                         strikeTemplate: KickSlotTemplate, strikeSlots: [Int?]) -> [String: [Double]]
  {
    precondition(template.roles.count == k - 1, "template has \(template.roles.count) roles, need \(k - 1)")
    var r = neutral
    var f = [Double](), amp = [Double](), dec = [Double](), rise = [Double](), ph = [Double]()
    func push(_ m: ModeEstimate) { f.append(m.F); amp.append(m.A); dec.append(m.D); rise.append(m.R); ph.append(m.phi) }
    let m0 = s.modes[0]
    push(m0)
    for (i, role) in template.roles.enumerated() {
      if let j = a.roleModes[i] { push(s.modes[j]) } else {
        f.append(role.frequency); amp.append(silent); dec.append(role.decay); rise.append(0.0007); ph.append(0)
      }
    }
    r["f"] = f; r["amp"] = amp; r["dec"] = dec; r["rise"] = rise; r["ph"] = ph
    let tail = s.modes.first { $0.region == "fundamental-tail" }
    r["tamp"] = [tail?.A ?? silent]; r["tdec"] = [tail?.D ?? 0.3]
    // shared glide; below the effective minimum depth every mode's settled phase is corrected
    let gBox = effectiveBox("g", slots: k)!, tdBox = effectiveBox("td", slots: k)!
    let T = min(max(m0.T, tdBox.0), tdBox.1)
    if m0.G < gBox.0 {
      r["g"] = [gBox.0]; r["td"] = [tdBox.0]
      r["ph"] = zip(f, ph).map { F, p in p + 2 * Double.pi * F * (m0.G * T - gBox.0 * tdBox.0) }
    } else {
      r["g"] = [m0.G]; r["td"] = [T]
    }
    // strikes
    let st = s.strikes ?? []
    var xf = [Double](), xg = [Double](), xtd = [Double](), xamp = [Double](), xdec = [Double](), xph = [Double]()
    let xgBox = effectiveBox("xg", slots: k)!, xtdBox = effectiveBox("xtd", slots: k)!
    for (i, role) in strikeTemplate.roles.enumerated() {
      if i < strikeSlots.count, let j = strikeSlots[i] {
        let m = st[j]
        let (g, t, dph) = KickRowsV3.glideFloor(F: m.F, g: m.G, T: min(max(m.T, xtdBox.0), xtdBox.1),
                                                gMin: xgBox.0, tMin: xtdBox.0)
        xf.append(m.F); xg.append(g); xtd.append(t); xamp.append(m.A); xdec.append(m.D); xph.append(m.phi + dph)
      } else {
        xf.append(role.frequency); xg.append(0.1); xtd.append(0.003); xamp.append(silent); xdec.append(0.002)
        xph.append(0)
      }
    }
    r["xf"] = xf; r["xg"] = xg; r["xtd"] = xtd; r["xamp"] = xamp; r["xdec"] = xdec; r["xph"] = xph
    r["end"] = [s.gate.E]; r["fade"] = [s.gate.Fd]
    // wires (decaying) + hiss (stationary floor)
    let nb = 16
    precondition(s.noise.bands.count == nb, "snare rows need 16 noise bands (got \(s.noise.bands.count))")
    let wmBox = effectiveBox("wm", slots: k)!, wd2Box = effectiveBox("wd2", slots: k)!
    var wm = s.noise.bands.map(\.m), wd1 = s.noise.bands.map(\.d1), wd2 = s.noise.bands.map(\.d2)
    for b in 0..<nb {
      // m beyond the box would leave 1% of L on the slow decay: give that 1% the fast decay
      if wm[b] > wmBox.1 { wm[b] = wmBox.1; wd2[b] = max(wd1[b], wd2Box.0) }
      if wm[b] < wmBox.0 { wm[b] = wmBox.0 }
    }
    r["wl"] = s.noise.bands.map(\.L); r["wm"] = wm; r["wd1"] = wd1; r["wd2"] = wd2; r["wrise"] = [s.noise.rn]
    r["hl"] = s.noise.bands.map { max($0.H ?? 0, 1e-6) }
    // crackle
    let cr = s.crackle ?? Crackle.silent()
    r["cr_lvl"] = [cr.present ? cr.level : silent]; r["cr_rate"] = [cr.rate]
    r["cr_col"] = cr.colour; r["cr_dec"] = [cr.decay]
    // events (by time; unused: silent, colour 0.01); colour peaks at the box maximum
    let evs = (s.noise.events ?? []).sorted { $0.t < $1.t }
    let ecBox = effectiveBox("ev_col", slots: k)!
    var evt = [Double](), evl = [Double](), evd = [Double](), evr = [Double](), evc = [Double]()
    for e in 0..<2 {
      if e < evs.count {
        let ev = evs[e]
        var col = (0..<nb).map { $0 < ev.colour.count ? ev.colour[$0] : 0.01 }
        var lvl = ev.level
        let mx = col.max() ?? 1
        if mx > ecBox.1 { col = col.map { $0 * ecBox.1 / mx }; lvl *= mx / ecBox.1 }
        evt.append(ev.t); evl.append(lvl); evd.append(ev.decay); evr.append(ev.rise); evc += col
      } else {
        evt.append(e == 0 ? 0.1 : 0.3); evl.append(silent); evd.append(0.05); evr.append(0.001)
        evc += [Double](repeating: 0.01, count: nb)
      }
    }
    r["ev_t"] = evt; r["ev_lvl"] = evl; r["ev_dec"] = evd; r["ev_rise"] = evr; r["ev_col"] = evc
    // clamp to the effective boxes; wrap phases
    for (name, width, _, _) in layout(slots: k) {
      guard var v = r[name] else { preconditionFailure("snare row misses \(name)") }
      precondition(v.count == width, "snare row \(name): \(v.count) values, layout says \(width)")
      if let (lo, hi) = effectiveBox(name, slots: k) { v = v.map { min(max($0, lo), hi) } } else {
        v = v.map { remainder($0, 2 * Double.pi) }
      }
      r[name] = v
    }
    return r
  }
}

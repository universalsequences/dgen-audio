// KickSlots.swift — slot correspondence (SPEC §5): every kick's modes are assigned to the same
// K modal role slots, so slot k plays the same role across the set and a latent decoder sees
// smooth targets.
//
//   slot 0         the fundamental (its pitch-drop law; the analysis' `fundamental-tail` mode
//                  becomes slot 0's slow-tail share/decay);
//   slots 1..K-1   roles estimated from the set: energy-weighted k-means in log-frequency over the
//                  ringing (D >= 10 ms) non-fundamental modes, sorted by frequency;
//   per kick       Hungarian assignment of its non-fundamental modes to roles, cost = octave
//                  distance + an energy penalty (quiet modes are cheap to drop); a role may stay
//                  empty (cost `emptyCost`), and then plays silently (A = 1e-5) at the role's
//                  median frequency and decay.
//
// `KickRows.rows` writes the result in the voice's row layout (`train-dataset --rows`).

import Foundation

public struct KickRole: Codable {
  public var frequency: Double
  public var decay: Double
}

public struct KickSlotTemplate: Codable {
  public var roles: [KickRole]  // slots 1..K-1

  /// `energyWeighted`: k-means weights √energy (crowds roles where the loud low modes are);
  /// otherwise every ringing mode counts once (roles follow where modes occur).
  public static func estimate(_ samples: [KickAnalysis], slots: Int, ringing: Double = 0.01,
                              energyWeighted: Bool = false, iterations: Int = 50) -> KickSlotTemplate {
    var pts: [(lf: Double, w: Double, d: Double)] = []
    for s in samples {
      for m in s.modes where (m.region == "low" || m.region == "mid") && m.D >= ringing && m.A > 0 {
        pts.append((log2(m.F), energyWeighted ? max(m.energy, 1e-12).squareRoot() : 1, m.D))
      }
    }
    return cluster(pts, roles: slots - 1, iterations: iterations)
  }

  /// Strike roles (v3): per strike region (StrikeFit.regions), count-weighted k-means in
  /// log-frequency over every kick's strikes in it; roles = `perRegion` per region, low first.
  public static func estimateStrikes(_ samples: [KickAnalysis], perRegion: Int = 2, iterations: Int = 50)
    -> KickSlotTemplate
  {
    var roles: [KickRole] = []
    for band in StrikeFit.regions {
      var pts: [(lf: Double, w: Double, d: Double)] = []
      for s in samples {
        for m in s.strikes ?? [] where m.A > 0 && m.F >= band[0] && m.F < band[1] { pts.append((log2(m.F), 1, m.D)) }
      }
      var t = cluster(pts, roles: perRegion, iterations: iterations)
      if t.roles.count < perRegion {  // too few strikes in the region: log-spaced nominal roles
        t.roles = (0..<perRegion).map {
          KickRole(frequency: band[0] * pow(band[1] / band[0], (Double($0) + 0.5) / Double(perRegion)), decay: 0.002)
        }
      }
      roles += t.roles
    }
    return KickSlotTemplate(roles: roles)
  }

  static func cluster(_ pts0: [(lf: Double, w: Double, d: Double)], roles n: Int, iterations: Int) -> KickSlotTemplate {
    var pts = pts0
    guard n > 0, !pts.isEmpty else { return KickSlotTemplate(roles: []) }
    pts.sort { $0.lf < $1.lf }
    // init: weighted quantiles
    let total = pts.reduce(0) { $0 + $1.w }
    var centers: [Double] = (0..<n).map { k in
      let target = total * (Double(k) + 0.5) / Double(n)
      var acc = 0.0
      for p in pts { acc += p.w; if acc >= target { return p.lf } }
      return pts.last!.lf
    }
    for _ in 0..<iterations {
      var sum = [Double](repeating: 0, count: n), wsum = [Double](repeating: 0, count: n)
      for p in pts {
        let k = centers.indices.min { abs(centers[$0] - p.lf) < abs(centers[$1] - p.lf) }!
        sum[k] += p.w * p.lf; wsum[k] += p.w
      }
      for k in 0..<n where wsum[k] > 0 { centers[k] = sum[k] / wsum[k] }
      centers.sort()
    }
    var decays = [[Double]](repeating: [], count: n)
    for p in pts {
      let k = centers.indices.min { abs(centers[$0] - p.lf) < abs(centers[$1] - p.lf) }!
      decays[k].append(p.d)
    }
    let roles = zip(centers, decays).map { c, ds -> KickRole in
      let sorted = ds.sorted()
      return KickRole(frequency: pow(2, c), decay: sorted.isEmpty ? 0.03 : sorted[sorted.count / 2])
    }
    return KickSlotTemplate(roles: roles)
  }
}

public struct KickSlotAssignment: Codable {
  /// slot k (1-based role index k-1) → index into the sample's `modes`, nil = empty role.
  public var roleModes: [Int?]
  /// Modes not placed in any slot (their energy is left to the noise bands).
  public var dropped: [Int]
  public var droppedEnergy: Double
}

public enum KickSlots {
  public static var emptyCost = 1.5
  public static var energyWeight = 0.5

  /// v3 assignment (12 slots): every mode costs `emptyCost` to drop whatever its energy, and its
  /// assignment costs octave distance + 0.25 per decade of energy below the kick's loudest
  /// candidate (capped at 6 decades), so a free role within ~1.5 octaves always takes a mode, a
  /// contested role goes to the louder mode, and no mode moves more than `maxOctaves`.
  /// (The v2 rule made dropping free for modes 30 dB below the loudest: short or quiet real
  /// modes lost their slot even when one was free.)
  public static func assignV3(_ s: KickAnalysis, template: KickSlotTemplate, maxOctaves: Double = 1.5)
    -> KickSlotAssignment
  {
    let roles = template.roles
    let cand = s.modes.indices.filter { (s.modes[$0].region == "low" || s.modes[$0].region == "mid") && s.modes[$0].A > 0 }
    let eMax = cand.map { s.modes[$0].energy }.max() ?? 1
    let nR = roles.count, nM = cand.count
    let size = nR + nM
    var cost = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
    for r in 0..<size {
      for c in 0..<size {
        if r < nR && c < nM {
          let m = s.modes[cand[c]]
          let dist = abs(log2(m.F / roles[r].frequency))
          let quiet = min(6, max(0, -log10(max(m.energy, 1e-30) / max(eMax, 1e-30))))
          cost[r][c] = dist > maxOctaves ? 100 : dist + 0.25 * quiet
        } else if r < nR || c < nM {
          cost[r][c] = emptyCost  // role left empty / mode dropped
        }
      }
    }
    let match = Hungarian.solve(cost)
    var roleModes = [Int?](repeating: nil, count: nR)
    var used = Set<Int>()
    for r in 0..<nR where match[r] < nM && cost[r][match[r]] < 100 {
      roleModes[r] = cand[match[r]]
      used.insert(cand[match[r]])
    }
    let dropped = cand.filter { !used.contains($0) }
    return KickSlotAssignment(roleModes: roleModes, dropped: dropped,
                              droppedEnergy: dropped.reduce(0) { $0 + s.modes[$1].energy })
  }

  /// Strike sines to strike roles: within each strike region (roles split evenly, low first) by
  /// octave distance; a region with more strikes than roles keeps the strongest.
  public static func assignStrikes(_ s: KickAnalysis, template: KickSlotTemplate) -> [Int?] {
    let st = s.strikes ?? []
    let nR = template.roles.count
    var out = [Int?](repeating: nil, count: nR)
    let per = nR / StrikeFit.regions.count
    guard per > 0 else { return out }
    for (g, band) in StrikeFit.regions.enumerated() {
      var idx = st.indices.filter { st[$0].F >= band[0] && st[$0].F < band[1] }
      idx.sort { st[$0].energy > st[$1].energy }
      idx = Array(idx.prefix(per))
      let roles = Array(template.roles[(g * per)..<((g + 1) * per)])
      let size = max(per, idx.count)
      var cost = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
      for r in 0..<size {
        for c in 0..<size where r < per && c < idx.count {
          cost[r][c] = abs(log2(st[idx[c]].F / roles[r].frequency))
        }
      }
      let match = Hungarian.solve(cost)
      for r in 0..<per where match[r] < idx.count { out[g * per + r] = idx[match[r]] }
    }
    return out
  }

  public static func assign(_ s: KickAnalysis, template: KickSlotTemplate) -> KickSlotAssignment {
    let roles = template.roles
    let cand = s.modes.indices.filter { (s.modes[$0].region == "low" || s.modes[$0].region == "mid") && s.modes[$0].A > 0 }
    let eMax = cand.map { s.modes[$0].energy }.max() ?? 1
    let nR = roles.count, nM = cand.count
    let size = nR + nM
    // rows: roles then dummy rows (a mode left unassigned); cols: modes then dummy cols (empty role)
    var cost = [[Double]](repeating: [Double](repeating: 0, count: size), count: size)
    for r in 0..<size {
      for c in 0..<size {
        if r < nR && c < nM {
          let m = s.modes[cand[c]]
          let dist = abs(log2(m.F / roles[r].frequency))
          let quiet = max(0, -log10(max(m.energy, 1e-30) / max(eMax, 1e-30)))
          cost[r][c] = dist + energyWeight * quiet
        } else if r < nR {
          cost[r][c] = emptyCost  // role left empty
        } else if c < nM {
          let m = s.modes[cand[c]]
          // dropping a mode costs what an empty role would, scaled down for quiet modes
          let quiet = max(0, -log10(max(m.energy, 1e-30) / max(eMax, 1e-30)))
          cost[r][c] = max(0, emptyCost - energyWeight * quiet)
        } else {
          cost[r][c] = 0
        }
      }
    }
    let match = Hungarian.solve(cost)
    var roleModes = [Int?](repeating: nil, count: nR)
    var used = Set<Int>()
    for r in 0..<nR where match[r] < nM {
      roleModes[r] = cand[match[r]]
      used.insert(cand[match[r]])
    }
    let dropped = cand.filter { !used.contains($0) }
    return KickSlotAssignment(roleModes: roleModes, dropped: dropped,
                              droppedEnergy: dropped.reduce(0) { $0 + s.modes[$1].energy })
  }
}

/// Minimum-cost perfect matching on a square cost matrix (Kuhn–Munkres, O(n³)).
/// Returns, for each row, the assigned column.
public enum Hungarian {
  public static func solve(_ a: [[Double]]) -> [Int] {
    let n = a.count
    guard n > 0 else { return [] }
    let inf = Double.infinity
    var u = [Double](repeating: 0, count: n + 1), v = [Double](repeating: 0, count: n + 1)
    var p = [Int](repeating: 0, count: n + 1), way = [Int](repeating: 0, count: n + 1)
    for i in 1...n {
      p[0] = i
      var j0 = 0
      var minv = [Double](repeating: inf, count: n + 1)
      var used = [Bool](repeating: false, count: n + 1)
      repeat {
        used[j0] = true
        let i0 = p[j0]
        var delta = inf, j1 = 0
        for j in 1...n where !used[j] {
          let cur = a[i0 - 1][j - 1] - u[i0] - v[j]
          if cur < minv[j] { minv[j] = cur; way[j] = j0 }
          if minv[j] < delta { delta = minv[j]; j1 = j }
        }
        for j in 0...n {
          if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
        }
        j0 = j1
      } while p[j0] != 0
      repeat {
        let j1 = way[j0]
        p[j0] = p[j1]
        j0 = j1
      } while j0 != 0
    }
    var rowToCol = [Int](repeating: 0, count: n)
    for j in 1...n where p[j] > 0 { rowToCol[p[j] - 1] = j - 1 }
    return rowToCol
  }
}

/// Rows in the Latent Kick voice's `kick` row layout (natural units, per segment).
public enum KickRows {
  public static let silentAmplitude = 1e-5

  public static func row(_ s: KickAnalysis, template: KickSlotTemplate, assignment a: KickSlotAssignment)
    -> [String: [Double]]
  {
    let K = template.roles.count + 1
    var f = [Double](), g = [Double](), td = [Double](), amp = [Double](), dec = [Double]()
    var rise = [Double](), ph = [Double]()
    func push(_ m: ModeEstimate) {
      f.append(m.F); g.append(m.G); td.append(m.T); amp.append(m.A); dec.append(m.D)
      rise.append(m.R); ph.append(m.phi)
    }
    let m0 = s.modes[0]
    push(m0)
    for (k, role) in template.roles.enumerated() {
      if let i = a.roleModes[k] {
        push(s.modes[i])
      } else {
        f.append(role.frequency); g.append(0); td.append(0.03); amp.append(silentAmplitude)
        dec.append(role.decay); rise.append(0.0007); ph.append(0)
      }
    }
    precondition(f.count == K)
    let tail = s.modes.first { $0.region == "fundamental-tail" }
    var r: [String: [Double]] = [
      "f": f, "g": g, "td": td, "amp": amp, "dec": dec, "rise": rise, "ph": ph,
      "g2": [m0.G2], "t2": [m0.T2 > 0 ? m0.T2 : 0.003],
      "tamp": [tail?.A ?? silentAmplitude], "tdec": [tail?.D ?? 0.3],
      "nl": s.noise.bands.map(\.L), "nm": s.noise.bands.map(\.m),
      "nd1": s.noise.bands.map(\.d1), "nd2": s.noise.bands.map(\.d2), "nrise": [s.noise.rn],
      // a near-linear clipper start (knee well above the 0.9 peak): the trainer learns it
      "knee": [4], "cmix": [0.5],
      "end": [s.gate.E], "fade": [s.gate.Fd], "level": [1],
    ]
    if s.noise.bands.isEmpty { r["nl"] = nil }
    return r
  }
}

/// Rows in voice v3's `kick` row layout (voice/voice3.lisp): every segment, natural units,
/// clamped to what the trainer renders: `train-dataset --rows` maps natural values into the squash
/// boxes 1% inside the edges (in σ space: lin boxes lose 1% of the range, log boxes 1% of the
/// log range). A drop depth g = 0 would therefore play as g = 0.08 (an 8% glide over td): such
/// modes get td at its effective minimum (~1 ms) and their phase compensated by the glide's phase
/// integral 2π F g td (same for g2/t2 on slot 0 and xg/xtd on strikes). ev_t cannot go below 8 ms.
public enum KickRowsV3 {
  public static let silent = 1e-5
  public static let boxMargin = 0.01

  /// The effective [lo, hi] the trainer renders for a segment.
  public static func effectiveBox(_ name: String) -> (Double, Double)? {
    guard let seg = layout.first(where: { $0.0 == name }), let lo = seg.2, let hi = seg.3 else { return nil }
    if logSegments.contains(name) {
      let r = log(hi / lo)
      return (lo * exp(boxMargin * r), lo * exp((1 - boxMargin) * r))
    }
    return (lo + boxMargin * (hi - lo), hi - boxMargin * (hi - lo))
  }

  static let logSegments: Set<String> = [
    "f", "td", "amp", "dec", "rise", "t2", "tamp", "tdec", "xf", "xtd", "xamp", "xdec", "gr_amp", "gr_f", "gr_td",
    "gr_d1", "gr_d2", "bu_amp", "bu_f", "bu_w", "sh_in", "bk_amt", "c_pos", "rl_hz", "fade", "gain", "nl", "nd1",
    "nd2", "nrise", "cr_lvl", "cr_rate", "cr_col", "cr_dec", "ev_lvl", "ev_dec", "ev_rise", "ev_col",
  ]

  /// A glide (g, T) below the effective minimum depth: play it as (gMin, tMin) and return the
  /// phase correction that keeps the settled phase (2π F (g T - gMin tMin)).
  static func glideFloor(F: Double, g: Double, T: Double, gMin: Double, tMin: Double) -> (Double, Double, Double) {
    guard g < gMin else { return (g, T, 0) }
    return (gMin, tMin, 2 * Double.pi * F * (g * T - gMin * tMin))
  }

  /// (name, width, lo, hi) of voice3.lisp's row-layout kick, in order (raw segments: nil range).
  public static let layout: [(String, Int, Double?, Double?)] = [
    ("f", 12, 20, 12000), ("g", 12, 0, 8), ("td", 12, 0.001, 0.5), ("amp", 12, 0.000001, 20),
    ("dec", 12, 0.001, 6), ("rise", 12, 0.00001, 0.05), ("ph", 12, nil, nil),
    ("g2", 1, 0, 8), ("t2", 1, 0.0003, 0.05), ("tamp", 1, 0.000001, 4), ("tdec", 1, 0.01, 6),
    ("xf", 4, 100, 16000), ("xg", 4, 0, 4), ("xtd", 4, 0.0003, 0.05), ("xamp", 4, 0.000001, 4),
    ("xdec", 4, 0.0002, 0.05), ("xph", 4, nil, nil),
    ("gr_amp", 1, 0.000001, 2), ("gr_f", 1, 20, 90), ("gr_g", 1, 0, 6), ("gr_td", 1, 0.002, 0.3),
    ("gr_clip", 1, 0.05, 1), ("gr_d1", 1, 0.005, 1), ("gr_d2", 1, 0.01, 2), ("gr_m", 1, 0, 1),
    ("bu_amp", 1, 0.000001, 2), ("bu_f", 1, 50, 8000), ("bu_tc", 1, 0, 0.03), ("bu_w", 1, 0.0003, 0.02),
    ("bu_ph", 1, nil, nil),
    ("sh_in", 1, 0.05, 20), ("sh_odd", 1, 0, 1), ("sh_even", 1, -1, 1), ("bk_amt", 1, 0.000001, 2),
    ("c_pos", 1, 0.05, 4), ("c_neg", 1, 0.3, 1.5), ("c_off", 1, -0.3, 0.3), ("c_mix", 1, 0, 1),
    ("rl_hz", 1, 300, 20000), ("rl_mix", 1, 0, 1),
    ("t0", 1, -0.002, 0.008), ("end", 1, 0.02, 1.2), ("fade", 1, 0.0002, 0.2), ("gain", 1, 0.05, 20),
    ("nl", 16, 0.0000001, 4), ("nm", 16, 0, 1), ("nd1", 16, 0.0002, 0.5), ("nd2", 16, 0.002, 6),
    ("nrise", 1, 0.00001, 0.05),
    ("cr_lvl", 1, 0.000001, 4), ("cr_rate", 1, 1, 4000), ("cr_col", 4, 0.001, 10), ("cr_dec", 1, 0.05, 20),
    ("ev_t", 2, 0, 0.8), ("ev_lvl", 2, 0.000001, 4), ("ev_dec", 2, 0.002, 1), ("ev_rise", 2, 0.0001, 0.02),
    ("ev_col", 32, 0.0001, 1),
  ]

  /// Character knobs the analysis cannot measure: neutral starting values (the refine learns them).
  public static let neutral: [String: [Double]] = [
    "gr_amp": [silent], "gr_f": [45], "gr_g": [1], "gr_td": [0.02], "gr_clip": [0.5], "gr_d1": [0.05],
    "gr_d2": [0.2], "gr_m": [0.5],
    "bu_amp": [silent], "bu_f": [500], "bu_tc": [0.005], "bu_w": [0.002], "bu_ph": [0],
    "sh_in": [1], "sh_odd": [0.02], "sh_even": [0], "bk_amt": [silent],
    "c_pos": [4], "c_neg": [1], "c_off": [0], "c_mix": [0.5], "rl_hz": [18000], "rl_mix": [0.05],
    "gain": [1],
    // onset delay: analysis parameters are referenced to t = 0 (prepared sample 0); the refine moves it
    "t0": [0],
  ]

  public static func row(_ s: KickAnalysis, template: KickSlotTemplate, assignment a: KickSlotAssignment,
                         strikeTemplate: KickSlotTemplate, strikeSlots: [Int?]) -> [String: [Double]]
  {
    var r = neutral
    var f = [Double](), g = [Double](), td = [Double](), amp = [Double](), dec = [Double]()
    var rise = [Double](), ph = [Double]()
    func push(_ m: ModeEstimate) {
      f.append(m.F); g.append(m.G); td.append(m.T); amp.append(m.A); dec.append(m.D)
      rise.append(m.R); ph.append(m.phi)
    }
    let m0 = s.modes[0]
    push(m0)
    for (k, role) in template.roles.enumerated() {
      if let i = a.roleModes[k] {
        push(s.modes[i])
      } else {
        f.append(role.frequency); g.append(0); td.append(0.03); amp.append(silent)
        dec.append(role.decay); rise.append(0.0007); ph.append(0)
      }
    }
    r["f"] = f; r["g"] = g; r["td"] = td; r["amp"] = amp; r["dec"] = dec; r["rise"] = rise; r["ph"] = ph
    let tail = s.modes.first { $0.region == "fundamental-tail" }
    r["g2"] = [m0.G2]; r["t2"] = [m0.T2 > 0 ? m0.T2 : 0.003]
    r["tamp"] = [tail?.A ?? silent]; r["tdec"] = [tail?.D ?? 0.3]
    // strikes
    let st = s.strikes ?? []
    var xf = [Double](), xg = [Double](), xtd = [Double](), xamp = [Double](), xdec = [Double](), xph = [Double]()
    for (k, role) in strikeTemplate.roles.enumerated() {
      if k < strikeSlots.count, let i = strikeSlots[k] {
        let m = st[i]
        xf.append(m.F); xg.append(m.G); xtd.append(m.T); xamp.append(m.A); xdec.append(m.D); xph.append(m.phi)
      } else {
        xf.append(role.frequency); xg.append(0.1); xtd.append(0.003); xamp.append(silent)
        xdec.append(0.002); xph.append(0)
      }
    }
    r["xf"] = xf; r["xg"] = xg; r["xtd"] = xtd; r["xamp"] = xamp; r["xdec"] = xdec; r["xph"] = xph
    r["end"] = [s.gate.E]; r["fade"] = [s.gate.Fd]
    // hiss
    let nb = 16
    precondition(s.noise.bands.count == nb, "v3 rows need 16 noise bands (got \(s.noise.bands.count))")
    r["nl"] = s.noise.bands.map(\.L); r["nm"] = s.noise.bands.map(\.m)
    r["nd1"] = s.noise.bands.map(\.d1); r["nd2"] = s.noise.bands.map(\.d2); r["nrise"] = [s.noise.rn]
    // crackle
    let cr = s.crackle ?? Crackle.silent()
    r["cr_lvl"] = [cr.present ? cr.level : silent]; r["cr_rate"] = [cr.rate]
    r["cr_col"] = cr.colour; r["cr_dec"] = [cr.decay]
    // events (by time; unused: silent, colour 0.01)
    let evs = (s.noise.events ?? []).sorted { $0.t < $1.t }
    var evt = [Double](), evl = [Double](), evd = [Double](), evr = [Double](), evc = [Double]()
    for k in 0..<2 {
      if k < evs.count {
        let e = evs[k]
        evt.append(e.t); evl.append(e.level); evd.append(e.decay); evr.append(e.rise)
        evc += (0..<nb).map { $0 < e.colour.count ? e.colour[$0] : 0.01 }
      } else {
        evt.append(k == 0 ? 0.1 : 0.3); evl.append(silent); evd.append(0.05); evr.append(0.001)
        evc += [Double](repeating: 0.01, count: nb)
      }
    }
    r["ev_t"] = evt; r["ev_lvl"] = evl; r["ev_dec"] = evd; r["ev_rise"] = evr; r["ev_col"] = evc
    // glide floors (see the type comment)
    let gBox = effectiveBox("g")!, tdBox = effectiveBox("td")!
    var phs = r["ph"]!, gs = r["g"]!, tds = r["td"]!
    for k in 0..<gs.count {
      let F = r["f"]![k]
      let (g, t, dph) = glideFloor(F: F, g: gs[k], T: min(max(tds[k], tdBox.0), tdBox.1), gMin: gBox.0, tMin: tdBox.0)
      gs[k] = g; tds[k] = t; phs[k] += dph
    }
    let g2Box = effectiveBox("g2")!, t2Box = effectiveBox("t2")!
    let (g2, t2, dph2) = glideFloor(F: r["f"]![0], g: r["g2"]![0], T: min(max(r["t2"]![0], t2Box.0), t2Box.1),
                                    gMin: g2Box.0, tMin: t2Box.0)
    r["g2"] = [g2]; r["t2"] = [t2]; phs[0] += dph2
    r["g"] = gs; r["td"] = tds; r["ph"] = phs
    // hiss bands with m beyond the effective box: m = 0.99 would leave 1% of L on the slow
    // decay (a floor ~40 dB under a fast burst that lasts nd2); give that 1% the fast decay
    let nmBox = effectiveBox("nm")!, nd2Box = effectiveBox("nd2")!
    var nm = r["nm"]!, nd1 = r["nd1"]!, nd2 = r["nd2"]!
    for b in 0..<nm.count {
      if nm[b] > nmBox.1 { nm[b] = nmBox.1; nd2[b] = max(nd1[b], nd2Box.0) }
      if nm[b] < nmBox.0 { nm[b] = nmBox.0 }  // 1% on the short fast decay: harmless
    }
    r["nm"] = nm; r["nd1"] = nd1; r["nd2"] = nd2
    // event colours top out at the box's effective maximum: rescale colour and level together
    let ecBox = effectiveBox("ev_col")!
    var evc2 = r["ev_col"]!, evl2 = r["ev_lvl"]!
    for e in 0..<2 {
      let mx = evc2[(16 * e)..<(16 * e + 16)].max() ?? 1
      if mx > ecBox.1 {
        for b in 0..<16 { evc2[16 * e + b] *= ecBox.1 / mx }
        evl2[e] *= mx / ecBox.1
      }
    }
    r["ev_col"] = evc2; r["ev_lvl"] = evl2
    let xgBox = effectiveBox("xg")!, xtdBox = effectiveBox("xtd")!
    var sph = r["xph"]!, sg = r["xg"]!, std = r["xtd"]!
    for k in 0..<sg.count {
      let (g, t, dph) = glideFloor(F: r["xf"]![k], g: sg[k], T: min(max(std[k], xtdBox.0), xtdBox.1),
                                   gMin: xgBox.0, tMin: xtdBox.0)
      sg[k] = g; std[k] = t; sph[k] += dph
    }
    r["xg"] = sg; r["xtd"] = std; r["xph"] = sph
    // clamp to the effective boxes; wrap phases
    for (name, width, _, _) in layout {
      guard var v = r[name] else { preconditionFailure("v3 row misses \(name)") }
      precondition(v.count == width, "v3 row \(name): \(v.count) values, layout says \(width)")
      if let (lo, hi) = effectiveBox(name) { v = v.map { min(max($0, lo), hi) } }
      else { v = v.map { remainder($0, 2 * Double.pi) } }
      r[name] = v
    }
    return r
  }
}

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
    let n = slots - 1
    var pts: [(lf: Double, w: Double, d: Double)] = []
    for s in samples {
      for m in s.modes where (m.region == "low" || m.region == "mid") && m.D >= ringing && m.A > 0 {
        pts.append((log2(m.F), energyWeighted ? max(m.energy, 1e-12).squareRoot() : 1, m.D))
      }
    }
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

// Noise band envelopes of the residual (prepared - modal resynth).
// Model: x_noise = Σ_b N_b(t) band_b(t), band_b = unit-RMS 2nd-order
// band-pass of white (NoiseBands).
//
// With ONE shared white (SPEC §3.1) the band signals are correlated: the
// 2nd-order skirts overlap and add coherently, so the power seen through
// analysis filter a is a quadratic form
//   e_a(t) = Σ_{b,c} N_b(t) N_c(t) K_a[b][c],
//   K_a[b][c] = ∫ |H_a|² Re(H_b conj H_c) df / sqrt(∫|H_b|² df ∫|H_c|² df).
// With independent whites per band only the diagonal survives
// (K_a[b][b] = M_ab). Analysis filters are the same band-passes applied
// zero-phase (|H_a|) so the envelopes are not delayed.
// Steps:
//   1. e_a(t): squared zero-phase band output, Hann-smoothed
//      (clamp(4/bw, 2 ms, 30 ms)), on a frame grid that is dense early
//      (0.5 ms to onset+20 ms, 1 ms to onset+60 ms, 4 ms after);
//   2. per-frame NNLS with the diagonal approximation → N_b(t)² tracks;
//   3. per-band LM fit (dB, floor) of L_b, m_b, d1_b, d2_b with r_n fixed;
//   4. joint LM over all bands + shared r_n against e_a(t) in dB with the
//      exact (quadratic) model.

import Foundation

public struct NoiseFitResult: Codable {
    public var bands: [NoiseBandParams]
    public var rn: Double
    public var fitRMSdB: Double
    public var sharedWhite: Bool
    public var sections: Int
    public var trackT: [Double]?
    public var trackN2: [[Double]]?  // [band][frame] unmixed N_b² (debug)
    /// v3: noise events (bursts on the same bands) admitted by the joint fit
    public var events: [NoiseEventParams]? = nil
    /// v3: secondary-onset candidates (time, mean dB rise over the bands >= 1 kHz)
    public var eventCandidates: [[Double]]? = nil
}

public enum NoiseFit {
    public struct Options {
        public var floorDB = 70.0
        public var jointRefine = true
        public var jointIterations = 30
        public var sharedWhite = true
        public var keepTracks = false
        public var analysisPerBand = 3
        public var sections = 1
        /// weight of the neighbour-band smoothness prior
        public var smoothness = 0.05
        /// neighbour-band prior of the per-frame unmixing (per dB of N²)
        public var unmixSmoothness = 0.3
        /// v3: per-sample weights in [0, 1] (0 = gated out, e.g. crackle clicks); band powers are
        /// measured on the gated residual and divided by the equally smoothed mask
        public var sampleMask: [Double]? = nil
        /// the mask gates only the residual above this frequency (gating the loud low end would
        /// splatter it over the quiet high bands); bands centred below 0.75x it are not renormalised
        public var maskHighpassHz = 800.0
        /// v3: up to this many noise events (0 = none, the v1/v2 model)
        public var maxEvents = 0
        /// minimum mean dB rise (bands >= 1 kHz, vs the minimum of the preceding 40 ms) of a
        /// secondary-onset candidate
        public var eventRiseDB = 6.0
        /// an event is admitted when the dB RMS of the band-power fit falls by at least this share
        public var eventMinGain = 0.03
        public var onsetEventMinGain = 0.10
        /// snare: a stationary per-band floor H_b (record hiss) beside the decaying two-stage
        /// envelope (the wires): N_b = (L[m e^{-t/d1} + (1-m) e^{-t/d2}] + H_b)(1 - e^{-t/r_n})
        public var floorTerm = false
        /// upper bound of the slow decay d2 (s); with floorTerm a short cap keeps stationary noise
        /// out of the wire envelope
        public var maxSlowDecay = 3.0
        public init() {}
    }

    /// K[j][b][c] (see header) for analysis bands given as weights w_j(f)
    /// on the grid. Diagonal-only in (b, c) when !shared.
    public static func kernel(edges: [Double], analysis: [(lo: Double, hi: Double)], sr: Double, shared: Bool,
                              sections: Int = 1, gridPoints: Int = 8192) -> [[[Double]]]
    {
        let nb = edges.count - 1
        let cs = (0..<nb).map { NoiseBands.bandpass(lo: edges[$0], hi: edges[$0 + 1], sampleRate: sr) }
        let fgrid = (1..<gridPoints).map { Double($0) * 0.5 * sr / Double(gridPoints) }
        var Hr = [[Double]](), Hi = [[Double]]()
        for c in cs {
            var r = [Double](), i = [Double]()
            for f in fgrid {
                let w = 2 * Double.pi * f / sr
                let nr = c.b0 + c.b1 * cos(w) + c.b2 * cos(2 * w), ni = -(c.b1 * sin(w) + c.b2 * sin(2 * w))
                let dr = 1 + c.a1 * cos(w) + c.a2 * cos(2 * w), di = -(c.a1 * sin(w) + c.a2 * sin(2 * w))
                let d = dr * dr + di * di
                let hr = (nr * dr + ni * di) / d, hi = (ni * dr - nr * di) / d
                var pr = 1.0, pi = 0.0
                for _ in 0..<sections { (pr, pi) = (pr * hr - pi * hi, pr * hi + pi * hr) }
                r.append(pr)
                i.append(pi)
            }
            Hr.append(r)
            Hi.append(i)
        }
        var norms = [Double](repeating: 0, count: nb)
        for b in 0..<nb { for k in 0..<fgrid.count { norms[b] += Hr[b][k] * Hr[b][k] + Hi[b][k] * Hi[b][k] } }
        let nj = analysis.count
        var K = [[[Double]]](repeating: [[Double]](repeating: [Double](repeating: 0, count: nb), count: nb), count: nj)
        for j in 0..<nj {
            let g = FFT.bandGain(lo: analysis[j].lo, hi: analysis[j].hi, taper: analysisTaper)
            let ks = fgrid.indices.filter { g(fgrid[$0]) > 0 }
            let wk = ks.map { let v = g(fgrid[$0]); return v * v }
            for b in 0..<nb {
                for c in b..<nb {
                    if !shared && b != c { continue }
                    var s = 0.0
                    for (q, k) in ks.enumerated() { s += wk[q] * (Hr[b][k] * Hr[c][k] + Hi[b][k] * Hi[c][k]) }
                    let v = s / (norms[b] * norms[c]).squareRoot()
                    K[j][b][c] = v
                    K[j][c][b] = v
                }
            }
        }
        return K
    }

    static let analysisTaper = 0.04

    /// Analysis bands: `perBand` x NB log-spaced FFT bands spanning the
    /// synthesis range widened by half an octave each side.
    static func analysisBands(edges: [Double], perBand: Int, sr: Double) -> [(lo: Double, hi: Double)] {
        let lo = max(20, edges.first! / 1.41), hi = min(edges.last! * 1.41, 0.47 * sr)
        let n = perBand * (edges.count - 1) + 2
        let e = (0...n).map { lo * pow(hi / lo, Double($0) / Double(n)) }
        return (0..<n).map { (e[$0], e[$0 + 1]) }
    }

    static func frameGrid(onset: Double, end: Double) -> [Double] {
        var t: [Double] = []
        var x = 0.0
        while x < end {
            t.append(x)
            x += x < onset + 0.02 ? 0.0005 : (x < onset + 0.06 ? 0.001 : 0.004)
        }
        return t
    }

    public static func fit(residual r: [Double], sr: Double, onset: Double, end: Double, edges: [Double],
                           options: Options = Options()) -> NoiseFitResult
    {
        let nb = edges.count - 1
        let abands = analysisBands(edges: edges, perBand: options.analysisPerBand, sr: sr)
        let na = abands.count
        let K = kernel(edges: edges, analysis: abands, sr: sr, shared: options.sharedWhite, sections: options.sections)
        let M = (0..<na).map { a in (0..<nb).map { b in K[a][b][b] } }
        let nEnd = min(r.count, Int(end * sr))
        let ft = frameGrid(onset: onset, end: Double(nEnd) / sr)
        let frames = ft.map { Int($0 * sr) }
        // 1. band powers (on the gated residual, renormalised by the smoothed gate)
        let mask = options.sampleMask.flatMap { $0.count == r.count ? $0 : nil }
        var rg = r
        if let m = mask {
            let hp = FFT.spectralFilter(r, sampleRate: sr, analytic: false,
                                        gain: FFT.bandGain(lo: options.maskHighpassHz, hi: 0, taper: 0.25)).re
            for i in 0..<r.count { rg[i] = r[i] - (1 - m[i]) * hp[i] }
        }
        var e = [[Double]](repeating: [Double](repeating: 0, count: frames.count), count: na)
        for a in 0..<na {
            let mask = (abands[a].lo * abands[a].hi).squareRoot() >= 0.75 * options.maskHighpassHz ? mask : nil
            let y = FFT.spectralFilter(rg, sampleRate: sr, analytic: false,
                                       gain: FFT.bandGain(lo: abands[a].lo, hi: abands[a].hi, taper: analysisTaper)).re
            let bw = abands[a].hi - abands[a].lo
            let wl = max(3, Int(min(max(4 / bw, 0.002), 0.03) * sr))
            let w = hann(wl)
            let wsum = w.reduce(0, +)
            for (fi, c0) in frames.enumerated() {
                var s = 0.0, ms = 0.0
                for k in 0..<wl {
                    let i = c0 - wl / 2 + k
                    if i >= 0 && i < nEnd {
                        s += w[k] * y[i] * y[i]
                        if let m = mask { ms += w[k] * m[i] }
                    }
                }
                e[a][fi] = mask == nil ? s / wsum : (ms >= 0.5 * wsum ? s / ms : -1)
            }
            // frames mostly gated out (a click under the window): interpolate in dB from the
            // neighbouring measured frames of the band
            if mask != nil {
                let ok = frames.indices.filter { e[a][$0] >= 0 }
                for fi in frames.indices where e[a][fi] < 0 {
                    let prev = ok.last { $0 < fi }, next = ok.first { $0 > fi }
                    switch (prev, next) {
                    case let (p?, q?):
                        let w = (ft[fi] - ft[p]) / max(ft[q] - ft[p], 1e-9)
                        e[a][fi] = exp((1 - w) * log(e[a][p] + 1e-30) + w * log(e[a][q] + 1e-30))
                    case let (p?, nil): e[a][fi] = e[a][p]
                    case let (nil, q?): e[a][fi] = e[a][q]
                    default: e[a][fi] = 0
                    }
                }
            }
        }
        let emax = e.flatMap { $0 }.max() ?? 1
        let floorE = max(emax, 1e-20) * pow(10, -options.floorDB / 10)
        // 2. unmix per frame (diagonal approximation), in the log domain
        //    with a neighbour-band smoothness prior; warm-started frame to frame
        var n2 = [[Double]](repeating: [Double](repeating: 0, count: frames.count), count: nb)
        var q = [Double](repeating: log(max(emax, 1e-20)) - 2, count: nb)  // ln N_b²
        let qlo = log(floorE) - 6, qhi = log(max(emax, 1e-20)) + 6
        let wq = options.unmixSmoothness
        for fi in 0..<frames.count {
            let eb = (0..<na).map { e[$0][fi] }
            let f = LinAlg.levenbergMarquardt(x0: q, lower: [Double](repeating: qlo, count: nb),
                                              upper: [Double](repeating: qhi, count: nb), maxIter: 30) { qq in
                var out = [Double](repeating: 0, count: na + nb - 1)
                let v = qq.map { exp($0) }
                for a in 0..<na {
                    var s = 0.0
                    for b in 0..<nb { s += M[a][b] * v[b] }
                    out[a] = 10 * log10(s + floorE) - 10 * log10(eb[a] + floorE)
                }
                for b in 0..<(nb - 1) { out[na + b] = wq * 4.343 * (qq[b] - qq[b + 1]) }
                return out
            }
            q = f.x
            for b in 0..<nb { n2[b][fi] = exp(q[b]) }
        }
        let n2max = n2.flatMap { $0 }.max() ?? 1
        let floorN = max(n2max, 1e-20) * pow(10, -options.floorDB / 10)
        let fitIdx = frames.indices.filter { ft[$0] >= onset * 0.5 }
        // 3. per-band fits
        var rn = 0.0005
        let ps = options.floorTerm ? 5 : 4  // per-band parameter stride: [ln L, m, d1, d2(, ln H)]
        @inline(__always) func env(_ p: [Double], _ o: Int, _ t: Double, _ rn: Double) -> Double {
            let L = exp(p[o]), m = p[o + 1], d1 = p[o + 2], d2 = p[o + 3]
            let h = ps == 5 ? exp(p[o + 4]) : 0
            return (L * (m * exp(-t / d1) + (1 - m) * exp(-t / d2)) + h) * (1 - exp(-t / rn))
        }
        var params: [Double] = []
        // L may sit at most 20 dB below the noise floor (no dead bands)
        let lnLo = 0.5 * log(floorN) - 2.3
        var lo4 = [lnLo, 0.0, 0.0005, 0.004], hi4 = [log(10.0), 1.0, 0.04, options.maxSlowDecay]
        if ps == 5 { lo4.append(lnLo - 2.3); hi4.append(log(10.0)) }
        for b in 0..<nb {
            let tr = n2[b]
            let early = fitIdx.filter { ft[$0] < onset + 0.012 }.map { tr[$0] }.max() ?? 0
            let L0 = max(early, floorN * 10).squareRoot()
            // floor start: the 20th percentile of the track's last 40% (the stationary part)
            var hx: [Double] = []
            if ps == 5 {
                let late = fitIdx.filter { ft[$0] > onset + 0.6 * (ft[fitIdx.last!] - onset) }.map { tr[$0] }.sorted()
                let h0 = late.isEmpty ? floorN : late[Int(0.2 * Double(late.count - 1))]
                hx = [min(max(0.5 * log(max(h0, floorN)), lo4[4] + 0.1), hi4[4] - 0.1)]
            }
            var best: [Double] = [log(L0), 0.7, 0.004, 0.08] + hx
            var bestC = Double.infinity
            for d1i in [0.002, 0.008] {
                for d2i in [0.04, 0.2] {
                    let f = LinAlg.levenbergMarquardt(x0: [log(L0 * 1.3), 0.7, d1i, d2i] + hx, lower: lo4, upper: hi4,
                                                      maxIter: 60) { p in
                        fitIdx.map { i in
                            let v = env(p, 0, ft[i], rn)
                            return 10 * log10(v * v + floorN) - 10 * log10(tr[i] + floorN)
                        }
                    }
                    if f.cost < bestC {
                        bestC = f.cost
                        best = f.x
                    }
                }
            }
            params += best
        }
        // 4. joint refine incl. r_n (and v3 noise events) against e_a in dB, exact mixing model.
        //    Parameter vector: [ln L, m, d1, d2] x nb, r_n, then per event
        //    [t, ln level, ln decay, ln rise, ln colour x nb].
        let ev0 = ps * nb + 1, evW = 4 + nb
        func residualE(_ p: [Double]) -> [Double] {
            let rnv = p[ps * nb]
            let nev = (p.count - ev0) / evW
            var out = [Double](repeating: 0, count: fitIdx.count * na + ps * (nb - 1) + nev * (nb - 1))
            var N = [Double](repeating: 0, count: nb)
            let evs: [(t: Double, L: Double, d: Double, r: Double, c: [Double])] = (0..<nev).map { k in
                let o = ev0 + k * evW
                return (p[o], exp(p[o + 1]), exp(p[o + 2]), exp(p[o + 3]), (0..<nb).map { exp(p[o + 4 + $0]) })
            }
            var o = 0
            for i in fitIdx {
                let t = ft[i]
                for b in 0..<nb { N[b] = env(p, ps * b, t, rnv) }
                for ev in evs where t > ev.t {
                    let d = t - ev.t
                    let g = ev.L * (1 - exp(-d / ev.r)) * exp(-d / ev.d)
                    for b in 0..<nb { N[b] += g * ev.c[b] }
                }
                for a in 0..<na {
                    var s = 0.0
                    if options.sharedWhite {
                        let Ka = K[a]
                        for b in 0..<nb {
                            var t = 0.0
                            let row = Ka[b]
                            for c in 0..<nb { t += row[c] * N[c] }
                            s += N[b] * t
                        }
                    } else {
                        for b in 0..<nb { s += M[a][b] * N[b] * N[b] }
                    }
                    out[o] = 10 * log10(max(s, 0) + floorE) - 10 * log10(e[a][i] + floorE)
                    o += 1
                }
            }
            // smoothness across neighbouring bands (dB of L, log of times, m):
            // makes the overlapping-band unmixing well posed
            let w = options.smoothness * Double(fitIdx.count * na).squareRoot() / Double(nb)
            for b in 0..<(nb - 1) {
                let p0 = ps * b, p1 = ps * b + ps
                out[o] = w * 8.686 * (p[p0] - p[p1]); o += 1
                out[o] = w * 4 * (p[p0 + 1] - p[p1 + 1]); o += 1
                out[o] = w * 2 * log(p[p0 + 2] / p[p1 + 2]); o += 1
                out[o] = w * 2 * log(p[p0 + 3] / p[p1 + 3]); o += 1
                if ps == 5 { out[o] = w * 8.686 * (p[p0 + 4] - p[p1 + 4]); o += 1 }
            }
            for k in 0..<nev {
                let c0 = ev0 + k * evW + 4
                for b in 0..<(nb - 1) { out[o] = w * 8.686 * (p[c0 + b] - p[c0 + b + 1]); o += 1 }
            }
            return out
        }
        func dataRMS(_ p: [Double]) -> Double {
            let res = residualE(p).prefix(fitIdx.count * na)
            return (res.reduce(0) { $0 + $1 * $1 } / Double(max(res.count, 1))).squareRoot()
        }
        var hissLo = [Double](), hissHi = [Double]()
        for _ in 0..<nb {
            hissLo += lo4
            hissHi += hi4
        }
        hissLo.append(0.0001)
        hissHi.append(0.01)
        var full = params + [rn]
        if options.jointRefine {
            let f = LinAlg.levenbergMarquardt(x0: full, lower: hissLo, upper: hissHi, maxIter: options.jointIterations,
                                              residuals: residualE)
            full = f.x
        }
        // v3 noise events: secondary onsets first, then an onset event; each admitted only when the
        // band-power fit improves by the set share (model selection, not a threshold on levels)
        var events: [NoiseEventParams] = []
        var candidates: [[Double]] = []
        if options.maxEvents > 0 {
            let hb = (0..<na).filter { (abands[$0].lo * abands[$0].hi).squareRoot() >= 1000 }
            let dB = (0..<na).map { a in e[a].map { 10 * log10($0 + floorE) } }
            var O = [Double](repeating: 0, count: frames.count)
            for fi in frames.indices where ft[fi] >= onset + 0.02 && ft[fi] <= end - 0.03 {
                let t = ft[fi]
                let prev = frames.indices.filter { ft[$0] >= t - 0.04 && ft[$0] <= t - 0.003 }
                guard !prev.isEmpty else { continue }
                var s = 0.0
                for a in hb {
                    let mn = prev.map { dB[a][$0] }.min()!
                    s += max(0, dB[a][fi] - mn)
                }
                O[fi] = s / Double(max(hb.count, 1))
            }
            var peaks: [(fi: Int, v: Double)] = []
            for fi in frames.indices where O[fi] >= options.eventRiseDB {
                let lo = fi > 0 ? O[fi - 1] : 0, hi = fi + 1 < frames.count ? O[fi + 1] : 0
                if O[fi] >= lo && O[fi] >= hi { peaks.append((fi, O[fi])) }
            }
            peaks.sort { $0.v > $1.v }
            var picked: [(fi: Int, v: Double)] = []
            for pk in peaks where !picked.contains(where: { abs(ft[$0.fi] - ft[pk.fi]) < 0.03 }) {
                picked.append(pk)
                if picked.count >= options.maxEvents + 2 { break }
            }
            // the rise peaks after the onset: walk back to where O first exceeded half of it
            func startTime(_ fi: Int) -> Double {
                var j = fi
                while j > 0 && O[j - 1] > 0.3 * O[fi] && ft[j - 1] > ft[fi] - 0.03 { j -= 1 }
                return ft[j]
            }
            candidates = picked.map { [startTime($0.fi), $0.v] }
            var evLo = [Double](), evHi = [Double](), evKinds = [String](), evGains = [Double]()
            var rms = dataRMS(full)
            func tryEvent(_ init0: [Double], lo: [Double], hi: [Double], kind: String, minGain: Double) {
                let x0 = full + init0
                let f = LinAlg.levenbergMarquardt(x0: x0, lower: hissLo + evLo + lo, upper: hissHi + evHi + hi,
                                                  maxIter: options.jointIterations, residuals: residualE)
                let r2 = dataRMS(f.x)
                let gain = 1 - r2 / max(rms, 1e-9)
                if ProcessInfo.processInfo.environment["DGEN_ANALYSIS_TRACE"] != nil {
                    fputs(String(format: "event %@ t0 %.3f: rms %.3f -> %.3f dB (gain %.3f)\n", kind, init0[0], rms, r2, gain), stderr)
                }
                if gain >= minGain {
                    full = f.x
                    rms = r2
                    evLo += lo
                    evHi += hi
                    evKinds.append(kind)
                    evGains.append(gain)
                }
            }
            let lnFloorL = 0.5 * log(floorN) - 2.3
            for c in candidates {
                if evKinds.count >= options.maxEvents { break }
                let tc = c[0]
                // initial level / colour: excess of the unmixed band tracks over the model, 0..25 ms after tc
                let win = frames.indices.filter { ft[$0] >= tc && ft[$0] <= tc + 0.025 }
                guard !win.isEmpty else { continue }
                var ex = [Double](repeating: 0, count: nb)
                for b in 0..<nb {
                    for fi in win {
                        let m = env(full, ps * b, ft[fi], full[ps * nb])
                        ex[b] = max(ex[b], max(0, n2[b][fi] - m * m).squareRoot())
                    }
                }
                let L0 = max(ex.max() ?? 0, exp(lnFloorL + 1))
                let col = ex.map { log(min(max($0 / L0, 1e-3), 1)) }
                let t0 = max(0, tc - 0.002)
                tryEvent([t0, log(L0), log(0.05), log(0.001)] + col,
                         lo: [max(0, tc - 0.015), lnFloorL, log(0.002), log(0.0001)] + [Double](repeating: log(1e-4), count: nb),
                         hi: [tc + 0.01, log(10.0), log(1.0), log(0.02)] + [Double](repeating: 0, count: nb),
                         kind: "secondary", minGain: options.eventMinGain)
            }
            if evKinds.count < options.maxEvents {
                // an onset event (e.g. a hat on the hit): coloured like the hiss's fast stage above 2 kHz
                let fast = (0..<nb).map { b -> Double in
                    let fc = (edges[b] * edges[b + 1]).squareRoot()
                    return fc >= 2000 ? exp(full[ps * b]) * max(full[ps * b + 1], 0.05) : 0
                }
                let L0 = max(fast.max() ?? 0, exp(lnFloorL + 1))
                let col = fast.map { log(min(max($0 / L0, 1e-3), 1)) }
                let t0 = max(0, onset - 0.0005)
                tryEvent([t0, log(0.7 * L0), log(0.06), log(0.0005)] + col,
                         lo: [max(0, onset - 0.003), lnFloorL, log(0.002), log(0.0001)] + [Double](repeating: log(1e-4), count: nb),
                         hi: [onset + 0.005, log(10.0), log(1.0), log(0.02)] + [Double](repeating: 0, count: nb),
                         kind: "onset", minGain: options.onsetEventMinGain)
            }
            for (k, kind) in evKinds.enumerated() {
                let o = ev0 + k * evW
                var col = (0..<nb).map { exp(full[o + 4 + $0]) }
                let cm = col.max() ?? 1
                col = col.map { min(max($0 / cm, 1e-4), 1) }
                events.append(NoiseEventParams(t: full[o], level: exp(full[o + 1]) * cm, decay: exp(full[o + 2]),
                                               rise: exp(full[o + 3]), colour: col, kind: kind, gain: evGains[k]))
            }
            events.sort { $0.t < $1.t }
        }
        rn = full[ps * nb]
        let rms = dataRMS(full)
        var bands: [NoiseBandParams] = []
        for b in 0..<nb {
            var m = full[ps * b + 1], d1 = full[ps * b + 2], d2 = full[ps * b + 3]
            if d1 > d2 {
                swap(&d1, &d2)
                m = 1 - m
            }
            bands.append(NoiseBandParams(lo: edges[b], hi: edges[b + 1], L: exp(full[ps * b]), m: m, d1: d1, d2: d2,
                                         H: ps == 5 ? exp(full[ps * b + 4]) : nil))
        }
        var result = NoiseFitResult(bands: bands, rn: rn, fitRMSdB: rms, sharedWhite: options.sharedWhite,
                                    sections: options.sections, trackT: options.keepTracks ? ft : nil,
                                    trackN2: options.keepTracks ? n2 : nil)
        if options.maxEvents > 0 {
            result.events = events
            result.eventCandidates = candidates
        }
        return result
    }
}

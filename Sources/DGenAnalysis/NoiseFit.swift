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
        // 1. band powers
        var e = [[Double]](repeating: [Double](repeating: 0, count: frames.count), count: na)
        for a in 0..<na {
            let y = FFT.spectralFilter(r, sampleRate: sr, analytic: false,
                                       gain: FFT.bandGain(lo: abands[a].lo, hi: abands[a].hi, taper: analysisTaper)).re
            let bw = abands[a].hi - abands[a].lo
            let wl = max(3, Int(min(max(4 / bw, 0.002), 0.03) * sr))
            let w = hann(wl)
            let wsum = w.reduce(0, +)
            for (fi, c0) in frames.enumerated() {
                var s = 0.0
                for k in 0..<wl {
                    let i = c0 - wl / 2 + k
                    if i >= 0 && i < nEnd { s += w[k] * y[i] * y[i] }
                }
                e[a][fi] = s / wsum
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
        @inline(__always) func env(_ p: [Double], _ o: Int, _ t: Double, _ rn: Double) -> Double {
            let L = exp(p[o]), m = p[o + 1], d1 = p[o + 2], d2 = p[o + 3]
            return L * (m * exp(-t / d1) + (1 - m) * exp(-t / d2)) * (1 - exp(-t / rn))
        }
        var params: [Double] = []
        // L may sit at most 20 dB below the noise floor (no dead bands)
        let lo4 = [0.5 * log(floorN) - 2.3, 0.0, 0.0005, 0.004], hi4 = [log(10.0), 1.0, 0.04, 3.0]
        for b in 0..<nb {
            let tr = n2[b]
            let early = fitIdx.filter { ft[$0] < onset + 0.012 }.map { tr[$0] }.max() ?? 0
            let L0 = max(early, floorN * 10).squareRoot()
            var best: [Double] = [log(L0), 0.7, 0.004, 0.08]
            var bestC = Double.infinity
            for d1i in [0.002, 0.008] {
                for d2i in [0.04, 0.2] {
                    let f = LinAlg.levenbergMarquardt(x0: [log(L0 * 1.3), 0.7, d1i, d2i], lower: lo4, upper: hi4,
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
        // 4. joint refine incl. r_n against e_a in dB, exact mixing model
        func residualE(_ p: [Double]) -> [Double] {
            let rnv = p[4 * nb]
            var out = [Double](repeating: 0, count: fitIdx.count * na + 4 * (nb - 1))
            var N = [Double](repeating: 0, count: nb)
            var o = 0
            for i in fitIdx {
                for b in 0..<nb { N[b] = env(p, 4 * b, ft[i], rnv) }
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
                let p0 = 4 * b, p1 = 4 * b + 4
                out[o] = w * 8.686 * (p[p0] - p[p1]); o += 1
                out[o] = w * 4 * (p[p0 + 1] - p[p1 + 1]); o += 1
                out[o] = w * 2 * log(p[p0 + 2] / p[p1 + 2]); o += 1
                out[o] = w * 2 * log(p[p0 + 3] / p[p1 + 3]); o += 1
            }
            return out
        }
        var full = params + [rn]
        if options.jointRefine {
            var lo = [Double](), hi = [Double]()
            for _ in 0..<nb {
                lo += lo4
                hi += hi4
            }
            lo.append(0.0001)
            hi.append(0.01)
            let f = LinAlg.levenbergMarquardt(x0: full, lower: lo, upper: hi, maxIter: options.jointIterations,
                                              residuals: residualE)
            full = f.x
        }
        rn = full[4 * nb]
        let res = residualE(full).prefix(fitIdx.count * na)
        let rms = (res.reduce(0) { $0 + $1 * $1 } / Double(max(res.count, 1))).squareRoot()
        var bands: [NoiseBandParams] = []
        for b in 0..<nb {
            var m = full[4 * b + 1], d1 = full[4 * b + 2], d2 = full[4 * b + 3]
            if d1 > d2 {
                swap(&d1, &d2)
                m = 1 - m
            }
            bands.append(NoiseBandParams(lo: edges[b], hi: edges[b + 1], L: exp(full[4 * b]), m: m, d1: d1, d2: d2))
        }
        return NoiseFitResult(bands: bands, rn: rn, fitRMSdB: rms, sharedWhite: options.sharedWhite,
                              sections: options.sections, trackT: options.keepTracks ? ft : nil, trackN2: options.keepTracks ? n2 : nil)
    }
}

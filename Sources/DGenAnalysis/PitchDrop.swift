// Fundamental pitch-drop law from the instantaneous frequency of the
// low-passed analytic signal:
//   f(t) = F (1 + G e^{-t/T} [+ G2 e^{-t/T2}])
// 1. analytic signal of x band-limited to [hp, lp] Hz (FFT Hilbert);
// 2. frame IF = arg Σ z[i+1] conj z[i] (amplitude-weighted within a frame),
//    frame weight = Σ |z|²;
// 3. robust (Cauchy IRLS) weighted Levenberg-Marquardt fit of the law;
// 4. phase-locked refinement: demodulate z by the fitted phase law,
//    low-pass the baseband (isolates the fundamental from neighbouring
//    partials), add its IF to the model and refit;
// 5. optional second drop term, accepted only when it reduces the
//    weighted residual by `secondDropGain` and is distinct in time scale.

import Foundation

public struct PitchDropFit: Codable {
    public var F: Double
    public var G: Double
    public var T: Double
    public var G2: Double
    public var T2: Double
    public var secondDrop: Bool
    public var weightedRMSHz: Double
    public var weightedRMSHzSingle: Double
    public var fStartHz: Double   // law at the onset
    public var frames: Int
}

public struct IFTrack: Codable {
    public var t: [Double]
    public var f: [Double]
    public var w: [Double]
}

public enum PitchDrop {
    /// Bounds of (F, G, T) and (F, G, T, G2, T2).
    public static let lo3 = [25.0, 0, 0.0008], hi3 = [500.0, 20, 0.15]
    public static let lo5 = [25.0, 0, 0.0008, 0, 0.0005], hi5 = [500.0, 20, 0.15, 20, 0.15]
    public struct Options {
        public var highpassHz = 25.0
        public var lowpassHz = 400.0
        public var frameSeconds = 0.0005
        public var floorDB = 35.0           // frames below peak - floorDB are ignored
        public var basebandHz = 30.0        // phase-locked baseband low-pass
        public var refinements = 3
        public var allowSecondDrop = true
        public var secondDropGain = 0.7     // accept if SSE2 < gain * SSE1
        public init() {}
    }

    public struct Analytic {
        public var re: [Double]
        public var im: [Double]
    }

    public static func analytic(_ x: [Double], sr: Double, lo: Double, hi: Double) -> Analytic {
        let z = FFT.spectralFilter(x, sampleRate: sr, analytic: true, gain: FFT.bandGain(lo: lo, hi: hi, taper: 0.3))
        return Analytic(re: z.re, im: z.im)
    }

    /// Frame IF track of a complex signal between sample indices [a, b).
    static func ifTrack(re: [Double], im: [Double], sr: Double, a: Int, b: Int, hop: Int) -> IFTrack {
        var t: [Double] = [], f: [Double] = [], w: [Double] = []
        var s = a
        while s + hop < b {
            var cr = 0.0, ci = 0.0, e = 0.0
            for i in s..<(s + hop) {
                // z[i+1] * conj(z[i])
                cr += re[i + 1] * re[i] + im[i + 1] * im[i]
                ci += im[i + 1] * re[i] - re[i + 1] * im[i]
                e += re[i] * re[i] + im[i] * im[i]
            }
            t.append((Double(s) + Double(hop) / 2) / sr)
            f.append(atan2(ci, cr) * sr / (2 * Double.pi))
            w.append(e)
            s += hop
        }
        return IFTrack(t: t, f: f, w: w)
    }

    static func law(_ p: [Double], _ t: Double) -> Double {
        var g = 1 + p[1] * exp(-t / p[2])
        if p.count >= 5 { g += p[3] * exp(-t / p[4]) }
        return p[0] * g
    }

    static func phaseLaw(_ p: [Double], _ t: Double) -> Double {
        var q = p[0] * t + p[0] * p[1] * p[2] * (1 - exp(-t / p[2]))
        if p.count >= 5 { q += p[0] * p[3] * p[4] * (1 - exp(-t / p[4])) }
        return 2 * Double.pi * q
    }

    static func weightedMedian(_ v: [Double], _ w: [Double]) -> Double {
        let idx = v.indices.sorted { v[$0] < v[$1] }
        let tot = w.reduce(0, +)
        var acc = 0.0
        for i in idx {
            acc += w[i]
            if acc >= tot / 2 { return v[i] }
        }
        return v.last ?? 0
    }

    /// Robust weighted LM fit of the law to (t, f, w); returns params, SSE.
    /// With `fixedF`, p[0] is held at that value.
    static func fitLaw(track: IFTrack, init p0: [Double], lower: [Double], upper: [Double],
                       fixedF: Double? = nil) -> ([Double], Double)
    {
        let n = track.t.count
        let wsum = track.w.reduce(0, +)
        let wn = track.w.map { $0 / max(wsum, 1e-300) }
        var robust = [Double](repeating: 1, count: n)
        var p = p0
        var lo = lower, hi = upper
        if let F = fixedF {
            p[0] = F
            lo[0] = F
            hi[0] = F
        }
        var sse = 0.0
        for _ in 0..<4 {
            let rw = robust
            let fit = LinAlg.levenbergMarquardt(x0: p, lower: lo, upper: hi, maxIter: 80) { q in
                (0..<n).map { i in (wn[i] * rw[i]).squareRoot() * (law(q, track.t[i]) - track.f[i]) }
            }
            p = fit.x
            // Cauchy weights with a MAD scale
            let res = (0..<n).map { law(p, track.t[$0]) - track.f[$0] }
            let mad = weightedMedian(res.map { abs($0) }, wn)
            let c = max(2.0, 2.5 * mad)
            robust = res.map { 1 / (1 + ($0 / c) * ($0 / c)) }
            var num = 0.0, den = 0.0
            for i in 0..<n {
                let w = wn[i] * robust[i]
                num += w * res[i] * res[i]
                den += w
            }
            sse = num / max(den, 1e-300)
        }
        return (p, sse)
    }

    /// Cauchy cost Σ w c² ln(1 + (r/c)²) with a given scale (model comparison).
    static func cauchyCost(track: IFTrack, p: [Double], scale c: Double) -> Double {
        let wsum = track.w.reduce(0, +)
        var s = 0.0
        for i in track.t.indices {
            let r = (law(p, track.t[i]) - track.f[i]) / c
            s += track.w[i] / max(wsum, 1e-300) * c * c * log(1 + r * r)
        }
        return s
    }

    /// Refit G, T (and G2, T2 when the fit has a second drop) with F held.
    public static func refit(_ f: PitchDropFit, track: IFTrack, fixedF F: Double, onset: Double) -> PitchDropFit {
        guard track.t.count >= 8 else { var g = f; g.F = F; return g }
        var out = f
        if f.secondDrop {
            let (p, sse) = fitLaw(track: track, init: [F, f.G, f.T, f.G2, f.T2],
                                  lower: lo5, upper: hi5, fixedF: F)
            out.F = p[0]; out.G = p[1]; out.T = p[2]; out.G2 = p[3]; out.T2 = p[4]
            out.weightedRMSHz = sse.squareRoot()
            out.fStartHz = law(p, onset)
        } else {
            let (p, sse) = fitLaw(track: track, init: [F, f.G, f.T], lower: lo3, upper: hi3, fixedF: F)
            out.F = p[0]; out.G = p[1]; out.T = p[2]
            out.weightedRMSHz = sse.squareRoot()
            out.fStartHz = law(p, onset)
        }
        return out
    }

    public static func fit(_ x: [Double], sr: Double, onset: Double, end: Double,
                           options: Options = Options(), track trackOut: inout IFTrack?) -> PitchDropFit
    {
        let z = analytic(x, sr: sr, lo: options.highpassHz, hi: options.lowpassHz)
        let hop = max(4, Int(options.frameSeconds * sr))
        let a = min(x.count - 2, Int((onset + 0.001) * sr))
        let b = min(x.count - 1, Int(end * sr))
        var tr = ifTrack(re: z.re, im: z.im, sr: sr, a: max(0, a), b: b, hop: hop)
        let wmax = tr.w.max() ?? 0
        let keep = tr.w.indices.filter { tr.w[$0] > wmax * pow(10, -options.floorDB / 10) && tr.f[$0] > 5 }
        tr = IFTrack(t: keep.map { tr.t[$0] }, f: keep.map { tr.f[$0] }, w: keep.map { tr.w[$0] })
        trackOut = tr
        guard tr.t.count >= 8 else {
            return PitchDropFit(F: 50, G: 0, T: 0.03, G2: 0, T2: 0, secondDrop: false, weightedRMSHz: .nan,
                                weightedRMSHzSingle: .nan, fStartHz: 50, frames: tr.t.count)
        }
        // initial guess
        let tLast = tr.t.last!
        let lateIdx = tr.t.indices.filter { tr.t[$0] > onset + 0.5 * (tLast - onset) }
        let earlyIdx = tr.t.indices.filter { tr.t[$0] < onset + 0.006 }
        let fEnd = weightedMedian(lateIdx.map { tr.f[$0] }, lateIdx.map { tr.w[$0] })
        let fBeg = earlyIdx.isEmpty ? fEnd : weightedMedian(earlyIdx.map { tr.f[$0] }, earlyIdx.map { tr.w[$0] })
        var T0 = 0.03
        if fBeg > fEnd * 1.05 {
            let target = fEnd + (fBeg - fEnd) / M_E
            if let i = tr.t.indices.first(where: { tr.t[$0] > onset + 0.004 && tr.f[$0] < target }) {
                T0 = max(0.003, tr.t[i] - onset)
            }
        }
        let G0 = max(0, fBeg / max(fEnd, 1) - 1) * exp((onset + 0.003) / T0)
        var (p1, _) = fitLaw(track: tr, init: [max(fEnd, 20), min(G0, 19), T0], lower: lo3, upper: hi3)
        // phase-locked refinement
        let fullLen = x.count
        for _ in 0..<options.refinements {
            var ur = [Double](repeating: 0, count: fullLen), ui = ur
            for i in 0..<fullLen {
                let th = phaseLaw(p1, Double(i) / sr)
                let c = cos(th), s = sin(th)
                // z * e^{-iθ}
                ur[i] = z.re[i] * c + z.im[i] * s
                ui[i] = z.im[i] * c - z.re[i] * s
            }
            let u = FFT.complexFilter(re: ur, im: ui, sampleRate: sr,
                                      gain: FFT.bandGain(lo: 0, hi: options.basebandHz, taper: 0.5))
            var bt = ifTrack(re: u.re, im: u.im, sr: sr, a: max(0, a), b: b, hop: hop)
            let bw = bt.w.max() ?? 0
            let k2 = bt.w.indices.filter { bt.w[$0] > bw * pow(10, -options.floorDB / 10) }
            bt = IFTrack(t: k2.map { bt.t[$0] }, f: k2.map { bt.f[$0] + law(p1, bt.t[$0]) }, w: k2.map { bt.w[$0] })
            if bt.t.count < 8 { break }
            tr = bt
            (p1, _) = fitLaw(track: tr, init: p1, lower: lo3, upper: hi3)
        }
        trackOut = tr
        let (pS, sseS) = fitLaw(track: tr, init: p1, lower: lo3, upper: hi3)
        var best = pS + [0, 0]
        var bestSSE = sseS
        var second = false
        if options.allowSecondDrop {
            // compare models with one fixed Cauchy scale (from the single fit)
            let scale = max(2.0, 2.5 * sseS.squareRoot())
            let costS = cauchyCost(track: tr, p: pS, scale: scale)
            var bestCost = costS
            for (gs, ts) in [(0.3, 0.2), (0.5, 0.25), (0.3, 4.0)] {
                let start: [Double] = [pS[0], pS[1] * (1 - gs), pS[2], pS[1] * gs, max(0.0006, pS[2] * ts)]
                let (p2, sse2) = fitLaw(track: tr, init: start, lower: lo5, upper: hi5)
                let cost2 = cauchyCost(track: tr, p: p2, scale: scale)
                let ratio = max(p2[2], p2[4]) / max(min(p2[2], p2[4]), 1e-9)
                if cost2 < options.secondDropGain * bestCost && ratio > 2.5 && min(p2[1], p2[3]) > 0.03 {
                    bestCost = cost2
                    // order: (G, T) = slower term, (G2, T2) = faster term
                    best = p2[2] >= p2[4] ? p2 : [p2[0], p2[3], p2[4], p2[1], p2[2]]
                    bestSSE = sse2
                    second = true
                }
            }
        }
        return PitchDropFit(F: best[0], G: best[1], T: best[2], G2: best[3], T2: best[4], secondDrop: second,
                            weightedRMSHz: bestSSE.squareRoot(), weightedRMSHzSingle: sseS.squareRoot(),
                            fStartHz: law(second ? best : pS, onset), frames: tr.t.count)
    }
}

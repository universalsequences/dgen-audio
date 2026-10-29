// Strike sines (voice3 `xf xg xtd xamp xdec xph`): short decaying sines with an optional glide that
// carry the attack / overtone transients left after the modal part.
//
//   strike(t) = A e^{-t/D} (1 - e^{-10000 t}) sin(2π[F t + F G T (1 - e^{-t/T})] + φ)
//
// which is a ModeParams with R = 0.1 ms and its own (G, T): the analysis keeps strikes as
// ModeEstimate (region "strike") so the joint amplitude LS treats them like modes.
//
// Method: matrix pencil on the high-passed early residual (onset .. onset + window) gives
// candidate damped sinusoids; greedy forward selection by joint-LS SSE on the fit window
// (onset + fitWindow), a candidate admitted only when it removes >= minGain of the window's
// energy; then a joint variable-projection LM over (F, G, T, D) of the admitted strikes.

import Foundation

public enum StrikeFit {
    public struct Options {
        public var count = 4
        public var window = 0.03
        public var fitWindow = 0.05
        public var highpassHz = 100.0
        public var minGain = 0.02
        public var fRange = [100.0, 16000.0]
        public var gMax = 4.0
        public var tRange = [0.0003, 0.05]
        public var dRange = [0.0002, 0.05]
        public init() {}
    }

    public static let rise = 0.0001

    static func strike(F: Double, G: Double, T: Double, D: Double) -> ModeEstimate {
        ModeEstimate(F: F, G: G, T: T, A: 0, D: D, R: rise, phi: 0, G2: 0, T2: 0, region: "strike", energy: 0,
                     dropTestRatio: nil, amplitudeCapped: false)
    }

    /// Unused strike slots in the voice (silent; nominal values).
    public static let nominalF = [300.0, 900.0, 2500.0, 6000.0]

    public static func fit(residual r: [Double], sr: Double, onset: Double, options o: Options = Options())
        -> [ModeEstimate]
    {
        let n = r.count
        let hi = min(o.fRange[1] * 1.1, 0.47 * sr)
        let y = FFT.spectralFilter(r, sampleRate: sr, analytic: false,
                                   gain: FFT.bandGain(lo: o.highpassHz, hi: hi, taper: 0.1)).re
        let a = Int(onset * sr)
        let b = min(n, a + Int(o.window * sr))
        let nFit = min(n, a + Int(o.fitWindow * sr))
        guard b - a > 64 else { return [] }
        var ey = 0.0
        for i in 0..<nFit { ey += y[i] * y[i] }
        guard ey > 0 else { return [] }
        var mpo = MatrixPencil.Options()
        mpo.maxPencil = 100
        mpo.dynamicRangeDB = 50
        let mp = MatrixPencil.estimate(Array(y[a..<b]), sampleRate: sr, options: mpo).modes
        var cands = mp.filter {
            $0.frequency >= o.fRange[0] && $0.frequency <= o.fRange[1] && $0.decay >= o.dRange[0] * 0.5
        }.map { m in
            strike(F: m.frequency, G: 0, T: 0.003, D: min(max(m.decay, o.dRange[0]), o.dRange[1]))
        }
        guard !cands.isEmpty else { return [] }
        // greedy forward selection
        var chosen: [ModeEstimate] = []
        var base = ey
        while chosen.count < o.count && !cands.isEmpty {
            var best: (k: Int, sse: Double)? = nil
            for (k, c) in cands.enumerated() {
                var e = 0.0
                _ = KickAnalyzer.jointAmplitudes(chosen + [c], x: y, count: nFit, sr: sr, sse: &e)
                if best == nil || e < best!.sse { best = (k, e) }
            }
            guard let bk = best, base - bk.sse >= o.minGain * ey else { break }
            chosen.append(cands.remove(at: bk.k))
            base = bk.sse
            // drop candidates within 3% of a chosen frequency
            cands.removeAll { c in chosen.contains { abs($0.F / c.F - 1) < 0.03 } }
        }
        guard !chosen.isEmpty else { return [] }
        // joint refinement (variable projection): q = [ln F, G, ln T, ln D] per strike
        var x0 = [Double](), lo = [Double](), up = [Double]()
        for c in chosen {
            x0 += [log(c.F), 0.02, log(c.T), log(c.D)]
            lo += [log(max(o.fRange[0], c.F * 0.8)), 0, log(o.tRange[0]), log(o.dRange[0])]
            up += [log(min(o.fRange[1], c.F * 1.25)), o.gMax, log(o.tRange[1]), log(o.dRange[1])]
        }
        func modes(_ q: [Double]) -> [ModeEstimate] {
            (0..<chosen.count).map { j in
                strike(F: exp(q[4 * j]), G: q[4 * j + 1], T: exp(q[4 * j + 2]), D: exp(q[4 * j + 3]))
            }
        }
        func residual(_ q: [Double]) -> [Double] {
            let ms = KickAnalyzer.jointAmplitudes(modes(q), x: y, count: nFit, sr: sr)
            let s = KickVoice.renderModes(ms.map { $0.params }, count: nFit, sampleRate: sr)
            return (0..<nFit).map { y[$0] - s[$0] }
        }
        let f = LinAlg.levenbergMarquardt(x0: x0, lower: lo, upper: up, maxIter: 40, residuals: residual)
        var out = KickAnalyzer.jointAmplitudes(modes(f.x), x: y, count: nFit, sr: sr)
        // a strike that ended up negligible is dropped
        out = out.filter { $0.A > 0 && $0.energy > 1e-6 * ey }
        return out.sorted { $0.F < $1.F }
    }
}

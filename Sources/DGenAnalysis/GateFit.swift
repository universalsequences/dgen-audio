// End gate from the sample's own ending.
// E0 = last sample whose |x| exceeds peak·10^(endThresholdDB/20) (+1 sample).
// On the last `window` seconds: RMS envelope (1 ms window, 0.25 ms hop) is
// fitted in dB by  trend(t) · gate(t; E, Fd)  where trend is a log-linear
// decay (a + b t dB) and gate = 1 - smoothstep((t - E + Fd)/Fd).
// A hard chop yields Fd at the lower bound; a natural fade a longer Fd.

import Foundation

public struct GateFit: Codable {
    public var E: Double
    public var Fd: Double
    public var chopped: Bool       // Fd <= 3 ms
    public var lastLoudSeconds: Double
    public var endLevelDB: Double  // trend level at E - Fd re peak
}

public enum GateEstimator {
    public static func fit(_ x: [Double], sr: Double, endThresholdDB: Double = -60,
                           window: Double = 0.08, minFade: Double = 0.0004) -> GateFit
    {
        let pk = x.map { abs($0) }.max() ?? 1
        let thr = pk * pow(10, endThresholdDB / 20)
        let last = x.lastIndex { abs($0) > thr } ?? (x.count - 1)
        let E0 = Double(last + 1) / sr
        let hop = max(1, Int(0.00025 * sr))
        let half = max(1, Int(0.0005 * sr))
        let start = max(0, last - Int(window * sr))
        // analytic amplitude (a short RMS of a 50 Hz tone would oscillate)
        let z = FFT.spectralFilter(x, sampleRate: sr, analytic: true, gain: FFT.bandGain(lo: 20, hi: 0))
        var ts: [Double] = [], ed: [Double] = []
        var c = start
        while c < x.count {
            var s = 0.0, n = 0.0
            for i in max(0, c - half)..<min(x.count, c + half) {
                s += z.re[i] * z.re[i] + z.im[i] * z.im[i]
                n += 1
            }
            ts.append(Double(c) / sr)
            ed.append(10 * log10(s / max(n, 1) / (pk * pk) + 1e-10))
            c += hop
        }
        // beyond the file end the signal is zero
        for k in 1...8 {
            ts.append(Double(x.count - 1) / sr + Double(k * hop) / sr)
            ed.append(-200)
        }
        let floorDB = -90.0
        guard ts.count > 8 else {
            return GateFit(E: E0, Fd: minFade, chopped: true, lastLoudSeconds: E0, endLevelDB: 0)
        }
        // trend init: line through the first half of the window
        let h = ts.count / 2
        let tm = ts[0..<h].reduce(0, +) / Double(h), em = ed[0..<h].reduce(0, +) / Double(h)
        var sxy = 0.0, sxx = 0.0
        for i in 0..<h { sxy += (ts[i] - tm) * (ed[i] - em); sxx += (ts[i] - tm) * (ts[i] - tm) }
        let b0 = sxx > 0 ? sxy / sxx : 0
        let a0 = em - b0 * tm
        func model(_ p: [Double], _ t: Double) -> Double {
            let g = GateParams(E: p[2], Fd: p[3]).value(t)
            // RMS of a gated signal ≈ trend + 20log10(gate) (gate is the amplitude factor)
            return max(p[0] + p[1] * t + 20 * log10(max(g, 1e-9)), floorDB)
        }
        let fit = LinAlg.levenbergMarquardt(
            x0: [a0, b0, E0, 0.002], lower: [-120, -2000, E0 - window * 0.8, minFade],
            upper: [20, 200, E0 + 0.002, window * 0.8], maxIter: 150
        ) { p in ts.indices.map { model(p, ts[$0]) - max(ed[$0], floorDB) } }
        // the RMS window (1 ms) smears a hard chop; subtract its share
        let p = fit.x
        let E = min(p[2], Double(x.count) / sr)
        let Fd = max(minFade, p[3])
        return GateFit(E: E, Fd: Fd, chopped: Fd <= 0.003, lastLoudSeconds: E0,
                       endLevelDB: p[0] + p[1] * (E - Fd))
    }
}

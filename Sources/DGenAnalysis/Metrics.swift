// The kick-track check metrics, ported 1:1 from the research scripts
// (modal-autograd/kick/scripts/kick_metric.py band_score and
// modal-autograd/e1/metric.py score with windows 64..4096).
// Note: both scripts compute band membership with bin frequencies of a
// 48 kHz grid regardless of the actual sample rate; the port keeps that so
// numbers are comparable with the baseline.

import Foundation

public enum KickMetrics {
    // band_score: W = 2048, hop 256, 32 log bands 25 Hz..20 kHz (48 kHz bins),
    // W/2 zeros prepended, 4-frame moving average, 70 dB floor.
    static let bandW = 2048
    static let bandHop = 256

    static let bandMatrix: [[Double]] = {
        let edges = (0...32).map { exp(log(25.0) + (log(20000.0) - log(25.0)) * Double($0) / 32) }
        var rows: [[Double]] = []
        for b in 0..<32 {
            var row = [Double](repeating: 0, count: bandW / 2 + 1)
            var cnt = 0.0
            for k in 0...(bandW / 2) {
                let f = Double(k) * 48000.0 / Double(bandW)
                if f >= edges[b] && f < edges[b + 1] { row[k] = 1; cnt += 1 }
            }
            if cnt > 0 { rows.append(row) }
        }
        return rows
    }()

    /// Band envelopes in dB: [frame][band].
    public static func bandEnvDB(_ x0: [Double]) -> [[Double]] {
        let W = bandW, hop = bandHop
        let x = [Double](repeating: 0, count: W / 2) + x0
        if x.count < W { return [] }
        let win = hann(W)
        let nFrames = (x.count - W) / hop + 1
        let B = bandMatrix
        var E = [[Double]](repeating: [Double](repeating: 0, count: B.count), count: nFrames)
        let counts = B.map { max($0.reduce(0, +), 1) }
        for f in 0..<nFrames {
            let st = f * hop
            let frame = (0..<W).map { x[st + $0] * win[$0] }
            let P = FFT.powerSpectrum(frame, n: W)
            for b in 0..<B.count {
                var s = 0.0
                let row = B[b]
                for k in 0..<P.count where row[k] != 0 { s += P[k] }
                E[f][b] = s / counts[b]
            }
        }
        // np.convolve(c, ones(4)/4, mode="same") along time
        var S = E
        for b in 0..<B.count {
            for f in 0..<nFrames {
                // 'same' for len-4 kernel: out[i] = sum_{j} in[i + 1 - j], j=0..3 → window [i-2, i+1]
                var s = 0.0
                for d in -2...1 {
                    let g = f + d
                    if g >= 0 && g < nFrames { s += E[g][b] }
                }
                S[f][b] = 10 * log10(s / 4 + 1e-12)
            }
        }
        return S
    }

    public static func bandScore(_ p: [Double], _ t: [Double], floorDB: Double = 70) -> Double {
        let n = min(p.count, t.count)
        let P = bandEnvDB(Array(p[0..<n])), T = bandEnvDB(Array(t[0..<n]))
        return bandScore(envP: P, envT: T, floorDB: floorDB)
    }

    /// Band score from precomputed envelopes; optional frame range.
    public static func bandScore(envP P: [[Double]], envT T: [[Double]], floorDB: Double = 70,
                                 frames: Range<Int>? = nil) -> Double {
        let tmax = T.flatMap { $0 }.max() ?? 0
        let lo = tmax - floorDB
        let r = frames ?? 0..<T.count
        var s = 0.0, c = 0.0
        for f in r where f < T.count && f < P.count {
            for b in 0..<T[f].count {
                s += abs(max(P[f][b], lo) - max(T[f][b], lo))
                c += 1
            }
        }
        return c > 0 ? s / c : 0
    }

    /// Frame index of band_env_db whose window centre is at sample `i` of the
    /// unpadded signal (frames are centred thanks to the W/2 prepend).
    public static func bandFrame(forSample i: Int) -> Int { max(0, i / bandHop) }

    /// MR-STFT score (e1/metric.py `score`), default windows 64..4096.
    public static func mrScore(_ p0: [Double], _ t0: [Double],
                               windows: [Int] = [64, 128, 256, 512, 1024, 2048, 4096], eps: Double = 1e-3) -> Double {
        let n = min(p0.count, t0.count)
        var tot = 0.0
        for w in windows {
            let hop = w / 4
            let win = hann(w)
            if n < w { continue }
            let nFrames = (n - w) / hop + 1
            var acc = 0.0
            for f in 0..<nFrames {
                let st = f * hop
                let P = FFT.powerSpectrum((0..<w).map { p0[st + $0] * win[$0] }, n: w)
                let T = FFT.powerSpectrum((0..<w).map { t0[st + $0] * win[$0] }, n: w)
                for k in 0..<P.count {
                    acc += abs(P[k].squareRoot() - T[k].squareRoot())
                        + abs(0.5 * log(P[k] + eps * eps) - 0.5 * log(T[k] + eps * eps))
                }
            }
            tot += acc / Double(nFrames * (w / 2 + 1))
        }
        return tot / Double(windows.count)
    }

    public static func relRMS(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        var e = 0.0, s = 0.0
        for i in 0..<n {
            e += (a[i] - b[i]) * (a[i] - b[i])
            s += b[i] * b[i]
        }
        return (e / max(s, 1e-30)).squareRoot()
    }
}

// Portable radix-2 complex FFT and the spectral helpers the analysis needs
// (real spectra, analytic signal, zero-phase FFT filtering). No Accelerate:
// the analysis must build anywhere dgen builds, and the sizes involved
// (<= 65536 points, a few hundred transforms per sample) are small.

import Foundation

public enum FFT {
    /// Smallest power of two >= n.
    public static func nextPow2(_ n: Int) -> Int {
        var p = 1
        while p < n { p <<= 1 }
        return p
    }

    /// Cached twiddles per size (cos, sin of -2πk/n for k < n/2).
    private static var twiddleCache: [Int: ([Double], [Double])] = [:]
    private static let cacheLock = NSLock()

    private static func twiddles(_ n: Int) -> ([Double], [Double]) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let t = twiddleCache[n] { return t }
        var c = [Double](repeating: 0, count: n / 2)
        var s = [Double](repeating: 0, count: n / 2)
        for k in 0..<(n / 2) {
            let a = -2.0 * Double.pi * Double(k) / Double(n)
            c[k] = cos(a)
            s[k] = sin(a)
        }
        twiddleCache[n] = (c, s)
        return (c, s)
    }

    /// In-place complex FFT (forward: e^{-i...}; inverse scales by 1/n).
    public static func transform(_ re: inout [Double], _ im: inout [Double], inverse: Bool = false) {
        let n = re.count
        precondition(n == im.count && n > 0 && (n & (n - 1)) == 0, "FFT size must be a power of two")
        if n == 1 { return }
        // bit reversal
        var j = 0
        for i in 1..<n {
            var bit = n >> 1
            while j & bit != 0 {
                j ^= bit
                bit >>= 1
            }
            j ^= bit
            if i < j {
                re.swapAt(i, j)
                im.swapAt(i, j)
            }
        }
        let (tc, ts) = twiddles(n)
        let sign: Double = inverse ? -1 : 1
        var len = 2
        while len <= n {
            let half = len / 2
            let step = n / len
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var start = 0
                    while start < n {
                        var k = 0
                        for p in start..<(start + half) {
                            let wr = tc[k]
                            let wi = sign * ts[k]
                            let q = p + half
                            let xr = rp[q] * wr - ip[q] * wi
                            let xi = rp[q] * wi + ip[q] * wr
                            rp[q] = rp[p] - xr
                            ip[q] = ip[p] - xi
                            rp[p] += xr
                            ip[p] += xi
                            k += step
                        }
                        start += len
                    }
                }
            }
            len <<= 1
        }
        if inverse {
            let s = 1.0 / Double(n)
            for i in 0..<n {
                re[i] *= s
                im[i] *= s
            }
        }
    }

    /// Power spectrum |X_k|^2 for k = 0...n/2 of a real frame (zero-padded to n).
    public static func powerSpectrum(_ x: [Double], n: Int) -> [Double] {
        var re = [Double](repeating: 0, count: n)
        var im = [Double](repeating: 0, count: n)
        for i in 0..<min(n, x.count) { re[i] = x[i] }
        transform(&re, &im)
        var p = [Double](repeating: 0, count: n / 2 + 1)
        for k in 0...(n / 2) { p[k] = re[k] * re[k] + im[k] * im[k] }
        return p
    }

    /// Zero-phase spectral filter: X(f) * gain(f_hz), returning the real
    /// signal (analytic == false) or the analytic signal of the filtered
    /// signal (analytic == true: negative frequencies zeroed, positive x2).
    /// The input is zero-padded to >= 2x its length to avoid wrap-around.
    public static func spectralFilter(
        _ x: [Double], sampleRate: Double, analytic: Bool,
        gain: (Double) -> Double
    ) -> (re: [Double], im: [Double]) {
        let n = nextPow2(max(2, 2 * x.count))
        var re = [Double](repeating: 0, count: n)
        var im = [Double](repeating: 0, count: n)
        for i in 0..<x.count { re[i] = x[i] }
        transform(&re, &im)
        let df = sampleRate / Double(n)
        for k in 0..<n {
            let kk = k <= n / 2 ? k : k - n
            let f = abs(Double(kk)) * df
            var g = gain(f)
            if analytic {
                if kk < 0 { g = 0 } else if kk > 0 && k != n / 2 { g *= 2 }
            }
            re[k] *= g
            im[k] *= g
        }
        transform(&re, &im, inverse: true)
        return (Array(re[0..<x.count]), Array(im[0..<x.count]))
    }

    /// Zero-phase filter of a complex signal with a gain symmetric in |f|.
    public static func complexFilter(
        re x: [Double], im y: [Double], sampleRate: Double, gain: (Double) -> Double
    ) -> (re: [Double], im: [Double]) {
        let n = nextPow2(max(2, 2 * x.count))
        var re = [Double](repeating: 0, count: n)
        var im = [Double](repeating: 0, count: n)
        for i in 0..<x.count {
            re[i] = x[i]
            im[i] = y[i]
        }
        transform(&re, &im)
        let df = sampleRate / Double(n)
        for k in 0..<n {
            let kk = k <= n / 2 ? k : k - n
            let g = gain(abs(Double(kk)) * df)
            re[k] *= g
            im[k] *= g
        }
        transform(&re, &im, inverse: true)
        return (Array(re[0..<x.count]), Array(im[0..<x.count]))
    }

    /// Raised-cosine band gain: 1 inside [lo, hi], cosine tapers of width
    /// `taper` (fraction of the edge frequency) outside, 0 beyond.
    public static func bandGain(lo: Double, hi: Double, taper: Double = 0.25) -> (Double) -> Double {
        return { f in
            var g = 1.0
            if lo > 0 {
                let a = lo * (1 - taper)
                if f <= a { return 0 }
                if f < lo { g *= 0.5 - 0.5 * cos(Double.pi * (f - a) / (lo - a)) }
            }
            if hi > 0 {
                let b = hi * (1 + taper)
                if f >= b { return 0 }
                if f > hi { g *= 0.5 + 0.5 * cos(Double.pi * (f - hi) / (b - hi)) }
            }
            return g
        }
    }
}

@inline(__always) func hann(_ n: Int) -> [Double] {
    (0..<n).map { 0.5 * (1 - cos(2 * Double.pi * Double($0) / Double(n))) }
}

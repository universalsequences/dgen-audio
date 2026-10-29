// Matrix pencil estimation of damped sinusoids in a real signal.
//   x[n] ≈ Σ_k A_k r_k^n sin(ω_k n + φ_k),  n = 0..N-1
// Hankel matrix Y (N-L) x (L+1); dominant right-singular subspace V
// (eigenvectors of YᵀY); poles = eig(pinv(V1) V2). Amplitudes and phases by
// a joint real least-squares fit over all retained poles.

import Foundation

public struct DampedSinusoid: Codable {
    public var frequency: Double  // Hz
    public var decay: Double      // time constant, s
    public var amplitude: Double  // at n = 0 of the analysed segment
    public var phase: Double      // sin convention at n = 0
}

public enum MatrixPencil {
    public struct Options {
        public var maxPencil: Int = 120
        /// Keep singular values within this range of the largest.
        public var dynamicRangeDB: Double = 60
        /// ... and above this multiple of the median singular value.
        public var noiseFactor: Double = 3
        public var maxComponents: Int = 48
        public init() {}
    }

    public struct Result {
        public var modes: [DampedSinusoid]
        public var singularValues: [Double]
        public var order: Int
    }

    public static func estimate(_ x: [Double], sampleRate fs: Double, options: Options = Options()) -> Result {
        let N = x.count
        guard N >= 16 else { return Result(modes: [], singularValues: [], order: 0) }
        let L = max(4, min(options.maxPencil, N / 3))
        let rows = N - L
        let cols = L + 1
        // C = YᵀY with Y[i][j] = x[i + j]
        var C = [[Double]](repeating: [Double](repeating: 0, count: cols), count: cols)
        x.withUnsafeBufferPointer { xp in
            for a in 0..<cols {
                for b in a..<cols {
                    var s = 0.0
                    for i in 0..<rows { s += xp[i + a] * xp[i + b] }
                    C[a][b] = s
                    C[b][a] = s
                }
            }
        }
        let (vals, vecs) = LinAlg.symmetricEigen(C)
        let sv = vals.map { sqrt(max($0, 0)) }
        let smax = sv.first ?? 0
        guard smax > 0 else { return Result(modes: [], singularValues: sv, order: 0) }
        let sorted = sv.sorted()
        let median = sorted[sorted.count / 2]
        let thrDR = smax * pow(10, -options.dynamicRangeDB / 20)
        var M = 0
        for s in sv where s > thrDR && s > options.noiseFactor * median { M += 1 }
        M = min(max(M, 1), options.maxComponents, L - 1)
        // V1, V2 (L x M)
        var V1 = [[Double]](repeating: [Double](repeating: 0, count: M), count: L)
        var V2 = V1
        for i in 0..<L {
            for j in 0..<M {
                V1[i][j] = vecs[i][j]
                V2[i][j] = vecs[i + 1][j]
            }
        }
        // Φ = pinv(V1) V2, column by column
        var Phi = [[Double]](repeating: [Double](repeating: 0, count: M), count: M)
        for j in 0..<M {
            let col = LinAlg.leastSquares(V1, (0..<L).map { V2[$0][j] })
            for i in 0..<M { Phi[i][j] = col[i] }
        }
        let poles = LinAlg.eigenvalues(Phi)
        // keep upper half-plane poles; real poles go into the LS as columns
        var osc: [(r: Double, w: Double)] = []
        var real: [Double] = []
        for p in poles {
            let r = (p.re * p.re + p.im * p.im).squareRoot()
            let w = atan2(p.im, p.re)
            if !r.isFinite || r <= 0 { continue }
            if abs(p.im) < 1e-9 {
                if r < 1.2 { real.append(p.re) }
            } else if w > 0 {
                osc.append((min(r, 1.05), w))
            }
        }
        let amps = fitAmplitudes(x, osc: osc, real: real)
        var modes: [DampedSinusoid] = []
        for (k, o) in osc.enumerated() {
            let (a, b) = amps[k]  // x ≈ a cos + b sin
            let A = (a * a + b * b).squareRoot()
            let phi = atan2(a, b)
            let decay = o.r < 1 ? -1 / (fs * log(o.r)) : Double.infinity
            modes.append(DampedSinusoid(frequency: o.w * fs / (2 * Double.pi), decay: decay, amplitude: A, phase: phi))
        }
        return Result(modes: modes, singularValues: sv, order: M)
    }

    /// LS amplitudes (a_k, b_k) of r^n cos(ωn), r^n sin(ωn) (+ real-pole columns).
    public static func fitAmplitudes(_ x: [Double], osc: [(r: Double, w: Double)], real: [Double] = [])
        -> [(Double, Double)]
    {
        let N = x.count
        let nc = 2 * osc.count + real.count
        if nc == 0 { return [] }
        var A = [[Double]](repeating: [Double](repeating: 0, count: nc), count: N)
        for (k, o) in osc.enumerated() {
            let lr = log(o.r)
            for n in 0..<N {
                let e = exp(lr * Double(n))
                A[n][2 * k] = e * cos(o.w * Double(n))
                A[n][2 * k + 1] = e * sin(o.w * Double(n))
            }
        }
        for (j, p) in real.enumerated() {
            var v = 1.0
            for n in 0..<N {
                A[n][2 * osc.count + j] = v
                v *= p
                if abs(v) > 1e6 { v = 1e6 * (v > 0 ? 1 : -1) }
            }
        }
        let c = LinAlg.leastSquares(A, x, ridge: 1e-10 * x.reduce(0) { $0 + $1 * $1 })
        return (0..<osc.count).map { (c[2 * $0], c[2 * $0 + 1]) }
    }

    /// Stabilisation: modes of the full-segment estimate that reappear (|ΔF|
    /// < relF·F + 0.2 Hz, |ΔD|/D < relD) in both a shortened segment (75%)
    /// and a smaller pencil (70%). Noise poles move between runs; physical
    /// modes do not.
    public static func stableModes(_ x: [Double], sampleRate fs: Double, options: Options = Options(),
                                   relF: Double = 0.006, relD: Double = 0.35) -> [DampedSinusoid]
    {
        let full = estimate(x, sampleRate: fs, options: options).modes
        let short = estimate(Array(x[0..<(x.count * 3 / 4)]), sampleRate: fs, options: options).modes
        var o2 = options
        o2.maxPencil = max(8, Int(Double(min(options.maxPencil, x.count / 3)) * 0.7))
        let small = estimate(x, sampleRate: fs, options: o2).modes
        func present(_ m: DampedSinusoid, in set: [DampedSinusoid]) -> Bool {
            set.contains { abs($0.frequency - m.frequency) < relF * m.frequency + 0.2
                && abs($0.decay - m.decay) < relD * m.decay }
        }
        return full.filter { present($0, in: short) && present($0, in: small) }
    }

    /// Variable-projection refinement of (F, D) of all modes jointly on the
    /// segment (amplitudes by linear LS at every step). Modes that leave
    /// their ±2% frequency window or end up with D outside [Dmin, 5 s] keep
    /// their matrix-pencil values.
    public static func refine(_ x: [Double], sampleRate fs: Double, modes: [DampedSinusoid], minDecay: Double)
        -> [DampedSinusoid]
    {
        guard !modes.isEmpty else { return modes }
        let k = modes.count
        var p0 = [Double](), lo = [Double](), hi = [Double]()
        for m in modes {
            p0 += [m.frequency, log(m.decay)]
            lo += [m.frequency * 0.98, log(minDecay)]
            hi += [m.frequency * 1.02, log(5.0)]
        }
        func poles(_ p: [Double]) -> [(r: Double, w: Double)] {
            var out: [(r: Double, w: Double)] = []
            for j in 0..<k {
                let d = exp(p[2 * j + 1])
                out.append((r: exp(-1 / (fs * d)), w: 2 * Double.pi * p[2 * j] / fs))
            }
            return out
        }
        func residual(_ p: [Double]) -> [Double] {
            let ps = poles(p)
            let amps = fitAmplitudes(x, osc: ps)
            var r = x
            for (j, o) in ps.enumerated() {
                let (a, b) = amps[j]
                let lr = log(o.r)
                for n in 0..<x.count {
                    let e = exp(lr * Double(n))
                    r[n] -= e * (a * cos(o.w * Double(n)) + b * sin(o.w * Double(n)))
                }
            }
            return r
        }
        let fit = LinAlg.levenbergMarquardt(x0: p0, lower: lo, upper: hi, maxIter: 30, residuals: residual)
        let ps = poles(fit.x)
        let amps = fitAmplitudes(x, osc: ps)
        return (0..<k).map { j in
            let (a, b) = amps[j]
            return DampedSinusoid(frequency: fit.x[2 * j], decay: exp(fit.x[2 * j + 1]),
                                  amplitude: (a * a + b * b).squareRoot(), phase: atan2(a, b))
        }
    }
}

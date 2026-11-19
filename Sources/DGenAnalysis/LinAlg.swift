// Small dense linear algebra for the analysis: symmetric Jacobi eigen,
// Householder least squares, NNLS, eigenvalues of a real general matrix
// (Hessenberg reduction + Francis double-shift QR), and a bounded
// Levenberg-Marquardt. Row-major [[Double]] throughout; sizes are small
// (<= a few hundred), so clarity wins over blocking.

import DGenAnalysisC
import Foundation

/// A per-thread instance of T under `key` (analysis caches: each kick runs on one worker thread).
func threadLocal<T: AnyObject>(_ key: String, _ make: () -> T) -> T {
    let td = Thread.current.threadDictionary
    if let v = td[key] as? T { return v }
    let v = make()
    td[key] = v
    return v
}

public enum LinAlg {
    // MARK: symmetric eigen (cyclic Jacobi)

    /// Eigen-decomposition of a symmetric matrix. Returns eigenvalues sorted
    /// descending and the matching eigenvectors as columns of `vectors`.
    public static func symmetricEigen(_ a0: [[Double]], maxSweeps: Int = 60)
        -> (values: [Double], vectors: [[Double]])
    {
        symmetricEigen(flat: a0.flatMap { $0 }, n: a0.count, maxSweeps: maxSweeps)
    }

    /// symmetricEigen on a row-major n×n matrix: LAPACK dsyevd, or the Jacobi sweeps where LAPACK is
    /// unavailable (non-Apple builds) or fails.
    static func symmetricEigen(flat: [Double], n: Int, maxSweeps: Int = 60) -> (values: [Double], vectors: [[Double]]) {
        if n > 0 {
            var vals = [Double](repeating: 0, count: n), vecs = [Double](repeating: 0, count: n * n)
            let info = flat.withUnsafeBufferPointer { F in vals.withUnsafeMutableBufferPointer { V in
                vecs.withUnsafeMutableBufferPointer { E in dgen_sym_eigen(F.baseAddress, Int32(n), V.baseAddress, E.baseAddress) } } }
            if info == 0 { return (vals, (0..<n).map { Array(vecs[($0 * n)..<(($0 + 1) * n)]) }) }
        }
        return jacobiEigen((0..<n).map { Array(flat[($0 * n)..<(($0 + 1) * n)]) }, maxSweeps: maxSweeps)
    }

    /// Cyclic Jacobi sweeps (the portable fallback of symmetricEigen).
    static func jacobiEigen(_ a0: [[Double]], maxSweeps: Int = 60)
        -> (values: [Double], vectors: [[Double]])
    {
        let n = a0.count
        var a = a0
        var v = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n { v[i][i] = 1 }
        for _ in 0..<maxSweeps {
            var off = 0.0
            var diag = 0.0
            for i in 0..<n {
                diag += a[i][i] * a[i][i]
                for j in (i + 1)..<max(i + 1, n) { off += a[i][j] * a[i][j] }
            }
            if off <= 1e-26 * max(diag, 1e-300) { break }
            for p in 0..<n {
                for q in (p + 1)..<max(p + 1, n) {
                    let apq = a[p][q]
                    if abs(apq) < 1e-300 { continue }
                    let theta = (a[q][q] - a[p][p]) / (2 * apq)
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1))
                    let c = 1 / sqrt(t * t + 1)
                    let s = t * c
                    for k in 0..<n {
                        let akp = a[k][p], akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for k in 0..<n {
                        let apk = a[p][k], aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for k in 0..<n {
                        let vkp = v[k][p], vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }
        let order = (0..<n).sorted { a[$0][$0] > a[$1][$1] }
        let vals = order.map { a[$0][$0] }
        var vecs = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for (newj, oldj) in order.enumerated() { for i in 0..<n { vecs[i][newj] = v[i][oldj] } }
        return (vals, vecs)
    }

    // MARK: least squares

    /// min ||A x - b||^2 + ridge ||x||^2 via Householder QR. A is m x n.
    /// Least squares on a row-major m×n matrix (LAPACK QR; see dgen_lstsq).
    static func leastSquares(flat A: [Double], rows m: Int, cols n: Int, _ b: [Double], ridge: Double = 0) -> [Double] {
        var x = [Double](repeating: 0, count: n)
        if n == 0 { return x }
        let info = A.withUnsafeBufferPointer { F in b.withUnsafeBufferPointer { B in x.withUnsafeMutableBufferPointer { X in
            dgen_lstsq(F.baseAddress, Int32(m), Int32(n), B.baseAddress, ridge, X.baseAddress)
        }}}
        assert(info == 0, "dgen_lstsq failed (\(info))")
        return x
    }

    public static func leastSquares(_ A: [[Double]], _ b: [Double], ridge: Double = 0) -> [Double] {
        let n = A.first?.count ?? 0
        if n == 0 { return [] }
        return leastSquares(flat: A.flatMap { $0 }, rows: A.count, cols: n, b, ridge: ridge)
    }

    /// Householder QR least squares on a column-major m×n matrix `a` (destroyed) and `y` (destroyed),
    /// in Swift: LM's step solve on the reference paths (DGEN_NOISEFIT=swift / fd keep the original
    /// operation order, so those runs reproduce the pre-C analysis bit for bit).
    public static func leastSquaresColumnMajor(_ a: inout [Double], _ y: inout [Double], m: Int, n: Int) -> [Double] {
        var scale = [Double](repeating: 1, count: n)
        var x = [Double](repeating: 0, count: n)
        var v = [Double](repeating: 0, count: m)
        a.withUnsafeMutableBufferPointer { A in
        y.withUnsafeMutableBufferPointer { Y in
        v.withUnsafeMutableBufferPointer { V in
            // column scaling for conditioning
            for j in 0..<n {
                let c = A.baseAddress! + j * m
                var s = 0.0
                for i in 0..<m { s += c[i] * c[i] }
                scale[j] = s > 0 ? 1 / sqrt(s) : 1
                let sc = scale[j]
                for i in 0..<m { c[i] *= sc }
            }
            let kmax = min(m, n)
            for k in 0..<kmax {
                let ck = A.baseAddress! + k * m
                var norm = 0.0
                for i in k..<m { norm += ck[i] * ck[i] }
                norm = sqrt(norm)
                if norm == 0 { continue }
                let alpha = ck[k] > 0 ? -norm : norm
                for i in k..<m { V[i - k] = ck[i] }
                V[0] -= alpha
                var vn = 0.0
                for i in 0..<(m - k) { vn += V[i] * V[i] }
                if vn == 0 { continue }
                for j in k..<n {
                    let cj = A.baseAddress! + j * m
                    var d = 0.0
                    for i in k..<m { d += V[i - k] * cj[i] }
                    let f = 2 * d / vn
                    for i in k..<m { cj[i] -= f * V[i - k] }
                }
                var d = 0.0
                for i in k..<m { d += V[i - k] * Y[i] }
                let f = 2 * d / vn
                for i in k..<m { Y[i] -= f * V[i - k] }
            }
            let rmax = abs(A[0])
            for k in stride(from: kmax - 1, through: 0, by: -1) {
                var s = Y[k]
                for j in (k + 1)..<max(k + 1, n) { s -= A[j * m + k] * x[j] }
                x[k] = abs(A[k * m + k]) > 1e-13 * rmax ? s / A[k * m + k] : 0
            }
        }}}
        for j in 0..<n { x[j] *= scale[j] }
        return x
    }

    // MARK: NNLS (Lawson-Hanson)

    public static func nnls(_ A: [[Double]], _ b: [Double], maxIter: Int = 300) -> [Double] {
        let m = A.count
        let n = A.first?.count ?? 0
        var x = [Double](repeating: 0, count: n)
        var passive = [Bool](repeating: false, count: n)
        func gradient() -> [Double] {
            var r = b
            for i in 0..<m { for j in 0..<n where x[j] != 0 { r[i] -= A[i][j] * x[j] } }
            var w = [Double](repeating: 0, count: n)
            for j in 0..<n { for i in 0..<m { w[j] += A[i][j] * r[i] } }
            return w
        }
        func solvePassive() -> [Double] {
            let idx = (0..<n).filter { passive[$0] }
            let sub = A.map { row in idx.map { row[$0] } }
            let z = leastSquares(sub, b)
            var full = [Double](repeating: 0, count: n)
            for (k, j) in idx.enumerated() { full[j] = z[k] }
            return full
        }
        var iter = 0
        while iter < maxIter {
            iter += 1
            let w = gradient()
            var best = -1
            var bestW = 1e-12 * (b.map { abs($0) }.max() ?? 1)
            for j in 0..<n where !passive[j] && w[j] > bestW {
                bestW = w[j]
                best = j
            }
            if best < 0 { break }
            passive[best] = true
            while true {
                let z = solvePassive()
                var ok = true
                for j in 0..<n where passive[j] && z[j] <= 0 { ok = false }
                if ok {
                    x = z
                    break
                }
                var alpha = Double.infinity
                for j in 0..<n where passive[j] && z[j] <= 0 {
                    let d = x[j] - z[j]
                    if d > 0 { alpha = min(alpha, x[j] / d) }
                }
                if !alpha.isFinite { alpha = 0 }
                for j in 0..<n { x[j] += alpha * (z[j] - x[j]) }
                for j in 0..<n where passive[j] && x[j] <= 1e-15 {
                    passive[j] = false
                    x[j] = 0
                }
                if !passive.contains(true) { break }
            }
        }
        return x
    }

    // MARK: general real eigenvalues

    /// Eigenvalues (re, im) of a real square matrix.
    public static func eigenvalues(_ m: [[Double]]) -> [(re: Double, im: Double)] {
        let n = m.count
        if n == 0 { return [] }
        if n == 1 { return [(m[0][0], 0)] }
        // 1-indexed copy for a literal port of the classical algorithms.
        var a = [[Double]](repeating: [Double](repeating: 0, count: n + 1), count: n + 1)
        for i in 0..<n { for j in 0..<n { a[i + 1][j + 1] = m[i][j] } }
        balance(&a, n)
        elmhes(&a, n)
        for i in 1...n { for j in 1...n where i > j + 1 { a[i][j] = 0 } }
        return hqr(&a, n)
    }

    private static func balance(_ a: inout [[Double]], _ n: Int) {
        let radix = 2.0
        let sqrdx = radix * radix
        var last = false
        while !last {
            last = true
            for i in 1...n {
                var r = 0.0, c = 0.0
                for j in 1...n where j != i {
                    c += abs(a[j][i])
                    r += abs(a[i][j])
                }
                if c != 0 && r != 0 {
                    var g = r / radix
                    var f = 1.0
                    let s = c + r
                    while c < g {
                        f *= radix
                        c *= sqrdx
                    }
                    g = r * radix
                    while c > g {
                        f /= radix
                        c /= sqrdx
                    }
                    if (c + r) / f < 0.95 * s {
                        last = false
                        g = 1 / f
                        for j in 1...n { a[i][j] *= g }
                        for j in 1...n { a[j][i] *= f }
                    }
                }
            }
        }
    }

    private static func elmhes(_ a: inout [[Double]], _ n: Int) {
        if n < 3 { return }
        for m in 2..<n {
            var x = 0.0
            var i = m
            for j in m...n where abs(a[j][m - 1]) > abs(x) {
                x = a[j][m - 1]
                i = j
            }
            if i != m {
                for j in (m - 1)...n { let t = a[i][j]; a[i][j] = a[m][j]; a[m][j] = t }
                for j in 1...n { let t = a[j][i]; a[j][i] = a[j][m]; a[j][m] = t }
            }
            if x != 0 {
                if m + 1 <= n {
                    for i2 in (m + 1)...n {
                        var y = a[i2][m - 1]
                        if y != 0 {
                            y /= x
                            a[i2][m - 1] = y
                            for j in m...n { a[i2][j] -= y * a[m][j] }
                            for j in 1...n { a[j][m] += y * a[j][i2] }
                        }
                    }
                }
            }
        }
    }

    private static func sign(_ a: Double, _ b: Double) -> Double { b >= 0 ? abs(a) : -abs(a) }

    private static func hqr(_ a: inout [[Double]], _ n: Int) -> [(re: Double, im: Double)] {
        var wr = [Double](repeating: 0, count: n + 1)
        var wi = [Double](repeating: 0, count: n + 1)
        var anorm = 0.0
        for i in 1...n { for j in max(i - 1, 1)...n { anorm += abs(a[i][j]) } }
        var nn = n
        var t = 0.0
        var p = 0.0, q = 0.0, r = 0.0, s = 0.0, w = 0.0, x = 0.0, y = 0.0, z = 0.0
        while nn >= 1 {
            var its = 0
            var l = 0
            repeat {
                l = nn
                while l >= 2 {
                    s = abs(a[l - 1][l - 1]) + abs(a[l][l])
                    if s == 0 { s = anorm }
                    if abs(a[l][l - 1]) + s == s {
                        a[l][l - 1] = 0
                        break
                    }
                    l -= 1
                }
                x = a[nn][nn]
                if l == nn {
                    wr[nn] = x + t
                    wi[nn] = 0
                    nn -= 1
                } else {
                    y = a[nn - 1][nn - 1]
                    w = a[nn][nn - 1] * a[nn - 1][nn]
                    if l == nn - 1 {
                        p = 0.5 * (y - x)
                        q = p * p + w
                        z = sqrt(abs(q))
                        x += t
                        if q >= 0 {
                            z = p + sign(z, p)
                            wr[nn - 1] = x + z
                            wr[nn] = x + z
                            if z != 0 { wr[nn] = x - w / z }
                            wi[nn - 1] = 0
                            wi[nn] = 0
                        } else {
                            wr[nn - 1] = x + p
                            wr[nn] = x + p
                            wi[nn - 1] = -z
                            wi[nn] = z
                        }
                        nn -= 2
                    } else {
                        if its == 60 {
                            // give up on this block: report the diagonal
                            for i in 1...nn {
                                wr[i] = a[i][i] + t
                                wi[i] = 0
                            }
                            nn = 0
                            break
                        }
                        if its == 10 || its == 20 || its == 40 {
                            t += x
                            for i in 1...nn { a[i][i] -= x }
                            s = abs(a[nn][nn - 1]) + abs(a[nn - 1][nn - 2])
                            x = 0.75 * s
                            y = x
                            w = -0.4375 * s * s
                        }
                        its += 1
                        var m = nn - 2
                        while m >= l {
                            z = a[m][m]
                            r = x - z
                            s = y - z
                            p = (r * s - w) / a[m + 1][m] + a[m][m + 1]
                            q = a[m + 1][m + 1] - z - r - s
                            r = a[m + 2][m + 1]
                            s = abs(p) + abs(q) + abs(r)
                            p /= s
                            q /= s
                            r /= s
                            if m == l { break }
                            let u = abs(a[m][m - 1]) * (abs(q) + abs(r))
                            let v = abs(p) * (abs(a[m - 1][m - 1]) + abs(z) + abs(a[m + 1][m + 1]))
                            if u + v == v { break }
                            m -= 1
                        }
                        if m + 2 <= nn {
                            for i in (m + 2)...nn {
                                a[i][i - 2] = 0
                                if i != m + 2 { a[i][i - 3] = 0 }
                            }
                        }
                        var k = m
                        while k <= nn - 1 {
                            if k != m {
                                p = a[k][k - 1]
                                q = a[k + 1][k - 1]
                                r = 0
                                if k != nn - 1 { r = a[k + 2][k - 1] }
                                x = abs(p) + abs(q) + abs(r)
                                if x != 0 {
                                    p /= x
                                    q /= x
                                    r /= x
                                }
                            }
                            s = sign(sqrt(p * p + q * q + r * r), p)
                            if s != 0 {
                                if k == m {
                                    if l != m { a[k][k - 1] = -a[k][k - 1] }
                                } else {
                                    a[k][k - 1] = -s * x
                                }
                                p += s
                                x = p / s
                                y = q / s
                                z = r / s
                                q /= p
                                r /= p
                                for j in k...nn {
                                    p = a[k][j] + q * a[k + 1][j]
                                    if k != nn - 1 {
                                        p += r * a[k + 2][j]
                                        a[k + 2][j] -= p * z
                                    }
                                    a[k + 1][j] -= p * y
                                    a[k][j] -= p * x
                                }
                                let mmin = nn < k + 3 ? nn : k + 3
                                for i in l...mmin {
                                    p = x * a[i][k] + y * a[i][k + 1]
                                    if k != nn - 1 {
                                        p += z * a[i][k + 2]
                                        a[i][k + 2] -= p * r
                                    }
                                    a[i][k + 1] -= p * q
                                    a[i][k] -= p
                                }
                            }
                            k += 1
                        }
                    }
                }
            } while nn >= 1 && l < nn - 1
        }
        return (1...n).map { (wr[$0], wi[$0]) }
    }

    // MARK: Levenberg-Marquardt (bounded by projection)

    public struct LMResult {
        public var x: [Double]
        public var cost: Double
        public var iterations: Int
    }

    /// Minimise 0.5 * ||r(x)||^2 with box bounds. Numerical (forward
    /// difference) Jacobian; the step is projected onto the box.
    public static func levenbergMarquardt(
        x0: [Double], lower: [Double], upper: [Double], maxIter: Int = 100,
        relStep: Double = 1e-6, residuals: ([Double]) -> [Double],
        batchResiduals: ((_ flat: [Double], _ count: Int) -> [Double])? = nil,
        jacobian: ((_ x: [Double]) -> (r: [Double], J: [Double]))? = nil
    ) -> LMResult {
        let n = x0.count
        func clampv(_ v: [Double]) -> [Double] {
            (0..<n).map { min(max(v[$0], lower[$0]), upper[$0]) }
        }
        var x = clampv(x0)
        var r = residuals(x)
        var cost = 0.5 * r.reduce(0) { $0 + $1 * $1 }
        var lambda = 1e-3
        var it = 0
        while it < maxIter {
            it += 1
            let m = r.count
            var JtJ = [Double](repeating: 0, count: n * n)
            var Jtr = [Double](repeating: 0, count: n)
            if let jacobian {
                // analytic Jacobian: normal equations by BLAS
                let (_, J) = jacobian(x)
                J.withUnsafeBufferPointer { Jb in r.withUnsafeBufferPointer { R in
                    JtJ.withUnsafeMutableBufferPointer { G in Jtr.withUnsafeMutableBufferPointer { g in
                        dgen_normal_equations(Jb.baseAddress, R.baseAddress, Int32(m), Int32(n), G.baseAddress, g.baseAddress)
                    }}
                }}
            } else {
            // Jacobian (row-major flat m×n), by forward differences; with batchResiduals the n
            // perturbed points are evaluated in one call
            var J = [Double](repeating: 0, count: m * n)
            var hs = [Double](repeating: 0, count: n)
            for j in 0..<n {
                var h = relStep * max(abs(x[j]), 1e-3 * max(abs(upper[j] - lower[j]), 1e-12))
                if x[j] + h > upper[j] { h = -h }
                hs[j] = h
            }
            if let batchResiduals {
                var flat = [Double](repeating: 0, count: n * n)
                for j in 0..<n {
                    for k in 0..<n { flat[j * n + k] = x[k] }
                    flat[j * n + j] = x[j] + hs[j]
                }
                let rp = batchResiduals(flat, n)
                J.withUnsafeMutableBufferPointer { Jb in
                    rp.withUnsafeBufferPointer { R in r.withUnsafeBufferPointer { R0 in
                        for j in 0..<n { for i in 0..<m { Jb[i * n + j] = (R[j * m + i] - R0[i]) / hs[j] } }
                    }}
                }
            } else {
                var xp = x
                for j in 0..<n {
                    xp[j] = x[j] + hs[j]
                    let rp = residuals(xp)
                    xp[j] = x[j]
                    J.withUnsafeMutableBufferPointer { Jb in
                        rp.withUnsafeBufferPointer { R in r.withUnsafeBufferPointer { R0 in
                            for i in 0..<m { Jb[i * n + j] = (R[i] - R0[i]) / hs[j] }
                        }}
                    }
                }
            }
            J.withUnsafeBufferPointer { Jb in JtJ.withUnsafeMutableBufferPointer { G in Jtr.withUnsafeMutableBufferPointer { g in
                r.withUnsafeBufferPointer { R in
                    for i in 0..<m {
                        let row = Jb.baseAddress! + i * n
                        for a in 0..<n where row[a] != 0 {
                            g[a] += row[a] * R[i]
                            for b in a..<n { G[a * n + b] += row[a] * row[b] }
                        }
                    }
                }
                for a in 0..<n { for b in 0..<a { G[a * n + b] = G[b * n + a] } }
            }}}
            }
            var improved = false
            for _ in 0..<12 {
                // column-major copy of JtJ + λ·diag (JtJ is symmetric, so row- and column-major agree)
                var Aug = JtJ
                for a in 0..<n { Aug[a * n + a] += lambda * max(JtJ[a * n + a], 1e-12) }
                var rhs = Jtr.map { -$0 }
                var delta: [Double]
                if jacobian != nil {
                    // Cholesky (the damped normal matrix is SPD); QR if it is not
                    var b = rhs
                    let ok = Aug.withUnsafeBufferPointer { A in b.withUnsafeMutableBufferPointer { B in
                        dgen_cholesky_solve(A.baseAddress, Int32(n), B.baseAddress) == 0 } }
                    delta = ok ? b : leastSquaresColumnMajor(&Aug, &rhs, m: n, n: n)
                } else {
                    delta = leastSquaresColumnMajor(&Aug, &rhs, m: n, n: n)
                }
                let xn = clampv((0..<n).map { x[$0] + delta[$0] })
                let rn = residuals(xn)
                let cn = 0.5 * rn.reduce(0) { $0 + $1 * $1 }
                if cn.isFinite && cn < cost {
                    let rel = (cost - cn) / max(cost, 1e-300)
                    x = xn
                    r = rn
                    cost = cn
                    lambda = max(lambda * 0.3, 1e-9)
                    improved = true
                    if rel < 1e-10 { return LMResult(x: x, cost: cost, iterations: it) }
                    break
                }
                lambda *= 10
            }
            if !improved { break }
        }
        return LMResult(x: x, cost: cost, iterations: it)
    }
}

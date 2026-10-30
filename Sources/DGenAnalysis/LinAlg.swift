// Small dense linear algebra for the analysis: symmetric Jacobi eigen,
// Householder least squares, NNLS, eigenvalues of a real general matrix
// (Hessenberg reduction + Francis double-shift QR), and a bounded
// Levenberg-Marquardt. Row-major [[Double]] throughout; sizes are small
// (<= a few hundred), so clarity wins over blocking.

import Foundation

public enum LinAlg {
    // MARK: symmetric eigen (cyclic Jacobi)

    /// Eigen-decomposition of a symmetric matrix. Returns eigenvalues sorted
    /// descending and the matching eigenvectors as columns of `vectors`.
    public static func symmetricEigen(_ a0: [[Double]], maxSweeps: Int = 60)
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
    public static func leastSquares(_ A: [[Double]], _ b: [Double], ridge: Double = 0) -> [Double] {
        var a = A
        var y = b
        let n = A.first?.count ?? 0
        if n == 0 { return [] }
        if ridge > 0 {
            let r = sqrt(ridge)
            for j in 0..<n {
                var row = [Double](repeating: 0, count: n)
                row[j] = r
                a.append(row)
                y.append(0)
            }
        }
        let m = a.count
        // column scaling for conditioning
        var scale = [Double](repeating: 1, count: n)
        for j in 0..<n {
            var s = 0.0
            for i in 0..<m { s += a[i][j] * a[i][j] }
            scale[j] = s > 0 ? 1 / sqrt(s) : 1
            for i in 0..<m { a[i][j] *= scale[j] }
        }
        let kmax = min(m, n)
        for k in 0..<kmax {
            var norm = 0.0
            for i in k..<m { norm += a[i][k] * a[i][k] }
            norm = sqrt(norm)
            if norm == 0 { continue }
            let alpha = a[k][k] > 0 ? -norm : norm
            var v = [Double](repeating: 0, count: m - k)
            for i in k..<m { v[i - k] = a[i][k] }
            v[0] -= alpha
            var vn = 0.0
            for x in v { vn += x * x }
            if vn == 0 { continue }
            for j in k..<n {
                var d = 0.0
                for i in k..<m { d += v[i - k] * a[i][j] }
                let f = 2 * d / vn
                for i in k..<m { a[i][j] -= f * v[i - k] }
            }
            var d = 0.0
            for i in k..<m { d += v[i - k] * y[i] }
            let f = 2 * d / vn
            for i in k..<m { y[i] -= f * v[i - k] }
        }
        var x = [Double](repeating: 0, count: n)
        let rmax = abs(a[0][0])
        for k in stride(from: kmax - 1, through: 0, by: -1) {
            var s = y[k]
            for j in (k + 1)..<max(k + 1, n) { s -= a[k][j] * x[j] }
            x[k] = abs(a[k][k]) > 1e-13 * rmax ? s / a[k][k] : 0
        }
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
        relStep: Double = 1e-6, residuals: ([Double]) -> [Double]
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
            // Jacobian
            var J = [[Double]](repeating: [Double](repeating: 0, count: n), count: m)
            for j in 0..<n {
                var h = relStep * max(abs(x[j]), 1e-3 * max(abs(upper[j] - lower[j]), 1e-12))
                if x[j] + h > upper[j] { h = -h }
                var xp = x
                xp[j] += h
                let rp = residuals(xp)
                for i in 0..<m { J[i][j] = (rp[i] - r[i]) / h }
            }
            var JtJ = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
            var Jtr = [Double](repeating: 0, count: n)
            for i in 0..<m {
                let row = J[i]
                for a in 0..<n where row[a] != 0 {
                    Jtr[a] += row[a] * r[i]
                    for b in a..<n { JtJ[a][b] += row[a] * row[b] }
                }
            }
            for a in 0..<n { for b in 0..<a { JtJ[a][b] = JtJ[b][a] } }
            var improved = false
            for _ in 0..<12 {
                var Aug = JtJ
                for a in 0..<n { Aug[a][a] += lambda * max(JtJ[a][a], 1e-12) }
                let delta = leastSquares(Aug, Jtr.map { -$0 })
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

import Foundation

/// Canonical full-covariance CMA-ES state. All optimizer arithmetic and RNG
/// state use Float64; callers convert candidates to Float only for rendering.
struct CMAES: Codable {
    struct Candidate: Codable, Equatable {
        var index: Int
        var values: [Double]
        var reflectedCoordinates: Int
    }

    enum StopReason: String, Codable {
        case covarianceCollapse = "covariance_collapse"
        case numericalFailure = "numerical_failure"
    }

    var dimension: Int
    var population: Int
    var mean: [Double]
    var sigma: Double
    var covariance: [Double]       // row major
    var eigenvectors: [Double]     // columns are eigenvectors
    var eigenvalues: [Double]
    var pc: [Double]
    var ps: [Double]
    var generation: Int
    var rng: GaussianRNG
    private(set) var conditionNumber: Double
    private(set) var stopReason: StopReason?

    private var mu: Int
    private var weights: [Double]
    private var muEff: Double
    private var cc: Double
    private var cs: Double
    private var c1: Double
    private var cmu: Double
    private var damping: Double
    private var chiN: Double

    init(mean: [Double], sigma: Double, population: Int, seed: UInt64) {
        precondition(!mean.isEmpty && population >= 2 && sigma > 0)
        dimension = mean.count
        self.population = population
        self.mean = mean
        self.sigma = sigma
        covariance = CMAES.identity(mean.count)
        eigenvectors = CMAES.identity(mean.count)
        eigenvalues = [Double](repeating: 1, count: mean.count)
        pc = [Double](repeating: 0, count: mean.count)
        ps = pc
        generation = 0
        rng = GaussianRNG(seed: seed)
        conditionNumber = 1
        stopReason = nil

        mu = population / 2
        var raw = (0..<mu).map { log((Double(population) + 1) / 2) - log(Double($0 + 1)) }
        let sum = raw.reduce(0, +)
        raw = raw.map { $0 / sum }
        weights = raw
        muEff = 1 / raw.reduce(0) { $0 + $1 * $1 }
        let n = Double(mean.count)
        cc = (4 + muEff / n) / (n + 4 + 2 * muEff / n)
        cs = (muEff + 2) / (n + muEff + 5)
        c1 = 2 / (pow(n + 1.3, 2) + muEff)
        cmu = min(1 - c1, 2 * (muEff - 2 + 1 / muEff) / (pow(n + 2, 2) + muEff))
        damping = 1 + 2 * max(0, sqrt((muEff - 1) / (n + 1)) - 1) + cs
        chiN = sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    }

    /// Samples a generation and replaces trailing slots with anchors. Anchor
    /// order is stable; if there are more anchors than slots, the last ones win.
    mutating func ask(anchors: [[Double]] = []) -> [Candidate] {
        guard stopReason == nil else { return [] }
        var result: [Candidate] = []
        result.reserveCapacity(population)
        for index in 0..<population {
            let normal = (0..<dimension).map { _ in rng.normal() }
            var step = [Double](repeating: 0, count: dimension)
            for row in 0..<dimension {
                for col in 0..<dimension {
                    step[row] += eigenvectors[row * dimension + col]
                        * sqrt(max(eigenvalues[col], 1e-14)) * normal[col]
                }
            }
            let raw = zip(mean, step).map { $0 + sigma * $1 }
            let reflected = raw.map(Self.reflect)
            let count = zip(raw, reflected).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
            result.append(Candidate(index: index, values: reflected, reflectedCoordinates: count))
        }
        for (offset, anchor) in anchors.suffix(population).enumerated() {
            guard anchor.count == dimension, anchor.allSatisfy(\.isFinite) else { continue }
            let slot = population - min(anchors.count, population) + offset
            let reflected = anchor.map(Self.reflect)
            let count = zip(anchor, reflected).reduce(0) { $0 + ($1.0 == $1.1 ? 0 : 1) }
            result[slot] = Candidate(index: slot, values: reflected, reflectedCoordinates: count)
        }
        return result
    }

    /// Updates from candidates sorted by `(fitness, candidateIndex)` by the caller.
    mutating func tell(ranked: [Candidate]) {
        guard stopReason == nil, ranked.count >= mu else {
            stopReason = .numericalFailure
            return
        }
        let oldMean = mean
        var selectedSteps = [[Double]]()
        selectedSteps.reserveCapacity(mu)
        for i in 0..<mu {
            guard ranked[i].values.count == dimension,
                  ranked[i].values.allSatisfy(\.isFinite) else {
                stopReason = .numericalFailure
                return
            }
            selectedSteps.append(zip(ranked[i].values, oldMean).map { ($0 - $1) / sigma })
        }
        var weightedStep = [Double](repeating: 0, count: dimension)
        for i in 0..<mu {
            for d in 0..<dimension { weightedStep[d] += weights[i] * selectedSteps[i][d] }
        }
        mean = zip(oldMean, weightedStep).map { $0 + sigma * $1 }

        var inverseStep = [Double](repeating: 0, count: dimension)
        for col in 0..<dimension {
            var projection = 0.0
            for row in 0..<dimension {
                projection += eigenvectors[row * dimension + col] * weightedStep[row]
            }
            projection /= sqrt(max(eigenvalues[col], 1e-14))
            for row in 0..<dimension {
                inverseStep[row] += eigenvectors[row * dimension + col] * projection
            }
        }
        let psScale = sqrt(cs * (2 - cs) * muEff)
        for d in 0..<dimension { ps[d] = (1 - cs) * ps[d] + psScale * inverseStep[d] }
        let psNorm = norm(ps)
        let correction = sqrt(max(1e-30, 1 - pow(1 - cs, 2 * Double(generation + 1))))
        let hsig = psNorm / correction / chiN < 1.4 + 2 / (Double(dimension) + 1)
        let pcScale = (hsig ? 1.0 : 0.0) * sqrt(cc * (2 - cc) * muEff)
        for d in 0..<dimension { pc[d] = (1 - cc) * pc[d] + pcScale * weightedStep[d] }

        let oldC = covariance
        let base = 1 - c1 - cmu
        for row in 0..<dimension {
            for col in 0..<dimension {
                let idx = row * dimension + col
                var rankMu = 0.0
                for i in 0..<mu {
                    rankMu += weights[i] * selectedSteps[i][row] * selectedSteps[i][col]
                }
                covariance[idx] = base * oldC[idx] + c1 * pc[row] * pc[col]
                    + (hsig ? 0 : c1 * cc * (2 - cc) * oldC[idx]) + cmu * rankMu
            }
        }
        symmetrize()
        sigma *= exp((cs / damping) * (psNorm / chiN - 1))
        generation += 1
        guard sigma.isFinite, sigma > 1e-16, covariance.allSatisfy(\.isFinite) else {
            stopReason = .numericalFailure
            return
        }
        refreshEigendecomposition()
        if let maxAxis = eigenvalues.max(), sigma * sqrt(maxAxis) > 1_000 {
            sigma = 1_000 / sqrt(maxAxis)
        }
        if eigenvalues.allSatisfy({ sigma * sqrt($0) < 1e-12 }) {
            stopReason = .covarianceCollapse
        }
    }

    static func reflect(_ value: Double) -> Double {
        guard value.isFinite else { return .nan }
        var x = value.truncatingRemainder(dividingBy: 2)
        if x < 0 { x += 2 }
        return x <= 1 ? x : 2 - x
    }

    private mutating func symmetrize() {
        for row in 0..<dimension {
            for col in (row + 1)..<dimension {
                let value = 0.5 * (covariance[row * dimension + col] + covariance[col * dimension + row])
                covariance[row * dimension + col] = value
                covariance[col * dimension + row] = value
            }
        }
    }

    private mutating func refreshEigendecomposition() {
        let decomposition = Self.jacobi(covariance, size: dimension)
        eigenvalues = decomposition.values.map { max($0, 1e-14) }
        let largest = eigenvalues.max() ?? 1
        let floor = max(1e-14, largest / 1e14)
        eigenvalues = eigenvalues.map { max($0, floor) }
        eigenvectors = decomposition.vectors
        conditionNumber = (eigenvalues.max() ?? 1) / max(eigenvalues.min() ?? 1, 1e-14)
        // Reconstruct after flooring so serialized C is positive definite too.
        covariance = [Double](repeating: 0, count: dimension * dimension)
        for r in 0..<dimension {
            for c in 0..<dimension {
                for k in 0..<dimension {
                    covariance[r * dimension + c] += eigenvectors[r * dimension + k]
                        * eigenvalues[k] * eigenvectors[c * dimension + k]
                }
            }
        }
        symmetrize()
    }

    private static func identity(_ n: Int) -> [Double] {
        var result = [Double](repeating: 0, count: n * n)
        for i in 0..<n { result[i * n + i] = 1 }
        return result
    }

    /// Deterministic symmetric Jacobi eigensolver; adequate for CMA's 20-100D CPU update.
    private static func jacobi(_ input: [Double], size n: Int) -> (values: [Double], vectors: [Double]) {
        var a = input
        var v = identity(n)
        let iterations = max(32, 20 * n * n)
        for _ in 0..<iterations {
            var p = 0, q = min(1, n - 1), largest = 0.0
            if n > 1 {
                for i in 0..<n { for j in (i + 1)..<n where abs(a[i * n + j]) > largest {
                    largest = abs(a[i * n + j]); p = i; q = j
                }}
            }
            if largest < 1e-15 { break }
            let app = a[p * n + p], aqq = a[q * n + q], apq = a[p * n + q]
            let phi = 0.5 * atan2(2 * apq, aqq - app)
            let c = cos(phi), s = sin(phi)
            for k in 0..<n where k != p && k != q {
                let akp = a[k * n + p], akq = a[k * n + q]
                a[k * n + p] = c * akp - s * akq; a[p * n + k] = a[k * n + p]
                a[k * n + q] = s * akp + c * akq; a[q * n + k] = a[k * n + q]
            }
            a[p * n + p] = c*c*app - 2*s*c*apq + s*s*aqq
            a[q * n + q] = s*s*app + 2*s*c*apq + c*c*aqq
            a[p * n + q] = 0; a[q * n + p] = 0
            for k in 0..<n {
                let vkp = v[k * n + p], vkq = v[k * n + q]
                v[k * n + p] = c * vkp - s * vkq
                v[k * n + q] = s * vkp + c * vkq
            }
        }
        let order = (0..<n).sorted { a[$0 * n + $0] < a[$1 * n + $1] }
        var values = [Double](), vectors = [Double](repeating: 0, count: n * n)
        for (newColumn, oldColumn) in order.enumerated() {
            values.append(a[oldColumn * n + oldColumn])
            for row in 0..<n { vectors[row * n + newColumn] = v[row * n + oldColumn] }
        }
        return (values, vectors)
    }
}

struct GaussianRNG: Codable {
    var state: UInt64
    var spare: Double?

    init(seed: UInt64) { state = seed; spare = nil }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func uniform() -> Double {
        Double(next() >> 11) * (1 / 9_007_199_254_740_992.0)
    }

    mutating func normal() -> Double {
        if let value = spare { spare = nil; return value }
        let u1 = max(uniform(), Double.ulpOfOne), u2 = uniform()
        let radius = sqrt(-2 * log(u1)), angle = 2 * Double.pi * u2
        spare = radius * sin(angle)
        return radius * cos(angle)
    }
}

private func norm(_ values: [Double]) -> Double {
    sqrt(values.reduce(0) { $0 + $1 * $1 })
}

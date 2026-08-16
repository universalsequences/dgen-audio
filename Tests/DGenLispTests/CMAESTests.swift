import XCTest
@testable import DGenLisp

final class CMAESTests: XCTestCase {
    func testReflectionAlwaysReturnsUnitInterval() {
        let values = [-1000.25, -3.2, -0.2, 0, 0.3, 1, 1.3, 7.8, 1000.25]
        let reflected = values.map(CMAES.reflect)
        XCTAssertTrue(reflected.allSatisfy { $0 >= 0 && $0 <= 1 })
        XCTAssertEqual(CMAES.reflect(-0.2), 0.2, accuracy: 1e-15)
        XCTAssertEqual(CMAES.reflect(1.3), 0.7, accuracy: 1e-15)
    }

    func testSamplingIsDeterministicAndResumeIsIdentical() throws {
        var first = CMAES(mean: [0.3, 0.7, 0.5], sigma: 0.2, population: 8, seed: 17)
        var second = first
        XCTAssertEqual(first.ask(), second.ask())

        let ranked = first.ask().sorted { sphere($0.values) < sphere($1.values) }
        first.tell(ranked: ranked)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(first)
        var resumed = try JSONDecoder().decode(CMAES.self, from: bytes)
        XCTAssertEqual(first.ask(), resumed.ask())
        XCTAssertEqual(try encoder.encode(first), try encoder.encode(resumed))
    }

    func testSphereConvergenceAndPositiveDefiniteCovariance() {
        var cma = CMAES(mean: [0.85, 0.8, 0.75, 0.7], sigma: 0.25, population: 24, seed: 1)
        let target = [Double](repeating: 0.2, count: 4)
        for _ in 0..<100 {
            let candidates = cma.ask()
            cma.tell(ranked: candidates.sorted {
                squaredDistance($0.values, target) < squaredDistance($1.values, target)
            })
        }
        XCTAssertLessThan(squaredDistance(cma.mean, target), 1e-5)
        XCTAssertTrue(cma.eigenvalues.allSatisfy { $0 > 0 && $0.isFinite })
        for row in 0..<cma.dimension { for column in 0..<cma.dimension {
            XCTAssertEqual(
                cma.covariance[row * cma.dimension + column],
                cma.covariance[column * cma.dimension + row], accuracy: 1e-12)
        }}
    }

    func testRotatedEllipsoidConvergence() {
        var cma = CMAES(mean: [0.8, 0.15, 0.75, 0.2], sigma: 0.25, population: 32, seed: 9)
        let target = [0.25, 0.35, 0.45, 0.55]
        // Two planar rotations make the narrow axes non-coordinate-aligned.
        func objective(_ x: [Double]) -> Double {
            let d = zip(x, target).map(-)
            let a = 0.8 * d[0] - 0.6 * d[1]
            let b = 0.6 * d[0] + 0.8 * d[1]
            let c = 0.7071067811865476 * (d[2] - d[3])
            let e = 0.7071067811865476 * (d[2] + d[3])
            return a*a + 100*b*b + 10_000*c*c + 10*e*e
        }
        for _ in 0..<180 {
            let candidates = cma.ask()
            cma.tell(ranked: candidates.sorted { objective($0.values) < objective($1.values) })
        }
        XCTAssertLessThan(objective(cma.mean), 1e-4)
        XCTAssertGreaterThan(cma.conditionNumber, 10, "optimizer should adapt non-spherical covariance")
    }

    /// Elite selection must not hand back the same point twice: the CMA archive
    /// clusters tightly as it converges, and the multistart-specific "always
    /// keep indices 0 and 1" convention bypasses the diversity floor.
    func testDiverseSelectionKeepsOnlyMandatoryWithoutFloor() {
        let candidates: [[Float]] = [
            [0.5, 0.5],        // best
            [0.5, 0.5001],     // near-duplicate of the best
            [0.9, 0.1],        // genuinely different
            [0.1, 0.9],
        ]
        let scores: [Float] = [1.0, 1.0001, 2.0, 3.0]
        let elites = BatchMultistart.diverseSelection(
            candidates: candidates, scores: scores, count: 3, mandatory: [0]
        ).map { candidates[$0] }

        XCTAssertEqual(elites.count, 3)
        XCTAssertEqual(elites[0], candidates[0])
        XCTAssertFalse(
            elites.contains(candidates[1]),
            "near-duplicate of the best must be rejected by the diversity floor")
        XCTAssertEqual(Set(elites.map(\.description)).count, elites.count, "elites must be distinct")
    }

    private func sphere(_ x: [Double]) -> Double { x.reduce(0) { $0 + $1 * $1 } }
    private func squaredDistance(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
    }
}

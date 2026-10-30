import XCTest

@testable import DGenAnalysis

final class KickSlotsTests: XCTestCase {
  func testHungarianFindsTheOptimum() {
    // brute force over all permutations of a 5x5 matrix
    var rng = SystemRandomNumberGenerator()
    for _ in 0..<20 {
      let n = 5
      let a = (0..<n).map { _ in (0..<n).map { _ in Double.random(in: 0...10, using: &rng) } }
      let m = Hungarian.solve(a)
      XCTAssertEqual(Set(m).count, n)
      let got = (0..<n).reduce(0.0) { $0 + a[$1][m[$1]] }
      var best = Double.infinity
      func permute(_ p: inout [Int], _ k: Int) {
        if k == n { best = min(best, (0..<n).reduce(0.0) { $0 + a[$1][p[$1]] }); return }
        for i in k..<n { p.swapAt(k, i); permute(&p, k + 1); p.swapAt(k, i) }
      }
      var p = Array(0..<n)
      permute(&p, 0)
      XCTAssertEqual(got, best, accuracy: 1e-9)
    }
  }
}

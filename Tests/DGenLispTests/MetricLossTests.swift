import Foundation
import XCTest

@testable import DGenLisp

/// Finite-difference checks of the CPU metric losses' gradients.
final class MetricLossTests: XCTestCase {
  private func signal(_ n: Int, seed: UInt64, f: Double, decay: Double) -> [Double] {
    var rng = SplitMix64(seed: seed)
    return (0..<n).map { i in
      let t = Double(i) / 44100
      return 0.8 * exp(-t / decay) * sin(2 * .pi * f * t) + 0.05 * Double(rng.gaussian()) * exp(-t / 0.02)
    }
  }

  private func check(_ f: ([Double]) -> (Double, [Double]), _ y: [Double], probes: [Int], h: Double,
                     tol: Double, file: StaticString = #filePath, line: UInt = #line) {
    let (_, g) = f(y)
    for i in probes {
      var a = y, b = y
      a[i] += h; b[i] -= h
      let fd = (f(a).0 - f(b).0) / (2 * h)
      let scale = max(abs(fd), abs(g[i]), 1e-9)
      XCTAssertLessThan(abs(fd - g[i]) / scale, tol, "index \(i): fd \(fd) vs ad \(g[i])", file: file, line: line)
    }
  }

  func testMRGradientMatchesFiniteDifferences() {
    let n = 4096
    let y = signal(n, seed: 1, f: 60, decay: 0.08), t = signal(n, seed: 2, f: 55, decay: 0.1)
    check({ MetricLoss.mr($0, t, windows: [64, 256, 1024]) }, y, probes: [5, 300, 1500, 2900, 4000], h: 1e-6, tol: 2e-2)
  }

  func testBandGradientMatchesFiniteDifferences() {
    let n = 8192
    let y = signal(n, seed: 3, f: 60, decay: 0.08), t = signal(n, seed: 4, f: 55, decay: 0.1)
    check({ MetricLoss.band($0, t) }, y, probes: [10, 700, 2500, 5000, 8000], h: 1e-6, tol: 2e-2)
  }
}

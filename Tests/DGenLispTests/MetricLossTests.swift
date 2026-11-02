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

  func testLowWaveGradientMatchesFiniteDifferences() {
    let n = 4096
    let y = signal(n, seed: 5, f: 60, decay: 0.08), t = signal(n, seed: 6, f: 55, decay: 0.1)
    check({ MetricLoss.lowWave($0, t, cutoff: 600, sampleRate: 44100) }, y, probes: [3, 900, 2048, 4000], h: 1e-6, tol: 1e-2)
  }

  func testBandGradientMatchesFiniteDifferences() {
    let n = 8192
    let y = signal(n, seed: 3, f: 60, decay: 0.08), t = signal(n, seed: 4, f: 55, decay: 0.1)
    check({ MetricLoss.band($0, t) }, y, probes: [10, 700, 2500, 5000, 8000], h: 1e-6, tol: 2e-2)
  }

  /// --hf-weight: the score stays the plain metric; the gradient is the low bands' part plus W x the
  /// high bands' part (bands whose lowest bin is at or above hfFrom).
  func testBandHFWeightScalesOnlyHighBandGradient() {
    let n = 8192
    let y = signal(n, seed: 7, f: 60, decay: 0.08), t = signal(n, seed: 8, f: 55, decay: 0.1)
    let w = MetricLoss.bandWeights(hfWeight: 3, hfFrom: 4000)
    let high = w.map { $0 == 3 ? 1.0 : 0 }, low = w.map { $0 == 3 ? 0 : 1.0 }
    XCTAssertTrue(high.contains(1) && low.contains(1), "4 kHz splits the bands")
    let (s0, g0) = MetricLoss.band(y, t)
    let (sw, gw) = MetricLoss.band(y, t, weights: w)
    let (_, gl) = MetricLoss.band(y, t, weights: low), (_, gh) = MetricLoss.band(y, t, weights: high)
    XCTAssertEqual(s0, sw)
    XCTAssertEqual(MetricLoss.band(y, t, weights: MetricLoss.bandWeights(hfWeight: 1, hfFrom: 4000)).1, g0)
    let scale = gw.map(abs).max()!
    XCTAssertGreaterThan(gh.map(abs).max()!, 0)
    for i in 0..<n {
      XCTAssertEqual(gw[i], gl[i] + 3 * gh[i], accuracy: 1e-9 * scale)
    }
  }

  /// A trainer's cached target spectra give the same score and gradient as computing them per call.
  func testCachedTargetMatchesUncached() {
    let n = 8192
    let y = signal(n, seed: 9, f: 60, decay: 0.08), t = signal(n, seed: 10, f: 55, decay: 0.1)
    let tg = MetricLoss.target(t, windows: [64, 256, 1024])
    let (b0, gb0) = MetricLoss.band(y, t), (b1, gb1) = MetricLoss.band(y, t, target: tg)
    let (m0, gm0) = MetricLoss.mr(y, t, windows: [64, 256, 1024])
    let (m1, gm1) = MetricLoss.mr(y, t, windows: [64, 256, 1024], target: tg)
    XCTAssertEqual(b0, b1)
    XCTAssertEqual(m0, m1)
    XCTAssertEqual(gb0, gb1)
    XCTAssertEqual(gm0, gm1)
  }
}

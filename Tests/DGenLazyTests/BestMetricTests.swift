import XCTest

@testable import DDSPE2E

/// The checkpoint-selection metric decides which model a training chain hands
/// forward, so its vDSP implementation is checked against a plainly-written
/// oracle rather than trusted on inspection.
final class BestMetricTests: XCTestCase {

  private func signal(count: Int, seed: UInt64, tone: Float) -> [Float] {
    var state = seed
    return (0..<count).map { i in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let noise = Float(state >> 40) / Float(1 << 24) * 2.0 - 1.0
      return 0.3 * noise + Foundation.sin(tone * Float(i))
    }
  }

  func testVDSPMatchesReferenceImplementation() throws {
    let a = signal(count: 4096, seed: 11, tone: 0.05)
    let b = signal(count: 4096, seed: 22, tone: 0.08)
    let windows = [64, 128, 256, 512, 1024]

    for epsilon: Float in [1e-3, 1e-8] {
      let fast = BestCheckpointScorer.multiScaleSpectralScore(
        prediction: a, target: b, windowSizes: windows, hopDivisor: 4, logEpsilon: epsilon)
      let reference = BestCheckpointScorer.referenceScore(
        prediction: a, target: b, windowSizes: windows, hopDivisor: 4, logEpsilon: epsilon)

      let fastValue = try XCTUnwrap(fast)
      let referenceValue = try XCTUnwrap(reference)
      XCTAssertEqual(
        fastValue, referenceValue, accuracy: Swift.max(1e-4, referenceValue * 1e-3),
        "vDSP score diverged from reference at epsilon \(epsilon)")
    }
  }

  func testIdenticalSignalsScoreZero() throws {
    let a = signal(count: 2048, seed: 7, tone: 0.03)
    let score = try XCTUnwrap(
      BestCheckpointScorer.multiScaleSpectralScore(
        prediction: a, target: a, windowSizes: [256, 1024], hopDivisor: 4, logEpsilon: 1e-3))
    XCTAssertEqual(score, 0, accuracy: 1e-5)
  }

  func testCloserSignalScoresLower() throws {
    let target = signal(count: 2048, seed: 3, tone: 0.04)
    let near = zip(target, signal(count: 2048, seed: 99, tone: 0.04)).map { $0 * 0.95 + $1 * 0.05 }
    let far = signal(count: 2048, seed: 99, tone: 0.21)

    let nearScore = try XCTUnwrap(
      BestCheckpointScorer.multiScaleSpectralScore(
        prediction: near, target: target, windowSizes: [256, 1024], hopDivisor: 4,
        logEpsilon: 1e-3))
    let farScore = try XCTUnwrap(
      BestCheckpointScorer.multiScaleSpectralScore(
        prediction: far, target: target, windowSizes: [256, 1024], hopDivisor: 4,
        logEpsilon: 1e-3))
    XCTAssertLessThan(nearScore, farScore)
  }
}

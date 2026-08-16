import XCTest

@testable import DGenLazy

/// Regression tests for the compile-cache fingerprint in Realize.swift.
///
/// Constant node values are baked into kernel source as literals. The
/// fingerprint must therefore include them: a rebuilt graph that changes only
/// a constant (e.g. a scheduled loss weight ramping across training steps) has
/// identical node/tensor counts, and a count-only fingerprint silently reuses
/// stale kernels with the old constant baked in.
final class CompilationCacheTests: XCTestCase {

  /// Rebuild the same-topology graph each epoch with a different constant
  /// weight, tinygrad-style (same LazyGraph, graph cleared after backward).
  /// The realized loss must track the constant, not the first-compiled value.
  func testRebuiltGraphPicksUpChangedConstant() throws {
    let frameCount = 8
    LazyGraphContext.reset()
    let p = Signal.param(3.0)

    var losses: [Float] = []
    for w: Float in [1.0, 2.0, 4.0, 4.0] {
      let loss = p * Signal.constant(w)
      let values = try loss.backward(frames: frameCount)
      losses.append(values[0])
    }

    XCTAssertEqual(losses[0], 3.0, accuracy: 1e-5)
    XCTAssertEqual(losses[1], 6.0, accuracy: 1e-5, "stale compile cache: constant change ignored")
    XCTAssertEqual(losses[2], 12.0, accuracy: 1e-5, "stale compile cache: constant change ignored")
    XCTAssertEqual(losses[3], 12.0, accuracy: 1e-5)
  }
}

import XCTest

@testable import DGen
@testable import DGenLazy

/// Regressions found while building Examples/ModalSnareAD (closed-form [B,K]
/// modal banks driven by a frame-parallel clock).
final class FrameTensorRegionRegressionTests: XCTestCase {
  let frames = 512
  let sr: Float = 48_000

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sr
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func seconds() -> Signal {
    let rampHz = sr / (2 * Float(frames))
    return Signal.phasor(Tensor([rampHz])).sum() * (2 * Float(frames) / sr)
  }

  /// `(x * a).sum(axis: 1)` where x is a frame tensor produced in the same
  /// block used to return all zeros: the inline mul-reduce read x from memory,
  /// but x's only consumer was the elided mul, so its store was skipped.
  func testSumAxisOfMulReadsInBlockOperand() throws {
    let data: [Float] = [1, 2, 3, 4, 5, 6]
    let a = Tensor(data).reshape([2, 3])
    let x = seconds() * a
    let out = try (x * a).sum(axis: 1).realize(frames: frames)
    XCTAssertEqual(out.count, frames * 2)
    for n in [1, 100, 511] {
      let t = Float(n) / sr
      XCTAssertEqual(out[n * 2], t * (1 + 4 + 9), accuracy: 1e-5)
      XCTAssertEqual(out[n * 2 + 1], t * (16 + 25 + 36), accuracy: 1e-5)
    }
  }
}

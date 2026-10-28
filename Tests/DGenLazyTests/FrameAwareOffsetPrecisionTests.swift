import XCTest

@testable import DGen
@testable import DGenLazy

/// Frame-aware tensor offsets `cell + frame * size + elem` were computed in
/// float whenever the element index was float-typed (expandAxis in the
/// sumAxis backward), so beyond 2^24 elements (here 48000 frames x [32, 16])
/// reads/writes landed on rounded addresses and late-frame gradients were
/// wrong (one element was off by 100%).
final class FrameAwareOffsetPrecisionTests: XCTestCase {
  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  func testLateFrameGradientPastFloatIntegerRange() throws {
    let frames = 48_000, B = 32, K = 16
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
    let w0 = (0..<(B * K)).map { Float($0 % 7) * 0.1 + 0.3 }
    let w = Tensor.param([B, K], data: w0)
    let rate = Float(48_000) / Float(2 * frames)
    let t = Signal.phasor(Tensor([rate])).sum() * (1 / rate)
    let late = relu(t - 0.75)  // nonzero only in the last quarter of the render
    let row = (w * late).sum(axis: 1)
    _ = try (row * row).sum().backward(frames: frames)
    let g = try XCTUnwrap(w.grad?.getData())
    var s2 = 0.0
    for n in 0..<frames {
      let l = max(0, Double(n) / 48_000 - 0.75)
      s2 += l * l
    }
    for b in 0..<B {
      let rowSum = (0..<K).reduce(Float(0)) { $0 + w0[b * K + $1] }
      for k in 0..<K {
        let expect = Float(2 * s2) * rowSum
        XCTAssertEqual(g[b * K + k], expect, accuracy: 1e-4 * expect, "w[\(b), \(k)]")
      }
    }
  }
}

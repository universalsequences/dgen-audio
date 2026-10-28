import XCTest

@testable import DGen
@testable import DGenLazy

/// Metal compiles with fast math, under which metal::tanh returned 0 at 44
/// and NaN above, e.g. a tanh saturator fed a loud training render.
final class MetalTanhOverflowTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  func testTensorTanhSaturatesForLargeInputs() throws {
    let values: [Float] = [0.5, 5, 20, 44, 50, 100, 1e4]
    let x = Tensor(values + values.map { -$0 })
    let ramp = Signal.phasor(Tensor([1])).sum() * 0.0 + 1.0  // frame tensor, not folded
    let out = try tanh(ramp * x).realize(frames: 64)
    let n = values.count * 2
    for (i, v) in (values + values.map { -$0 }).enumerated() {
      XCTAssertEqual(out[n * 10 + i], Foundation.tanh(v), accuracy: 1e-6, "tanh(\(v))")
    }
  }

  func testTanhGradientIsFiniteWhenSaturated() throws {
    let w = Tensor.param([2], data: [60, -60])
    let ramp = Signal.phasor(Tensor([1])).sum() * 0.0 + 1.0
    let loss = tanh(ramp * w).sum()
    _ = try loss.backward(frames: 64)
    let g = try XCTUnwrap(w.grad?.getData())
    XCTAssertTrue(g.allSatisfy { $0.isFinite }, "grad \(g)")
  }
}

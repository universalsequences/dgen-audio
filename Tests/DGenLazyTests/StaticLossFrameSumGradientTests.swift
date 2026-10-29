import XCTest

@testable import DGen
@testable import DGenLazy

/// `loss.backward(frames: N)` minimises the SUM of the per-frame loss values it returns.
/// A loss that is frame-invariant (a static tensor expression peeked into a Signal, e.g.
/// an L2 regulariser or a parameter-space anchor) must therefore get N times its per-frame
/// gradient, exactly as a frame-varying loss gets the sum of its per-frame gradients.
/// Otherwise mixing a static regulariser with an audio loss in one graph silently
/// re-weights it by 1/N.
final class StaticLossFrameSumGradientTests: XCTestCase {
  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func run(frames: Int, maxFrames: Int, mixed: Bool) throws -> (grad: [Float], lossSum: Float) {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = maxFrames
    LazyGraphContext.reset()
    let p = Tensor.param([2], data: [3, -1.5])
    // static: per frame (p0^2 + p1^2)
    var loss = (Tensor([0.0]) + (p * p).sum()).peek(Signal.constant(0))
    if mixed {
      // plus a frame-varying term that does not depend on p
      let rate = Float(48_000) / Float(2 * maxFrames)
      loss = loss + Signal.phasor(Tensor([rate])).sum() * 0.0
    }
    let v = try loss.backward(frames: frames)
    return (p.grad?.getData() ?? [], v.reduce(0, +))
  }

  func testStaticLossGradientIsSummedOverFrames() throws {
    for (frames, maxFrames, mixed) in [(64, 64, false), (64, 4096, false), (64, 64, true), (256, 512, true)] {
      let r = try run(frames: frames, maxFrames: maxFrames, mixed: mixed)
      let n = Float(frames)
      XCTAssertEqual(r.lossSum, n * (9 + 2.25), accuracy: 1e-3 * n, "loss frames=\(frames) max=\(maxFrames) mixed=\(mixed)")
      XCTAssertEqual(r.grad.count, 2)
      // d/dp sum_frames (p.p) = N * 2p
      XCTAssertEqual(r.grad.first ?? .nan, n * 6, accuracy: 1e-3 * n, "frames=\(frames) max=\(maxFrames) mixed=\(mixed)")
      XCTAssertEqual(r.grad.last ?? .nan, n * -3, accuracy: 1e-3 * n, "frames=\(frames) max=\(maxFrames) mixed=\(mixed)")
    }
  }
}

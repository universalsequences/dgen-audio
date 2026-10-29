import XCTest

@testable import DGen
@testable import DGenLazy

/// The backward of `sum(u * g)` (u = x @ W + expand(b)) contains sumAxis0(mul(expand(seed), g)):
/// GEMMPass rewrote it to sumMulAxis0 over the view chains' SOURCES, dropping the expand of
/// the scalar seed, so shape inference failed ("requires two matching 2D tensor inputs").
/// This is the chain-rule term a two-pass trainer uses to push a frame-summed dL/du
/// through a static decoder.
final class StaticSumOfMulMatmulGradientTests: XCTestCase {
  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func grads(withExtra: Bool) throws -> [Float] {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    let W = Tensor.param([3, 4], data: (0..<12).map { Float($0) * 0.1 - 0.5 })
    let b = Tensor.param([1, 4], data: [0.1, 0.2, 0.3, 0.4])
    let x = Tensor((0..<6).map { Float($0) * 0.3 - 0.7 }).reshape([2, 3])
    let g = Tensor((0..<8).map { Float($0) * 0.25 - 1 }).reshape([2, 4])
    let u = x.matmul(W) + b.expand([2, 4])
    var loss = (u * g).sum()
    if withExtra { loss = loss + (W * W).sum() * 0.0 }
    _ = try (Tensor([0.0]) + loss).peek(Signal.constant(0)).backward(frames: 64)
    return (W.grad?.getData() ?? []) + (b.grad?.getData() ?? [])
  }

  func testSumOfMulOverMatmulOutput() throws {
    // analytic: dW = x^T g * 64 frames, db = sum_rows g * 64
    let x = (0..<6).map { Float($0) * 0.3 - 0.7 }, g = (0..<8).map { Float($0) * 0.25 - 1 }
    var expect = [Float](repeating: 0, count: 16)
    for i in 0..<3 { for j in 0..<4 { for r in 0..<2 { expect[i * 4 + j] += x[r * 3 + i] * g[r * 4 + j] * 64 } } }
    for j in 0..<4 { for r in 0..<2 { expect[12 + j] += g[r * 4 + j] * 64 } }
    for extra in [true, false] {
      let got = try grads(withExtra: extra)
      XCTAssertEqual(got.count, 16, "extra=\(extra)")
      for (a, e) in zip(got, expect) { XCTAssertEqual(a, e, accuracy: 1e-3 * Swift.max(1, Swift.abs(e)), "extra=\(extra)") }
    }
  }
}

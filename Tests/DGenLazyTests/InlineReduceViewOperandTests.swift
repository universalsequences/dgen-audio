import XCTest

@testable import DGen
@testable import DGenLazy

/// `(w * s.reshape([1, K]).expand([B, K])).sum(axis: 1)`: a shared frame
/// signal broadcast across a batch, weighted and reduced (filter-bank noise
/// shared by B voices). The inlined sumAxis(mul) read the expanded view with
/// base strides; debug builds asserted, release builds read wrong addresses
/// and produced wrong values/gradients.
final class InlineReduceViewOperandTests: XCTestCase {
  let frames = 512
  let freqs: [Float] = [300, 700, 1100]
  let w0: [Float] = [0.3, -0.2, 0.5, 0.7, 0.1, -0.4]

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func shared() -> SignalTensor {
    sin(Signal.phasor(Tensor(freqs)) * (2 * Float.pi)).reshape([1, 3]).expand([2, 3])
  }

  private func loss(_ w: DGenLazy.Tensor) -> Signal {
    let rate: Float = 48_000.0 / 1024
    let t = Signal.phasor(Tensor([rate])).sum() * (1 / rate)
    let env = exp(t * w * -10.0) * w
    let y = (env * shared()).sum(axis: 1)
    return (y * y).sum()
  }

  func testForwardMatchesCPU() throws {
    let y = try (shared() * Tensor(w0).reshape([2, 3])).sum(axis: 1).realize(frames: frames)
    for n in [1, 77, 300] {
      for b in 0..<2 {
        var acc: Float = 0
        for k in 0..<3 {
          acc += w0[b * 3 + k] * Foundation.sin(2 * Float.pi * freqs[k] * Float(n) / 48_000)
        }
        XCTAssertEqual(y[n * 2 + b], acc, accuracy: 1e-4)
      }
    }
  }

  func testGradientMatchesFiniteDifference() throws {
    let w = Tensor.param([2, 3], data: w0)
    _ = try loss(w).backward(frames: frames)
    let ad = try XCTUnwrap(w.grad?.getData())
    for i in 0..<6 {
      func lv(_ d: Float) throws -> Double {
        LazyGraphContext.reset()
        var v = w0
        v[i] += d
        return try loss(Tensor(v).reshape([2, 3])).realize(frames: frames).reduce(0.0) { $0 + Double($1) }
      }
      let fd = Float((try lv(1e-2) - (try lv(-1e-2))) / 2e-2)
      XCTAssertEqual(ad[i], fd, accuracy: 1e-3 * Swift.max(1, Swift.abs(fd)), "w[\(i)]")
    }
  }
}

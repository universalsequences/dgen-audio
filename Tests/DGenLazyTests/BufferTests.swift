import XCTest

@testable import DGenLazy

/// Tests for Signal.buffer() — sliding window view over a signal's history.
///
/// buffer(size) writes each frame's signal value into a flat history array
/// and returns a [1, size] tensor view via a slidingWindow transform.
/// Element i at frame f reads history[f - size + 1 + i], with out-of-bounds → 0.
final class BufferTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  /// Verify buffer sum matches expected sliding window sums.
  /// Counter 0, 1, 2, ... with buffer(4) should sum the last 4 values.
  func testBufferSum() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let result = try counter.buffer(size: 4).sum().realize(frames: 8)

    //   f=0: [0,0,0,0]→0   f=1: [0,0,0,1]→1   f=2: [0,0,1,2]→3   f=3: [0,1,2,3]→6
    //   f=4: [1,2,3,4]→10  f=5: [2,3,4,5]→14  f=6: [3,4,5,6]→18  f=7: [4,5,6,7]→22
    let expected: [Float] = [0, 1, 3, 6, 10, 14, 18, 22]
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.01, "Frame \(i) sum mismatch")
    }
  }

  /// Verify element ORDER using a weighted conv2d kernel [1, 10, 100, 1000].
  /// Each buffer position contributes to a unique decimal digit, so misordering
  /// produces a visibly wrong number (e.g. 1234 vs 4321).
  func testBufferElementOrder() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let buf = counter.buffer(size: 4)

    let kernel = Tensor([[1, 10, 100, 1000]])
    let result = try buf.conv2d(kernel).sum().realize(frames: 8)

    // Buffer at frame f = [max(f-3,0), max(f-2,0), max(f-1,0), f] (zeros for early frames)
    // Weighted: 1*buf[0] + 10*buf[1] + 100*buf[2] + 1000*buf[3]
    let expected: [Float] = [0, 1000, 2100, 3210, 4321, 5432, 6543, 7654]
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.5,
        "Frame \(i): buffer element order is wrong")
    }
  }

  /// buffer composes with conv2d on a real signal (sine wave).
  func testBufferConv2d() throws {
    let sig = sin(Signal.phasor(440.0) * Signal.constant(2.0 * .pi))
    let filtered = sig.buffer(size: 128).conv2d(Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]))

    let result = try filtered.realize(frames: 200)
    let tensorSize = 124  // 128 - 5 + 1

    let lastFrame = Array(result[(199 * tensorSize)..<(200 * tensorSize)])
    let range = (lastFrame.max() ?? 0) - (lastFrame.min() ?? 0)

    XCTAssertGreaterThan(range, 0.01, "Filtered output should have non-trivial variation")
  }

  /// Gradients flow through buffer -> conv2d to the kernel.
  func testBufferGradient() throws {
    let sig = sin(Signal.phasor(440.0) * Signal.constant(2.0 * .pi))
    let kernel = Tensor.param([1, 5], data: [0.2, 0.2, 0.2, 0.2, 0.2])
    let filtered = sig.buffer(size: 32).conv2d(kernel)

    let loss = (filtered * filtered).sum()
    let _ = try loss.backward(frames: 64)

    let gradData = kernel.grad?.getData() ?? []
    XCTAssertFalse(gradData.isEmpty, "Kernel should have gradients")
    XCTAssertTrue(gradData.contains(where: { abs($0) > 1e-6 }), "At least one gradient should be non-zero")
  }
}

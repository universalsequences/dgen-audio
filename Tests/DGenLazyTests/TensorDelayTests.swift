import DGen
import XCTest

@testable import DGenLazy

/// Tests for per-lane tensor delay lines (`Graph.tensorDelay` / `.delayLine`).
///
/// Exactness strategy: drive each lane with a frame-counter ramp scaled by a
/// distinct per-lane factor, so `input[lane][n] = factor[lane] * n`. A correct
/// delay of `d` samples then satisfies `out[n] == input[n - d]` for `n >= d`
/// and `0` before that (cells are zero-initialized) — no cross-implementation
/// alignment assumptions needed.
final class TensorDelayTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 48000.0
    DGenConfig.backend = .c
    DGenConfig.maxFrameCount = 128
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100.0
    super.tearDown()
  }

  /// Frame counter: 0, 1, 2, ... (accum outputs the value BEFORE incrementing).
  private func frameCounter() -> Signal {
    return Signal.accum(Signal.constant(1.0), min: 0.0, max: 1_000_000.0)
  }

  /// Extract lane `i` of a [2] SignalTensor as a scalar Signal via a one-hot mask.
  private func lane(_ st: SignalTensor, _ i: Int) -> Signal {
    var mask: [Float] = [0, 0]
    mask[i] = 1
    return (st * Tensor(mask)).sum()
  }

  private func assertDelayedRamp(
    _ result: [Float], factor: Float, delay: Int, accuracy: Float = 1e-3,
    _ label: String, file: StaticString = #file, line: UInt = #line
  ) {
    for n in 0..<result.count {
      let expected: Float = n >= delay ? factor * Float(n - delay) : 0.0
      XCTAssertEqual(
        result[n], expected, accuracy: accuracy,
        "\(label): frame \(n)", file: file, line: line)
    }
  }

  // MARK: - Scalar regression

  func testScalarDelayShiftsByRequestedSamples() throws {
    let input = frameCounter()
    let delayed = input.delay(10.0)
    let result = try delayed.realize(frames: 64)
    assertDelayedRamp(result, factor: 1.0, delay: 10, "scalar delay")
  }

  func testScalarDelayHonorsMaxDelayOverride() throws {
    // Small buffer, delay within it — still an exact shift.
    let input = frameCounter()
    let delayed = input.delay(10.0, maxDelay: 32)
    let result = try delayed.realize(frames: 30)
    assertDelayedRamp(result, factor: 1.0, delay: 10, "scalar delay @max-delay 32")
  }

  // MARK: - Tensor delay, broadcast scalar time

  func testTensorDelayScalarTimeDelaysLanesIndependently() throws {
    let factors = Tensor([2.0, 5.0])
    let input = factors * frameCounter()  // lane i = factor[i] * n
    let delayed = input.delay(10.0)
    XCTAssertEqual(delayed.shape, [2])

    let lane0 = try lane(delayed, 0).realize(frames: 64)
    LazyGraphContext.reset()
    let input2 = Tensor([2.0, 5.0]) * frameCounter()
    let delayed2 = input2.delay(10.0)
    let lane1 = try lane(delayed2, 1).realize(frames: 64)

    assertDelayedRamp(lane0, factor: 2.0, delay: 10, "tensor delay lane 0")
    assertDelayedRamp(lane1, factor: 5.0, delay: 10, "tensor delay lane 1")
  }

  // MARK: - Tensor delay, per-lane times

  func testTensorDelayPerLaneTimes() throws {
    let times = Tensor([5.0, 9.0])

    let input0 = Tensor([2.0, 5.0]) * frameCounter()
    let lane0 = try lane(input0.delay(times), 0).realize(frames: 64)
    LazyGraphContext.reset()
    let input1 = Tensor([2.0, 5.0]) * frameCounter()
    let lane1 = try lane(input1.delay(Tensor([5.0, 9.0])), 1).realize(frames: 64)

    assertDelayedRamp(lane0, factor: 2.0, delay: 5, "per-lane time lane 0")
    assertDelayedRamp(lane1, factor: 5.0, delay: 9, "per-lane time lane 1")
  }

  // MARK: - Fractional + clamped times

  func testTensorDelayFractionalTimeInterpolates() throws {
    let input = Tensor([1.0, 1.0]) * frameCounter()
    let delayed = input.delay(10.5)
    let result = try lane(delayed, 0).realize(frames: 64)
    // On a ramp, linear interpolation between n-10 and n-11 gives n - 10.5.
    for n in 12..<result.count {
      XCTAssertEqual(
        result[n], Float(n) - 10.5, accuracy: 1e-3, "fractional delay frame \(n)")
    }
  }

  func testTensorDelayClampsOversizedTime() throws {
    // maxDelay 32 with a requested delay of 1000: clamps to 31 instead of
    // wrapping into garbage.
    let input = Tensor([1.0, 1.0]) * frameCounter()
    let delayed = input.delay(1000.0, maxDelay: 32)
    let result = try lane(delayed, 0).realize(frames: 64)
    assertDelayedRamp(result, factor: 1.0, delay: 31, "clamped tensor delay")
  }

  // MARK: - Backend parity

  func testTensorDelayCMetalParity() throws {
    func build() -> Signal {
      let input = Tensor([2.0, 5.0]) * frameCounter()
      let delayed = input.delay(Tensor([5.0, 9.0]))
      return delayed.sum()
    }

    DGenConfig.backend = .c
    LazyGraphContext.reset()
    let c = try build().realize(frames: 128)

    DGenConfig.backend = .metal
    LazyGraphContext.reset()
    let metal = try build().realize(frames: 128)

    XCTAssertEqual(c.count, metal.count)
    for i in 0..<c.count {
      XCTAssertEqual(c[i], metal[i], accuracy: 1e-3, "C/Metal parity frame \(i)")
    }
  }
}

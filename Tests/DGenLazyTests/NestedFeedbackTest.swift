import XCTest

@testable import DGenLazy

/// Tests for identity-folding (x*1, x+0, x-0, x/1) interacting with
/// cross-block dependencies. When an arithmetic identity is folded away,
/// the node must still emit a UOp so `defineGlobal` propagates correctly.
final class IdentityFoldingCrossBlockTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100.0
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100.0
    DGenConfig.debug = false
    super.tearDown()
  }

  // MARK: - Chained delay with identity folding

  /// delay() uses .seq to order write-before-read, creating cross-block
  /// dependencies. When the result is folded by * 1.0, the output node
  /// must still produce a UOp for defineGlobal to work.
  func testDelayChainWithMulByOne() throws {
    func fn(_ input: Signal) -> Signal {
      return input.delay(220) * 1.0
    }
    let input = Signal.phasor(430)
    let a1 = fn(input)
    let a2 = fn(a1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testDelayChainWithOneMulX() throws {
    func fn(_ input: Signal) -> Signal {
      return 1.0 * input.delay(220)
    }
    let input = Signal.phasor(430)
    let a1 = fn(input)
    let a2 = fn(a1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testDelayChainWithAddZero() throws {
    func fn(_ input: Signal) -> Signal {
      return input.delay(220) + 0.0
    }
    let input = Signal.phasor(430)
    let a1 = fn(input)
    let a2 = fn(a1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testDelayChainWithZeroAddX() throws {
    func fn(_ input: Signal) -> Signal {
      return 0.0 + input.delay(220)
    }
    let input = Signal.phasor(430)
    let a1 = fn(input)
    let a2 = fn(a1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testDelayChainWithSubZero() throws {
    func fn(_ input: Signal) -> Signal {
      return input.delay(220) - 0.0
    }
    let input = Signal.phasor(430)
    let a1 = fn(input)
    let a2 = fn(a1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testDelayChainWithDivByOne() throws {
    func fn(_ input: Signal) -> Signal {
      return input.delay(220) / 1.0
    }
    let input = Signal.phasor(430)
    let a1 = fn(input)
    let a2 = fn(a1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  // MARK: - Nested feedback (allpass filter) with identity folding

  /// Allpass filters in series use history() feedback + delay, creating
  /// scalar corridors with cross-block dependencies. Identity folds on
  /// the output path must not break the defineGlobal chain.
  func testNestedFeedbackAllpassWithMulByOne() throws {
    func allpass(_ input: Signal, _ delayTime: Signal, _ delayMult: Signal) -> Signal {
      let (prev, write) = Signal.history()
      let delayed = prev.delay(delayTime * delayMult)
      let alpha = Signal.constant(0.6)
      let sum = (input * 1.0 + -1.0 * delayed * alpha) * alpha
      _ = write(sum)
      return input * 1.0
    }

    let input = Signal.phasor(430)
    let delayTime = Signal.constant(120)
    let mult = Signal.constant(0.9)
    let a1 = allpass(input, delayTime, mult)
    let a2 = allpass(a1, delayTime, mult)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testNestedFeedbackAllpassWithAddZero() throws {
    func allpass(_ input: Signal, _ delayTime: Signal, _ delayMult: Signal) -> Signal {
      let (prev, write) = Signal.history()
      let delayed = prev.delay(delayTime * delayMult)
      let alpha = Signal.constant(0.6)
      let sum = (input + 0.0 + -1.0 * delayed * alpha) * alpha
      _ = write(sum)
      return input + 0.0
    }

    let input = Signal.phasor(430)
    let delayTime = Signal.constant(120)
    let mult = Signal.constant(0.9)
    let a1 = allpass(input, delayTime, mult)
    let a2 = allpass(a1, delayTime, mult)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

  func testNestedFeedbackAllpassWithDivByOne() throws {
    func allpass(_ input: Signal, _ delayTime: Signal, _ delayMult: Signal) -> Signal {
      let (prev, write) = Signal.history()
      let delayed = prev.delay(delayTime * delayMult)
      let alpha = Signal.constant(0.6)
      let sum = (input / 1.0 + -1.0 * delayed * alpha) * alpha
      _ = write(sum)
      return input / 1.0
    }

    let input = Signal.phasor(430)
    let delayTime = Signal.constant(120)
    let mult = Signal.constant(0.9)
    let a1 = allpass(input, delayTime, mult)
    let a2 = allpass(a1, delayTime, mult)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }
}

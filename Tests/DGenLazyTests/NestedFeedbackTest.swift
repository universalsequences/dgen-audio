import XCTest

@testable import DGenLazy

final class NestedFeedbackTest: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .c
    DGenConfig.debug = true
    DGenConfig.sampleRate = 44100.0
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100.0
    DGenConfig.debug = false
    super.tearDown()
  }

  func testNestedFeedback() throws {
    func allpass(_ input: Signal, _ delayTime: Signal, _ delayMult: Signal) -> Signal {
      let (prev, write) = Signal.history()
      let delayed = prev.delay(delayTime * delayMult)
      let alpha = Signal.constant(0.6)
      let sum = (input * 1.0 + -1.0 * delayed * alpha) * alpha
      _ = write(sum)
      let sum2 = sum + delayed
      return sum2 * 1.0
    }

    let input = Signal.phasor(430)
    let delayTime1 = Signal.constant(120)
    let mult1 = Signal.constant(0.9)
    let a1 = allpass(input, delayTime1, mult1)
    let a2 = allpass(a1, delayTime1, mult1)
    let result = try a2.realize(frames: 8)
    XCTAssertEqual(result.count, 8)
  }

}

import DGenLazy
import XCTest

@testable import DGen
@testable import DGenLisp

final class BatchLispEvaluatorTests: XCTestCase {
  func testScalarPatchLiftsParamsPhasorsHistoryAndOutputIntoLanes() throws {
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 2_000
    DGenConfig.maxFrameCount = 32
    LazyGraphContext.reset()

    let gain = Tensor([0.25, 0.75], requiresGrad: true) * Signal.constant(1)
    let frequency = Tensor([100, 200], requiresGrad: true) * Signal.constant(1)
    let evaluator = LispEvaluator(
      batchLaneCount: 2,
      batchParameterValues: ["gain": .signalTensor(gain), "frequency": .signalTensor(frequency)])
    try evaluator.evaluate(source: """
      (defmacro leaky (x)
        (make-history state)
        (def previous (read-history state))
        (def next (+ x (* previous 0.5)))
        (write-history state next))
      (param gain @default 0.5 @min 0 @max 1)
      (param frequency @default 100 @min 20 @max 500)
      (def wave (sin (* (phasor frequency) 6.283185307)))
      (out (* gain (leaky wave)) 1)
      """)

    XCTAssertTrue(evaluator.outputs.isEmpty)
    let output = try XCTUnwrap(evaluator.tensorOutputs.first?.signal)
    XCTAssertEqual(output.shape, [2])
    let flat = try output.realize(frames: 32)
    XCTAssertEqual(flat.count, 64)
    XCTAssertNotEqual(flat[10], flat[11], "independent lane params should produce distinct samples")
  }
}

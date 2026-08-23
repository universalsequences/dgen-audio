import DGenLazy
import XCTest

@testable import DGen
@testable import DGenLisp

final class BatchLispEvaluatorTests: XCTestCase {
  override func tearDown() {
    DGenConfig.maxFrameCount = 4096
    DGenConfig.backend = .metal
    super.tearDown()
  }

  func testBatchEvaluationBroadcastsFrozenSeedValuesAlongsideCandidateParams() throws {
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 2_000
    DGenConfig.maxFrameCount = 1
    LazyGraphContext.reset()

    let transform = DirectionTrainer.TransformedParam(
      LearnableParam(name: "gain", min: 0, max: 1, seedValue: 0.5))
    let nodes = try parseSource("""
      (param fixed_gain @default 0.1 @min 0 @max 1)
      (param mode @default 1 @min 0 @max 1)
      (param gain @default 0.5 @min 0 @max 1)
      (out (selector mode 0 (+ fixed_gain gain)) 1)
      """)
    let output = try BatchMultistart.evaluate(
      z: [Tensor([0.0, 1.0], requiresGrad: true)],
      transforms: [transform],
      parameterValues: ["fixed_gain": 0.75, "mode": 2, "gain": 0.5],
      nodes: nodes, lanes: 2)

    let samples = try output.realize(frames: 1)
    XCTAssertEqual(samples.count, 2)
    XCTAssertEqual(samples[0], 0.75, accuracy: 1e-6)
    XCTAssertEqual(samples[1], 1.75, accuracy: 1e-6)
  }

  func testBatchBackwardPreservesTensorGradientThroughFloor() throws {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 2_000
    DGenConfig.maxFrameCount = 8
    LazyGraphContext.reset()

    let transform = DirectionTrainer.TransformedParam(
      LearnableParam(name: "semitone", min: -12, max: 12, seedValue: 0))
    let semitone = Tensor([0.25, 0.75], requiresGrad: true)
    let nodes = try parseSource("""
      (param semitone @default 0 @min -12 @max 12)
      (out (floor semitone) 1)
      """)
    let output = try BatchMultistart.evaluate(
      z: [semitone], transforms: [transform],
      parameterValues: ["semitone": 0], nodes: nodes, lanes: 2)

    let loss = (output * output).sum()
    _ = try loss.backward(frames: 8)

    let gradient = try XCTUnwrap(semitone.grad?.getData())
    XCTAssertEqual(gradient, [0, 0])
  }

  func testBatchEvaluationLiftsBiquadInputAndAllControlsIntoLanes() throws {
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 2_000
    DGenConfig.maxFrameCount = 32
    LazyGraphContext.reset()

    let transforms = [
      DirectionTrainer.TransformedParam(
        LearnableParam(name: "cutoff", min: 50, max: 900, seedValue: 100)),
      DirectionTrainer.TransformedParam(
        LearnableParam(name: "resonance", min: 0.1, max: 10, seedValue: 0.5)),
    ]
    let nodes = try parseSource("""
      (param cutoff @default 100 @min 50 @max 900)
      (param resonance @default 0.5 @min 0.1 @max 10)
      (def wave (phasor 100))
      (out (biquad wave cutoff resonance 1 0) 1)
      """)
    let output = try BatchMultistart.evaluate(
      z: [Tensor([0.1, 0.9], requiresGrad: true),
          Tensor([0.2, 0.8], requiresGrad: true)],
      transforms: transforms,
      parameterValues: ["cutoff": 100, "resonance": 0.5],
      nodes: nodes, lanes: 2)

    XCTAssertEqual(output.shape, [2])
    XCTAssertTrue(output.requiresGrad)
  }

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

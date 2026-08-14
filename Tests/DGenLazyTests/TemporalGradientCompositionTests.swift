import DGen
import XCTest

@testable import DGenLazy

/// Composability coverage for temporal adjoints. The isolated stateful-phasor
/// suffix scan and history BPTT each pass finite differences on their own; these
/// tests pin the point where enabling both reverse paths changes an unrelated
/// parameter's gradient despite leaving the forward graph unchanged.
final class TemporalGradientCompositionTests: XCTestCase {
  private let frames = 256
  private let detune: Float = 0.65

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100
    DGenConfig.maxFrameCount = 512
    DGenGradientConfig.detachPhasorFrequency = false
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenGradientConfig.detachPhasorFrequency = false
    DGenConfig.backend = .metal
    DGenConfig.maxFrameCount = 4096
    super.tearDown()
  }

  private func oscillator(detune: Signal) -> Signal {
    DGenLazy.sin(Signal.statefulPhasor(55.0 - detune) * (2.0 * Float.pi))
  }

  private func onePole(_ input: Signal, coefficient: Signal) -> Signal {
    let state = Signal.history()
    let output = coefficient * input + (1.0 - coefficient) * state.read
    return state.write(output)
  }

  private func statelessLoss(gain: Signal, detune: Signal) -> Signal {
    mse(oscillator(detune: detune) * gain, Signal.constant(0.1))
  }

  private func onePoleLoss(coefficient: Signal, detune: Signal) -> Signal {
    mse(onePole(oscillator(detune: detune), coefficient: coefficient), Signal.constant(0.1))
  }

  private func twoPoleLoss(coefficient: Signal, detune: Signal) -> Signal {
    let first = onePole(oscillator(detune: detune), coefficient: coefficient)
    return mse(onePole(first, coefficient: Signal.constant(0.12)), Signal.constant(0.1))
  }

  private func svfLowpass(_ input: Signal, g: Signal) -> Signal {
    let damping = Signal.constant(1.0 / 1.2)
    let a1 = 1.0 / (1.0 + g * (g + damping))
    let a2 = g * a1
    let ic1 = Signal.history()
    let ic2 = Signal.history()
    let v3 = input - ic2.read
    let v1 = a1 * ic1.read + a2 * v3
    let ic1New = ic1.write(2.0 * v1 - ic1.read)
    let v1PassThrough = (ic1New + ic1.read) * 0.5
    let v2 = ic2.read + g * v1PassThrough
    let ic2New = ic2.write(2.0 * v2 - ic2.read)
    return (ic2New + ic2.read) * 0.5
  }

  private func svfLoss(cutoff: Signal, detune: Signal) -> Signal {
    let g = DGenLazy.tan(Signal.constant(Float.pi / 44100.0) * cutoff)
    return mse(svfLowpass(oscillator(detune: detune), g: g), Signal.constant(0.1))
  }

  private func lossValue(
    _ build: (Signal, Signal) -> Signal,
    parameter: Float,
    detune: Float = 0.65
  ) throws -> Float {
    LazyGraphContext.reset()
    return try build(Signal.constant(parameter), Signal.constant(detune))
      .realize(frames: frames).reduce(0, +)
  }

  private func finiteDifference(
    _ build: (Signal, Signal) -> Signal,
    parameter: Float,
    epsilon: Float
  ) throws -> Float {
    let plus = try lossValue(build, parameter: parameter + epsilon)
    let minus = try lossValue(build, parameter: parameter - epsilon)
    return (plus - minus) / (2.0 * epsilon)
  }

  private func relativeError(_ actual: Float, _ expected: Float) -> Float {
    abs(actual - expected) / max(abs(expected), 1e-9)
  }

  func testTrainablePhasorComposesWithStatelessGainGradient() throws {
    let initial: Float = 0.7
    let epsilon: Float = 1e-3
    LazyGraphContext.reset()
    let gain = Signal.param(initial)
    let frequencyDetune = Signal.param(detune)
    let loss = statelessLoss(gain: gain, detune: frequencyDetune)
    _ = try loss.backward(frames: frames)

    let fd = try finiteDifference(statelessLoss, parameter: initial, epsilon: epsilon)
    let autograd = try XCTUnwrap(gain.grad?.data)
    XCTAssertLessThan(relativeError(autograd, fd), 0.02)
  }

  func testOnePoleGradientWithConstantPhasorFrequency() throws {
    let initial: Float = 0.08
    let epsilon: Float = 1e-3
    LazyGraphContext.reset()
    let coefficient = Signal.param(initial)
    let loss = onePoleLoss(coefficient: coefficient, detune: Signal.constant(detune))
    _ = try loss.backward(frames: frames)

    let fd = try finiteDifference(onePoleLoss, parameter: initial, epsilon: epsilon)
    let autograd = try XCTUnwrap(coefficient.grad?.data)
    XCTAssertLessThan(relativeError(autograd, fd), 0.02)
  }

  func testTrainablePhasorComposesWithOnePoleGradient() throws {
    let initial: Float = 0.08
    let epsilon: Float = 1e-3
    LazyGraphContext.reset()
    let coefficient = Signal.param(initial)
    let frequencyDetune = Signal.param(detune)
    let loss = onePoleLoss(coefficient: coefficient, detune: frequencyDetune)
    _ = try loss.backward(frames: frames)

    let fd = try finiteDifference(onePoleLoss, parameter: initial, epsilon: epsilon)
    let autograd = try XCTUnwrap(coefficient.grad?.data)
    let error = relativeError(autograd, fd)

    XCTAssertLessThan(error, 0.02, "autograd=\(autograd), fd=\(fd), relError=\(error)")
  }

  func testTrainablePhasorComposesWithTwoCascadedHistoryGradients() throws {
    let initial: Float = 0.08
    let epsilon: Float = 1e-3
    LazyGraphContext.reset()
    let coefficient = Signal.param(initial)
    let frequencyDetune = Signal.param(detune)
    let loss = twoPoleLoss(coefficient: coefficient, detune: frequencyDetune)
    _ = try loss.backward(frames: frames)

    let fd = try finiteDifference(twoPoleLoss, parameter: initial, epsilon: epsilon)
    let autograd = try XCTUnwrap(coefficient.grad?.data)
    XCTAssertLessThan(
      relativeError(autograd, fd), 0.02, "autograd=\(autograd), fd=\(fd)")
  }

  func testCoupledSVFGradientWithConstantPhasorFrequency() throws {
    let initial: Float = 300
    let epsilon: Float = 1
    LazyGraphContext.reset()
    let cutoff = Signal.param(initial)
    let loss = svfLoss(cutoff: cutoff, detune: Signal.constant(detune))
    _ = try loss.backward(frames: frames)

    let fd = try finiteDifference(svfLoss, parameter: initial, epsilon: epsilon)
    let autograd = try XCTUnwrap(cutoff.grad?.data)
    XCTAssertLessThan(
      relativeError(autograd, fd), 0.02, "autograd=\(autograd), fd=\(fd)")
  }

  func testTrainablePhasorComposesWithCoupledSVFGradient() throws {
    let initial: Float = 300
    let epsilon: Float = 1
    LazyGraphContext.reset()
    let cutoff = Signal.param(initial)
    let frequencyDetune = Signal.param(detune)
    let loss = svfLoss(cutoff: cutoff, detune: frequencyDetune)
    _ = try loss.backward(frames: frames)

    let fd = try finiteDifference(svfLoss, parameter: initial, epsilon: epsilon)
    let autograd = try XCTUnwrap(cutoff.grad?.data)
    let error = relativeError(autograd, fd)
    XCTExpectFailure(
      "A phasor suffix-scan adjoint corrupts an unrelated coupled-history BPTT gradient") {
        XCTAssertLessThan(error, 0.02, "autograd=\(autograd), fd=\(fd), relError=\(error)")
      }
  }
}

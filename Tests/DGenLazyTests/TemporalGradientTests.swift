import DGen
import XCTest

@testable import DGenLazy

/// Finite-difference coverage for stateful operators whose current input affects
/// future outputs. These checks fail with a local or frame-index-only derivative.
final class TemporalGradientTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100
    DGenConfig.maxFrameCount = 256
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.maxFrameCount = 4096
    DGenConfig.kernelOutputPath = nil
    super.tearDown()
  }

  func testSweptPhasorGradientMatchesFiniteDifference() throws {
    let frames = 128
    let slope: Float = 1200
    let epsilon: Float = 1

    for backend in [Backend.metal, Backend.c] {
      DGenConfig.backend = backend
      func evaluate(_ slopeValue: Float) throws -> (loss: Float, grad: Float?) {
        LazyGraphContext.reset()
        let slopeParam = Signal.param(slopeValue)
        let time = Signal.accum(
          Signal.constant(1.0 / DGenConfig.sampleRate),
          reset: 0,
          min: 0,
          max: 1)
        let frequency = Signal.constant(80) + slopeParam * time
        let phase = Signal.statefulPhasor(frequency)
        let loss = mse(phase, Signal.constant(0.08))
        let total = try loss.backward(frames: frames).reduce(0, +)
        return (total, slopeParam.grad?.data)
      }

      let center = try evaluate(slope)
      let plus = try evaluate(slope + epsilon).loss
      let minus = try evaluate(slope - epsilon).loss
      let finiteDifference = (plus - minus) / (2 * epsilon)

      guard let gradient = center.grad else {
        XCTFail("Missing swept-phasor gradient for backend=\(backend)")
        continue
      }
      XCTAssertEqual(
        gradient,
        finiteDifference,
        accuracy: max(abs(finiteDifference) * 0.02, 1e-7),
        "backend=\(backend)")
    }
  }

  func testAccumulatorGradientMatchesFiniteDifferenceAcrossResetBoundary() throws {
    let frames = 128
    let increment: Float = 0.001
    let epsilon: Float = 1e-5

    for backend in [Backend.metal, Backend.c] {
      DGenConfig.backend = backend
      func evaluate(_ incrementValue: Float) throws -> (loss: Float, grad: Float?) {
        LazyGraphContext.reset()
        let incrementParam = Signal.param(incrementValue)
        let time = Signal.accum(
          Signal.constant(1.0 / DGenConfig.sampleRate),
          reset: 0,
          min: 0,
          max: 1)
        // Once the midpoint is crossed, every later frame resets. This makes the
        // expected adjoint stop at a known temporal boundary.
        let reset = gswitch(
          time > Signal.constant(Float(frames / 2) / DGenConfig.sampleRate),
          Signal.constant(1),
          Signal.constant(0))
        let modulation = 1.0 + Signal.phasor(12.0) * 0.25
        let accumulated = Signal.accum(
          incrementParam * modulation,
          reset: reset,
          min: Signal.constant(0),
          max: Signal.constant(10))
        let loss = mse(accumulated, Signal.constant(0.02))
        let total = try loss.backward(frames: frames).reduce(0, +)
        return (total, incrementParam.grad?.data)
      }

      let center = try evaluate(increment)
      let plus = try evaluate(increment + epsilon).loss
      let minus = try evaluate(increment - epsilon).loss
      let finiteDifference = (plus - minus) / (2 * epsilon)

      guard let gradient = center.grad else {
        XCTFail("Missing accumulator gradient for backend=\(backend)")
        continue
      }
      XCTAssertEqual(
        gradient,
        finiteDifference,
        accuracy: max(abs(finiteDifference) * 0.02, 1e-4),
        "backend=\(backend)")
    }
  }
}

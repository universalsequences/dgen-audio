import XCTest

@testable import DGenLazy

final class SpectralHopGradientTests: XCTestCase {
  private let frameCount = 256
  private let windowSize = 64
  private let sampleRate: Float = 2000.0

  private func configure() {
    LazyGraphContext.reset()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = frameCount
    DGenConfig.debug = false
  }

  private func spectralLossSignal(
    studentFreq: Signal,
    targetFreq: Signal,
    hop: Int
  ) -> Signal {
    let twoPi = Float.pi * 2.0
    let student = sin(Signal.phasor(studentFreq) * twoPi)
    let teacher = sin(Signal.phasor(targetFreq) * twoPi)
    return spectralLossFFT(student, teacher, windowSize: windowSize, hop: hop, normalize: true)
  }

  private func lossAtFrequency(
    studentFreq: Float,
    targetFreq: Float,
    hop: Int
  ) throws -> Float {
    configure()
    let loss = spectralLossSignal(
      studentFreq: Signal.constant(studentFreq),
      targetFreq: Signal.constant(targetFreq),
      hop: hop
    )
    let values = try loss.backward(frames: frameCount)
    return values.reduce(0, +) / Float(max(1, values.count))
  }

  private func analyticalGradientAtFrequency(
    studentFreq: Float,
    targetFreq: Float,
    hop: Int
  ) throws -> Float {
    configure()
    let student = Signal.param(studentFreq)
    let loss = spectralLossSignal(
      studentFreq: student,
      targetFreq: Signal.constant(targetFreq),
      hop: hop
    )
    _ = try loss.backward(frames: frameCount)
    return student.grad?.data ?? 0.0
  }

  func testSpectralLossHopForwardFiniteAndPositive() throws {
    for hop in [1, 2, 4, 8, 16] {
      let loss = try lossAtFrequency(studentFreq: 120.0, targetFreq: 260.0, hop: hop)
      XCTAssertTrue(loss.isFinite, "Loss should be finite for hop=\(hop), got \(loss)")
      XCTAssertGreaterThan(loss, 0.0, "Loss should be > 0 for different frequencies, hop=\(hop)")
    }
  }

  func testSpectralLossHopGradientDirectionMatchesFiniteDifferenceSmallHops() throws {
    let startFreq: Float = 140.0
    let targetFreq: Float = 240.0
    let epsilon: Float = 2.0

    for hop in [1, 2, 4] {
      let analytical = try analyticalGradientAtFrequency(
        studentFreq: startFreq,
        targetFreq: targetFreq,
        hop: hop
      )
      let lossPlus = try lossAtFrequency(
        studentFreq: startFreq + epsilon,
        targetFreq: targetFreq,
        hop: hop
      )
      let lossMinus = try lossAtFrequency(
        studentFreq: startFreq - epsilon,
        targetFreq: targetFreq,
        hop: hop
      )
      let numerical = (lossPlus - lossMinus) / (2.0 * epsilon)

      XCTAssertTrue(analytical.isFinite, "Analytical grad must be finite for hop=\(hop)")
      XCTAssertTrue(numerical.isFinite, "Numerical grad must be finite for hop=\(hop)")

      if abs(numerical) < 1e-7 {
        XCTAssertLessThan(
          abs(analytical),
          1e-4,
          "Near-zero numerical grad should imply tiny analytical grad for hop=\(hop)")
      } else {
        XCTAssertEqual(
          analytical.sign == numerical.sign,
          true,
          "Gradient direction mismatch for hop=\(hop): analytical=\(analytical), numerical=\(numerical)"
        )
      }
    }
  }

  func testSpectralLossHopGradientDirectionMatchesFiniteDifferenceLargeHops() throws {
    let startFreq: Float = 140.0
    let targetFreq: Float = 240.0
    let epsilon: Float = 2.0

    for hop in [8, 16] {
      let analytical = try analyticalGradientAtFrequency(
        studentFreq: startFreq,
        targetFreq: targetFreq,
        hop: hop
      )
      let lossPlus = try lossAtFrequency(
        studentFreq: startFreq + epsilon,
        targetFreq: targetFreq,
        hop: hop
      )
      let lossMinus = try lossAtFrequency(
        studentFreq: startFreq - epsilon,
        targetFreq: targetFreq,
        hop: hop
      )
      let numerical = (lossPlus - lossMinus) / (2.0 * epsilon)

      XCTAssertTrue(analytical.isFinite, "Analytical grad must be finite for hop=\(hop)")
      XCTAssertTrue(numerical.isFinite, "Numerical grad must be finite for hop=\(hop)")
      XCTAssertGreaterThan(abs(numerical), 1e-7, "Numerical grad should not vanish for hop=\(hop)")

      if abs(numerical) < 1e-7 {
        XCTAssertLessThan(
          abs(analytical),
          1e-4,
          "Near-zero numerical grad should imply tiny analytical grad for hop=\(hop)")
      } else {
        XCTAssertEqual(
          analytical.sign == numerical.sign,
          true,
          "Gradient direction mismatch for hop=\(hop): analytical=\(analytical), numerical=\(numerical)"
        )
      }
    }
  }
}

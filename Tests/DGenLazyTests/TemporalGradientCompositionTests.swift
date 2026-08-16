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
    XCTAssertLessThan(error, 0.02, "autograd=\(autograd), fd=\(fd), relError=\(error)")
  }

  // MARK: - Spectral (isolated-pass) composition — the Korg1 patch-learn shape
  //
  // With an inline MSE loss the backward recurrence shares its kernel with the
  // forward history loop. A spectral loss inserts isolated-pass kernels, so the
  // recurrence becomes a detached backward block — and a trainable phasor's
  // temporalGradStore/Scan/Read tape used to split that block mid-recurrence
  // (carry reads in one block, carry writes stranded after the scan),
  // truncating the SVF BPTT and corrupting every filter-parameter gradient.
  // L2 on linear magnitudes at 2 kHz keeps finite differences well-conditioned
  // (the trainer's L1/log modes share the same backward block structure).

  private let spectralSampleRate: Float = 2000.0

  private func spectralSVF(_ input: Signal, cutoff: Signal) -> Signal {
    svfLowpass(input, g: DGenLazy.tan(Signal.constant(Float.pi / spectralSampleRate) * cutoff))
  }

  private func spectralSVFLoss(cutoff: Signal, detune: Signal) -> Signal {
    let student = spectralSVF(oscillator(detune: detune), cutoff: cutoff)
    let teacher = spectralSVF(
      oscillator(detune: Signal.constant(-7.0)), cutoff: Signal.constant(500.0))
    return spectralLossFFT(
      student, teacher, windowSize: 128, useHannWindow: true,
      useLogMagnitude: false, lossMode: .l2, hop: 32, normalize: true)
  }

  /// Korg1 sandwich: a smooth envelope modulates BOTH the phasor frequency
  /// (temporal tape path) and the SVF cutoff (per-frame path inside the BPTT
  /// recurrence), so the envelope's gradient merges tape and recurrence
  /// contributions.
  private func envModulatedSpectralLoss(cutoff: Signal, pitchEnvAmount: Signal) -> Signal {
    let t = Signal.accum(
      Signal.constant(1.0 / spectralSampleRate), reset: 0.0, min: 0.0, max: 1000.0)
    let env = DGenLazy.exp(t * -3.0)
    let osc = DGenLazy.sin(
      Signal.statefulPhasor(55.0 + env * pitchEnvAmount) * (2.0 * Float.pi))
    let student = spectralSVF(osc, cutoff: cutoff + env * 40.0)
    let teacher = spectralSVF(
      oscillator(detune: Signal.constant(-7.0)), cutoff: Signal.constant(500.0))
    return spectralLossFFT(
      student, teacher, windowSize: 128, useHannWindow: true,
      useLogMagnitude: false, lossMode: .l2, hop: 32, normalize: true)
  }

  func testTrainablePhasorComposesWithSVFSpectralGradient() throws {
    DGenConfig.sampleRate = spectralSampleRate
    let initialCutoff: Float = 300
    LazyGraphContext.reset()
    let cutoff = Signal.param(initialCutoff)
    let frequencyDetune = Signal.param(detune)
    let loss = spectralSVFLoss(cutoff: cutoff, detune: frequencyDetune)
    _ = try loss.backward(frames: frames)

    let cutoffFd = try finiteDifference(
      spectralSVFLoss, parameter: initialCutoff, epsilon: 1.0)
    let cutoffAuto = try XCTUnwrap(cutoff.grad?.data)
    XCTAssertLessThan(
      relativeError(cutoffAuto, cutoffFd), 0.02,
      "cutoff: autograd=\(cutoffAuto), fd=\(cutoffFd)")

    func detuneLoss(_ d: Signal, _ unused: Signal) -> Signal {
      spectralSVFLoss(cutoff: Signal.constant(initialCutoff), detune: d)
    }
    let detuneFd = try finiteDifference(detuneLoss, parameter: detune, epsilon: 1e-2)
    let detuneAuto = try XCTUnwrap(frequencyDetune.grad?.data)
    XCTAssertLessThan(
      relativeError(detuneAuto, detuneFd), 0.05,
      "detune: autograd=\(detuneAuto), fd=\(detuneFd)")
  }

  func testEnvModulatedPhasorAndCutoffComposeWithSVFSpectralGradient() throws {
    DGenConfig.sampleRate = spectralSampleRate
    let initialCutoff: Float = 300
    let initialAmount: Float = 12.0
    LazyGraphContext.reset()
    let cutoff = Signal.param(initialCutoff)
    let amount = Signal.param(initialAmount)
    let loss = envModulatedSpectralLoss(cutoff: cutoff, pitchEnvAmount: amount)
    _ = try loss.backward(frames: frames)

    func cutoffLoss(_ c: Signal, _ unused: Signal) -> Signal {
      envModulatedSpectralLoss(cutoff: c, pitchEnvAmount: Signal.constant(initialAmount))
    }
    let cutoffFd = try finiteDifference(cutoffLoss, parameter: initialCutoff, epsilon: 1.0)
    let cutoffAuto = try XCTUnwrap(cutoff.grad?.data)
    XCTAssertLessThan(
      relativeError(cutoffAuto, cutoffFd), 0.02,
      "cutoff: autograd=\(cutoffAuto), fd=\(cutoffFd)")

    func amountLoss(_ a: Signal, _ unused: Signal) -> Signal {
      envModulatedSpectralLoss(cutoff: Signal.constant(initialCutoff), pitchEnvAmount: a)
    }
    let amountFd = try finiteDifference(amountLoss, parameter: initialAmount, epsilon: 1e-2)
    let amountAuto = try XCTUnwrap(amount.grad?.data)
    XCTAssertLessThan(
      relativeError(amountAuto, amountFd), 0.05,
      "pitchEnvAmount: autograd=\(amountAuto), fd=\(amountFd)")
  }
}

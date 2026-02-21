import XCTest

@testable import DGen
@testable import DGenLazy

/// Tests for batched spectral loss: [B]-shaped SignalTensors → scalar loss → [B] gradients.
final class BatchedSpectralLossTests: XCTestCase {
  private let sampleRate: Float = 2000.0

  private func configure(frames: Int) {
    LazyGraphContext.reset()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = frames
    DGenConfig.debug = false
  }

  // MARK: - Forward: batched loss == mean of individual losses

  /// The core correctness test: batched spectral loss on B signals should produce
  /// the same result as computing B individual scalar spectral losses and averaging them.
  func testBatchedLossMatchesMeanOfScalarLosses() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0
    let batchSize = 3
    // Use well-separated frequencies for clear spectral differences
    let studentFreqs: [Float] = [100, 200, 300]
    let teacherFreqs: [Float] = [120, 220, 320]

    // Compute B individual scalar losses
    var scalarLosses: [Float] = []
    for i in 0..<batchSize {
      configure(frames: frameCount)
      let student = sin(Signal.phasor(Signal.constant(studentFreqs[i])) * twoPi)
      let teacher = sin(Signal.phasor(Signal.constant(teacherFreqs[i])) * twoPi)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
      let values = try loss.backward(frames: frameCount)
      let avg = values.reduce(0, +) / Float(values.count)
      scalarLosses.append(avg)
    }
    let expectedMean = scalarLosses.reduce(0, +) / Float(batchSize)

    // Compute batched loss
    configure(frames: frameCount)
    let freqsStudent = Tensor(studentFreqs)
    let freqsTeacher = Tensor(teacherFreqs)
    let phases_s = SignalTensor.phasor(freqsStudent)
    let phases_t = SignalTensor.phasor(freqsTeacher)
    let student_b = sin(phases_s * twoPi)
    let teacher_b = sin(phases_t * twoPi)
    let batchedLoss = spectralLossFFT(student_b, teacher_b, windowSize: windowSize)
    let batchedValues = try batchedLoss.backward(frames: frameCount)
    let batchedAvg = batchedValues.reduce(0, +) / Float(batchedValues.count)

    // They should be close (not exact due to mean vs sum-then-divide ordering)
    XCTAssertEqual(
      Double(batchedAvg), Double(expectedMean), accuracy: Double(abs(expectedMean) * 0.05 + 1e-4),
      "Batched loss (\(batchedAvg)) should match mean of scalar losses (\(expectedMean))")
  }

  func testSpectralLossL1ModeIsSelectableAndProducesDifferentValue() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    func evaluate(_ mode: SpectralLossMode) throws -> Float {
      configure(frames: frameCount)
      let student = sin(SignalTensor.phasor(Tensor([110.0, 220.0])) * twoPi)
      let teacher = sin(SignalTensor.phasor(Tensor([180.0, 260.0])) * twoPi)
      let loss = spectralLossFFT(
        student, teacher, windowSize: windowSize, lossMode: mode, normalize: true)
      let values = try loss.backward(frames: frameCount)
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let l2 = try evaluate(.l2)
    let l1 = try evaluate(.l1)

    XCTAssertGreaterThan(l2, 0.0, "L2 spectral loss should be positive, got \(l2)")
    XCTAssertGreaterThan(l1, 0.0, "L1 spectral loss should be positive, got \(l1)")
    XCTAssertNotEqual(
      Double(l1), Double(l2), accuracy: 1e-8,
      "L1 and L2 spectral modes should produce different values on this input")
  }

  func testSpectralLossL1ModeBackpropagatesToParams() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)
    let ampParam = Tensor.param([2], data: [0.6, 0.4])
    let freqs = Tensor([150.0, 310.0])
    let student = sin(SignalTensor.phasor(freqs) * twoPi) * ampParam
    let teacher = sin(SignalTensor.phasor(freqs) * twoPi)
    let loss = spectralLossFFT(
      student, teacher, windowSize: windowSize, lossMode: .l1, normalize: true)
    _ = try loss.backward(frames: frameCount)

    guard let grad = ampParam.grad?.getData() else {
      XCTFail("Expected gradients for ampParam with L1 spectral loss")
      return
    }
    XCTAssertEqual(grad.count, 2, "Expected two gradients for ampParam")
    XCTAssertTrue(grad.allSatisfy { !$0.isNaN && !$0.isInfinite }, "Gradients must be finite")
    XCTAssertTrue(grad.contains { abs($0) > 1e-9 }, "Expected at least one non-zero gradient")
  }

  // MARK: - Determinism

  func testBatchedSpectralLossIsDeterministic() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    func evaluate() throws -> Float {
      configure(frames: frameCount)
      let freqs_s = Tensor([140.0, 280.0])
      let freqs_t = Tensor([200.0, 350.0])
      let student = sin(SignalTensor.phasor(freqs_s) * twoPi)
      let teacher = sin(SignalTensor.phasor(freqs_t) * twoPi)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
      let values = try loss.backward(frames: frameCount)
      return values.reduce(0, +) / Float(values.count)
    }

    let r1 = try evaluate()
    let r2 = try evaluate()
    let r3 = try evaluate()

    XCTAssertEqual(r1, r2, "Batched spectral loss run 1 vs 2: \(r1) vs \(r2)")
    XCTAssertEqual(r2, r3, "Batched spectral loss run 2 vs 3: \(r2) vs \(r3)")
  }

  // MARK: - Gradient flow

  /// Verify that gradients from batched spectral loss flow through to parameters.
  /// Checks that parameter gradients are non-zero for each batch element.
  func testBatchedSpectralLossProducesNonZeroGradients() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)
    // Learnable amplitude scaling per batch element
    let ampParam = Tensor.param([2], data: [0.8, 0.6])
    let freqs_s = Tensor([150.0, 300.0])
    let freqs_t = Tensor([200.0, 400.0])

    let student = sin(SignalTensor.phasor(freqs_s) * twoPi) * ampParam
    let teacher = sin(SignalTensor.phasor(freqs_t) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: windowSize, normalize: true)
    let _ = try loss.backward(frames: frameCount)

    // Gradients should flow back to the amplitude parameter
    let grad = ampParam.grad?.getData()
    XCTAssertNotNil(grad, "Amplitude parameter should have gradients")
    if let grad = grad {
      XCTAssertEqual(grad.count, 2, "Should have 2 gradient values (one per batch)")
      for i in 0..<grad.count {
        XCTAssertFalse(grad[i].isNaN, "Gradient[\(i)] should not be NaN")
        XCTAssertFalse(grad[i].isInfinite, "Gradient[\(i)] should not be infinite")
        XCTAssertNotEqual(
          grad[i], 0.0, accuracy: 1e-10,
          "Gradient[\(i)] should be non-zero, got \(grad[i])")
      }
    }
  }

  // MARK: - Non-zero loss

  /// Identical signals should produce zero (or near-zero) loss.
  func testBatchedIdenticalSignalsGiveZeroLoss() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)
    let freqs = Tensor([200.0, 400.0])
    let phases = SignalTensor.phasor(freqs)
    let signal = sin(phases * twoPi)
    // Same signal as both student and teacher
    let loss = spectralLossFFT(signal, signal, windowSize: windowSize)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(values.count)

    XCTAssertEqual(
      Double(avg), 0.0, accuracy: 0.01,
      "Identical batched signals should give near-zero loss, got \(avg)")
  }

  /// Different signals should produce non-zero loss.
  func testBatchedDifferentSignalsGiveNonZeroLoss() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)
    let student = sin(SignalTensor.phasor(Tensor([100.0, 200.0])) * twoPi)
    let teacher = sin(SignalTensor.phasor(Tensor([300.0, 500.0])) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(values.count)

    XCTAssertGreaterThan(
      avg, 0.1,
      "Different batched signals should give non-zero loss, got \(avg)")
  }

  // MARK: - Batch size variations

  func testBatchSize1MatchesScalar() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0
    let studentFreq: Float = 150.0
    let teacherFreq: Float = 250.0

    // Scalar version
    configure(frames: frameCount)
    let scalarStudent = sin(Signal.phasor(Signal.constant(studentFreq)) * twoPi)
    let scalarTeacher = sin(Signal.phasor(Signal.constant(teacherFreq)) * twoPi)
    let scalarLoss = spectralLossFFT(scalarStudent, scalarTeacher, windowSize: windowSize)
    let scalarValues = try scalarLoss.backward(frames: frameCount)
    let scalarAvg = scalarValues.reduce(0, +) / Float(scalarValues.count)

    // Batched version with B=1
    configure(frames: frameCount)
    let batchStudent = sin(SignalTensor.phasor(Tensor([studentFreq])) * twoPi)
    let batchTeacher = sin(SignalTensor.phasor(Tensor([teacherFreq])) * twoPi)
    let batchLoss = spectralLossFFT(batchStudent, batchTeacher, windowSize: windowSize)
    let batchValues = try batchLoss.backward(frames: frameCount)
    let batchAvg = batchValues.reduce(0, +) / Float(batchValues.count)

    XCTAssertEqual(
      Double(batchAvg), Double(scalarAvg), accuracy: Double(abs(scalarAvg) * 0.05 + 1e-4),
      "B=1 batched loss (\(batchAvg)) should match scalar loss (\(scalarAvg))")
  }

  func testBatchSize4Works() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)
    let student = sin(SignalTensor.phasor(Tensor([100.0, 200.0, 300.0, 400.0])) * twoPi)
    let teacher = sin(SignalTensor.phasor(Tensor([150.0, 250.0, 350.0, 450.0])) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(values.count)

    XCTAssertGreaterThan(avg, 0.0, "B=4 batched spectral loss should be positive, got \(avg)")
    XCTAssertFalse(avg.isNaN, "Loss should not be NaN")
    XCTAssertFalse(avg.isInfinite, "Loss should not be infinite")
  }

  // MARK: - Gradient descent training

  /// Verify the scalar (non-batched) spectral loss correctly trains amplitude.
  /// This serves as a baseline to diagnose whether issues are in the batched path.
  func testScalarSpectralLossAmplitudeTraining() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)

    let ampParam = Signal.param(0.5, min: 0.01, max: 2.0)
    let freq: Float = 200.0
    let optimizer = Adam(params: [ampParam], lr: 0.01)

    // Warmup
    do {
      let student = sin(Signal.phasor(freq) * twoPi) * ampParam
      let teacher = sin(Signal.phasor(freq) * twoPi)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
      _ = try loss.backward(frames: frameCount)
      optimizer.zeroGrad()
    }

    var losses: [Float] = []
    for _ in 0..<20 {
      let student = sin(Signal.phasor(freq) * twoPi) * ampParam
      let teacher = sin(Signal.phasor(freq) * twoPi)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
      let values = try loss.backward(frames: frameCount)
      losses.append(values.reduce(0, +) / Float(values.count))
      optimizer.step()
      optimizer.zeroGrad()
    }

    XCTAssertGreaterThan(
      losses.first!, losses.last!,
      "Scalar loss should decrease: first=\(losses.first!), last=\(losses.last!)")
  }

  /// Train learnable amplitude parameters via batched spectral loss with Adam optimizer.
  /// Verifies that loss decreases over training epochs.
  func testBatchedSpectralLossGradientDescentReducesLoss() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0
    let batchSize = 2

    configure(frames: frameCount)

    // Learnable amplitude per batch element, starting off from target (1.0)
    let ampParam = Tensor.param([batchSize], data: [0.5, 0.5])
    ampParam.minBound = 0.01
    ampParam.maxBound = 2.0

    // Fixed frequencies — same for student and teacher so only amplitude differs
    let freqs = Tensor([150.0, 350.0])

    let optimizer = Adam(params: [ampParam], lr: 0.005)

    // Warmup (compiles kernels)
    do {
      let student = sin(SignalTensor.phasor(freqs) * twoPi) * ampParam
      let teacher = sin(SignalTensor.phasor(freqs) * twoPi)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
      _ = try loss.backward(frames: frameCount)
      optimizer.zeroGrad()
    }

    var losses: [Float] = []
    let epochs = 20

    for _ in 0..<epochs {
      let student = sin(SignalTensor.phasor(freqs) * twoPi) * ampParam
      let teacher = sin(SignalTensor.phasor(freqs) * twoPi)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
      let values = try loss.backward(frames: frameCount)
      let epochLoss = values.reduce(0, +) / Float(values.count)
      losses.append(epochLoss)

      optimizer.step()
      optimizer.zeroGrad()
    }

    // Loss should decrease from first to last epoch
    XCTAssertGreaterThan(
      losses.first!, losses.last!,
      "Loss should decrease: first=\(losses.first!), last=\(losses.last!), all=\(losses)")
    // At least 30% reduction over 20 epochs
    XCTAssertLessThan(
      losses.last!, losses.first! * 0.7,
      "Loss should decrease by at least 30%: first=\(losses.first!), last=\(losses.last!)")
  }

  // MARK: - Batched harmonic synth → batched spectral loss

  /// Core batch-synth pipeline: [B, K] frequencies → statefulPhasor → sin → sumAxis → [B] signal
  /// → batched spectral loss. This is the critical path for batched DDSP training.
  func testBatchedHarmonicSynthProducesCorrectShapes() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0
    let B = 2
    let K = 3

    configure(frames: frameCount)

    // [B, K] frequencies: batch 0 = harmonics of 100Hz, batch 1 = harmonics of 150Hz
    let freqs = Tensor([
      [100.0, 200.0, 300.0],
      [150.0, 300.0, 450.0],
    ])  // [B=2, K=3]

    // Build student with learnable [B, K] amplitudes
    let ampParam = Tensor.param(
      [B, K],
      data: [
        0.5, 0.3, 0.2,
        0.4, 0.6, 0.1,
      ])

    let phases = SignalTensor.phasor(freqs)  // [B, K]
    let sines = sin(phases * twoPi)  // [B, K]
    let weighted = sines * ampParam  // [B, K]
    let student = weighted.sum(axis: 1)  // [B] — sum harmonics

    // Teacher: same freqs, different amps
    let teacherAmps = Tensor([
      [1.0, 0.5, 0.25],
      [0.8, 0.4, 0.2],
    ])  // [B=2, K=3]
    let teacherSines = sin(SignalTensor.phasor(freqs) * twoPi) * teacherAmps
    let teacher = teacherSines.sum(axis: 1)  // [B]

    let loss = spectralLossFFT(student, teacher, windowSize: windowSize)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(values.count)

    XCTAssertGreaterThan(avg, 0.0, "Loss should be positive, got \(avg)")
    XCTAssertFalse(avg.isNaN, "Loss should not be NaN")

    let grad = ampParam.grad?.getData()
    XCTAssertNotNil(grad, "Should have gradients")
    if let grad = grad {
      XCTAssertEqual(grad.count, B * K, "Should have B*K=\(B*K) gradient values, got \(grad.count)")
      for i in 0..<grad.count {
        XCTAssertFalse(grad[i].isNaN, "Gradient[\(i)] should not be NaN")
        XCTAssertFalse(grad[i].isInfinite, "Gradient[\(i)] should not be infinite")
      }
    }
  }

  /// Train [B, K] amplitude parameters via batched harmonic synth + spectral loss.
  func testBatchedHarmonicSynthTrainingReducesLoss() throws {
    let frameCount = 256
    let windowSize = 64
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)

    // Create 2D tensors directly (not via reshape) so they survive graph clear
    let freqs = Tensor([
      [100.0, 200.0, 300.0],
      [150.0, 300.0, 450.0],
    ])  // [B=2, K=3]

    // Learnable amps start at 0.5 uniform, target is non-uniform
    let ampParam = Tensor.param(
      [2, 3],
      data: [
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
      ])
    ampParam.minBound = 0.01
    ampParam.maxBound = 2.0

    let teacherAmps = Tensor([
      [1.0, 0.5, 0.25],
      [0.8, 0.4, 0.2],
    ])  // [B=2, K=3]

    let optimizer = Adam(params: [ampParam], lr: 0.005)

    // Warmup
    do {
      let student = sin(SignalTensor.phasor(freqs) * twoPi) * ampParam
      let teacher = sin(SignalTensor.phasor(freqs) * twoPi) * teacherAmps
      let loss = spectralLossFFT(
        student.sum(axis: 1), teacher.sum(axis: 1), windowSize: windowSize)
      _ = try loss.backward(frames: frameCount)
      optimizer.zeroGrad()
    }

    var losses: [Float] = []
    for _ in 0..<20 {
      let student = sin(SignalTensor.phasor(freqs) * twoPi) * ampParam
      let teacher = sin(SignalTensor.phasor(freqs) * twoPi) * teacherAmps

      // batched spectral loss
      let loss = spectralLossFFT(
        student.sum(axis: 1), teacher.sum(axis: 1), windowSize: windowSize)
      let values = try loss.backward(frames: frameCount)
      losses.append(values.reduce(0, +) / Float(values.count))
      optimizer.step()
      optimizer.zeroGrad()
    }

    XCTAssertGreaterThan(
      losses.first!, losses.last!,
      "Loss should decrease: first=\(losses.first!), last=\(losses.last!)")
  }

  // MARK: - Hop support

  func testBatchedWithHop() throws {
    let frameCount = 256
    let windowSize = 64
    let hop = 4
    let twoPi = Float.pi * 2.0

    configure(frames: frameCount)
    let student = sin(SignalTensor.phasor(Tensor([100.0, 200.0])) * twoPi)
    let teacher = sin(SignalTensor.phasor(Tensor([300.0, 400.0])) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: windowSize, hop: hop)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(values.count)

    XCTAssertGreaterThan(avg, 0.0, "Batched spectral loss with hop should be positive, got \(avg)")
    XCTAssertFalse(avg.isNaN, "Loss should not be NaN with hop")
  }
}

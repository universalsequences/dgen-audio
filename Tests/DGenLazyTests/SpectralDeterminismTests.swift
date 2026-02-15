import XCTest

@testable import DGenLazy

/// Tests that spectral loss evaluation is deterministic —
/// identical parameters must produce identical loss values across runs.
final class SpectralDeterminismTests: XCTestCase {
  private let frameCount = 256
  private let sampleRate: Float = 2000.0

  private func configure(frames: Int? = nil) {
    LazyGraphContext.reset()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = frames ?? frameCount
    DGenConfig.debug = false
  }

  // MARK: - Helpers

  /// Evaluate spectral loss with pure sine waves (no noise).
  private func spectralLoss(
    studentFreq: Float,
    targetFreq: Float,
    windowSize: Int,
    hop: Int
  ) throws -> (loss: Float, perFrame: [Float]) {
    configure()
    let twoPi = Float.pi * 2.0
    let student = sin(Signal.phasor(Signal.constant(studentFreq)) * twoPi)
    let teacher = sin(Signal.phasor(Signal.constant(targetFreq)) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: windowSize, hop: hop, normalize: true)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(max(1, values.count))
    return (avg, values)
  }

  /// Evaluate MSE loss between two sine waves.
  private func mseLoss(
    studentFreq: Float,
    targetFreq: Float
  ) throws -> (loss: Float, perFrame: [Float]) {
    configure()
    let twoPi = Float.pi * 2.0
    let student = sin(Signal.phasor(Signal.constant(studentFreq)) * twoPi)
    let teacher = sin(Signal.phasor(Signal.constant(targetFreq)) * twoPi)
    let loss = mse(student, teacher)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(max(1, values.count))
    return (avg, values)
  }

  /// Evaluate spectral loss with noise added to the student signal.
  private func spectralLossWithNoise(
    studentFreq: Float,
    targetFreq: Float,
    windowSize: Int,
    hop: Int,
    noiseGain: Float
  ) throws -> (loss: Float, perFrame: [Float]) {
    configure()
    let twoPi = Float.pi * 2.0
    let student = sin(Signal.phasor(Signal.constant(studentFreq)) * twoPi)
      + Signal.noise() * Signal.constant(noiseGain)
    let teacher = sin(Signal.phasor(Signal.constant(targetFreq)) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: windowSize, hop: hop, normalize: true)
    let values = try loss.backward(frames: frameCount)
    let avg = values.reduce(0, +) / Float(max(1, values.count))
    return (avg, values)
  }

  // MARK: - Basic determinism tests

  func testMSEIsDeterministic() throws {
    let result1 = try mseLoss(studentFreq: 140, targetFreq: 240)
    let result2 = try mseLoss(studentFreq: 140, targetFreq: 240)

    XCTAssertEqual(result1.loss, result2.loss,
      "MSE loss should be identical across runs: \(result1.loss) vs \(result2.loss)")
    for i in 0..<min(result1.perFrame.count, result2.perFrame.count) {
      XCTAssertEqual(result1.perFrame[i], result2.perFrame[i],
        "MSE per-frame[\(i)] mismatch: \(result1.perFrame[i]) vs \(result2.perFrame[i])")
    }
  }

  func testSpectralLossHop1IsDeterministic() throws {
    let result1 = try spectralLoss(studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 1)
    let result2 = try spectralLoss(studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 1)

    XCTAssertEqual(result1.loss, result2.loss,
      "Spectral loss (hop=1) should be identical: \(result1.loss) vs \(result2.loss)")
  }

  func testSpectralLossHop4IsDeterministic() throws {
    let result1 = try spectralLoss(studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 4)
    let result2 = try spectralLoss(studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 4)

    XCTAssertEqual(result1.loss, result2.loss,
      "Spectral loss (hop=4) should be identical: \(result1.loss) vs \(result2.loss)")
  }

  func testSpectralLossHop8IsDeterministic() throws {
    let result1 = try spectralLoss(studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 8)
    let result2 = try spectralLoss(studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 8)

    XCTAssertEqual(result1.loss, result2.loss,
      "Spectral loss (hop=8) should be identical: \(result1.loss) vs \(result2.loss)")
  }

  func testSpectralLossWithNoiseIsDeterministic() throws {
    let result1 = try spectralLossWithNoise(
      studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 1, noiseGain: 0.1)
    let result2 = try spectralLossWithNoise(
      studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 1, noiseGain: 0.1)

    XCTAssertEqual(result1.loss, result2.loss,
      "Spectral loss with noise should be identical: \(result1.loss) vs \(result2.loss)")
  }

  func testSpectralLossWithNoiseHop4IsDeterministic() throws {
    let result1 = try spectralLossWithNoise(
      studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 4, noiseGain: 0.1)
    let result2 = try spectralLossWithNoise(
      studentFreq: 140, targetFreq: 240, windowSize: 64, hop: 4, noiseGain: 0.1)

    XCTAssertEqual(result1.loss, result2.loss,
      "Spectral+noise loss (hop=4) should be identical: \(result1.loss) vs \(result2.loss)")
  }

  func testSpectralLossThreeRunsIdentical() throws {
    let r1 = try spectralLoss(studentFreq: 200, targetFreq: 300, windowSize: 64, hop: 4)
    let r2 = try spectralLoss(studentFreq: 200, targetFreq: 300, windowSize: 64, hop: 4)
    let r3 = try spectralLoss(studentFreq: 200, targetFreq: 300, windowSize: 64, hop: 4)

    XCTAssertEqual(r1.loss, r2.loss, "Run 1 vs 2 mismatch: \(r1.loss) vs \(r2.loss)")
    XCTAssertEqual(r2.loss, r3.loss, "Run 2 vs 3 mismatch: \(r2.loss) vs \(r3.loss)")
  }

  // MARK: - DDSP pipeline determinism
  // These mirror the actual training pipeline: matmul → peek → statefulPhasor → harmonic sum → spectral loss

  /// Minimal DDSP-like pipeline: matmul MLP, peek, statefulPhasor, harmonic sum, spectral loss.
  private func ddspPipelineLoss(
    f0Frames: [Float],
    uvFrames: [Float],
    targetAudio: [Float],
    weightData: [Float],
    windowSize: Int,
    hop: Int,
    numHarmonics: Int,
    frames: Int
  ) throws -> (loss: Float, perFrame: [Float]) {
    configure(frames: frames)

    let featureFrames = f0Frames.count
    let featureMaxIndex = Float(max(0, featureFrames - 1))
    let frameDenom = Float(max(1, frames - 1))
    let playheadStep = featureMaxIndex / frameDenom

    // Playhead: linear sweep through feature frames
    let playheadRaw = Signal.accum(
      Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex)
    let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

    // Feature lookups
    let f0Tensor = Tensor(f0Frames)
    let uvTensor = Tensor(uvFrames)
    let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
    let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

    // Simple "model": fixed weight tensor controls harmonic amps
    let ampTensor = Tensor.param([1, numHarmonics], data: weightData)
    let amps = sigmoid(ampTensor)
    let ampsRow = amps.peekRow(Signal.constant(0.0))

    // Harmonic synthesis with statefulPhasor
    let twoPi = Float.pi * 2.0
    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
    let harmonicFreqs = harmonicIndices * f0
    let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
    let harmonicSines = sin(harmonicPhases * twoPi)
    let harmonic = (harmonicSines * ampsRow).sum() * uv

    // Target
    let target = Tensor(targetAudio).toSignal(maxFrames: frames)

    // Spectral loss
    let loss = spectralLossFFT(harmonic, target, windowSize: windowSize, hop: hop, normalize: true)
    let values = try loss.backward(frames: frames)
    let avg = values.reduce(0, +) / Float(max(1, values.count))
    return (avg, values)
  }

  func testDDSPPipelineSpectralIsDeterministic() throws {
    let numHarmonics = 8
    let featureFrames = 16
    let frames = 512

    // Fixed f0 and uv
    let f0Frames = [Float](repeating: 220.0, count: featureFrames)
    let uvFrames = [Float](repeating: 1.0, count: featureFrames)

    // Fixed target (sine wave at 220 Hz)
    let targetAudio = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sampleRate)
    }

    // Fixed "model weights"
    let weights = (0..<numHarmonics).map { i in Float(i) * 0.1 - 0.3 }

    let r1 = try ddspPipelineLoss(
      f0Frames: f0Frames, uvFrames: uvFrames, targetAudio: targetAudio,
      weightData: weights, windowSize: 64, hop: 4, numHarmonics: numHarmonics, frames: frames)
    let r2 = try ddspPipelineLoss(
      f0Frames: f0Frames, uvFrames: uvFrames, targetAudio: targetAudio,
      weightData: weights, windowSize: 64, hop: 4, numHarmonics: numHarmonics, frames: frames)

    XCTAssertEqual(r1.loss, r2.loss,
      "DDSP pipeline spectral loss should be deterministic: \(r1.loss) vs \(r2.loss)")
  }

  func testDDSPPipelineSpectralLargerScale() throws {
    let numHarmonics = 16
    let featureFrames = 32
    let frames = 2048

    let f0Frames = (0..<featureFrames).map { _ in Float.random(in: 100...400) }
    let uvFrames = [Float](repeating: 1.0, count: featureFrames)
    let targetAudio = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sampleRate)
    }
    let weights = (0..<numHarmonics).map { _ in Float.random(in: -0.5...0.5) }

    let r1 = try ddspPipelineLoss(
      f0Frames: f0Frames, uvFrames: uvFrames, targetAudio: targetAudio,
      weightData: weights, windowSize: 64, hop: 8, numHarmonics: numHarmonics, frames: frames)
    let r2 = try ddspPipelineLoss(
      f0Frames: f0Frames, uvFrames: uvFrames, targetAudio: targetAudio,
      weightData: weights, windowSize: 64, hop: 8, numHarmonics: numHarmonics, frames: frames)

    XCTAssertEqual(r1.loss, r2.loss,
      "DDSP pipeline (larger scale) should be deterministic: \(r1.loss) vs \(r2.loss)")
  }

  // MARK: - Training lifecycle determinism (clearComputationGraph pattern)
  // These tests mimic the actual training loop: tensors are pre-allocated ONCE,
  // data is injected via updateDataLazily, and backward() clears the graph
  // (no LazyGraphContext.reset() between evaluations).

  func testSpectralLossTrainingLifecycleIsDeterministic() throws {
    // Setup: single reset, then pre-allocate tensors
    configure()

    let twoPi = Float.pi * 2.0
    let studentFreq: Float = 140
    let targetFreq: Float = 240

    // Evaluate spectral loss — backward() internally calls clearComputationGraph()
    let student1 = sin(Signal.phasor(Signal.constant(studentFreq)) * twoPi)
    let teacher1 = sin(Signal.phasor(Signal.constant(targetFreq)) * twoPi)
    let loss1 = spectralLossFFT(student1, teacher1, windowSize: 64, hop: 4, normalize: true)
    let values1 = try loss1.backward(frames: frameCount)
    let avg1 = values1.reduce(0, +) / Float(max(1, values1.count))

    // Second evaluation — graph was cleared by backward(), rebuild on same graph context
    // NO reset() here, matching the training loop pattern
    let student2 = sin(Signal.phasor(Signal.constant(studentFreq)) * twoPi)
    let teacher2 = sin(Signal.phasor(Signal.constant(targetFreq)) * twoPi)
    let loss2 = spectralLossFFT(student2, teacher2, windowSize: 64, hop: 4, normalize: true)
    let values2 = try loss2.backward(frames: frameCount)
    let avg2 = values2.reduce(0, +) / Float(max(1, values2.count))

    XCTAssertEqual(avg1, avg2,
      "Spectral loss should be identical across clearComputationGraph cycles: \(avg1) vs \(avg2)")
  }

  func testDDSPPipelineTrainingLifecycleIsDeterministic() throws {
    let numHarmonics = 8
    let featureFrames = 16
    let frames = 512

    let f0Data = [Float](repeating: 220.0, count: featureFrames)
    let uvData = [Float](repeating: 1.0, count: featureFrames)
    let targetData = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sampleRate)
    }
    let weights = (0..<numHarmonics).map { i in Float(i) * 0.1 - 0.3 }

    // Single reset, then pre-allocate ALL tensors before the "training loop"
    configure(frames: frames)
    let f0Tensor = Tensor(f0Data)
    let uvTensor = Tensor(uvData)
    let targetTensor = Tensor(targetData)
    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
    let ampTensor = Tensor.param([1, numHarmonics], data: weights)

    /// Build and evaluate the DDSP pipeline
    func evaluate() throws -> Float {
      let featureMaxIndex = Float(max(0, featureFrames - 1))
      let frameDenom = Float(max(1, frames - 1))
      let playheadStep = featureMaxIndex / frameDenom
      let playheadRaw = Signal.accum(
        Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex)
      let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

      let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
      let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

      let amps = sigmoid(ampTensor)
      let ampsRow = amps.peekRow(Signal.constant(0.0))
      let twoPi = Float.pi * 2.0
      let harmonicFreqs = harmonicIndices * f0
      let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
      let harmonicSines = sin(harmonicPhases * twoPi)
      let harmonic = (harmonicSines * ampsRow).sum() * uv

      let target = targetTensor.toSignal(maxFrames: frames)
      let loss = spectralLossFFT(harmonic, target, windowSize: 64, hop: 4, normalize: true)
      let values = try loss.backward(frames: frames)
      // backward() calls clearComputationGraph() internally
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let r1 = try evaluate()
    // Inject SAME data via updateDataLazily (mimics training loop data injection)
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r2 = try evaluate()

    XCTAssertEqual(r1, r2,
      "DDSP pipeline should be deterministic across clearComputationGraph cycles: \(r1) vs \(r2)")
  }

  func testDDSPPipelineTrainingLifecycleThreeRuns() throws {
    let numHarmonics = 8
    let featureFrames = 16
    let frames = 512

    let f0Data = [Float](repeating: 220.0, count: featureFrames)
    let uvData = [Float](repeating: 1.0, count: featureFrames)
    let targetData = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sampleRate)
    }
    let weights = (0..<numHarmonics).map { i in Float(i) * 0.1 - 0.3 }

    configure(frames: frames)
    let f0Tensor = Tensor(f0Data)
    let uvTensor = Tensor(uvData)
    let targetTensor = Tensor(targetData)
    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
    let ampTensor = Tensor.param([1, numHarmonics], data: weights)

    func evaluate() throws -> Float {
      let featureMaxIndex = Float(max(0, featureFrames - 1))
      let frameDenom = Float(max(1, frames - 1))
      let playheadStep = featureMaxIndex / frameDenom
      let playheadRaw = Signal.accum(
        Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex)
      let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

      let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
      let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

      let amps = sigmoid(ampTensor)
      let ampsRow = amps.peekRow(Signal.constant(0.0))
      let twoPi = Float.pi * 2.0
      let harmonicFreqs = harmonicIndices * f0
      let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
      let harmonicSines = sin(harmonicPhases * twoPi)
      let harmonic = (harmonicSines * ampsRow).sum() * uv

      let target = targetTensor.toSignal(maxFrames: frames)
      let loss = spectralLossFFT(harmonic, target, windowSize: 64, hop: 4, normalize: true)
      let values = try loss.backward(frames: frames)
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let r1 = try evaluate()
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r2 = try evaluate()
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r3 = try evaluate()

    XCTAssertEqual(r1, r2, "Run 1 vs 2: \(r1) vs \(r2)")
    XCTAssertEqual(r2, r3, "Run 2 vs 3: \(r2) vs \(r3)")
  }

  // MARK: - Large-scale determinism (training-scale)
  // These test at the same scale as the DDSPE2E training pipeline

  func testSpectralLossLargeWindowIsDeterministic() throws {
    let frames = 2048
    configure(frames: frames)

    let twoPi = Float.pi * 2.0
    let student = sin(Signal.phasor(Signal.constant(140.0)) * twoPi)
    let teacher = sin(Signal.phasor(Signal.constant(240.0)) * twoPi)
    let loss = spectralLossFFT(student, teacher, windowSize: 512, hop: 128, normalize: true)
    let values1 = try loss.backward(frames: frames)
    let avg1 = values1.reduce(0, +) / Float(max(1, values1.count))

    // Second eval on same graph context (no reset)
    let student2 = sin(Signal.phasor(Signal.constant(140.0)) * twoPi)
    let teacher2 = sin(Signal.phasor(Signal.constant(240.0)) * twoPi)
    let loss2 = spectralLossFFT(student2, teacher2, windowSize: 512, hop: 128, normalize: true)
    let values2 = try loss2.backward(frames: frames)
    let avg2 = values2.reduce(0, +) / Float(max(1, values2.count))

    XCTAssertEqual(avg1, avg2,
      "Spectral loss (w=512 hop=128) should be deterministic at 2048 frames: \(avg1) vs \(avg2)")
  }

  func testMultiWindowSpectralLossIsDeterministic() throws {
    let frames = 4096
    configure(frames: frames)

    func evaluate() throws -> Float {
      let twoPi = Float.pi * 2.0
      let student = sin(Signal.phasor(Signal.constant(140.0)) * twoPi)
      let teacher = sin(Signal.phasor(Signal.constant(240.0)) * twoPi)
      let s1 = spectralLossFFT(student, teacher, windowSize: 512, hop: 128, normalize: true)
      let s2 = spectralLossFFT(student, teacher, windowSize: 1024, hop: 256, normalize: true)
      let loss = (s1 + s2) * 0.5
      let values = try loss.backward(frames: frames)
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let r1 = try evaluate()
    let r2 = try evaluate()
    let r3 = try evaluate()

    XCTAssertEqual(r1, r2, "Multi-window spectral run 1 vs 2: \(r1) vs \(r2)")
    XCTAssertEqual(r2, r3, "Multi-window spectral run 2 vs 3: \(r2) vs \(r3)")
  }

  func testDDSPPipelineLargeScaleTrainingLifecycle() throws {
    let numHarmonics = 64
    let featureFrames = 64
    let frames = 16384

    let f0Data = [Float](repeating: 220.0, count: featureFrames)
    let uvData = [Float](repeating: 1.0, count: featureFrames)
    let targetData = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sampleRate)
    }
    let weights = (0..<numHarmonics).map { i in Float(i) * 0.01 - 0.3 }

    configure(frames: frames)
    let f0Tensor = Tensor(f0Data)
    let uvTensor = Tensor(uvData)
    let targetTensor = Tensor(targetData)
    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
    let ampTensor = Tensor.param([1, numHarmonics], data: weights)

    func evaluate() throws -> Float {
      let featureMaxIndex = Float(max(0, featureFrames - 1))
      let frameDenom = Float(max(1, frames - 1))
      let playheadStep = featureMaxIndex / frameDenom
      let playheadRaw = Signal.accum(
        Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex)
      let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

      let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
      let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

      let amps = sigmoid(ampTensor)
      let ampsRow = amps.peekRow(Signal.constant(0.0))
      let twoPi = Float.pi * 2.0
      let harmonicFreqs = harmonicIndices * f0
      let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
      let harmonicSines = sin(harmonicPhases * twoPi)
      let harmonic = (harmonicSines * ampsRow).sum() * uv

      let target = targetTensor.toSignal(maxFrames: frames)
      let s1 = spectralLossFFT(harmonic, target, windowSize: 512, hop: 128, normalize: true)
      let s2 = spectralLossFFT(harmonic, target, windowSize: 1024, hop: 256, normalize: true)
      let loss = (s1 + s2) * 0.5
      let values = try loss.backward(frames: frames)
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let r1 = try evaluate()
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r2 = try evaluate()
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r3 = try evaluate()

    XCTAssertEqual(r1, r2, "Large-scale DDSP run 1 vs 2: \(r1) vs \(r2)")
    XCTAssertEqual(r2, r3, "Large-scale DDSP run 2 vs 3: \(r2) vs \(r3)")
  }

  func testFullMLPDDSPPipelineTrainingLifecycle() throws {
    let numHarmonics = 64
    let hiddenSize = 64
    let featureFrames = 64
    let frames = 16384
    let sr: Float = 16000.0

    // Fixed conditioning data (f0Norm, loudNorm, uv) as flat [featureFrames * 3]
    let condData: [Float] = (0..<featureFrames).flatMap { i -> [Float] in
      let f0Norm = Float(log2(220.0 / 440.0))
      let loudNorm: Float = 0.5
      let uv: Float = 1.0
      return [f0Norm, loudNorm, uv]
    }
    let f0Data = [Float](repeating: 220.0, count: featureFrames)
    let uvData = [Float](repeating: 1.0, count: featureFrames)
    let targetData = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sr)
    }

    // Setup once
    LazyGraphContext.reset()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sr
    DGenConfig.maxFrameCount = frames
    DGenConfig.debug = false

    // Pre-allocate data tensors — initialize with ACTUAL data so r1 uses same input
    let featuresTensor = Tensor(
      (0..<featureFrames).map { _ -> [Float] in
        [Float(log2(220.0 / 440.0)), 0.5, 1.0]
      }
    )
    let f0Tensor = Tensor(f0Data)
    let uvTensor = Tensor(uvData)
    let targetTensor = Tensor(targetData)
    let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })

    // MLP model params — deterministic initialization
    let inputSize = 3
    func seededWeights(count: Int, scale: Float, offset: Int) -> [Float] {
      (0..<count).map { i in
        let r = Float((i + offset) % 997) / 997.0
        return (r * 2.0 - 1.0) * scale
      }
    }
    let W1 = Tensor.param([inputSize, hiddenSize],
      data: seededWeights(count: inputSize*hiddenSize, scale: 0.08, offset: 0))
    let b1 = Tensor.param([1, hiddenSize], data: [Float](repeating: 0, count: hiddenSize))
    let W_harm = Tensor.param([hiddenSize, numHarmonics],
      data: seededWeights(count: hiddenSize*numHarmonics, scale: 0.06, offset: 1000))
    let b_harm = Tensor.param([1, numHarmonics], data: [Float](repeating: 0, count: numHarmonics))

    let W_hgain = Tensor.param([hiddenSize, 1],
      data: seededWeights(count: hiddenSize, scale: 0.05, offset: 2000))
    let b_hgain = Tensor.param([1, 1], data: [0.0])
    let W_noise = Tensor.param([hiddenSize, 1],
      data: seededWeights(count: hiddenSize, scale: 0.05, offset: 3000))
    let b_noise = Tensor.param([1, 1], data: [0.0])

    func evaluate() throws -> Float {
      // Full MLP forward
      let hidden = tanh(featuresTensor.matmul(W1) + b1)
      let harmonicAmps = sigmoid(hidden.matmul(W_harm) + b_harm)
      let harmonicGain = sigmoid(hidden.matmul(W_hgain) + b_hgain)
      let noiseGain = sigmoid(hidden.matmul(W_noise) + b_noise)

      // Synth
      let featureMaxIndex = Float(max(0, featureFrames - 1))
      let frameDenom = Float(max(1, frames - 1))
      let playheadStep = featureMaxIndex / frameDenom
      let playheadRaw = Signal.accum(
        Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex)
      let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

      let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
      let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

      let ampsAtTime = harmonicAmps.peekRow(playhead)
      let twoPi = Float.pi * 2.0
      let harmonicFreqs = harmonicIndices * f0
      let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
      let harmonicSines = sin(harmonicPhases * twoPi)
      let harmonic = (harmonicSines * ampsAtTime).sum() * uv

      let hGain = harmonicGain.peek(playhead, channel: Signal.constant(0.0))
      let harmonicOut = harmonic * hGain * (1.0 / Float(max(1, numHarmonics)))
      _ = noiseGain.peek(playhead, channel: Signal.constant(0.0))

      let target = targetTensor.toSignal(maxFrames: frames)

      let s1 = spectralLossFFT(harmonicOut, target, windowSize: 512, hop: 128, normalize: true)
      let s2 = spectralLossFFT(harmonicOut, target, windowSize: 1024, hop: 256, normalize: true)
      let loss = (s1 + s2) * 0.5
      let values = try loss.backward(frames: frames)
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let r1 = try evaluate()
    // Inject same data (no change, same values)
    featuresTensor.updateDataLazily(condData)
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r2 = try evaluate()
    featuresTensor.updateDataLazily(condData)
    f0Tensor.updateDataLazily(f0Data)
    uvTensor.updateDataLazily(uvData)
    targetTensor.updateDataLazily(targetData)
    let r3 = try evaluate()

    XCTAssertEqual(r1, r2, "Full MLP DDSP run 1 vs 2: \(r1) vs \(r2)")
    XCTAssertEqual(r2, r3, "Full MLP DDSP run 2 vs 3: \(r2) vs \(r3)")
  }

  func testDDSPPipelineWithNoiseIsDeterministic() throws {
    let numHarmonics = 8
    let featureFrames = 16
    let frames = 512

    let f0Frames = [Float](repeating: 220.0, count: featureFrames)
    let uvFrames = [Float](repeating: 1.0, count: featureFrames)
    let targetAudio = (0..<frames).map { i in
      sinf(2.0 * Float.pi * 220.0 * Float(i) / sampleRate)
    }
    let weights = (0..<numHarmonics).map { i in Float(i) * 0.1 - 0.3 }

    // Build pipeline with noise
    func buildWithNoise() throws -> Float {
      configure(frames: frames)
      let featureMaxIndex = Float(max(0, featureFrames - 1))
      let frameDenom = Float(max(1, frames - 1))
      let playheadStep = featureMaxIndex / frameDenom
      let playheadRaw = Signal.accum(
        Signal.constant(playheadStep), reset: 0.0, min: 0.0, max: featureMaxIndex)
      let playhead = playheadRaw.clip(0.0, Double(max(0.0, featureMaxIndex - 1e-4)))

      let f0Tensor = Tensor(f0Frames)
      let uvTensor = Tensor(uvFrames)
      let f0 = min(max(f0Tensor.peek(playhead), 20.0), 500.0)
      let uv = uvTensor.peek(playhead).clip(0.0, 1.0)

      let ampTensor = Tensor.param([1, numHarmonics], data: weights)
      let amps = sigmoid(ampTensor)
      let ampsRow = amps.peekRow(Signal.constant(0.0))

      let twoPi = Float.pi * 2.0
      let harmonicIndices = Tensor((0..<numHarmonics).map { Float($0 + 1) })
      let harmonicFreqs = harmonicIndices * f0
      let harmonicPhases = Signal.statefulPhasor(harmonicFreqs)
      let harmonicSines = sin(harmonicPhases * twoPi)
      let harmonic = (harmonicSines * ampsRow).sum() * uv

      // Add noise branch
      let noiseOut = Signal.noise() * Signal.constant(0.1) * (1.0 - uv)
      let prediction = harmonic + noiseOut

      let target = Tensor(targetAudio).toSignal(maxFrames: frames)
      let loss = spectralLossFFT(prediction, target, windowSize: 64, hop: 4, normalize: true)
      let values = try loss.backward(frames: frames)
      return values.reduce(0, +) / Float(max(1, values.count))
    }

    let r1 = try buildWithNoise()
    let r2 = try buildWithNoise()

    XCTAssertEqual(r1, r2,
      "DDSP pipeline with noise should be deterministic: \(r1) vs \(r2)")
  }
}

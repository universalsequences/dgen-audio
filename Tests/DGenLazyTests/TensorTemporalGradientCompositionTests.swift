import DGen
import XCTest

@testable import DGenLazy

/// Tensor (batched, B=2) analogue of TemporalGradientCompositionTests.
///
/// A trainable tensor statefulPhasor frequency emits the three-phase suffix
/// scan adjoint (temporalGradStore → temporalGradScan → temporalGradRead) as
/// per-lane tensor ops. A coupled TensorHistory SVF downstream emits a
/// reverse-time BPTT recurrence. Each temporal path passes finite differences
/// on its own; these tests pin the composition, where the consolidated tensor
/// BPTT block must execute after the temporalGradRead that feeds it.
final class TensorTemporalGradientCompositionTests: XCTestCase {
  private let frames = 256
  private let lanes = 2
  private let baseFreqs: [Float] = [55.0, 63.0]
  private let baseCutoffs: [Float] = [300.0, 420.0]

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100
    DGenConfig.maxFrameCount = 512
    DGenGradientConfig.detachPhasorFrequency = false
    DGenGradientConfig.detachAccumInputs = false
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenGradientConfig.detachPhasorFrequency = false
    DGenConfig.backend = .metal
    DGenConfig.maxFrameCount = 4096
    super.tearDown()
  }

  // MARK: - Graph builders

  private func oscillator(freqs: DGenLazy.Tensor) -> SignalTensor {
    // Promote to SignalTensor: the batched trainer computes lane frequencies
    // from transformed z tensors, so the phasor sees a SignalTensor input.
    let freqLanes: SignalTensor = freqs * Signal.constant(1.0)
    return DGenLazy.sin(SignalTensor.statefulPhasor(freqLanes) * (2.0 * Float.pi))
  }

  /// Coupled two-cell ZDF SVF lowpass over B lanes (Lisp-style: two
  /// TensorHistory cells with pass-through writes).
  private func tensorSVFLowpass(_ input: SignalTensor, g: SignalTensor) -> SignalTensor {
    let damping: Float = 1.0 / 1.2
    let gPlusDamp = g + damping
    let denom = (g * gPlusDamp) + 1.0
    let a1 = 1.0 / denom
    let a2 = g * a1
    let ic1 = TensorHistory(shape: [lanes], requiresGrad: true)
    let ic2 = TensorHistory(shape: [lanes], requiresGrad: true)
    let ic1Read = ic1.read()
    let ic2Read = ic2.read()
    let v3 = input - ic2Read
    let v1 = a1 * ic1Read + a2 * v3
    let ic1New = ic1.write(v1 * 2.0 - ic1Read)
    let v1PassThrough = (ic1New + ic1Read) * 0.5
    let v2 = ic2Read + g * v1PassThrough
    let ic2New = ic2.write(v2 * 2.0 - ic2Read)
    return (ic2New + ic2Read) * 0.5
  }

  private func gFromCutoffs(
    _ cutoffs: DGenLazy.Tensor, sampleRate: Float = 44100.0
  ) -> SignalTensor {
    DGenLazy.tan(cutoffs * Signal.constant(Float.pi / sampleRate))
  }

  /// Per-lane squared error against a constant, summed over lanes.
  private func batchedLoss(_ output: SignalTensor) -> Signal {
    let diff = output - 0.1
    return (diff * diff).sum() / Float(frames)
  }

  private func svfLoss(freqs: DGenLazy.Tensor, cutoffs: DGenLazy.Tensor) -> Signal {
    batchedLoss(tensorSVFLowpass(oscillator(freqs: freqs), g: gFromCutoffs(cutoffs)))
  }

  private func gainLoss(freqs: DGenLazy.Tensor, gains: DGenLazy.Tensor) -> Signal {
    let gainLanes: SignalTensor = gains * Signal.constant(1.0)
    let scaled = oscillator(freqs: freqs) * gainLanes
    return batchedLoss(scaled)
  }

  /// The trainer's loss shape: batched spectral loss against a fixed teacher.
  /// Run at a low sample rate so the 128-sample window resolves the target
  /// frequencies (resolution = sampleRate / windowSize; see CLAUDE.md).
  private let spectralSampleRate: Float = 2000.0

  private func spectralSVFLoss(freqs: DGenLazy.Tensor, cutoffs: DGenLazy.Tensor) -> Signal {
    let student = tensorSVFLowpass(
      oscillator(freqs: freqs), g: gFromCutoffs(cutoffs, sampleRate: spectralSampleRate))
    let teacherFreqs = Tensor([49.0, 71.0])
    let teacherCutoffs = Tensor([500.0, 350.0])
    let teacher = tensorSVFLowpass(
      oscillator(freqs: teacherFreqs),
      g: gFromCutoffs(teacherCutoffs, sampleRate: spectralSampleRate))
    // L2 on linear magnitudes: the smooth mode, so finite differences are
    // well-conditioned. The trainer's L1/log modes share the same backward
    // block structure.
    return spectralLossFFT(
      student, teacher, windowSize: 128, useHannWindow: true,
      useLogMagnitude: false, lossMode: .l2, hop: 32, normalize: true)
  }

  func testConstantPhasorFrequencyWithTensorSVFSpectralGradient() throws {
    DGenConfig.sampleRate = spectralSampleRate
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs)
    let cutoffs = Tensor(baseCutoffs, requiresGrad: true)
    let loss = spectralSVFLoss(freqs: freqs, cutoffs: cutoffs)
    _ = try loss.backward(frames: frames)
    let grads = try XCTUnwrap(cutoffs.grad?.getData())

    for lane in 0..<lanes {
      let fd = try finiteDifference(
        spectralSVFLoss, a: baseFreqs, b: baseCutoffs, vary: 1, lane: lane, epsilon: 1.0)
      let error = relativeError(grads[lane], fd)
      XCTAssertLessThan(
        error, 0.05, "lane \(lane): autograd=\(grads[lane]), fd=\(fd), relError=\(error)")
    }
  }

  /// Sandwich shape (the Korg1 failure): the freq param feeds both the phasor
  /// (temporal tape path) and the cutoff (per-frame path inside the SVF BPTT
  /// recurrence). The grad contributions merge in an add that is a carry-read
  /// descendant AND a temporalGradRead consumer — it must be evicted from the
  /// consolidated reverse loop, since the read only exists after it.
  private func entangledSVFLoss(freqs: DGenLazy.Tensor, cutoffs: DGenLazy.Tensor) -> Signal {
    let osc = oscillator(freqs: freqs)
    let freqLanes: SignalTensor = freqs * Signal.constant(1.0)
    let cutoffLanes: SignalTensor = (cutoffs * Signal.constant(1.0)) + freqLanes * 2.0
    let g = DGenLazy.tan(cutoffLanes * Signal.constant(Float.pi / 44100.0))
    return batchedLoss(tensorSVFLowpass(osc, g: g))
  }

  func testFreqFeedingBothPhasorAndCutoffComposes() throws {
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs, requiresGrad: true)
    let cutoffs = Tensor(baseCutoffs, requiresGrad: true)
    let loss = entangledSVFLoss(freqs: freqs, cutoffs: cutoffs)
    _ = try loss.backward(frames: frames)
    let freqGrads = try XCTUnwrap(freqs.grad?.getData())
    let cutoffGrads = try XCTUnwrap(cutoffs.grad?.getData())

    for lane in 0..<lanes {
      let cutoffFd = try finiteDifference(
        entangledSVFLoss, a: baseFreqs, b: baseCutoffs, vary: 1, lane: lane, epsilon: 1.0)
      XCTAssertLessThan(
        relativeError(cutoffGrads[lane], cutoffFd), 0.02,
        "cutoff lane \(lane): autograd=\(cutoffGrads[lane]), fd=\(cutoffFd)")

      let freqFd = try finiteDifference(
        entangledSVFLoss, a: baseFreqs, b: baseCutoffs, vary: 0, lane: lane, epsilon: 1e-2)
      XCTAssertLessThan(
        relativeError(freqGrads[lane], freqFd), 0.05,
        "freq lane \(lane): autograd=\(freqGrads[lane]), fd=\(freqFd)")
    }
  }

  // MARK: - Helpers

  private func lossValue(_ build: (DGenLazy.Tensor, DGenLazy.Tensor) -> Signal, a: [Float], b: [Float]) throws
    -> Float
  {
    LazyGraphContext.reset()
    let loss = build(Tensor(a), Tensor(b))
    return try loss.realize(frames: frames).reduce(0, +)
  }

  /// Central finite difference of the summed loss w.r.t. one lane of one input.
  private func finiteDifference(
    _ build: (DGenLazy.Tensor, DGenLazy.Tensor) -> Signal, a: [Float], b: [Float],
    vary: Int, lane: Int, epsilon: Float
  ) throws -> Float {
    func perturbed(_ values: [Float], by delta: Float) -> [Float] {
      var out = values
      out[lane] += delta
      return out
    }
    let plus = try lossValue(
      build,
      a: vary == 0 ? perturbed(a, by: epsilon) : a,
      b: vary == 1 ? perturbed(b, by: epsilon) : b)
    let minus = try lossValue(
      build,
      a: vary == 0 ? perturbed(a, by: -epsilon) : a,
      b: vary == 1 ? perturbed(b, by: -epsilon) : b)
    let step: Float = 2.0 * epsilon
    let delta: Float = plus - minus
    return delta / step
  }

  private func relativeError(_ actual: Float, _ expected: Float) -> Float {
    abs(actual - expected) / max(abs(expected), 1e-9)
  }

  // MARK: - Controls

  func testConstantPhasorFrequencyWithTensorSVFGradient() throws {
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs)
    let cutoffs = Tensor(baseCutoffs, requiresGrad: true)
    let loss = svfLoss(freqs: freqs, cutoffs: cutoffs)
    _ = try loss.backward(frames: frames)
    let grads = try XCTUnwrap(cutoffs.grad?.getData())

    for lane in 0..<lanes {
      let fd = try finiteDifference(
        svfLoss, a: baseFreqs, b: baseCutoffs, vary: 1, lane: lane, epsilon: 1.0)
      let error = relativeError(grads[lane], fd)
      XCTAssertLessThan(
        error, 0.02, "lane \(lane): autograd=\(grads[lane]), fd=\(fd), relError=\(error)")
    }
  }

  func testTrainableTensorPhasorWithStatelessGainGradient() throws {
    let baseGains: [Float] = [0.7, 0.5]
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs, requiresGrad: true)
    let gains = Tensor(baseGains, requiresGrad: true)
    let loss = gainLoss(freqs: freqs, gains: gains)
    _ = try loss.backward(frames: frames)
    let gainGrads = try XCTUnwrap(gains.grad?.getData())

    for lane in 0..<lanes {
      let fd = try finiteDifference(
        gainLoss, a: baseFreqs, b: baseGains, vary: 1, lane: lane, epsilon: 1e-3)
      let error = relativeError(gainGrads[lane], fd)
      XCTAssertLessThan(
        error, 0.02, "lane \(lane): autograd=\(gainGrads[lane]), fd=\(fd), relError=\(error)")
    }
  }

  // MARK: - The composition under test

  func testTrainableTensorPhasorComposesWithTensorSVFGradient() throws {
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs, requiresGrad: true)
    let cutoffs = Tensor(baseCutoffs, requiresGrad: true)
    let loss = svfLoss(freqs: freqs, cutoffs: cutoffs)
    _ = try loss.backward(frames: frames)
    let cutoffGrads = try XCTUnwrap(cutoffs.grad?.getData())
    let freqGrads = try XCTUnwrap(freqs.grad?.getData())

    for lane in 0..<lanes {
      let fd = try finiteDifference(
        svfLoss, a: baseFreqs, b: baseCutoffs, vary: 1, lane: lane, epsilon: 1.0)
      let error = relativeError(cutoffGrads[lane], fd)
      XCTAssertLessThan(
        error, 0.02,
        "cutoff lane \(lane): autograd=\(cutoffGrads[lane]), fd=\(fd), relError=\(error)")

      let freqFd = try finiteDifference(
        svfLoss, a: baseFreqs, b: baseCutoffs, vary: 0, lane: lane, epsilon: 1e-2)
      let freqError = relativeError(freqGrads[lane], freqFd)
      XCTAssertLessThan(
        freqError, 0.05,
        "freq lane \(lane): autograd=\(freqGrads[lane]), fd=\(freqFd), relError=\(freqError)")
    }
  }

  func testTrainableTensorPhasorComposesWithTensorSVFSpectralGradient() throws {
    DGenConfig.sampleRate = spectralSampleRate
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs, requiresGrad: true)
    let cutoffs = Tensor(baseCutoffs, requiresGrad: true)
    let loss = spectralSVFLoss(freqs: freqs, cutoffs: cutoffs)
    _ = try loss.backward(frames: frames)
    let cutoffGrads = try XCTUnwrap(cutoffs.grad?.getData())
    let freqGrads = try XCTUnwrap(freqs.grad?.getData())

    for lane in 0..<lanes {
      let fd = try finiteDifference(
        spectralSVFLoss, a: baseFreqs, b: baseCutoffs, vary: 1, lane: lane, epsilon: 1.0)
      let error = relativeError(cutoffGrads[lane], fd)
      XCTAssertLessThan(
        error, 0.05,
        "cutoff lane \(lane): autograd=\(cutoffGrads[lane]), fd=\(fd), relError=\(error)")

      let freqFd = try finiteDifference(
        spectralSVFLoss, a: baseFreqs, b: baseCutoffs, vary: 0, lane: lane, epsilon: 1e-2)
      let freqError = relativeError(freqGrads[lane], freqFd)
      XCTAssertLessThan(
        freqError, 0.05,
        "freq lane \(lane): autograd=\(freqGrads[lane]), fd=\(freqFd), relError=\(freqError)")
    }
  }

  /// Shape-[1] tensor phasors share elementCount == 1 with scalar phasors but
  /// keep their grad in tensor storage; the store must take the tensor path.
  func testSingleLaneTensorPhasorFreqGradient() throws {
    func build(_ fr: Float, requiresGrad: Bool) -> (DGenLazy.Tensor, Signal) {
      let f = Tensor([fr], requiresGrad: requiresGrad)
      let fl: SignalTensor = f * Signal.constant(1.0)
      let osc = DGenLazy.sin(SignalTensor.statefulPhasor(fl) * (2.0 * Float.pi))
      let diff = osc - 0.1
      let sq = diff * diff
      return (f, sq.sum() / Float(frames))
    }
    LazyGraphContext.reset()
    let (f, loss) = build(55.0, requiresGrad: true)
    _ = try loss.backward(frames: frames)
    let grad = try XCTUnwrap(f.grad?.getData()?.first)

    func lv(_ fr: Float) throws -> Float {
      LazyGraphContext.reset()
      let (_, loss) = build(fr, requiresGrad: false)
      return try loss.realize(frames: frames).reduce(0, +)
    }
    let fd = (try lv(55.01) - (try lv(54.99))) / 0.02
    XCTAssertLessThan(
      relativeError(grad, fd), 0.02, "autograd=\(grad), fd=\(fd)")
  }

  /// A shared scalar accum clock (trainable rate) driving a per-frame cutoff
  /// envelope through the tensor SVF: the accum's three-phase adjoint must
  /// compose with the tensor-history BPTT the same way the phasor's does.
  func testSharedAccumClockComposesWithTensorSVFGradient() throws {
    func build(_ rateValue: Float, requiresGrad: Bool) -> (Signal, DGenLazy.Tensor, Signal) {
      let rate =
        requiresGrad ? Signal.param(rateValue) : Signal.constant(rateValue)
      let t = Signal.accum(
        rate * Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 100.0)
      let env = DGenLazy.exp(t * Signal.constant(-8.0))
      let cutoffs = Tensor(baseCutoffs, requiresGrad: requiresGrad)
      let cutoffLanes: SignalTensor = cutoffs * env
      let g = DGenLazy.tan(cutoffLanes * Signal.constant(Float.pi / 44100.0))
      let osc = oscillator(freqs: Tensor(baseFreqs))
      let loss = batchedLoss(tensorSVFLowpass(osc, g: g))
      return (rate, cutoffs, loss)
    }

    LazyGraphContext.reset()
    let (rate, cutoffs, loss) = build(1.0, requiresGrad: true)
    _ = try loss.backward(frames: frames)
    let rateGrad = try XCTUnwrap(rate.grad?.data)
    let cutoffGrads = try XCTUnwrap(cutoffs.grad?.getData())

    func lv(_ rateValue: Float) throws -> Float {
      LazyGraphContext.reset()
      let (_, _, loss) = build(rateValue, requiresGrad: false)
      return try loss.realize(frames: frames).reduce(0, +)
    }
    let eps: Float = 1e-3
    let fd = (try lv(1.0 + eps) - (try lv(1.0 - eps))) / (2 * eps)
    XCTAssertLessThan(
      relativeError(rateGrad, fd), 0.05, "rate: autograd=\(rateGrad), fd=\(fd)")
    for lane in 0..<lanes {
      XCTAssertNotEqual(cutoffGrads[lane], 0.0, "cutoff lane \(lane) grad is zero")
    }
  }

  /// Per-lane gradients in the B=2 batch must match two independent B=1 runs
  /// (no cross-lane gradient/state mixing).
  func testBatchedGradientsMatchSingleLaneRuns() throws {
    LazyGraphContext.reset()
    let freqs = Tensor(baseFreqs, requiresGrad: true)
    let cutoffs = Tensor(baseCutoffs, requiresGrad: true)
    let loss = svfLoss(freqs: freqs, cutoffs: cutoffs)
    _ = try loss.backward(frames: frames)
    let batchedCutoffGrads = try XCTUnwrap(cutoffs.grad?.getData())
    let batchedFreqGrads = try XCTUnwrap(freqs.grad?.getData())

    for lane in 0..<lanes {
      LazyGraphContext.reset()
      let laneFreq = Tensor([baseFreqs[lane]], requiresGrad: true)
      let laneCutoff = Tensor([baseCutoffs[lane]], requiresGrad: true)
      let laneFreqLanes: SignalTensor = laneFreq * Signal.constant(1.0)
      let osc = DGenLazy.sin(SignalTensor.statefulPhasor(laneFreqLanes) * (2.0 * Float.pi))
      let g = DGenLazy.tan(laneCutoff * Signal.constant(Float.pi / 44100.0))
      let damping: Float = 1.0 / 1.2
      let gPlusDamp = g + damping
      let denom = (g * gPlusDamp) + 1.0
      let a1 = 1.0 / denom
      let a2 = g * a1
      let ic1 = TensorHistory(shape: [1], requiresGrad: true)
      let ic2 = TensorHistory(shape: [1], requiresGrad: true)
      let ic1Read = ic1.read()
      let ic2Read = ic2.read()
      let v3 = osc - ic2Read
      let v1 = a1 * ic1Read + a2 * v3
      let ic1New = ic1.write(v1 * 2.0 - ic1Read)
      let v1PassThrough = (ic1New + ic1Read) * 0.5
      let v2 = ic2Read + g * v1PassThrough
      let ic2New = ic2.write(v2 * 2.0 - ic2Read)
      let out = (ic2New + ic2Read) * 0.5
      let diff = out - 0.1
      let squared = diff * diff
      let laneLoss = squared.sum() / Float(frames)
      _ = try laneLoss.backward(frames: frames)
      let laneCutoffGrad = try XCTUnwrap(laneCutoff.grad?.getData()?.first)
      let laneFreqGrad = try XCTUnwrap(laneFreq.grad?.getData()?.first)

      XCTAssertLessThan(
        relativeError(batchedCutoffGrads[lane], laneCutoffGrad), 5e-3,
        "cutoff lane \(lane): batched=\(batchedCutoffGrads[lane]), single=\(laneCutoffGrad)")
      XCTAssertLessThan(
        relativeError(batchedFreqGrads[lane], laneFreqGrad), 5e-3,
        "freq lane \(lane): batched=\(batchedFreqGrads[lane]), single=\(laneFreqGrad)")
    }
  }
}

import DGen
import Foundation
import XCTest

@testable import DGenLazy

/// R5 gate (docs/DDSP_REVIVAL_SPEC.md): the DDSP paper's learned-IR reverb on
/// the differentiable buffer -> tensorFFT -> per-bin multiply -> tensorIFFT ->
/// overlapAdd route.
///
/// Three layers of proof, cheapest first:
///  1. Forward exactness: `spectralConvolve` equals CPU direct convolution
///     delayed by `spectralConvolveLatency` — pins the block-tiling argument
///     in LearnedReverb.swift (no windowing, no circular aliasing).
///  2. Finite-difference gradient check on a small hop-gated config — this is
///     exactly the code region where hop-sliced adjoint bugs have lived.
///  3. Known-IR recovery: dry audio rendered through a fixed synthetic IR
///     (echoes + exponential decay); the IR is recovered as a trainable
///     parameter from the wet audio via multi-scale spectral loss.
final class LearnedReverbTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = 4096
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-8
    LazyGraphContext.reset()
    super.tearDown()
  }

  // MARK: - Helpers

  private func makeNoise(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(state >> 40) / Float(1 << 24) * 2.0 - 1.0
    }
  }

  /// Causal direct convolution, output length = x.count.
  private func directConvolve(_ x: [Float], _ h: [Float]) -> [Float] {
    var y = [Float](repeating: 0, count: x.count)
    for n in 0..<x.count {
      for l in 0..<Swift.min(h.count, n + 1) {
        y[n] += h[l] * x[n - l]
      }
    }
    return y
  }

  /// The wet target as `spectralConvolve` should produce it: direct
  /// convolution delayed by the operator's fixed latency.
  private func delayedWet(dry: [Float], ir: [Float], fftSize: Int) -> [Float] {
    let conv = directConvolve(dry, ir)
    let d = spectralConvolveLatency(fftSize: fftSize)
    var out = [Float](repeating: 0, count: dry.count)
    for n in d..<dry.count { out[n] = conv[n - d] }
    return out
  }

  /// Synthetic room-ish IR: unit tap, two discrete echoes, exponential tail.
  private func syntheticIR(length L: Int) -> [Float] {
    var ir = [Float](repeating: 0, count: L)
    ir[0] = 1.0
    if L > 13 { ir[13] += 0.6 }
    if L > 29 { ir[29] += -0.35 }
    for l in 1..<L {
      ir[l] += 0.5 * Foundation.exp(-Float(l) * 8.0 / Float(L)) * (l % 2 == 0 ? 1.0 : -0.7)
    }
    return ir
  }

  private func cosine(_ a: [Float], _ b: [Float]) -> Float {
    let dot = zip(a, b).map(*).reduce(0, +)
    let na = a.map { $0 * $0 }.reduce(0, +).squareRoot()
    let nb = b.map { $0 * $0 }.reduce(0, +).squareRoot()
    return dot / Swift.max(na * nb, 1e-12)
  }

  // MARK: - 1. Forward exactness

  func testForwardMatchesDirectConvolution() throws {
    let N = 64
    let hop = 16
    let L = 32  // hop + L - 1 = 47 <= 64
    let frameCount = 512
    DGenConfig.maxFrameCount = frameCount
    LazyGraphContext.reset()

    let dryData = makeNoise(count: frameCount, seed: 0xDD5)
    let irData = syntheticIR(length: L)

    let dry = Tensor(dryData).toSignal(maxFrames: frameCount)
    let ir = Tensor(irData)
    let wet = spectralConvolve(dry, ir: ir, fftSize: N, hop: hop)
    let result = try wet.realize(frames: frameCount)

    let expected = delayedWet(dry: dryData, ir: irData, fftSize: N)
    var maxDiff: Float = 0
    for n in 0..<frameCount { maxDiff = Swift.max(maxDiff, abs(result[n] - expected[n])) }
    print("spectralConvolve forward max |err| = \(maxDiff)")
    XCTAssertLessThan(
      maxDiff, 1e-3,
      "spectralConvolve should be exact linear convolution delayed by fftSize-1")
  }

  /// learnedReverb with a zero wet IR is a pure delay of the dry signal.
  func testLearnedReverbZeroIRIsDelayedIdentity() throws {
    let N = 64
    let hop = 16
    let L = 32
    let frameCount = 512
    DGenConfig.maxFrameCount = frameCount
    LazyGraphContext.reset()

    let dryData = makeNoise(count: frameCount, seed: 0xBEE5)
    let dry = Tensor(dryData).toSignal(maxFrames: frameCount)
    let ir = Tensor([Float](repeating: 0, count: L))
    let out = try learnedReverb(dry, ir: ir, fftSize: N, hop: hop).realize(frames: frameCount)

    let d = spectralConvolveLatency(fftSize: N)
    var maxDiff: Float = 0
    for n in d..<frameCount { maxDiff = Swift.max(maxDiff, abs(out[n] - dryData[n - d])) }
    print("learnedReverb zero-IR max |err| = \(maxDiff)")
    XCTAssertLessThan(maxDiff, 1e-3, "zero wet IR should reduce to a delayed dry path")
  }

  // MARK: - 2. Finite-difference gradient check

  func testGradientMatchesFiniteDifferencesHop1() throws {
    try runGradientCheck(hop: 1)
  }

  func testGradientMatchesFiniteDifferences() throws {
    try runGradientCheck(hop: 8)
  }

  private func runGradientCheck(hop: Int) throws {
    let N = 32
    let L = 8  // hop + L - 1 = 15 <= 32
    let frameCount = 256
    DGenConfig.sampleRate = 2000.0
    DGenConfig.maxFrameCount = frameCount
    defer { DGenConfig.sampleRate = 44100.0 }

    let dryData = makeNoise(count: frameCount, seed: 0xFD_2026)
    let trueIR: [Float] = [0.9, 0.0, -0.4, 0.0, 0.25, 0.1, -0.05, 0.3]
    let wetData = delayedWet(dry: dryData, ir: trueIR, fftSize: N)
    let start = [Float](repeating: 0.1, count: L)

    func loss(_ irData: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let ir = Tensor.param([L], data: irData)
      let dry = Tensor(dryData).toSignal(maxFrames: frameCount)
      let target = Tensor(wetData).toSignal(maxFrames: frameCount)
      let pred = spectralConvolve(dry, ir: ir, fftSize: N, hop: hop)
      let values = try mse(pred, target).backward(frames: frameCount)
      let mean = values.reduce(0, +) / Float(values.count)
      return (mean, ir.grad?.getData() ?? [])
    }

    let (_, auto) = try loss(start)
    XCTAssertEqual(auto.count, L, "expected a gradient per IR tap")

    // The loss is quadratic in the IR, so central differences are exact up to
    // float noise at any step size.
    let eps: Float = 1e-2
    var fd = [Float](repeating: 0, count: L)
    for i in 0..<L {
      var plus = start
      plus[i] += eps
      var minus = start
      minus[i] -= eps
      fd[i] = ((try loss(plus).0) - (try loss(minus).0)) / (2 * eps) * Float(frameCount)
    }

    let cos = cosine(auto, fd)
    let scale = Swift.max(fd.map { abs($0) }.max() ?? 1, 1e-6)
    let relErr = (0..<L).map { abs(auto[$0] - fd[$0]) / scale }.max() ?? 0
    print("reverb IR grad: cosine=\(cos) maxRelErr=\(relErr)\n  auto=\(auto)\n  fd=\(fd)")
    XCTAssertGreaterThan(cos, 0.999, "autograd should match finite differences in direction")
    XCTAssertLessThan(relErr, 1e-2, "autograd should match finite differences in magnitude")
  }

  // MARK: - 3. Known-IR recovery gate

  func testKnownIRRecoveryViaMultiScaleSpectralLoss() throws {
    let N = 256
    let hop = 64
    let L = 128  // hop + L - 1 = 191 <= 256
    let frameCount = 4096
    DGenConfig.sampleRate = 16_000.0
    DGenConfig.maxFrameCount = frameCount
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-3
    defer { DGenConfig.sampleRate = 44100.0 }
    LazyGraphContext.reset()

    // Dry excitation: broadband deterministic noise, so every bin the IR can
    // shape is actually excited.
    let dryData = makeNoise(count: frameCount, seed: 0x1234_5678)
    let trueIR = syntheticIR(length: L)
    let wetData = delayedWet(dry: dryData, ir: trueIR, fftSize: N)

    let ir = Tensor.param([L], data: [Float](repeating: 0.01, count: L))
    let optimizer = Adam(params: [ir], lr: 0.02)

    func buildLoss() -> Signal {
      let dry = Tensor(dryData).toSignal(maxFrames: frameCount)
      let target = Tensor(wetData).toSignal(maxFrames: frameCount)
      let pred = spectralConvolve(dry, ir: ir, fftSize: N, hop: hop)
      // Same loss family as DDSPE2E's fullLoss: multi-scale spectral (linear +
      // log-magnitude) plus a waveform MSE term. Spectral-only collapses the
      // loss (92x observed) but log-magnitudes are phase-blind, so the tap
      // signs/positions stay under-determined; the MSE term supplies phase.
      // (MSE alone converges to machine precision — the problem is convex —
      // so it is weighted up to stay visible next to the ~600-scale spectral
      // sum, mirroring how fullLoss balances mseWeight against spectral.)
      var loss = mse(pred, target) * 50.0
      for w in [512, 128, 32] {
        loss =
          loss
          + spectralLossFFT(pred, target, windowSize: w, hop: max(1, w / 4))
          + spectralLossFFT(
            pred, target, windowSize: w, useLogMagnitude: true, useSmoothLogMagnitude: true,
            hop: max(1, w / 4))
      }
      return loss
    }

    func lossValue(_ values: [Float]) -> Float {
      values.reduce(0, +) / Float(max(1, values.count))
    }

    let initialLoss = lossValue(try buildLoss().backward(frames: frameCount))
    optimizer.zeroGrad()

    var finalLoss = initialLoss
    let steps = 400
    for step in 0..<steps {
      let loss = buildLoss()
      finalLoss = lossValue(try loss.backward(frames: frameCount))
      optimizer.step()
      optimizer.zeroGrad()
      if step % 50 == 0 { print("step \(step): loss=\(finalLoss)") }
    }

    let recovered = ir.getData() ?? []
    XCTAssertEqual(recovered.count, L)
    let corr = cosine(recovered, trueIR)
    print("IR recovery: loss \(initialLoss) -> \(finalLoss) (\(initialLoss / max(finalLoss, 1e-12))x)")
    print("IR cosine similarity vs true IR: \(corr)")
    print("true IR head:      \(Array(trueIR.prefix(16)))")
    print("recovered IR head: \(Array(recovered.prefix(16)))")

    XCTAssertLessThan(
      finalLoss, initialLoss / 10.0,
      "reverb recovery loss should collapse >10x; got \(initialLoss) -> \(finalLoss)")
    XCTAssertGreaterThan(
      corr, 0.9,
      "recovered IR should correlate with the true IR; cosine=\(corr)")
  }
}

import XCTest

@testable import DGenLazy

/// Gradients through a **hop-gated** tensor region (anything downstream of
/// `buffer(size:hop:)` with `hop > 1`).
///
/// The graph below is about as small as an STFT filter gets — buffer, FFT,
/// multiply by a gain vector, sum — and it is entirely linear in the gain
/// vector, so the loss is an exact quadratic and central finite differences are
/// the true gradient with no truncation error at any step size.
///
/// Two regimes, and only one of them was the bug this file was written for:
///
///  - **A frame-rate operand multiplying into the hop-gated region** — a
///    `sampleRow` driven by a frame-rate index, which is what `spectralFilter`
///    and the DDSP frequency-sampled noise branch actually build. This was
///    wrong for every `hop > 1`; hop-sliced adjoint tapes were read at frame
///    rate with *hold* semantics, so each tick's adjoint was replayed `hop`
///    times into the frame-summing gradient reducers. Fixed — see
///    `Graph.frameAwareCellScatter` and the hop/operand matrix in
///    `HopGatedGradientFDMatrixTests`.
///  - **Every node scheduled at hop rate, the loss included** — reached by
///    indexing with `Signal.constant`, which leaves the graph with no
///    frame-rate node at all. The forward then emits the loss only on hop ticks
///    (its total is exactly `1/hop` of the frame-rate formulation's) and the
///    entire backward chain is hop-scheduled. That gradient is still wrong, and
///    it is a *separate* defect: it reproduces with a plain static
///    `Tensor.param` and no `sampleRow`, so it is not about frame-rate
///    operands. Pinned below.
final class HopGatedSpectralGradientTests: XCTestCase {
  private let frameCount = 256
  private let N = 16

  override func setUp() {
    super.setUp()
    // Other suites leave the C backend selected; these graphs are spectral
    // BPTT, which only the Metal backend compiles.
    DGenConfig.backend = .metal
    DGenConfig.maxFrameCount = frameCount
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
    super.tearDown()
  }

  private func makeNoise(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(state >> 40) / Float(1 << 24) * 2.0 - 1.0
    }
  }

  /// Returns (autograd gradient, finite-difference gradient) for
  /// `sum(FFT(buffer(noise, hop)) * gains)` scored against a fixed target gain
  /// vector. Both are in "sum over frames" units.
  ///
  /// `frameRateIndex` selects which of the two regimes above is exercised: a
  /// frame-rate `accum` (value pinned at 0, so the maths is identical) keeps
  /// the graph's consumers running every frame, while `Signal.constant` lets
  /// the whole graph — loss included — collapse to hop rate.
  private func gradients(hop: Int, frameRateIndex: Bool) throws -> (auto: [Float], fd: [Float]) {
    let noiseData = makeNoise(count: frameCount, seed: 99)
    let targetRow = (0..<N).map { c in 0.3 + Float(c % 3) * 0.2 }
    let start = [Float](repeating: 0.5, count: N)
    let width = N

    func loss(_ gains: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let learned = Tensor.param([1, width], data: gains)
      let noise = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let target = Tensor([targetRow])
      let index: Signal =
        frameRateIndex
        ? Signal.accum(Signal.constant(0.0), reset: 0.0, min: 0.0, max: 1.0)
        : Signal.constant(0.0)
      let frame = noise.buffer(size: width, hop: hop).reshape([width]) * hann(width)
      let (re, _) = signalTensorFFT(frame, N: width)
      func branch(_ m: Tensor) -> Signal { (re * m.sampleRow(index)).sum() }
      let values = try mse(branch(learned), branch(target)).backward(frames: frameCount)
      let mean = values.reduce(0, +) / Float(values.count)
      return (mean, learned.grad?.getData() ?? [])
    }

    let (_, auto) = try loss(start)
    var fd = [Float](repeating: 0, count: N)
    for i in 0..<N {
      var plus = start
      plus[i] += 1e-2
      var minus = start
      minus[i] -= 1e-2
      fd[i] = ((try loss(plus).0) - (try loss(minus).0)) / 2e-2 * Float(frameCount)
    }
    return (auto, fd)
  }

  private func relativeError(_ auto: [Float], _ fd: [Float]) -> Float {
    let scale = Swift.max(fd.map { Swift.abs($0) }.max() ?? 1, 1e-6)
    return (0..<auto.count).map { Swift.abs(auto[$0] - fd[$0]) / scale }.max() ?? 0
  }

  /// Control: without hop gating the same graph differentiates correctly.
  func testFrameRateOperandGradientIsCorrectWithoutHopGating() throws {
    let (auto, fd) = try gradients(hop: 1, frameRateIndex: false)
    XCTAssertLessThan(
      relativeError(auto, fd), 1e-3,
      "hop=1 autograd should match finite differences; auto=\(auto) fd=\(fd)")
  }

  /// The regime this file was written for, and the one every real hop-gated
  /// spectral graph is in: a frame-rate operand crossing the hop gate.
  func testFrameRateOperandGradientAcrossHopGate() throws {
    for hop in [2, 4, N] {
      let (auto, fd) = try gradients(hop: hop, frameRateIndex: true)
      XCTAssertLessThan(
        relativeError(auto, fd), 1e-3,
        "hop=\(hop) autograd should match finite differences; auto=\(auto) fd=\(fd)")
    }
  }

  /// Known failure, and a different defect from the one above: with no
  /// frame-rate node anywhere the loss itself is only evaluated on hop ticks
  /// and the whole backward chain is hop-scheduled. Marked strict so fixing it
  /// surfaces as a loud "unexpected pass".
  func testFullyHopScheduledGraphGradient() throws {
    XCTExpectFailure(
      "known DGen bug, separate from hop-gated operand BPTT: when every node "
        + "including the loss is scheduled at hop rate, the backward chain returns "
        + "a gradient unrelated to the true one (reproduces without sampleRow too)"
    )
    let (auto, fd) = try gradients(hop: N, frameRateIndex: false)
    XCTAssertLessThan(
      relativeError(auto, fd), 1e-3,
      "hop=\(N) autograd should match finite differences; auto=\(auto) fd=\(fd)")
  }
}

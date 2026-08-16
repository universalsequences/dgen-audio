import XCTest

@testable import DGenLazy

/// Pins a gradient bug found while hoisting `spectralFilter`: BPTT is wrong for
/// a **frame-rate** operand that multiplies into a **hop-gated** tensor region
/// (anything downstream of `buffer(size:hop:)` with `hop > 1`).
///
/// The graph below is about as small as an STFT filter gets — buffer, FFT,
/// multiply by a per-frame gain vector, sum — and it is entirely linear in the
/// gain vector, so the loss is an exact quadratic and central finite
/// differences are the true gradient with no truncation error at any step size.
///
/// With `hop = 1` autograd matches finite differences to ~1e-5 relative. With
/// `hop = 16` it does not just lose precision: the returned gradient carries
/// none of the spectrum's structure and points in a different direction
/// entirely. The directional derivative along the returned gradient `g` equals
/// `<fd, g>`, not `<g, g>` — i.e. `fd` is the true gradient and `g` is not.
///
/// Consequences: the DDSP frequency-sampled noise branch (hop = 32) and any
/// other hop-gated spectral path have never trained on a correct gradient.
/// They descend only because the wrong direction happens to correlate with the
/// right one. Fixing this is an engine change (hop-gated BPTT), out of scope
/// for the `spectralFilter` hoist that surfaced it.
final class HopGatedSpectralGradientTests: XCTestCase {
  private let frameCount = 256
  private let N = 16

  override func setUp() {
    super.setUp()
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
  private func gradients(hop: Int) throws -> (auto: [Float], fd: [Float]) {
    let noiseData = makeNoise(count: frameCount, seed: 99)
    let targetRow = (0..<N).map { c in 0.3 + Float(c % 3) * 0.2 }
    let start = [Float](repeating: 0.5, count: N)
    let width = N

    func loss(_ gains: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let learned = Tensor.param([1, width], data: gains)
      let noise = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let target = Tensor([targetRow])
      let frame = noise.buffer(size: width, hop: hop).reshape([width]) * hann(width)
      let (re, _) = signalTensorFFT(frame, N: width)
      func branch(_ m: Tensor) -> Signal { (re * m.sampleRow(Signal.constant(0.0))).sum() }
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
    let (auto, fd) = try gradients(hop: 1)
    XCTAssertLessThan(
      relativeError(auto, fd), 1e-3,
      "hop=1 autograd should match finite differences; auto=\(auto) fd=\(fd)")
  }

  /// Known failure: see the type comment. Marked strict so that fixing the
  /// engine bug turns this into a loud "unexpected pass" rather than silence.
  func testFrameRateOperandGradientAcrossHopGate() throws {
    XCTExpectFailure(
      "known DGen bug: BPTT for a frame-rate operand multiplying into a hop-gated "
        + "region (buffer(hop:) with hop > 1) returns a gradient unrelated to the true one"
    )
    let (auto, fd) = try gradients(hop: N)
    XCTAssertLessThan(
      relativeError(auto, fd), 1e-3,
      "hop=\(N) autograd should match finite differences; auto=\(auto) fd=\(fd)")
  }
}

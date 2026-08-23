import XCTest

@testable import DGenLazy

/// Finite-difference validation of BPTT through a **hop-gated** spectral region
/// (anything downstream of `Signal.buffer(size:hop:)` with `hop > 1`).
///
/// The graphs here are exactly quadratic in the parameter, so central
/// differences are the true gradient with no truncation error. The metric is
/// cosine similarity, which is scale-free — loss reduction conventions (sum vs
/// mean over frames) cannot mask or manufacture a failure.
///
/// The matrix covers both axes that can drive the error: the hop size, and how
/// the frame-rate operand reaches the multiply (a static `Tensor.param`, or a
/// `sampleRow` of a `[frames, n]` param).
final class HopGatedGradientFDMatrixTests: XCTestCase {
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

  private func playhead(featureFrames: Int) -> Signal {
    let maxIndex = Float(Swift.max(0, featureFrames - 1))
    let step = maxIndex / Float(Swift.max(1, frameCount - 1))
    let raw = Signal.accum(
      Signal.constant(step), reset: 0.0, min: 0.0, max: maxIndex)
    return raw.clip(0.0, Double(Swift.max(0.0, maxIndex - 1e-4)))
  }

  private func cosine(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count && !a.isEmpty)
    var dot: Double = 0
    var na: Double = 0
    var nb: Double = 0
    for i in 0..<a.count {
      dot += Double(a[i]) * Double(b[i])
      na += Double(a[i]) * Double(a[i])
      nb += Double(b[i]) * Double(b[i])
    }
    guard na > 0, nb > 0 else { return 0 }
    return Float(dot / (na.squareRoot() * nb.squareRoot()))
  }

  // MARK: - Harness

  /// Runs `build` under autograd and under central finite differences, and
  /// returns the cosine between the two gradient vectors.
  ///
  /// `build` receives the parameter tensor and returns a scalar-per-frame loss.
  private func cosineAgainstFD(
    paramShape: Shape,
    start: [Float],
    epsilon: Float = 1e-2,
    build: @escaping (Tensor) -> Signal
  ) throws -> (cos: Float, auto: [Float], fd: [Float]) {
    func run(_ values: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let p = Tensor.param(paramShape, data: values)
      let loss = build(p)
      let perFrame = try loss.backward(frames: frameCount)
      let total = perFrame.reduce(0, +)
      return (total, p.grad?.getData() ?? [])
    }

    let (_, auto) = try run(start)
    XCTAssertEqual(auto.count, start.count, "autograd gradient has wrong length")

    var fd = [Float](repeating: 0, count: start.count)
    for i in 0..<start.count {
      var plus = start
      plus[i] += epsilon
      var minus = start
      minus[i] -= epsilon
      fd[i] = (try run(plus).0 - (try run(minus).0)) / (2 * epsilon)
    }
    return (cosine(auto, fd), auto, fd)
  }

  // MARK: - Graphs

  /// Static `Tensor.param` scaling the half of the STFT, IFFT'd and
  /// overlap-added back to a signal, scored against a fixed target signal.
  private func staticOperandLoss(hop: Int) -> (Tensor) -> Signal {
    let noiseData = makeNoise(count: frameCount, seed: 99)
    let targetData = makeNoise(count: frameCount, seed: 7)
    let N = self.N
    let frameCount = self.frameCount
    return { p in
      let noise = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let target = Tensor(targetData).toSignal(maxFrames: frameCount)
      let window = hann(N)
      let frame = noise.buffer(size: N, hop: hop).reshape([N]) * window
      let (re, im) = signalTensorFFT(frame, N: N)
      let out = signalTensorIFFT(re * p, im * p, N: N)
      let signal = (out * window).overlapAdd(hop: hop)
      return mse(signal, target)
    }
  }

  /// Same graph, but the frame-rate operand arrives through `sampleRow` of a
  /// `[frames, N]` parameter driven by a playhead ramp.
  private func sampleRowOperandLoss(hop: Int, rows: Int) -> (Tensor) -> Signal {
    let noiseData = makeNoise(count: frameCount, seed: 99)
    let targetData = makeNoise(count: frameCount, seed: 7)
    let N = self.N
    let frameCount = self.frameCount
    let pos = { self.playhead(featureFrames: rows) }
    return { p in
      let noise = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let target = Tensor(targetData).toSignal(maxFrames: frameCount)
      let window = hann(N)
      let gains = p.sampleRow(pos())
      let frame = noise.buffer(size: N, hop: hop).reshape([N]) * window
      let (re, im) = signalTensorFFT(frame, N: N)
      let out = signalTensorIFFT(re * gains, im * gains, N: N)
      let signal = (out * window).overlapAdd(hop: hop)
      return mse(signal, target)
    }
  }

  // MARK: - Tests

  private func assertStatic(hop: Int, file: StaticString = #filePath, line: UInt = #line) throws {
    let start = (0..<N).map { 0.5 + Float($0 % 3) * 0.1 }
    let (cos, auto, fd) = try cosineAgainstFD(
      paramShape: [N], start: start, build: staticOperandLoss(hop: hop))
    XCTAssertGreaterThan(
      cos, 0.999,
      "static operand, hop=\(hop): cosine=\(cos)\nauto=\(auto)\nfd=\(fd)",
      file: file, line: line)
  }

  private func assertSampleRow(hop: Int, file: StaticString = #filePath, line: UInt = #line) throws
  {
    let rows = 4
    let start = (0..<(rows * N)).map { 0.5 + Float($0 % 5) * 0.05 }
    let (cos, auto, fd) = try cosineAgainstFD(
      paramShape: [rows, N], start: start,
      build: sampleRowOperandLoss(hop: hop, rows: rows))
    XCTAssertGreaterThan(
      cos, 0.999,
      "sampleRow operand, hop=\(hop): cosine=\(cos)\nauto=\(auto)\nfd=\(fd)",
      file: file, line: line)
  }

  func testStaticOperandHop1() throws { try assertStatic(hop: 1) }
  func testStaticOperandHop2() throws { try assertStatic(hop: 2) }
  func testStaticOperandHop4() throws { try assertStatic(hop: 4) }
  func testStaticOperandHop16() throws { try assertStatic(hop: 16) }

  func testSampleRowOperandHop1() throws { try assertSampleRow(hop: 1) }
  func testSampleRowOperandHop2() throws { try assertSampleRow(hop: 2) }
  func testSampleRowOperandHop4() throws { try assertSampleRow(hop: 4) }
  func testSampleRowOperandHop16() throws { try assertSampleRow(hop: 16) }
}

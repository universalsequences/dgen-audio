import XCTest

@testable import DGenLazy

/// Gradients of a *static* `Tensor.param` whose derived tensor is multiplied
/// elementwise into a frame-varying SignalTensor (no sampleRow in between).
///
/// Direct product, pad, and matmul-derived tensors differentiate correctly.
/// A static `tensorFFT` chain does NOT: the frame-varying adjoint comes back
/// with wrong magnitudes (deterministically — the C backend produces the exact
/// same wrong values as Metal, so it is a backward-graph construction defect,
/// not a GPU race). Found while building the R5 learned reverb, which now maps
/// IR taps to their spectrum with a constant DFT matmul instead (see
/// `LearnedReverb.swift`). The two FFT cases below pin the defect as expected
/// failures until the static-FFT adjoint path is fixed.
final class StaticTensorFrameGradProbeTests: XCTestCase {
  private let frameCount = 64
  private let N = 8

  override func setUp() {
    super.setUp()
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

  private func runCase(_ name: String, transform: @escaping (Tensor) -> Tensor) throws {
    let noiseData = makeNoise(count: frameCount, seed: 7)
    let L = N / 2
    let start = (0..<L).map { 0.2 + 0.05 * Float($0) }

    func loss(_ hData: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let h = Tensor.param([L], data: hData)
      let hFull = transform(h)  // [N]
      let sig = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let frame = sig.buffer(size: N, hop: 1).reshape([N])
      let pred = (frame * hFull).sum()
      let values = try mse(pred, Signal.constant(0.0)).backward(frames: frameCount)
      return (values.reduce(0, +) / Float(values.count), h.grad?.getData() ?? [])
    }

    let (_, auto) = try loss(start)
    var fd = [Float](repeating: 0, count: L)
    let eps: Float = 1e-2
    for i in 0..<L {
      var p = start; p[i] += eps
      var m = start; m[i] -= eps
      fd[i] = ((try loss(p).0) - (try loss(m).0)) / (2 * eps) * Float(frameCount)
    }
    let scale = Swift.max(fd.map { abs($0) }.max() ?? 1, 1e-6)
    let relErr = (0..<L).map { abs(auto[$0] - fd[$0]) / scale }.max() ?? 0
    print("probe \(name): relErr=\(relErr)\n  auto=\(auto)\n  fd=\(fd)")
    XCTAssertLessThan(relErr, 1e-3, "case \(name)")
  }

  func testPadProductGradient() throws {
    let half = N / 2
    try runCase("pad") { $0.pad([(0, half)]) }
  }

  func testFFTProductGradient() throws {
    XCTExpectFailure("static tensorFFT chain mis-propagates frame-varying adjoints (R5 finding)")
    let n = N
    let half = N / 2
    try runCase("fft") { h in
      let (re, _) = tensorFFT(h.pad([(0, half)]), N: n)
      return re
    }
  }

  func testFFTProductGradientCBackend() throws {
    XCTExpectFailure("static tensorFFT chain mis-propagates frame-varying adjoints (R5 finding)")
    DGenConfig.backend = .c
    defer { DGenConfig.backend = .metal }
    let n = N
    let half = N / 2
    try runCase("fft-c") { h in
      let (re, _) = tensorFFT(h.pad([(0, half)]), N: n)
      return re
    }
  }

  func testMatmulProductGradient() throws {
    let n = N
    let half = N / 2
    // Constant [L, N] matrix (DFT-like) between the param and the product.
    var rows = [[Float]]()
    for l in 0..<half {
      rows.append((0..<n).map { f in Foundation.cos(2.0 * Float.pi * Float(l * f) / Float(n)) })
    }
    try runCase("matmul") { h in
      h.reshape([1, half]).matmul(Tensor(rows)).reshape([n])
    }
  }

  private func runSignalCase(
    _ name: String, hop: Int, tolerance: Float = 1e-3,
    build: @escaping (Tensor, SignalTensor, SignalTensor) -> Signal
  ) throws {
    let noiseData = makeNoise(count: frameCount, seed: 7)
    let n = N
    let L = N / 2
    let start = (0..<L).map { 0.2 + 0.05 * Float($0) }

    func loss(_ hData: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let h = Tensor.param([L], data: hData)
      let sig = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let frame = sig.buffer(size: n, hop: hop).reshape([n])
      let (reX, imX) = signalTensorFFT(frame, N: n)
      let pred = build(h, reX, imX)
      let values = try mse(pred, Signal.constant(0.0)).backward(frames: frameCount)
      return (values.reduce(0, +) / Float(values.count), h.grad?.getData() ?? [])
    }

    let (_, auto) = try loss(start)
    var fd = [Float](repeating: 0, count: L)
    let eps: Float = 1e-2
    for i in 0..<L {
      var p = start; p[i] += eps
      var m = start; m[i] -= eps
      fd[i] = ((try loss(p).0) - (try loss(m).0)) / (2 * eps) * Float(frameCount)
    }
    let scale = Swift.max(fd.map { abs($0) }.max() ?? 1, 1e-6)
    let relErr = (0..<L).map { abs(auto[$0] - fd[$0]) / scale }.max() ?? 0
    print("probe \(name) hop=\(hop): relErr=\(relErr)\n  auto=\(auto)\n  fd=\(fd)")
    XCTAssertLessThan(relErr, tolerance, "case \(name) hop=\(hop)")
  }

  private func dftRows(_ n: Int, _ L: Int, imag: Bool) -> Tensor {
    var rows = [[Float]]()
    for l in 0..<L {
      rows.append((0..<n).map { f in
        let angle = 2.0 * Float.pi * Float((l * f) % n) / Float(n)
        return imag ? -Foundation.sin(angle) : Foundation.cos(angle)
      })
    }
    return Tensor(rows)
  }

  func testSingleConsumerStreamingFFTProduct() throws {
    let n = N
    let L = N / 2
    for hop in [1, 4] {
      try runSignalCase("singleConsumer", hop: hop) { h, reX, _ in
        let reH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: false)).reshape([n])
        return (reX * reH).sum()
      }
    }
  }

  func testDualConsumerComplexProduct() throws {
    let n = N
    let L = N / 2
    for hop in [1, 4] {
      try runSignalCase("dualConsumer", hop: hop) { h, reX, imX in
        let reH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: false)).reshape([n])
        let imH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: true)).reshape([n])
        let reY = reX * reH - imX * imH
        let imY = reX * imH + imX * reH
        return (reY + imY).sum()
      }
    }
  }

  func testComplexProductIFFTOverlapAdd() throws {
    let n = N
    let L = N / 2
    for hop in [1, 4] {
      try runSignalCase("ifft-ola", hop: hop) { h, reX, imX in
        let reH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: false)).reshape([n])
        let imH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: true)).reshape([n])
        let reY = reX * reH - imX * imH
        let imY = reX * imH + imX * reH
        return signalTensorIFFT(reY, imY, N: n).overlapAdd(hop: hop)
      }
    }
  }

  func testIFFTSumGradient() throws {
    let n = N
    let L = N / 2
    for hop in [1, 4] {
      try runSignalCase("ifft-sum", hop: hop) { h, reX, imX in
        let reH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: false)).reshape([n])
        let imH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: true)).reshape([n])
        let reY = reX * reH - imX * imH
        let imY = reX * imH + imX * reH
        return signalTensorIFFT(reY, imY, N: n).sum()
      }
    }
  }

  func testOverlapAddOnlyGradient() throws {
    let n = N
    let L = N / 2
    for hop in [1, 4] {
      try runSignalCase("ola-only", hop: hop) { h, reX, imX in
        let reH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: false)).reshape([n])
        let imH = h.reshape([1, L]).matmul(self.dftRows(n, L, imag: true)).reshape([n])
        let reY = reX * reH - imX * imH
        let imY = reX * imH + imX * reH
        return (reY + imY).overlapAdd(hop: hop)
      }
    }
  }

  func testDirectProductGradient() throws {
    let noiseData = makeNoise(count: frameCount, seed: 7)
    let start = (0..<N).map { 0.2 + 0.05 * Float($0) }

    func loss(_ hData: [Float]) throws -> (Float, [Float]) {
      LazyGraphContext.reset()
      let h = Tensor.param([N], data: hData)
      let sig = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let frame = sig.buffer(size: N, hop: 1).reshape([N])
      let pred = (frame * h).sum()
      let values = try mse(pred, Signal.constant(0.0)).backward(frames: frameCount)
      return (values.reduce(0, +) / Float(values.count), h.grad?.getData() ?? [])
    }

    let (_, auto) = try loss(start)
    var fd = [Float](repeating: 0, count: N)
    let eps: Float = 1e-2
    for i in 0..<N {
      var p = start; p[i] += eps
      var m = start; m[i] -= eps
      fd[i] = ((try loss(p).0) - (try loss(m).0)) / (2 * eps) * Float(frameCount)
    }
    let scale = Swift.max(fd.map { abs($0) }.max() ?? 1, 1e-6)
    let relErr = (0..<N).map { abs(auto[$0] - fd[$0]) / scale }.max() ?? 0
    print("probe direct product: relErr=\(relErr)\n  auto=\(auto)\n  fd=\(fd)")
    XCTAssertLessThan(relErr, 1e-3)
  }
}

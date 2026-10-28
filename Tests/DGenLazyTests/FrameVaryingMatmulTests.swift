import XCTest

import enum DGen.Backend
@testable import DGenLazy

/// Forward values and gradients of `matmul` over FRAME-VARYING activations — the
/// MLP-decoder shape: per-frame x[1,K] @ W1[K,H] -> tanh -> @ W2[H,N].
///
/// The axis reduction that fuses matmul's broadcast multiply used to index a
/// frame-aware view operand with its base strides, dropping the view's extra
/// dims (x[i][0] read for every k). The same emitter serves C and Metal, so
/// both backends are checked.
final class FrameVaryingMatmulTests: XCTestCase {
  private let frameCount = 64
  private var savedBackend: Backend = .metal
  private var savedMaxFrames = 0

  override func setUp() {
    super.setUp()
    savedBackend = DGenConfig.backend
    savedMaxFrames = DGenConfig.maxFrameCount
    DGenConfig.maxFrameCount = frameCount
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = savedBackend
    DGenConfig.maxFrameCount = savedMaxFrames
    LazyGraphContext.reset()
    super.tearDown()
  }

  private func noise(_ count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(state >> 40) / Float(1 << 24) * 2.0 - 1.0
    }
  }

  private let w1Start: [Float] = [0.5, -0.3, 0.8, 0.2]  // [2,2]
  private let w2Start: [Float] = [1.0, 2.0, 3.0, -1.0, 0.5, 0.25]  // [2,3]
  private let readout: [Float] = [1.0, 0.5, 0.25]

  /// Builds pred(t) = sum((tanh(x(t) @ W1) @ W2) * readout), x(t) = [s0(t), s1(t)].
  private func buildPred(w1: Tensor, w2: Tensor) throws -> Signal {
    let s0 = Tensor(noise(frameCount, seed: 3)).toSignal(maxFrames: frameCount)
    let s1 = Tensor(noise(frameCount, seed: 11)).toSignal(maxFrames: frameCount)
    let x = Tensor([[1.0, 0.0]]) * s0 + Tensor([[0.0, 1.0]]) * s1  // [1,2], per frame
    let h = tanh(try x.matmul(w1))  // [1,2]
    let y = try h.matmul(w2)  // [1,3]
    return (y * Tensor([readout])).sum()
  }

  private func referencePred(w1: [Float], w2: [Float]) -> [Float] {
    let s0 = noise(frameCount, seed: 3)
    let s1 = noise(frameCount, seed: 11)
    return (0..<frameCount).map { t in
      let x = [s0[t], s1[t]]
      let h = (0..<2).map { j in Foundation.tanh(x[0] * w1[j] + x[1] * w1[2 + j]) }
      return (0..<3).reduce(Float(0)) { acc, n in
        acc + (h[0] * w2[n] + h[1] * w2[3 + n]) * readout[n]
      }
    }
  }

  private func checkForward(_ backend: Backend) throws {
    DGenConfig.backend = backend
    LazyGraphContext.reset()
    let pred = try buildPred(
      w1: Tensor(w1Start).reshape([2, 2]), w2: Tensor(w2Start).reshape([2, 3]))
    let got = try pred.realize(frames: frameCount)
    let want = referencePred(w1: w1Start, w2: w2Start)
    XCTAssertEqual(got.count, frameCount)
    let err = zip(got, want).map { abs($0 - $1) }.max() ?? .infinity
    XCTAssertLessThan(err, 1e-4, "\(backend) forward max error \(err)")
  }

  private func checkGradients(_ backend: Backend) throws {
    DGenConfig.backend = backend
    let target = noise(frameCount, seed: 29)

    func loss(_ w1Data: [Float], _ w2Data: [Float]) throws -> (Float, [Float], [Float]) {
      LazyGraphContext.reset()
      let w1 = Tensor.param([2, 2], data: w1Data)
      let w2 = Tensor.param([2, 3], data: w2Data)
      let pred = try buildPred(w1: w1, w2: w2)
      let tgt = Tensor(target).toSignal(maxFrames: frameCount)
      let values = try mse(pred, tgt).backward(frames: frameCount)
      return (
        values.reduce(0, +) / Float(values.count),
        w1.grad?.getData() ?? [], w2.grad?.getData() ?? []
      )
    }

    // Reference loss: mean over frames of (pred - target)^2.
    func refLoss(_ w1: [Float], _ w2: [Float]) -> Float {
      let p = referencePred(w1: w1, w2: w2)
      var total: Float = 0
      for t in 0..<frameCount {
        let d: Float = p[t] - target[t]
        total += d * d
      }
      return total / Float(frameCount)
    }

    let (_, g1, g2) = try loss(w1Start, w2Start)
    XCTAssertEqual(g1.count, 4)
    XCTAssertEqual(g2.count, 6)
    let eps: Float = 1e-2
    func fd(_ which: Int, _ i: Int) -> Float {
      var p1 = w1Start, m1 = w1Start, p2 = w2Start, m2 = w2Start
      if which == 1 { p1[i] += eps; m1[i] -= eps } else { p2[i] += eps; m2[i] -= eps }
      // Gradients accumulate over frames: scale the mean-loss FD by frameCount.
      return (refLoss(p1, p2) - refLoss(m1, m2)) / (2 * eps) * Float(frameCount)
    }
    let fd1 = (0..<4).map { fd(1, $0) }
    let fd2 = (0..<6).map { fd(2, $0) }
    let scale = Swift.max((fd1 + fd2).map { abs($0) }.max() ?? 1, 1e-6)
    let err1 = zip(g1, fd1).map { abs($0 - $1) / scale }.max() ?? .infinity
    let err2 = zip(g2, fd2).map { abs($0 - $1) / scale }.max() ?? .infinity
    XCTAssertLessThan(err1, 2e-3, "\(backend) dW1 auto=\(g1) fd=\(fd1)")
    XCTAssertLessThan(err2, 2e-3, "\(backend) dW2 auto=\(g2) fd=\(fd2)")
  }

  func testForwardC() throws { try checkForward(.c) }
  func testForwardMetal() throws { try checkForward(.metal) }
  func testGradientsC() throws { try checkGradients(.c) }
  func testGradientsMetal() throws { try checkGradients(.metal) }
}

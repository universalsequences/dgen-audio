import XCTest

@testable import DGen
@testable import DGenLazy

/// Gradient of a parameter that reaches a matmul through a DERIVED right operand:
/// u = X @ (W0 + N @ C) + (b0 + n @ C). (E6 latent-path --readout-null: dL/dC was wrong.)
final class MatmulDerivedOperandGradientTests: XCTestCase {
  override func tearDown() { DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount; super.tearDown() }

  private func lossAndGrad(_ c: [Float], variant: Int) throws -> (Float, [Float]) {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    let B = 5, H = 6, k = 3, P = 4
    let X = Tensor((0..<(B * H)).map { sin(Float($0) * 0.7) }).reshape([B, H])
    let W0 = Tensor((0..<(H * P)).map { cos(Float($0) * 0.3) * 0.5 }).reshape([H, P])
    let N = Tensor((0..<(H * k)).map { sin(Float($0) * 1.3 + 0.2) }).reshape([H, k])
    let C = Tensor.param([k, P], data: c)
    let u: DGenLazy.Tensor
    switch variant {
    case 0: u = X.matmul(W0 + N.matmul(C))
    case 1: u = X.matmul(N.matmul(C))
    default: u = X.matmul(N).matmul(C)
    }
    let loss = (tanh(u) * tanh(u)).sum()
    let v = try (Tensor([0.0]) + loss).peek(Signal.constant(0)) * (1 / Float(64))
    let lv = try v.backward(frames: 64).reduce(0, +)
    return (lv, C.grad?.getData() ?? [])
  }

  func testDerivedRightOperandGradientMatchesFD() throws {
    let c0: [Float] = (0..<12).map { Float($0) * 0.05 - 0.2 }
    for variant in 0..<3 {
      let (_, g) = try lossAndGrad(c0, variant: variant)
      XCTAssertEqual(g.count, 12)
      for i in [0, 5, 11] {
        var cp = c0, cm = c0; cp[i] += 1e-2; cm[i] -= 1e-2
        let fd = (try lossAndGrad(cp, variant: variant).0 - lossAndGrad(cm, variant: variant).0) / 2e-2
        XCTAssertEqual(g[i], fd, accuracy: 2e-2 * max(1, abs(fd)), "variant \(variant) index \(i)")
      }
    }
  }
}

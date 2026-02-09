import DGen
import XCTest

@testable import DGenLazy

/// Tests that verify BPTT (Backpropagation Through Time) correctness for Signal.history().
///
/// BPTT splits scalar feedback blocks into a forward loop (0→N-1) and a reverse
/// backward loop (N-1→0), enabling correct temporal gradient propagation through
/// history() feedback. These tests compare BPTT gradients against analytical
/// formulas and finite-difference numerical baselines.
final class BPTTTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
  }

  // MARK: - Helpers

  /// Pure exponential decay: y[0]=1 (click seed), y[n]=y[n-1]*rate for n>0.
  /// Produces y[n] = rate^n.
  private func buildDecay(rate: Signal) -> Signal {
    let trigger = Signal.click()
    let (prev, write) = Signal.history()
    return write(gswitch(trigger, 1.0, prev * rate))
  }

  /// Run forward+backward for a decay signal.
  /// Loss = Σ (y[n] - 0)^2 = Σ y[n]^2 = Σ rate^(2n).
  private func evalDecay(rate: Float, frameCount: Int) throws -> (totalLoss: Float, grad: Float?) {
    LazyGraphContext.reset()
    let param = Signal.param(rate)
    let y = buildDecay(rate: param)
    let loss = mse(y, Signal.constant(0.0))
    let lossValues = try loss.backward(frames: frameCount)
    let totalLoss = lossValues.reduce(0, +)
    return (totalLoss, param.grad?.data)
  }

  /// Run forward+backward for an onepole filter with MSE loss against a target.
  private func evalOnepole(cutoff: Float, targetCutoff: Float, frameCount: Int) throws
    -> (totalLoss: Float, grad: Float?)
  {
    LazyGraphContext.reset()
    let param = Signal.param(cutoff)

    let phase1 = Signal.phasor(440.0)
    let (prev1, write1) = Signal.history()
    let learnable = write1(Signal.mix(phase1, prev1, param))

    let phase2 = Signal.phasor(440.0)
    let (prev2, write2) = Signal.history()
    let target = write2(Signal.mix(phase2, prev2, targetCutoff))

    let loss = mse(learnable, target)
    let lossValues = try loss.backward(frames: frameCount)
    let totalLoss = lossValues.reduce(0, +)
    return (totalLoss, param.grad?.data)
  }

  /// 1st-order IIR: y[n] = b*x[n] - a*y[n-1]
  /// Single history pair for output feedback. Tests both feedforward (b) and feedback (a).
  private func buildIIR(input: Signal, b: Signal, a: Signal) -> Signal {
    let (prevY, writeY) = Signal.history()
    let y = b * input - a * prevY
    return writeY(y)
  }

  /// Run forward+backward for an IIR filter with MSE loss against a target.
  private func evalIIR(
    bValue: Float, aValue: Float,
    targetB: Float, targetA: Float,
    frameCount: Int
  ) throws -> (totalLoss: Float, gradB: Float?, gradA: Float?) {
    LazyGraphContext.reset()
    let bParam = Signal.param(bValue)
    let aParam = Signal.param(aValue)

    let input1 = Signal.phasor(440.0)
    let learnable = buildIIR(input: input1, b: bParam, a: aParam)

    let input2 = Signal.phasor(440.0)
    let target = buildIIR(
      input: input2,
      b: Signal.constant(targetB), a: Signal.constant(targetA))

    let loss = mse(learnable, target)
    let lossValues = try loss.backward(frames: frameCount)
    let totalLoss = lossValues.reduce(0, +)
    return (totalLoss, bParam.grad?.data, aParam.grad?.data)
  }

  // MARK: - Gradient Correctness: Exponential Decay

  /// Analytical gradient for exponential decay.
  ///
  /// y[n] = rate^n, Loss = Σ y[n]^2 = Σ rate^(2n)
  /// dL/drate = Σ_{n=1}^{N-1} 2n * rate^(2n-1)
  func testAnalyticalGradientDecay() throws {
    let frameCount = 64
    let rate: Float = 0.8

    let (totalLoss, bpttGrad) = try evalDecay(rate: rate, frameCount: frameCount)

    var analyticalLoss: Float = 0
    var analyticalGrad: Float = 0
    for n in 0..<frameCount {
      analyticalLoss += pow(rate, Float(2 * n))
      if n >= 1 {
        analyticalGrad += 2.0 * Float(n) * pow(rate, Float(2 * n - 1))
      }
    }

    print("=== Analytical Gradient (Decay) ===")
    print("Loss:     analytical=\(analyticalLoss)  gpu=\(totalLoss)")
    print("Gradient: analytical=\(analyticalGrad)  bptt=\(bpttGrad ?? -999)")
    let relError = abs((bpttGrad ?? 0) - analyticalGrad) / abs(analyticalGrad)
    print("Relative error: \(String(format: "%.6f", relError))")

    XCTAssertNotNil(bpttGrad, "BPTT gradient should exist")
    XCTAssertEqual(
      totalLoss, analyticalLoss, accuracy: abs(analyticalLoss) * 0.01,
      "GPU loss should match analytical")
    XCTAssertEqual(
      bpttGrad!, analyticalGrad, accuracy: abs(analyticalGrad) * 0.05,
      "BPTT gradient should match analytical within 5%")
  }

  /// Numerical (finite-difference) gradient for exponential decay.
  func testNumericalGradientDecay() throws {
    let frameCount = 64
    let rate: Float = 0.8
    let eps: Float = 1e-3

    let (_, bpttGrad) = try evalDecay(rate: rate, frameCount: frameCount)
    let (lossPlus, _) = try evalDecay(rate: rate + eps, frameCount: frameCount)
    let (lossMinus, _) = try evalDecay(rate: rate - eps, frameCount: frameCount)
    let numericalGrad = (lossPlus - lossMinus) / (2 * eps)

    print("=== Numerical Gradient (Decay) ===")
    print("BPTT gradient:      \(bpttGrad ?? -999)")
    print("Numerical gradient: \(numericalGrad)")
    let relError = abs((bpttGrad ?? 0) - numericalGrad) / abs(numericalGrad)
    print("Relative error: \(String(format: "%.6f", relError))")

    XCTAssertNotNil(bpttGrad, "BPTT gradient should exist")
    XCTAssertEqual(
      bpttGrad!, numericalGrad, accuracy: abs(numericalGrad) * 0.05,
      "BPTT should match numerical gradient within 5%")
  }

  // MARK: - Gradient Correctness: Onepole Filter

  /// Numerical gradient for onepole lowpass filter.
  func testNumericalGradientOnepole() throws {
    let frameCount = 128
    let cutoff: Float = 0.5
    let targetCutoff: Float = 0.2
    let eps: Float = 1e-3

    let (_, bpttGrad) = try evalOnepole(
      cutoff: cutoff, targetCutoff: targetCutoff, frameCount: frameCount)
    let (lossPlus, _) = try evalOnepole(
      cutoff: cutoff + eps, targetCutoff: targetCutoff, frameCount: frameCount)
    let (lossMinus, _) = try evalOnepole(
      cutoff: cutoff - eps, targetCutoff: targetCutoff, frameCount: frameCount)
    let numericalGrad = (lossPlus - lossMinus) / (2 * eps)

    print("=== Numerical Gradient (Onepole) ===")
    print("BPTT gradient:      \(bpttGrad ?? -999)")
    print("Numerical gradient: \(numericalGrad)")
    let relError = abs((bpttGrad ?? 0) - numericalGrad) / abs(numericalGrad)
    print("Relative error: \(String(format: "%.6f", relError))")

    XCTAssertNotNil(bpttGrad, "BPTT gradient should exist")
    XCTAssertEqual(
      bpttGrad!, numericalGrad, accuracy: abs(numericalGrad) * 0.1,
      "BPTT should match numerical gradient within 10%")
  }

  // MARK: - Multiple Independent Histories

  /// Two independent decay loops with separate params.
  /// Verifies carry cells don't interfere between history pairs.
  func testMultipleHistories() throws {
    let frameCount = 64
    let r1: Float = 0.9
    let r2: Float = 0.5
    let eps: Float = 1e-3

    func evalMulti(rate1: Float, rate2: Float) throws
      -> (totalLoss: Float, grad1: Float?, grad2: Float?)
    {
      LazyGraphContext.reset()
      let param1 = Signal.param(rate1)
      let param2 = Signal.param(rate2)

      let y1 = buildDecay(rate: param1)
      let y2 = buildDecay(rate: param2)

      // Use mse for each branch, sum the losses
      let loss = mse(y1, Signal.constant(0.0)) + mse(y2, Signal.constant(0.0))
      let lossValues = try loss.backward(frames: frameCount)
      let totalLoss = lossValues.reduce(0, +)
      return (totalLoss, param1.grad?.data, param2.grad?.data)
    }

    let (_, grad1, grad2) = try evalMulti(rate1: r1, rate2: r2)

    // Numerical gradient for rate1
    let (lossP1, _, _) = try evalMulti(rate1: r1 + eps, rate2: r2)
    let (lossM1, _, _) = try evalMulti(rate1: r1 - eps, rate2: r2)
    let numGrad1 = (lossP1 - lossM1) / (2 * eps)

    // Numerical gradient for rate2
    let (lossP2, _, _) = try evalMulti(rate1: r1, rate2: r2 + eps)
    let (lossM2, _, _) = try evalMulti(rate1: r1, rate2: r2 - eps)
    let numGrad2 = (lossP2 - lossM2) / (2 * eps)

    print("=== Multiple Histories ===")
    print("Rate1 (\(r1)): BPTT=\(grad1 ?? -999) numerical=\(numGrad1)")
    print("Rate2 (\(r2)): BPTT=\(grad2 ?? -999) numerical=\(numGrad2)")

    XCTAssertNotNil(grad1, "Gradient for rate1 should exist")
    XCTAssertNotNil(grad2, "Gradient for rate2 should exist")
    XCTAssertEqual(
      grad1!, numGrad1, accuracy: abs(numGrad1) * 0.1,
      "Rate1 BPTT should match numerical")
    XCTAssertEqual(
      grad2!, numGrad2, accuracy: abs(numGrad2) * 0.1,
      "Rate2 BPTT should match numerical")
  }

  // MARK: - Gradient Direction

  /// Stepping in the negative gradient direction should decrease loss.
  func testGradientDirection() throws {
    let frameCount = 128
    let rate: Float = 0.8

    let (loss0, grad0) = try evalDecay(rate: rate, frameCount: frameCount)
    XCTAssertNotNil(grad0, "Gradient should exist")
    XCTAssertGreaterThan(grad0!, 0, "Gradient should be positive (larger rate -> larger loss)")

    let lr: Float = 0.001
    let newRate = rate - lr * grad0!
    let (loss1, _) = try evalDecay(rate: newRate, frameCount: frameCount)

    print("=== Gradient Direction ===")
    print("Rate: \(rate) -> \(newRate)")
    print("Loss: \(loss0) -> \(loss1) (delta=\(loss1 - loss0))")

    XCTAssertLessThan(loss1, loss0, "Loss should decrease after gradient step")
  }

  // MARK: - IIR (Biquad-like) Filter

  /// Numerical gradient for a 1st-order IIR with output feedback.
  /// y[n] = b*x[n] - a*y[n-1]
  ///
  /// Tests both:
  /// - b gradient (feedforward, correct without temporal carry)
  /// - a gradient (feedback, requires BPTT for correctness)
  func testNumericalGradientIIR() throws {
    let frameCount = 128
    let b: Float = 0.8
    let a: Float = 0.3
    let targetB: Float = 1.0
    let targetA: Float = 0.5
    let eps: Float = 1e-3

    let (_, bpttGradB, bpttGradA) = try evalIIR(
      bValue: b, aValue: a,
      targetB: targetB, targetA: targetA,
      frameCount: frameCount)

    // Numerical gradient for b
    let (lossP_b, _, _) = try evalIIR(
      bValue: b + eps, aValue: a,
      targetB: targetB, targetA: targetA,
      frameCount: frameCount)
    let (lossM_b, _, _) = try evalIIR(
      bValue: b - eps, aValue: a,
      targetB: targetB, targetA: targetA,
      frameCount: frameCount)
    let numGradB = (lossP_b - lossM_b) / (2 * eps)

    // Numerical gradient for a
    let (lossP_a, _, _) = try evalIIR(
      bValue: b, aValue: a + eps,
      targetB: targetB, targetA: targetA,
      frameCount: frameCount)
    let (lossM_a, _, _) = try evalIIR(
      bValue: b, aValue: a - eps,
      targetB: targetB, targetA: targetA,
      frameCount: frameCount)
    let numGradA = (lossP_a - lossM_a) / (2 * eps)

    print("=== Numerical Gradient (IIR) ===")
    print("b: BPTT=\(bpttGradB ?? -999) numerical=\(numGradB)")
    print("a: BPTT=\(bpttGradA ?? -999) numerical=\(numGradA)")

    XCTAssertNotNil(bpttGradB, "Gradient for b should exist")
    XCTAssertNotNil(bpttGradA, "Gradient for a should exist")
    XCTAssertEqual(
      bpttGradB!, numGradB, accuracy: abs(numGradB) * 0.1,
      "b BPTT should match numerical")
    XCTAssertEqual(
      bpttGradA!, numGradA, accuracy: abs(numGradA) * 0.1,
      "a BPTT should match numerical")
  }

  /// Learn IIR filter coefficients (b, a) via BPTT.
  /// Target: b=1.0, a=0.4.  Learnable starts at b=0.8, a=0.1.
  func testLearnIIRCoefficients() throws {
    let frameCount = 128
    let targetB: Float = 1.0
    let targetA: Float = 0.4

    LazyGraphContext.reset()
    let bParam = Signal.param(0.8)
    let aParam = Signal.param(0.1)
    let optimizer = Adam(params: [bParam, aParam], lr: 0.05)

    var firstLoss: Float = 0
    var lastLoss: Float = 0

    for epoch in 0..<200 {
      let input1 = Signal.phasor(440.0)
      let learnable = buildIIR(input: input1, b: bParam, a: aParam)

      let input2 = Signal.phasor(440.0)
      let target = buildIIR(
        input: input2,
        b: Signal.constant(targetB), a: Signal.constant(targetA))

      let loss = mse(learnable, target)
      let lossValues = try loss.backward(frames: frameCount)
      let avgLoss = lossValues.reduce(0, +) / Float(frameCount)

      if epoch == 0 { firstLoss = avgLoss }
      lastLoss = avgLoss

      if epoch % 40 == 0 {
        print(
          "IIR epoch \(epoch): loss=\(String(format: "%.6f", avgLoss)) "
            + "b=\(bParam.data ?? -1) a=\(aParam.data ?? -1)")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print(
      "Final: b=\(bParam.data ?? -1) (target \(targetB)) "
        + "a=\(aParam.data ?? -1) (target \(targetA))")

    XCTAssertLessThan(lastLoss, firstLoss * 0.1, "Loss should decrease by >10x")
    XCTAssertEqual(
      bParam.data ?? -1, targetB, accuracy: 0.3,
      "b should approach target")
    XCTAssertEqual(
      aParam.data ?? -1, targetA, accuracy: 0.3,
      "a should approach target")
  }
}

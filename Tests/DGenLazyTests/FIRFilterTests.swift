import XCTest

@testable import DGenLazy

/// Tests for FIR (Finite Impulse Response) lowpass filtering using conv2d.
final class FIRFilterTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func generateTestSignals(n: Int = 64, freq: Float = 4.0) -> (clean: [Float], noisy: [Float]) {
    var clean = [Float](repeating: 0, count: n)
    var noisy = [Float](repeating: 0, count: n)
    for i in 0..<n {
      let phase = Float(i) / Float(n) * freq * 2.0 * .pi
      clean[i] = sin(phase)
      let noise = 0.3 * sin(Float(i) / Float(n) * 25.0 * 2.0 * .pi)
      noisy[i] = clean[i] + noise
    }
    return (clean, noisy)
  }

  /// MSE + regularization penalizing kernel sum deviating from 1.0
  private func firLoss(filtered: Tensor, target: Tensor, kernel: Tensor) -> Tensor {
    let diff = filtered - target
    let mse = (diff * diff).mean()
    let sumDev = kernel.sum() - Tensor([1.0])
    return mse + sumDev * sumDev * Tensor([0.1])
  }

  /// Shared setup for FIR lowpass tests: noisy input, clean target, and learnable kernel.
  private func makeFIRLowpassFixture(
    n: Int = 64, kernelSize: Int = 7
  ) -> (noisyTensor: Tensor, targetTensor: Tensor, kernel: Tensor) {
    let (clean, noisy) = generateTestSignals(n: n)
    let offset = (kernelSize - 1) / 2
    let noisyTensor = Tensor([noisy])
    let targetTensor = Tensor([Array(clean[offset..<(offset + n - kernelSize + 1)])])
    let kernel = Tensor.param(
      [1, kernelSize], data: [Float](repeating: 1.0 / Float(kernelSize), count: kernelSize))
    return (noisyTensor, targetTensor, kernel)
  }

  /// conv2d with a box filter (moving average) should smooth a signal.
  func testMovingAverageLowpass() throws {
    let n = 64
    let (clean, noisy) = generateTestSignals(n: n)

    let noisyTensor = Tensor([noisy])
    let kernel = Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])

    let filtered = noisyTensor.conv2d(kernel)
    XCTAssertEqual(filtered.shape, [1, n - 4])

    let result = try filtered.realize()

    var mseFiltered: Float = 0
    var mseNoisy: Float = 0
    for i in 0..<(n - 4) {
      let cleanVal = clean[i + 2]
      mseFiltered += (result[i] - cleanVal) * (result[i] - cleanVal)
      mseNoisy += (noisy[i + 2] - cleanVal) * (noisy[i + 2] - cleanVal)
    }
    mseFiltered /= Float(n - 4)
    mseNoisy /= Float(n - 4)

    print("=== Moving Average FIR Lowpass ===")
    print("Noise reduction ratio: \(String(format: "%.2fx", mseNoisy / mseFiltered))")

    XCTAssertLessThan(mseFiltered, mseNoisy, "Filtering should reduce noise")
  }

  /// Train a conv2d kernel to act as a lowpass filter.
  func testLearnedFIRLowpass() throws {
    let (noisyTensor, targetTensor, kernel) = makeFIRLowpassFixture()

    func buildLoss() -> Tensor {
      firLoss(filtered: noisyTensor.conv2d(kernel), target: targetTensor, kernel: kernel)
    }

    let optimizer = SGD(params: [kernel], lr: 0.01)
    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    print("\n=== Learned FIR Lowpass ===")
    print("Initial loss: \(String(format: "%.6f", initialLoss))")

    var finalLoss = initialLoss
    for epoch in 0..<50 {
      finalLoss = try buildLoss().backward(frameCount: 1)[0]

      if epoch % 20 == 0 || epoch == 49 {
        let k = kernel.getData() ?? []
        print("Epoch \(epoch): loss = \(String(format: "%.6f", finalLoss)), kernel = \(k.map { String(format: "%.3f", $0) })")
      }

      optimizer.step()
      if var k = kernel.getData() {
        let s = k.reduce(0, +)
        if abs(s) > 1e-8 { k = k.map { $0 / s } }
        kernel.updateDataLazily(k)
      }
      optimizer.zeroGrad()
    }

    let kernelSum = (kernel.getData() ?? []).reduce(0, +)
    XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease")
    XCTAssertEqual(kernelSum, 1.0, accuracy: 0.3, "Kernel should have ~unity DC gain")
  }

  /// Forward-only: realize loss without backward to verify output readout
  /// works for shape-[1] tensor losses.
  func testFIRLossForwardOnly() throws {
    let (noisyTensor, targetTensor, kernel) = makeFIRLowpassFixture()
    let loss = firLoss(filtered: noisyTensor.conv2d(kernel), target: targetTensor, kernel: kernel)
    let result = try loss.realize()

    XCTAssertGreaterThan(result[0], 0, "Loss should be non-zero")
  }

  /// Learn FIR coefficients that approximate a one-pole IIR filter.
  ///
  /// Teacher: y[n] = (1-α)*x[n] + α*y[n-1]  (IIR via history feedback)
  /// Student: y[n] = Σ h[k]*x[n-k]           (FIR via buffer + conv2d)
  ///
  /// The one-pole impulse response is h[n] = (1-α)*α^n, so a large enough
  /// FIR kernel should converge to this exponential decay.
  func testFIRApproximatesOnepole() throws {
    let kernelSize = 32
    let alpha: Float = 0.5
    let frameCount = 256

    func buildLoss(firKernel: Tensor) -> Signal {
      let phase = Signal.phasor(440.0)

      // Teacher: one-pole IIR
      let (prev, write) = Signal.history()
      let target = Signal.mix(phase, prev, alpha)
      write(target)

      // Student: FIR via buffer + conv2d → single dot product per frame
      // bufferSize == kernelSize so conv2d output is [1,1], .sum() collapses to scalar
      let student = phase.buffer(size: kernelSize).conv2d(firKernel).sum()

      // MSE via arithmetic
      let diff = student - target
      return diff * diff
    }

    let firKernel = Tensor.param(
      [1, kernelSize], data: [Float](repeating: 1.0 / Float(kernelSize), count: kernelSize))

    let optimizer = Adam(params: [firKernel], lr: 0.01)

    // Warmup
    _ = try buildLoss(firKernel: firKernel).backward(frames: frameCount)
    optimizer.zeroGrad()

    print("\n=== FIR Approximates One-Pole ===")
    var firstLoss: Float = 0
    var lastLoss: Float = 0
    for epoch in 0..<200 {
      let loss = buildLoss(firKernel: firKernel)
      let lossValues = try loss.backward(frames: frameCount)
      let avgLoss = lossValues.reduce(0, +) / Float(frameCount)

      if epoch == 0 { firstLoss = avgLoss }
      lastLoss = avgLoss

      if epoch % 50 == 0 || epoch == 199 {
        let k = firKernel.getData() ?? []
        print("Epoch \(epoch): loss=\(String(format: "%.6f", avgLoss)) kernel[0..4]=\(k.prefix(5).map { String(format: "%.4f", $0) })")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    // Verify loss decreased
    XCTAssertLessThan(lastLoss, firstLoss, "Loss should decrease")

    // Compare learned kernel to theoretical impulse response.
    // Buffer stores oldest→newest, so kernel[k] = (1-α)*α^(N-1-k)
    // (largest weight at the end = most recent sample)
    let learned = firKernel.getData() ?? []
    let theoretical = (0..<kernelSize).map { (1 - alpha) * pow(alpha, Float(kernelSize - 1 - $0)) }
    print("Learned (last 8):      \(learned.suffix(8).map { String(format: "%.4f", $0) })")
    print("Theoretical (last 8):  \(theoretical.suffix(8).map { String(format: "%.4f", $0) })")

    // The learned kernel should have its largest weight at the end (most recent sample)
    // matching the one-pole's exponential decay characteristic
    let lastTap = learned.last ?? 0
    XCTAssertGreaterThan(lastTap, 0.1, "Last kernel tap (newest sample) should dominate")
    XCTAssertLessThan(lastLoss, firstLoss * 0.1, "Loss should decrease by 10x")
  }
}

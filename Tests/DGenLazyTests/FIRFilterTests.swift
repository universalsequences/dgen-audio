import XCTest

@testable import DGenLazy

/// Tests for FIR (Finite Impulse Response) lowpass filtering using conv2d.
///
/// An FIR filter is just a convolution — each output sample is a weighted sum
/// of a window of input samples. By making the kernel learnable, the optimizer
/// discovers filter coefficients.
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

  // MARK: - Manual FIR Lowpass

  /// conv2d with a box filter (moving average) should smooth a signal.
  func testMovingAverageLowpass() throws {
    let n = 64
    let (clean, noisy) = generateTestSignals(n: n)

    let noisyTensor = Tensor([noisy])  // [1, 64]
    let kernel = Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])  // 5-tap box filter

    let filtered = noisyTensor.conv2d(kernel)  // [1, 60]
    XCTAssertEqual(filtered.shape, [1, n - 4])

    let result = try filtered.realize()

    var mseFiltered: Float = 0
    var mseNoisy: Float = 0
    for i in 0..<(n - 4) {
      let cleanVal = clean[i + 2]  // align with conv center
      mseFiltered += (result[i] - cleanVal) * (result[i] - cleanVal)
      mseNoisy += (noisy[i + 2] - cleanVal) * (noisy[i + 2] - cleanVal)
    }
    mseFiltered /= Float(n - 4)
    mseNoisy /= Float(n - 4)

    print("=== Moving Average FIR Lowpass ===")
    print("MSE noisy vs clean:    \(String(format: "%.6f", mseNoisy))")
    print("MSE filtered vs clean: \(String(format: "%.6f", mseFiltered))")
    print("Noise reduction ratio: \(String(format: "%.2fx", mseNoisy / mseFiltered))")

    XCTAssertLessThan(mseFiltered, mseNoisy, "Filtering should reduce noise")
  }

  // MARK: - Signal.buffer() Tests

  /// buffer(4).sum() on a counter should give the sum of the last 4 values.
  func testBufferNumerical() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let buf = counter.buffer(size: 4)
    let s = buf.sum()
    let result = try s.realize(frames: 8)

    let expected: [Float] = [0, 1, 3, 6, 10, 14, 18, 22]
    print("\n=== Buffer Numerical Test ===")
    for i in 0..<8 {
      print("Frame \(i): sum=\(result[i]), expected=\(expected[i])")
    }
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.01, "Frame \(i) sum mismatch")
    }
  }

  /// Verify buffer element ORDER using a weighted sum (conv2d with asymmetric kernel).
  /// Plain sum can't distinguish [1,2,3,4] from [4,3,2,1], but a weighted sum can.
  func testBufferElementOrder() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let buf = counter.buffer(size: 4)  // [1, 4]

    // Weighted sum: kernel [1, 10, 100, 1000] encodes position into the result
    let kernel = Tensor([[1, 10, 100, 1000]])
    let weighted = buf.conv2d(kernel)  // [1, 1] per frame
    let result = try weighted.sum().realize(frames: 8)

    // Counter: 0, 1, 2, 3, 4, 5, 6, 7
    // slidingWindow: element i at frame f → base[f - 4 + 1 + i] = base[f - 3 + i]
    // So at frame f, buffer = [base[f-3], base[f-2], base[f-1], base[f]]
    // with base[k] = k for k >= 0, and 0 for k < 0 (out of bounds)
    //
    // Frame 0: [0, 0, 0, 0]       → 1*0 + 10*0 + 100*0 + 1000*0 = 0
    // Frame 1: [0, 0, 0, 1]       → 1*0 + 10*0 + 100*0 + 1000*1 = 1000
    // Frame 2: [0, 0, 1, 2]       → 1*0 + 10*0 + 100*1 + 1000*2 = 2100
    // Frame 3: [0, 1, 2, 3]       → 1*0 + 10*1 + 100*2 + 1000*3 = 3210
    // Frame 4: [1, 2, 3, 4]       → 1*1 + 10*2 + 100*3 + 1000*4 = 4321
    // Frame 5: [2, 3, 4, 5]       → 1*2 + 10*3 + 100*4 + 1000*5 = 5432
    // Frame 6: [3, 4, 5, 6]       → 1*3 + 10*4 + 100*5 + 1000*6 = 6543
    // Frame 7: [4, 5, 6, 7]       → 1*4 + 10*5 + 100*6 + 1000*7 = 7654
    let expected: [Float] = [0, 1000, 2100, 3210, 4321, 5432, 6543, 7654]

    print("\n=== Buffer Element Order Test ===")
    for i in 0..<8 {
      print("Frame \(i): weighted=\(result[i]), expected=\(expected[i])")
    }
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.5,
        "Frame \(i) weighted sum mismatch — buffer element order is wrong")
    }
  }

  /// buffer composes with conv2d: phasor -> buffer -> conv2d
  func testBufferConv2d() throws {
    let phasor = Signal.phasor(440.0)
    let sig = sin(phasor * Signal.constant(2.0 * .pi))
    let buf = sig.buffer(size: 128)

    let kernel = Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
    let filtered = buf.conv2d(kernel)  // [1, 124]

    let result = try filtered.realize(frames: 200)
    let tensorSize = 124

    let lastFrame = Array(result[(199 * tensorSize)..<(200 * tensorSize)])
    let maxVal = lastFrame.max() ?? 0
    let minVal = lastFrame.min() ?? 0

    print("\n=== Buffer Conv2d Test ===")
    print("Last frame: max=\(maxVal), min=\(minVal)")
    print("First 10 values: \(lastFrame.prefix(10).map { String(format: "%.4f", $0) })")

    XCTAssertGreaterThan(maxVal - minVal, 0.01, "Filtered output should have non-trivial variation")
  }

  /// Gradients flow through buffer -> conv2d to the kernel.
  func testBufferGradient() throws {
    let phasor = Signal.phasor(440.0)
    let sig = sin(phasor * Signal.constant(2.0 * .pi))
    let buf = sig.buffer(size: 32)

    let kernel = Tensor.param([1, 5], data: [0.2, 0.2, 0.2, 0.2, 0.2])
    let filtered = buf.conv2d(kernel)  // [1, 28]

    // Energy minimization: push all filtered values toward zero
    let loss = (filtered * filtered).sum()
    let lossValues = try loss.backward(frames: 64)

    let gradData = kernel.grad?.getData() ?? []
    print("\n=== Buffer Gradient Test ===")
    print("Loss: \(lossValues.last ?? 0)")
    print("Kernel grad: \(gradData.map { String(format: "%.6f", $0) })")

    XCTAssertFalse(gradData.isEmpty, "Kernel should have gradients")
    XCTAssertTrue(
      gradData.contains(where: { abs($0) > 1e-6 }), "At least one gradient should be non-zero")
  }

  // MARK: - Learned FIR Lowpass

  /// Train a conv2d kernel to act as a lowpass filter.
  /// MSE loss + regularization to keep kernel sum near 1.0 (unity DC gain).
  func testLearnedFIRLowpass() throws {
    let n = 64
    let kernelSize = 7
    let (clean, noisy) = generateTestSignals(n: n)

    let noisyTensor = Tensor([noisy])

    let offset = (kernelSize - 1) / 2
    let outLen = n - kernelSize + 1
    let cleanTrimmed = Array(clean[offset..<(offset + outLen)])
    let targetTensor = Tensor([cleanTrimmed])

    let learnedKernel = Tensor.param(
      [1, kernelSize], data: [Float](repeating: 1.0 / Float(kernelSize), count: kernelSize))

    func buildLoss() -> Tensor {
      let filtered = noisyTensor.conv2d(learnedKernel)
      let diff = filtered - targetTensor
      let mse = (diff * diff).mean()
      let sumDev = learnedKernel.sum() - Tensor([1.0])
      let reg = sumDev * sumDev * Tensor([0.1])
      return mse + reg
    }

    let optimizer = SGD(params: [learnedKernel], lr: 0.01)
    let epochs = 50

    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    print("\n=== Learned FIR Lowpass ===")
    print("Kernel size: \(kernelSize)")
    print("Initial loss: \(String(format: "%.6f", initialLoss))")

    var finalLoss = initialLoss

    for epoch in 0..<epochs {
      let loss = buildLoss()
      let lossValue = try loss.backward(frameCount: 1)[0]
      finalLoss = lossValue

      if epoch % 20 == 0 || epoch == epochs - 1 {
        let k = learnedKernel.getData() ?? []
        let g = learnedKernel.grad?.getData() ?? []
        print(
          "Epoch \(epoch): loss = \(String(format: "%.6f", lossValue)), kernel = \(k.map { String(format: "%.3f", $0) }), grad = \(g.map { String(format: "%.3f", $0) })"
        )
      }

      optimizer.step()

      // Project kernel: normalize to sum = 1.0 (unity DC gain)
      if var k = learnedKernel.getData() {
        let s = k.reduce(0, +)
        if abs(s) > 1e-8 { k = k.map { $0 / s } }
        learnedKernel.updateDataLazily(k)
      }

      optimizer.zeroGrad()
    }

    let finalKernel = learnedKernel.getData() ?? []
    let kernelSum = finalKernel.reduce(0, +)
    print("\nFinal kernel: \(finalKernel.map { String(format: "%.4f", $0) })")
    print("Kernel sum (should be ~1.0): \(String(format: "%.4f", kernelSum))")

    XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease")
    XCTAssertEqual(kernelSum, 1.0, accuracy: 0.3, "Lowpass kernel should have ~unity DC gain")
  }

  /// Forward-only: realize loss without backward to verify output readout.
  func testFIRLossForwardOnly() throws {
    let n = 64
    let kernelSize = 7
    let (clean, noisy) = generateTestSignals(n: n)

    let noisyTensor = Tensor([noisy])
    let offset = (kernelSize - 1) / 2
    let outLen = n - kernelSize + 1
    let cleanTrimmed = Array(clean[offset..<(offset + outLen)])
    let targetTensor = Tensor([cleanTrimmed])

    let learnedKernel = Tensor.param(
      [1, kernelSize], data: [Float](repeating: 1.0 / Float(kernelSize), count: kernelSize))

    let filtered = noisyTensor.conv2d(learnedKernel)
    let diff = filtered - targetTensor
    let mse = (diff * diff).mean()
    let sumDev = learnedKernel.sum() - Tensor([1.0])
    let reg = sumDev * sumDev * Tensor([0.1])
    let loss = mse + reg

    let result = try loss.realize()

    print("\n=== FIR Forward Only (no backward) ===")
    print("Loss realize() result: \(result)")
    XCTAssertFalse(result.isEmpty, "Should have a result")
    XCTAssertGreaterThan(result[0], 0, "Loss should be non-zero")
  }
}

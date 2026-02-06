import XCTest

@testable import DGenLazy

/// Tests for FIR (Finite Impulse Response) lowpass filtering using conv2d.
///
/// Key insight: An FIR filter is just a convolution. Each output sample is a
/// weighted sum of a window of *input* samples — no feedback, fully parallel.
/// By making the kernel learnable, the optimizer discovers filter coefficients.
final class FIRFilterTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Manual FIR Lowpass

  /// Verify that conv2d with a box filter (moving average) smooths a signal.
  /// A moving average is the simplest FIR lowpass — equal weights.
  func testMovingAverageLowpass() throws {
    let n = 64
    let freq: Float = 4.0  // 4 cycles across 64 samples

    // Build a sine wave with high-freq noise
    var clean = [Float](repeating: 0, count: n)
    var noisy = [Float](repeating: 0, count: n)
    for i in 0..<n {
      let phase = Float(i) / Float(n) * freq * 2.0 * .pi
      clean[i] = sin(phase)
      // Add high-frequency noise (25 cycles across n samples — avoids integer aliasing)
      let noise = 0.3 * sin(Float(i) / Float(n) * 25.0 * 2.0 * .pi)
      noisy[i] = clean[i] + noise
    }

    // Reshape as [1, N] for conv2d (faking 2D: height=1, width=N)
    let noisyTensor = Tensor([noisy])  // [1, 64]

    // 5-tap moving average kernel: [1, 5]
    let kernel = Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])

    let filtered = noisyTensor.conv2d(kernel)  // [1, 60]
    XCTAssertEqual(filtered.shape, [1, n - 4])

    let result = try filtered.realize()

    // Compare filtered output to the clean signal (trimmed to match conv output)
    // The moving average should reduce the noise, bringing us closer to clean
    var mseFiltered: Float = 0
    var mseNoisy: Float = 0
    // conv2d output is offset by (kernelSize-1)/2 = 2 samples
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

  /// Verify buffer + sum produces known per-frame values.
  /// accum(1, max: large) produces 0, 1, 2, 3, ...
  /// buffer(4).sum() should give the sum of the last 4 values written.
  func testBufferNumerical() throws {
    let counter = Signal.accum(Signal.constant(1.0), min: 0.0, max: 10000.0)
    let buf = counter.buffer(size: 4)  // SignalTensor [1, 4]
    let s = buf.sum()  // Signal
    let result = try s.realize(frames: 8)

    // Expected sums per frame (see derivation in test comments):
    let expected: [Float] = [0, 1, 3, 6, 10, 14, 18, 22]
    print("\n=== Buffer Numerical Test ===")
    for i in 0..<8 {
      print("Frame \(i): sum=\(result[i]), expected=\(expected[i])")
    }
    for i in 0..<8 {
      XCTAssertEqual(result[i], expected[i], accuracy: 0.01, "Frame \(i) sum mismatch")
    }
  }

  /// Verify buffer composes with conv2d: phasor → buffer → conv2d with a simple kernel
  func testBufferConv2d() throws {
    let phasor = Signal.phasor(440.0)
    let sig = sin(phasor * Signal.constant(2.0 * .pi))  // sine wave
    let buf = sig.buffer(size: 128)  // [1, 128]

    // Simple averaging kernel [1, 5]
    let kernel = Tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
    let filtered = buf.conv2d(kernel)  // SignalTensor [1, 124]

    // Run enough frames to fill the buffer
    let result = try filtered.realize(frames: 200)
    let tensorSize = 124  // 128 - 5 + 1

    // Check last frame output is non-trivial (buffer should be full of sine data)
    let lastFrame = Array(result[(199 * tensorSize)..<(200 * tensorSize)])
    let maxVal = lastFrame.max() ?? 0
    let minVal = lastFrame.min() ?? 0

    print("\n=== Buffer Conv2d Test ===")
    print("Last frame: max=\(maxVal), min=\(minVal)")
    print("First 10 values: \(lastFrame.prefix(10).map { String(format: "%.4f", $0) })")

    // After 200 frames at 440Hz (sr=44100), the buffer should have ~1 cycle of sine
    // The moving average should produce a smoothed but non-zero signal
    XCTAssertGreaterThan(maxVal - minVal, 0.01, "Filtered output should have non-trivial variation")
  }

  /// Verify gradients flow through buffer to a conv2d kernel.
  /// TODO(human): Design the loss function and target signal below.
  func testBufferGradient() throws {
    let phasor = Signal.phasor(440.0)
    let sig = sin(phasor * Signal.constant(2.0 * .pi))
    let buf = sig.buffer(size: 32)  // [1, 32]

    // Learnable kernel [1, 5]
    let kernel = Tensor.param([1, 5], data: [0.2, 0.2, 0.2, 0.2, 0.2])
    let filtered = buf.conv2d(kernel)  // SignalTensor [1, 28]

    // TODO(human): Design a loss that gives the optimizer a clear gradient signal.
    // `filtered` is the conv2d output, a SignalTensor [1, 28].
    // Return a scalar Signal loss.
    let loss = bufferGradLoss(filtered: filtered)

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
  /// The optimizer discovers FIR coefficients that minimize MSE against a clean signal.
  func testLearnedFIRLowpass() throws {
    let n = 64
    let freq: Float = 4.0
    let kernelSize = 7

    // Generate clean + noisy signals
    var clean = [Float](repeating: 0, count: n)
    var noisy = [Float](repeating: 0, count: n)
    for i in 0..<n {
      let phase = Float(i) / Float(n) * freq * 2.0 * .pi
      clean[i] = sin(phase)
      let noise = 0.3 * sin(Float(i) / Float(n) * 25.0 * 2.0 * .pi)
      noisy[i] = clean[i] + noise
    }

    let noisyTensor = Tensor([noisy])  // [1, 64]

    // Trim clean signal to match conv2d output size, centered
    let offset = (kernelSize - 1) / 2
    let outLen = n - kernelSize + 1
    let cleanTrimmed = Array(clean[offset..<(offset + outLen)])
    let targetTensor = Tensor([cleanTrimmed])  // [1, outLen]

    // TODO(human): Design the initial kernel data and choose the loss function.
    // Learnable FIR kernel — your design choice below:
    let learnedKernel = Tensor.param([1, kernelSize], data: firKernelInit(kernelSize))

    func buildLoss() -> Tensor {
      let filtered = noisyTensor.conv2d(learnedKernel)  // [1, outLen]
      return firLoss(prediction: filtered, target: targetTensor, kernel: learnedKernel)
    }

    let optimizer = SGD(params: [learnedKernel], lr: 0.01)
    let epochs = 50

    DGenConfig.kernelOutputPath = "/tmp/learn_fir_lowpass_loop.metal"
    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()
    DGenConfig.kernelOutputPath = nil

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

      // Project kernel onto constraint: sum = 1.0 (unity DC gain)
      if var k = learnedKernel.getData() {
        let s = k.reduce(0, +)
        if abs(s) > 1e-8 { k = k.map { $0 / s } }
        learnedKernel.updateDataLazily(k)
      }

      optimizer.zeroGrad()
    }

    // The learned kernel should have lowpass characteristics:
    // - Positive, roughly symmetric weights
    // - Weights sum to ~1.0 (unity gain at DC)
    let finalKernel = learnedKernel.getData() ?? []
    let kernelSum = finalKernel.reduce(0, +)
    print("\nFinal kernel: \(finalKernel.map { String(format: "%.4f", $0) })")
    print("Kernel sum (should be ~1.0): \(String(format: "%.4f", kernelSum))")

    XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease")
    XCTAssertEqual(kernelSum, 1.0, accuracy: 0.3, "Lowpass kernel should have ~unity DC gain")
  }

  /// Forward-only test: realize the loss without backward to isolate output readout.
  func testFIRLossForwardOnly() throws {
    let n = 64
    let freq: Float = 4.0
    let kernelSize = 7

    var clean = [Float](repeating: 0, count: n)
    var noisy = [Float](repeating: 0, count: n)
    for i in 0..<n {
      let phase = Float(i) / Float(n) * freq * 2.0 * .pi
      clean[i] = sin(phase)
      let noise = 0.3 * sin(Float(i) / Float(n) * 25.0 * 2.0 * .pi)
      noisy[i] = clean[i] + noise
    }

    let noisyTensor = Tensor([noisy])
    let offset = (kernelSize - 1) / 2
    let outLen = n - kernelSize + 1
    let cleanTrimmed = Array(clean[offset..<(offset + outLen)])
    let targetTensor = Tensor([cleanTrimmed])

    let learnedKernel = Tensor.param([1, kernelSize], data: firKernelInit(kernelSize))

    let filtered = noisyTensor.conv2d(learnedKernel)
    let loss = firLoss(prediction: filtered, target: targetTensor, kernel: learnedKernel)

    DGenConfig.kernelOutputPath = "/tmp/fir_forward_only.metal"
    let result = try loss.realize()
    DGenConfig.kernelOutputPath = nil

    print("\n=== FIR Forward Only (no backward) ===")
    print("Loss realize() result: \(result)")
    XCTAssertFalse(result.isEmpty, "Should have a result")
    XCTAssertGreaterThan(result[0], 0, "Loss should be non-zero")
  }
}

// MARK: - TODO(human): Design the FIR filter initialization and loss

/// Initialize FIR kernel weights and compute training loss.
///
/// firKernelInit: Return an array of `size` floats for starting kernel weights.
///   Options to consider: uniform (1/size each), triangular window, random small values
///
/// firLoss: Return a scalar Tensor loss given the filtered output and clean target.
///   Options: plain MSE, or MSE + a regularization term (e.g., penalize kernel sum != 1)
//
func firKernelInit(_ size: Int) -> [Float] {
  return [Float](repeating: 1.0 / Float(size), count: size)
}

func firLoss(prediction: Tensor, target: Tensor, kernel: Tensor) -> Tensor {
  let diff = prediction - target
  let mse = (diff * diff).mean()
  let reg = (kernel.sum() - Tensor([1.0])) * (kernel.sum() - Tensor([1.0])) * Tensor([0.1])
  return mse + reg
}

// MARK: - TODO(human): Design the buffer gradient test loss

/// Build a loss from the conv2d-filtered buffer output.
/// `filtered` is a SignalTensor [1, 28] — the result of convolving a sine wave
/// buffer with a learnable 5-tap kernel.
///
/// The goal: produce a scalar Signal loss that gives meaningful gradients
/// to the kernel weights. The optimizer should have a clear direction to move.
///
/// Options to consider:
/// - Sum-of-squares of the filtered output (minimize energy)
/// - MSE against a known target signal
/// - Difference from a desired magnitude (e.g., push mean toward a target value)
func bufferGradLoss(filtered: SignalTensor) -> Signal {
  // TODO(human): Replace this with your loss design.
  // Currently uses sum-of-squares (energy minimization) as a placeholder.
  let energy = (filtered * filtered).sum()
  return energy
}

import XCTest

@testable import DGenLazy

/// Tests for conv2d and the underlying windows (im2col) operation.
final class Conv2DLearningTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Windows (im2col) Tests
  //
  // These test the low-level `windows()` operation that extracts sliding windows
  // as extra dimensions. This is the building block for conv2d.

  /// Test that windows() extracts correct sliding window positions
  func testWindowsBasic() throws {
    // 3x3 input
    let input = Tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ])

    // Extract 2x2 windows -> [2, 2, 2, 2] (2x2 output positions, 2x2 kernel windows)
    let windows = input.windows([2, 2])

    XCTAssertEqual(windows.shape, [2, 2, 2, 2])

    let result = try windows.realize()
    print("Windows shape: \(windows.shape)")
    print("Windows data: \(result)")

    // Window at (0,0) should be top-left 2x2: [1,2,4,5]
    // Window at (0,1) should be top-right 2x2: [2,3,5,6]
    // Window at (1,0) should be bottom-left 2x2: [4,5,7,8]
    // Window at (1,1) should be bottom-right 2x2: [5,6,8,9]
    XCTAssertEqual(result[0], 1.0, accuracy: 1e-5)  // windows[0,0,0,0]
    XCTAssertEqual(result[1], 2.0, accuracy: 1e-5)  // windows[0,0,0,1]
  }

  /// Test manual convolution using windows + multiply + sum
  func testWindowsManualConvolution() throws {
    // 4x4 input
    let input = Tensor([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ])

    // 2x2 averaging kernel
    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    // Step 1: Extract windows [4,4] -> [3,3,2,2]
    let windows = input.windows([2, 2])
    XCTAssertEqual(windows.shape, [3, 3, 2, 2])

    // Step 2: Broadcast multiply with kernel
    // kernel [2,2] broadcasts to [3,3,2,2]
    let multiplied = windows * kernel

    // Step 3: Sum over kernel dimensions
    let sumKW = multiplied.sum(axis: -1)  // [3,3,2,2] -> [3,3,2]
    let sumKH = sumKW.sum(axis: -1)  // [3,3,2] -> [3,3]

    XCTAssertEqual(sumKH.shape, [3, 3])

    let result = try sumKH.realize()
    print("Manual convolution result shape: \(sumKH.shape)")
    print("Manual convolution result: \(result)")

    // First output position: avg of [1,2,5,6] = 3.5
    XCTAssertEqual(result[0], 3.5, accuracy: 1e-5)
  }

  // MARK: - Conv2D Function Tests
  //
  // These test the high-level conv2d() function that wraps windows + multiply + sum.

  /// Test conv2d forward pass produces correct output shape and values
  func testConv2DForward() throws {
    // 4x4 input
    let input = Tensor([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    ])

    // 2x2 averaging kernel (should compute local averages)
    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    // Use the high-level conv2d function
    let output = input.conv2d(kernel)

    // Output shape: [4-2+1, 4-2+1] = [3, 3]
    XCTAssertEqual(output.shape, [3, 3])

    let result = try output.realize()
    print("Conv2D output: \(result)")

    // First position: average of [1,2,5,6] = 3.5
    XCTAssertEqual(result[0], 3.5, accuracy: 1e-5)

    // Last position: average of [11,12,15,16] = 13.5
    XCTAssertEqual(result[8], 13.5, accuracy: 1e-5)
  }

  /// Test conv2d with identity kernel (center element = 1)
  func testConv2DIdentityKernel() throws {
    let input = Tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ])

    // 1x1 identity kernel - should just copy values
    let identityKernel = Tensor([[1.0]])
    let output = input.conv2d(identityKernel)

    XCTAssertEqual(output.shape, [3, 3])

    let result = try output.realize()

    // Should be identical to input
    XCTAssertEqual(result[0], 1.0, accuracy: 1e-5)
    XCTAssertEqual(result[4], 5.0, accuracy: 1e-5)  // center
    XCTAssertEqual(result[8], 9.0, accuracy: 1e-5)
  }

  /// Test conv2d with edge detection (Sobel-like) kernel
  func testConv2DEdgeDetection() throws {
    // Gradient image (increases left to right)
    let input = Tensor([
      [0.0, 1.0, 2.0, 3.0],
      [0.0, 1.0, 2.0, 3.0],
      [0.0, 1.0, 2.0, 3.0],
      [0.0, 1.0, 2.0, 3.0],
    ])

    // Horizontal edge detector: [-1, 1]
    let kernel = Tensor([[-1.0, 1.0]])
    let output = input.conv2d(kernel)

    // Output shape: [4, 3] (width reduced by 1)
    XCTAssertEqual(output.shape, [4, 3])

    let result = try output.realize()
    print("Edge detection result: \(result)")

    // All outputs should be 1.0 (constant horizontal gradient)
    for val in result {
      XCTAssertEqual(val, 1.0, accuracy: 1e-5)
    }
  }

  // MARK: - Conv2D Learning Tests
  //
  // These test that gradients flow correctly through conv2d for learning.

  /// Learn a kernel to match a target convolution output (element-wise MSE)
  func testConv2DLearnKernel() throws {
    DGenConfig.kernelOutputPath = "/tmp/test_conv2d_learn_kernel.metal"
    // Fixed input
    let input = Tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ])

    // Target kernel we want to learn
    let targetKernel = Tensor([
      [0.0, 1.0],
      [1.0, 0.0],
    ])

    // Learnable kernel (starts with wrong values)
    let learnedKernel = Tensor.param([2, 2], data: [0.25, 0.25, 0.25, 0.25])

    func buildLoss() -> Tensor {
      // Element-wise MSE: compare each output position
      let prediction = input.conv2d(learnedKernel)
      let target = input.conv2d(targetKernel)
      let diff = prediction - target
      return (diff * diff).sum()
    }

    let optimizer = SGD(params: [learnedKernel], lr: 0.002)
    let epochs = 200

    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    DGenConfig.kernelOutputPath = nil

    print("\n=== Learn Conv2D Kernel (element-wise MSE) ===")
    print("Target kernel: \(targetKernel.getData() ?? [])")
    print("Initial loss: \(initialLoss)")

    var finalLoss = initialLoss

    for epoch in 0..<epochs {
      let loss = buildLoss()
      let lossValue = try loss.backward(frameCount: 1)[0]
      finalLoss = lossValue

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let kernelData = learnedKernel.getData() ?? []
        print(
          "Epoch \(epoch): loss = \(String(format: "%.4f", lossValue)), kernel = \(kernelData.map { String(format: "%.3f", $0) })"
        )
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    XCTAssertLessThan(finalLoss, initialLoss * 0.1, "Loss should decrease significantly")
  }

  /// Learn a scale factor applied after convolution
  func testConv2DLearnScale() throws {
    let input = Tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ])

    // Fixed kernel
    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    let targetScale: Float = 2.0
    // Use same shape as conv output for element-wise multiply
    let learnedScale = Tensor.param([2, 2], data: [0.5, 0.5, 0.5, 0.5])

    func buildLoss() -> Tensor {
      let convOutput = input.conv2d(kernel)  // [2, 2]
      let scaled = convOutput * learnedScale
      let target = convOutput * targetScale
      let diff = scaled - target
      return (diff * diff).sum()
    }

    let optimizer = SGD(params: [learnedScale], lr: 0.01)
    let epochs = 50

    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    print("\n=== Learn Conv2D Scale ===")
    print("Target scale: \(targetScale)")
    print("Initial loss: \(initialLoss)")

    for epoch in 0..<epochs {
      let loss = buildLoss()
      let lossValue = try loss.backward(frameCount: 1)[0]

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let scale = learnedScale.getData()?[0] ?? 0
        print(
          "Epoch \(epoch): loss = \(String(format: "%.4f", lossValue)), scale = \(String(format: "%.3f", scale))"
        )
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    let finalScales = learnedScale.getData() ?? []
    let avgScale = finalScales.reduce(0, +) / Float(finalScales.count)
    XCTAssertEqual(avgScale, targetScale, accuracy: 0.1, "Should learn correct scale")
  }

  /// Learn Laplacian kernel coefficients for wave equation simulation (element-wise MSE)
  func testConv2DLearnLaplacianCoefficients() throws {
    // 5x5 input with a peak in the center
    let input = Tensor([
      [0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 1.0, 4.0, 1.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    // Target: standard Laplacian kernel (for wave equation: ∇²u)
    // This kernel computes the discrete Laplacian
    let targetKernel = Tensor([
      [0.0, 1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0, 1.0, 0.0],
    ])

    // Learnable kernel (starts at zero)
    let learnedKernel = Tensor.param([3, 3], data: [Float](repeating: 0.0, count: 9))

    func buildLoss() -> Tensor {
      // Element-wise MSE: compare each output position
      let prediction = input.conv2d(learnedKernel)
      let target = input.conv2d(targetKernel)
      let diff = prediction - target
      return (diff * diff).sum()
    }

    let optimizer = SGD(params: [learnedKernel], lr: 0.01)
    let epochs = 200

    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    print("\n=== Learn Laplacian Kernel ===")
    print("Target kernel: \(targetKernel.getData() ?? [])")
    print("Initial loss: \(initialLoss)")

    var finalLoss = initialLoss

    for epoch in 0..<epochs {
      let loss = buildLoss()
      let lossValue = try loss.backward(frameCount: 1)[0]
      finalLoss = lossValue

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let kernelData = learnedKernel.getData() ?? []
        print("Epoch \(epoch): loss = \(String(format: "%.6f", lossValue))")
        print("  Kernel: \(kernelData.map { String(format: "%.2f", $0) })")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    // Check that we learned something close to the Laplacian
    let finalKernel = learnedKernel.getData() ?? []
    let targetData = targetKernel.getData() ?? []

    print("\nFinal kernel vs target:")
    for i in 0..<min(finalKernel.count, targetData.count) {
      print(
        "  [\(i)]: learned=\(String(format: "%.2f", finalKernel[i])), target=\(String(format: "%.2f", targetData[i]))"
      )
    }

    XCTAssertLessThan(finalLoss, initialLoss * 0.01, "Loss should decrease significantly")
  }
}

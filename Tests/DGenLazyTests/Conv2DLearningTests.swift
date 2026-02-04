import XCTest

@testable import DGenLazy

/// Conv2D learning tests using im2col view + sum(axis:)
final class Conv2DLearningTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Basic conv2d view test

  /// Test that conv2d view extracts correct windows
  func testConv2DViewBasic() throws {
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
    print("Conv2D view windows shape: \(windows.shape)")
    print("Windows data: \(result)")

    // Window at (0,0) should be [1,2,4,5]
    // Window at (0,1) should be [2,3,5,6]
    // Window at (1,0) should be [4,5,7,8]
    // Window at (1,1) should be [5,6,8,9]
    XCTAssertEqual(result[0], 1.0, accuracy: 1e-5)  // windows[0,0,0,0]
    XCTAssertEqual(result[1], 2.0, accuracy: 1e-5)  // windows[0,0,0,1]
  }

  // MARK: - Full convolution using im2col

  /// Convolution via im2col: windows * kernel, then sum over kernel dims
  func testConv2DWithSum() throws {
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
    let sumKH = sumKW.sum(axis: -1)       // [3,3,2] -> [3,3]

    XCTAssertEqual(sumKH.shape, [3, 3])

    let result = try sumKH.realize()
    print("Convolution result shape: \(sumKH.shape)")
    print("Convolution result: \(result)")

    // First output position: avg of [1,2,5,6] = 3.5
    XCTAssertEqual(result[0], 3.5, accuracy: 1e-5)
  }

  // MARK: - Membrane-inspired: Learn wave speed

  /// Learn c² (wave speed) using im2col convolution
  /// Similar structure to testLearnKernel but learns a scalar multiplier
  func testLearnWaveSpeed() throws {
    // 3x3 input (asymmetric to produce non-zero conv output)
    let input = Tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ])

    // Fixed 2x2 averaging kernel
    let kernel = Tensor([
      [0.25, 0.25],
      [0.25, 0.25],
    ])

    // Target: convolution result scaled by target c²
    let targetCSquared: Float = 0.2

    // Learnable c² multiplier (starts wrong)
    // Shape [2,2] to match conv output - element-wise scaling
    let learnedScale = Tensor.param([2, 2], data: [0.05, 0.05, 0.05, 0.05])

    func buildLoss() -> Tensor {
      // im2col convolution
      let windows = input.windows([2, 2])  // [2, 2, 2, 2]
      let multiplied = windows * kernel
      let convResult = multiplied.sum(axis: -1).sum(axis: -1)  // [2, 2]

      // Scale by learned values
      let studentOutput = convResult * learnedScale
      let targetOutput = convResult * targetCSquared

      // MSE loss
      let diff = studentOutput - targetOutput
      return (diff * diff).sum()
    }

    let optimizer = SGD(params: [learnedScale], lr: 0.005)
    let epochs = 30

    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    print("\n=== Learn Scale Factor ===")
    print("Target scale: \(targetCSquared)")
    print("Initial loss: \(initialLoss)")

    var finalLoss = initialLoss

    for epoch in 0..<epochs {
      let loss = buildLoss()
      let lossValue = try loss.backward(frameCount: 1)[0]
      finalLoss = lossValue

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let scaleData = learnedScale.getData() ?? []
        let avgScale = scaleData.reduce(0, +) / Float(scaleData.count)
        print("Epoch \(epoch): loss = \(String(format: "%.6f", lossValue)), avg scale = \(String(format: "%.4f", avgScale))")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    let finalScale = learnedScale.getData()?[0] ?? 0
    print("Final scale: \(String(format: "%.4f", finalScale)) (target: \(targetCSquared))")

    XCTAssertEqual(finalScale, targetCSquared, accuracy: 0.02, "Should learn correct scale")
  }

  // MARK: - Learn a simple kernel

  /// Learn a kernel to match a target output
  func testLearnKernel() throws {
    // Simple 3x3 input
    let input = Tensor([
      [0.0, 1.0, 0.0],
      [1.0, 2.0, 1.0],
      [0.0, 1.0, 0.0],
    ])

    // Target: sum of input (a scalar we want to match)
    let target: Float = 6.0  // sum of input

    // Learnable 2x2 kernel
    let kernel = Tensor.param([2, 2], data: [0.1, 0.1, 0.1, 0.1])

    func buildLoss() -> Tensor {
      // im2col convolution
      let windows = input.windows([2, 2])  // [2, 2, 2, 2]
      let multiplied = windows * kernel
      let convResult = multiplied.sum(axis: -1).sum(axis: -1)  // [2, 2]

      // Sum all output positions to get a scalar prediction
      let prediction = convResult.sum()  // [1]

      // MSE loss
      let diff = prediction - target
      return diff * diff
    }

    let optimizer = SGD(params: [kernel], lr: 0.001)
    let epochs = 30

    // Use backward() instead of realize() to get gradients!
    let initialLossValues = try buildLoss().backward(frameCount: 1)
    let initialLoss = initialLossValues[0]
    optimizer.zeroGrad()

    print("\n=== Learn Kernel via im2col ===")
    print("Target sum: \(target)")
    print("Initial loss: \(initialLoss)")

    var finalLoss = initialLoss

    for epoch in 0..<epochs {
      let loss = buildLoss()
      let lossValues = try loss.backward(frameCount: 1)
      let lossValue = lossValues[0]
      finalLoss = lossValue

      let gradNorm = kernel.grad?.getData()?.map { $0 * $0 }.reduce(0, +) ?? 0

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let kernelData = kernel.getData() ?? []
        print("Epoch \(epoch): loss = \(String(format: "%.4f", lossValue)), grad²= \(String(format: "%.4f", gradNorm)), kernel = \(kernelData.map { String(format: "%.3f", $0) })")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("Final loss: \(String(format: "%.4f", finalLoss))")
    XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease significantly")
  }
}

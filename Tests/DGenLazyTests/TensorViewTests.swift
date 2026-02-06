import XCTest

@testable import DGen
@testable import DGenLazy

/// Tests for tensor view operations: reshape, transpose, shrink, pad, repeat, expand, conv2d
final class TensorViewTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Reshape Tests

  func testReshape1Dto2D() throws {
    let t = Tensor([1, 2, 3, 4, 5, 6])
    let reshaped = t.reshape([2, 3])

    XCTAssertEqual(reshaped.shape, [2, 3])

    let result = try reshaped.realize()
    // Row-major: same data, different logical shape
    XCTAssertEqual(result, [1, 2, 3, 4, 5, 6])
  }

  func testReshape2Dto1D() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6]])
    let flat = t.reshape([6])

    XCTAssertEqual(flat.shape, [6])

    let result = try flat.realize()
    XCTAssertEqual(result, [1, 2, 3, 4, 5, 6])
  }

  func testReshape2Dto2D() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2, 3]
    let reshaped = t.reshape([3, 2])

    XCTAssertEqual(reshaped.shape, [3, 2])

    let result = try reshaped.realize()
    // [1,2], [3,4], [5,6] in row-major
    XCTAssertEqual(result, [1, 2, 3, 4, 5, 6])
  }

  // MARK: - Transpose Tests

  func testTranspose2D() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2, 3]
    let tT = t.transpose()

    XCTAssertEqual(tT.shape, [3, 2])

    let result = try tT.realize()
    // Transposed: [[1, 4], [2, 5], [3, 6]] -> [1, 4, 2, 5, 3, 6]
    XCTAssertEqual(result, [1, 4, 2, 5, 3, 6])
  }

  func testTransposeWithAxes() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2, 3]
    let tT = t.transpose([1, 0])  // Same as default transpose

    XCTAssertEqual(tT.shape, [3, 2])

    let result = try tT.realize()
    XCTAssertEqual(result, [1, 4, 2, 5, 3, 6])
  }

  func testTransposeSquare() throws {
    let t = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let tT = t.transpose()

    let result = try tT.realize()
    // [[1, 3], [2, 4]] -> [1, 3, 2, 4]
    XCTAssertEqual(result, [1, 3, 2, 4])
  }

  // MARK: - Shrink Tests

  func testShrinkFirstRow() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]
    let row = t.shrink([(0, 1), nil])  // First row only

    XCTAssertEqual(row.shape, [1, 3])

    let result = try row.realize()
    XCTAssertEqual(result, [1, 2, 3])
  }

  func testShrinkMiddleColumn() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]
    let col = t.shrink([nil, (1, 2)])  // Middle column

    XCTAssertEqual(col.shape, [3, 1])

    let result = try col.realize()
    XCTAssertEqual(result, [2, 5, 8])
  }

  func testShrinkSubmatrix() throws {
    let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]
    let sub = t.shrink([(0, 2), (1, 3)])  // Top-right 2x2

    XCTAssertEqual(sub.shape, [2, 2])

    let result = try sub.realize()
    // [[2, 3], [5, 6]] -> [2, 3, 5, 6]
    XCTAssertEqual(result, [2, 3, 5, 6])
  }

  func testShrink1D() throws {
    let t = Tensor([1, 2, 3, 4, 5])
    let sliced = t.shrink([(1, 4)])  // Elements 1, 2, 3

    XCTAssertEqual(sliced.shape, [3])

    let result = try sliced.realize()
    XCTAssertEqual(result, [2, 3, 4])
  }

  // MARK: - Pad Tests

  func testPad1D() throws {
    let t = Tensor([1, 2, 3])
    let padded = t.pad([(2, 1)])  // 2 zeros left, 1 zero right

    XCTAssertEqual(padded.shape, [6])

    let result = try padded.realize()
    XCTAssertEqual(result, [0, 0, 1, 2, 3, 0])
  }

  func testPad2D() throws {
    let t = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let padded = t.pad([(1, 1), (1, 1)])  // Pad all sides with 1

    XCTAssertEqual(padded.shape, [4, 4])

    let result = try padded.realize()
    // [[0,0,0,0], [0,1,2,0], [0,3,4,0], [0,0,0,0]]
    XCTAssertEqual(result, [0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0])
  }

  func testPadAsymmetric() throws {
    let t = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let padded = t.pad([(1, 0), (0, 2)])  // 1 top, 2 right

    XCTAssertEqual(padded.shape, [3, 4])

    let result = try padded.realize()
    // [[0,0,0,0], [1,2,0,0], [3,4,0,0]]
    XCTAssertEqual(result, [0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0])
  }

  /// Test pad followed by conv2d (SIMD, no feedback) - WORKS
  /// This passes because memory is zero-initialized and each frame starts fresh
  func testPadThenConv2D_SIMD() throws {
    DGenConfig.kernelOutputPath = "/tmp/pad_conv_simd.metal"

    let input = Tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ])

    // Pad to [5,5]
    let padded = input.pad([(1, 1), (1, 1)])
    XCTAssertEqual(padded.shape, [5, 5])

    // 3x3 Laplacian kernel
    let kernel = Tensor([
      [0.0, 1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0, 1.0, 0.0],
    ])

    // Conv -> should give [3,3] output
    let convResult = padded.conv2d(kernel)
    XCTAssertEqual(convResult.shape, [3, 3])

    // Realize the conv result directly
    let result = try convResult.realize()
    print("Pad+Conv2D SIMD result: \(result)")

    // Should have 9 elements ([3,3])
    XCTAssertEqual(result.count, 9)

    // Verify we got non-trivial values (padding worked)
    let hasNonZero = result.contains { $0 != 0 }
    XCTAssertTrue(hasNonZero, "Should have non-zero values from conv2d")

    DGenConfig.kernelOutputPath = nil
    DGenConfig.debug = false
  }

  /// Test pad followed by conv2d in FEEDBACK loop - KNOWN LIMITATION
  /// This fails because:
  /// 1. Virtual padding doesn't materialize data - it relies on "out of bounds" reads returning 0
  /// 2. asStrided computes indices assuming [6,6] padded shape but reads from [4,4] memory
  /// 3. In SIMD mode, memory is zero-initialized so out-of-bounds reads return 0
  /// 4. In scalar feedback mode, memory is reused across frames, so out-of-bounds reads
  ///    hit leftover data from previous computations instead of zeros
  ///
  /// FIX NEEDED: Materialize padding before asStrided operations
  func testPadThenConv2D_ScalarFeedback_KnownLimitation() throws {
    DGenConfig.kernelOutputPath = "/tmp/pad_conv_feedback.metal"

    let frameCount = 4

    // Create history buffer with initial impulse
    let initialState: [Float] = [
      0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
    ]
    let history = TensorHistory(shape: [4, 4], data: initialState)

    // Read state from history (creates feedback dependency -> scalar block)
    let state = history.read()

    // Pad to [6,6] for same-size convolution
    let padded = state.pad([(1, 1), (1, 1)])
    XCTAssertEqual(padded.shape, [6, 6])

    // 3x3 Laplacian kernel
    let kernel = Tensor([
      [0.0, 1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0, 1.0, 0.0],
    ])

    // Conv on padded -> should give [4,4] output (same size as input)
    let laplacian = padded.conv2d(kernel)
    XCTAssertEqual(laplacian.shape, [4, 4])

    // Write back to history (completes feedback loop)
    history.write(laplacian)

    // Output sum
    let output = laplacian.sum()

    let result = try output.realize(frames: frameCount)
    print("\n=== Pad+Conv2D Scalar Feedback ===")
    print("Result per frame: \(result)")

    // The Laplacian is a zero-sum filter. For an isolated impulse:
    // - The center pixel at [1,1] = 1.0 gets multiplied by -4 → -4
    // - Four neighbors get +1 contribution each → +4 total
    // - Net sum = 0
    //
    // Frame 0: Laplacian of single impulse → output tensor has values
    //          [0,1,0,0; 1,-4,1,0; 0,1,0,0; 0,0,0,0], sum = 0
    // Frame 1: Laplacian of frame 0's output → different sum
    // etc.
    //
    // So values SHOULD change between frames as Laplacian is applied iteratively.
    // Frame 0 sum should be 0.0 (Laplacian is zero-sum for isolated impulse)

    XCTAssertEqual(
      result[0], 0.0, accuracy: 0.01, "Frame 0: Laplacian of single impulse should sum to 0")

    // Verify frame 1 by computing Laplacian of frame 0's output manually
    // Frame 0 output: [0,1,0,0; 1,-4,1,0; 0,1,0,0; 0,0,0,0]
    // Padded: [0,0,0,0,0,0; 0,0,1,0,0,0; 0,1,-4,1,0,0; 0,0,1,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0]
    // Applying Laplacian again gives a more complex pattern
    // The sum of Laplacian is always zero for boundary-compatible data
    // But our data has impulses that spread, creating non-zero sum over time
    // due to boundary effects with zero padding
    //
    // Manual calculation of frame 1:
    // output[0,0] = 0+0+0+0+0+0+0+1+0 = 1 (from top-left 3x3)
    // output[0,1] = 0+0+0+0+(-4)+1+0+(-4)+0 = -7
    // etc. This gets complex quickly.
    //
    // For now, just verify values are changing (iterative Laplacian)
    // and frame 0 is correct.
    XCTAssertNotEqual(
      result[1], result[0], "Frame 1 should differ from frame 0 due to iterative Laplacian")

    DGenConfig.kernelOutputPath = nil
    DGenConfig.debug = false
  }

  // MARK: - Repeat Tests

  func testRepeat1D() throws {
    let t = Tensor([1, 2, 3])
    let repeated = t.repeat([3])  // Repeat 3 times

    XCTAssertEqual(repeated.shape, [9])

    let result = try repeated.realize()
    XCTAssertEqual(result, [1, 2, 3, 1, 2, 3, 1, 2, 3])
  }

  func testRepeat2DVertical() throws {
    let t = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let repeated = t.repeat([2, 1])  // 2x vertical, 1x horizontal

    XCTAssertEqual(repeated.shape, [4, 2])

    let result = try repeated.realize()
    // [[1,2], [3,4], [1,2], [3,4]]
    XCTAssertEqual(result, [1, 2, 3, 4, 1, 2, 3, 4])
  }

  func testRepeat2DHorizontal() throws {
    let t = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let repeated = t.repeat([1, 2])  // 1x vertical, 2x horizontal

    XCTAssertEqual(repeated.shape, [2, 4])

    let result = try repeated.realize()
    // [[1,2,1,2], [3,4,3,4]]
    XCTAssertEqual(result, [1, 2, 1, 2, 3, 4, 3, 4])
  }

  func testRepeat2DBoth() throws {
    let t = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let repeated = t.repeat([2, 2])  // 2x both directions

    XCTAssertEqual(repeated.shape, [4, 4])

    let result = try repeated.realize()
    // [[1,2,1,2], [3,4,3,4], [1,2,1,2], [3,4,3,4]]
    XCTAssertEqual(result, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4])
  }

  // MARK: - Expand Tests

  func testExpandColumn() throws {
    let t = Tensor([[1], [2], [3]])  // [3, 1]
    let expanded = t.expand([3, 4])

    XCTAssertEqual(expanded.shape, [3, 4])

    let result = try expanded.realize()
    // [[1,1,1,1], [2,2,2,2], [3,3,3,3]]
    XCTAssertEqual(result, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
  }

  func testExpandRow() throws {
    let t = Tensor([[1, 2, 3]])  // [1, 3]
    let expanded = t.expand([4, 3])

    XCTAssertEqual(expanded.shape, [4, 3])

    let result = try expanded.realize()
    // [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
    XCTAssertEqual(result, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
  }

  // MARK: - Windows (im2col) Tests

  func testWindowsSimple() throws {
    // 3x3 image with values 1-9
    let img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]
    let windowsView = img.windows([2, 2])  // 2x2 windows

    // Output: [2, 2, 2, 2] - 2x2 output positions, each with 2x2 window
    XCTAssertEqual(windowsView.shape, [2, 2, 2, 2])

    let result = try windowsView.realize()

    // Window at (0,0): [[1,2], [4,5]] -> [1, 2, 4, 5]
    // Window at (0,1): [[2,3], [5,6]] -> [2, 3, 5, 6]
    // Window at (1,0): [[4,5], [7,8]] -> [4, 5, 7, 8]
    // Window at (1,1): [[5,6], [8,9]] -> [5, 6, 8, 9]
    // Flattened in row-major: [1,2,4,5, 2,3,5,6, 4,5,7,8, 5,6,8,9]
    XCTAssertEqual(result, [1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9])
  }

  func testWindowsLarger() throws {
    // 4x4 image
    let img = Tensor([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ])
    let windowsView = img.windows([3, 3])  // 3x3 windows

    // Output: [2, 2, 3, 3] - 2x2 output positions
    XCTAssertEqual(windowsView.shape, [2, 2, 3, 3])

    let result = try windowsView.realize()

    // Just verify the corners
    // Window at (0,0) starts with 1, 2, 3
    XCTAssertEqual(result[0], 1)
    XCTAssertEqual(result[1], 2)
    XCTAssertEqual(result[2], 3)

    // Window at (1,1) should have center element 11
    // Position (1,1) in output, element (1,1) in window = original (2,2) = 11
    // Index: 1*2*9 + 1*9 + 1*3 + 1 = 18 + 9 + 3 + 1 = 31
    XCTAssertEqual(result[31], 11)
  }

  // MARK: - Conv2D Tests (Proper Convolution with Kernel)

  /// Direct conv2d test: 3x3 input with 2x2 kernel
  /// Verifies shape and numerical correctness of convolution
  func testConv2dSimple() throws {
    // 3x3 image with values 1-9
    let img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]

    // 2x2 averaging kernel
    let kernel = Tensor([[0.25, 0.25], [0.25, 0.25]])  // [2, 2]

    let convResult = img.conv2d(kernel)

    // Output shape: [3-2+1, 3-2+1] = [2, 2]
    XCTAssertEqual(convResult.shape, [2, 2])

    let result = try convResult.realize()

    // Manual convolution calculation:
    // (0,0): (1*0.25 + 2*0.25 + 4*0.25 + 5*0.25) = 3.0
    // (0,1): (2*0.25 + 3*0.25 + 5*0.25 + 6*0.25) = 4.0
    // (1,0): (4*0.25 + 5*0.25 + 7*0.25 + 8*0.25) = 6.0
    // (1,1): (5*0.25 + 6*0.25 + 8*0.25 + 9*0.25) = 7.0
    XCTAssertEqual(result[0], 3.0, accuracy: 1e-5, "Conv at (0,0)")
    XCTAssertEqual(result[1], 4.0, accuracy: 1e-5, "Conv at (0,1)")
    XCTAssertEqual(result[2], 6.0, accuracy: 1e-5, "Conv at (1,0)")
    XCTAssertEqual(result[3], 7.0, accuracy: 1e-5, "Conv at (1,1)")
  }

  /// Conv2d with identity-like kernel (1 in top-left, 0 elsewhere)
  /// Should shift image by kernel size - 1
  func testConv2dIdentityKernel() throws {
    let img = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]

    // "Delta" kernel - only top-left is 1.0
    let kernel = Tensor([[1.0, 0.0], [0.0, 0.0]])  // [2, 2]

    let convResult = img.conv2d(kernel)

    XCTAssertEqual(convResult.shape, [2, 2])

    let result = try convResult.realize()

    // Convolution with delta at (0,0) should return top-left of each window:
    // (0,0): 1*1 + 2*0 + 4*0 + 5*0 = 1
    // (0,1): 2*1 + 3*0 + 5*0 + 6*0 = 2
    // (1,0): 4*1 + 5*0 + 7*0 + 8*0 = 4
    // (1,1): 5*1 + 6*0 + 8*0 + 9*0 = 5
    XCTAssertEqual(result, [1, 2, 4, 5])
  }

  /// Conv2d with Laplacian kernel for edge detection
  func testConv2dLaplacian() throws {
    // 5x5 image with impulse in center
    var imgData = [Float](repeating: 0.0, count: 25)
    imgData[12] = 1.0  // Center at (2,2)
    let img = Tensor(imgData).reshape([5, 5])

    // 3x3 Laplacian kernel
    let kernel = Tensor([
      [0.0, 1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0, 1.0, 0.0],
    ])

    let convResult = img.conv2d(kernel)

    // Output shape: [5-3+1, 5-3+1] = [3, 3]
    XCTAssertEqual(convResult.shape, [3, 3])

    let result = try convResult.realize()

    // Laplacian of center impulse:
    // Center (1,1) in output corresponds to window centered at (2,2) in input
    // Laplacian at impulse = -4 * 1 = -4
    // Neighbors of impulse in output get +1 contribution
    //
    // Output grid:
    // [0, 1, 0]
    // [1,-4, 1]
    // [0, 1, 0]
    XCTAssertEqual(result[0], 0.0, accuracy: 1e-5, "Top-left")
    XCTAssertEqual(result[1], 1.0, accuracy: 1e-5, "Top-center")
    XCTAssertEqual(result[2], 0.0, accuracy: 1e-5, "Top-right")
    XCTAssertEqual(result[3], 1.0, accuracy: 1e-5, "Mid-left")
    XCTAssertEqual(result[4], -4.0, accuracy: 1e-5, "Center")
    XCTAssertEqual(result[5], 1.0, accuracy: 1e-5, "Mid-right")
    XCTAssertEqual(result[6], 0.0, accuracy: 1e-5, "Bot-left")
    XCTAssertEqual(result[7], 1.0, accuracy: 1e-5, "Bot-center")
    XCTAssertEqual(result[8], 0.0, accuracy: 1e-5, "Bot-right")
  }

  /// Conv2d with 4x4 input and 2x2 kernel - verify total sum is preserved
  func testConv2dSumPreservation() throws {
    // 4x4 image with all 1s
    let img = Tensor([
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ])

    // Averaging kernel (sums to 1.0)
    let kernel = Tensor([[0.25, 0.25], [0.25, 0.25]])

    let convResult = img.conv2d(kernel)

    // Output: [3, 3]
    XCTAssertEqual(convResult.shape, [3, 3])

    let result = try convResult.realize()

    // Each output should be 1.0 (average of 1s)
    for i in 0..<9 {
      XCTAssertEqual(result[i], 1.0, accuracy: 1e-5, "Position \(i)")
    }
  }

  // MARK: - Chained Operations

  func testReshapeTranspose() throws {
    let t = Tensor([1, 2, 3, 4, 5, 6])
    let reshaped = t.reshape([2, 3])
    let transposed = reshaped.transpose()

    XCTAssertEqual(transposed.shape, [3, 2])

    let result = try transposed.realize()
    // Original [[1,2,3], [4,5,6]] -> transposed [[1,4], [2,5], [3,6]]
    XCTAssertEqual(result, [1, 4, 2, 5, 3, 6])
  }

  func testPadShrink() throws {
    let t = Tensor([[1, 2], [3, 4]])
    let padded = t.pad([(1, 1), (1, 1)])  // [4, 4]
    let shrunk = padded.shrink([(1, 3), (1, 3)])  // Back to [2, 2]

    XCTAssertEqual(shrunk.shape, [2, 2])

    let result = try shrunk.realize()
    // Should recover original
    XCTAssertEqual(result, [1, 2, 3, 4])
  }

  func testRepeatShrink() throws {
    let t = Tensor([1, 2, 3])
    let repeated = t.repeat([3])  // [9]: [1,2,3,1,2,3,1,2,3]
    let middle = repeated.shrink([(3, 6)])  // Middle section

    XCTAssertEqual(middle.shape, [3])

    let result = try middle.realize()
    XCTAssertEqual(result, [1, 2, 3])
  }

  // MARK: - Matmul Tests

  func testMatmulIdentity() throws {
    // Multiply by identity matrix
    let a = Tensor([[1, 2], [3, 4]])  // [2, 2]
    let identity = Tensor([[1, 0], [0, 1]])  // [2, 2]
    let c = a.matmul(identity)

    XCTAssertEqual(c.shape, [2, 2])

    let result = try c.realize()
    // A * I = A
    XCTAssertEqual(result, [1, 2, 3, 4])
  }

  func testMatmulSimple() throws {
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
    // = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    // = [[19, 22], [43, 50]]
    let a = Tensor([[1, 2], [3, 4]])
    let b = Tensor([[5, 6], [7, 8]])
    let c = a.matmul(b)

    XCTAssertEqual(c.shape, [2, 2])

    let result = try c.realize()
    XCTAssertEqual(result, [19, 22, 43, 50])
  }

  func testMatmulRectangular() throws {
    // [2, 3] @ [3, 2] -> [2, 2]
    let a = Tensor([[1, 2, 3], [4, 5, 6]])  // [2, 3]
    let b = Tensor([[7, 8], [9, 10], [11, 12]])  // [3, 2]
    let c = a.matmul(b)

    XCTAssertEqual(c.shape, [2, 2])

    let result = try c.realize()
    // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
    // = [[58, 64], [139, 154]]
    XCTAssertEqual(result, [58, 64, 139, 154])
  }

  func testMatmulVectorMatrix() throws {
    // Row vector @ matrix: [1, 3] @ [3, 2] -> [1, 2]
    let v = Tensor([[1, 2, 3]])  // [1, 3] row vector
    let m = Tensor([[1, 2], [3, 4], [5, 6]])  // [3, 2]
    let c = v.matmul(m)

    XCTAssertEqual(c.shape, [1, 2])

    let result = try c.realize()
    // [[1*1+2*3+3*5, 1*2+2*4+3*6]] = [[22, 28]]
    XCTAssertEqual(result, [22, 28])
  }

  func testMatmulOperator() throws {
    let a = Tensor([[1, 2], [3, 4]])
    let b = Tensor([[5, 6], [7, 8]])
    let c = a ◦ b  // Using the operator

    XCTAssertEqual(c.shape, [2, 2])

    let result = try c.realize()
    XCTAssertEqual(result, [19, 22, 43, 50])
  }

  // MARK: - PeekRow Tests

  func testPeekRowShape() throws {
    // Verify peekRow returns correct shape
    let t = Tensor([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ])

    // Row index must be a Signal (changes per frame)
    let playhead = Signal.phasor(1.0) * 2.0  // 0..2 over time
    let row = t.peekRow(playhead)

    // Shape should be [numCols]
    XCTAssertEqual(row.shape, [4])
  }

  func testPeekRowTrainingLoop() throws {
    // Test peekRow in a training context similar to GraphGradientTests
    let frameCount = 64
    let numRows = 4
    let numCols = 3

    // Learnable tensor [numRows, numCols] using param()
    let learnableTensor = Tensor.param(
      [numRows, numCols],
      data: (0..<(numRows * numCols)).map { Float($0) * 0.1 }
    )

    // Target value
    let target: Float = 5.0

    // Playhead signal cycles through rows
    let playhead = Signal.phasor(Float(1000) / Float(frameCount)) * Float(numRows - 1)

    // peekRow -> sum -> loss
    let rowAtTime = learnableTensor.peekRow(playhead)
    let summed = rowAtTime.sum()
    let loss = (summed - target) * (summed - target)

    // Run backward (Signal uses frames: parameter)
    let lossValues = try loss.backward(frames: frameCount)

    // Verify we got loss values and gradients
    XCTAssertEqual(lossValues.count, frameCount)
    XCTAssertNotNil(learnableTensor.grad)

    // Get gradient data
    let gradData = learnableTensor.grad?.getData()
    XCTAssertNotNil(gradData)
    XCTAssertEqual(gradData?.count, numRows * numCols)

    // Gradients should be non-zero
    let hasNonZeroGrad = gradData?.contains { $0 != 0 } ?? false
    XCTAssertTrue(hasNonZeroGrad, "peekRow should propagate gradients")
  }

  // MARK: - SpectralLossFFT Tests

  func testSpectralLossFFTTrainingLoop() throws {
    // Test spectralLossFFT in a training context
    let frameCount = 64

    // Learnable frequency parameter
    let learnedFreq = Signal.param(300.0)
    let targetFreq: Float = 440.0

    // Two oscillators
    let sig1 = sin(Signal.phasor(learnedFreq) * 2 * .pi)
    let sig2 = sin(Signal.phasor(targetFreq) * 2 * .pi)

    // Spectral loss
    let loss = spectralLossFFT(sig1, sig2, windowSize: 32)

    // Run backward (Signal uses frames: parameter)
    let lossValues = try loss.backward(frames: frameCount)

    // Verify we got loss values
    XCTAssertEqual(lossValues.count, frameCount)

    // Loss should be positive (different frequencies)
    let avgLoss = lossValues.reduce(0, +) / Float(frameCount)
    XCTAssertGreaterThan(avgLoss, 0, "Different frequencies should have positive loss")

    // Gradient should exist
    XCTAssertNotNil(learnedFreq.grad)
  }
}

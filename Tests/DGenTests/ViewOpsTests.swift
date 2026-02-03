import XCTest

@testable import DGen
@testable import DGenFrontend

/// Tests for view operations: expandView and pool
/// These operations manipulate tensor strides/shapes without copying data
final class ViewOpsTests: XCTestCase {

  // MARK: - Helper to run a graph and get output

  func runMetalGraph(
    _ graphBuilder: (Graph) throws -> Void,
    frameCount: Int = 1
  ) throws -> [Float] {
    let g = Graph()
    try graphBuilder(g)

    let result = try CompilationPipeline.compile(
      graph: g,
      backend: .metal,
      options: .init(frameCount: frameCount, debug: false)
    )

    let runtime = try MetalCompiledKernel(
      kernels: result.kernels,
      cellAllocations: result.cellAllocations,
      context: result.context
    )

    // Inject tensor data
    if let memBuffer = runtime.getBuffer(name: "memory") {
      let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
      injectTensorData(result: result, memory: memPtr)
    }

    runtime.runNoCopy(frameCount: frameCount)

    // Read outputs
    var outputs = [Float](repeating: 0, count: frameCount)
    if let outBuffer = runtime.getBuffer(name: "outputs") {
      let outPtr = outBuffer.contents().assumingMemoryBound(to: Float.self)
      for i in 0..<frameCount {
        outputs[i] = outPtr[i]
      }
    }

    return outputs
  }

  // MARK: - expandView Tests

  func testExpandViewBasic() throws {
    // Test: [2, 1, 3] expanded to [2, 4, 3]
    // Sum should be 4 * (1+2+3+4+5+6) = 84
    let outputs = try runMetalGraph { g in
      let input = g.tensor(shape: [2, 1, 3], data: [1, 2, 3, 4, 5, 6])
      let expanded = try g.expandView(input, to: [2, 4, 3])
      let sumResult = g.n(.sum, expanded)
      _ = g.n(.output(0), sumResult)
    }

    XCTAssertEqual(outputs[0], 84.0, accuracy: 1e-4, "expandView should broadcast via stride=0")
  }

  // MARK: - pool Tests with Peek Verification

  func testPoolVerifyEachElement() throws {
    // Test: [4, 4] with kernel [2, 2], stride [2, 2] -> [2, 2, 2, 2]
    //
    // Input:
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    //
    // After pool, shape is [oH=2, oW=2, kH=2, kW=2]
    // Position (0,0) window: [[1,2],[5,6]]   - top-left 2x2
    // Position (0,1) window: [[3,4],[7,8]]   - top-right 2x2
    // Position (1,0) window: [[9,10],[13,14]] - bottom-left 2x2
    // Position (1,1) window: [[11,12],[15,16]] - bottom-right 2x2
    //
    // We'll shrink to each single element and sum to verify

    // Expected values at each position [oH, oW, kH, kW]:
    let expectedValues: [(indices: [Int], value: Float)] = [
      // Window (0,0) - top-left
      ([0, 0, 0, 0], 1), ([0, 0, 0, 1], 2), ([0, 0, 1, 0], 5), ([0, 0, 1, 1], 6),
      // Window (0,1) - top-right
      ([0, 1, 0, 0], 3), ([0, 1, 0, 1], 4), ([0, 1, 1, 0], 7), ([0, 1, 1, 1], 8),
      // Window (1,0) - bottom-left
      ([1, 0, 0, 0], 9), ([1, 0, 0, 1], 10), ([1, 0, 1, 0], 13), ([1, 0, 1, 1], 14),
      // Window (1,1) - bottom-right
      ([1, 1, 0, 0], 11), ([1, 1, 0, 1], 12), ([1, 1, 1, 0], 15), ([1, 1, 1, 1], 16),
    ]

    for (indices, expectedValue) in expectedValues {
      let outputs = try runMetalGraph { g in
        let input = g.tensor(
          shape: [4, 4],
          data: [
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
          ])

        // Pool: [4,4] -> [2,2,2,2]
        let pooled = try g.pool(input, kernelSize: [2, 2], stride: [2, 2])

        // Shrink to single element at indices [oH, oW, kH, kW]
        // Each range is (start, start+1) to get exactly one element
        let singleElement = try g.shrink(pooled, ranges: [
          (indices[0], indices[0] + 1),
          (indices[1], indices[1] + 1),
          (indices[2], indices[2] + 1),
          (indices[3], indices[3] + 1),
        ])

        // Sum the 1-element tensor to get a scalar
        let value = g.n(.sum, singleElement)
        _ = g.n(.output(0), value)
      }

      XCTAssertEqual(
        outputs[0], expectedValue, accuracy: 1e-4,
        "pool[\(indices)] should be \(expectedValue), got \(outputs[0])"
      )
    }
  }

  func testPoolSumPreservesData() throws {
    // Verify pool doesn't lose data - sum before and after should match
    let outputs = try runMetalGraph { g in
      let input = g.tensor(
        shape: [4, 4],
        data: [
          1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16,
        ])

      let pooled = try g.pool(input, kernelSize: [2, 2], stride: [2, 2])
      let sumResult = g.n(.sum, pooled)
      _ = g.n(.output(0), sumResult)
    }

    // Sum of 1..16 = 136
    XCTAssertEqual(outputs[0], 136.0, accuracy: 1e-4, "pool should preserve all data")
  }

  func testPoolWithBatchDims() throws {
    // Test [2, 4, 4] -> [2, 2, 2, 2, 2] (batch dim preserved)
    let outputs = try runMetalGraph { g in
      // 2 batches, each 4x4 with values 1-16 and 17-32
      var data = [Float]()
      for i in 1...32 {
        data.append(Float(i))
      }
      let input = g.tensor(shape: [2, 4, 4], data: data)

      let pooled = try g.pool(input, kernelSize: [2, 2], stride: [2, 2])
      let sumResult = g.n(.sum, pooled)
      _ = g.n(.output(0), sumResult)
    }

    // Sum of 1..32 = 528
    XCTAssertEqual(outputs[0], 528.0, accuracy: 1e-4, "pool with batch dims should preserve all data")
  }

  func testExpandViewThenMul() throws {
    // Test using expandView for broadcasting in multiplication
    // a: [2, 1] = [[1], [2]]
    // b: [1, 3] = [[10, 20, 30]]
    // expand a to [2, 3], expand b to [2, 3]
    // multiply element-wise, sum should be (1*10 + 1*20 + 1*30) + (2*10 + 2*20 + 2*30)
    //                                    = 60 + 120 = 180
    let outputs = try runMetalGraph { g in
      let a = g.tensor(shape: [2, 1], data: [1, 2])
      let b = g.tensor(shape: [1, 3], data: [10, 20, 30])

      let aExpanded = try g.expandView(a, to: [2, 3])
      let bExpanded = try g.expandView(b, to: [2, 3])

      let product = g.n(.mul, aExpanded, bExpanded)
      let sumResult = g.n(.sum, product)
      _ = g.n(.output(0), sumResult)
    }

    XCTAssertEqual(outputs[0], 180.0, accuracy: 1e-4, "expandView should enable broadcasting for mul")
  }

  // MARK: - repeatView Tests

  func testRepeatViewBasic() throws {
    // Test: [2, 3] repeated by [2, 3] -> [4, 9]
    // Input has 6 elements, output "appears" to have 36
    // Each element appears 2*3 = 6 times
    // Sum should be 6 * (1+2+3+4+5+6) = 6 * 21 = 126
    let outputs = try runMetalGraph { g in
      let input = g.tensor(shape: [2, 3], data: [1, 2, 3, 4, 5, 6])
      let repeated = try g.repeatView(input, repeats: [2, 3])
      let sumResult = g.n(.sum, repeated)
      _ = g.n(.output(0), sumResult)
    }

    XCTAssertEqual(outputs[0], 126.0, accuracy: 1e-4, "repeatView should tile data via modular indexing")
  }

  func testRepeatViewVerifyElements() throws {
    // Test: [2, 2] repeated by [2, 2] -> [4, 4]
    // Input:     After repeat:
    //  1 2       1 2 1 2
    //  3 4       3 4 3 4
    //            1 2 1 2
    //            3 4 3 4
    //
    // Now with shrinkStart fix, shrink should compose correctly with repeat!

    let expectedValues: [(indices: [Int], value: Float)] = [
      // Row 0
      ([0, 0], 1), ([0, 1], 2), ([0, 2], 1), ([0, 3], 2),
      // Row 1
      ([1, 0], 3), ([1, 1], 4), ([1, 2], 3), ([1, 3], 4),
      // Row 2 (repeats row 0)
      ([2, 0], 1), ([2, 1], 2), ([2, 2], 1), ([2, 3], 2),
      // Row 3 (repeats row 1)
      ([3, 0], 3), ([3, 1], 4), ([3, 2], 3), ([3, 3], 4),
    ]

    for (indices, expectedValue) in expectedValues {
      let outputs = try runMetalGraph { g in
        let input = g.tensor(shape: [2, 2], data: [1, 2, 3, 4])
        let repeated = try g.repeatView(input, repeats: [2, 2])

        // Shrink to single element - now works with shrinkStart!
        let singleElement = try g.shrink(repeated, ranges: [
          (indices[0], indices[0] + 1),
          (indices[1], indices[1] + 1),
        ])

        let value = g.n(.sum, singleElement)
        _ = g.n(.output(0), value)
      }

      XCTAssertEqual(
        outputs[0], expectedValue, accuracy: 1e-4,
        "repeat[\(indices)] should be \(expectedValue), got \(outputs[0])"
      )
    }
  }

  func testRepeatViewForPoolOverlapping() throws {
    // Test repeat in the context of overlapping pooling
    // This simulates what we'd need for conv2d with stride < kernel
    //
    // For a 3x3 input with 2x2 kernel and stride 1:
    // - Output is 2x2 (4 positions)
    // - Each position needs a 2x2 window, but windows overlap
    //
    // We can use repeat to tile the input, then shrink/reshape/permute
    // For now, just verify repeat + sum preserves data * repeat_factor

    let outputs = try runMetalGraph { g in
      // 3x3 input
      let input = g.tensor(shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9])

      // Repeat by [2, 2] to get [6, 6]
      let repeated = try g.repeatView(input, repeats: [2, 2])

      // Sum should be 4 * (1+2+...+9) = 4 * 45 = 180
      let sumResult = g.n(.sum, repeated)
      _ = g.n(.output(0), sumResult)
    }

    XCTAssertEqual(outputs[0], 180.0, accuracy: 1e-4, "repeat [2,2] should quadruple the sum")
  }

  // MARK: - Overlapping Pool Tests (kernel > stride)

  func testOverlappingPool1D() throws {
    // 1D overlapping pool: input [4], kernel 2, stride 1
    // Output shape: [3, 2] (3 windows of size 2)
    //
    // Input: [a, b, c, d] = [1, 2, 3, 4]
    // Windows:
    //   o=0: [1, 2]
    //   o=1: [2, 3]  <- overlaps with o=0
    //   o=2: [3, 4]  <- overlaps with o=1
    //
    // Sum of all windows = (1+2) + (2+3) + (3+4) = 3 + 5 + 7 = 15
    // Note: elements 2 and 3 are counted twice due to overlap

    let outputs = try runMetalGraph { g in
      let input = g.tensor(shape: [4], data: [1, 2, 3, 4])
      let pooled = try g.pool(input, kernelSize: [2], stride: [1])
      let sumResult = g.n(.sum, pooled)
      _ = g.n(.output(0), sumResult)
    }

    XCTAssertEqual(outputs[0], 15.0, accuracy: 1e-4, "overlapping 1D pool sum")
  }

  func testOverlappingPool2D() throws {
    // 2D overlapping pool: input [3, 3], kernel [2, 2], stride [1, 1]
    // Output shape: [2, 2, 2, 2] (2x2 output positions, each with 2x2 window)
    //
    // Input:
    //   1 2 3
    //   4 5 6
    //   7 8 9
    //
    // Windows (each 2x2):
    //   (0,0): [1,2,4,5]  (0,1): [2,3,5,6]
    //   (1,0): [4,5,7,8]  (1,1): [5,6,8,9]
    //
    // Sum = (1+2+4+5) + (2+3+5+6) + (4+5+7+8) + (5+6+8+9)
    //     = 12 + 16 + 24 + 28 = 80
    // Center element (5) appears in all 4 windows!

    let outputs = try runMetalGraph { g in
      let input = g.tensor(shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9])
      let pooled = try g.pool(input, kernelSize: [2, 2], stride: [1, 1])
      let sumResult = g.n(.sum, pooled)
      _ = g.n(.output(0), sumResult)
    }

    XCTAssertEqual(outputs[0], 80.0, accuracy: 1e-4, "overlapping 2D pool sum")
  }

  func testOverlappingPoolVerifyElements() throws {
    // Verify specific elements in overlapping pool
    // Input [3, 3], kernel [2, 2], stride [1, 1] -> output [2, 2, 2, 2]
    //
    // Input:
    //   1 2 3
    //   4 5 6
    //   7 8 9

    let expectedValues: [(indices: [Int], value: Float)] = [
      // Window (0,0) - top-left
      ([0, 0, 0, 0], 1), ([0, 0, 0, 1], 2), ([0, 0, 1, 0], 4), ([0, 0, 1, 1], 5),
      // Window (0,1) - top-right (shifted by stride=1 in W)
      ([0, 1, 0, 0], 2), ([0, 1, 0, 1], 3), ([0, 1, 1, 0], 5), ([0, 1, 1, 1], 6),
      // Window (1,0) - bottom-left (shifted by stride=1 in H)
      ([1, 0, 0, 0], 4), ([1, 0, 0, 1], 5), ([1, 0, 1, 0], 7), ([1, 0, 1, 1], 8),
      // Window (1,1) - bottom-right
      ([1, 1, 0, 0], 5), ([1, 1, 0, 1], 6), ([1, 1, 1, 0], 8), ([1, 1, 1, 1], 9),
    ]

    for (indices, expectedValue) in expectedValues {
      let outputs = try runMetalGraph { g in
        let input = g.tensor(shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9])
        let pooled = try g.pool(input, kernelSize: [2, 2], stride: [1, 1])

        // Shrink to single element
        let singleElement = try g.shrink(pooled, ranges: [
          (indices[0], indices[0] + 1),
          (indices[1], indices[1] + 1),
          (indices[2], indices[2] + 1),
          (indices[3], indices[3] + 1),
        ])

        let value = g.n(.sum, singleElement)
        _ = g.n(.output(0), value)
      }

      XCTAssertEqual(
        outputs[0], expectedValue, accuracy: 1e-4,
        "overlapping pool[\(indices)] should be \(expectedValue), got \(outputs[0])"
      )
    }
  }

  // MARK: - conv2dView Tests

  func testConv2dViewIdentityKernel() throws {
    // Test with a kernel that picks the top-left element of each window
    // Kernel: [[1, 0], [0, 0]] - only top-left contributes
    //
    // Input [3, 3]:     Output [2, 2] (stride 1):
    //   1 2 3           1 2
    //   4 5 6           4 5
    //   7 8 9
    //
    // Each output is just the top-left of its 2x2 window

    let expectedValues: [(indices: [Int], value: Float)] = [
      ([0, 0], 1), ([0, 1], 2), ([1, 0], 4), ([1, 1], 5),
    ]

    for (indices, expectedValue) in expectedValues {
      let outputs = try runMetalGraph { g in
        let input = g.tensor(shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9])
        let kernel = g.tensor(shape: [2, 2], data: [1, 0, 0, 0])

        let result = try g.conv2dView(input, kernel: kernel, stride: [1, 1])

        // Shrink to single element and sum
        let singleElement = try g.shrink(result, ranges: [
          (indices[0], indices[0] + 1),
          (indices[1], indices[1] + 1),
        ])
        let value = g.n(.sum, singleElement)
        _ = g.n(.output(0), value)
      }

      XCTAssertEqual(
        outputs[0], expectedValue, accuracy: 1e-4,
        "conv2d[\(indices)] with identity kernel should be \(expectedValue)"
      )
    }
  }

  func testConv2dViewSumKernel() throws {
    // Test with all-ones kernel (box filter / sum)
    // Kernel: [[1, 1], [1, 1]] - sums all elements in window
    //
    // Input [3, 3]:     Windows:
    //   1 2 3           [0,0]: 1+2+4+5=12   [0,1]: 2+3+5+6=16
    //   4 5 6           [1,0]: 4+5+7+8=24   [1,1]: 5+6+8+9=28
    //   7 8 9

    let expectedValues: [(indices: [Int], value: Float)] = [
      ([0, 0], 12), ([0, 1], 16), ([1, 0], 24), ([1, 1], 28),
    ]

    for (indices, expectedValue) in expectedValues {
      let outputs = try runMetalGraph { g in
        let input = g.tensor(shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9])
        let kernel = g.tensor(shape: [2, 2], data: [1, 1, 1, 1])

        let result = try g.conv2dView(input, kernel: kernel, stride: [1, 1])

        let singleElement = try g.shrink(result, ranges: [
          (indices[0], indices[0] + 1),
          (indices[1], indices[1] + 1),
        ])
        let value = g.n(.sum, singleElement)
        _ = g.n(.output(0), value)
      }

      XCTAssertEqual(
        outputs[0], expectedValue, accuracy: 1e-4,
        "conv2d[\(indices)] box filter should be \(expectedValue)"
      )
    }
  }

  func testConv2dViewEdgeDetect() throws {
    // Test with edge detection kernel (Sobel-like horizontal)
    // Kernel: [[-1, 1], [-1, 1]] - detects vertical edges
    //
    // Uniform input should give 0 (no edges)
    // Input: all 5s

    let outputs = try runMetalGraph { g in
      let input = g.tensor(shape: [3, 3], data: [5, 5, 5, 5, 5, 5, 5, 5, 5])
      let kernel = g.tensor(shape: [2, 2], data: [-1, 1, -1, 1])

      let result = try g.conv2dView(input, kernel: kernel, stride: [1, 1])
      let sumResult = g.n(.sum, result)
      _ = g.n(.output(0), sumResult)
    }

    // All outputs should be 0 (uniform input, edge detector)
    XCTAssertEqual(outputs[0], 0.0, accuracy: 1e-4, "edge detect on uniform input")
  }

  func testConv2dViewStride2() throws {
    // Test with stride 2 (non-overlapping)
    // Input [4, 4], kernel [2, 2], stride [2, 2] -> output [2, 2]
    //
    // Input:              Kernel (all 1s):
    //   1  2  3  4        1 1
    //   5  6  7  8        1 1
    //   9 10 11 12
    //  13 14 15 16
    //
    // Windows (non-overlapping):
    //   [0,0]: 1+2+5+6=14     [0,1]: 3+4+7+8=22
    //   [1,0]: 9+10+13+14=46  [1,1]: 11+12+15+16=54

    let expectedValues: [(indices: [Int], value: Float)] = [
      ([0, 0], 14), ([0, 1], 22), ([1, 0], 46), ([1, 1], 54),
    ]

    for (indices, expectedValue) in expectedValues {
      let outputs = try runMetalGraph { g in
        var data = [Float]()
        for i in 1...16 { data.append(Float(i)) }

        let input = g.tensor(shape: [4, 4], data: data)
        let kernel = g.tensor(shape: [2, 2], data: [1, 1, 1, 1])

        let result = try g.conv2dView(input, kernel: kernel, stride: [2, 2])

        let singleElement = try g.shrink(result, ranges: [
          (indices[0], indices[0] + 1),
          (indices[1], indices[1] + 1),
        ])
        let value = g.n(.sum, singleElement)
        _ = g.n(.output(0), value)
      }

      XCTAssertEqual(
        outputs[0], expectedValue, accuracy: 1e-4,
        "conv2d stride2 [\(indices)] should be \(expectedValue)"
      )
    }
  }

  // MARK: - conv2dView Training/Gradient Tests

  func testConv2dViewGradientFlow() throws {
    // Test that gradients flow through conv2dView
    // Train a kernel to match a target output
    //
    // Setup:
    //   - Fixed input [3, 3]
    //   - Learnable kernel [2, 2] (starts at zeros)
    //   - Target: conv2d output should sum to 12
    //
    // The kernel should learn values that make the output sum to 12

    let g = Graph()

    // Fixed input
    let input = g.tensor(shape: [3, 3], data: [1, 2, 3, 4, 5, 6, 7, 8, 9])

    // Learnable kernel - starts at small random-ish values
    let kernelParam = TensorParameter(
      graph: g,
      shape: [2, 2],
      data: [0.1, 0.1, 0.1, 0.1],
      name: "kernel"
    )

    // conv2dView: [3, 3] with [2, 2] kernel, stride [1, 1] -> [2, 2]
    let convResult = try g.conv2dView(input, kernel: kernelParam.node(), stride: [1, 1])

    // Sum the conv output to scalar
    let outputSum = g.n(.sum, convResult)

    // Target: we want the sum to be 80 (same as all-ones kernel)
    // With kernel [1,1,1,1], windows sum to [12, 16, 24, 28] = 80 total
    let target = g.n(.constant(80.0))

    // MSE loss
    let loss = g.n(.mse, [outputSum, target])
    _ = g.n(.output(0), loss)

    // Create training context with very small learning rate (gradients are large due to broadcasting)
    let ctx = try GraphTrainingContext(
      graph: g,
      loss: loss,
      tensorParameters: [kernelParam],
      optimizer: GraphSGD(),
      learningRate: 0.00001,  // Small LR - gradients are ~4k magnitude
      frameCount: 1,
      kernelDebugOutput: "/tmp/conv2d_view_gradient.metal"
    )

    print("\n=== conv2dView Gradient Flow Test ===")
    print("Initial kernel: \(kernelParam.data)")

    // Get initial loss (first trainStep)
    let initialLoss = ctx.trainStep(fullReset: true)
    print("Initial loss: \(initialLoss)")

    // Train for several steps (need more steps due to small learning rate)
    var finalLoss = initialLoss
    for step in 0..<500 {
      finalLoss = ctx.trainStep(fullReset: true)
      if step % 40 == 0 {
        let grads = ctx.getTensorGradients()
        let gradMax = grads.map { abs($0) }.max() ?? 0
        print("Step \(step): loss = \(String(format: "%.2f", finalLoss)), kernel[0] = \(String(format: "%.4f", kernelParam.data[0])), gradMax = \(String(format: "%.2f", gradMax))")
      }
    }

    print("Final loss: \(finalLoss)")
    print("Final kernel: \(kernelParam.data)")

    // Verify:
    // 1. Loss decreased
    XCTAssertLessThan(finalLoss, initialLoss, "Loss should decrease during training")

    // 2. Gradients were non-zero (kernel values changed from initial)
    let kernelChanged = kernelParam.data != [0.1, 0.1, 0.1, 0.1]
    XCTAssertTrue(kernelChanged, "Kernel should have been updated by gradients")

    // 3. Loss is small (converged close to target)
    XCTAssertLessThan(finalLoss, 1.0, "Loss should be small after training")
  }

  // MARK: - Isolated Gradient Tests

  func testSumAxisGradient() throws {
    // Simple test: [2,2] tensor -> sumAxis(1) -> [2] -> sum -> scalar
    // Gradient should flow back correctly
    let g = Graph()

    // Learnable tensor [2,2] = [[1,2],[3,4]]
    let param = TensorParameter(
      graph: g,
      shape: [2, 2],
      data: [1, 2, 3, 4],
      name: "param"
    )

    // sumAxis along axis 1: [[1,2],[3,4]] -> [3, 7]
    let summed = try g.sum(param.node(), axis: 1)  // [2]

    // sum to scalar: 3 + 7 = 10
    let total = g.n(.sum, [summed])

    // MSE with target 20: (10-20)^2 = 100
    let target = g.n(.constant(20.0))
    let loss = g.n(.mse, [total, target])
    _ = g.n(.output(0), loss)

    let ctx = try GraphTrainingContext(
      graph: g,
      loss: loss,
      tensorParameters: [param],
      optimizer: GraphSGD(),
      learningRate: 0.01,
      frameCount: 1,
      kernelDebugOutput: "/tmp/sumaxis_grad.metal"
    )

    print("\n=== sumAxis Gradient Test ===")
    let initialLoss = ctx.trainStep(fullReset: true)
    print("Initial loss: \(initialLoss)")  // Should be 100

    let grads = ctx.getTensorGradients()
    print("Gradients: \(grads)")
    // Expected: d(loss)/d(param) = 2*(10-20)*1 = -20 for each element
    // Because sumAxis and sum both have gradient 1 for each input

    let gradMax = grads.map { abs($0) }.max() ?? 0
    print("Grad max: \(gradMax)")

    // Each element should have gradient -20
    XCTAssertEqual(gradMax, 20.0, accuracy: 1.0, "Gradient should be ~20")
  }

  func testExpandViewGradient() throws {
    // Simple test: [1,2] tensor -> expandView to [2,2] -> sum -> scalar
    // Gradient should sum back along expanded dimension
    let g = Graph()

    // Learnable tensor [1,2] = [[1,2]]
    let param = TensorParameter(
      graph: g,
      shape: [1, 2],
      data: [1, 2],
      name: "param"
    )

    // expandView to [2,2]: [[1,2],[1,2]]
    let expanded = try g.expandView(param.node(), to: [2, 2])

    // sum to scalar: 1+2+1+2 = 6
    let total = g.n(.sum, [expanded])

    // MSE with target 10: (6-10)^2 = 16
    let target = g.n(.constant(10.0))
    let loss = g.n(.mse, [total, target])
    _ = g.n(.output(0), loss)

    let ctx = try GraphTrainingContext(
      graph: g,
      loss: loss,
      tensorParameters: [param],
      optimizer: GraphSGD(),
      learningRate: 0.01,
      frameCount: 1,
      kernelDebugOutput: "/tmp/expandview_grad.metal"
    )

    print("\n=== expandView Gradient Test ===")
    let initialLoss = ctx.trainStep(fullReset: true)
    print("Initial loss: \(initialLoss)")  // Should be 16

    let grads = ctx.getTensorGradients()
    print("Gradients: \(grads)")
    // d(loss)/d(total) = 2*(6-10) = -8
    // Each expanded element contributes 1 to the sum, so upstream grad is -8
    // But param[0,0] appears twice (expanded), so grad = -8 * 2 = -16
    // Similarly param[0,1] appears twice, so grad = -8 * 2 = -16

    let gradMax = grads.map { abs($0) }.max() ?? 0
    print("Grad max: \(gradMax)")

    // Each element should have gradient -16 (sum of 2 copies * -8)
    XCTAssertEqual(gradMax, 16.0, accuracy: 1.0, "Gradient should be ~16")
  }

  func testMulExpandViewGradient() throws {
    // Test mul where one input is expanded: pool * expandedKernel
    // This is the core of conv2dView
    let g = Graph()

    // Fixed "pool" tensor [2,2] = [[1,2],[3,4]]
    let pool = g.tensor(shape: [2, 2], data: [1, 2, 3, 4])

    // Learnable kernel [1,2] = [[0.1, 0.1]] -> expandView to [2,2]
    let kernelParam = TensorParameter(
      graph: g,
      shape: [1, 2],
      data: [0.1, 0.1],
      name: "kernel"
    )
    let expandedKernel = try g.expandView(kernelParam.node(), to: [2, 2])

    // Multiply: pool * expandedKernel
    let product = g.n(.mul, [pool, expandedKernel])

    // Sum to scalar
    let total = g.n(.sum, [product])

    // MSE with target 10
    let target = g.n(.constant(10.0))
    let loss = g.n(.mse, [total, target])
    _ = g.n(.output(0), loss)

    let ctx = try GraphTrainingContext(
      graph: g,
      loss: loss,
      tensorParameters: [kernelParam],
      optimizer: GraphSGD(),
      learningRate: 0.001,
      frameCount: 1,
      kernelDebugOutput: "/tmp/mul_expandview_grad.metal"
    )

    print("\n=== mul + expandView Gradient Test ===")
    // Forward: pool * 0.1 = [0.1, 0.2, 0.3, 0.4], sum = 1.0
    // Loss = (1.0 - 10)^2 = 81
    let initialLoss = ctx.trainStep(fullReset: true)
    print("Initial loss: \(initialLoss)")

    let grads = ctx.getTensorGradients()
    print("Gradients: \(grads)")
    // d(loss)/d(total) = 2*(1-10) = -18
    // d(total)/d(product[i,j]) = 1
    // d(product)/d(expandedKernel[i,j]) = pool[i,j]
    // d(expandedKernel)/d(kernel[0,j]) = sum over i (expanded dim)
    // So grad_kernel[0,0] = -18 * (pool[0,0] + pool[1,0]) = -18 * (1+3) = -72
    //    grad_kernel[0,1] = -18 * (pool[0,1] + pool[1,1]) = -18 * (2+4) = -108

    let gradMax = grads.map { abs($0) }.max() ?? 0
    print("Grad max: \(gradMax)")
    print("Expected grads: [-72, -108]")

    XCTAssertEqual(grads[0], -72.0, accuracy: 1.0, "kernel[0,0] grad should be ~-72")
    XCTAssertEqual(grads[1], -108.0, accuracy: 1.0, "kernel[0,1] grad should be ~-108")
  }

  func testConv2dViewStyleGradient() throws {
    // Test the exact pattern of conv2dView but with fixed pool data
    // pool [2,2,2,2] * kernel [1,1,2,2] expanded to [2,2,2,2] -> sumAxis x2 -> [2,2]
    let g = Graph()

    // Fixed "pool" tensor [2,2,2,2] - simulating overlapping windows
    // Each [oH,oW,:,:] is a 2x2 window
    let poolData: [Float] = [
      // [0,0,:,:] = window at (0,0)
      1, 2, 4, 5,
      // [0,1,:,:] = window at (0,1)
      2, 3, 5, 6,
      // [1,0,:,:] = window at (1,0)
      4, 5, 7, 8,
      // [1,1,:,:] = window at (1,1)
      5, 6, 8, 9
    ]
    let pool = g.tensor(shape: [2, 2, 2, 2], data: poolData)

    // Learnable kernel [2,2] -> reshape to [1,1,2,2] -> expandView to [2,2,2,2]
    let kernelParam = TensorParameter(
      graph: g,
      shape: [2, 2],
      data: [0.1, 0.1, 0.1, 0.1],
      name: "kernel"
    )

    // Reshape kernel to [1,1,2,2]
    let reshapedKernel = try g.reshape(kernelParam.node(), to: [1, 1, 2, 2])

    // Expand to [2,2,2,2]
    let expandedKernel = try g.expandView(reshapedKernel, to: [2, 2, 2, 2])

    // Multiply
    let product = g.n(.mul, [pool, expandedKernel])

    // sumAxis(-1) then sumAxis(-1) to get [2,2]
    let sum1 = try g.sum(product, axis: -1)  // [2,2,2]
    let sum2 = try g.sum(sum1, axis: -1)     // [2,2]

    // Sum to scalar
    let total = g.n(.sum, [sum2])

    // Target 80 (same as original test)
    let target = g.n(.constant(80.0))
    let loss = g.n(.mse, [total, target])
    _ = g.n(.output(0), loss)

    let ctx = try GraphTrainingContext(
      graph: g,
      loss: loss,
      tensorParameters: [kernelParam],
      optimizer: GraphSGD(),
      learningRate: 0.000001,
      frameCount: 1,
      kernelDebugOutput: "/tmp/conv2d_style_grad.metal"
    )

    print("\n=== conv2dView-style Gradient Test ===")
    // Forward with kernel [0.1,0.1,0.1,0.1]:
    // Each window: sum(pool[oH,oW,:,:] * 0.1) = 0.1 * sum(window)
    // Window sums: 12, 16, 24, 28 -> conv outputs: 1.2, 1.6, 2.4, 2.8
    // Total: 8.0, Loss: (8-80)^2 = 5184
    let initialLoss = ctx.trainStep(fullReset: true)
    print("Initial loss: \(initialLoss)")

    let grads = ctx.getTensorGradients()
    print("Gradients: \(grads)")

    // Expected gradient calculation:
    // d(loss)/d(total) = 2*(8-80) = -144
    // Upstream to each sum2[oH,oW] = -144 (expanded from scalar)
    // Upstream to each sum1[oH,oW,kH] = -144 (expanded)
    // Upstream to each product[oH,oW,kH,kW] = -144
    // d(product)/d(expandedKernel) = pool
    // d(expandedKernel)/d(kernel[kH,kW]) = sum over oH,oW
    // grad_kernel[kH,kW] = -144 * sum(pool[:,:,kH,kW])
    // pool[:,:,0,0] = [1,2,4,5] -> sum=12 -> grad=-1728
    // pool[:,:,0,1] = [2,3,5,6] -> sum=16 -> grad=-2304
    // pool[:,:,1,0] = [4,5,7,8] -> sum=24 -> grad=-3456
    // pool[:,:,1,1] = [5,6,8,9] -> sum=28 -> grad=-4032

    let gradMax = grads.map { abs($0) }.max() ?? 0
    print("Grad max: \(gradMax)")
    print("Expected grads: [-1728, -2304, -3456, -4032]")
    print("Expected max: 4032")

    XCTAssertEqual(gradMax, 4032.0, accuracy: 100.0, "Gradient max should be ~4032, not ~98000")
  }
}

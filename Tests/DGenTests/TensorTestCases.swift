import XCTest

@testable import DGen

/// Shared test case definitions for tensor operations.
/// Each test case defines a graph and expected output values.
/// Both C and Metal backends can run these same tests.

struct TensorTestCase {
  let name: String
  let frameCount: Int
  let expectedOutputs: [Float]?
  let accuracy: Float
  let graphBuilder: (Graph) throws -> Void
  let injectData: Bool
  let customValidator: (([Float]) -> Void)?

  init(
    name: String,
    frameCount: Int = 1,
    expectedOutputs: [Float],
    accuracy: Float = 0.001,
    injectData: Bool = true,
    graphBuilder: @escaping (Graph) throws -> Void
  ) {
    self.name = name
    self.frameCount = frameCount
    self.expectedOutputs = expectedOutputs
    self.accuracy = accuracy
    self.injectData = injectData
    self.graphBuilder = graphBuilder
    self.customValidator = nil
  }

  /// For tests that need custom validation instead of exact expected values
  init(
    name: String,
    frameCount: Int,
    injectData: Bool = true,
    graphBuilder: @escaping (Graph) throws -> Void,
    validator: @escaping ([Float]) -> Void
  ) {
    self.name = name
    self.frameCount = frameCount
    self.expectedOutputs = nil
    self.accuracy = 0.001
    self.injectData = injectData
    self.graphBuilder = graphBuilder
    self.customValidator = validator
  }
}

/// All tensor operation test cases
enum TensorTestCases {

  // MARK: - Basic Operations

  static let sumReduceExecution = TensorTestCase(
    name: "SumReduceExecution",
    expectedOutputs: [10.0],
    graphBuilder: { g in
      let tensorNode = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
      let sumResult = g.n(.sum, tensorNode)
      _ = g.n(.output(0), sumResult)
    }
  )

  static let tensorAddScalarExecution = TensorTestCase(
    name: "TensorAddScalarExecution",
    expectedOutputs: [30.0],
    graphBuilder: { g in
      let tensorNode = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
      let scalar = g.n(.constant(5.0))
      let result = g.n(.add, tensorNode, scalar)
      let sumResult = g.n(.sum, result)
      _ = g.n(.output(0), sumResult)
    }
  )

  static let tensorMulTensor = TensorTestCase(
    name: "TensorMulTensor",
    expectedOutputs: [40.0],
    graphBuilder: { g in
      let t1 = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
      let t2 = g.tensor(shape: [2, 2], data: [2.0, 3.0, 4.0, 5.0])
      let result = g.n(.mul, t1, t2)
      let sumResult = g.n(.sum, result)
      _ = g.n(.output(0), sumResult)
    }
  )

  static let broadcastScalarTensor = TensorTestCase(
    name: "BroadcastScalarTensor",
    expectedOutputs: [81.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let scalar = g.n(.constant(10.0))
      let added = g.n(.add, t, scalar)
      let result = g.n(.sum, added)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Tensor History

  static let tensorHistoryExecution = TensorTestCase(
    name: "TensorHistoryExecution",
    frameCount: 4,
    expectedOutputs: [4.0, 8.0, 12.0, 16.0],
    injectData: false,
    graphBuilder: { g in
      let stateBuffer = g.tensorHistoryBuffer(shape: [2, 2])
      let state = g.tensorHistoryRead(stateBuffer)
      let newState = g.n(.add, state, g.n(.constant(1.0)))
      g.tensorHistoryWrite(stateBuffer, newState)
      _ = g.n(.output(0), g.n(.sum, newState))
    }
  )

  // MARK: - Conv2d

  static let conv2dExecution = TensorTestCase(
    name: "Conv2dExecution",
    expectedOutputs: [9.0],
    graphBuilder: { g in
      let inputNode = g.ones(shape: [3, 3])
      let kernelNode = g.tensor(
        shape: [3, 3],
        data: [
          0, 0, 0,
          0, 1, 0,
          0, 0, 0,
        ])
      let convResult = g.n(.conv2d([3, 3]), inputNode, kernelNode)
      let sumResult = g.n(.sum, convResult)
      _ = g.n(.output(0), sumResult)
    }
  )

  // MARK: - Conv1d

  static let conv1dExecution = TensorTestCase(
    name: "Conv1dExecution",
    expectedOutputs: [39.0],
    graphBuilder: { g in
      let inputNode = g.tensor(shape: [5], data: [1, 2, 3, 4, 5])
      let kernelNode = g.tensor(shape: [3], data: [1, 1, 1])
      let convResult = g.n(.conv1d(3), inputNode, kernelNode)
      let sumResult = g.n(.sum, convResult)
      _ = g.n(.output(0), sumResult)
    }
  )

  static let conv1dIdentityKernel = TensorTestCase(
    name: "Conv1dIdentityKernel",
    expectedOutputs: [15.0],
    graphBuilder: { g in
      let inputNode = g.tensor(shape: [5], data: [1, 2, 3, 4, 5])
      let kernelNode = g.tensor(shape: [3], data: [0, 1, 0])
      let convResult = g.n(.conv1d(3), inputNode, kernelNode)
      let sumResult = g.n(.sum, convResult)
      _ = g.n(.output(0), sumResult)
    }
  )

  // MARK: - Reshape

  static let reshapeExecution = TensorTestCase(
    name: "ReshapeExecution",
    expectedOutputs: [21.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let reshaped = try g.reshape(t, to: [3, 2])
      let result = g.n(.sum, reshaped)
      _ = g.n(.output(0), result)
    }
  )

  static let reshapeToFlat = TensorTestCase(
    name: "ReshapeToFlat",
    expectedOutputs: [91.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let flat = try g.reshape(t, to: [6])
      let weights = g.tensor(shape: [6], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let weighted = g.n(.mul, flat, weights)
      let result = g.n(.sum, weighted)
      _ = g.n(.output(0), result)
    }
  )

  static let reshapeFromFlat = TensorTestCase(
    name: "ReshapeFromFlat",
    expectedOutputs: [91.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [6], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let reshaped = try g.reshape(t, to: [2, 3])
      let weights = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let weighted = g.n(.mul, reshaped, weights)
      let result = g.n(.sum, weighted)
      _ = g.n(.output(0), result)
    }
  )

  static let reshapeThenSumAxisExecution = TensorTestCase(
    name: "ReshapeThenSumAxisExecution",
    expectedOutputs: [50.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let reshaped = try g.reshape(t, to: [3, 2])
      let summed = try g.sum(reshaped, axis: 1)
      let weights = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
      let weighted = g.n(.mul, summed, weights)
      let result = g.n(.sum, weighted)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - SumAxis

  static let sumAxisExecution = TensorTestCase(
    name: "SumAxisExecution",
    expectedOutputs: [21.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let summed = try g.sum(t, axis: -1)
      let result = g.n(.sum, summed)
      _ = g.n(.output(0), result)
    }
  )

  static let sumAxisAxis0 = TensorTestCase(
    name: "SumAxisAxis0",
    expectedOutputs: [46.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let summed = try g.sum(t, axis: 0)
      let weights = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
      let weighted = g.n(.mul, summed, weights)
      let result = g.n(.sum, weighted)
      _ = g.n(.output(0), result)
    }
  )

  static let sumAxisToScalar = TensorTestCase(
    name: "SumAxisToScalar",
    expectedOutputs: [10.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [4], data: [1.0, 2.0, 3.0, 4.0])
      let summed = try g.sum(t, axis: 0)
      _ = g.n(.output(0), summed)
    }
  )

  static let nestedParallelRangeDebug = TensorTestCase(
    name: "NestedParallelRangeDebug",
    expectedOutputs: [78.0],
    graphBuilder: { g in
      let t = g.tensor(
        shape: [4, 3],
        data: [
          1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          7.0, 8.0, 9.0,
          10.0, 11.0, 12.0,
        ])
      let summed = try g.sum(t, axis: 1)
      let result = g.n(.sum, summed)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Matmul

  static let matmulExecution = TensorTestCase(
    name: "MatmulExecution",
    expectedOutputs: [415.0],
    graphBuilder: { g in
      let a = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let b = g.tensor(shape: [3, 2], data: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
      let c = try g.matmul(a, b)
      let result = g.n(.sum, c)
      _ = g.n(.output(0), result)
    }
  )

  static let matmulWithScalarMul = TensorTestCase(
    name: "MatmulWithScalarMul",
    expectedOutputs: [830.0],
    graphBuilder: { g in
      let a = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let b = g.tensor(shape: [3, 2], data: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
      let scalar = g.n(.constant(2.0))
      let bScaled = g.n(.mul, b, scalar)
      let c = try g.matmul(a, bScaled)
      let result = g.n(.sum, c)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Transpose

  static let transposeExecution = TensorTestCase(
    name: "TransposeExecution",
    expectedOutputs: [46.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 3], data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      let transposed = try g.transpose(t)
      let summed = try g.sum(transposed, axis: 1)
      let weights = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
      let weighted = g.n(.mul, summed, weights)
      let result = g.n(.sum, weighted)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Pad

  static let padExecution = TensorTestCase(
    name: "PadExecution",
    expectedOutputs: [10.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
      let padded = try g.pad(t, padding: [(1, 1), (1, 1)])
      let result = g.n(.sum, padded)
      _ = g.n(.output(0), result)
    }
  )

  static let padAsymmetric = TensorTestCase(
    name: "PadAsymmetric",
    expectedOutputs: [6.0],
    graphBuilder: { g in
      let t = g.tensor(shape: [3], data: [1.0, 2.0, 3.0])
      let padded = try g.pad(t, padding: [(2, 1)])
      let result = g.n(.sum, padded)
      _ = g.n(.output(0), result)
    }
  )

  static let concatViaPadAndAdd = TensorTestCase(
    name: "ConcatViaPadAndAdd",
    expectedOutputs: [10.0],
    graphBuilder: { g in
      let t1 = g.tensor(shape: [2], data: [1.0, 2.0])
      let t2 = g.tensor(shape: [2], data: [3.0, 4.0])
      let t1Padded = try g.pad(t1, padding: [(0, 2)])
      let t2Padded = try g.pad(t2, padding: [(2, 0)])
      let concat = g.n(.add, t1Padded, t2Padded)
      let result = g.n(.sum, concat)
      _ = g.n(.output(0), result)
    }
  )

  static let concat2DViaPadAndAdd = TensorTestCase(
    name: "Concat2DViaPadAndAdd",
    expectedOutputs: [36.0],
    graphBuilder: { g in
      let t1 = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
      let t2 = g.tensor(shape: [2, 2], data: [5.0, 6.0, 7.0, 8.0])
      let t1Padded = try g.pad(t1, padding: [(0, 2), (0, 0)])
      let t2Padded = try g.pad(t2, padding: [(2, 0), (0, 0)])
      let concat = g.n(.add, t1Padded, t2Padded)
      let result = g.n(.sum, concat)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Shrink

  static let shrinkExecution = TensorTestCase(
    name: "ShrinkExecution",
    expectedOutputs: [34.0],
    graphBuilder: { g in
      let t = g.tensor(
        shape: [4, 4],
        data: [
          1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16,
        ])
      let shrunk = try g.shrink(t, ranges: [(1, 3), (1, 3)])
      let result = g.n(.sum, shrunk)
      _ = g.n(.output(0), result)
    }
  )

  static let shrinkColumnOnly = TensorTestCase(
    name: "ShrinkColumnOnly",
    expectedOutputs: [90.0],
    graphBuilder: { g in
      let t = g.tensor(
        shape: [3, 6],
        data: [
          1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
          7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
          13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ])
      let shrunk = try g.shrink(t, ranges: [nil, (2, 5)])
      let result = g.n(.sum, shrunk)
      _ = g.n(.output(0), result)
    }
  )

  static let shrinkWithScalarOp = TensorTestCase(
    name: "ShrinkWithScalarOp",
    expectedOutputs: [68.0],
    graphBuilder: { g in
      let t = g.tensor(
        shape: [4, 4],
        data: [
          1.0, 2.0, 3.0, 4.0,
          5.0, 6.0, 7.0, 8.0,
          9.0, 10.0, 11.0, 12.0,
          13.0, 14.0, 15.0, 16.0,
        ])
      let shrunk = try g.shrink(t, ranges: [(1, 3), (1, 3)])
      let scaled = g.n(.mul, shrunk, g.n(.constant(2.0)))
      let result = g.n(.sum, scaled)
      _ = g.n(.output(0), result)
    }
  )

  static let shrinkWithSumAxis = TensorTestCase(
    name: "ShrinkWithSumAxis",
    expectedOutputs: [110.0],
    graphBuilder: { g in
      let t = g.tensor(
        shape: [4, 4],
        data: [
          1.0, 2.0, 3.0, 4.0,
          5.0, 6.0, 7.0, 8.0,
          9.0, 10.0, 11.0, 12.0,
          13.0, 14.0, 15.0, 16.0,
        ])
      let shrunk = try g.shrink(t, ranges: [(1, 3), nil])
      let summed = try g.sum(shrunk, axis: 1)
      let weights = g.tensor(shape: [2], data: [1.0, 2.0])
      let weighted = g.n(.mul, summed, weights)
      let result = g.n(.sum, weighted)
      _ = g.n(.output(0), result)
    }
  )

  static let chainedShrink = TensorTestCase(
    name: "ChainedShrink",
    expectedOutputs: [74.0],
    graphBuilder: { g in
      var data = [Float]()
      for i in 0..<36 { data.append(Float(i + 1)) }
      let t = g.tensor(shape: [6, 6], data: data)
      let shrunk1 = try g.shrink(t, ranges: [(1, 5), (1, 5)])
      let shrunk2 = try g.shrink(shrunk1, ranges: [(1, 3), (1, 3)])
      let result = g.n(.sum, shrunk2)
      _ = g.n(.output(0), result)
    }
  )

  static let shrinkWithBroadcastOp = TensorTestCase(
    name: "ShrinkWithBroadcastOp",
    expectedOutputs: [68.0],
    graphBuilder: { g in
      let a = g.tensor(
        shape: [4, 4],
        data: [
          1.0, 2.0, 3.0, 4.0,
          5.0, 6.0, 7.0, 8.0,
          9.0, 10.0, 11.0, 12.0,
          13.0, 14.0, 15.0, 16.0,
        ])
      let b = g.tensor(shape: [4, 4], data: [Float](repeating: 2.0, count: 16))
      let shrunkA = try g.shrink(a, ranges: [(1, 3), (1, 3)])
      let shrunkB = try g.shrink(b, ranges: [(0, 2), (0, 2)])
      let product = g.n(.mul, shrunkA, shrunkB)
      let result = g.n(.sum, product)
      _ = g.n(.output(0), result)
    }
  )

  static let shrinkThenConv2d = TensorTestCase(
    name: "ShrinkThenConv2d",
    expectedOutputs: [4.0],
    graphBuilder: { g in
      // 4x4 tensor where first 2x2 block is zeros, rest is ones
      let t = g.tensor(
        shape: [4, 4],
        data: [
          0.0, 0.0, 1.0, 1.0,
          0.0, 0.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0,
          1.0, 1.0, 1.0, 1.0,
        ])
      // Shrink to bottom-right 2x2 which should be all 1s
      let shrunk = try g.shrink(t, ranges: [(2, 4), (2, 4)])
      // Identity kernel
      let kernel = g.tensor(
        shape: [3, 3],
        data: [
          0.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 0.0,
        ])
      let convResult = g.n(.conv2d([3, 3]), shrunk, kernel)
      let result = g.n(.sum, convResult)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Stack

  static let stackBasic = TensorTestCase(
    name: "StackBasic",
    expectedOutputs: [10.0],
    graphBuilder: { g in
      let s1 = g.n(.constant(1.0))
      let s2 = g.n(.constant(2.0))
      let s3 = g.n(.constant(3.0))
      let s4 = g.n(.constant(4.0))
      let stacked = try g.stack([s1, s2, s3, s4])
      let result = g.n(.sum, stacked)
      _ = g.n(.output(0), result)
    }
  )

  static let stackWithShape = TensorTestCase(
    name: "StackWithShape",
    expectedOutputs: [10.0],
    graphBuilder: { g in
      let s1 = g.n(.constant(1.0))
      let s2 = g.n(.constant(2.0))
      let s3 = g.n(.constant(3.0))
      let s4 = g.n(.constant(4.0))
      let stacked = try g.stack([s1, s2, s3, s4], shape: [2, 2])
      let result = g.n(.sum, stacked)
      _ = g.n(.output(0), result)
    }
  )

  // MARK: - Latch

  static let latchWithTensorInputs = TensorTestCase(
    name: "LatchWithTensorInputs",
    frameCount: 10,
    injectData: true,
    graphBuilder: { g in
      let values = g.tensor(shape: [2, 2], data: [1.0, 2.0, 3.0, 4.0])
      let trigger = g.n(.constant(1.0))
      let latchCell = g.alloc()
      let latchNode = g.n(.latch(latchCell), values, trigger)
      let result = g.n(.sum, latchNode)
      _ = g.n(.output(0), result)
    },
    validator: { output in
      // With constant trigger=1, latch captures values immediately
      // Frame 0: returns old latched value (0), Frame 1+: returns 10
      XCTAssertEqual(output[1], 10.0, accuracy: 0.001, "Latch should capture tensor values")
    }
  )

  // MARK: - Phasor with Tensor

  static let phasorWithTensorFrequencies = TensorTestCase(
    name: "PhasorWithTensorFrequencies",
    frameCount: 100,
    injectData: true,
    graphBuilder: { g in
      let freqs = g.tensor(shape: [2, 2], data: [100.0, 200.0, 300.0, 400.0])
      let cellId = g.alloc()
      let zeroReset = g.n(.constant(0.0))
      let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)
      let result = g.n(.sum, phasorNode)
      _ = g.n(.output(0), result)
    },
    validator: { output in
      XCTAssertGreaterThan(output[50], output[0], "Phasors should accumulate")
    }
  )

  // MARK: - Accum with Tensor

  static let accumWithTensorInputs = TensorTestCase(
    name: "AccumWithTensorInputs",
    frameCount: 100,
    injectData: true,
    graphBuilder: { g in
      let increments = g.tensor(shape: [2, 2], data: [0.1, 0.2, 0.3, 0.4])
      let reset = g.n(.constant(0.0))
      let min = g.n(.constant(0.0))
      let max = g.n(.constant(10.0))
      let cellId = g.alloc()
      let accumNode = g.n(.accum(cellId), increments, reset, min, max)
      let result = g.n(.sum, accumNode)
      _ = g.n(.output(0), result)
    },
    validator: { output in
      XCTAssertGreaterThan(output[50], output[10], "Accum should accumulate over time")
    }
  )

  // MARK: - Cos Phasor Tensor

  static let cosPhasorTensor = TensorTestCase(
    name: "CosPhasorTensor",
    frameCount: 441,
    injectData: true,
    graphBuilder: { g in
      let freqs = g.tensor(shape: [2, 2], data: [100.0, 200.0, 300.0, 400.0])
      let cellId = g.alloc()
      let zeroReset = g.n(.constant(0.0))
      let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)
      let twopi = g.n(.constant(Float.pi * 2.0))
      let scaled = g.n(.mul, phasorNode, twopi)
      let cosNode = g.n(.cos, scaled)
      let result = g.n(.sum, cosNode)
      _ = g.n(.output(0), result)
    },
    validator: { output in
      let hasNaN = output.contains { $0.isNaN }
      let hasInf = output.contains { $0.isInfinite }
      XCTAssertFalse(hasNaN, "Output should not contain NaN")
      XCTAssertFalse(hasInf, "Output should not contain Inf")
      let minVal = output.min() ?? 0
      let maxVal = output.max() ?? 0
      XCTAssertGreaterThanOrEqual(minVal, -4.1, "Sum of 4 cos should be >= -4")
      XCTAssertLessThanOrEqual(maxVal, 4.1, "Sum of 4 cos should be <= 4")
    }
  )

  static let cosPhasorTensorLarge = TensorTestCase(
    name: "CosPhasorTensorLarge",
    frameCount: 4096,
    injectData: true,
    graphBuilder: { g in
      let freqs = g.tensor(
        shape: [4, 4],
        data: [
          50.0, 100.0, 150.0, 200.0,
          250.0, 300.0, 350.0, 400.0,
          450.0, 500.0, 600.0, 700.0,
          800.0, 1000.0, 1500.0, 2000.0,
        ])
      let cellId = g.alloc()
      let zeroReset = g.n(.constant(0.0))
      let phasorNode = g.n(.phasor(cellId), freqs, zeroReset)
      let twopi = g.n(.constant(Float.pi * 2.0))
      let scaled = g.n(.mul, phasorNode, twopi)
      let cosNode = g.n(.cos, scaled)
      let result = g.n(.sum, cosNode)
      _ = g.n(.output(0), result)
    },
    validator: { output in
      let hasNaN = output.contains { $0.isNaN }
      let hasInf = output.contains { $0.isInfinite }
      XCTAssertFalse(hasNaN, "Output should not contain NaN")
      XCTAssertFalse(hasInf, "Output should not contain Inf")
      let minVal = output.min() ?? 0
      let maxVal = output.max() ?? 0
      XCTAssertGreaterThanOrEqual(minVal, -16.1, "Sum of 16 cos should be >= -16")
      XCTAssertLessThanOrEqual(maxVal, 16.1, "Sum of 16 cos should be <= 16")
    }
  )

  // MARK: - Peek on Phasor Tensor

  static let peekOnPhasorTensor = TensorTestCase(
    name: "PeekOnPhasorTensor",
    frameCount: 100,
    injectData: true,
    graphBuilder: { g in
      let freqs = g.tensor(shape: [3, 1], data: [100.0, 200.0, 300.0])
      let phasorCell = g.alloc()
      let zeroReset = g.n(.constant(0.0))
      let phasorTensor = g.n(.phasor(phasorCell), freqs, zeroReset)
      let zero = g.n(.constant(0.0))
      let peekResult = try g.peek(tensor: phasorTensor, index: zero, channel: zero)
      _ = g.n(.output(0), peekResult)
    },
    validator: { output in
      XCTAssertGreaterThan(output[50], output[0], "Peek should increase over time")
      XCTAssertGreaterThan(output[10], 0, "Peek should return non-zero from phasor tensor")
    }
  )

  // MARK: - All test cases

  static let allCases: [TensorTestCase] = [
    // Basic ops
    sumReduceExecution,
    tensorAddScalarExecution,
    tensorMulTensor,
    broadcastScalarTensor,

    // History
    tensorHistoryExecution,

    // Conv
    conv2dExecution,
    conv1dExecution,
    conv1dIdentityKernel,

    // Reshape
    reshapeExecution,
    reshapeToFlat,
    reshapeFromFlat,
    reshapeThenSumAxisExecution,

    // SumAxis
    sumAxisExecution,
    sumAxisAxis0,
    sumAxisToScalar,
    nestedParallelRangeDebug,

    // Matmul
    matmulExecution,
    matmulWithScalarMul,

    // Transpose
    transposeExecution,

    // Pad
    padExecution,
    padAsymmetric,
    concatViaPadAndAdd,
    concat2DViaPadAndAdd,

    // Shrink
    shrinkExecution,
    shrinkColumnOnly,
    shrinkWithScalarOp,
    shrinkWithSumAxis,
    chainedShrink,
    shrinkWithBroadcastOp,
    shrinkThenConv2d,

    // Stack
    stackBasic,
    stackWithShape,

    // Latch
    latchWithTensorInputs,

    // Phasor with Tensor
    phasorWithTensorFrequencies,

    // Accum with Tensor
    accumWithTensorInputs,

    // Cos Phasor
    cosPhasorTensor,
    cosPhasorTensorLarge,

    // Peek
    peekOnPhasorTensor,

    // Complex simulation
    membraneSimulationExecute,
  ]

  // MARK: - Complex Membrane Simulation (simplified - no phasor trigger)

  static let membraneSimulationExecute = TensorTestCase(
    name: "MembraneSimulationExecute",
    frameCount: 100,
    injectData: true,
    graphBuilder: { g in
      let gridShape: Shape = [4, 4]

      // History buffers
      let stateBuffer = g.tensorHistoryBuffer(shape: gridShape)
      let prevStateBuffer = g.tensorHistoryBuffer(shape: gridShape)

      // Initial excitation - just add to state on first frame
      // Using a constant excitation that gets added every frame
      let excite = g.tensor(
        shape: [4, 4],
        data: [
          0, 0, 0, 0,
          0, 0.01, 0.01, 0,
          0, 0.01, 0.01, 0,
          0, 0, 0, 0,
        ])

      let state_t = g.n(.add, excite, g.tensorHistoryRead(stateBuffer))
      let state_t_1 = g.tensorHistoryRead(prevStateBuffer)

      // Laplacian kernel
      let kernel = g.tensor(
        shape: [3, 3],
        data: [
          0, 1, 0,
          1, -4, 1,
          0, 1, 0,
        ])

      let laplacian = g.n(.conv2d([3, 3]), state_t, kernel)

      // Physics coefficients
      let c_squared = g.n(.constant(0.1))
      let d = g.n(.constant(0.03))
      let two = g.n(.constant(2.0))
      let one = g.n(.constant(1.0))

      let twoMinusD = g.n(.sub, two, d)
      let oneMinusD = g.n(.sub, one, d)

      let scaledState = g.n(.mul, twoMinusD, state_t)
      let scaledPrev = g.n(.mul, oneMinusD, state_t_1)
      let scaledLaplacian = g.n(.mul, c_squared, laplacian)

      let diff = g.n(.sub, scaledState, scaledPrev)
      let state_t_plus_1 = g.n(.add, diff, scaledLaplacian)

      g.tensorHistoryWrite(prevStateBuffer, state_t)
      g.tensorHistoryWrite(stateBuffer, state_t_plus_1)

      let sumOutput = g.n(.sum, state_t_plus_1)
      _ = g.n(.output(0), sumOutput)
    },
    validator: { output in
      // With constant excitation, simulation should produce non-zero output
      let hasNonZero = output.contains { abs($0) > 0.001 }
      XCTAssertTrue(hasNonZero, "Simulation should produce non-zero output")

      // Check no NaN/Inf
      let hasNaN = output.contains { $0.isNaN }
      let hasInf = output.contains { $0.isInfinite }
      XCTAssertFalse(hasNaN, "Output should not contain NaN")
      XCTAssertFalse(hasInf, "Output should not contain Inf")
    }
  )
}

// MARK: - Test Runner Utilities

/// Run a test case with the C backend
func runCTest(_ testCase: TensorTestCase) throws {
  let g = Graph()
  try testCase.graphBuilder(g)

  let cResult = try CompilationPipeline.compile(
    graph: g,
    backend: .c,
    options: .init(frameCount: testCase.frameCount, debug: false)
  )

  let cRuntime = CCompiledKernel(
    source: cResult.source,
    cellAllocations: cResult.cellAllocations,
    memorySize: cResult.totalMemorySlots
  )
  try cRuntime.compileAndLoad()

  guard let mem = cRuntime.allocateNodeMemory() else {
    XCTFail("\(testCase.name): Failed to allocate memory")
    return
  }
  defer { cRuntime.deallocateNodeMemory(mem) }

  if testCase.injectData {
    injectTensorData(result: cResult, memory: mem.assumingMemoryBound(to: Float.self))
  }

  var output = [Float](repeating: 0, count: testCase.frameCount)
  let input = [Float](repeating: 0, count: testCase.frameCount)

  output.withUnsafeMutableBufferPointer { outPtr in
    input.withUnsafeBufferPointer { inPtr in
      cRuntime.runWithMemory(
        outputs: outPtr.baseAddress!,
        inputs: inPtr.baseAddress!,
        memory: mem,
        frameCount: testCase.frameCount
      )
    }
  }

  if let validator = testCase.customValidator {
    validator(output)
  } else if let expectedOutputs = testCase.expectedOutputs {
    for (i, expected) in expectedOutputs.enumerated() {
      XCTAssertEqual(
        output[i], expected,
        accuracy: testCase.accuracy,
        "\(testCase.name) [C]: output[\(i)] should be \(expected)"
      )
    }
  }
}

/// Run a test case with the Metal backend
func runMetalTest(_ testCase: TensorTestCase) throws {
  let g = Graph()
  try testCase.graphBuilder(g)

  let mResult = try CompilationPipeline.compile(
    graph: g,
    backend: .metal,
    options: .init(frameCount: testCase.frameCount, debug: false)
  )

  let metalRuntime = try MetalCompiledKernel(
    kernels: mResult.kernels,
    cellAllocations: mResult.cellAllocations,
    context: mResult.context,
    frameCount: testCase.frameCount
  )

  if testCase.injectData, let memoryBuffer = metalRuntime.getBuffer(name: "memory") {
    let memPtr = memoryBuffer.contents().assumingMemoryBound(to: Float.self)
    injectTensorData(result: mResult, memory: memPtr)
  }

  var output = [Float](repeating: 0, count: testCase.frameCount)
  let input = [Float](repeating: 0, count: testCase.frameCount)

  output.withUnsafeMutableBufferPointer { outPtr in
    input.withUnsafeBufferPointer { inPtr in
      metalRuntime.run(
        outputs: outPtr.baseAddress!,
        inputs: inPtr.baseAddress!,
        frameCount: testCase.frameCount
      )
    }
  }

  // Print output values for comparison with C backend
  print("\n=== Metal OUTPUT VALUES ===")
  var maxOutput: Float = 0
  var maxFrame = 0
  for (i, x) in output.enumerated() {
    if i < 20 || i % 50 == 0 || i >= testCase.frameCount - 10 {
      print("frame \(i): output=\(x)")
    }
    if abs(x) > maxOutput {
      maxOutput = abs(x)
      maxFrame = i
    }
  }
  print("Peak output: \(maxOutput) at frame \(maxFrame)")

  if let validator = testCase.customValidator {
    validator(output)
  } else if let expectedOutputs = testCase.expectedOutputs {
    for (i, expected) in expectedOutputs.enumerated() {
      XCTAssertEqual(
        output[i], expected,
        accuracy: testCase.accuracy,
        "\(testCase.name) [Metal]: output[\(i)] should be \(expected)"
      )
    }
  }
}

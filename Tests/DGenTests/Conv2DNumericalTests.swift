import XCTest

@testable import DGen

/// Numerical correctness tests for the SIMD-unrolled conv2d emission.
/// Exercises both the feedback-free (one-shot conv2d → sum → output) and the
/// sample-by-sample feedback (history-loop) paths, with constant and runtime
/// kernel tensors, across grid sizes that do and don't hit the SIMD path.
final class Conv2DNumericalTests: XCTestCase {

  // MARK: - Reference implementation

  /// Naive SAME-padding 2D convolution — matches `.conv2d` emission semantics:
  /// output shape = input shape, kernel centered with `padH = kH/2`, `padW = kW/2`,
  /// OOB reads are zero.
  static func referenceConv2D(
    input: [Float], inH: Int, inW: Int, kernel: [Float], kH: Int, kW: Int
  ) -> [Float] {
    let padH = kH / 2
    let padW = kW / 2
    var out = [Float](repeating: 0, count: inH * inW)
    for outY in 0..<inH {
      for outX in 0..<inW {
        var acc: Float = 0
        for ky in 0..<kH {
          for kx in 0..<kW {
            let inY = outY + ky - padH
            let inX = outX + kx - padW
            if inY >= 0 && inY < inH && inX >= 0 && inX < inW {
              acc += input[inY * inW + inX] * kernel[ky * kW + kx]
            }
          }
        }
        out[outY * inW + outX] = acc
      }
    }
    return out
  }

  // MARK: - Test harness

  /// Compile + run a one-frame graph that computes conv2d(input, kernel) and returns
  /// the full output tensor (read directly out of the output cell's memory).
  /// If `kernelData` is nil, the kernel tensor is allocated without baked data;
  /// `runtimeKernelValues` will be written into the kernel cell after `injectTensorData`.
  private func runConv2D(
    inputData: [Float], inShape: [Int],
    kernelShape: [Int], constantKernel: [Float]?, runtimeKernel: [Float]? = nil,
    expectOptimized: Bool
  ) throws -> [Float] {
    let (inH, inW) = (inShape[0], inShape[1])
    XCTAssertEqual(inputData.count, inH * inW, "input data mismatch")

    let g = Graph()
    let inputNode = g.tensor(shape: inShape, data: inputData)

    let kernelNode: NodeID
    if let data = constantKernel {
      kernelNode = g.tensor(shape: kernelShape, data: data)
    } else {
      // Runtime kernel: no .data. Values are written into the cell below.
      kernelNode = g.tensor(shape: kernelShape)
    }

    let convResult = g.n(.conv2d(kernelShape), inputNode, kernelNode)
    // Force materialization: write 16-to-sum so the whole output cell is populated,
    // then read cell memory directly for per-element verification.
    _ = g.n(.output(0), g.n(.sum, convResult))

    let result = try CompilationPipeline.compile(
      graph: g, backend: .c,
      options: .init(frameCount: 1, debug: false))

    if expectOptimized {
      XCTAssertFalse(
        g.simdOptimizedConv2Ds.isEmpty,
        "Conv2DPass should have annotated this conv2d as SIMD-eligible")
    } else {
      XCTAssertTrue(
        g.simdOptimizedConv2Ds.isEmpty,
        "Conv2DPass should NOT have annotated this conv2d "
          + "(ineligible shape or kernel)")
    }
    if ProcessInfo.processInfo.environment["DGEN_DUMP_CONV"] != nil {
      print("=== GENERATED C ===\n\(result.source)\n=== END ===")
    }

    let kernelRuntime = CCompiledKernel(
      source: result.source,
      cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots)
    try kernelRuntime.compileAndLoad()

    guard let mem = kernelRuntime.allocateNodeMemory() else {
      XCTFail("failed to allocate memory")
      return []
    }
    defer { kernelRuntime.deallocateNodeMemory(mem) }

    let memPtr = mem.assumingMemoryBound(to: Float.self)
    injectTensorData(result: result, memory: memPtr)

    // For the runtime-kernel path, hand-inject the kernel values into the kernel cell.
    if let rk = runtimeKernel {
      guard let kTensorId = g.nodeToTensor[kernelNode],
        let kTensor = g.tensors[kTensorId]
      else {
        XCTFail("kernel tensor missing"); return []
      }
      let phys =
        result.cellAllocations.cellMappings[kTensor.cellId] ?? kTensor.cellId
      for (i, v) in rk.enumerated() { memPtr[phys + i] = v }
    }

    var out = [Float](repeating: 0, count: 1)
    let input = [Float](repeating: 0, count: 1)
    out.withUnsafeMutableBufferPointer { outPtr in
      input.withUnsafeBufferPointer { inPtr in
        kernelRuntime.runWithMemory(
          outputs: outPtr.baseAddress!, inputs: inPtr.baseAddress!,
          memory: mem, frameCount: 1)
      }
    }

    // Read the conv2d output cell directly — convResult's tensor cell.
    guard let convTensorId = g.nodeToTensor[convResult],
      let convTensor = g.tensors[convTensorId]
    else {
      XCTFail("conv2d output tensor missing"); return []
    }
    let convPhys =
      result.cellAllocations.cellMappings[convTensor.cellId] ?? convTensor.cellId
    var readOut = [Float](repeating: 0, count: inH * inW)
    for i in 0..<(inH * inW) { readOut[i] = memPtr[convPhys + i] }
    return readOut
  }

  private func assertClose(
    _ got: [Float], _ expected: [Float], accuracy: Float = 1e-4,
    file: StaticString = #file, line: UInt = #line
  ) {
    XCTAssertEqual(got.count, expected.count, file: file, line: line)
    for i in 0..<min(got.count, expected.count) {
      XCTAssertEqual(
        got[i], expected[i], accuracy: accuracy,
        "mismatch at index \(i): got \(got[i]), expected \(expected[i])",
        file: file, line: line)
    }
  }

  // MARK: - Non-feedback: constant kernel, SIMD path

  func testConv2D_Identity_4x4() throws {
    // Identity kernel: center=1, rest=0. Output should equal input.
    let input: [Float] = (0..<16).map { Float($0 + 1) }  // [1..16]
    let kernel: [Float] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    let got = try runConv2D(
      inputData: input, inShape: [4, 4],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, input)
  }

  func testConv2D_Identity_8x8() throws {
    let input: [Float] = (0..<64).map { Float($0 + 1) }
    let kernel: [Float] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    let got = try runConv2D(
      inputData: input, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, input)
  }

  func testConv2D_Laplacian_4x4() throws {
    // Discrete Laplacian. With zero-padding, edges see extra -4 weight on themselves.
    let input: [Float] = [
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
    ]
    let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    let expected = Self.referenceConv2D(
      input: input, inH: 4, inW: 4, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [4, 4],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_Averaging_8x8() throws {
    let input: [Float] = (0..<64).map { _ in 1.0 }
    // 3×3 box filter with weight 1/9.
    let kernel: [Float] = Array(repeating: Float(1.0 / 9.0), count: 9)
    let expected = Self.referenceConv2D(
      input: input, inH: 8, inW: 8, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_AsymmetricKernel_4x4() throws {
    // Non-symmetric kernel shakes out any (ky,kx) transposition bugs.
    let input: [Float] = (0..<16).map { Float($0) }
    let kernel: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    let expected = Self.referenceConv2D(
      input: input, inH: 4, inW: 4, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [4, 4],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_SingleImpulse_8x8() throws {
    // Impulse at position (3, 3). Asymmetric kernel. Output should show
    // the kernel imprinted around (3,3).
    var input = [Float](repeating: 0, count: 64)
    input[3 * 8 + 3] = 1.0
    let kernel: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    let expected = Self.referenceConv2D(
      input: input, inH: 8, inW: 8, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_NonSquareWide_4x8() throws {
    let input: [Float] = (0..<32).map { Float($0) }
    let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    let expected = Self.referenceConv2D(
      input: input, inH: 4, inW: 8, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [4, 8],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_NonSquareTall_8x4() throws {
    let input: [Float] = (0..<32).map { Float($0) }
    let kernel: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    let expected = Self.referenceConv2D(
      input: input, inH: 8, inW: 4, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [8, 4],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_Large_16x16() throws {
    let input: [Float] = (0..<256).map { Float($0) * 0.01 }
    let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    let expected = Self.referenceConv2D(
      input: input, inH: 16, inW: 16, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [16, 16],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  // MARK: - Non-feedback: narrower kernels

  func testConv2D_RowKernel_1x3_4x4() throws {
    // 1×3 row kernel: padH=0, padW=1. Horizontal blur.
    let input: [Float] = (1...16).map(Float.init)
    let kernel: [Float] = [0.25, 0.5, 0.25]
    let expected = Self.referenceConv2D(
      input: input, inH: 4, inW: 4, kernel: kernel, kH: 1, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [4, 4],
      kernelShape: [1, 3], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_ColumnKernel_3x1_4x4() throws {
    // 3×1 column kernel: padH=1, padW=0. Only row bounds matter; no column mask.
    let input: [Float] = (1...16).map(Float.init)
    let kernel: [Float] = [0.25, 0.5, 0.25]
    let expected = Self.referenceConv2D(
      input: input, inH: 4, inW: 4, kernel: kernel, kH: 3, kW: 1)
    let got = try runConv2D(
      inputData: input, inShape: [4, 4],
      kernelShape: [3, 1], constantKernel: kernel, expectOptimized: true)
    assertClose(got, expected)
  }

  // MARK: - Non-feedback: runtime kernel (no .data baked at graph-build)

  func testConv2D_RuntimeKernel_4x4() throws {
    let input: [Float] = (0..<16).map { Float($0) + 1 }
    let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    let expected = Self.referenceConv2D(
      input: input, inH: 4, inW: 4, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [4, 4],
      kernelShape: [3, 3], constantKernel: nil, runtimeKernel: kernel,
      expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_RuntimeKernel_8x8_Asymmetric() throws {
    let input: [Float] = (0..<64).map { Float($0) * 0.1 }
    let kernel: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    let expected = Self.referenceConv2D(
      input: input, inH: 8, inW: 8, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: nil, runtimeKernel: kernel,
      expectOptimized: true)
    assertClose(got, expected)
  }

  func testConv2D_RuntimeKernel_Identity_8x8() throws {
    let input: [Float] = (1...64).map(Float.init)
    let kernel: [Float] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    let got = try runConv2D(
      inputData: input, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: nil, runtimeKernel: kernel,
      expectOptimized: true)
    assertClose(got, input)
  }

  // MARK: - Feedback: conv2d inside history loop

  /// Run a minimal membrane-style feedback graph and return the full output-cell
  /// trajectory per frame. The per-frame recurrence matches `testMembraneSimulationExecute`
  /// but strips out excitation timing so each frame is deterministic.
  private func runFeedbackConv2D(
    initialState: [Float], inShape: [Int],
    kernelShape: [Int], constantKernel: [Float]?, runtimeKernel: [Float]? = nil,
    frameCount: Int
  ) throws -> [[Float]] {
    let (inH, inW) = (inShape[0], inShape[1])
    XCTAssertEqual(initialState.count, inH * inW)

    let g = Graph()
    let stateBuffer = g.tensorHistoryBuffer(shape: inShape, data: initialState)
    let state_t = g.tensorHistoryRead(stateBuffer)

    let kernelNode: NodeID
    if let data = constantKernel {
      kernelNode = g.tensor(shape: kernelShape, data: data)
    } else {
      kernelNode = g.tensor(shape: kernelShape)
    }

    let conv = g.n(.conv2d(kernelShape), state_t, kernelNode)
    g.tensorHistoryWrite(stateBuffer, conv)
    _ = g.n(.output(0), g.n(.sum, conv))

    let result = try CompilationPipeline.compile(
      graph: g, backend: .c,
      options: .init(frameCount: frameCount, debug: false))

    let kernelRuntime = CCompiledKernel(
      source: result.source,
      cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots)
    try kernelRuntime.compileAndLoad()

    guard let mem = kernelRuntime.allocateNodeMemory() else {
      XCTFail("failed to allocate memory"); return []
    }
    defer { kernelRuntime.deallocateNodeMemory(mem) }

    let memPtr = mem.assumingMemoryBound(to: Float.self)
    injectTensorData(result: result, memory: memPtr)

    if let rk = runtimeKernel {
      guard let kTensorId = g.nodeToTensor[kernelNode],
        let kTensor = g.tensors[kTensorId]
      else {
        XCTFail("kernel tensor missing"); return []
      }
      let phys =
        result.cellAllocations.cellMappings[kTensor.cellId] ?? kTensor.cellId
      for (i, v) in rk.enumerated() { memPtr[phys + i] = v }
    }

    // Run one frame at a time and snapshot the state cell each time.
    var trajectory: [[Float]] = []
    trajectory.reserveCapacity(frameCount)
    var out = [Float](repeating: 0, count: 1)
    let input = [Float](repeating: 0, count: 1)
    let stateCellPhys =
      result.cellAllocations.cellMappings[stateBuffer.cellId] ?? stateBuffer.cellId
    for _ in 0..<frameCount {
      out.withUnsafeMutableBufferPointer { outPtr in
        input.withUnsafeBufferPointer { inPtr in
          kernelRuntime.runWithMemory(
            outputs: outPtr.baseAddress!, inputs: inPtr.baseAddress!,
            memory: mem, frameCount: 1)
        }
      }
      var snap = [Float](repeating: 0, count: inH * inW)
      for i in 0..<(inH * inW) { snap[i] = memPtr[stateCellPhys + i] }
      trajectory.append(snap)
    }
    return trajectory
  }

  /// Analytic feedback: state_{t+1} = conv2d(state_t, kernel).
  private static func referenceFeedback(
    initialState: [Float], inH: Int, inW: Int,
    kernel: [Float], kH: Int, kW: Int, frames: Int
  ) -> [[Float]] {
    var traj: [[Float]] = []
    var state = initialState
    for _ in 0..<frames {
      state = referenceConv2D(
        input: state, inH: inH, inW: inW, kernel: kernel, kH: kH, kW: kW)
      traj.append(state)
    }
    return traj
  }

  func testConv2DFeedback_Averaging_4x4() throws {
    // Box filter applied repeatedly — should diffuse an impulse outward.
    var init4: [Float] = Array(repeating: 0, count: 16)
    init4[1 * 4 + 1] = 1.0
    let kernel: [Float] = Array(repeating: Float(1.0 / 9.0), count: 9)
    let frames = 8
    let got = try runFeedbackConv2D(
      initialState: init4, inShape: [4, 4],
      kernelShape: [3, 3], constantKernel: kernel, frameCount: frames)
    let expected = Self.referenceFeedback(
      initialState: init4, inH: 4, inW: 4,
      kernel: kernel, kH: 3, kW: 3, frames: frames)
    XCTAssertEqual(got.count, expected.count)
    for f in 0..<frames {
      assertClose(got[f], expected[f], accuracy: 1e-4)
    }
  }

  func testConv2DFeedback_Identity_8x8() throws {
    // Identity kernel through feedback should leave state unchanged every frame.
    let init8: [Float] = (1...64).map(Float.init)
    let kernel: [Float] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    let frames = 5
    let got = try runFeedbackConv2D(
      initialState: init8, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: kernel, frameCount: frames)
    for f in 0..<frames {
      assertClose(got[f], init8, accuracy: 1e-4)
    }
  }

  func testConv2DFeedback_Laplacian_8x8_DecayWithDamping() throws {
    // Matches the membrane recurrence (minus excitation): state carries through
    // a Laplacian-times-damping kernel. All-ones initial state with a diffusive
    // kernel summing to just under 1 should decay monotonically on the sum.
    let init8: [Float] = Array(repeating: 1.0, count: 64)
    // Laplacian-ish with a dominant center; 9 weights sum to 0.5 so energy leaks each step.
    let kernel: [Float] = [
      0.01, 0.05, 0.01,
      0.05, 0.26, 0.05,
      0.01, 0.05, 0.01,
    ]
    let frames = 4
    let got = try runFeedbackConv2D(
      initialState: init8, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: kernel, frameCount: frames)
    let expected = Self.referenceFeedback(
      initialState: init8, inH: 8, inW: 8,
      kernel: kernel, kH: 3, kW: 3, frames: frames)
    for f in 0..<frames {
      assertClose(got[f], expected[f], accuracy: 1e-4)
    }
  }

  func testConv2DFeedback_RuntimeKernel_4x4() throws {
    var init4: [Float] = Array(repeating: 0, count: 16)
    init4[1 * 4 + 2] = 1.0
    let kernel: [Float] = [0, 0.125, 0, 0.125, 0.5, 0.125, 0, 0.125, 0]
    let frames = 6
    let got = try runFeedbackConv2D(
      initialState: init4, inShape: [4, 4],
      kernelShape: [3, 3], constantKernel: nil, runtimeKernel: kernel,
      frameCount: frames)
    let expected = Self.referenceFeedback(
      initialState: init4, inH: 4, inW: 4,
      kernel: kernel, kH: 3, kW: 3, frames: frames)
    for f in 0..<frames {
      assertClose(got[f], expected[f], accuracy: 1e-4)
    }
  }

  func testConv2DFeedback_RuntimeKernel_8x8() throws {
    let init8: [Float] = (0..<64).map { Float($0 % 7) * 0.1 }
    let kernel: [Float] = [0.05, 0.1, 0.05, 0.1, 0.4, 0.1, 0.05, 0.1, 0.05]
    let frames = 4
    let got = try runFeedbackConv2D(
      initialState: init8, inShape: [8, 8],
      kernelShape: [3, 3], constantKernel: nil, runtimeKernel: kernel,
      frameCount: frames)
    let expected = Self.referenceFeedback(
      initialState: init8, inH: 8, inW: 8,
      kernel: kernel, kH: 3, kW: 3, frames: frames)
    for f in 0..<frames {
      assertClose(got[f], expected[f], accuracy: 1e-4)
    }
  }

  // MARK: - Non-SIMD fallback: ineligible shapes still give correct output

  func testConv2D_ScalarFallback_3x3_Correctness() throws {
    // inW=3 is NOT divisible by 4 → Conv2DPass skips → scalar emission path.
    // Mainly checking nothing changed at this layer when we added SIMD.
    let input: [Float] = (1...9).map(Float.init)
    let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    let expected = Self.referenceConv2D(
      input: input, inH: 3, inW: 3, kernel: kernel, kH: 3, kW: 3)
    let got = try runConv2D(
      inputData: input, inShape: [3, 3],
      kernelShape: [3, 3], constantKernel: kernel, expectOptimized: false)
    assertClose(got, expected)
  }
}

import DGen
import XCTest

@testable import DGenLazy

/// Focused tests for matmul kernel generation and correctness.
/// Goal: understand what the compiler generates and identify fusion opportunities.
final class MatmulKernelTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  /// Simplest matmul: [2,2] @ [2,2] → [2,2]
  /// Should need only 1-2 kernels ideally, let's see what we get.
  func testMatmul2x2Kernel() throws {
    DGenConfig.kernelOutputPath = "/tmp/matmul_2x2.metal"
    defer { DGenConfig.kernelOutputPath = nil }

    let A = Tensor([
      [1.0, 2.0],
      [3.0, 4.0],
    ])
    let B = Tensor([
      [5.0, 6.0],
      [7.0, 8.0],
    ])

    let C = A.matmul(B)
    let result = try C.realize()

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    XCTAssertEqual(result[0], 19.0, accuracy: 1e-4)
    XCTAssertEqual(result[1], 22.0, accuracy: 1e-4)
    XCTAssertEqual(result[2], 43.0, accuracy: 1e-4)
    XCTAssertEqual(result[3], 50.0, accuracy: 1e-4)

    print("matmul 2x2 result: \(result)")
    print("Kernel written to /tmp/matmul_2x2.metal")
  }

  /// CPU reference matmul for verification: A[M,K] @ B[K,N] → C[M,N]
  private func cpuMatmul(_ a: [Float], _ b: [Float], M: Int, K: Int, N: Int) -> [Float] {
    var c = [Float](repeating: 0, count: M * N)
    for i in 0..<M {
      for j in 0..<N {
        var sum: Float = 0
        for p in 0..<K {
          sum += a[i * K + p] * b[p * N + j]
        }
        c[i * N + j] = sum
      }
    }
    return c
  }

  /// 8x8 matmul — smallest GEMM-eligible size (M,N,K all divisible by 8).
  /// Exercises the simdgroup_float8x8 path with exactly 1 tile per dimension.
  func testMatmul8x8Kernel() throws {
    DGenConfig.kernelOutputPath = "/tmp/matmul_8x8.metal"
    defer { DGenConfig.kernelOutputPath = nil }

    let M = 8
    let K = 8
    let N = 8
    // Deterministic small values to avoid float precision issues
    let dataA = (0..<M * K).map { Float($0 + 1) * 0.01 }
    let dataB = (0..<K * N).map { Float($0 + 1) * 0.01 }
    let expected = cpuMatmul(dataA, dataB, M: M, K: K, N: N)

    let A = Tensor(dataA).reshape([M, K])
    let B = Tensor(dataB).reshape([K, N])
    let C = A.matmul(B)
    let result = try C.realize()

    XCTAssertEqual(result.count, M * N)
    for i in 0..<result.count {
      XCTAssertEqual(
        result[i], expected[i], accuracy: 1e-3,
        "Mismatch at index \(i): got \(result[i]), expected \(expected[i])")
    }
    print("matmul 8x8 GEMM result verified against CPU reference")
  }

  /// 64x64 matmul — exercises multi-tile GEMM (8 tiles per dimension).
  func testMatmul64x64Kernel() throws {
    DGenConfig.kernelOutputPath = "/tmp/matmul_64x64.metal"
    defer { DGenConfig.kernelOutputPath = nil }

    let M = 64
    let K = 64
    let N = 64
    // Use small values scaled down to keep products in reasonable float range
    let dataA = (0..<M * K).map { Float($0 % 17) * 0.1 - 0.8 }
    let dataB = (0..<K * N).map { Float($0 % 13) * 0.1 - 0.6 }
    let expected = cpuMatmul(dataA, dataB, M: M, K: K, N: N)

    let A = Tensor(dataA).reshape([M, K])
    let B = Tensor(dataB).reshape([K, N])
    let C = A.matmul(B)
    let result = try C.realize()

    XCTAssertEqual(result.count, M * N)
    var maxErr: Float = 0
    for i in 0..<result.count {
      let err = abs(result[i] - expected[i])
      maxErr = max(maxErr, err)
      XCTAssertEqual(
        result[i], expected[i], accuracy: 1e-2,
        "Mismatch at [\(i/N),\(i%N)]: got \(result[i]), expected \(expected[i])")
    }
    print("matmul 64x64 GEMM result verified, max error: \(maxErr)")
  }

  /// 8x8 matmul backward — verify gradients are numerically correct.
  /// For loss = sum(A @ B), dL/dA[i,k] = sum_j B[k,j] (sum of row k of B).
  /// Both forward and backward matmul patterns are GEMM-eligible (8-divisible).
  func testMatmulGemmBackward() throws {
    DGenConfig.kernelOutputPath = "/tmp/matmul_gemm_backward.metal"
    defer { DGenConfig.kernelOutputPath = nil }

    let M = 8
    let K = 8
    let N = 8
    let dataA = (0..<M * K).map { Float($0 + 1) * 0.01 }
    let dataB = (0..<K * N).map { Float($0 + 1) * 0.01 }

    let A = Tensor.param([M, K], data: dataA)
    let B = Tensor(dataB).reshape([K, N])

    let C = A.matmul(B)
    let loss = C.sum()
    let _ = try loss.backward(frameCount: 1)

    // Verify gradients: dL/dA[i,k] = sum_j B[k,j]
    guard let gradA = A.grad else {
      XCTFail("A.grad is nil after backward")
      return
    }

    var expectedGradA = [Float](repeating: 0, count: M * K)
    for i in 0..<M {
      for k in 0..<K {
        var rowSum: Float = 0
        for j in 0..<N { rowSum += dataB[k * N + j] }
        expectedGradA[i * K + k] = rowSum
      }
    }

    for i in 0..<expectedGradA.count {
      XCTAssertEqual(
        gradA.getData()![i], expectedGradA[i], accuracy: 1e-2,
        "Gradient mismatch at A[\(i/K),\(i%K)]: got \(gradA.getData()![i]), expected \(expectedGradA[i])"
      )
    }
    print("GEMM backward gradients verified against analytical reference")
  }

  /// 8x8 backward with both A and B as params — verify both gradients.
  /// dL/dA[i,k] = sum_j B[k,j], dL/dB[k,j] = sum_i A[i,k]
  func testMatmulGemmBackwardBothParams() throws {
    let M = 8
    let K = 8
    let N = 8
    let dataA = (0..<M * K).map { Float($0 + 1) * 0.01 }
    let dataB = (0..<K * N).map { Float(65 - $0) * 0.01 }

    let A = Tensor.param([M, K], data: dataA)
    let B = Tensor.param([K, N], data: dataB)

    let C = A.matmul(B)
    let loss = (C - 0).sum()
    let lossValue = try loss.backward(frameCount: 1)
    print(dataA)
    print(dataB)
    print(lossValue)
    print("loss value = \(lossValue.reduce(0, +))")

    // dL/dA[i,k] = sum_j B[k,j]
    guard let gradA = A.grad else {
      XCTFail("A.grad nil")
      return
    }
    for i in 0..<M {
      for k in 0..<K {
        var rowSum: Float = 0
        for j in 0..<N { rowSum += dataB[k * N + j] }
        XCTAssertEqual(
          gradA.getData()![i * K + k], rowSum, accuracy: 1e-2,
          "dA[\(i),\(k)] mismatch")
      }
    }

    // dL/dB[k,j] = sum_i A[i,k]
    guard let gradB = B.grad else {
      XCTFail("B.grad nil")
      return
    }
    for k in 0..<K {
      for j in 0..<N {
        var colSum: Float = 0
        for i in 0..<M { colSum += dataA[i * K + k] }
        XCTAssertEqual(
          gradB.getData()![k * N + j], colSum, accuracy: 1e-2,
          "dB[\(k),\(j)] mismatch")
      }
    }
    print("GEMM backward both-param gradients verified")
  }

  /// Matmul [2,2] @ [2,2] → [2,2] with backward pass.
  /// Verifies that the static matmul kernel parallelizes over output elements (4 threads, no element loop).
  func testMatmul2x2KernelBackward() throws {
    DGenConfig.kernelOutputPath = "/tmp/matmul_2x2_backward.metal"
    defer { DGenConfig.kernelOutputPath = nil }

    let A = Tensor([
      [1.0, 2.0],
      [3.0, 4.0],
    ])
    let B = Tensor([
      [5.0, 6.0],
      [7.0, 8.0],
    ])

    let C = A.matmul(B)
    let _ = try C.sum().backward(frameCount: 4)

    // Read generated kernel and verify parallelization of kernel_0 (the matmul)
    let source = try String(contentsOfFile: "/tmp/matmul_2x2_backward.metal", encoding: .utf8)
    let kernel0 = extractKernel(source, index: 0)

    // Must dispatch 4 threads (one per output element), not 1
    XCTAssertTrue(
      kernel0.contains("id < (uint)(4)"),
      "Matmul kernel should dispatch 4 threads (one per output element), got single-thread dispatch"
    )
    XCTAssertFalse(
      kernel0.contains("id < (uint)(1)"),
      "Matmul kernel must NOT use single-thread dispatch"
    )

    // Must NOT have a sequential loop over 4 elements (the outer element loop should be parallelized)
    XCTAssertFalse(
      kernel0.contains("t8 < 4") || kernel0.contains("< 4;"),
      "Matmul kernel should not have a sequential loop over 4 output elements"
    )

    // Should have ThreadCountScale = 4
    XCTAssertTrue(
      kernel0.contains("ThreadCountScale Optional(4)"),
      "Matmul kernel should have ThreadCountScale = 4"
    )

    print("Kernel written to /tmp/matmul_2x2_backward.metal")
  }

  /// Extract a single kernel's source from concatenated Metal output.
  private func extractKernel(_ source: String, index: Int) -> String {
    let marker = "// KERNEL \(index)\n"
    let nextMarker = "// KERNEL \(index + 1)\n"
    guard let start = source.range(of: marker) else { return "" }
    if let end = source.range(of: nextMarker) {
      return String(source[start.lowerBound..<end.lowerBound])
    }
    return String(source[start.lowerBound...])
  }

  /// Benchmark: attention training step time (10 epochs, [4,4] matrices)
  func testAttentionStepBenchmark() throws {
    let numWindows = 4
    let numBins = 4

    // Fixed spectrogram-like input [4,4]
    let spectrogramData: [Float] = [
      0.8, 0.3, 0.1, 0.5,
      0.0, 0.0, 0.0, 0.0,
      0.6, 0.4, 0.2, 0.7,
      0.0, 0.0, 0.0, 0.0,
    ]
    let pattern: [Float] = [1, 0, 1, 0]

    // Non-uniform init
    let WQ = Tensor.param(
      [4, 4],
      data: [
        0.20, -0.10, 0.05, 0.15, -0.05, 0.20, 0.10, -0.15,
        0.10, 0.05, -0.20, 0.15, -0.15, 0.10, 0.05, 0.20,
      ])
    let WK = Tensor.param(
      [4, 4],
      data: [
        0.15, 0.05, -0.10, 0.20, 0.10, -0.15, 0.20, 0.05,
        -0.20, 0.15, 0.05, -0.10, 0.05, 0.20, -0.15, 0.10,
      ])
    let WV = Tensor.param(
      [4, 4],
      data: [
        -0.10, 0.20, 0.15, 0.05, 0.15, -0.05, 0.20, -0.10,
        0.05, 0.10, -0.15, 0.20, 0.20, -0.10, 0.05, 0.15,
      ])
    let WO = Tensor.param([4, 1], data: [0.20, -0.15, 0.10, -0.05])

    let optimizer = Adam(params: [WQ, WK, WV, WO], lr: 0.05)
    let epochs = 10

    // Warmup (first epoch compiles kernels)
    let spectrogram = Tensor(spectrogramData).reshape([numWindows, numBins])
    let target = Tensor(pattern).reshape([numWindows, 1])
    let dk = WQ.shape[1]
    let Q = spectrogram.matmul(WQ)
    let K = spectrogram.matmul(WK)
    let V = spectrogram.matmul(WV)
    let scores = Q.matmul(K.transpose()) / Foundation.sqrt(Float(dk))
    let weights = scores.softmax(axis: -1)
    let attnOut = weights.matmul(V)
    let predictions = attnOut.matmul(WO)
    let diff = predictions - target
    let warmupLoss = (diff * diff).sum()
    let _ = try warmupLoss.backward(frameCount: 1)
    optimizer.step()
    optimizer.zeroGrad()

    // Timed epochs
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<epochs {
      let spectrogram = Tensor(spectrogramData).reshape([numWindows, numBins])
      let target = Tensor(pattern).reshape([numWindows, 1])
      let dk = WQ.shape[1]
      let Q = spectrogram.matmul(WQ)
      let K = spectrogram.matmul(WK)
      let V = spectrogram.matmul(WV)
      let scores = Q.matmul(K.transpose()) / Foundation.sqrt(Float(dk))
      let weights = scores.softmax(axis: -1)
      let attnOut = weights.matmul(V)
      let predictions = attnOut.matmul(WO)
      let diff = predictions - target
      let loss = (diff * diff).sum()
      let _ = try loss.backward(frameCount: 1)
      optimizer.step()
      optimizer.zeroGrad()
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start

    let msPerStep = (elapsed / Double(epochs)) * 1000
    print("\n=== Attention Step Benchmark ===")
    print("Epochs: \(epochs)")
    print("Total: \(String(format: "%.1f", elapsed * 1000)) ms")
    print("Per step: \(String(format: "%.1f", msPerStep)) ms")
  }
}

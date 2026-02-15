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

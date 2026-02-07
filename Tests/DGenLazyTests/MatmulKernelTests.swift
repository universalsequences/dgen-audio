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
}

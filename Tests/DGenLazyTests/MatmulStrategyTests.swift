import DGen
import XCTest

@testable import DGenLazy

/// Tests for the `DGenConfig.gemmStrategy` enum. Each strategy
/// (.none, .registerTiled, .threadgroupStaged) is exercised against a CPU
/// reference matmul, and the emitted kernel source is grepped for the
/// expected codegen signature.
final class MatmulStrategyTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
    DGenConfig.gemmStrategy = .registerTiled
  }

  override func tearDown() {
    DGenConfig.gemmStrategy = .registerTiled
    DGenConfig.kernelOutputPath = nil
    super.tearDown()
  }

  // MARK: - CPU reference

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

  private func runMatmul(M: Int, K: Int, N: Int, kernelPath: String,
                         tolerance: Float, file: StaticString = #file, line: UInt = #line) throws -> String {
    DGenConfig.kernelOutputPath = kernelPath
    // Use small bounded values so products stay in a safe float32 range, matching
    // the convention in MatmulKernelTests.testMatmul64x64Kernel.
    let dataA: [Float] = (0..<M * K).map { Float($0 % 17) * 0.1 - 0.8 }
    let dataB: [Float] = (0..<K * N).map { Float($0 % 13) * 0.1 - 0.6 }
    let expected = cpuMatmul(dataA, dataB, M: M, K: K, N: N)
    let A = Tensor(dataA).reshape([M, K])
    let B = Tensor(dataB).reshape([K, N])
    let C = A.matmul(B)
    let result = try C.realize()
    XCTAssertEqual(result.count, M * N, file: file, line: line)
    for i in 0..<result.count {
      XCTAssertEqual(result[i], expected[i], accuracy: tolerance,
                     "Mismatch at index \(i)", file: file, line: line)
    }
    return try String(contentsOfFile: kernelPath, encoding: .utf8)
  }

  // MARK: - 32×32 strategy variants

  func testMatmul32x32_NoneStrategy() throws {
    DGenConfig.gemmStrategy = .none
    let src = try runMatmul(M: 32, K: 32, N: 32,
                            kernelPath: "/tmp/matmul_32x32_none.metal",
                            tolerance: 1e-3)
    XCTAssertFalse(src.contains("simdgroup_float8x8"),
                   "naive strategy must not emit simdgroup_float8x8")
    XCTAssertFalse(src.contains("DispatchMode: gemm"),
                   "naive strategy must not produce a gemm dispatch")
  }

  func testMatmul32x32_RegisterTiledStrategy() throws {
    DGenConfig.gemmStrategy = .registerTiled
    let src = try runMatmul(M: 32, K: 32, N: 32,
                            kernelPath: "/tmp/matmul_32x32_register.metal",
                            tolerance: 1e-3)
    XCTAssertTrue(src.contains("simdgroup_multiply_accumulate"),
                  "registerTiled must emit simdgroup MAC")
    XCTAssertFalse(src.contains("threadgroup_barrier"),
                   "registerTiled must NOT emit threadgroup_barrier")
    XCTAssertTrue(src.contains("DispatchMode: gemm("),
                  "registerTiled must produce gemm dispatch (not gemmStaged)")
  }

  func testMatmul32x32_ThreadgroupStagedStrategy() throws {
    DGenConfig.gemmStrategy = .threadgroupStaged
    let src = try runMatmul(M: 32, K: 32, N: 32,
                            kernelPath: "/tmp/matmul_32x32_staged.metal",
                            tolerance: 1e-3)
    XCTAssertTrue(src.contains("simdgroup_multiply_accumulate"),
                  "staged must emit simdgroup MAC")
    XCTAssertTrue(src.contains("threadgroup_barrier"),
                  "staged must emit threadgroup_barrier between cooperative load and MAC")
    XCTAssertTrue(src.contains("simdgroup_index_in_threadgroup"),
                  "staged must reference simdgroup_index_in_threadgroup attribute")
    XCTAssertTrue(src.contains("threadgroup float scratch_"),
                  "staged must allocate threadgroup scratch for A/B strips")
    XCTAssertTrue(src.contains("DispatchMode: gemmStaged"),
                  "staged must produce a gemmStaged dispatch")
  }

  // MARK: - 64×64 staged variant (4×4 threadgroup grid for 16×16 blocks)

  func testMatmul64x64_ThreadgroupStagedStrategy() throws {
    DGenConfig.gemmStrategy = .threadgroupStaged
    let src = try runMatmul(M: 64, K: 64, N: 64,
                            kernelPath: "/tmp/matmul_64x64_staged.metal",
                            tolerance: 1e-2)
    XCTAssertTrue(src.contains("threadgroup_barrier"))
    XCTAssertTrue(src.contains("DispatchMode: gemmStaged"),
                  "64×64 must use staged dispatch")
  }

  // MARK: - Fallback for ineligible shapes

  /// 8×8 with .threadgroupStaged: shape doesn't meet 16-aligned M/N requirement,
  /// so the pass should fall back to register-tiled .gemm.
  func testMatmul8x8_StagedFallsBackToRegisterTiled() throws {
    DGenConfig.gemmStrategy = .threadgroupStaged
    let src = try runMatmul(M: 8, K: 8, N: 8,
                            kernelPath: "/tmp/matmul_8x8_staged_fallback.metal",
                            tolerance: 1e-3)
    XCTAssertFalse(src.contains("threadgroup_barrier"),
                   "8×8 staged must fall back to register-tiled (no barrier)")
    XCTAssertFalse(src.contains("DispatchMode: gemmStaged"),
                   "8×8 staged must fall back to register-tiled gemm dispatch")
    XCTAssertTrue(src.contains("simdgroup_multiply_accumulate"),
                  "fallback should still go through GEMM path")
  }
}

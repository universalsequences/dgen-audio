import DGen
import XCTest

@testable import DGenLazy

/// GPU-timed micro-benchmarks comparing .registerTiled vs .threadgroupStaged.
/// Uses MetalRuntime.profileKernels (MTLCommandBuffer GPU timestamps) to measure
/// the matmul kernel in isolation, skipping realize() overhead entirely.
/// Gate with DGEN_BENCH=1.
final class MatmulStrategyBench: XCTestCase {

  override func tearDown() {
    DGenConfig.gemmStrategy = .registerTiled
    super.tearDown()
  }

  /// Build a matmul graph, realize once to compile & prime buffers,
  /// then run profileKernels `samples` times and return the median GPU ms
  /// of the matmul kernel (identified by the gemm/gemmStaged dispatchInfo prefix).
  private func gpuTimeMatmul(M: Int, K: Int, N: Int, samples: Int) throws -> (median: Double, dispatch: String) {
    LazyGraphContext.reset()
    let dataA: [Float] = (0..<M * K).map { Float($0 % 17) * 0.1 - 0.8 }
    let dataB: [Float] = (0..<K * N).map { Float($0 % 13) * 0.1 - 0.6 }
    let A = Tensor(dataA).reshape([M, K])
    let B = Tensor(dataB).reshape([K, N])
    let C = A.matmul(B)
    _ = try C.realize()

    let graph = C.graph
    guard let cached = graph.fullCompilationCache else {
      throw XCTSkip("no compilation cache")
    }

    var matmulTimes: [Double] = []
    var dispatchInfo = ""
    for _ in 0..<samples {
      let timings = cached.runtime.profileKernels(frameCount: 1)
      // Matmul kernel is the one whose dispatchInfo starts with "gemm".
      if let matmul = timings.first(where: { $0.dispatchInfo.hasPrefix("gemm") }) {
        matmulTimes.append(matmul.gpuMs)
        dispatchInfo = matmul.dispatchInfo
      }
    }
    guard !matmulTimes.isEmpty else { throw XCTSkip("no matmul kernel found in timings") }
    matmulTimes.sort()
    let median = matmulTimes[matmulTimes.count / 2]
    return (median, dispatchInfo)
  }

  /// Build a matmul with both params requiring grad, run loss.backward() once
  /// to compile the full forward+backward graph, then call profileKernels to get
  /// per-kernel GPU timings. Returns all kernels whose dispatchInfo starts with
  /// "gemm" (forward + the two backward gradient GEMMs).
  private func gpuTimeBackward(M: Int, K: Int, N: Int, samples: Int)
    throws -> (gemmKernels: [(dispatch: String, gpuMs: Double)], allUs: Double)
  {
    LazyGraphContext.reset()
    let dataA: [Float] = (0..<M * K).map { Float($0 % 17) * 0.1 - 0.8 }
    let dataB: [Float] = (0..<K * N).map { Float($0 % 13) * 0.1 - 0.6 }
    let A = Tensor.param([M, K], data: dataA)
    let B = Tensor.param([K, N], data: dataB)
    let C = A.matmul(B)
    let loss = C.sum()
    _ = try loss.backward(frameCount: 1)

    guard let cached = loss.graph.fullCompilationCache else {
      throw XCTSkip("no compilation cache")
    }

    // Accumulate per-kernel samples, keyed by (index, dispatch).
    var perKernelSamples: [Int: (dispatch: String, times: [Double])] = [:]
    var totalSamples: [Double] = []
    for _ in 0..<samples {
      let timings = cached.runtime.profileKernels(frameCount: 1)
      var total = 0.0
      for t in timings {
        total += t.gpuMs
        var entry = perKernelSamples[t.index] ?? (t.dispatchInfo, [])
        entry.times.append(t.gpuMs)
        entry.dispatch = t.dispatchInfo
        perKernelSamples[t.index] = entry
      }
      totalSamples.append(total)
    }
    totalSamples.sort()
    let totalMedian = totalSamples[totalSamples.count / 2]

    let gemmKernels = perKernelSamples.values
      .filter { $0.dispatch.hasPrefix("gemm") }
      .map { entry -> (dispatch: String, gpuMs: Double) in
        let sorted = entry.times.sorted()
        return (entry.dispatch, sorted[sorted.count / 2])
      }
      .sorted { $0.gpuMs > $1.gpuMs }
    return (gemmKernels, totalMedian * 1000.0)
  }

  func testBenchmarkBackwardStrategies() throws {
    guard ProcessInfo.processInfo.environment["DGEN_BENCH"] != nil else {
      throw XCTSkip("Set DGEN_BENCH=1 to run benchmarks")
    }

    let shapes: [(Int, Int, Int)] = [
      (128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
    ]
    let samples = 20

    print("")
    print("BACKWARD PASS (forward matmul + 2 gradient matmuls + sum/misc):")
    print("shape              | reg total μs | staged total μs | total speedup | per-GEMM breakdown (dispatch → μs)")
    print("-------------------|--------------|-----------------|---------------|-----------------------------------")
    for (M, K, N) in shapes {
      DGenConfig.gemmStrategy = .registerTiled
      let reg = try gpuTimeBackward(M: M, K: K, N: N, samples: samples)
      DGenConfig.gemmStrategy = .threadgroupStaged
      let stg = try gpuTimeBackward(M: M, K: K, N: N, samples: samples)
      let speedup = reg.allUs / stg.allUs
      let shape = "\(M)x\(K)x\(N)"
      let regGemms = reg.gemmKernels
        .map { String(format: "%@ %.1f", $0.dispatch, $0.gpuMs * 1000) }
        .joined(separator: ", ")
      let stgGemms = stg.gemmKernels
        .map { String(format: "%@ %.1f", $0.dispatch, $0.gpuMs * 1000) }
        .joined(separator: ", ")
      print(
        shape.padding(toLength: 18, withPad: " ", startingAt: 0) + " | "
          + String(format: "%10.1f", reg.allUs) + "   | "
          + String(format: "%13.1f", stg.allUs) + "   | "
          + String(format: "%11.2fx", speedup) + "   |")
      print("                   |              |                 |               | reg:    " + regGemms)
      print("                   |              |                 |               | staged: " + stgGemms)
    }
    print("")
  }

  func testBenchmarkMatmulStrategies() throws {
    guard ProcessInfo.processInfo.environment["DGEN_BENCH"] != nil else {
      throw XCTSkip("Set DGEN_BENCH=1 to run benchmarks")
    }

    let shapes: [(Int, Int, Int)] = [
      (64, 64, 64), (128, 128, 128), (256, 256, 256),
      (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
    ]
    let samples = 20

    print("")
    print("shape              | registerTiled μs | staged μs | speedup | reg dispatch        | staged dispatch")
    print("-------------------|------------------|-----------|---------|---------------------|-------------------------------")
    for (M, K, N) in shapes {
      DGenConfig.gemmStrategy = .registerTiled
      let reg = try gpuTimeMatmul(M: M, K: K, N: N, samples: samples)
      DGenConfig.gemmStrategy = .threadgroupStaged
      let stg = try gpuTimeMatmul(M: M, K: K, N: N, samples: samples)
      let speedup = reg.median / stg.median
      let regUs = reg.median * 1000.0
      let stgUs = stg.median * 1000.0
      let shape = "\(M)x\(K)x\(N)"
      print(
        shape.padding(toLength: 18, withPad: " ", startingAt: 0) + " | "
          + String(format: "%12.1f", regUs) + "     | "
          + String(format: "%7.1f", stgUs) + "   | "
          + String(format: "%5.2fx", speedup) + "   | "
          + reg.dispatch.padding(toLength: 19, withPad: " ", startingAt: 0) + " | "
          + stg.dispatch)
    }
    print("")
  }
}

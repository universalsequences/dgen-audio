import XCTest
@testable import DGenLazy
@testable import DGen

final class KernelAnalysisTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 44100.0
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.sampleRate = 44100.0
    super.tearDown()
  }

  // MARK: - Tensor Analysis

  func testAnalyzeSimpleTensor() throws {
    let a = Tensor([1, 2, 3, 4])
    let b = Tensor([5, 6, 7, 8])
    let c = (a + b).sum()
    let analysis = try c.analyze()

    XCTAssertGreaterThan(analysis.kernelCount, 0)
    XCTAssertGreaterThan(analysis.work, 0)
    XCTAssertGreaterThan(analysis.memoryBytes, 0)
    XCTAssertFalse(analysis.includesBackward)
    print("Simple tensor - Kernels: \(analysis.kernelCount), Work: \(analysis.work), Span: \(analysis.span), Memory: \(analysis.memoryBytes) bytes")
  }

  func testAnalyzeBackward() throws {
    // Forward-only analysis
    let w1 = Tensor([1, 2, 3, 4], requiresGrad: true)
    let fwdLoss = (w1 * 2.0).sum()
    let fwdAnalysis = try fwdLoss.analyze(backward: false)
    XCTAssertFalse(fwdAnalysis.includesBackward)
    XCTAssertGreaterThan(fwdAnalysis.kernelCount, 0)

    // Backward analysis (need fresh graph)
    LazyGraphContext.reset()
    let w2 = Tensor([1, 2, 3, 4], requiresGrad: true)
    let bwdLoss = (w2 * 2.0).sum()
    let bwdAnalysis = try bwdLoss.analyze(backward: true)
    XCTAssertTrue(bwdAnalysis.includesBackward)
    XCTAssertGreaterThan(bwdAnalysis.kernelCount, 0)

    // Backward should have at least as many kernels or more work
    XCTAssertGreaterThanOrEqual(bwdAnalysis.work, fwdAnalysis.work)

    print("Forward - Kernels: \(fwdAnalysis.kernelCount), Work: \(fwdAnalysis.work)")
    print("Backward - Kernels: \(bwdAnalysis.kernelCount), Work: \(bwdAnalysis.work)")
  }

  // MARK: - Signal Analysis

  func testAnalyzeSignal() throws {
    DGenConfig.sampleRate = 1000.0
    let freq = Signal.param(100.0)
    let phase = Signal.phasor(freq)
    let wave = sin(phase * 2 * .pi)
    let analysis = try wave.analyze(frames: 64)

    XCTAssertGreaterThan(analysis.kernelCount, 0)
    XCTAssertGreaterThan(analysis.work, 0)
    XCTAssertGreaterThan(analysis.span, 0)
    XCTAssertFalse(analysis.includesBackward)
    print("Signal - Kernels: \(analysis.kernelCount), Work: \(analysis.work), Span: \(analysis.span)")
  }

  // MARK: - Per-Kernel Breakdown

  func testKernelBreakdown() throws {
    let a = Tensor([1, 2, 3, 4])
    let b = Tensor([5, 6, 7, 8])
    let c = (a + b).sum()
    let analysis = try c.analyze()

    for kernel in analysis.kernels {
      print("\(kernel.name): kind=\(kernel.kind), work=\(kernel.work), span=\(kernel.span), "
            + "arith=\(kernel.arithmeticOps), trans=\(kernel.transcendentalOps), mem=\(kernel.memoryOps)")
    }

    // At least one kernel should have arithmetic ops
    let hasArith = analysis.kernels.contains { $0.arithmeticOps > 0 }
    XCTAssertTrue(hasArith)
  }

  // MARK: - Memory Consistency

  func testMemorySlots() throws {
    let a = Tensor([1, 2, 3, 4])
    let b = Tensor([5, 6, 7, 8])
    let c = (a + b).sum()
    let analysis = try c.analyze()

    XCTAssertEqual(analysis.memoryBytes, analysis.memorySlots * 4)
    XCTAssertGreaterThan(analysis.memorySlots, 0)
  }

  // MARK: - Graph Cleanup

  func testAnalyzeDoesNotPolluteGraph() throws {
    // analyze() should call clearComputationGraph(), so a second
    // analyze on a fresh graph should also work
    let a = Tensor([1, 2, 3])
    let b = a * 2
    let _ = try b.analyze()

    // Build a new graph on the same context — should not crash
    LazyGraphContext.reset()
    let c = Tensor([4, 5, 6])
    let d = c + 1
    let analysis = try d.analyze()
    XCTAssertGreaterThan(analysis.kernelCount, 0)
  }
}

import XCTest

@testable import DGen
@testable import DGenLazy

/// MetalCompiledKernel.run now records a failed command buffer (status .error, e.g. a GPU
/// timeout or fault under memory pressure) in `lastRunError`, and LazyGraph.run throws it.
/// Before, a failed run left zeroed buffers that backward() returned as a loss of exactly 0
/// with garbage gradients (seen in a long batched training run: ~90 steps of loss 0.00000,
/// then diverged weights). A GPU fault cannot be provoked deterministically here, so this
/// pins the other half: ordinary runs must not report an error.
final class MetalRunErrorSurfacingTests: XCTestCase {
  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  func testSuccessfulRunsReportNoError() throws {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = 256
    LazyGraphContext.reset()
    let p = Tensor.param([2], data: [0.5, -0.25])
    for _ in 0..<3 {
      LazyGraphContext.current.clearComputationGraph()
      let s = sin(Signal.phasor(Tensor([440, 660])) * (2 * Float.pi)) * p
      let v = try (s.sum() * s.sum()).backward(frames: 256)
      XCTAssertEqual(v.count, 256)
      XCTAssertTrue(v.contains { $0 != 0 })
      XCTAssertNotNil(p.grad)
    }
    LazyGraphContext.current.clearComputationGraph()
    let y = try (Signal.phasor(Tensor([100])).sum() * 2.0).realize(frames: 256)
    XCTAssertEqual(y.count, 256)
  }

  /// Splitting a run into several command buffers must not change any result.
  func testSplitCommandBuffersGiveIdenticalResults() throws {
    func run(split: Int) throws -> ([Float], [Float]) {
      DGenConfig.metalKernelsPerCommandBuffer = split
      defer { DGenConfig.metalKernelsPerCommandBuffer = 0 }
      DGenConfig.backend = .metal
      DGenConfig.sampleRate = 48_000
      DGenConfig.maxFrameCount = 512
      LazyGraphContext.reset()
      let p = Tensor.param([2, 3], data: [0.5, -0.25, 1, 2, -1, 0.3])
      let q = Tensor.param([1, 3], data: [0.1, 0.2, 0.3])
      let t = Signal.phasor(Tensor([30])).sum()
      let x = exp(t * (p * -3.0)) * q.expand([2, 3])
      let y = (x * x).sum(axis: 1)
      let v = try (y.sum() + (Tensor([0.0]) + (p * p).sum()).peek(Signal.constant(0)) * 0.1).backward(frames: 512)
      return (v, (p.grad?.getData() ?? []) + (q.grad?.getData() ?? []))
    }
    let (v0, g0) = try run(split: 0)
    let (v1, g1) = try run(split: 1)
    XCTAssertEqual(v0, v1)
    XCTAssertEqual(g0.count, 9)
    XCTAssertEqual(g0, g1)
  }
}

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
}

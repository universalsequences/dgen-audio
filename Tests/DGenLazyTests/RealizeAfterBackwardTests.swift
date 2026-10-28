import XCTest

@testable import DGen
@testable import DGenLazy

/// A forward `realize()` issued after `backward()` on the same lazy graph
/// (the train-then-preview loop every trainer uses) must render the current
/// parameter values, however many times it is repeated.
final class RealizeAfterBackwardTests: XCTestCase {
  let frames = 256
  let updated: [Float] = [1, 1, 1, -1, -1, -1]

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    LazyGraphContext.preserveCompilationCaches = false
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func render(_ w: DGenLazy.Tensor) -> SignalTensor {
    let freqs = Tensor([100, 200, 300, 400, 500, 600]).reshape([2, 3])
    return (sin(Signal.phasor(freqs) * (2 * Float.pi)) * w).sum(axis: 1)
  }

  private func reference() throws -> [Float] {
    LazyGraphContext.reset()
    return try render(Tensor(updated).reshape([2, 3])).realize(frames: frames)
  }

  private func trainThenPreview(preserveCaches: Bool) throws -> [[Float]] {
    LazyGraphContext.preserveCompilationCaches = preserveCaches
    LazyGraphContext.reset()
    let w = Tensor.param([2, 3], data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    _ = try render(w).realize(frames: frames)
    LazyGraphContext.current.clearComputationGraph()
    let y = render(w)
    _ = try (y * y).sum().backward(frames: frames)
    w.updateDataLazily(updated)
    w.grad = nil
    var previews: [[Float]] = []
    for _ in 0..<3 {
      LazyGraphContext.current.clearComputationGraph()
      previews.append(try render(w).realize(frames: frames))
    }
    return previews
  }

  func testRealizeAfterBackwardSeesUpdatedParams() throws {
    let ref = try reference()
    XCTAssertEqual(ref.count, frames * 2)
    for preserve in [false, true] {
      let previews = try trainThenPreview(preserveCaches: preserve)
      for (i, p) in previews.enumerated() {
        XCTAssertEqual(p.count, ref.count, "preserveCaches=\(preserve) preview \(i) count")
        guard p.count == ref.count else { continue }
        let diff = zip(p, ref).map { Swift.abs($0 - $1) }.max() ?? 0
        XCTAssertLessThan(diff, 1e-5, "preserveCaches=\(preserve) preview \(i)")
      }
    }
  }
}

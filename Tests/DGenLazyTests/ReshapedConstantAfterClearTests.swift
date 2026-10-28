import XCTest

@testable import DGen
@testable import DGenLazy

/// Training loops build constant tables once (`Tensor(flat).reshape([F, B])`)
/// and rebuild the graph every step. A reshaped constant carries its data but
/// was neither registered for refresh nor refreshed by every consumer (e.g.
/// sampleRow), so after clearComputationGraph() its stale node ID aliased an
/// unrelated node: wrong gradients, or a shape error when the alias was a
/// different shape.
final class ReshapedConstantAfterClearTests: XCTestCase {
  let frames = 512

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func loss(_ a: DGenLazy.Tensor, _ c: DGenLazy.Tensor, table: DGenLazy.Tensor, target: DGenLazy.Tensor) -> Signal {
    let rate = Float(48_000) / Float(2 * frames)
    let t = Signal.phasor(Tensor([rate])).sum() * (1 / rate)
    let ph = floor(t * 48_000 + 0.5)
    let env = exp(t * (a * -20.0)) * c.expand([2, 3])
    let bands = table.sampleRow(ph).reshape([1, 3]).expand([2, 3])
    let d = (env * bands).sum(axis: 1) - target.sampleRow(ph)
    return (d * d).sum()
  }

  private func gradients(clearFirst: Bool) throws -> [Float] {
    LazyGraphContext.reset()
    let tdata = (0..<(frames * 3)).map { Float(Foundation.sin(Double($0) * 0.37)) }
    let gdata = (0..<(frames * 2)).map { Float(Foundation.cos(Double($0) * 0.11)) * 0.3 }
    let table = Tensor(tdata).reshape([frames, 3])
    let target = Tensor(gdata).reshape([frames, 2])
    let a = Tensor.param([2, 3], data: [1, 2, 3, 4, 5, 6])
    let c = Tensor.param([2, 1], data: [0.5, -0.7])
    let unused = Tensor.param([2, 3], data: [0, 0, 0, 0, 0, 0])
    _ = unused
    if clearFirst { LazyGraphContext.current.clearComputationGraph() }
    _ = try loss(a, c, table: table, target: target).backward(frames: frames)
    return (a.grad?.getData() ?? []) + (c.grad?.getData() ?? [])
  }

  func testGradientsAreIdenticalAfterClear() throws {
    let fresh = try gradients(clearFirst: false)
    let cleared = try gradients(clearFirst: true)
    XCTAssertEqual(fresh.count, 8)
    XCTAssertEqual(cleared.count, 8)
    for (f, c) in zip(fresh, cleared) {
      XCTAssertEqual(f, c, accuracy: 1e-4 * Swift.max(1, Swift.abs(f)))
    }
  }
}

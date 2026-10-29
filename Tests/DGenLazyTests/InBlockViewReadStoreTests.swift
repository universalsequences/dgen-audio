import XCTest

@testable import DGen
@testable import DGenLazy

/// A static tensor computed in a block and read by a node of the same block through a
/// view (shrink/expand/pad/...) must be stored: the view maps the consumer's element to
/// a different base element and reads it from memory. Found building
/// Examples/ModalKickAD (per-lane "spread about the sub's log f" knob): on the C
/// backend `log(x).shrink(col 0).expand([B, K])` read the clock phasor's frame buffer.
final class InBlockViewReadStoreTests: XCTestCase {
  let frames = 16

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = frames
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func frameIndex() -> Signal {
    let rampHz = Float(48_000) / (2 * Float(frames))
    return Signal.phasor(Tensor([rampHz])).sum() * (2 * Float(frames))
  }

  private let base: [Float] = [1, 2, 3, 4, 10, 20, 30, 40]  // [B=2, K=4]

  private func runShrinkExpandOfComputed(_ backend: Backend) throws -> [Float] {
    DGenConfig.backend = backend
    LazyGraphContext.reset()
    let x = log(Tensor(base).reshape([2, 4]))
    let col0 = x.shrink([(0, 2), (0, 1)]).expand([2, 4])
    let y = (frameIndex() * col0).sum(axis: 1)  // [2]
    let flat = try y.realize(frames: frames)
    return [flat[5 * 2], flat[5 * 2 + 1]]  // frame 5
  }

  private func runLaneScaledLogMix(_ backend: Backend) throws -> [Float] {
    DGenConfig.backend = backend
    LazyGraphContext.reset()
    let x = log(Tensor(base).reshape([2, 4]))
    let sub = x.shrink([(0, 2), (0, 1)]).expand([2, 4])
    let kmul = Tensor([1, 1, 1, 1, 1, 1, 1.5, 1.5]).reshape([2, 4])
    let f = (sub + kmul * (x - sub)).exp()
    let y = (frameIndex() * f).sum(axis: 1)
    let flat = try y.realize(frames: frames)
    return [flat[5 * 2], flat[5 * 2 + 1]]
  }

  private func expectedLaneScaledLogMix() -> [Float] {
    let km: [Float] = [1, 1, 1, 1, 1, 1, 1.5, 1.5]
    return (0..<2).map { b in
      var s: Float = 0
      for k in 0..<4 {
        let xv = Foundation.log(base[b * 4 + k]), sv = Foundation.log(base[b * 4])
        s += Foundation.exp(sv + km[b * 4 + k] * (xv - sv))
      }
      return 5 * s
    }
  }

  func testShrinkExpandOfInBlockComputedTensorC() throws {
    let got = try runShrinkExpandOfComputed(.c)
    XCTAssertEqual(got[0], 0, accuracy: 1e-4)
    XCTAssertEqual(got[1], 5 * 4 * Foundation.log(10), accuracy: 1e-3)  // was 2.5
  }

  func testLaneScaledLogMixC() throws {
    let got = try runLaneScaledLogMix(.c)
    let want = expectedLaneScaledLogMix()
    XCTAssertEqual(got[0], want[0], accuracy: 1e-3 * want[0])
    XCTAssertEqual(got[1], want[1], accuracy: 1e-3 * want[1])  // was 2110 vs 809.8
  }
}

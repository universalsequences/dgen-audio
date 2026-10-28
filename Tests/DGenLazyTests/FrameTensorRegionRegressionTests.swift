import XCTest

@testable import DGen
@testable import DGenLazy

/// Regressions found while building Examples/ModalSnareAD (closed-form [B,K]
/// modal banks driven by a frame-parallel clock).
final class FrameTensorRegionRegressionTests: XCTestCase {
  let frames = 512
  let sr: Float = 48_000

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sr
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = DGenConfig.defaultFrameCount
    super.tearDown()
  }

  private func seconds() -> Signal {
    let rampHz = sr / (2 * Float(frames))
    return Signal.phasor(Tensor([rampHz])).sum() * (2 * Float(frames) / sr)
  }

  /// `(x * a).sum(axis: 1)` where x is a frame tensor produced in the same
  /// block used to return all zeros: the inline mul-reduce read x from memory,
  /// but x's only consumer was the elided mul, so its store was skipped.
  func testSumAxisOfMulReadsInBlockOperand() throws {
    let data: [Float] = [1, 2, 3, 4, 5, 6]
    let a = Tensor(data).reshape([2, 3])
    let x = seconds() * a
    let out = try (x * a).sum(axis: 1).realize(frames: frames)
    XCTAssertEqual(out.count, frames * 2)
    for n in [1, 100, 511] {
      let t = Float(n) / sr
      XCTAssertEqual(out[n * 2], t * (1 + 4 + 9), accuracy: 1e-5)
      XCTAssertEqual(out[n * 2 + 1], t * (16 + 25 + 36), accuracy: 1e-5)
    }
  }

  /// Literal scalars (`x * -1.0 + 1.0`) used to open a [1] shape region each,
  /// splitting a frame-tensor elementwise chain into one element loop per op
  /// and materializing every intermediate as a [frames, B*K] buffer.
  func testLiteralsDoNotSplitElementwiseChain() throws {
    let B = 4, K = 8
    let rates = (0..<(B * K)).map { Float($0 + 1) * 3 }
    func chain() -> SignalTensor {
      let r = Tensor(rates).reshape([B, K])
      let t = seconds() * r
      var y = exp(t * -1.0)
      for _ in 0..<12 { y = (y * -1.0 + 1.0) * 0.5 + 0.25 }
      return y
    }
    let out = try chain().sum(axis: 1).realize(frames: frames)
    // CPU reference.
    for n in [0, 37, 511] {
      for b in 0..<B {
        var acc: Float = 0
        for k in 0..<K {
          var y = Foundation.exp(-Float(n) / sr * rates[b * K + k])
          for _ in 0..<12 { y = (1 - y) * 0.5 + 0.25 }
          acc += y
        }
        XCTAssertEqual(out[n * B + b], acc, accuracy: 1e-4)
      }
    }
    // Count frame-tensor stores in the emitted Metal: the split chain stored
    // every intermediate (~50 stores); fused, only the reduce input/output remain.
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let root = chain().sum(axis: 1).sum()
    _ = lazy.node(.output(0), [root.nodeId])
    let result = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .metal, options: .init(frameCount: frames))
    let stores = result.kernels.map { k in
      k.source.split(separator: "\n").filter { $0.hasPrefix("memory[") }.count
    }.reduce(0, +)
    XCTAssertLessThan(stores, 8, "elementwise chain was split into per-op loops")
  }
}

import XCTest

@testable import DGen
@testable import DGenLazy

private typealias Tensor = DGenLazy.Tensor

/// `RunInvariantCache`: a cached build renders exactly what the live graph renders, and only the
/// run-invariant part (noise through a literal-coefficient filter) is cached.
final class RunInvariantCacheTests: XCTestCase {
  private let frames = 512

  /// White noise through a one-pole y = c·y[-1] + noise, scaled by `gain`.
  private func voice(gain: Tensor, coefficient: Tensor) -> Signal {
    let h = TensorHistory(shape: [1])
    let y = h.read() * coefficient + Signal.noise()
    h.write(y)
    return (y * gain).sum()
  }

  /// The frame number, as an integer-valued signal (a phasor at the graph's own sample rate: other
  /// suites change the default).
  private func frameIndex() -> Signal {
    let n = Float(frames), sr = LazyGraphContext.current.graph.sampleRate
    return floor(Signal.phasor(Tensor([sr / (2 * n)])).sum() * (2 * n) + 0.5)
  }

  private func literal(_ v: Float) -> Tensor {
    let t = Tensor([v])
    t.markConstantData()
    return t
  }

  private func hasNoise() -> Bool {
    LazyGraphContext.current.graph.nodes.values.contains { if case .noise = $0.op { return true } else { return false } }
  }

  func testCachedRenderMatchesLiveRender() throws {
    LazyGraphContext.reset()
    let gain = Tensor([0.7], requiresGrad: true)
    let live = try voice(gain: gain, coefficient: literal(0.9)).realize(frames: frames)
    XCTAssertGreaterThan(live.map(abs).max() ?? 0, 0.01, "the noise must be audible for the check to mean anything")

    let cache = RunInvariantCache()
    for build in 0..<2 {  // the first build probes, the second reads the cached table only
      LazyGraphContext.current.clearComputationGraph()
      let out = try cache.build(frames: frames, frameIndex: frameIndex) { voice(gain: gain, coefficient: literal(0.9)) }
      XCTAssertEqual(cache.lastSubstituted, 1, "build \(build)")
      XCTAssertEqual(cache.cachedCount, 1, "build \(build)")
      XCTAssertFalse(hasNoise(), "build \(build): the noise subgraph should be gone")
      XCTAssertEqual(try out.realize(frames: frames), live, "build \(build)")
    }
  }

  func testLearnableCoefficientIsNotCached() throws {
    LazyGraphContext.reset()
    let gain = Tensor([0.7], requiresGrad: true)
    let coefficient = Tensor([0.9], requiresGrad: true)
    let live = try voice(gain: gain, coefficient: coefficient).realize(frames: frames)
    LazyGraphContext.current.clearComputationGraph()
    let cache = RunInvariantCache()
    let out = try cache.build(frames: frames, frameIndex: frameIndex) { voice(gain: gain, coefficient: coefficient) }
    // the filter state depends on a parameter: only the raw noise feeding it is invariant
    XCTAssertEqual(cache.lastSubstituted, 1)
    // the same values, but the filter's multiply-add now reads its noise from a table: the
    // compiler may contract it differently, so allow last-bit differences
    let got = try out.realize(frames: frames)
    let scale = live.map(abs).max() ?? 1
    XCTAssertEqual(got.count, live.count)
    XCTAssertLessThanOrEqual(zip(live, got).map { abs($0 - $1) }.max() ?? 0, 1e-6 * scale)
  }
}

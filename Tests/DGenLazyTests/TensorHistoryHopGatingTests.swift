import XCTest

@testable import DGen
@testable import DGenLazy

/// Can `TensorHistory` feedback be HOP-GATED (advance once per hop) so it works
/// inside STFT-style spectral pipelines (per-bin dynamics, spectral freeze,
/// temporal blur, the soothe-class per-bin compressor)?
///
/// Without a hop, tensor-history feedback advances once per FRAME (see
/// `HistoryTensorTests.testTensorAccumulator`). `TensorHistory(shape:hop:)` makes
/// the same feedback advance once every `hop` frames (fs/hop) instead — the state
/// at hop N depends on hop N-1, ticking at the STFT frame rate.
///
/// Test: accumulate a hop-gated buffered window into per-element state.
///   - input is constant 1.0, so once the ring fills the [1,size] window sums to
///     `size`, and each *update* adds `size`.
///   - the update only fires on hop frames (0, hop, 2*hop, ...).
///
/// With `size=4, hop=4`, the per-frame sum is identical on C and Metal:
///   [1, 0,0,0, 5, 0,0,0, 9, 0,0,0, 13, 0,0,0, 17, 0,0,0]
/// i.e. on hop frames it accumulates 1 → 5 → 9 → 13 → 17 (state persists across
/// hops; +4 per hop after the ring fills). Between hops the hop-gated value reads
/// 0 — a hop value is only defined on its hop frame; real spectral pipelines
/// consume it inside the hop region (multiply → IFFT → overlapAdd, where the OLA
/// ring holds per frame). So we assert on the hop-frame samples, not the gaps.
final class TensorHistoryHopGatingTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private let size = 4
  private let hop = 4
  private let frameCount = 20

  /// Constant 1.0 input → hop-gated [size] window → per-element state accumulator.
  /// `historyHop == nil`: per-frame feedback. `historyHop == hop`: per-hop feedback.
  private func runHopAccumulator(backend: Backend, historyHop: Int?) throws -> [Float] {
    DGenConfig.backend = backend
    LazyGraphContext.reset()

    let sig = Signal.constant(1.0)
    let buffered = sig.buffer(size: size, hop: hop).reshape([size])  // hop-gated [size]

    let history = TensorHistory(shape: [size], hop: historyHop)
    let prev = history.read()
    let newValue = prev + buffered  // update driven by the hop-gated buffer
    history.write(newValue)

    return try newValue.sum().realize(frames: frameCount)
  }

  private func deltas(_ r: [Float]) -> [Float] { zip(r.dropFirst(), r).map { $0 - $1 } }

  // MARK: - Default: per-frame feedback (unchanged; per-sample history intact)

  func testDefault_AdvancesPerFrame() throws {
    for backend in [Backend.c, .metal] {
      let r = try runHopAccumulator(backend: backend, historyHop: nil)
      let name = backend == .c ? "C" : "Metal"
      print("\n=== [\(name)] (no hop) per-frame sum: \(r)")

      // No hop: the accumulator ticks EVERY frame. Once the ring is full
      // (frame >= 2) every delta equals `size` (4.0).
      for i in 3..<r.count {
        XCTAssertEqual(
          r[i] - r[i - 1], 4.0, accuracy: 1e-4,
          "[\(name)] expected per-frame advance of 4.0 at frame \(i)")
      }
    }
  }

  // MARK: - Hop-gated: per-hop feedback (the STFT spectral-dynamics substrate)

  func testHopGated_AdvancesPerHop() throws {
    let c = try runHopAccumulator(backend: .c, historyHop: hop)
    let metal = try runHopAccumulator(backend: .metal, historyHop: hop)
    print("\n=== [C]     (hop=\(hop)) \(c)")
    print("=== [Metal] (hop=\(hop)) \(metal)")

    for (backend, r) in [("C", c), ("Metal", metal)] {
      // The update fires ONLY on hop frames — between hops nothing runs and the
      // hop-gated value reads 0.
      for i in 0..<r.count where i % hop != 0 {
        XCTAssertEqual(r[i], 0.0, accuracy: 1e-4, "[\(backend)] mid-hop frame \(i) should be 0")
      }

      // On hop frames the state PERSISTS across hops and accumulates: each hop
      // adds the buffered window sum. After the ring fills that is +size per hop,
      // so hop-frame values strictly increase. (If hop-gating silently failed and
      // history reset each hop, these would be flat — the original Metal bug.)
      let hopValues = stride(from: 0, to: r.count, by: hop).map { r[$0] }
      XCTAssertEqual(hopValues, [1, 5, 9, 13, 17], "[\(backend)] per-hop accumulation")
      for k in 1..<hopValues.count {
        XCTAssertGreaterThan(
          hopValues[k], hopValues[k - 1],
          "[\(backend)] hop \(k) did not accumulate (state not persisting across hops)")
      }
    }

    // Strongest signal the codegen is correct: both backends agree exactly.
    XCTAssertEqual(c, metal, "C and Metal must produce identical hop-gated output")
  }
}

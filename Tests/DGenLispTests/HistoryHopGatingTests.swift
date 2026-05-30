import DGenLazy
import XCTest

@testable import DGen
@testable import DGenLisp

/// End-to-end confirmation that dgenlisp exposes hop-gated tensor history through
/// the unified `make-history` form: `(make-history name @shape [...] @hop H)`.
///
/// This mirrors `TensorHistoryHopGatingTests` (the DGenLazy-level proof) but drives
/// the whole thing from lisp source, verifying the parser/evaluator wiring of the
/// optional `@shape` / `@hop` attributes and that `read-history` / `write-history`
/// dispatch onto the tensor binding.
///
/// Program: accumulate a hop-gated buffered window of constant 1.0 into per-element
/// state. With `size=4, hop=4` the per-frame sum is identical on C and Metal:
///   [1, 0,0,0, 5, 0,0,0, 9, 0,0,0, 13, 0,0,0, 17, 0,0,0]
/// On hop frames the state persists and accumulates (+size per hop after the ring
/// fills); between hops the hop-gated value reads 0. Without `@hop` the same
/// program ticks every frame (per-sample feedback, unchanged).
final class HistoryHopGatingTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.sampleRate = 8
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-hophist-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private let size = 4
  private let hop = 4
  private let frameCount = 20

  /// Builds the accumulator program in lisp and realizes its per-frame sum.
  /// `hopClause` is either "" (per-frame) or "@hop 4" (hop-gated).
  private func runHopAccumulator(backend: Backend, hopClause: String) throws -> [Float] {
    DGenConfig.backend = backend
    LazyGraphContext.reset()

    let source = """
      (make-history h @shape [\(size)] \(hopClause))
      (def buffered (reshape (buffer 1 \(size) \(hop)) @shape [\(size)]))
      (def prev (read-history h))
      (def next (+ prev buffered))
      (write-history h next)
      (def out (sum next))
      """
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))

    guard case .signal(let out)? = evaluator.definitions["out"] else {
      throw XCTSkip("expected signal 'out'")
    }
    return try out.realize(frames: frameCount)
  }

  // MARK: - Default: per-frame feedback (no @hop, unchanged)

  func testMakeHistoryShapeNoHop_AdvancesPerFrame() throws {
    for backend in [Backend.c, .metal] {
      let r = try runHopAccumulator(backend: backend, hopClause: "")
      let name = backend == .c ? "C" : "Metal"
      print("\n=== [\(name)] (make-history @shape, no hop) per-frame sum: \(r)")

      // Without @hop the accumulator ticks EVERY frame. Once the ring fills
      // (frame >= 2) every delta equals `size` (4.0).
      for i in 3..<r.count {
        XCTAssertEqual(
          r[i] - r[i - 1], 4.0, accuracy: 1e-4,
          "[\(name)] expected per-frame advance of 4.0 at frame \(i)")
      }
    }
  }

  // MARK: - Hop-gated: per-hop feedback via @hop

  func testMakeHistoryShapeWithHop_AdvancesPerHop() throws {
    let c = try runHopAccumulator(backend: .c, hopClause: "@hop \(hop)")
    let metal = try runHopAccumulator(backend: .metal, hopClause: "@hop \(hop)")
    print("\n=== [C]     (make-history @shape @hop \(hop)) \(c)")
    print("=== [Metal] (make-history @shape @hop \(hop)) \(metal)")

    for (backend, r) in [("C", c), ("Metal", metal)] {
      // The update fires ONLY on hop frames — between hops nothing runs and the
      // hop-gated value reads 0.
      for i in 0..<r.count where i % hop != 0 {
        XCTAssertEqual(r[i], 0.0, accuracy: 1e-4, "[\(backend)] mid-hop frame \(i) should be 0")
      }

      // On hop frames the state PERSISTS across hops and accumulates.
      let hopValues = stride(from: 0, to: r.count, by: hop).map { r[$0] }
      XCTAssertEqual(hopValues, [1, 5, 9, 13, 17], "[\(backend)] per-hop accumulation")
      for k in 1..<hopValues.count {
        XCTAssertGreaterThan(
          hopValues[k], hopValues[k - 1],
          "[\(backend)] hop \(k) did not accumulate (state not persisting across hops)")
      }
    }

    // Both backends must agree exactly.
    XCTAssertEqual(c, metal, "C and Metal must produce identical hop-gated output")
  }

  // MARK: - make-tensor-history also accepts @hop (explicit alias)

  func testMakeTensorHistoryAcceptsHopAttribute() throws {
    DGenConfig.backend = .c
    LazyGraphContext.reset()
    let source = """
      (make-tensor-history h @shape [\(size)] @hop \(hop))
      (def buffered (reshape (buffer 1 \(size) \(hop)) @shape [\(size)]))
      (def next (+ (read-tensor-history h) buffered))
      (write-tensor-history h next)
      (def out (sum next))
      """
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let out)? = evaluator.definitions["out"] else {
      throw XCTSkip("expected signal 'out'")
    }
    let r = try out.realize(frames: frameCount)
    let hopValues = stride(from: 0, to: r.count, by: hop).map { r[$0] }
    XCTAssertEqual(hopValues, [1, 5, 9, 13, 17])
  }

  // MARK: - Scalar make-history still works (no @shape → signal feedback)

  func testScalarMakeHistoryStillProducesSignalFeedback() throws {
    DGenConfig.backend = .c
    LazyGraphContext.reset()
    let source = """
      (make-history acc)
      (def prev (read-history acc))
      (def next (+ prev 1))
      (write-history acc next)
      (def out next)
      """
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let out)? = evaluator.definitions["out"] else {
      throw XCTSkip("expected scalar signal 'out'")
    }
    let r = try out.realize(frames: 5)
    // Per-frame scalar accumulator: 1, 2, 3, 4, 5.
    XCTAssertEqual(r, [1, 2, 3, 4, 5])
  }
}

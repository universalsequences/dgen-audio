import DGenLazy
import Foundation
import XCTest

@testable import DGenLisp

/// The dataset-training forms (DatasetForms.swift): identity-like in compiles, learnable leaves
/// under a DatasetContext, and the one-shot lowering's refusal of frame-varying latches.
final class DatasetFormsTests: XCTestCase {
  private var tempDir: URL!
  private let sr: Float = 1000
  private let frames = 64

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = sr
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-dataset-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    // w: [2,3] decoder weights, b: [1,3]
    try #"{"shape":[2,3],"data":[0.5,-0.25,1.0,0.0,0.75,-0.5]}"#.write(
      to: tempDir.appendingPathComponent("w.json"), atomically: true, encoding: .utf8)
    try #"{"shape":[1,3],"data":[0.1,0.2,-0.3]}"#.write(
      to: tempDir.appendingPathComponent("b.json"), atomically: true, encoding: .utf8)
  }

  override func tearDownWithError() throws {
    DGenConfig.maxFrameCount = 4096
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  /// One-row voice: u = z @ w + b, f/d/a from the layout, y = a e^{-t/d} sin(2π f t).
  private let voice = """
    (def trigger (click))
    (row-layout v (f 1 log 20 400) (d 1 log 0.001 1) (a 1 lin 0 2))
    (param z1 @default 0.5)
    (param z2 @default -0.25)
    (def zf (+ (* z1 (tensor @shape [1 2] @data [1 0])) (* z2 (tensor @shape [1 2] @data [0 1]))))
    (def z (event-hold (latent-table @name codes @dim 2 zf) trigger))
    (def w (tensor @shape [2 3] @file "w.json" @learnable true))
    (def b (tensor @shape [1 3] @file "b.json" @learnable true @init zeros))
    (def u (param-row (+ (matmul z w) b) @layout v))
    (def t (hit-time trigger))
    (def f (latch (seg u v f) trigger))
    (def d (latch (seg u v d) trigger))
    (def a (latch (seg u v a) trigger))
    (def y (sum-axis (* a (exp (/ (* -1 t) d)) (sin (* twopi f t))) @axis 1))
    (out (row-out y) 1)
    """

  private func sigmoid(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }

  private func expected(_ u: [Float]) -> [Float] {
    let f = 20 * Foundation.pow(400 / 20, sigmoid(u[0]))
    let d = 0.001 * Foundation.pow(1000, sigmoid(u[1]))
    let a = 2 * sigmoid(u[2])
    return (0..<frames).map { n in
      let t = Float(n) / sr
      return a * Foundation.exp(-t / d) * Foundation.sin(2 * .pi * f * t)
    }
  }

  func testCompileModeIsIdentityLike() throws {
    let ev = LispEvaluator(sourceDirectory: tempDir)
    try ev.evaluate(nodes: parseSource(voice))
    XCTAssertEqual(ev.rowLayouts["v"]?.width, 3)
    XCTAssertEqual(ev.latentTables, [LatentTableInfo(name: "codes", dim: 2)])
    XCTAssertEqual(ev.learnableTensors.map(\.name), ["w", "b"])
    XCTAssertEqual(ev.paramRows, [ParamRowInfo(name: "row", layout: "v")])
    guard let out = ev.outputs.first?.signal else { return XCTFail("row-out must be a scalar output") }
    let y = try out.realize(frames: frames)
    // z = [0.5, -0.25]; u = z @ w + b
    let u: [Float] = [0.5 * 0.5 + -0.25 * 0.0 + 0.1, 0.5 * -0.25 + -0.25 * 0.75 + 0.2, 0.5 * 1.0 + -0.25 * -0.5 - 0.3]
    for (a, e) in zip(y, expected(u)) { XCTAssertEqual(a, e, accuracy: 2e-4) }
  }

  func testDatasetModeUsesLeavesAndRows() throws {
    let rows: [[Float]] = [[0.2, -1.0, 0.5], [-0.4, 0.3, -0.2], [1.1, 0.0, 0.9]]
    let leaf = Tensor(rows.flatMap { $0 }).reshape([3, 3])
    var asked: [String] = []
    let ctx = DatasetContext(
      rows: 3, frames: frames, sampleRate: sr, rowMode: .leaf(leaf),
      learnable: { name, shape, _, _ in asked.append(name); return Tensor.param(shape, data: [Float](repeating: 0, count: shape.reduce(1, *))) },
      latentRows: { _, dim in Tensor.param([3, dim], data: [Float](repeating: 0, count: 3 * dim)) })
    let ev = LispEvaluator(sourceDirectory: tempDir)
    ev.dataset = ctx
    try ev.evaluate(nodes: parseSource(voice))
    // top-level defs still create their leaves; only param-row's argument is skipped
    XCTAssertEqual(asked, ["w", "b"])
    guard let out = ev.tensorOutputs.first?.signal else { return XCTFail("expected a [3] tensor output") }
    XCTAssertEqual(out.shape, [3])
    let flat = try out.realize(frames: frames)
    for r in 0..<3 {
      let e = expected(rows[r])
      for n in 0..<frames { XCTAssertEqual(flat[n * 3 + r], e[n], accuracy: 2e-4, "row \(r) frame \(n)") }
    }
  }

  func testCaptureStopsAtParamRow() throws {
    let ctx = DatasetContext(
      rows: 2, frames: frames, sampleRate: sr, rowMode: .capture,
      learnable: { _, shape, file, _ in Tensor.param(shape, data: file ?? []) },
      latentRows: { _, dim in Tensor.param([2, dim], data: [1, 0, 0, 1]) })
    let ev = LispEvaluator(sourceDirectory: tempDir)
    ev.dataset = ctx
    XCTAssertThrowsError(try ev.evaluate(nodes: parseSource(voice))) { XCTAssert($0 is DatasetCaptureDone) }
    XCTAssertEqual(ctx.captured["row"]?.shape, [2, 3])
    XCTAssertTrue(ev.tensorOutputs.isEmpty, "the voice must not be built in capture mode")
  }

  func testFrameVaryingLatchIsRefusedUnderTraining() throws {
    let ctx = DatasetContext(
      rows: 1, frames: frames, sampleRate: sr, rowMode: .decode,
      learnable: { _, shape, _, _ in Tensor.param(shape, data: [Float](repeating: 0, count: shape.reduce(1, *))) },
      latentRows: { _, dim in Tensor.param([1, dim], data: [Float](repeating: 0, count: dim)) })
    let ev = LispEvaluator(sourceDirectory: tempDir)
    ev.dataset = ctx
    XCTAssertThrowsError(try ev.evaluate(nodes: parseSource("(def x (latch (phasor 3) (click)))")))
    // static values pass through
    let ok = LispEvaluator(sourceDirectory: tempDir)
    ok.dataset = ctx
    XCTAssertNoThrow(try ok.evaluate(nodes: parseSource("(def x (event-hold (tensor @shape [1 2] @data [1 2]) (click)))")))
    guard case .tensor? = ok.definitions["x"] else { return XCTFail("one-shot event-hold of a tensor is the tensor") }
  }

  func testRowSegmentSquashRoundTrip() {
    let segs = [
      RowSegment(name: "a", offset: 0, width: 1, squash: .log, lo: 20, hi: 8000),
      RowSegment(name: "b", offset: 1, width: 1, squash: .lin, lo: -1, hi: 3),
      RowSegment(name: "c", offset: 2, width: 1, squash: .raw, lo: 0, hi: 0),
    ]
    for (s, v) in zip(segs, [Float(440), 0.7, -2.5]) {
      XCTAssertEqual(s.fromU(s.toU(v)), v, accuracy: abs(v) * 1e-4)
    }
    // out-of-box values clamp inside the box instead of becoming infinite
    XCTAssertTrue(segs[0].toU(1e6).isFinite)
  }

  func testCheckpointRoundTrip() throws {
    let store = LeafStore()
    let a = store.leaf("w:a", [2, 2]) { [1, 2, 3, 4] }
    _ = store.leaf("z:codes", [1, 3]) { [0.5, -0.5, 0.25] }
    var adam = AdamMoments()
    adam.step(store, names: ["w:a"], grads: ["w:a": [0.1, -0.2, 0.3, -0.4]], lr: 0.01)
    let url = tempDir.appendingPathComponent("ck.json")
    try DatasetCheckpoint.write(url, stage: "refine", step: 7, store: store, adam: adam)
    let l = try DatasetCheckpoint.load(url)
    XCTAssertEqual(l.step, 7)
    XCTAssertEqual(l.stage, "refine")
    XCTAssertEqual(l.leaves["w:a"], a.getData())
    XCTAssertEqual(l.leaves["z:codes"], [0.5, -0.5, 0.25])
    XCTAssertEqual(l.adam.t, 1)
    XCTAssertEqual(l.adam.m["w:a"], adam.m["w:a"])
    XCTAssertEqual(l.adam.v["w:a"], adam.v["w:a"])
  }
}

import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// `bindingShapes` / `failedForm`: what a patch editor reads to draw tensor
/// (multichannel) cables and to mark the node an error came from. A patch
/// editor emits one top-level form per node, so top-level bindings are nodes.
final class BindingShapesLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-binding-shapes-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  /// Evaluates `source`, returning the evaluator whether or not it threw.
  private func evaluate(_ source: String) -> (LispEvaluator, Error?) {
    let e = LispEvaluator(sourceDirectory: tempDir)
    do {
      try e.evaluate(nodes: parseSource(source))
      return (e, nil)
    } catch {
      return (e, error)
    }
  }

  func testTopLevelBindingsReportKindAndShape() {
    let (e, error) = evaluate(
      """
      (def p (in 2))
      (def k 0.5)
      (def t (* (iota 8) 2))
      (def ph (phasor (* p t)))
      (def osc (cos (* ph twopi)))
      (make-history h @shape [8])
      (def mix (sum osc))
      (out mix 1)
      """)
    XCTAssertNil(error)
    XCTAssertEqual(e.bindingShapes["p"], BindingShape(kind: .signal, shape: []))
    XCTAssertEqual(e.bindingShapes["k"], BindingShape(kind: .constant, shape: []))
    XCTAssertEqual(e.bindingShapes["t"], BindingShape(kind: .tensor, shape: [8]))
    XCTAssertEqual(e.bindingShapes["ph"], BindingShape(kind: .signalTensor, shape: [8]))
    XCTAssertEqual(e.bindingShapes["osc"], BindingShape(kind: .signalTensor, shape: [8]))
    XCTAssertEqual(e.bindingShapes["h"], BindingShape(kind: .tensorHistory, shape: [8]))
    XCTAssertEqual(
      e.bindingShapes["mix"], BindingShape(kind: .signal, shape: []),
      "sum folds the multichannel signal back to a scalar")
    XCTAssertNil(e.failedForm)
  }

  func testBroadcastShapesAreReportedAsInferred() {
    let (e, error) = evaluate(
      """
      (def a (phasor (* (in 1) (reshape (iota 8) @shape [8 1]))))
      (def b (+ a (iota 8)))
      (out (sum b) 1)
      """)
    XCTAssertNil(error)
    XCTAssertEqual(e.bindingShapes["a"]?.shape, [8, 1])
    XCTAssertEqual(e.bindingShapes["b"]?.shape, [8, 8], "[8 1] with [8] broadcasts NumPy-style")
  }

  func testMacroLocalDefsAreNotReportedButTheCallSiteIs() {
    let (e, error) = evaluate(
      """
      (defmacro spread (hz n) (def lanes (iota n)) (* hz lanes))
      (def bank (phasor (spread (in 1) 4)))
      (out (sum bank) 1)
      """)
    XCTAssertNil(error)
    XCTAssertEqual(e.bindingShapes["bank"], BindingShape(kind: .signalTensor, shape: [4]))
    XCTAssertFalse(
      e.bindingShapes.keys.contains { $0.contains("lanes") },
      "macro-local defs are scoped (_mN_…) and not nodes: \(e.bindingShapes.keys)")
  }

  func testAFailingDefIsNamedWithTheShapesBoundBeforeIt() {
    let (e, error) = evaluate(
      """
      (def p (in 2))
      (def t (* (iota 8) 2))
      (make-history h @shape [8 1])
      (def b (biquad (read-history h) (* p t) 0.777 0 1))
      (out (sum b) 1)
      """)
    XCTAssertNotNil(error)
    XCTAssertEqual(e.failedForm?.op, "def")
    XCTAssertEqual(e.failedForm?.name, "b")
    XCTAssertEqual(e.failedForm?.index, 3)
    XCTAssertEqual(e.failedForm?.label, "(def b)")
    XCTAssertTrue(e.failedForm?.message.contains("biquad") ?? false)
    XCTAssertEqual(e.bindingShapes["t"]?.shape, [8], "shapes before the error are kept")
    XCTAssertNil(e.bindingShapes["b"], "the failing binding has no shape")
  }

  func testAFailingHistoryWriteNamesTheHistory() {
    let (e, error) = evaluate(
      """
      (def t (iota 8))
      (make-history h @shape [8])
      (def big (+ (read-history h) (reshape t @shape [8 1])))
      (write-history h big)
      (out (sum big) 1)
      """)
    XCTAssertNotNil(error)
    XCTAssertEqual(e.failedForm?.op, "write-history")
    XCTAssertEqual(e.failedForm?.name, "h", "the history node is the one to mark")
    XCTAssertEqual(e.bindingShapes["big"]?.shape, [8, 8])
  }

  func testAFailingFormInsideAMacroIsAttributedToTheCallingNode() {
    let (e, error) = evaluate(
      """
      (defmacro bad (x) (def y (frobnicate x)) y)
      (def v (bad (iota 4)))
      (out 0 1)
      """)
    XCTAssertNotNil(error)
    XCTAssertEqual(e.failedForm?.op, "def")
    XCTAssertEqual(e.failedForm?.name, "v")
  }

  func testShapesReportFileIsWrittenWithFailedForm() throws {
    let (e, _) = evaluate("(def t (iota 4))\n(def u (frobnicate t))\n(out 0 1)\n")
    let dir = tempDir.appendingPathComponent("out/nested").path
    writeBindingShapes(e, to: dir, name: "patch")
    let data = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/patch.shapes.json"))
    let report = try JSONDecoder().decode(BindingShapesReport.self, from: data)
    XCTAssertEqual(report.bindingShapes["t"]?.shape, [4])
    XCTAssertEqual(report.failedForm?.name, "u")
    XCTAssertEqual(report.failedForm?.op, "def")
  }
}

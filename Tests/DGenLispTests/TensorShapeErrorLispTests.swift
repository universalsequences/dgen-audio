import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Shape mismatches reachable from user source must be thrown `LispError`s,
/// not Swift traps: a `precondition` in an optimized CLI build exits with
/// SIGTRAP (133) and prints nothing, which a host cannot report.
///
/// Shape semantics pinned here:
/// - Elementwise ops broadcast NumPy-style, so `[8 1]` with `[8]` is `[8 8]`.
/// - A tensor `biquad` requires every tensor operand to have the filter's exact
///   shape (scalars are lifted); it does not broadcast implicitly.
/// - `write-history` accepts any value with the history's element count (same
///   row-major layout, e.g. `[8]` into `[8 1]`) and rejects everything else.
final class TensorShapeErrorLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-shape-errors-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private func evaluator(_ source: String) throws -> LispEvaluator {
    let e = LispEvaluator(sourceDirectory: tempDir)
    try e.evaluate(nodes: parseSource(source))
    return e
  }

  /// Evaluates `source`, expecting a LispError whose message contains every
  /// fragment in `expected`.
  private func assertLispError(
    _ source: String, contains expected: [String], file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertThrowsError(try evaluator(source), file: file, line: line) { error in
      guard let lispError = error as? LispError else {
        return XCTFail("expected LispError, got \(error)", file: file, line: line)
      }
      for fragment in expected {
        XCTAssertTrue(
          lispError.message.contains(fragment),
          "message \"\(lispError.message)\" lacks \"\(fragment)\"", file: file, line: line)
      }
    }
  }

  private func compile(_ e: LispEvaluator, name: String) throws -> CompilerResult {
    try compilePatch(
      graph: LazyGraphContext.current,
      outputs: e.outputs,
      options: CompilerOptions(
        outputDir: tempDir.path, name: name, sampleRate: 48000, maxFrames: 64,
        voiceCount: 1, skipInlineAudit: true, debug: false))
  }

  private func biquadFeedbackSource(historyShape: String) -> String {
    """
    (def p (in 2))
    (def t (* (iota 8) 2))
    (make-history h @shape \(historyShape))
    (def b (biquad (read-history h) (* p t) 0.777 0 1))
    (def c (cos (+ b (* (phasor (* p t)) twopi))))
    (write-history h c)
    (out (sum c) 1)
    """
  }

  // MARK: - biquad

  /// The user repro: an [8 1] history fed through biquad with an [8] cutoff
  /// used to trap in DGenLazy's per-element biquad precondition.
  func testBiquadControlShapeMismatchIsAnErrorNotATrap() {
    assertLispError(
      biquadFeedbackSource(historyShape: "[8 1]"),
      contains: ["biquad", "cutoff", "[8]", "[8, 1]", "[8, 8]"])
  }

  func testBiquadFeedbackWithMatchingRank1HistoryCompiles() throws {
    let e = try evaluator(biquadFeedbackSource(historyShape: "[8]"))
    let result = try compile(e, name: "biquad-feedback-8")
    XCTAssertTrue(FileManager.default.fileExists(atPath: result.dylibPath))
  }

  func testBiquadScalarControlsLiftToTensorShape() throws {
    let e = try evaluator(
      """
      (make-history h @shape [8 1])
      (def b (biquad (read-history h) 440 0.7 0 0))
      (write-history h b)
      """)
    guard case .signalTensor(let b)? = e.definitions["b"] else {
      return XCTFail("expected signalTensor biquad output")
    }
    XCTAssertEqual(b.shape, [8, 1])
  }

  // MARK: - write-history

  /// Without biquad, [8 1] + [8] broadcasts to [8 8] (intended NumPy
  /// semantics); writing that into an [8 1] history used to overrun silently.
  func testWriteHistoryRejectsBroadcastGrownValue() {
    assertLispError(
      """
      (def t (* (iota 8) 2))
      (make-history h @shape [8 1])
      (def c (cos (+ (read-history h) (phasor t))))
      (write-history h c)
      (out (sum c) 1)
      """,
      contains: ["write-history", "[8, 8]", "[8, 1]", "64", "8 elements"])
  }

  func testWriteTensorHistoryRejectsElementCountMismatch() {
    assertLispError(
      """
      (make-tensor-history h @shape [4])
      (write-tensor-history h (tensor @shape [2] @data [1 2]))
      """,
      contains: ["write-history", "[2]", "[4]"])
  }

  func testWriteHistoryAcceptsSameElementCountReshape() throws {
    // [8] into [8 1]: identical row-major layout, kept valid.
    _ = try evaluator(
      """
      (make-history h @shape [8 1])
      (write-history h (phasor (* (iota 8) 2)))
      """)
  }

  func testExistingTensorHistoryFeedbackStillCompiles() throws {
    let e = try evaluator(
      """
      (def lap (tensor @shape [3 3] @data [0 1 0  1 -4 1  0 1 0]))
      (make-tensor-history p @shape [4 4])
      (def prev (read-tensor-history p))
      (def next (+ prev (* 0.1 (conv2d prev lap @padding same))))
      (write-tensor-history p next)
      (out (sum next) 1)
      """)
    let result = try compile(e, name: "tensor-history-plate")
    XCTAssertTrue(FileManager.default.fileExists(atPath: result.dylibPath))
  }

  // MARK: - Other former traps

  func testFormerFatalErrorsAreLispErrors() {
    assertLispError(
      """
      (def a (tensor @shape [4] @data [1 2 3 4]))
      (def k (tensor @shape [2 2] @data [1 0 0 1]))
      (def y (conv2d a k))
      """,
      contains: ["conv2d", "[4]"])
    assertLispError(
      """
      (def a (tensor @shape [2 2] @data [1 2 3 4]))
      (def y (conv1d a a))
      """,
      contains: ["conv1d", "[2, 2]"])
    assertLispError(
      """
      (def a (tensor @shape [4] @data [1 2 3 4]))
      (def y (sum-axis a @axis 3))
      """,
      contains: ["sum-axis", "@axis 3"])
    assertLispError(
      """
      (def a (tensor @shape [4] @data [1 2 3 4]))
      (def y (max-axis a @axis 0))
      """,
      contains: ["max-axis", "1-D"])
    assertLispError("(def w (hann 0))", contains: ["hann", "positive"])
  }
}

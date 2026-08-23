import DGenLazy
import XCTest

@testable import DGen
@testable import DGenLisp

/// Confirms the `cumsum` op is reachable from dgenlisp on static and per-frame
/// tensors, with the optional `@axis` attribute (default: last axis).
final class CumsumLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 8
    DGenConfig.maxFrameCount = 32
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-cumsum-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    DGenConfig.maxFrameCount = 4096
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private func evaluator(_ source: String) throws -> LispEvaluator {
    let e = LispEvaluator(sourceDirectory: tempDir)
    try e.evaluate(nodes: parseSource(source))
    return e
  }

  func testCumsumStaticTensorShapeAndValues() throws {
    let e = try evaluator(
      """
      (def x (tensor @shape [5] @data [1 2 3 4 5]))
      (def y (cumsum x))
      """)
    guard case .tensor(let y)? = e.definitions["y"] else { return XCTFail("expected tensor y") }
    XCTAssertEqual(y.shape, [5])
    XCTAssertEqual(try y.realize(), [1, 3, 6, 10, 15])
  }

  func testCumsumAxisAttributeOnMatrix() throws {
    let e = try evaluator(
      """
      (def x (tensor @shape [2 3] @data [1 2 3 4 5 6]))
      (def y (cumsum x @axis 0))
      """)
    guard case .tensor(let y)? = e.definitions["y"] else { return XCTFail("expected tensor y") }
    XCTAssertEqual(y.shape, [2, 3])
    XCTAssertEqual(try y.realize(), [1, 2, 3, 5, 7, 9])
  }

  func testCumsumOnSignalTensor() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def y (cumsum frame))
      """)
    guard case .signalTensor(let y)? = e.definitions["y"] else {
      return XCTFail("expected signalTensor y")
    }
    XCTAssertEqual(y.shape, [4])
  }
}

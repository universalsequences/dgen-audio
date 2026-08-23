import DGenLazy
import XCTest

@testable import DGenLisp

final class IotaLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.maxFrameCount = 32
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-iota-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    DGenConfig.maxFrameCount = 4096
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private func evaluator(_ source: String) throws -> LispEvaluator {
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    return evaluator
  }

  func testIotaProducesIndexRamp() throws {
    let e = try evaluator("(def idx (iota 5))")
    guard case .tensor(let idx)? = e.definitions["idx"] else {
      return XCTFail("expected tensor idx")
    }
    XCTAssertEqual(idx.shape, [5])
    XCTAssertEqual(try idx.realize(), [0, 1, 2, 3, 4])
  }

  func testIotaWithStartAndStep() throws {
    let e = try evaluator("(def bins (iota 4 10 0.5))")
    guard case .tensor(let bins)? = e.definitions["bins"] else {
      return XCTFail("expected tensor bins")
    }
    XCTAssertEqual(try bins.realize(), [10, 10.5, 11, 11.5])
  }

  /// The motivating use: enumerate positions, scale them, and read a table.
  func testIotaDrivesGather() throws {
    let e = try evaluator(
      """
      (def table (tensor @shape [6] @data [0 10 20 30 40 50]))
      (def out (gather table (* (iota 3) 2)))
      """)
    guard case .tensor(let out)? = e.definitions["out"] else {
      return XCTFail("expected tensor out")
    }
    XCTAssertEqual(try out.realize(), [0, 20, 40])
  }

  func testIotaRejectsEmptyCount() throws {
    XCTAssertThrowsError(try evaluator("(def bad (iota 0))"))
  }
}

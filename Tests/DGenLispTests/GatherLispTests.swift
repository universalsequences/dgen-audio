import DGenLazy
import XCTest

@testable import DGenLisp

final class GatherLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.maxFrameCount = 32
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-gather-\(UUID().uuidString)", isDirectory: true)
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

  func testStaticTensorGather() throws {
    let e = try evaluator(
      """
      (def source (tensor @shape [5] @data [10 20 30 40 50]))
      (def idx (tensor @shape [4] @data [4 4 2 0]))
      (def out (gather source idx))
      """)
    guard case .tensor(let out)? = e.definitions["out"] else {
      return XCTFail("expected tensor out")
    }
    XCTAssertEqual(out.shape, [4])
    XCTAssertEqual(try out.realize(), [50, 50, 30, 10])
  }

  func testSignalTensorGather() throws {
    let e = try evaluator(
      """
      (def s (+ 2 (phasor 0)))
      (def source (* (tensor @shape [5] @data [10 20 30 40 50]) s))
      (def idx (tensor @shape [2] @data [3 1]))
      (def picked (gather source idx))
      (def out (sum picked))
      """)
    guard case .signal(let out)? = e.definitions["out"] else {
      return XCTFail("expected signal out")
    }
    XCTAssertEqual(try out.realize(frames: 4), [120, 120, 120, 120])
  }

  func testSignalTensorComparisonsForSpectralMasks() throws {
    let e = try evaluator(
      """
      (def s (+ 2 (phasor 0)))
      (def bins (* (tensor @shape [3] @data [1 2 3]) s))
      (def threshold (+ 3 (phasor 0)))
      (def dynamic-mask (gt bins threshold))
      (def scalar-mask (gt bins 3.5))
      (def static-mask (gte bins (tensor @shape [3] @data [2 5 5])))
      (def out (+ (sum dynamic-mask) (sum scalar-mask) (sum static-mask)))
      """)
    guard case .signal(let out)? = e.definitions["out"] else {
      return XCTFail("expected signal out")
    }
    XCTAssertEqual(try out.realize(frames: 4), [6, 6, 6, 6])
  }

  func testSignalTensorLatchSyntax() throws {
    let e = try evaluator(
      """
      (def ramp (accum 1 0 0 100))
      (def values (+ (tensor @shape [3] @data [1 2 3]) ramp))
      (def trigger (eq (% ramp 3) 0))
      (def held (latch values trigger))
      (def out (sum held))
      """)
    guard case .signal(let out)? = e.definitions["out"] else {
      return XCTFail("expected signal out")
    }
    XCTAssertEqual(try out.realize(frames: 8), [6, 6, 6, 15, 15, 15, 24, 24])
  }
}

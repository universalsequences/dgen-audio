import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Evaluator-level tests for tensor-aware `delay`:
/// `(delay <tensor> <time>)` builds per-lane delay lines, `@max-delay`
/// bounds the per-lane buffer, and misuse produces actionable errors.
final class TensorDelayLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-tensor-delay-\(UUID().uuidString)", isDirectory: true)
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

  // MARK: - The multichannel scenario that used to throw

  func testTensorPhasorIntoDelayIsSignalTensor() throws {
    let e = try evaluator(
      """
      (def freqs (tensor @shape [2 2] @data [90 120 53 300]))
      (def ph (phasor freqs))
      (def delayed (delay ph 500))
      """)
    guard case .signalTensor(let delayed)? = e.definitions["delayed"] else {
      return XCTFail("expected signalTensor from (delay <signalTensor> 500)")
    }
    XCTAssertEqual(delayed.shape, [2, 2])
  }

  func testTensorDelayWithPerLaneTimesAndMaxDelay() throws {
    let e = try evaluator(
      """
      (def freqs (tensor @shape [2] @data [90 120]))
      (def times (tensor @shape [2] @data [100 200]))
      (def delayed (delay (phasor freqs) times @max-delay 4800))
      """)
    guard case .signalTensor(let delayed)? = e.definitions["delayed"] else {
      return XCTFail("expected signalTensor from per-lane delay")
    }
    XCTAssertEqual(delayed.shape, [2])
  }

  func testTensorDelayRendersEndToEnd() throws {
    // Compile and run: 2 delayed ramps, summed. Exact expectation mirrors
    // TensorDelayTests: sum over lanes of factor * (n - d) with zero-fill.
    let e = try evaluator(
      """
      (def ramp (accum 1 0 0 1000000))
      (def factors (tensor @shape [2] @data [2 5]))
      (def delayed (delay (* factors ramp) 10 @max-delay 64))
      (def outsig (sum delayed))
      """)
    guard case .signal(let outSig)? = e.definitions["outsig"] else {
      return XCTFail("expected scalar sum of delayed tensor")
    }
    let result = try outSig.realize(frames: 40)
    for n in 0..<result.count {
      let expected: Float = n >= 10 ? Float(n - 10) * (2 + 5) : 0
      XCTAssertEqual(result[n], expected, accuracy: 1e-3, "frame \(n)")
    }
  }

  func testTensorDelayInFeedbackLoopCompiles() throws {
    // Feedback through a tensor history whose write side passes through a
    // per-lane delay: the classic multichannel echo topology.
    let e = try evaluator(
      """
      (make-tensor-history h @shape [2])
      (def inject (tensor @shape [2] @data [0.5 0.25]))
      (def fb (read-tensor-history h))
      (def wet (+ inject (* fb 0.5)))
      (def delayed (delay wet 100 @max-delay 4800))
      (write-tensor-history h delayed)
      (def outsig (sum delayed))
      """)
    guard case .signal(let outSig)? = e.definitions["outsig"] else {
      return XCTFail("expected scalar output from feedback graph")
    }
    let result = try outSig.realize(frames: 64)
    XCTAssertEqual(result.count, 64)
    XCTAssertTrue(result.allSatisfy { $0.isFinite }, "feedback output must stay finite")
  }

  // MARK: - Error quality

  func testScalarInputWithTensorTimeIsActionable() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def times (tensor @shape [2] @data [100 200]))
        (def delayed (delay (phasor 440) times))
        """)
    ) { error in
      let message = "\(error)"
      XCTAssertTrue(
        message.contains("tensor delay time") && message.contains("scalar"),
        "error should explain the scalar-input/tensor-time mismatch, got: \(message)")
    }
  }

  func testMismatchedTimeShapeIsActionable() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def freqs (tensor @shape [2 2] @data [90 120 53 300]))
        (def times (tensor @shape [3] @data [1 2 3]))
        (def delayed (delay (phasor freqs) times))
        """)
    ) { error in
      let message = "\(error)"
      XCTAssertTrue(
        message.contains("[3]") && message.contains("[2, 2]"),
        "error should name both shapes, got: \(message)")
    }
  }

  func testInvalidMaxDelayIsActionable() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def freqs (tensor @shape [2] @data [90 120]))
        (def delayed (delay (phasor freqs) 10 @max-delay 0.5))
        """)
    ) { error in
      let message = "\(error)"
      XCTAssertTrue(
        message.contains("@max-delay"),
        "error should name the @max-delay attribute, got: \(message)")
    }
  }
}

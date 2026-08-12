import DGenLazy
import XCTest

@testable import DGen
@testable import DGenLisp

/// Regression tests for numeric type promotion in the dgenlisp evaluator.
///
/// Two gaps used to surface as `Type error: Expected tensor, got other type`:
///   1. a tensor combined with a scalar signal (a `param`) produced a
///      signalTensor that downstream tensor ops (notably `matmul`) rejected;
///   2. shaping ops such as `triangle` only accepted scalar signals, so
///      `(triangle (phasor <tensor>))` failed even though `cos`/`sin` worked.
final class TypePromotionLispTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-promotion-\(UUID().uuidString)", isDirectory: true)
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

  // MARK: - Bug 1: tensor combined with a scalar param

  func testTensorTimesParamIsSignalTensor() throws {
    let e = try evaluator(
      """
      (param xyz @default 0.5)
      (def t (tensor @shape [2 2] @data [2 1 4 1]))
      (def scaled (* t xyz))
      """)
    guard case .signalTensor(let scaled)? = e.definitions["scaled"] else {
      return XCTFail("expected signalTensor from tensor * param")
    }
    XCTAssertEqual(scaled.shape, [2, 2])
  }

  func testSumOfTensorTimesParamEvaluates() throws {
    let e = try evaluator(
      """
      (param xyz @default 0.5)
      (def total (sum (* (tensor @shape [2 2] @data [2 1 4 1]) xyz)))
      """)
    guard case .signal(let total)? = e.definitions["total"] else {
      return XCTFail("expected scalar signal from sum")
    }
    let frames = try total.realize(frames: 4)
    for value in frames {
      XCTAssertEqual(value, (2 + 1 + 4 + 1) * 0.5, accuracy: 1e-5)
    }
  }

  /// Every arithmetic op — not just `*` — must accept the mixed pair.
  func testAllArithmeticOpsAcceptTensorAndSignal() throws {
    for op in ["+", "-", "*", "/", "min", "max", "%", "pow", "gt", "lt", "gte", "lte", "eq"] {
      LazyGraphContext.reset()
      let e = try evaluator(
        """
        (param xyz @default 0.5)
        (def y (\(op) (tensor @shape [2] @data [2 4]) xyz))
        """)
      guard case .signalTensor(let y)? = e.definitions["y"] else {
        return XCTFail("\(op): expected signalTensor, got \(String(describing: e.definitions["y"]))")
      }
      XCTAssertEqual(y.shape, [2], "\(op) should keep the tensor shape")
    }
  }

  func testMatmulAcceptsSignalTensorOperand() throws {
    let e = try evaluator(
      """
      (param xyz @default 0.5)
      (def a (tensor @shape [2 2] @data [1 2 3 4]))
      (def b (* (tensor @shape [2 2] @data [5 6 7 8]) xyz))
      (def total (sum (matmul a b)))
      """)
    guard case .signal(let total)? = e.definitions["total"] else {
      return XCTFail("expected scalar signal from sum of matmul")
    }
    // [1 2; 3 4] @ ([5 6; 7 8] * 0.5) = [9.5 11; 21.5 25], sum = 67
    let frames = try total.realize(frames: 4)
    for value in frames {
      XCTAssertEqual(value, 67.0, accuracy: 1e-3)
    }
  }

  func testMatmulRejectsShapeMismatchWithActionableMessage() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (param xyz @default 0.5)
        (def y (matmul (tensor @shape [2 3] @data [1 2 3 4 5 6])
                       (* (tensor @shape [2 2] @data [1 2 3 4]) xyz)))
        """)
    ) { error in
      XCTAssertTrue(
        "\(error)".contains("dimension mismatch"), "unexpected error: \(error)")
    }
  }

  // MARK: - Bug 2: unary/shaping ops over signalTensors

  func testTriangleOverSignalTensor() throws {
    let e = try evaluator(
      """
      (def phase (phasor (tensor @shape [2] @data [12000 24000])))
      (def tri (triangle phase))
      """)
    guard case .signalTensor(let tri)? = e.definitions["tri"] else {
      return XCTFail("expected signalTensor from triangle over a signalTensor")
    }
    XCTAssertEqual(tri.shape, [2])

    // Phases advance by f/sr per frame: 0.25 and 0.5 respectively.
    // triangle(p) = p/0.5 while p < 0.5, else (1-p)/0.5.
    let data = try tri.realize(frames: 4)
    XCTAssertEqual(data.count, 8)
    func triangleModel(_ p: Float) -> Float { p < 0.5 ? p / 0.5 : (1 - p) / 0.5 }
    var phases: [Float] = [0, 0]
    let increments: [Float] = [0.25, 0.5]
    for frame in 0..<4 {
      for element in 0..<2 {
        XCTAssertEqual(
          data[frame * 2 + element], triangleModel(phases[element]), accuracy: 1e-4,
          "frame \(frame) element \(element)")
        phases[element] += increments[element]
        if phases[element] >= 1 { phases[element] -= 1 }
      }
    }
  }

  func testTriangleWithSignalDutyOverSignalTensor() throws {
    let e = try evaluator(
      """
      (param duty @default 0.25)
      (def tri (triangle (phasor (tensor @shape [2] @data [90 91])) duty))
      """)
    guard case .signalTensor(let tri)? = e.definitions["tri"] else {
      return XCTFail("expected signalTensor from triangle with a scalar duty")
    }
    XCTAssertEqual(tri.shape, [2])
  }

  /// The elementwise unary family must all accept signalTensors.
  func testUnaryMathFamilyAcceptsSignalTensor() throws {
    let fns = [
      "sin", "cos", "tan", "atan", "tanh", "exp", "log", "log10", "sqrt", "abs", "sign",
      "floor", "ceil", "round", "relu", "sigmoid",
    ]
    for fn in fns {
      LazyGraphContext.reset()
      let e = try evaluator(
        """
        (def y (\(fn) (phasor (tensor @shape [2] @data [90 91]))))
        """)
      guard case .signalTensor(let y)? = e.definitions["y"] else {
        return XCTFail("\(fn): expected signalTensor")
      }
      XCTAssertEqual(y.shape, [2], "\(fn) should keep the tensor shape")
    }
  }

  /// `scale`, `wrap` and `clip` share the shaping-op dispatch.
  func testShapingOpsAcceptSignalTensor() throws {
    let e = try evaluator(
      """
      (def phase (phasor (tensor @shape [2] @data [90 91])))
      (def scaled (scale phase 0 1 -1 1))
      (def wrapped (wrap phase 0 0.5))
      (def clipped (clip phase 0.25 0.75))
      """)
    for name in ["scaled", "wrapped", "clipped"] {
      guard case .signalTensor(let value)? = e.definitions[name] else {
        return XCTFail("\(name): expected signalTensor")
      }
      XCTAssertEqual(value.shape, [2], "\(name) should keep the tensor shape")
    }
  }

  func testGswitchAndMixAcceptSignalTensor() throws {
    let e = try evaluator(
      """
      (param xyz @default 0.5)
      (def phase (phasor (tensor @shape [2] @data [90 91])))
      (def picked (gswitch (gt phase 0.5) phase xyz))
      (def blended (mix phase (* (tensor @shape [2] @data [1 2]) xyz) 0.25))
      """)
    for name in ["picked", "blended"] {
      guard case .signalTensor(let value)? = e.definitions[name] else {
        return XCTFail("\(name): expected signalTensor")
      }
      XCTAssertEqual(value.shape, [2], "\(name) should keep the tensor shape")
    }
  }

  /// Scalar behaviour must be unchanged by the promotion rework.
  func testScalarPathsUnchanged() throws {
    let e = try evaluator(
      """
      (def a (scale 0.25 0 1 0 10))
      (def b (clip 5.0 0 1))
      (def c (min 3 7))
      (def d (pow 2 10))
      (def f (% 7 4))
      (def g (triangle 0.25))
      """)
    guard case .float(let a)? = e.definitions["a"] else { return XCTFail("a") }
    XCTAssertEqual(a, 2.5, accuracy: 1e-6)
    guard case .float(let b)? = e.definitions["b"] else { return XCTFail("b") }
    XCTAssertEqual(b, 1.0, accuracy: 1e-6)
    guard case .float(let c)? = e.definitions["c"] else { return XCTFail("c") }
    XCTAssertEqual(c, 3.0, accuracy: 1e-6)
    guard case .float(let d)? = e.definitions["d"] else { return XCTFail("d") }
    XCTAssertEqual(d, 1024.0, accuracy: 1e-6)
    guard case .float(let f)? = e.definitions["f"] else { return XCTFail("f") }
    XCTAssertEqual(f, 3.0, accuracy: 1e-6)
    guard case .signal? = e.definitions["g"] else { return XCTFail("g should stay a signal") }
  }
}

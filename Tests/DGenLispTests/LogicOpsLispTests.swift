import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// `and`/`or`/`xor`/`not` treat any nonzero input as true and yield 1.0/0.0,
/// matching the comparison operators, in every numeric domain.
final class LogicOpsLispTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 128
    LazyGraphContext.reset()
  }

  /// Signal domain: a and b step through the whole truth table, using
  /// non-unit truthy values to prove nonzero (not == 1) is the test.
  func testSignalTruthTable() throws {
    let output = try render(
      """
      (def x (accum 1 0 0 4))
      (def a (* 3 (gte x 2)))
      (def b (* -0.5 (% x 2)))
      (out (+ (and a b) (* 2 (or a b)) (* 4 (xor a b)) (* 8 (not a))) 1)
      """, frames: 8)
    // (a, b) = (0,0) (0,1) (1,0) (1,1), repeating.
    let expected: [Float] = [8, 14, 6, 3, 8, 14, 6, 3]
    XCTAssertEqual(output, expected)
  }

  func testNaryAndOrXorNestLeftAssociatively() throws {
    let output = try render(
      """
      (def x (accum 1 0 0 2))
      (out (+ (and 1 x 2) (* 2 (or 0 0 x)) (* 4 (xor 1 1 x))) 1)
      """, frames: 2)
    // x = 0: and 0, or 0, xor(xor(1,1),0) = 0. x = 1: 1, 1, 1.
    XCTAssertEqual(output, [0, 7])
  }

  func testFloatOperandsConstantFold() throws {
    let evaluator = LispEvaluator()
    for (source, expected) in [
      ("(and 2 -1)", 1), ("(and 2 0)", 0), ("(or 0 0)", 0), ("(or 0 0.25)", 1),
      ("(xor 5 0)", 1), ("(xor 5 5)", 0), ("(not 0)", 1), ("(not -3)", 0),
    ] as [(String, Float)] {
      guard case .float(let value) = try evaluator.evaluateAST(parseSource(source)[0]) else {
        XCTFail("\(source) did not fold to a float")
        continue
      }
      XCTAssertEqual(value, expected, source)
    }
  }

  func testTensorAndSignalTensorDomains() throws {
    let output = try render(
      """
      (def b (accum 1 0 0 2))
      (def t (tensor @shape [4] @data [0 1 2 0]))
      (def w (tensor @shape [4] @data [1 2 4 8]))
      (def last (tensor @shape [4] @data [0 0 0 1]))
      (def gain (+ b 1))
      (out (+ (sum (* (xor t b) w))
              (* 100 (sum (* (and t 0.5) w gain)))
              (* 1000 (sum (* (not t) w gain)))
              (* 10000 (sum (* (or t last) w gain)))) 1)
      """, frames: 2)
    // A pure tensor cannot reach `out`, so the tensor-domain results are
    // scaled by the frame-varying `gain` (1 then 2) after they are formed.
    // t truthy = [0 1 1 0]. Signal x tensor: xor -> [0 1 1 0] = 6 at b = 0,
    // [1 0 0 1] = 9 at b = 1. Tensor x float: (and t 0.5) = 6. Unary over a
    // tensor: (not t) = [1 0 0 1] = 9. Tensor x tensor: (or t last) = 14.
    XCTAssertEqual(output, [149_606, 299_209])
  }

  private func render(_ source: String, frames: Int) throws -> [Float] {
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))
    return try XCTUnwrap(evaluator.outputs.first?.signal).realize(frames: frames)
  }
}

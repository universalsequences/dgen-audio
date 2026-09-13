import XCTest

@testable import DGen

final class CIntegerIndexTests: XCTestCase {
  private func render(_ ops: [UOp]) -> String {
    let context = IRContext(g: Graph())
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = ops
    return CRenderer().compile(scheduleItems: [item], ctx: context,
      graph: context.g, totalMemorySlots: 1024)[0].source
  }

  func testTensorLoopCounterDoesNotUseFloatingFiniteGuard() {
    let index = Lazy.variable(1, nil)
    let source = render([
      UOp(op: .beginForLoop(index, .constant(0, 16)), value: .empty),
      UOp(op: .memoryRead(32, index), value: .variable(2, nil)),
      UOp(op: .endLoop, value: .empty),
    ])
    XCTAssertTrue(source.contains("for (int t1 = 0; t1 < 16; t1++)"))
    XCTAssertTrue(source.contains("memory[32 + t1]"))
    XCTAssertFalse(source.contains("isfinite(t1)"))
  }

  func testIntegerTapeIndexRetainsBoundsWithoutFloatingFiniteGuard() {
    let index = Lazy.variable(1, nil)
    let source = render([
      UOp(op: .frameIndex, value: index, scalarType: .int),
      UOp(op: .loadTape(.variable(2, nil), index), value: .variable(3, nil)),
    ])
    XCTAssertTrue(source.contains("int t1 = i;"))
    XCTAssertTrue(source.contains("t1 < 0 || t1 >= frameCount"))
    XCTAssertFalse(source.contains("isfinite(t1)"))
  }

  func testFloatingIndicesStillUseFiniteGuardsAndTapeBounds() {
    let index = Lazy.variable(1, nil)
    let source = render([
      UOp(op: .input(0), value: index),
      UOp(op: .memoryRead(32, index), value: .variable(2, nil)),
      UOp(op: .loadTape(.variable(3, nil), index), value: .variable(4, nil)),
    ])
    XCTAssertTrue(source.contains("isfinite(t1) ? (int) t1 : 0"))
    XCTAssertTrue(source.contains("isfinite(t1) ? (int)t1 : 0"))
    XCTAssertTrue(source.contains(">= frameCount"))
  }

  func testScalarMathUsesBuiltinsUnderFreestandingCompilation() {
    let input = Lazy.variable(1, nil)
    let source = render([
      UOp(op: .input(0), value: input),
      UOp(op: .floor(input), value: .variable(2, nil)),
      UOp(op: .ceil(input), value: .variable(3, nil)),
      UOp(op: .round(input), value: .variable(4, nil)),
      UOp(op: .min(input, .constant(0, 1)), value: .variable(5, nil)),
      UOp(op: .max(input, .constant(0, 0)), value: .variable(6, nil)),
      UOp(op: .abs(input), value: .variable(7, nil)),
      UOp(op: .sqrt(input), value: .variable(8, nil)),
      UOp(op: .pow(input, .constant(0, 0.5)), value: .variable(9, nil)),
    ])
    for name in ["floorf", "ceilf", "roundf", "fminf", "fmaxf", "fabsf", "sqrtf"] {
      XCTAssertTrue(source.contains("__builtin_\(name)("), name)
    }
    XCTAssertTrue(source.contains("t9 = __builtin_sqrtf(t1)"))
  }
}

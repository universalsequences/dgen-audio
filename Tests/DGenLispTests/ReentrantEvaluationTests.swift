import DGenLazy
import XCTest

@testable import DGenLisp

final class ReentrantEvaluationTests: XCTestCase {
  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  func testNamedParamSurvivesWhileComputedGraphIsRebuilt() throws {
    let nodes = try parseSource("""
      (def gain (param gain @default 0.5 @min 0 @max 1))
      (def voice (+ gain 0.25))
      (out voice 1)
      """)
    let evaluator = LispEvaluator(reusesRegisteredParameters: true)

    try evaluator.evaluate(nodes: nodes)
    guard case .signal(let firstParam)? = evaluator.definitions["gain"],
      case .signal(let firstVoice)? = evaluator.definitions["voice"]
    else {
      return XCTFail("expected parameter and voice definitions")
    }
    let firstCell = firstParam.memoryCellId
    let firstShape = (
      LazyGraphContext.current.debugNodeCount,
      LazyGraphContext.current.debugMemoryCellCount)
    firstParam.updateDataLazily(0.75)

    LazyGraphContext.current.clearComputationGraph()
    try evaluator.evaluate(nodes: nodes)
    guard case .signal(let secondParam)? = evaluator.definitions["gain"],
      case .signal(let secondVoice)? = evaluator.definitions["voice"]
    else {
      return XCTFail("expected rebuilt definitions")
    }

    XCTAssertTrue(firstParam === secondParam)
    XCTAssertFalse(firstVoice === secondVoice)
    XCTAssertEqual(secondParam.data, 0.75)
    XCTAssertEqual(secondParam.memoryCellId, firstCell)
    XCTAssertEqual(LazyGraphContext.current.debugNodeCount, firstShape.0)
    XCTAssertEqual(LazyGraphContext.current.debugMemoryCellCount, firstShape.1)
  }
}

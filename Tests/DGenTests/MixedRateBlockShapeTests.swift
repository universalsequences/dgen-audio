import XCTest

@testable import DGen

final class MixedRateBlockShapeTests: XCTestCase {
  func testRateSplitRemovesInheritedTensorLoopFromScalarFeedback() {
    let graph = Graph()
    let counter = graph.n(.constant(0))
    let input = graph.n(.input(0))
    let coefficients = graph.tensor(shape: [64], data: Array(repeating: 1, count: 64))
    let smaller = graph.tensor(shape: [32], data: Array(repeating: 1, count: 32))
    let hopTensor = graph.n(.mul, coefficients, input)
    let cell = graph.alloc()
    let history = graph.n(.historyRead(cell))
    let value = graph.n(.add, history, input)
    let write = graph.n(.historyWrite(cell), value)
    let nextTensor = graph.n(.mul, smaller, input)
    let context = IRContext(g: graph)
    var block = Block(frameOrder: .sequential)
    block.nodes = [hopTensor, history, value, write, nextTensor]
    block.shape = [64]
    block.tensorIndex = context.useVariable(src: nil)
    block.executionFrameGroup = 7
    var blocks = [block]
    XCTAssertTrue(TemporalityPass.splitMixedRateBlocks(
      blocks: &blocks, context: context,
      frameBasedNodes: [history, value, write],
      hopBasedNodes: [hopTensor: (16, counter), nextTensor: (16, counter)]))
    XCTAssertEqual(blocks.map(\.nodes), [[hopTensor], [history, value, write], [nextTensor]])
    XCTAssertEqual(blocks[0].shape, [64])
    XCTAssertNil(blocks[1].shape, "scalar feedback must not inherit 64 lane iterations")
    XCTAssertNil(blocks[1].tensorIndex)
    XCTAssertEqual(blocks[2].shape, [32])
    XCTAssertNotNil(blocks[0].tensorIndex)
    XCTAssertNotNil(blocks[2].tensorIndex)
    XCTAssertTrue(blocks.allSatisfy { $0.frameOrder == .sequential && $0.executionFrameGroup == 7 })
  }
}

import XCTest

@testable import DGen

final class MixedRateBlockShapeTests: XCTestCase {
  func testTensorGroupingKeepsScalarHistoryInOneFrameLoop() throws {
    let graph = Graph()
    let input = graph.n(.input(0))
    let tick = graph.n(.accum(graph.alloc()), graph.n(.constant(1)), graph.n(.constant(0)))
    let table = graph.tensor(shape: [8], data: Array(repeating: 0.5, count: 8))
    let cell = graph.alloc()
    let previous = graph.n(.historyRead(cell))
    let coefficients = graph.n(.mul, table, input)
    let selected = try graph.peek(tensor: coefficients, index: graph.n(.constant(0)), channel: graph.n(.constant(0)))
    let value = graph.n(.add, previous, selected)
    let write = graph.n(.historyWrite(cell), value)
    var block = Block(frameOrder: .sequential)
    block.nodes = [tick, previous, coefficients, selected, value, write]
    let parts = determineTensorBlocks([block], graph, IRContext(g: graph))
    let reader = try XCTUnwrap(parts.firstIndex { $0.nodes.contains(previous) })
    let writer = try XCTUnwrap(parts.firstIndex { $0.nodes.contains(write) })
    XCTAssertEqual(reader, writer, "tensor grouping must not make history block-stale")
    XCTAssertEqual(parts[reader].frameOrder, .sequential)
  }

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
    block.sequentialFrameGroup = 7
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
    XCTAssertTrue(blocks.allSatisfy { $0.frameOrder == .sequential && $0.sequentialFrameGroup == 7 })
  }
  func testBufferReuseKeepsEveryFeedbackFragmentLiveForWholeSampleLoop() throws {
    let zero = Lazy.constant(0, 0)
    let one = Lazy.constant(1, 1)
    var blocks = [
      BlockUOps(ops: [UOp(op: .memoryWrite(10, zero, one), value: .empty)],
        frameOrder: .sequential, vectorWidth: 1, temporality: .static_, dispatchMode: .singleThreaded),
      BlockUOps(ops: [UOp(op: .memoryRead(10, zero), value: .variable(2, nil))],
        frameOrder: .sequential, vectorWidth: 1, temporality: .frameBased, dispatchMode: .singleThreaded),
      BlockUOps(ops: [UOp(op: .memoryWrite(20, zero, one), value: .empty)],
        frameOrder: .sequential, vectorWidth: 1,
        temporality: .hopBased(hopSize: 16, counterNode: 100), dispatchMode: .singleThreaded),
      BlockUOps(ops: [UOp(op: .memoryRead(20, zero), value: .variable(3, nil))],
        frameOrder: .sequential, vectorWidth: 1, temporality: .frameBased, dispatchMode: .singleThreaded),
    ]
    for index in 1..<blocks.count { blocks[index].sequentialFrameGroup = 7 }
    let allocations = remapVectorMemorySlots(&blocks, cellSizes: [10: 64, 20: 32],
      voiceCellId: nil, enableBufferReuse: true)
    let first = try XCTUnwrap(allocations.cellMappings[10])
    let second = try XCTUnwrap(allocations.cellMappings[20])
    XCTAssertTrue(first + 64 <= second || second + 32 <= first,
      "a later fragment's write must not corrupt the preceding fragment's next sample")
  }

}

import XCTest

@testable import DGen

/// Predicate tests for lane-parallel tensor-history dispatch
/// (docs/TENSOR_BIQUAD_PARALLEL_LANES_SPEC.md, test 6): blocks containing a
/// shared-state scalar op (noise's single xorshift PRNG) or a hop-gated tensor
/// history must NOT lane-parallelize, while the clean per-lane biquad
/// recurrence must.
final class StatefulTensorParallelPolicyTests: XCTestCase {
  private let W = 8

  /// Builds a graph + block mimicking the batched-biquad forward recurrence:
  /// tensor historyRead → elementwise math → tensor historyWrite.
  private func makeForwardHistoryBlock() -> (Graph, Block) {
    let g = Graph()
    let shape: ValueShape = .tensor([W])
    g.nodes[0] = Node(id: 0, op: .historyRead(100), inputs: [], shape: shape)
    g.nodes[1] = Node(id: 1, op: .mul, inputs: [0, 0], shape: shape)
    g.nodes[2] = Node(id: 2, op: .add, inputs: [1, 0], shape: shape)
    g.nodes[3] = Node(id: 3, op: .historyWrite(100), inputs: [2], shape: shape)
    g.cellToTensor[100] = 0

    var block = Block(frameOrder: .sequential)
    block.nodes = [0, 1, 2, 3]
    block.shape = [W]
    block.temporality = .frameBased
    block.tensorIndex = .variable(999, nil)
    return (g, block)
  }

  func testTensorHistoryBlockLaneParallelizes() {
    let (g, block) = makeForwardHistoryBlock()
    let decision = StatefulTensorParallelPolicy.decide(block: block, graph: g, backend: .metal)
    XCTAssertTrue(decision.enabled, "clean per-lane tensor-history recurrence should enable")
    XCTAssertEqual(decision.tensorSize, W)
  }

  func testCBackendNeverLaneParallelizes() {
    let (g, block) = makeForwardHistoryBlock()
    let decision = StatefulTensorParallelPolicy.decide(block: block, graph: g, backend: .c)
    XCTAssertFalse(decision.enabled)
  }

  func testNoiseDisqualifiesLaneParallel() {
    let (g, block) = makeForwardHistoryBlock()
    var b = block
    // Shared single-cell xorshift PRNG: duplicating it per thread would change
    // the stream and race the state cell.
    g.nodes[4] = Node(id: 4, op: .noise(400), inputs: [], shape: .scalar)
    b.nodes.append(4)
    let decision = StatefulTensorParallelPolicy.decide(block: b, graph: g, backend: .metal)
    XCTAssertFalse(decision.enabled, "noise in the block must force the strict sequential path")
  }

  func testHopGatedTensorHistoryDisqualifiesLaneParallel() {
    let (g, block) = makeForwardHistoryBlock()
    // Hop-gated feedback keeps the strict frame-by-frame path.
    g.nodeHopRate[0] = (64, 0)
    let decision = StatefulTensorParallelPolicy.decide(block: block, graph: g, backend: .metal)
    XCTAssertFalse(decision.enabled, "hop-gated tensor history must not lane-parallelize")
  }

  func testScalarStatefulOpDisqualifiesLaneParallel() {
    let (g, block) = makeForwardHistoryBlock()
    var b = block
    g.nodes[4] = Node(id: 4, op: .accum(200), inputs: [], shape: .scalar)
    b.nodes.append(4)
    let decision = StatefulTensorParallelPolicy.decide(block: b, graph: g, backend: .metal)
    XCTAssertFalse(decision.enabled, "scalar single-cell accum must force the strict path")
  }

  func testScalarHistoryCellDisqualifiesLaneParallel() {
    let (g, block) = makeForwardHistoryBlock()
    var b = block
    // History op whose cell is NOT tensor-registered: single-cell state.
    g.nodes[4] = Node(id: 4, op: .historyRead(300), inputs: [], shape: .scalar)
    b.nodes.append(4)
    let decision = StatefulTensorParallelPolicy.decide(block: b, graph: g, backend: .metal)
    XCTAssertFalse(decision.enabled, "scalar history cell must force the strict path")
  }

  // MARK: - Detached BPTT backward decision

  /// Builds a graph + block mimicking the consolidated tensor BPTT backward
  /// recurrence: carry reads, grad math, an interleaved pure scalar node
  /// (creating the multi-region shape-aware layout), and carry writes.
  private func makeDetachedBackwardBlock() -> (Graph, Block) {
    let g = Graph()
    g.lastForwardNodeId = 9
    g.gradCarryCells[100] = 500
    g.tensorGradCarryCells.insert(500)

    let shape: ValueShape = .tensor([W])
    g.nodes[10] = Node(id: 10, op: .memoryRead(500), inputs: [], shape: shape)
    g.nodes[11] = Node(id: 11, op: .mul, inputs: [10, 10], shape: shape)
    // Pure scalar node consuming a tensor: forces a [1] region boundary,
    // giving the block the multi-transition layout the real consolidated
    // backward block has.
    g.nodes[12] = Node(id: 12, op: .seq, inputs: [11], shape: .scalar)
    g.nodes[13] = Node(id: 13, op: .memoryWrite(500), inputs: [12, 11], shape: shape)

    var block = Block(frameOrder: .sequential)
    block.nodes = [10, 11, 12, 13]
    block.shape = [W]
    block.temporality = .frameBased
    block.tensorIndex = .variable(998, nil)
    return (g, block)
  }

  func testDetachedBPTTBackwardLaneParallelizes() {
    let (g, block) = makeDetachedBackwardBlock()
    let decision = StatefulTensorParallelPolicy.decideDetachedBPTTBackward(
      block: block, graph: g, backend: .metal)
    XCTAssertTrue(decision.enabled, "per-lane carry recurrence should lane-parallelize")
    XCTAssertEqual(decision.tensorSize, W)
  }

  func testDetachedBPTTBackwardScalarWriteDisqualifies() {
    let (g, block) = makeDetachedBackwardBlock()
    var b = block
    // Scalar (non-carry, non-tensor) memory write: cannot be classified as
    // element-indexed → must fall back to single-threaded.
    g.nodes[14] = Node(id: 14, op: .memoryWrite(700), inputs: [12], shape: .scalar)
    b.nodes.append(14)
    let decision = StatefulTensorParallelPolicy.decideDetachedBPTTBackward(
      block: b, graph: g, backend: .metal)
    XCTAssertFalse(decision.enabled, "unclassifiable scalar write must force single-threaded")
  }

  func testForwardBlockIsNotDetachedBPTTBackward() {
    let (g, block) = makeForwardHistoryBlock()
    let decision = StatefulTensorParallelPolicy.decideDetachedBPTTBackward(
      block: block, graph: g, backend: .metal)
    XCTAssertFalse(decision.enabled)
  }
}

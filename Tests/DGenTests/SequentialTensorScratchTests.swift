import XCTest

@testable import DGen

final class SequentialTensorScratchTests: XCTestCase {
  private func allocation(grouped: Bool = true, escapes: Bool = false,
    materialized: Bool = false, materializedView: Bool = false,
    hop: Bool = false, training: Bool = false,
    backend: Backend = .c) throws -> (slots: Int, frameAware: Bool) {
    let g = Graph()
    let input = g.n(.input(0))
    let weights = g.tensor(shape: [4], data: [1, 2, 3, 4])
    let value = g.n(.mul, input, weights)
    let reduced = g.n(.sum, value)
    var nodes = [input, weights, value, reduced]
    if materializedView {
      let view = g.n(.reshape([2, 2]), value)
      nodes.append(view)
      g.materializeNodes.insert(view)
    }
    var producer = Block(frameOrder: .sequential)
    producer.nodes = [value]
    producer.shape = [4]
    producer.temporality = .frameBased
    producer.sequentialFrameGroup = grouped ? 0 : nil
    var consumer = Block(frameOrder: .sequential)
    consumer.nodes = [reduced]
    consumer.temporality = .frameBased
    consumer.sequentialFrameGroup = grouped ? 0 : nil
    var blocks = [producer, consumer]
    if escapes {
      let later = g.n(.neg, value)
      nodes.append(later)
      var outside = Block(frameOrder: .parallel)
      outside.nodes = [later]
      outside.shape = [4]
      outside.temporality = .frameBased
      blocks.append(outside)
    }
    try inferShapes(graph: g, sortedNodes: nodes)
    TensorOutputBindingPass.bindTensorOutputsAndReserveLazyCells(graph: g, sortedNodes: nodes)
    if materialized { g.materializeNodes.insert(value) }
    if training { g.lastForwardNodeId = reduced }
    TensorMemoryMaterializationPass.allocateTensorMemory(graph: g, blocks: blocks,
      frameBasedNodes: Set(nodes), hopBasedNodes: hop ? [value: (16, input)] : [:],
      backend: backend, frameCount: 128)
    let tensor = try g.getTensor(value)
    return (try XCTUnwrap(g.cellAllocationSizes[tensor.cellId]), g.frameAwareCells[tensor.cellId] != nil)
  }

  func testMandatoryFrameGroupUsesOneTensorOfScratch() throws {
    let result = try allocation()
    XCTAssertEqual(result.slots, 4)
    XCTAssertFalse(result.frameAware)
  }

  func testIndependentLoopsAndEscapingReadersKeepEveryFrame() throws {
    for result in [try allocation(grouped: false), try allocation(escapes: true)] {
      XCTAssertEqual(result.slots, 512)
      XCTAssertTrue(result.frameAware)
    }
  }

  func testHostMaterializationTrainingAndMetalKeepFrameTapes() throws {
    for result in [try allocation(materialized: true), try allocation(materializedView: true),
      try allocation(training: true),
      try allocation(backend: .metal)] {
      XCTAssertEqual(result.slots, 512)
      XCTAssertTrue(result.frameAware)
    }
  }

  func testHopTensorsRetainTheirSlicedFrameTape() throws {
    let result = try allocation(hop: true)
    XCTAssertEqual(result.slots, 32)
    XCTAssertTrue(result.frameAware)
  }
}

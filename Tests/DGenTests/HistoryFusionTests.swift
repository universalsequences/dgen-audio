import XCTest

@testable import DGen

/// `combineHistoryOpsNotInFeedback` fuses a history read/write pair into one
/// read-then-write node unless the write depends on the read within a frame.
final class HistoryFusionTests: XCTestCase {
  private func isReadWrite(_ graph: Graph, _ id: NodeID) -> Bool {
    if case .historyReadWrite = graph.nodes[id]?.op { return true }
    return false
  }

  /// A "started" flag whose read sits inside a feedback loop but whose write
  /// is a constant still fuses. Left apart, the constant write could run in
  /// an earlier frame loop than the read, which would then never see 0.
  func testIndependentWriteFusesEvenWhenReadIsInFeedback() {
    let g = Graph()
    let flagCell = g.alloc()
    let started = g.n(.historyRead(flagCell))
    let flagWrite = g.n(.historyWrite(flagCell), g.n(.constant(1)))
    let loopCell = g.alloc()
    let previous = g.n(.historyRead(loopCell))
    let value = g.n(.add, g.n(.mul, previous, g.n(.constant(0.5))),
      g.n(.sub, g.n(.constant(1)), started))
    _ = g.n(.historyWrite(loopCell), value)
    _ = g.n(.output(0), value)
    // The flag's read feeds the loop, as a cluster may record it.
    let clusters = findFeedbackLoops(g) + [[started, previous, value]]
    GraphPrepPasses.combineHistoryOpsNotInFeedback(
      g, feedbackClusters: clusters, options: .init())
    XCTAssertTrue(isReadWrite(g, started), "the flag pair should fuse")
    XCTAssertNil(g.nodes[flagWrite], "the fused write node is removed")
    XCTAssertFalse(isReadWrite(g, previous), "the loop's own recurrence stays a pair")
  }
}

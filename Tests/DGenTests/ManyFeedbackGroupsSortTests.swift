import XCTest

@testable import DGen

/// Feedback group ids are 0..<clusters; singleton groups used to be numbered
/// from a fixed 1000. With more than 1000 clusters (Plaits has ~1200 history
/// cells) the ranges collided, unrelated nodes shared a group, and the bogus
/// group cycles fell through to the sorted-order fallback — emitting a gswitch
/// before its `eq` input ("insufficientInputs(gswitch, expected 3, actual 2)").
final class ManyFeedbackGroupsSortTests: XCTestCase {
  func testEveryNodeIsSortedAfterItsInputsWithMoreThan1000FeedbackGroups() {
    let g = Graph()
    // 1200 counters, each its own cycle cluster (ids follow cell order). No
    // constants, so every node built here belongs to a cluster.
    var reads: [NodeID] = []
    var cells: [CellID] = []
    for _ in 0..<1200 {
      let cell = g.alloc()
      cells.append(cell)
      reads.append(g.n(.historyRead(cell)))
    }
    for i in 0..<1200 {
      // Counter 1100 also reads counter 1000: group 1100 depends on group 1000.
      let other = i == 1100 ? reads[1000] : reads[i]
      _ = g.n(.historyWrite(cells[i]), g.n(.add, reads[i], other))
    }
    // The first singleton gets id 1000 under the old numbering — feedback
    // group 1000's id — while it depends on group 1100: a bogus 2-cycle.
    let isZero = g.n(.eq, reads[1100], reads[1100])
    _ = g.n(.output(0), g.n(.gswitch, isZero, reads[0], reads[1]))

    let clusters = findFeedbackLoops(g)
    XCTAssertGreaterThan(clusters.count, 1000)
    let scalar = findSequentialNodes(g, feedbackClusters: clusters, backend: .c)
    let order = topologicalSort(g, feedbackClusters: clusters, scalarNodeSet: scalar)

    var position: [NodeID: Int] = [:]
    for (index, node) in order.enumerated() { position[node] = index }
    XCTAssertEqual(Set(order), Set(g.nodes.keys), "every node must be scheduled exactly once")
    XCTAssertEqual(order.count, g.nodes.count)
    for (id, node) in g.nodes {
      for input in node.inputs {
        guard let at = position[id], let inputAt = position[input] else { continue }
        XCTAssertLessThan(
          inputAt, at,
          "node \(id) (\(node.op)) is scheduled before its input \(input) (\(g.nodes[input]!.op))")
      }
    }
  }
}

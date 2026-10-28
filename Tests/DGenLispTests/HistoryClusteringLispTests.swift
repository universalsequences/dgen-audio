import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Every access to a history cell must land in one feedback cluster, cycle or
/// not. Biquad's x[n-1]/x[n-2] chain writes x[n] and reads it back next frame
/// with no path from the read to the write, so cycle detection alone never
/// grouped it; behind a tensor-feedback kernel the read was then scheduled
/// into an earlier frame loop than the write, and every frame of a multi-frame
/// block filtered with stale x state (`biquad <scalar> <[8] cutoffs>` sampled
/// at element 0 sounded nothing like the same scalar biquad). Which loop the
/// read lands in depends on the surrounding graph, so this asserts the
/// clustering invariant directly instead of a block-size render.
final class HistoryClusteringLispTests: XCTestCase {
  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func assertEveryHistoryCellIsInOneCluster(
    _ source: String, file: StaticString = #filePath, line: UInt = #line
  ) throws {
    let evaluator = LispEvaluator(sourceDirectory: URL(fileURLWithPath: NSTemporaryDirectory()))
    try evaluator.evaluate(nodes: parseSource(source))
    let g = LazyGraphContext.current.graph
    let clusters = findFeedbackLoops(g).map(Set.init)

    var accesses: [Int: Set<NodeID>] = [:]
    var written = Set<Int>()
    var read = Set<Int>()
    for node in g.nodes.values {
      switch node.op {
      case .historyRead(let cell):
        accesses[cell, default: []].insert(node.id)
        read.insert(cell)
      case .historyWrite(let cell):
        accesses[cell, default: []].insert(node.id)
        written.insert(cell)
      default: break
      }
    }
    let stateCells = read.intersection(written)
    XCTAssertFalse(stateCells.isEmpty, "source has no history state", file: file, line: line)
    for cell in stateCells.sorted() {
      let nodes = accesses[cell]!
      XCTAssertTrue(
        clusters.contains { nodes.isSubset(of: $0) },
        "history cell \(cell): reads and writes \(nodes.sorted()) are not in one cluster",
        file: file, line: line)
    }
  }

  func testScalarBiquadInputHistoryIsClustered() throws {
    try assertEveryHistoryCellIsInOneCluster(
      """
      (def x (cos (* (phasor 220 0) 6.283185)))
      (def outsig (biquad x 500 0.707 1 0))
      """)
  }

  func testTensorBiquadOverScalarInputHistoryIsClustered() throws {
    try assertEveryHistoryCellIsInOneCluster(
      """
      (def x (cos (* (phasor 220 0) 6.283185)))
      (def freqs (* 120 (pow 2.0 (/ (iota 8) 12.0))))
      (def outsig (sample (biquad x freqs 1 1 0) 0 0))
      """)
  }
}

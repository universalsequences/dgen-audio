import Foundation
import XCTest

@testable import DGen

final class ExecutionGateDemandTests: XCTestCase {
  private func gate(_ graph: Graph, _ condition: NodeID, _ body: NodeID, _ zero: NodeID) -> NodeID {
    let node = graph.n(.gswitch, condition, body, zero)
    graph.executionGates[node] = condition
    return node
  }

  func testBroadDemandPrecedesCombinatorialPathsThroughSharedHistory() throws {
    let graph = Graph()
    let zero = graph.n(.constant(0))
    let one = graph.n(.constant(1))
    let read = graph.n(.historyRead(0))
    let shared = graph.n(.add, read, one)
    let writer = graph.n(.historyWrite(0), shared)
    var value = shared
    var allValues = shared
    for index in 0..<22 {
      let a = graph.n(.param(index * 2 + 1))
      let b = graph.n(.param(index * 2 + 2))
      value = graph.n(.add, gate(graph, a, value, zero), gate(graph, b, value, zero))
      allValues = graph.n(.add, allValues, value)
    }
    let root = graph.n(.output(0), graph.n(.add, allValues, value))
    // A duplicate root seed is semantically inert. It fixes the seed visitation
    // order so the old depth-first walk explores the 2^22 narrow paths before
    // the unconditional demands for those same intermediate values.
    graph.gradientSideEffects.append(root)
    let start = Date()
    let plan = try ExecutionGatePass.prepare(graph: graph, backend: .c)
    defer { plan.restoreDependencies(graph: graph) }
    XCTAssertEqual(plan.demands[shared], .always)
    XCTAssertEqual(plan.demands[read], .always)
    XCTAssertEqual(plan.demands[writer], .always)
    // This small graph takes milliseconds. A generous ceiling catches the
    // former transient demand explosion without depending on microtimings.
    XCTAssertLessThan(Date().timeIntervalSince(start), 5)
  }

  func testNestedAndSharedDemandMatchesBooleanReachability() throws {
    let graph = Graph()
    let zero = graph.n(.constant(0))
    let one = graph.n(.constant(1))
    let conditions = (1...3).map { graph.n(.param($0)) }
    let read = graph.n(.historyRead(0))
    let shared = graph.n(.add, read, one)
    let writer = graph.n(.historyWrite(0), shared)
    let nested = gate(graph, conditions[0], gate(graph, conditions[1], shared, zero), zero)
    let alternative = gate(graph, conditions[2], shared, zero)
    _ = graph.n(.output(0), graph.n(.add, nested, alternative))
    let original = graph.nodes.mapValues(\.temporalDependencies)
    let plan = try ExecutionGatePass.prepare(graph: graph, backend: .c)
    for mask in 0..<8 {
      let enabled = Set(conditions.enumerated().compactMap { index, id in
        mask & (1 << index) == 0 ? nil : id
      })
      let expected = (mask & 1 != 0 && mask & 2 != 0) || mask & 4 != 0
      for id in [read, shared, writer] {
        let demand = try XCTUnwrap(plan.demands[id])
        XCTAssertEqual(demand.terms.contains { $0.isSubset(of: enabled) }, expected)
      }
    }
    plan.restoreDependencies(graph: graph)
    XCTAssertEqual(graph.nodes.mapValues(\.temporalDependencies), original)
    let second = try ExecutionGatePass.prepare(graph: graph, backend: .c)
    XCTAssertEqual(plan.demands, second.demands)
    second.restoreDependencies(graph: graph)
  }
}

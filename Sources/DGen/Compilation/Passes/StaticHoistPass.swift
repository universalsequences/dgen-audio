import Foundation

/// Lifts frame-invariant scalar math out of the per-sample loops.
///
/// `partitionIntoBlocks` groups nodes by adjacency in topological order, so a
/// parameter read, a `heat-db`-style exponential on it, or an envelope
/// coefficient lands in whatever frame-based block its neighbours are in and is
/// recomputed for every sample. Block temporality already distinguishes
/// `.static_` blocks (rendered once per process call, no loop) from frame-based
/// ones, and the C renderer broadcasts static globals into SIMD loops from lane
/// zero, so the only missing piece is forming such a block.
///
/// A node is hoistable when it is scalar, has no temporal dependencies, is not
/// frame- or hop-based, is not scheduled sequentially, uses an op from a pure
/// allowlist, and every value input is itself hoistable. Hoistable nodes depend
/// only on hoistable nodes, so emitting them all first is a valid schedule.
/// `DGEN_NO_STATIC_HOIST=1` disables the pass for A/B measurement.
enum StaticHoistPass {

  static var isEnabled: Bool {
    ProcessInfo.processInfo.environment["DGEN_NO_STATIC_HOIST"] != "1"
  }

  static func isHoistableOp(_ op: LazyOp) -> Bool {
    switch op {
    case .add, .sub, .div, .mul, .abs, .sign, .sin, .cos, .tan, .atan, .tanh, .exp, .log,
      .log10, .sqrt, .atan2, .gt, .gte, .lte, .lt, .eq, .gswitch, .mix, .pow, .floor, .ceil,
      .round, .mod, .min, .max, .and, .or, .xor, .neg,
      .selector, .constant, .hostSampleRate, .param:
      return true
    default:
      return false
    }
  }

  /// Returns the hoistable nodes in `sortedNodes` order.
  static func hoistableNodes(
    graph: Graph, sortedNodes: [NodeID], frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)], scalarNodeSet: Set<NodeID>
  ) -> [NodeID] {
    var hoistable = Set<NodeID>()
    var ordered: [NodeID] = []
    for id in sortedNodes {
      guard let node = graph.nodes[id],
        isHoistableOp(node.op),
        node.temporalDependencies.isEmpty,
        graph.nodeToTensor[id] == nil,
        !frameBasedNodes.contains(id),
        hopBasedNodes[id] == nil,
        !scalarNodeSet.contains(id),
        !graph.materializeNodes.contains(id),
        node.inputs.allSatisfy({ hoistable.contains($0) })
      else { continue }
      if case .tensor? = node.shape { continue }
      hoistable.insert(id)
      ordered.append(id)
    }
    return ordered
  }
}

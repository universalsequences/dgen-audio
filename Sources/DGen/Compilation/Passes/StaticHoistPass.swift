import Foundation

/// Lifts frame-invariant scalar and tensor math out of the per-sample loops.
///
/// `partitionIntoBlocks` groups nodes by adjacency in topological order, so a
/// parameter read, a `heat-db`-style exponential on it, or an envelope
/// coefficient lands in whatever frame-based block its neighbours are in and is
/// recomputed for every sample. Block temporality already distinguishes
/// `.static_` blocks (rendered once per process call, no loop) from frame-based
/// ones, and the C renderer broadcasts static globals into SIMD loops from lane
/// zero, so the only missing piece is forming such a block.
///
/// A node is hoistable when it has no temporal dependencies (an immutable
/// table peek's ordering dependencies are moot), is not
/// frame- or hop-based, uses an op from a pure
/// allowlist, and every value input is itself hoistable. Hoistable nodes depend
/// only on hoistable nodes, so emitting them all first is a valid schedule.
/// Stored tensors may change between process calls (host parameter updates),
/// but tensors written by the graph and streaming views must remain in place.
/// Sequential frame order alone does not imply a changing value: tensor math
/// is conservatively marked sequential when a graph contains scalar feedback.
/// The closed dependency proof, not that scheduling choice, establishes safety.
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
      .selector, .constant, .hostSampleRate, .param, .changed:
      return true
    default:
      return false
    }
  }

  /// Returns the hoistable nodes in `sortedNodes` order.
  static func hoistableNodes(
    graph: Graph, sortedNodes: [NodeID], frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) -> [NodeID] {
    var hoistable = Set<NodeID>()
    var ordered: [NodeID] = []
    for id in sortedNodes {
      guard let node = graph.nodes[id] else { continue }
      let tablePeek = isImmutableTablePeek(node, graph: graph)
      guard isHoistableOp(node.op) || isStoredTensorRead(node, graph: graph) || tablePeek,
        // seq construction orders every peek in a later operand's cone after
        // the earlier writes (e.g. a delay time behind `delay`'s write). A
        // table nothing writes reads the same whatever the order, so those
        // temporal dependencies do not pin its lookup to the frame loop.
        tablePeek || node.temporalDependencies.isEmpty,
        !frameBasedNodes.contains(id),
        hopBasedNodes[id] == nil,
        !graph.materializeNodes.contains(id),
        node.inputs.allSatisfy({ hoistable.contains($0) })
      else { continue }
      hoistable.insert(id)
      ordered.append(id)
    }
    return ordered
  }

  /// A `peek` into an immutable stored table is a pure function of its index
  /// and channel: when those are hoistable (e.g. derived from a param), the
  /// lookup is frame-invariant and runs once per process call instead of as a
  /// four-lane gather every sample. The source must be the stored tensor itself
  /// (no view transforms, never written by `poke`); the dependency closure in
  /// `hoistableNodes` guarantees index and channel are hoistable too.
  private static func isImmutableTablePeek(_ node: Node, graph: Graph) -> Bool {
    guard case .peek = node.op, node.inputs.count == 3,
      let source = graph.nodes[node.inputs[0]],
      isStoredTensorRead(source, graph: graph),
      graph.mutableTensorReadCell(node) == nil
    else { return false }
    return true
  }

  private static func isStoredTensorRead(_ node: Node, graph: Graph) -> Bool {
    guard case .tensorRef(let tensorId) = node.op,
      let tensor = graph.tensors[tensorId],
      node.inputs.isEmpty, tensor.transforms.isEmpty,
      !graph.mutableTensorCells.contains(tensor.cellId)
    else { return false }
    return true
  }
}

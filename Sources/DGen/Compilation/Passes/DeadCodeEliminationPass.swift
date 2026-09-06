import Foundation

/// Removes graph nodes whose values can never reach an output or a side effect.
///
/// The Lisp front end evaluates every top-level `def` eagerly, so a patch that
/// declares many `@mod true` parameters but only reads a few of them still
/// carries one `.modulatedParam` expansion per declaration (four multiply-adds,
/// a clamp and a select, per sample). Nothing downstream of the pipeline prunes
/// them: `topologicalSort` schedules every node in `graph.nodes`, so the dead
/// arithmetic lands in a block and is rendered into the per-sample loop.
///
/// This pass computes backward reachability from a conservative root set and
/// deletes the rest. Only nodes whose op is on an explicit pure allowlist and
/// that carry no tensor binding are ever removed; everything else (stateful ops,
/// writes, tensors, parameters, inputs, and anything a graph-side map refers to)
/// is treated as a root so the pass can never change observable behaviour or
/// cell allocation. `.param` and `.input` nodes are kept because the manifest's
/// physical cell mapping is derived from the cells the emitted UOps touch.
enum DeadCodeEliminationPass {

  /// Returns true for ops that have no side effects and no state cell, so a
  /// node using them is only worth keeping when something consumes its value.
  static func isPure(_ op: LazyOp) -> Bool {
    switch op {
    case .add, .sub, .div, .mul, .abs, .sign, .sin, .cos, .tan, .atan, .tanh, .exp, .log,
      .log10, .sqrt, .atan2, .gt, .gte, .lte, .lt, .eq, .gswitch, .mix, .pow, .floor, .ceil,
      .round, .mod, .min, .max, .and, .or, .xor, .neg, .mse,
      .selector, .modulatedParam, .constant, .hostSampleRate,
      .historyRead, .deterministicPhasor:
      return true
    default:
      return false
    }
  }

  /// Deletes unreachable pure scalar nodes in place and returns them so the
  /// caller can restore them once compilation is done: a lazy graph outlives a
  /// single compile, and a value nothing reads today may be consumed by nodes
  /// added before the next `realize()`.
  @discardableResult
  static func run(graph: Graph) -> [NodeID: Node] {
    var consumers: [NodeID: [NodeID]] = [:]
    var roots: [NodeID] = []
    for (id, node) in graph.nodes {
      for dep in node.allDependencies {
        consumers[dep, default: []].append(id)
      }
      let removable = isPure(node.op) && graph.nodeToTensor[id] == nil
        && !isTensorShaped(node)
      if !removable {
        roots.append(id)
      }
    }
    // Anything a graph-side map points at stays live regardless of its op.
    roots.append(contentsOf: graph.materializeNodes)
    roots.append(contentsOf: graph.gradientSideEffects)
    roots.append(contentsOf: graph.nodeHopRate.values.map { $0.1 })
    roots.append(contentsOf: graph.nodePositionDep.values)
    roots.append(contentsOf: graph.tensorGradCells.keys)
    roots.append(contentsOf: graph.simdOptimizedConv2Ds)
    roots.append(contentsOf: graph.conv2dMaskCells.keys)
    if let last = graph.lastForwardNodeId { roots.append(last) }

    var live = Set<NodeID>()
    var stack = roots
    while let id = stack.popLast() {
      guard !live.contains(id), let node = graph.nodes[id] else { continue }
      live.insert(id)
      stack.append(contentsOf: node.allDependencies)
    }

    var removed: [NodeID: Node] = [:]
    for id in graph.nodes.keys where !live.contains(id) {
      if let node = graph.nodes.removeValue(forKey: id) {
        removed[id] = node
      }
    }
    _ = consumers
    return removed
  }

  private static func isTensorShaped(_ node: Node) -> Bool {
    if case .tensor? = node.shape { return true }
    return false
  }
}

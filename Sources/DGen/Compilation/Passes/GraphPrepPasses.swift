import Foundation

/// Namespace for graph-preparation passes that run before block partitioning.
enum GraphPrepPasses {}

extension GraphPrepPasses {
  /// Propagates scalar requirements through `seq` inputs while preserving SIMD-safe atomics.
  static func propagateSeqScalarInputs(
    graph: Graph, initialScalarSet: Set<NodeID>
  ) -> Set<NodeID> {
    let simdSafeNodes = findSIMDSafeAtomicNodes(graph: graph)
    var scalarSet = initialScalarSet

    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
      guard case .seq = node.op else { continue }
      let hasScalarInput = node.inputs.contains { scalarSet.contains($0) }
      guard hasScalarInput else { continue }
      for inputId in node.inputs where !simdSafeNodes.contains(inputId) {
        if ProcessInfo.processInfo.environment["DGEN_DEBUG_SCALAR_HOP"] != nil,
          !scalarSet.contains(inputId), let inputNode = graph.nodes[inputId],
          case .tensor = inputNode.shape
        {
          print("[scalar-hop] seq-propagated id=\(inputId) op=\(inputNode.op) via seq=\(nodeId)")
        }
        scalarSet.insert(inputId)
      }
    }

    return scalarSet
  }

  /// Finds nodes that intentionally stay SIMD-safe even when traversed by scalar propagation.
  private static func findSIMDSafeAtomicNodes(graph: Graph) -> Set<NodeID> {
    var simdSafe = Set<NodeID>()
    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
      switch node.op {
      case .memoryAccumulate(_), .tensorAccumulate(_), .chunkPartialsReduceToCell(_, _, _, _, _):
        simdSafe.insert(nodeId)
      default:
        break
      }
    }
    return simdSafe
  }
}

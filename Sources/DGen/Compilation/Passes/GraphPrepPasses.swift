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

    for node in graph.nodes.values {
      guard case .seq = node.op else { continue }
      let hasScalarInput = node.inputs.contains { scalarSet.contains($0) }
      guard hasScalarInput else { continue }
      for inputId in node.inputs where !simdSafeNodes.contains(inputId) {
        scalarSet.insert(inputId)
      }
    }

    return scalarSet
  }

  /// Finds nodes that intentionally stay SIMD-safe even when traversed by scalar propagation.
  private static func findSIMDSafeAtomicNodes(graph: Graph) -> Set<NodeID> {
    var simdSafe = Set<NodeID>()
    for (nodeId, node) in graph.nodes {
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

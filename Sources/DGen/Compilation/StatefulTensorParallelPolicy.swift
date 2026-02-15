import Foundation

/// Shared eligibility checks for scalar frame-loop kernels that can safely parallelize
/// across tensor elements (`id < tensorSize`) while keeping a sequential frame loop.
enum StatefulTensorParallelPolicy {
  struct Decision {
    let enabled: Bool
    let tensorSize: Int
  }

  /// Strict predicate:
  /// - Metal backend only
  /// - scalar block with single known tensor shape
  /// - frameBased or hopBased temporality
  /// - contains stateful tensor-friendly ops (phasor/accum)
  /// - excludes tensor history read/write blocks
  static func decide(block: Block, graph: Graph, backend: Backend) -> Decision {
    guard backend == .metal else { return Decision(enabled: false, tensorSize: 0) }
    guard block.frameOrder == .sequential else { return Decision(enabled: false, tensorSize: 0) }
    guard let shape = block.shape else { return Decision(enabled: false, tensorSize: 0) }
    let tensorSize = shape.reduce(1, *)
    guard tensorSize > 1 else { return Decision(enabled: false, tensorSize: tensorSize) }

    switch block.temporality {
    case .frameBased, .hopBased:
      break
    case .static_:
      return Decision(enabled: false, tensorSize: tensorSize)
    }

    var hasCandidate = false
    for nodeId in block.nodes {
      guard let node = graph.nodes[nodeId] else { continue }
      switch node.op {
      case .phasor(_), .accum(_):
        hasCandidate = true
      case .historyRead(let cellId), .historyWrite(let cellId), .historyReadWrite(let cellId):
        // Keep tensor-history blocks on the existing strict frame-by-frame path.
        if graph.cellToTensor[cellId] != nil {
          return Decision(enabled: false, tensorSize: tensorSize)
        }
      default:
        break
      }
    }

    return Decision(enabled: hasCandidate, tensorSize: tensorSize)
  }
}

import Foundation

/// Combines `historyRead`/`historyWrite` pairs that are outside feedback loops into
/// `historyReadWrite` nodes to reduce graph surface area before scheduling.
extension GraphPrepPasses {
  static func combineHistoryOpsNotInFeedback(
    _ graph: Graph, feedbackClusters: [[NodeID]], options: CompilationPipeline.Options
  ) {
    // Create a set of all nodes that are in feedback loops.
    var nodesInFeedback = Set<NodeID>()
    for cluster in feedbackClusters {
      for nodeId in cluster {
        nodesInFeedback.insert(nodeId)
      }
    }

    // Find all historyRead and historyWrite nodes grouped by cellId.
    var historyReads: [CellID: NodeID] = [:]
    var historyWrites: [CellID: (nodeId: NodeID, inputs: [NodeID])] = [:]

    for (nodeId, node) in graph.nodes {
      switch node.op {
      case .historyRead(let cellId):
        historyReads[cellId] = nodeId
      case .historyWrite(let cellId):
        historyWrites[cellId] = (nodeId: nodeId, inputs: node.inputs)
      default:
        break
      }
    }

    // For each cellId that has both read and write, check if they're not in feedback loops.
    for (cellId, readNodeId) in historyReads {
      if let writeInfo = historyWrites[cellId] {
        // Check if neither the read nor write node is in a feedback loop.
        if !nodesInFeedback.contains(readNodeId) && !nodesInFeedback.contains(writeInfo.nodeId) {
          // Replace the historyRead node with historyReadWrite using the write's inputs.
          if graph.nodes[readNodeId] != nil {
            let newNode = Node(
              id: readNodeId,
              op: .historyReadWrite(cellId),
              inputs: writeInfo.inputs
            )
            graph.nodes[readNodeId] = newNode

            // Remove the historyWrite node.
            graph.nodes.removeValue(forKey: writeInfo.nodeId)

            if options.debug {
              print("   - Converted read node \(readNodeId) to historyReadWrite")
              print("   - Removed historyWrite node \(writeInfo.nodeId)")
              print("   - Inputs: \(writeInfo.inputs)")
            }
          }
        } else if options.debug {
          print("⚠️  Skipping combination for cell \(cellId) - nodes are in feedback loop")
        }
      }
    }
  }
}

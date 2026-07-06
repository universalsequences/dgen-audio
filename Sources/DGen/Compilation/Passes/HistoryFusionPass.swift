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

    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
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
    for cellId in historyReads.keys.sorted() {
      guard let readNodeId = historyReads[cellId] else { continue }
      if let writeInfo = historyWrites[cellId] {
        // Check if neither the read nor write node is in a feedback loop.
        // Skip writes that carry a reset (2nd input): historyReadWrite has no
        // reset path, so fusing would silently drop the reset.
        if writeInfo.inputs.count <= 1
          && !nodesInFeedback.contains(readNodeId) && !nodesInFeedback.contains(writeInfo.nodeId) {
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

            // historyWrite is pass-through: its output is its input's value.
            // Rewire any consumers of the removed write node to the write's
            // input so they keep receiving the same (current-frame) value.
            let passThroughSource = writeInfo.inputs[0]
            for (consumerId, consumer) in graph.nodes
            where consumer.inputs.contains(writeInfo.nodeId)
              || consumer.temporalDependencies.contains(writeInfo.nodeId) {
              var replacement = Node(
                id: consumerId,
                op: consumer.op,
                inputs: consumer.inputs.map {
                  $0 == writeInfo.nodeId ? passThroughSource : $0
                }
              )
              replacement.temporalDependencies = consumer.temporalDependencies.map {
                $0 == writeInfo.nodeId ? passThroughSource : $0
              }
              replacement.shape = consumer.shape
              graph.nodes[consumerId] = replacement
            }

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

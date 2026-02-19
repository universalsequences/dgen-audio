import Foundation

/// Temporality propagation result containing both frame-based and hop-based node sets.
public struct TemporalityResult {
  public let frameBasedNodes: Set<NodeID>
  public let hopBasedNodes: [NodeID: (Int, NodeID)]
}

/// Namespace for temporality analysis and block temporality assignment.
enum TemporalityPass {}

extension TemporalityPass {
  /// Returns true if an op is intrinsically frame-based (its value changes per frame).
  static func isIntrinsicallyFrameBased(_ op: LazyOp) -> Bool {
    switch op {
    case .phasor(_), .deterministicPhasor, .output(_), .accum(_), .input(_),
      .historyRead(_), .historyWrite(_), .historyReadWrite(_), .latch(_), .click(_),
      .overlapAdd(_, _, _, _, _):
      return true
    default:
      return false
    }
  }

  /// Returns intrinsic hop rate for ops that are natively hop-based.
  ///
  /// Current behavior: no op is intrinsically hop-based from a scheduling perspective.
  /// FFT/IFFT emit internal hop logic but remain frame-based producers with hop-based outputs.
  static func intrinsicHopRate(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> (Int, NodeID)? {
    _ = (op, graph, nodeId)
    return nil
  }

  /// Returns hop rate for nodes explicitly marked as hop-output producers.
  static func producesHopBasedOutput(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> (Int, NodeID)? {
    _ = op
    return graph.nodeHopRate[nodeId]
  }

  /// Infers node temporality from intrinsic op properties and input propagation.
  static func inferTemporality(graph: Graph, sortedNodes: [NodeID]) -> TemporalityResult {
    var frameBasedNodes = Set<NodeID>()
    var hopBasedNodes: [NodeID: (Int, NodeID)] = [:]
    var hopProducingNodes: [NodeID: (Int, NodeID)] = [:]

    for nodeId in sortedNodes {
      guard let node = graph.nodes[nodeId] else { continue }

      if let hopRate = producesHopBasedOutput(node.op, graph: graph, nodeId: nodeId) {
        hopProducingNodes[nodeId] = hopRate
        frameBasedNodes.insert(nodeId)
        continue
      }

      if isIntrinsicallyFrameBased(node.op) {
        frameBasedNodes.insert(nodeId)
        continue
      }

      if let hopRate = intrinsicHopRate(node.op, graph: graph, nodeId: nodeId) {
        hopBasedNodes[nodeId] = hopRate
        continue
      }

      let hasFrameBasedInput = node.inputs.contains { inputId in
        // Global reduction ops (sampleGradReduce, peekGradReduce, etc.)
        // aggregate across ALL frames internally. Their output is static,
        // so they should not propagate frame-based temporality downstream.
        if let inputNode = graph.nodes[inputId], isGlobalReductionOp(inputNode.op) {
          return false
        }
        return frameBasedNodes.contains(inputId) && !hopProducingNodes.keys.contains(inputId)
      }
      if hasFrameBasedInput {
        frameBasedNodes.insert(nodeId)
        continue
      }

      var hopInputRates: [(Int, NodeID)] = []
      for inputId in node.inputs {
        if let rate = hopBasedNodes[inputId] {
          hopInputRates.append(rate)
        } else if let rate = hopProducingNodes[inputId] {
          hopInputRates.append(rate)
        }
      }

      if !hopInputRates.isEmpty {
        let fastestRate = hopInputRates.min(by: { $0.0 < $1.0 })!
        hopBasedNodes[nodeId] = fastestRate

        let counterNode = fastestRate.1
        if !node.inputs.contains(counterNode) {
          graph.nodes[nodeId]?.temporalDependencies.append(counterNode)
        }
      }
    }

    // Propagate buffer position dependencies for circular sliding-window modes.
    var positionDeps = graph.nodePositionDep
    for nodeId in sortedNodes {
      guard let node = graph.nodes[nodeId] else { continue }
      for inputId in node.inputs {
        if let posNode = positionDeps[inputId], posNode != nodeId {
          positionDeps[nodeId] = posNode
          if !node.inputs.contains(posNode) && !node.temporalDependencies.contains(posNode) {
            graph.nodes[nodeId]?.temporalDependencies.append(posNode)
          }
        }
      }
    }

    return TemporalityResult(frameBasedNodes: frameBasedNodes, hopBasedNodes: hopBasedNodes)
  }

  /// Assigns block temporality from member node temporality.
  static func assignBlockTemporality(
    blocks: inout [Block],
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) {
    for i in 0..<blocks.count {
      blocks[i].temporality = determineBlockTemporality(
        block: blocks[i],
        frameBasedNodes: frameBasedNodes,
        hopBasedNodes: hopBasedNodes
      )
    }
  }

  /// Determines one block's temporality.
  private static func determineBlockTemporality(
    block: Block,
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) -> Temporality {
    if block.nodes.contains(where: { frameBasedNodes.contains($0) }) {
      return .frameBased
    }

    let hopRates = block.nodes.compactMap { hopBasedNodes[$0] }
    if let firstRate = hopRates.first {
      let allSameHopSize = hopRates.allSatisfy { $0.0 == firstRate.0 }
      if allSameHopSize {
        return .hopBased(hopSize: firstRate.0, counterNode: firstRate.1)
      }
      return .frameBased
    }

    return .static_
  }
}

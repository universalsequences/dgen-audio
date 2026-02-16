import Foundation

extension GraphPrepPasses {
  /// Walks backwards from a node through view-only ops to find the underlying
  /// compute/data node. Returns the source node ID and the IDs of intermediate view nodes.
  private static func traceViewChain(
    from nodeId: NodeID, graph: Graph
  ) -> (source: NodeID, viewNodeIds: [NodeID]) {
    var viewNodeIds: [NodeID] = []
    var currentId = nodeId
    while let node = graph.nodes[currentId], node.op.isViewOnly {
      viewNodeIds.append(currentId)
      guard let nextInput = node.inputs.first else { break }
      currentId = nextInput
    }
    return (currentId, viewNodeIds)
  }

  /// Detects `sumAxis(mul(a, b))` patterns that represent matrix multiplication and
  /// rewrites them as `.gemm` nodes, removing orphaned intermediate nodes.
  static func gemmPass(graph: Graph) {
    //if ProcessInfo.processInfo.environment["DGEN_NO_GEMM"] != nil { return }

    for (nodeId, node) in graph.nodes {
      guard node.inputs.count == 1,
        let mulNode = graph.nodes[node.inputs[0]]
      else { continue }

      guard case .sumAxis(let axis) = node.op,
        case .mul = mulNode.op,
        mulNode.inputs.count == 2
      else { continue }

      guard case .tensor(let mulShape) = mulNode.shape, mulShape.count == 3 else { continue }

      let K = mulShape[axis]
      let nonReduceAxes = (0..<3).filter { $0 != axis }
      let axisM = nonReduceAxes[0]
      let axisN = nonReduceAxes[1]
      let M = mulShape[axisM]
      let N = mulShape[axisN]

      guard let input0Node = graph.nodes[mulNode.inputs[0]],
        let input1Node = graph.nodes[mulNode.inputs[1]],
        case .tensor(let input0Shape) = input0Node.shape,
        case .tensor(let input1Shape) = input1Node.shape
      else { continue }

      // Identify left (M-owning) vs right (N-owning) input by broadcast pattern:
      // the input with size-1 at axisN owns rows (left), size-1 at axisM owns cols (right).
      let leftInputIdx: Int
      let rightInputIdx: Int
      if input0Shape[axisN] == 1 && input1Shape[axisM] == 1 {
        leftInputIdx = 0
        rightInputIdx = 1
      } else if input0Shape[axisM] == 1 && input1Shape[axisN] == 1 {
        leftInputIdx = 1
        rightInputIdx = 0
      } else {
        continue
      }

      guard M % 8 == 0, N % 8 == 0, K % 8 == 0 else { continue }

      let (leftSource, leftViewIds) = traceViewChain(
        from: mulNode.inputs[leftInputIdx], graph: graph)
      let (rightSource, rightViewIds) = traceViewChain(
        from: mulNode.inputs[rightInputIdx], graph: graph)

      // Replace sumAxis with gemm, preserving the nodeId so downstream consumers are unaffected
      graph.nodes[nodeId] = Node(
        id: nodeId, op: .gemm(M, N, K), inputs: [leftSource, rightSource])
      graph.nodes[nodeId]?.shape = .tensor([M, N])

      // Remove orphaned intermediate nodes (mul + view chains)
      let mulNodeId = node.inputs[0]
      let orphanCandidates = [mulNodeId] + leftViewIds + rightViewIds
      for candidateId in orphanCandidates {
        let hasOtherConsumer = graph.nodes.values.contains {
          $0.id != nodeId && $0.inputs.contains(candidateId)
        }
        if !hasOtherConsumer {
          graph.nodes.removeValue(forKey: candidateId)
        }
      }
    }
  }
}

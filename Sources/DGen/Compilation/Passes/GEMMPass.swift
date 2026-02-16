import Foundation

extension GraphPrepPasses {
  /// Walk backwards from a node through view-only ops (reshape, transpose, expandView, shrink, pad)
  /// to find the underlying compute/data node and the chain of view ops applied.
  /// Returns (sourceNodeID, [viewOps from source→target order])
  private static func traceViewChain(
    from nodeId: NodeID, graph: Graph
  ) -> (source: NodeID, viewOps: [(NodeID, LazyOp)]) {
    var viewOps: [(NodeID, LazyOp)] = []
    var currentId = nodeId
    while let node = graph.nodes[currentId], node.op.isViewOnly {
      viewOps.append((currentId, node.op))
      guard let nextInput = node.inputs.first else { break }
      currentId = nextInput
    }
    return (currentId, viewOps)
  }

  static func gemmPass(graph: Graph) {
    if ProcessInfo.processInfo.environment["DGEN_NO_GEMM"] != nil { return }
    for (nodeId, node) in graph.nodes {
      guard node.inputs.count == 1 else { continue }
      guard let mulNode = graph.nodes[node.inputs[0]] else { continue }

      // Pattern: sumAxis(mul(a, b))
      guard case .sumAxis(let axis) = node.op,
        case .mul = mulNode.op,
        mulNode.inputs.count == 2
      else { continue }

      // Get the mul's shape — this is [M, N, K] (or some permutation)
      guard case .tensor(let mulShape) = mulNode.shape, mulShape.count == 3 else { continue }

      // The reduce axis tells us which dimension is K
      let K = mulShape[axis]
      let nonReduceAxes = (0..<3).filter { $0 != axis }

      // Get shapes of the mul's direct inputs (the broadcast shapes before multiply)
      guard let input0Node = graph.nodes[mulNode.inputs[0]],
        let input1Node = graph.nodes[mulNode.inputs[1]],
        case .tensor(let input0Shape) = input0Node.shape,
        case .tensor(let input1Shape) = input1Node.shape
      else { continue }

      // Determine which input owns M (rows) vs N (cols) by finding broadcast dims.
      // Each input should have size-1 at exactly one non-reduce axis (the broadcast axis).
      let axisM = nonReduceAxes[0]
      let axisN = nonReduceAxes[1]
      let M = mulShape[axisM]
      let N = mulShape[axisN]

      // The input with size-1 at axisM is broadcast there → it owns N (right matrix).
      // The input with size-1 at axisN is broadcast there → it owns M (left matrix).
      let leftInputIdx: Int
      let rightInputIdx: Int
      if input0Shape[axisN] == 1 && input1Shape[axisM] == 1 {
        leftInputIdx = 0
        rightInputIdx = 1
      } else if input0Shape[axisM] == 1 && input1Shape[axisN] == 1 {
        leftInputIdx = 1
        rightInputIdx = 0
      } else {
        continue  // not a matmul broadcast pattern
      }

      // Validate WMMA compatibility
      guard M % 8 == 0, N % 8 == 0, K % 8 == 0 else { continue }

      // Trace each input back through view ops to find source tensors
      let (leftSource, leftViews) = traceViewChain(from: mulNode.inputs[leftInputIdx], graph: graph)
      let (rightSource, rightViews) = traceViewChain(
        from: mulNode.inputs[rightInputIdx], graph: graph)

      // Rewrite: replace the sumAxis node with a gemm node pointing directly
      // to source tensors. Same nodeId so downstream consumers are unaffected.
      graph.nodes[nodeId] = Node(
        id: nodeId, op: .gemm(M, N, K), inputs: [leftSource, rightSource])
      graph.nodes[nodeId]?.shape = .tensor([M, N])

      // Remove orphaned intermediate nodes (mul + view chains),
      // but only if no other node in the graph consumes them.
      let mulNodeId = node.inputs[0]
      let orphanCandidates = [mulNodeId] + leftViews.map(\.0) + rightViews.map(\.0)
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

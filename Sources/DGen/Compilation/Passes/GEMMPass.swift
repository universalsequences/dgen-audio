import Foundation

extension GraphPrepPasses {
  /// Walks backwards from a node through view-only ops to find the underlying
  /// compute/data node. Returns the source node ID, the IDs of intermediate view nodes,
  /// whether any transpose ops were encountered, and the nearest 2D shape found in the chain
  /// (used when the source itself is 1D, e.g., a tensorRef before reshape).
  private static func traceViewChain(
    from nodeId: NodeID, graph: Graph
  ) -> (source: NodeID, viewNodeIds: [NodeID], hasTranspose: Bool, nearest2DShape: [Int]?) {
    var viewNodeIds: [NodeID] = []
    var currentId = nodeId
    var hasTranspose = false
    var nearest2DShape: [Int]? = nil

    while let node = graph.nodes[currentId], node.op.isViewOnly {
      if case .tensor(let shape) = node.shape, shape.count == 2 {
        nearest2DShape = shape
      }
      if case .transpose = node.op {
        hasTranspose = !hasTranspose
      }
      viewNodeIds.append(currentId)
      guard let nextInput = node.inputs.first else { break }
      currentId = nextInput
    }

    if let sourceNode = graph.nodes[currentId],
      case .tensor(let shape) = sourceNode.shape, shape.count == 2
    {
      nearest2DShape = shape
    }

    return (currentId, viewNodeIds, hasTranspose, nearest2DShape)
  }

  /// Determine if a GEMM input needs a transposed load by comparing source shape
  /// against expected layout, using view chain transpose info to disambiguate square matrices.
  private static func needsTranspose(
    sourceShape: [Int], expected: [Int], viewChainHasTranspose: Bool
  ) -> Bool? {
    if sourceShape == expected {
      // Shapes match directly. For non-square matrices this means no transpose.
      // For square matrices (ambiguous), the view chain tells us if the physical
      // layout was transposed relative to what the 3D mul position implies.
      return viewChainHasTranspose
    }
    if sourceShape == [expected[1], expected[0]] {
      // Shapes are reversed — transpose detected by shape alone.
      // View chain transpose would cancel it out (double-transpose = no transpose).
      return !viewChainHasTranspose
    }
    return nil
  }

  /// Detects `sumAxis(mul(a, b))` patterns that represent matrix multiplication and
  /// rewrites them as `.gemm` nodes, removing orphaned intermediate nodes.
  /// Handles both forward patterns (broadcast size-1 dims) and backward patterns
  /// (expandAxis from sumAxis backward).
  static func gemmPass(graph: Graph) {
    for (nodeId, node) in graph.nodes {
      guard node.inputs.count == 1,
        let mulNode = graph.nodes[node.inputs[0]]
      else { continue }

      guard case .sumAxis(let axis) = node.op,
        case .mul = mulNode.op,
        mulNode.inputs.count == 2
      else { continue }

      guard let input0Node = graph.nodes[mulNode.inputs[0]],
        let input1Node = graph.nodes[mulNode.inputs[1]],
        case .tensor(let input0Shape) = input0Node.shape,
        case .tensor(let input1Shape) = input1Node.shape
      else { continue }

      guard case .tensor(let mulShape) = mulNode.shape else { continue }

      // Fused reduction pattern: sumAxis0(mul(A, B)) where A and B are 2D tensors [M, N].
      // This emits a dedicated reduction op that maps naturally to perFrameScaled(N)
      // and avoids the generic axis-reduce shape-region fallback.
      if mulShape.count == 2 {
        guard axis == 0,
          mulShape == input0Shape,
          mulShape == input1Shape,
          mulShape.count == 2
        else { continue }

        let (leftSource, leftViewIds, _, leftNearest2D) = traceViewChain(
          from: mulNode.inputs[0], graph: graph)
        let (rightSource, rightViewIds, _, rightNearest2D) = traceViewChain(
          from: mulNode.inputs[1], graph: graph)

        func sourceMatchesMulShape(_ sourceId: NodeID, _ nearest2D: [Int]?) -> Bool {
          if let node = graph.nodes[sourceId],
            case .tensor(let sourceShape) = node.shape, sourceShape.count == 2
          {
            return sourceShape == mulShape
          }
          return nearest2D == mulShape
        }

        guard sourceMatchesMulShape(leftSource, leftNearest2D),
          sourceMatchesMulShape(rightSource, rightNearest2D)
        else { continue }

        graph.nodes[nodeId] = Node(
          id: nodeId,
          op: .sumMulAxis0,
          inputs: [leftSource, rightSource]
        )
        graph.nodes[nodeId]?.shape = .tensor([mulShape[1]])

        let mulNodeId = node.inputs[0]
        let orphanCandidates = [mulNodeId] + leftViewIds + rightViewIds
        removeOrphans(orphanCandidates, excludingConsumer: nodeId, graph: graph)
        continue
      }

      guard mulShape.count == 3 else { continue }

      let K = mulShape[axis]
      let nonReduceAxes = (0..<3).filter { $0 != axis }
      let axisM = nonReduceAxes[0]
      let axisN = nonReduceAxes[1]
      let M = mulShape[axisM]
      let N = mulShape[axisN]

      guard M % 8 == 0, N % 8 == 0, K % 8 == 0 else { continue }

      // --- Forward pattern: identify left/right by broadcast size-1 dims ---
      //
      // Left (A) has size 1 along axisN, right (B) has size 1 along axisM.
      // If the roles are swapped, flip the input indices.
      let input0IsLeft = input0Shape[axisN] == 1 && input1Shape[axisM] == 1
      let input0IsRight = input0Shape[axisM] == 1 && input1Shape[axisN] == 1
      if input0IsLeft || input0IsRight {
        let leftInputIdx = input0IsLeft ? 0 : 1
        let rightInputIdx = 1 - leftInputIdx

        let (leftSource, leftViewIds, _, _) = traceViewChain(
          from: mulNode.inputs[leftInputIdx], graph: graph)
        let (rightSource, rightViewIds, _, _) = traceViewChain(
          from: mulNode.inputs[rightInputIdx], graph: graph)

        graph.nodes[nodeId] = Node(
          id: nodeId, op: .gemm(M, N, K, false, false), inputs: [leftSource, rightSource])
        graph.nodes[nodeId]?.shape = .tensor([M, N])

        let mulNodeId = node.inputs[0]
        let orphanCandidates = [mulNodeId] + leftViewIds + rightViewIds
        removeOrphans(orphanCandidates, excludingConsumer: nodeId, graph: graph)
        continue
      }

      // --- Backward pattern: one input is expandAxis (from sumAxis backward) ---
      guard let match = matchBackwardGemm(
        mulNode: mulNode,
        input0Node: input0Node,
        input1Node: input1Node,
        axis: axis,
        axisM: axisM,
        axisN: axisN,
        M: M,
        N: N,
        K: K,
        graph: graph
      ) else { continue }

      graph.nodes[nodeId] = Node(
        id: nodeId,
        op: .gemm(M, N, K, match.transA, match.transB),
        inputs: [match.leftSource, match.rightSource])
      graph.nodes[nodeId]?.shape = .tensor([M, N])

      let mulNodeId = node.inputs[0]
      let orphanCandidates = [mulNodeId, match.expandNodeId] + match.viewIds
      removeOrphans(orphanCandidates, excludingConsumer: nodeId, graph: graph)
    }

    fuseCrossFrameTensorAccumulateGemm(graph: graph)
  }

  /// Attempts to match the backward matmul pattern where one mul input is an
  /// `expandAxis` node (from sumAxis backward) and the other is a broadcast
  /// view chain from the forward pass.
  private static func matchBackwardGemm(
    mulNode: Node,
    input0Node: Node,
    input1Node: Node,
    axis: Int,
    axisM: Int,
    axisN: Int,
    M: Int,
    N: Int,
    K: Int,
    graph: Graph
  ) -> (leftSource: NodeID, rightSource: NodeID, transA: Bool, transB: Bool,
    expandNodeId: NodeID, viewIds: [NodeID])?
  {
    let inputNodes = [input0Node, input1Node]
    guard let expandIdx = inputNodes.firstIndex(where: {
      if case .expandAxis = $0.op { return true }; return false
    }) else { return nil }

    guard let expandNode = graph.nodes[mulNode.inputs[expandIdx]],
      case .expandAxis(_, let expandAxisPos) = expandNode.op,
      expandNode.inputs.count == 1
    else { return nil }

    let expandSource = expandNode.inputs[0]
    guard let expandSourceNode = graph.nodes[expandSource],
      case .tensor(let expandSrcShape) = expandSourceNode.shape,
      expandSrcShape.count == 2
    else { return nil }

    // The expand axis must not be the reduce axis (they'd cancel out)
    let remaining = [0, 1, 2].filter { $0 != expandAxisPos }
    guard remaining.contains(axis) else { return nil }

    // Trace the non-expandAxis input through view chain to its 2D source.
    // The source might be 1D (e.g., Tensor(data).reshape([K,N]) has a 1D tensorRef),
    // so we track the nearest 2D shape seen in the chain for shape comparison.
    let otherIdx = 1 - expandIdx
    let (otherSource, otherViewIds, otherHasTranspose, otherNearest2D) = traceViewChain(
      from: mulNode.inputs[otherIdx], graph: graph)

    // Use the traced source's shape if 2D, otherwise fall back to the nearest 2D shape in the chain
    let otherSrcShape: [Int]
    if let node = graph.nodes[otherSource],
      case .tensor(let shape) = node.shape, shape.count == 2
    {
      otherSrcShape = shape
    } else if let nearest = otherNearest2D {
      otherSrcShape = nearest
    } else {
      return nil
    }

    // Determine if expandAxis source is LEFT (owns M_out) or RIGHT (owns N_out).
    // The expandAxis inserts a dim at expandAxisPos. The two remaining 3D positions
    // map to the 2D source's dims. We check which remaining position corresponds to
    // the GEMM's M axis to decide if the expand source is left or right.
    let expandOtherAxis = remaining.first { $0 != axis }!
    let expandIsLeft = (expandOtherAxis == axisM)

    // Determine transposes using axis mapping (works for all matrix sizes including square).
    //
    // The expandAxis source maps its 2D dims to 3D positions `remaining[0]` and `remaining[1]`.
    // GEMM left expects [M, K] -> dim0=axisM, dim1=axis.
    // GEMM right expects [K, N] -> dim0=axis, dim1=axisN.
    //
    // If the expand source is LEFT, check if remaining[0]==axisM (no transpose) or ==axis (transpose).
    // For the OTHER input, use shape comparison + view chain transpose info.
    let expandTranspose: Bool
    if expandIsLeft {
      expandTranspose = (remaining[0] != axisM)
    } else {
      expandTranspose = (remaining[0] != axis)
    }

    let transA: Bool
    let transB: Bool
    let leftSource: NodeID
    let rightSource: NodeID

    if expandIsLeft {
      leftSource = expandSource
      rightSource = otherSource
      transA = expandTranspose
      guard let tb = needsTranspose(
        sourceShape: otherSrcShape, expected: [K, N],
        viewChainHasTranspose: otherHasTranspose)
      else { return nil }
      transB = tb
    } else {
      leftSource = otherSource
      rightSource = expandSource
      transB = expandTranspose
      guard let ta = needsTranspose(
        sourceShape: otherSrcShape, expected: [M, K],
        viewChainHasTranspose: otherHasTranspose)
      else { return nil }
      transA = ta
    }

    return (leftSource, rightSource, transA, transB, mulNode.inputs[expandIdx], otherViewIds)
  }

  /// Remove orphaned nodes that have no consumers besides the excluded node.
  private static func removeOrphans(
    _ candidates: [NodeID], excludingConsumer: NodeID, graph: Graph
  ) {
    for candidateId in candidates {
      let hasOtherConsumer = graph.nodes.values.contains {
        $0.id != excludingConsumer && $0.inputs.contains(candidateId)
      }
      if !hasOtherConsumer {
        graph.nodes.removeValue(forKey: candidateId)
      }
    }
  }

  /// Rewrites `tensorAccumulate(cell, gemm(...))` into a deterministic two-pass reduction.
  ///
  /// **Problem**: Backward parameter gradients for matmul sum per-frame GEMM results across
  /// all frames into a single gradient tensor. `tensorAccumulate` uses atomic adds, whose
  /// ordering varies between runs, producing non-deterministic floating-point results.
  ///
  /// **Solution**: Two-pass chunked reduction with fixed summation order:
  /// 1. `gemmChunkPartials` — groups frames into chunks of 64. Each chunk accumulates its
  ///    frames' GEMM tiles into one partial `[chunkCount, M, N]` tensor. Dispatched as
  ///    `tilesM × tilesN × chunkCount` threadgroups (one chunk per z-index).
  /// 2. `chunkPartialsReduceToCell` — one thread per output element sums partials across
  ///    chunks in fixed order `[0..chunkCount)`, then writes to the target cell. No atomics.
  ///
  /// Chunking (vs. a single kernel looping all frames) preserves GPU occupancy: multiple
  /// threadgroups work on different frame ranges in parallel.
  ///
  /// **Match conditions**:
  /// - node is `tensorAccumulate(targetCell)` with one input
  /// - that input traces through view-only ops to a `.gemm(...)` source
  /// - the traced gemm output has no other non-view consumers
  ///
  /// Typical trigger site is frame-based backward parameter gradients:
  /// per-frame gemm output reduced across frames into a single gradient tensor.
  private static func fuseCrossFrameTensorAccumulateGemm(graph: Graph) {
    struct Match {
      let nodeId: NodeID
      let targetCell: CellID
      let sourceNodeId: NodeID
      let viewNodeIds: [NodeID]
      let outputHasTranspose: Bool
      let gemmInputs: [NodeID]
      let M: Int
      let N: Int
      let K: Int
      let transA: Bool
      let transB: Bool
    }

    var matches: [Match] = []
    for nodeId in graph.nodes.keys.sorted() {
      guard let node = graph.nodes[nodeId] else { continue }
      guard case .tensorAccumulate(let targetCell) = node.op,
        node.inputs.count == 1
      else { continue }

      let (sourceNodeId, viewNodeIds, outputHasTranspose, _) = traceViewChain(
        from: node.inputs[0], graph: graph)
      guard let gemmNode = graph.nodes[sourceNodeId],
        case .gemm(let M, let N, let K, let transA, let transB) = gemmNode.op,
        gemmNode.inputs.count == 2
      else { continue }

      let hasOtherConsumer = graph.nodes.values.contains {
        $0.id != nodeId && !viewNodeIds.contains($0.id) && $0.inputs.contains(sourceNodeId)
      }
      // Keep semantics simple: only rewrite when this accumulate is the sole compute consumer.
      // If gemm has another real consumer, it must remain materialized as a normal tensor.
      guard !hasOtherConsumer else { continue }

      matches.append(
        Match(
          nodeId: nodeId,
          targetCell: targetCell,
          sourceNodeId: sourceNodeId,
          viewNodeIds: viewNodeIds,
          outputHasTranspose: outputHasTranspose,
          gemmInputs: gemmNode.inputs,
          M: M,
          N: N,
          K: K,
          transA: transA,
          transB: transB
        ))
    }

    // Fixed chunk size for now. We can tune or gate this later with a cost model.
    let chunkSize = 64
    let chunkCount = max(1, (graph.maxFrameCount + chunkSize - 1) / chunkSize)

    for match in matches {
      // Skip if rewritten by an earlier transform.
      guard let node = graph.nodes[match.nodeId],
        case .tensorAccumulate = node.op
      else { continue }

      let partialNodeId = graph.n(
        .gemmChunkPartials(
          match.M, match.N, match.K, match.transA, match.transB, chunkSize, chunkCount),
        match.gemmInputs,
        shape: .tensor([chunkCount, match.M, match.N])
      )

      graph.nodes[match.nodeId] = Node(
        id: match.nodeId,
        op: .chunkPartialsReduceToCell(
          match.targetCell, match.M, match.N, chunkCount, match.outputHasTranspose),
        inputs: [partialNodeId]
      )
      graph.nodes[match.nodeId]?.shape = .scalar

      removeOrphans(
        [match.sourceNodeId] + match.viewNodeIds, excludingConsumer: match.nodeId, graph: graph)
    }
  }
}

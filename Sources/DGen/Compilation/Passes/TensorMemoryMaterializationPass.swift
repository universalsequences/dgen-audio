import Foundation

/// Namespace for temporality-aware lazy tensor memory materialization.
enum TensorMemoryMaterializationPass {}

private struct TensorCellLivenessAnalysis {
  let cellToNode: [CellID: NodeID]
  let outboundCells: Set<CellID>
  let intraBlockFrameAwareCells: Set<CellID>
}

private struct TensorAllocationDecision {
  let shouldMaterialize: Bool
  let needsFrameAwareAlloc: Bool
}

extension TensorMemoryMaterializationPass {
  /// Materializes lazy tensor cells to real memory after temporality and block layout are finalized.
  ///
  /// Allocation policy:
  /// - outbound or materialized tensors are always allocated
  /// - frame/hop tensors that cross frame boundaries get `tensorSize * frameCount`
  /// - internal tensors that stay register-local keep lazy ids but still publish size metadata
  static func allocateTensorMemory(
    graph: Graph,
    blocks: [Block],
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)] = [:],
    feedbackClusterNodes: Set<NodeID> = [],
    backend: Backend = .metal,
    frameCount: Int
  ) {
    let liveness = analyzeTensorCellLiveness(
      graph: graph,
      blocks: blocks,
      frameBasedNodes: frameBasedNodes,
      hopBasedNodes: hopBasedNodes
    )

    var lazyToReal: [CellID: CellID] = [:]

    for (tensorId, tensor) in graph.tensors {
      guard tensor.isLazy else { continue }

      let tensorSize = tensor.shape.reduce(1, *)
      let lazyCellId = tensor.cellId
      let nodeId = liveness.cellToNode[lazyCellId]

      let decision = decideTensorAllocation(
        graph: graph,
        lazyCellId: lazyCellId,
        nodeId: nodeId,
        backend: backend,
        frameBasedNodes: frameBasedNodes,
        hopBasedNodes: hopBasedNodes,
        feedbackClusterNodes: feedbackClusterNodes,
        outboundCells: liveness.outboundCells,
        intraBlockFrameAwareCells: liveness.intraBlockFrameAwareCells
      )

      if !decision.shouldMaterialize {
        graph.cellAllocationSizes[lazyCellId] = tensorSize
        continue
      }

      let allocSize = decision.needsFrameAwareAlloc ? tensorSize * frameCount : tensorSize
      let realCellId = graph.allocateLazyCell(lazyCellId, vectorWidth: allocSize)
      lazyToReal[lazyCellId] = realCellId

      if decision.needsFrameAwareAlloc {
        graph.frameAwareCells[realCellId] = (tensorSize: tensorSize, frameCount: frameCount)
      }

      graph.tensors[tensorId] = Tensor(
        id: tensor.id,
        shape: tensor.shape,
        cellId: realCellId,
        data: tensor.data,
        baseShape: tensor.baseShape,
        baseStrides: tensor.baseStrides,
        transforms: tensor.transforms,
        isLazy: false,
        materialize: tensor.materialize
      )
      graph.cellToTensor[realCellId] = tensorId
    }

    patchViewCellIds(graph: graph, lazyToReal: lazyToReal)
  }

  /// Computes liveness facts used by allocation decisioning.
  private static func analyzeTensorCellLiveness(
    graph: Graph,
    blocks: [Block],
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) -> TensorCellLivenessAnalysis {
    var cellToNode: [CellID: NodeID] = [:]
    for (nodeId, tensorId) in graph.nodeToTensor {
      if let tensor = graph.tensors[tensorId] {
        cellToNode[tensor.cellId] = nodeId
      }
    }

    var outboundCells: Set<CellID> = []
    var intraBlockFrameAwareCells: Set<CellID> = []

    for (blockIdx, block) in blocks.enumerated() {
      let producedCells = producedTensorCells(in: block, graph: graph)
      if producedCells.isEmpty { continue }

      collectOutboundCells(
        into: &outboundCells,
        producedCells: producedCells,
        producerBlockIndex: blockIdx,
        blocks: blocks,
        graph: graph
      )

      let isHopBased: Bool
      if case .hopBased = block.temporality {
        isHopBased = true
      } else {
        isHopBased = false
      }
      let isFrameBasedTensorBlock = (block.temporality == .frameBased || isHopBased) && block.shape != nil

      if isFrameBasedTensorBlock {
        collectIntraBlockFrameAwareCells(
          into: &intraBlockFrameAwareCells,
          block: block,
          producedCells: producedCells,
          graph: graph,
          frameBasedNodes: frameBasedNodes,
          hopBasedNodes: hopBasedNodes
        )
      }
    }

    return TensorCellLivenessAnalysis(
      cellToNode: cellToNode,
      outboundCells: outboundCells,
      intraBlockFrameAwareCells: intraBlockFrameAwareCells
    )
  }

  /// Collects tensor cells produced by tensor-shaped nodes in one block.
  private static func producedTensorCells(in block: Block, graph: Graph) -> Set<CellID> {
    var produced: Set<CellID> = []
    for nodeId in block.nodes {
      guard let node = graph.nodes[nodeId], case .tensor = node.shape else { continue }
      if let tensorId = graph.nodeToTensor[nodeId], let tensor = graph.tensors[tensorId] {
        produced.insert(tensor.cellId)
      }
    }
    return produced
  }

  /// Marks produced cells that are consumed by nodes in later blocks.
  private static func collectOutboundCells(
    into outboundCells: inout Set<CellID>,
    producedCells: Set<CellID>,
    producerBlockIndex: Int,
    blocks: [Block],
    graph: Graph
  ) {
    guard producerBlockIndex + 1 < blocks.count else { return }

    for laterIdx in (producerBlockIndex + 1)..<blocks.count {
      for nodeId in blocks[laterIdx].nodes {
        guard let node = graph.nodes[nodeId] else { continue }

        markTensorInputDependenciesAsOutbound(
          nodeInputs: node.inputs,
          producedCells: producedCells,
          graph: graph,
          outboundCells: &outboundCells
        )

        if case .historyWrite = node.op {
          markTensorInputDependenciesAsOutbound(
            nodeInputs: node.inputs,
            producedCells: producedCells,
            graph: graph,
            outboundCells: &outboundCells
          )
        }
      }
    }
  }

  /// Marks produced tensor cells that are consumed by a set of input node IDs.
  private static func markTensorInputDependenciesAsOutbound(
    nodeInputs: [NodeID],
    producedCells: Set<CellID>,
    graph: Graph,
    outboundCells: inout Set<CellID>
  ) {
    for inputId in nodeInputs {
      guard let inputNode = graph.nodes[inputId], case .tensor = inputNode.shape else { continue }
      if let tensorId = graph.nodeToTensor[inputId], let tensor = graph.tensors[tensorId],
        producedCells.contains(tensor.cellId)
      {
        outboundCells.insert(tensor.cellId)
      }
    }
  }

  /// Marks cells produced+consumed inside one frame/hop tensor block.
  private static func collectIntraBlockFrameAwareCells(
    into intraBlockFrameAwareCells: inout Set<CellID>,
    block: Block,
    producedCells: Set<CellID>,
    graph: Graph,
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) {
    for nodeId in block.nodes {
      guard let node = graph.nodes[nodeId] else { continue }
      for inputId in node.inputs {
        guard let inputNode = graph.nodes[inputId], case .tensor = inputNode.shape else { continue }
        guard let tensorId = graph.nodeToTensor[inputId], let tensor = graph.tensors[tensorId] else {
          continue
        }

        let isFrameAwareTemporalInput = frameBasedNodes.contains(inputId) || hopBasedNodes[inputId] != nil
        if producedCells.contains(tensor.cellId) && isFrameAwareTemporalInput {
          intraBlockFrameAwareCells.insert(tensor.cellId)
        }
      }
    }
  }

  /// Decides if/when a lazy tensor cell should be materialized and frame-aware sized.
  private static func decideTensorAllocation(
    graph: Graph,
    lazyCellId: CellID,
    nodeId: NodeID?,
    backend: Backend,
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)],
    feedbackClusterNodes: Set<NodeID>,
    outboundCells: Set<CellID>,
    intraBlockFrameAwareCells: Set<CellID>
  ) -> TensorAllocationDecision {
    let isOutbound = outboundCells.contains(lazyCellId)
    let isFrameBasedByTemporality =
      nodeId.map { frameBasedNodes.contains($0) || hopBasedNodes[$0] != nil } ?? false
    let isChunkedGemmPartial = nodeId.flatMap { graph.nodes[$0] }.map {
      if case .gemmChunkPartials = $0.op { return true }
      return false
    } ?? false
    // Chunked GEMM partial outputs already encode chunk dimension in their tensor shape.
    // They are not per-frame tensors and must not be multiplied by frameCount again.
    let isFrameBased = isFrameBasedByTemporality && !isChunkedGemmPartial

    let isInFeedbackLoop = backend == .c && nodeId.map { feedbackClusterNodes.contains($0) } ?? false

    let needsFrameAwareForFlow = isFrameBased && !isInFeedbackLoop &&
      (isOutbound || intraBlockFrameAwareCells.contains(lazyCellId))

    let shouldMaterializeNode = nodeId.map { graph.materializeNodes.contains($0) } ?? false
    let shouldMaterialize = shouldMaterializeNode || isOutbound

    let needsFrameAwareForMaterialize = shouldMaterialize && isFrameBased && !isInFeedbackLoop
    let needsFrameAwareAlloc = needsFrameAwareForFlow || needsFrameAwareForMaterialize

    let shouldAllocate = shouldMaterialize || needsFrameAwareAlloc
    return TensorAllocationDecision(
      shouldMaterialize: shouldAllocate,
      needsFrameAwareAlloc: needsFrameAwareAlloc
    )
  }

  /// Updates view tensors that still reference lazy cell IDs after base-cell allocation.
  private static func patchViewCellIds(graph: Graph, lazyToReal: [CellID: CellID]) {
    for (tensorId, tensor) in graph.tensors {
      guard tensor.isView, let realCellId = lazyToReal[tensor.cellId] else { continue }

      graph.tensors[tensorId] = Tensor(
        id: tensor.id,
        shape: tensor.shape,
        cellId: realCellId,
        data: tensor.data,
        baseShape: tensor.baseShape,
        baseStrides: tensor.baseStrides,
        transforms: tensor.transforms,
        isLazy: false,
        materialize: tensor.materialize
      )
    }
  }
}

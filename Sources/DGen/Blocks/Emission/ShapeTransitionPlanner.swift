/// Shape-transition detection and region planning for block emission.
import Foundation

/// Check if a scalar block has multiple tensor shapes (shape transitions).
/// Returns the list of (nodeIndex, shape) pairs where shape changes.
///
/// Regions can also be forced around convolution ops and tensor-to-scalar reductions to
/// preserve execution ordering requirements.
///
/// - Parameters:
///   - block: Block to inspect.
///   - g: Graph containing node shape metadata.
/// - Returns: Ordered transition points used by region planning.
func detectShapeTransitions(block: Block, g: Graph) -> [(nodeIndex: Int, shape: [Int])] {
  var transitions: [(nodeIndex: Int, shape: [Int])] = []
  var currentShape: [Int]? = nil

  for (index, nodeId) in block.nodes.enumerated() {
    guard let node = g.nodes[nodeId] else { continue }

    if case .tensor(let shape) = node.shape {
      // Skip view-only ops for shape transition detection.
      // These emit no compute code (just marker UOps rendered as comments
      // or set ctx.values). Letting them trigger region boundaries creates
      // empty element loops.
      if node.op.isViewOnly { continue }

      // Check for shape change
      var needsNewRegion = shape != currentShape

      // Conv2d/conv1d have global read patterns - they need ALL elements of their
      // input tensor to be computed before ANY conv2d output is computed.
      // Force a region boundary BEFORE conv2d so the input is complete.
      // Also, conv2d emits its own parallelRange internally, so it must be
      // in its own region (not mixed with other element-wise ops).
      switch node.op {
      case .conv2d, .conv1d:
        // Always start a new region for conv2d, even if shape matches
        needsNewRegion = true
      default:
        break
      }

      if needsNewRegion {
        transitions.append((nodeIndex: index, shape: shape))
        currentShape = shape
      }

      // Force region boundary AFTER conv2d - it has its own internal loop,
      // so subsequent ops must be in a separate region
      switch node.op {
      case .conv2d, .conv1d:
        // Mark that next tensor node needs new region even if same shape
        currentShape = nil
      default:
        break
      }
    } else if case .scalar = node.shape {
      // Scalar node - check if it consumes a tensor (e.g., sum reduction)
      // These need their own region AFTER the tensor region completes
      // BUT: only create a new region when transitioning FROM tensor TO scalar,
      // not for consecutive scalar operations (which may depend on each other)
      let consumesTensor = node.inputs.contains { inputId in
        if let inputNode = g.nodes[inputId], case .tensor = inputNode.shape {
          return true
        }
        return false
      }
      let alreadyInScalarRegion = currentShape == [1]
      if consumesTensor && currentShape != nil && !alreadyInScalarRegion {
        // Create a "scalar reduction" region with shape [1] to separate it
        // This ensures the tensor computation completes before reduction runs
        transitions.append((nodeIndex: index, shape: [1]))
        currentShape = [1]
      }
    }
  }
  return transitions
}

/// A pre-classified region within a shape-transition block.
/// Separates analysis from emission for clarity.
struct EmissionRegion {
  let scalarNodes: [NodeID]  // Emitted before element loop (no tensorIndex)
  let tensorNodes: [NodeID]  // Emitted inside element loop (with tensorIndex)
  let shape: [Int]  // Tensor shape for this region
  let isConvOnly: Bool  // Conv has own internal loop; skip element loop wrapper
  let isSkipped: Bool  // Matmul fusion; only emit tensorRef nodes
  let hopCounter: Lazy?  // Non-nil if region needs hop-gating
}

/// Merge all sources of outbound tensor cells for a shape-transition block:
/// block-level (cross-block), cross-region (within-block), and conv inputs.
///
/// - Parameters:
///   - block: Block being emitted.
///   - blocks: Full block list for cross-block dependency checks.
///   - g: Graph containing node/tensor metadata.
///   - transitions: Shape-region boundaries for this block.
/// - Returns: Unified outbound tensor-cell set for shape-aware emission.
func computeShapeAwareOutboundCells(
  block: Block, blocks: [Block], g: Graph,
  transitions: [(nodeIndex: Int, shape: [Int])]
) -> Set<CellID> {
  var outbound = findOutboundTensorCells(blocks, g, block: block)
  outbound.formUnion(findCrossRegionOutboundCells(block: block, g: g, transitions: transitions))

  // Conv ops use memoryRead() directly; input MUST be in memory, not registers.
  for nodeId in block.nodes {
    guard let node = g.nodes[nodeId] else { continue }
    switch node.op {
    case .conv2d, .conv1d:
      if let inputId = node.inputs.first,
        let tensorId = g.nodeToTensor[inputId],
        let tensor = g.tensors[tensorId]
      {
        outbound.insert(tensor.cellId)
      }
    default:
      break
    }
  }

  return outbound
}

/// Candidate for fusing an expand/mul region into the following axis-reduce region.
private struct FusableReduceCandidate {
  let transitionIndex: Int
  let reduceNodeId: NodeID
  let mulNodeId: NodeID
  let regionNodeSlice: ArraySlice<NodeID>
  let regionNodeSet: Set<NodeID>
  let intermediateCell: CellID
  let mulInputTensors: (Tensor, Tensor)
}

/// Returns true when an op is allowed in a skipped expand/matmul-prep region.
///
/// - Parameter op: Operation to classify.
/// - Returns: `true` if the op is metadata or reshape/expand plumbing and safe to skip.
private func isAllowedExpandRegionOp(_ op: LazyOp) -> Bool {
  switch op {
  case .tensorRef(_), .reshape(_), .transpose(_), .expand(_),
    .expandAxis(_, _), .expandView(_), .shrink(_), .constant(_):
    return true
  default:
    return false
  }
}

/// Finds the node range covered by a transition, excluding the next transition boundary.
///
/// - Parameters:
///   - transitionIndex: Region transition index.
///   - transitions: All shape transitions.
///   - block: Block containing the nodes.
/// - Returns: Node slice for the transition region, or `nil` when there is no following transition.
private func regionNodeSlice(
  transitionIndex: Int,
  transitions: [(nodeIndex: Int, shape: [Int])],
  block: Block
) -> ArraySlice<NodeID>? {
  let nextIdx = transitionIndex + 1
  guard nextIdx < transitions.count else { return nil }
  let start = transitions[transitionIndex].nodeIndex
  let end = transitions[nextIdx].nodeIndex
  return block.nodes[start..<end]
}

/// Finds the leading axis-reduce node of the next transition.
///
/// - Parameters:
///   - transitionIndex: Current transition index.
///   - transitions: All shape transitions.
///   - block: Block containing the nodes.
///   - g: Graph containing node definitions.
/// - Returns: Node ID of the next region's reduce op when it is axis-reduce.
private func nextAxisReduceNodeId(
  transitionIndex: Int,
  transitions: [(nodeIndex: Int, shape: [Int])],
  block: Block,
  g: Graph
) -> NodeID? {
  let nextIdx = transitionIndex + 1
  guard nextIdx < transitions.count else { return nil }
  let nodeId = block.nodes[transitions[nextIdx].nodeIndex]
  guard let node = g.nodes[nodeId], isAxisReduceOp(node.op) else { return nil }
  return nodeId
}

/// Extracts mul producer metadata for an axis-reduce, validating regional ownership.
///
/// - Parameters:
///   - reduceNodeId: Axis-reduce node ID.
///   - regionNodeSet: Node IDs belonging to the candidate expand region.
///   - blockOutbound: Tensor cells required by later blocks.
///   - g: Graph containing node/tensor metadata.
/// - Returns: Mul node ID, intermediate product cell, and mul input tensors when valid.
private func extractMulProducerInfo(
  reduceNodeId: NodeID,
  regionNodeSet: Set<NodeID>,
  blockOutbound: Set<CellID>,
  g: Graph
) -> (mulNodeId: NodeID, intermediateCell: CellID, mulInputTensors: (Tensor, Tensor))? {
  guard let reduceNode = g.nodes[reduceNodeId],
    let mulNodeId = reduceNode.inputs.first,
    let mulNode = g.nodes[mulNodeId],
    case .mul = mulNode.op,
    mulNode.inputs.count == 2,
    regionNodeSet.contains(mulNodeId)
  else { return nil }

  guard let intermediateTensorId = g.nodeToTensor[mulNodeId],
    let intermediateTensor = g.tensors[intermediateTensorId]
  else { return nil }
  let intermediateCell = intermediateTensor.cellId
  guard !blockOutbound.contains(intermediateCell) else { return nil }

  guard let aTensorId = g.nodeToTensor[mulNode.inputs[0]],
    let aTensor = g.tensors[aTensorId],
    let bTensorId = g.nodeToTensor[mulNode.inputs[1]],
    let bTensor = g.tensors[bTensorId]
  else { return nil }

  return (
    mulNodeId: mulNodeId,
    intermediateCell: intermediateCell,
    mulInputTensors: (aTensor, bTensor)
  )
}

/// Validates that all nodes in the region are safe expand/matmul-prep operations.
///
/// - Parameters:
///   - regionNodeSlice: Ordered nodes in the candidate region.
///   - mulNodeId: Mul node that may be fused into the following reduce.
///   - g: Graph containing node definitions.
/// - Returns: `true` when every non-mul op is allowed in skipped regions.
private func regionContainsOnlyExpandPrepOps(
  regionNodeSlice: ArraySlice<NodeID>,
  mulNodeId: NodeID,
  g: Graph
) -> Bool {
  regionNodeSlice.allSatisfy { nodeId in
    guard let node = g.nodes[nodeId] else { return true }
    if nodeId == mulNodeId { return true }
    return isAllowedExpandRegionOp(node.op)
  }
}

/// Checks if nodes in the candidate region are consumed outside the region and reduce.
///
/// - Parameters:
///   - regionNodeSet: Candidate region nodes.
///   - reduceNodeId: Reduce node allowed to consume region outputs.
///   - g: Graph containing full consumer relationships.
/// - Returns: `true` when an external consumer exists and fusion would be unsafe.
private func hasExternalConsumersOutsideRegion(
  regionNodeSet: Set<NodeID>,
  reduceNodeId: NodeID,
  g: Graph
) -> Bool {
  let allowedConsumers = regionNodeSet.union([reduceNodeId])
  let externallyVisibleNodes = regionNodeSet.filter { nodeId in
    guard let node = g.nodes[nodeId] else { return false }
    if case .tensorRef(_) = node.op { return false }
    return true
  }

  guard !externallyVisibleNodes.isEmpty else { return false }
  for (consumerId, consumerNode) in g.nodes {
    guard !allowedConsumers.contains(consumerId) else { continue }
    if consumerNode.inputs.contains(where: { externallyVisibleNodes.contains($0) }) {
      return true
    }
  }
  return false
}

/// Attempts to build a fusion candidate for one transition pair.
///
/// - Parameters:
///   - transitionIndex: Current transition index.
///   - block: Block under analysis.
///   - transitions: Shape transitions for this block.
///   - blockOutbound: Tensor cells that cannot be removed due to cross-block use.
///   - g: Graph containing node/tensor metadata.
/// - Returns: Candidate when all structural and safety checks pass; otherwise `nil`.
private func makeFusableReduceCandidate(
  transitionIndex: Int,
  block: Block,
  transitions: [(nodeIndex: Int, shape: [Int])],
  blockOutbound: Set<CellID>,
  g: Graph
) -> FusableReduceCandidate? {
  guard let regionSlice = regionNodeSlice(
    transitionIndex: transitionIndex, transitions: transitions, block: block),
    let reduceNodeId = nextAxisReduceNodeId(
      transitionIndex: transitionIndex, transitions: transitions, block: block, g: g)
  else { return nil }

  let regionSet = Set(regionSlice)
  guard let mulInfo = extractMulProducerInfo(
    reduceNodeId: reduceNodeId, regionNodeSet: regionSet, blockOutbound: blockOutbound, g: g)
  else { return nil }

  guard regionContainsOnlyExpandPrepOps(
    regionNodeSlice: regionSlice, mulNodeId: mulInfo.mulNodeId, g: g)
  else { return nil }

  guard !hasExternalConsumersOutsideRegion(
    regionNodeSet: regionSet, reduceNodeId: reduceNodeId, g: g)
  else { return nil }

  return FusableReduceCandidate(
    transitionIndex: transitionIndex,
    reduceNodeId: reduceNodeId,
    mulNodeId: mulInfo.mulNodeId,
    regionNodeSlice: regionSlice,
    regionNodeSet: regionSet,
    intermediateCell: mulInfo.intermediateCell,
    mulInputTensors: mulInfo.mulInputTensors
  )
}

/// Detect fusable expand->axis-reduce pairs where we can skip the expand region
/// and have sumAxis compute the product inline.
/// Returns the set of transition indices whose regions should be skipped.
///
/// This helper also removes the skipped intermediate tensor from outbound cells and records
/// inline reduce inputs in the emission context.
///
/// - Parameters:
///   - block: Block under analysis.
///   - g: Graph containing node/tensor metadata.
///   - transitions: Shape-region boundaries.
///   - blockOutbound: Cross-block outbound tensor cells.
///   - outbound: Current outbound set, updated in place when regions are skipped.
///   - ctx: Emission context receiving inline-reduce metadata.
/// - Returns: Region transition indices that should be skipped during emission.
func detectFusableReduces(
  block: Block, g: Graph,
  transitions: [(nodeIndex: Int, shape: [Int])],
  blockOutbound: Set<CellID>,
  outbound: inout Set<CellID>,
  ctx: IRContext
) -> Set<Int> {
  var skipRegions = Set<Int>()

  for transitionIndex in transitions.indices {
    guard let candidate = makeFusableReduceCandidate(
      transitionIndex: transitionIndex,
      block: block,
      transitions: transitions,
      blockOutbound: blockOutbound,
      g: g
    ) else { continue }

    // This region no longer needs materialization; sumAxis will compute A*B inline.
    // Record it in ctx.inlineableReduceInputs using the skipped product cell as lookup key.
    skipRegions.insert(candidate.transitionIndex)
    outbound.remove(candidate.intermediateCell)
    ctx.inlineableReduceInputs[candidate.intermediateCell] = candidate.mulInputTensors
  }

  return skipRegions
}

/// Convert transitions into structured EmissionRegions.
/// Includes a scalar preamble (nodes before first transition) as a region with shape [1].
///
/// - Parameters:
///   - block: Block being planned.
///   - g: Graph containing node definitions.
///   - ctx: Emission context used to resolve hop-based region metadata.
///   - transitions: Ordered shape transitions.
///   - skipRegions: Transition indices to mark as skipped (e.g., fused expand regions).
/// - Returns: Emission regions ready for `emitRegion`.
func buildRegions(
  block: Block, g: Graph, ctx: IRContext,
  transitions: [(nodeIndex: Int, shape: [Int])],
  skipRegions: Set<Int>
) -> [EmissionRegion] {
  var regions: [EmissionRegion] = []

  // Scalar preamble: nodes before the first transition (e.g., gradient div from mean backward)
  if let firstTransition = transitions.first, firstTransition.nodeIndex > 0 {
    let preambleNodes = Array(block.nodes[0..<firstTransition.nodeIndex])
    if !preambleNodes.isEmpty {
      regions.append(
        EmissionRegion(
          scalarNodes: preambleNodes, tensorNodes: [], shape: [1],
          isConvOnly: false, isSkipped: false, hopCounter: nil
        ))
    }
  }

  for (transitionIdx, transition) in transitions.enumerated() {
    let regionEnd =
      transitionIdx + 1 < transitions.count
      ? transitions[transitionIdx + 1].nodeIndex
      : block.nodes.count

    let nodeRange = transition.nodeIndex..<regionEnd

    if skipRegions.contains(transitionIdx) {
      // Skipped expand region; only tensorRef nodes matter.
      let tensorRefNodes = nodeRange.compactMap { idx -> NodeID? in
        let nodeId = block.nodes[idx]
        guard let node = g.nodes[nodeId], case .tensorRef(_) = node.op else { return nil }
        return nodeId
      }
      regions.append(
        EmissionRegion(
          scalarNodes: tensorRefNodes, tensorNodes: [], shape: transition.shape,
          isConvOnly: false, isSkipped: true, hopCounter: nil
        ))
      continue
    }

    // Classify nodes as scalar (emitted outside element loop) or tensor.
    // View-only ops (reshape, transpose, shrink) are tensor-shaped metadata that
    // set ctx.values; they must be emitted so downstream ops find their inputs.
    var scalarNodes: [NodeID] = []
    var tensorNodes: [NodeID] = []
    for idx in nodeRange {
      let nodeId = block.nodes[idx]
      guard let node = g.nodes[nodeId] else { continue }
      if case .tensor = node.shape {
        tensorNodes.append(nodeId)
      } else if !node.op.isViewOnly {
        scalarNodes.append(nodeId)
      }
    }

    // Check if this region contains only conv2d (which has its own parallelRange)
    let firstNodeId = block.nodes[transition.nodeIndex]
    let isConvOnly: Bool
    if let firstNode = g.nodes[firstNodeId] {
      switch firstNode.op {
      case .conv2d, .conv1d:
        isConvOnly = (regionEnd - transition.nodeIndex == 1)
      default:
        isConvOnly = false
      }
    } else {
      isConvOnly = false
    }

    // Check if this tensor region is hop-based in a frame-based block
    let hopCounter: Lazy?
    if block.temporality == .frameBased {
      hopCounter =
        nodeRange.lazy
        .compactMap { idx -> Lazy? in
          let nodeId = block.nodes[idx]
          guard let (_, counterNodeId) = ctx.hopBasedNodes[nodeId] else { return nil }
          return ctx.values[counterNodeId]
        }
        .first
    } else {
      hopCounter = nil
    }

    regions.append(
      EmissionRegion(
        scalarNodes: scalarNodes, tensorNodes: tensorNodes, shape: transition.shape,
        isConvOnly: isConvOnly, isSkipped: false, hopCounter: hopCounter
      ))
  }

  return regions
}

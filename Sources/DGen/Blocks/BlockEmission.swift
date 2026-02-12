/// Block emission: UOp generation, dependency analysis, SIMD upgrade, shape transitions, BPTT.
import Foundation

// MARK: - Dependency Analysis

private func findNodesWithOutboundDependencies(_ blks: [Block], _ g: Graph, block: Block)
  -> [NodeID]
{
  // Map node -> block index
  var nodeBlock = [NodeID: Int]()
  for (bidx, b) in blks.enumerated() {
    b.nodes.forEach { nid in
      if nodeBlock[nid] == nil {
        nodeBlock[nid] = bidx
      }
    }
  }

  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  var need: Set<NodeID> = []
  for (consumerIdx, b) in blks.enumerated() {
    if thisIdx == consumerIdx { continue }
    for nID in b.nodes {
      g.nodes[nID]!.allDependencies.forEach { dep in
        if let producerIdx = nodeBlock[dep], producerIdx == thisIdx {
          if let depNode = g.nodes[dep], case .seq = depNode.op {
            if let lastInput = depNode.inputs.last { need.insert(lastInput) }
          } else {
            need.insert(dep)
          }
        }
      }
    }
  }
  return need.sorted()  // Return sorted array for stable ordering
}

private func findNodesAsInboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> [NodeID] {
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  // Map node -> block index
  var nodeBlock = [NodeID: Int]()
  for (bidx, b) in blks.enumerated() {
    b.nodes.forEach { nid in
      if nodeBlock[nid] == nil {
        nodeBlock[nid] = bidx
      }
    }
  }

  var need: Set<NodeID> = []
  // Collect only dependencies produced in a different group
  for nID in block.nodes {
    g.nodes[nID]!.allDependencies.forEach { dep in
      if let prodIdx = nodeBlock[dep] {
        if prodIdx != thisIdx { need.insert(dep) }
      }
    }
  }
  return need.sorted()  // Return sorted array for stable ordering
}

/// Compute which tensor cells in this block need to be written to memory because
/// they're used by later blocks. Cells only used within this block stay in registers.
private func findOutboundTensorCells(_ blks: [Block], _ g: Graph, block: Block) -> Set<CellID> {
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  // Collect all tensor cells produced by nodes in this block
  var producedCells: Set<CellID> = []
  for nodeId in block.nodes {
    if let node = g.nodes[nodeId], case .tensor = node.shape {
      if let tensorId = g.nodeToTensor[nodeId], let tensor = g.tensors[tensorId] {
        producedCells.insert(tensor.cellId)
      }
    }
  }

  // Check which cells are consumed by later blocks
  var outboundCells: Set<CellID> = []
  for (blockIdx, b) in blks.enumerated() {
    if blockIdx <= thisIdx { continue }  // Only look at later blocks

    for nodeId in b.nodes {
      guard let node = g.nodes[nodeId] else { continue }

      // Check if this node reads from any of our produced cells
      for inputId in node.inputs {
        if let inputNode = g.nodes[inputId], case .tensor = inputNode.shape {
          if let tensorId = g.nodeToTensor[inputId], let tensor = g.tensors[tensorId] {
            if producedCells.contains(tensor.cellId) {
              outboundCells.insert(tensor.cellId)
            }
          }
        }

      }

      // Also check historyRead/historyWrite operations that reference tensor cells
      switch node.op {
      case .historyRead(let cellId):
        // historyRead's cellId is the history buffer - mark if produced here
        if producedCells.contains(cellId) {
          outboundCells.insert(cellId)
        }
      case .historyWrite(_):
        // historyWrite reads from its INPUT tensor, not the history cell parameter
        // If the input tensor's cell was produced in this block, mark it as outbound
        for inputId in node.inputs {
          if let inputNode = g.nodes[inputId], case .tensor = inputNode.shape {
            if let tensorId = g.nodeToTensor[inputId], let tensor = g.tensors[tensorId] {
              if producedCells.contains(tensor.cellId) {
                outboundCells.insert(tensor.cellId)
              }
            }
          }
        }
      default:
        break
      }
    }
  }

  return outboundCells
}

/// Find tensor cells that cross shape region boundaries within a scalar block.
/// These must be written to memory (not kept in registers) because they're computed
/// in one loop and consumed in a different loop.
private func findCrossRegionOutboundCells(
  block: Block, g: Graph, transitions: [(nodeIndex: Int, shape: [Int])]
) -> Set<CellID> {
  guard !transitions.isEmpty else { return [] }

  var outbound: Set<CellID> = []

  // Build map: nodeId -> regionIndex
  var nodeToRegion: [NodeID: Int] = [:]
  for (regionIdx, transition) in transitions.enumerated() {
    let regionEnd =
      regionIdx + 1 < transitions.count
      ? transitions[regionIdx + 1].nodeIndex
      : block.nodes.count
    for nodeIndex in transition.nodeIndex..<regionEnd {
      nodeToRegion[block.nodes[nodeIndex]] = regionIdx
    }
  }

  // For each node, check if any of its inputs come from a different region
  for nodeId in block.nodes {
    guard let node = g.nodes[nodeId] else { continue }
    let myRegion = nodeToRegion[nodeId]  // May be nil for scalar nodes

    for inputId in node.inputs {
      let inputRegion = nodeToRegion[inputId]

      // Case 1: Both have regions and they differ (cross-region)
      // Case 2: Node is scalar (no region) but input has a region (tensor → scalar)
      let crossesRegion =
        (myRegion != nil && inputRegion != nil && myRegion != inputRegion)
        || (myRegion == nil && inputRegion != nil)

      guard crossesRegion else { continue }

      // This input crosses a region boundary - its cell must be outbound
      if let tensorId = g.nodeToTensor[inputId],
        let tensor = g.tensors[tensorId]
      {
        outbound.insert(tensor.cellId)
      }
    }
  }

  return outbound
}

// MARK: - SIMD Analysis

/// Check if any UOps contain patterns that prevent SIMD optimization:
/// - Inner loops (beginLoop, beginForLoop)
/// - View operations (reshape, transpose, shrink) that require complex index arithmetic (C only)
/// - Broadcast access (non-contiguous strides or shape mismatch) (C only)
private func containsSIMDBlockers(_ uops: [UOp], backend: Backend) -> Bool {
  for uop in uops {
    switch uop.op {
    case .beginLoop, .beginForLoop, .beginReverseLoop:
      return true
    case .reshape, .transpose, .shrink, .pad:
      // Metal handles these fine with per-thread execution
      if case .c = backend { return true }
    case .broadcastAccess:
      // Metal handles broadcast access fine with per-thread execution
      if case .c = backend { return true }
    default:
      break
    }
  }
  return false
}

/// Post-emission pass: upgrade eligible element loops to SIMD.
///
/// Handles both loop types:
/// - `beginParallelRange(count, 1)` / `endParallelRange` — standard tensor element loops
///   (forced to scalar by the pipeline when the block has SIMD blockers elsewhere)
/// - `beginForLoop(loopVar, count)` / `endLoop` — shape-transition inner element loops
///
/// Only for C backend. The renderer already supports SIMD: `beginParallelRange` with incr=4
/// emits `for (int simdN = 0; simdN < count; simdN += 4)`, and `memoryRead` with `.simd` kind
/// emits `vld1q_f32`. This pass identifies contiguous element loops that can use SIMD.
///
/// Eligibility: element count divisible by 4, no stateful/control-flow/non-contiguous ops,
/// no int-typed arithmetic (which indicates index decomposition or frame-aware offsets).
public func upgradeElementLoopsToSIMD(_ uops: inout [UOp]) {
  var i = 0
  while i < uops.count {
    // Match both loop types and extract element count + loop variable
    let elementCount: Int
    let loopVar: Lazy
    let isParallelRange: Bool

    switch uops[i].op {
    case .beginParallelRange(let count, _):
      elementCount = count
      loopVar = uops[i].value
      isParallelRange = true

    case .beginForLoop(let lv, let countLazy):
      guard case .constant(_, let countFloat) = countLazy else {
        i += 1
        continue
      }
      elementCount = Int(countFloat)
      loopVar = lv
      isParallelRange = false

    default:
      i += 1
      continue
    }

    // Must be divisible by 4
    guard elementCount >= 4 && elementCount % 4 == 0 else {
      i += 1
      continue
    }

    // Find matching end (track nesting depth)
    let beginIdx = i
    var depth = 1
    var endIdx: Int? = nil
    for j in (beginIdx + 1)..<uops.count {
      switch uops[j].op {
      case .beginForLoop, .beginLoop, .beginReverseLoop, .beginParallelRange:
        depth += 1
      case .endLoop, .endParallelRange:
        depth -= 1
        if depth == 0 {
          endIdx = j
        }
      default:
        break
      }
      if endIdx != nil { break }
    }

    guard let endIdx = endIdx else {
      i += 1
      continue
    }

    // Extract loop variable's VarID for offset matching
    let loopVarId: VarID?
    if case .variable(let vid, _) = loopVar {
      loopVarId = vid
    } else {
      loopVarId = nil
    }

    // Check for blocker UOps in the loop body
    var hasBlocker = false
    for k in (beginIdx + 1)..<endIdx {
      switch uops[k].op {
      // Stateful ops
      case .load, .store, .delay1, .memoryAccumulate:
        hasBlocker = true
      // Inherently scalar ops
      case .noise, .latch:
        hasBlocker = true
      // Control flow (nested loops, conditionals)
      case .beginForLoop, .endLoop, .beginParallelRange, .endParallelRange,
        .beginIf, .endIf, .gswitch:
        hasBlocker = true
      // Non-contiguous access
      case .broadcastAccess:
        hasBlocker = true
      // Hop gating
      case .beginHopCheck, .endHopCheck:
        hasBlocker = true
      // Accumulator (mutate is used by sum reduction — cross-iteration dependency)
      case .mutate:
        hasBlocker = true
      // Int-typed arithmetic (index decomposition / frame-aware offsets)
      case .add, .sub, .mul, .div:
        if uops[k].scalarType == .int {
          hasBlocker = true
        }
      // Memory access with non-loop-variable offset — scalar variable would be
      // broadcast to float32x4_t in SIMD mode, which can't be cast to int for indexing
      case .memoryRead(_, let offset), .memoryWrite(_, let offset, _):
        if case .variable(let vid, _) = offset, vid != loopVarId {
          hasBlocker = true
        }
      default:
        break
      }
      if hasBlocker { break }
    }

    guard !hasBlocker else {
      i = endIdx + 1
      continue
    }

    // Upgrade to SIMD
    uops[beginIdx] = UOp(
      op: .beginParallelRange(elementCount, 4),
      value: loopVar,
      kind: .simd
    )

    for k in (beginIdx + 1)..<endIdx {
      uops[k].kind = .simd
    }

    if isParallelRange {
      uops[endIdx].kind = .simd
    } else {
      // Replace endLoop → endParallelRange
      uops[endIdx] = UOp(op: .endParallelRange, value: uops[endIdx].value, kind: .simd)
    }

    i = endIdx + 1
  }
}

// MARK: - Shape Transitions

/// Check if a scalar block has multiple tensor shapes (shape transitions)
/// Returns the list of (nodeIndex, shape) pairs where shape changes
private func detectShapeTransitions(block: Block, g: Graph) -> [(nodeIndex: Int, shape: [Int])] {
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

// MARK: - Shape-Aware Emission

/// A pre-classified region within a shape-transition block.
/// Separates analysis from emission for clarity.
private struct EmissionRegion {
  let scalarNodes: [NodeID]  // Emitted before element loop (no tensorIndex)
  let tensorNodes: [NodeID]  // Emitted inside element loop (with tensorIndex)
  let shape: [Int]  // Tensor shape for this region
  let isConvOnly: Bool  // Conv has own internal loop — skip element loop wrapper
  let isSkipped: Bool  // Matmul fusion — only emit tensorRef nodes
  let hopCounter: Lazy?  // Non-nil if region needs hop-gating
}

/// Merge all sources of outbound tensor cells for a shape-transition block:
/// block-level (cross-block), cross-region (within-block), and conv inputs.
private func computeShapeAwareOutboundCells(
  block: Block, blocks: [Block], g: Graph,
  transitions: [(nodeIndex: Int, shape: [Int])]
) -> Set<CellID> {
  var outbound = findOutboundTensorCells(blocks, g, block: block)
  outbound.formUnion(findCrossRegionOutboundCells(block: block, g: g, transitions: transitions))

  // Conv ops use memoryRead() directly — input MUST be in memory, not registers
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

/// Detect fusable expand→axis-reduce pairs where we can skip the expand region
/// and have sumAxis compute the product inline.
/// Returns the set of transition indices whose regions should be skipped.
private func detectFusableReduces(
  block: Block, g: Graph,
  transitions: [(nodeIndex: Int, shape: [Int])],
  blockOutbound: Set<CellID>,
  outbound: inout Set<CellID>,
  ctx: IRContext
) -> Set<Int> {
  var skipRegions = Set<Int>()

  for (idx, transition) in transitions.enumerated() {
    let nextIdx = idx + 1
    guard nextIdx < transitions.count else { continue }

    let nextTransition = transitions[nextIdx]
    let regionEnd = nextTransition.nodeIndex

    // Next region must start with an axis reduce
    let nextNodeId = block.nodes[nextTransition.nodeIndex]
    guard let nextNode = g.nodes[nextNodeId],
      isAxisReduceOp(nextNode.op)
    else { continue }

    // Axis reduce's input must be a mul in this region
    let mulNodeId = nextNode.inputs[0]
    guard let mulNode = g.nodes[mulNodeId],
      case .mul = mulNode.op,
      mulNode.inputs.count == 2
    else { continue }
    let regionSlice = block.nodes[transition.nodeIndex..<regionEnd]
    let regionSet = Set(regionSlice)
    guard regionSet.contains(mulNodeId) else { continue }

    // Get the intermediate tensor (mul's output)
    // Only fuse if it's not needed by later blocks (block-level outbound)
    guard let intermediateTensorId = g.nodeToTensor[mulNodeId],
      let intermediateTensor = g.tensors[intermediateTensorId]
    else { continue }
    let intermediateCell = intermediateTensor.cellId
    guard !blockOutbound.contains(intermediateCell) else { continue }

    // Get mul's input tensors for inline computation
    guard let aTensorId = g.nodeToTensor[mulNode.inputs[0]],
      let aTensor = g.tensors[aTensorId],
      let bTensorId = g.nodeToTensor[mulNode.inputs[1]],
      let bTensor = g.tensors[bTensorId]
    else { continue }

    // Only skip if ALL nodes in the region are matmul-expand related
    // AND no node in the region is consumed by anything outside the region + sumAxis.
    let allExpandRelated = regionSlice.allSatisfy { nid in
      guard let n = g.nodes[nid] else { return true }
      if nid == mulNodeId { return true }
      switch n.op {
      case .tensorRef(_), .reshape(_), .transpose(_), .expand,
        .expandAxis(_, _), .expandView, .shrink(_), .constant(_):
        return true
      default:
        return false
      }
    }
    guard allExpandRelated else { continue }

    // Check that no non-tensorRef node in the region is consumed outside the region.
    // Must scan ALL graph nodes since ctx.values is shared across blocks.
    let safeConsumers = regionSet.union([nextNodeId])  // region nodes + the sumAxis
    let skippableNodeIds = regionSet.filter { nid in
      if let n = g.nodes[nid], case .tensorRef(_) = n.op { return false }
      return true
    }
    var hasExternalConsumers = false
    for (consumerId, consumerNode) in g.nodes {
      guard !safeConsumers.contains(consumerId) else { continue }
      if consumerNode.inputs.contains(where: { skippableNodeIds.contains($0) }) {
        hasExternalConsumers = true
        break
      }
    }
    guard !hasExternalConsumers else { continue }

    skipRegions.insert(idx)
    outbound.remove(intermediateCell)
    ctx.inlineableReduceInputs[intermediateCell] = (aTensor, bTensor)
  }

  return skipRegions
}

/// Convert transitions into structured EmissionRegions.
/// Includes a scalar preamble (nodes before first transition) as a region with shape [1].
private func buildRegions(
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
      // Skipped expand region — only tensorRef nodes matter
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
    // set ctx.values — they must be emitted so downstream ops find their inputs.
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

/// Emit UOps for a single EmissionRegion.
private func emitRegion(
  _ region: EmissionRegion, ctx: IRContext, g: Graph
) throws -> [UOp] {
  var uops: [UOp] = []

  if let counterLazy = region.hopCounter {
    uops.append(UOp(op: .beginHopCheck(counterLazy), value: .empty))
  }

  // Emit scalar nodes (no tensorIndex, kind: .scalar)
  for nodeId in region.scalarNodes {
    guard let node = g.nodes[nodeId] else { continue }
    for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
      var typedUop = uop
      typedUop.kind = .scalar
      uops.append(typedUop)
    }
  }

  // Emit tensor nodes
  if !region.tensorNodes.isEmpty && !region.isSkipped {
    let elemVar = ctx.useVariable(src: nil)
    let elementCount = region.shape.reduce(1, *)

    if !region.isConvOnly {
      var beginLoop = UOp(
        op: .beginForLoop(elemVar, .constant(0, Float(elementCount))),
        value: elemVar
      )
      beginLoop.kind = .scalar
      uops.append(beginLoop)
    }

    for nodeId in region.tensorNodes {
      ctx.tensorIndices[nodeId] = elemVar
      if let node = g.nodes[nodeId] {
        for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
          var typedUop = uop
          typedUop.kind = .scalar
          uops.append(typedUop)
        }
      }
    }

    if !region.isConvOnly {
      var endLoop = UOp(op: .endLoop, value: .empty)
      endLoop.kind = .scalar
      uops.append(endLoop)
    }
  }

  if region.hopCounter != nil {
    uops.append(UOp(op: .endHopCheck, value: .empty))
  }

  ctx.clearTensorRegisters()
  return uops
}

/// Emit UOps for a scalar block with shape transitions (e.g., conv2d in feedback loop).
/// Instead of one flat loop, emits nested loops: outer frame loop + inner element loops.
public func emitScalarBlockWithShapeTransitions(
  ctx: IRContext, block: Block, blocks: [Block], g: Graph,
  transitions: [(nodeIndex: Int, shape: [Int])]
) throws -> [UOp] {
  ctx.inlineableReduceInputs = [:]

  // Analysis: compute outbound cells and detect fusable reduces
  let blockOutbound = findOutboundTensorCells(blocks, g, block: block)
  var outbound = computeShapeAwareOutboundCells(
    block: block, blocks: blocks, g: g, transitions: transitions)
  let skipRegions = detectFusableReduces(
    block: block, g: g, transitions: transitions,
    blockOutbound: blockOutbound, outbound: &outbound, ctx: ctx)
  ctx.outboundTensorCells = outbound
  ctx.clearTensorRegisters()

  // Build structured regions from transitions
  let regions = buildRegions(
    block: block, g: g, ctx: ctx,
    transitions: transitions, skipRegions: skipRegions)

  // Emit each region
  var uops: [UOp] = []
  for region in regions {
    uops += try emitRegion(region, ctx: ctx, g: g)
  }
  return uops
}

// MARK: - Thread Count Scale

public func emitThreadCountScaleOpIfNeeded(ctx: IRContext, block: Block, g: Graph) -> [UOp] {
  guard let shape = block.shape else { return [] }

  let tensorSize = shape.reduce(1, *)

  var uops: [UOp] = []
  let setup = IRBuilder(ctx: ctx, nodeId: block.nodes[block.nodes.count - 1])
  let (frameIdx, binIdx) = setup.setupFlatThreading(tensorSize: tensorSize)
  uops.append(contentsOf: setup.ops)

  for nodeId in block.nodes {
    ctx.tensorIndices[nodeId] = binIdx.lazy
  }

  // Always set the decomposed frame index so currentFrameIndex() returns
  // the correct value in ThreadCountScale blocks (needed by slidingWindow
  // bounds checks, frame-aware tensor reads, etc.)
  ctx.frameAwareTensorFrameIndex = frameIdx.lazy
  ctx.frameAwareTensorElementIndex = binIdx.lazy

  // If this is a frame-based or hop-based block with frame-aware tensor outputs,
  // also set the flag that controls frame-aware memory addressing
  let isParallelFrameBlock: Bool
  if case .hopBased = block.temporality {
    isParallelFrameBlock = true
  } else {
    isParallelFrameBlock = block.temporality == .frameBased
  }
  if isParallelFrameBlock {
    let hasFrameAwareOutput = block.nodes.contains { nodeId in
      if let tensorId = g.nodeToTensor[nodeId],
        let tensor = g.tensors[tensorId]
      {
        return ctx.frameAwareTensorCells.contains(tensor.cellId)
      }
      return false
    }
    if hasFrameAwareOutput {
      ctx.isInFrameAwareTensorBlock = true
    }
  }

  return uops
}

// MARK: - Main Block Emission

public func emitBlockUOps(
  ctx: IRContext, block: Block, blocks: [Block], g: Graph, backend: Backend = .metal,
  debug: Bool = false
) throws -> (uops: [UOp], effectiveKind: Kind) {
  var emittedNodes: Set<NodeID> = []
  var bodyUops: [UOp] = []

  // Reset frame-aware tensor block context for each new block
  // These flags are set per-block in emitThreadCountScaleOpIfNeeded
  ctx.isInFrameAwareTensorBlock = false
  ctx.frameAwareTensorFrameIndex = nil
  ctx.frameAwareTensorElementIndex = nil

  // Tensor Register Optimization:
  // Compute which tensor cells need to be written to memory (used by later blocks)
  // and clear the register tracking for this new block.
  var outboundCells = findOutboundTensorCells(blocks, g, block: block)

  // Check for scalar blocks with shape transitions (e.g., conv2d in feedback loops)
  // These need nested element loops instead of flat threading
  let shapeTransitions = detectShapeTransitions(block: block, g: g)
  let hasMultipleShapes = shapeTransitions.count > 1

  // For ALL backends: include cross-region outbound cells (tensor → scalar reductions)
  // This ensures tensors are written to memory before scalar reductions read them
  if hasMultipleShapes {
    let crossRegion = findCrossRegionOutboundCells(
      block: block, g: g, transitions: shapeTransitions)
    outboundCells.formUnion(crossRegion)
  }

  // Mark conv2d/conv1d input tensors as outbound - they use memoryRead() directly
  // instead of tload(), so the input MUST be in memory not just in registers
  for nodeId in block.nodes {
    guard let node = g.nodes[nodeId] else { continue }
    switch node.op {
    case .conv2d, .conv1d:
      if let inputId = node.inputs.first,
        let tensorId = g.nodeToTensor[inputId],
        let tensor = g.tensors[tensorId]
      {
        outboundCells.insert(tensor.cellId)
      }
    default:
      break
    }
  }

  ctx.outboundTensorCells = outboundCells
  ctx.clearTensorRegisters()

  // Shape-aware emission for blocks with shape transitions
  // Both Metal and C need this when shapes change (e.g., matmul [M,K] @ [K,N] -> [M,N,K] product)
  // Without this, the outer loop uses the wrong shape, producing incorrect results
  // Triggers for any block with multiple shapes — not just scalar blocks — so that
  // fused axis reduces (which keep their block SIMD) still get per-region loops.
  let useShapeAwareEmission = hasMultipleShapes

  if useShapeAwareEmission {
    // Use specialized emission with per-shape element loops
    let shapeAwareUOps = try emitScalarBlockWithShapeTransitions(
      ctx: ctx, block: block, blocks: blocks, g: g, transitions: shapeTransitions
    )
    // Mark nodes as emitted
    for nodeId in block.nodes {
      emittedNodes.insert(nodeId)
    }
    bodyUops.append(contentsOf: shapeAwareUOps)
  } else {
    // Standard emission path
    // Thread count scaling is a Metal-specific parallelization optimization.
    // C backend uses sequential loops, so this would break feedback loop data flow.
    let threadScaleUOps =
      (backend == .metal)
      ? emitThreadCountScaleOpIfNeeded(ctx: ctx, block: block, g: g)
      : []
    bodyUops.append(contentsOf: threadScaleUOps)
    let emittedThreadScale = !threadScaleUOps.isEmpty

    // Track UOp boundary between forward and backward nodes for BPTT
    var backwardUOpsStartIndex: Int? = nil
    let lastForwardId = g.lastForwardNodeId

    for nodeId in block.nodes {
      if !emittedThreadScale, let tensorIndex = block.tensorIndex {
        // Don't give tensorIndex to inherently scalar stateful ops (accum, phasor, etc.)
        // These have single-cell state and their own scalar emit path. Giving them a
        // tensor index causes indexed memory access on single-cell state:
        // memory[cell + idx] for idx=0..N corrupts adjacent memory.
        let isScalarOp = g.nodes[nodeId]?.op.isInherentlyScalar ?? false
        if !isScalarOp {
          ctx.tensorIndices[nodeId] = tensorIndex
        }
      }

      // Track where backward UOps start
      if let lastFwd = lastForwardId, nodeId > lastFwd, backwardUOpsStartIndex == nil {
        backwardUOpsStartIndex = bodyUops.count
      }

      if let node = g.nodes[nodeId] {
        for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
          emittedNodes.insert(nodeId)
          bodyUops.append(uop)
        }
      }
    }

    // BPTT: If block has gradient carry cells and both forward+backward nodes,
    // split into forward loop (0→N-1) and reverse backward loop (N-1→0)
    //
    // Only activate when historyWrite is pass-through (its output is consumed by
    // other nodes). When historyWrite output is discarded (old API), BPTT doesn't
    // apply because historyWrite isn't on the loss path.
    if let backwardStart = backwardUOpsStartIndex {
      // Check if this block contains a pass-through historyWrite with carry cells.
      // Only activate BPTT when historyWrite's output is consumed by other nodes
      // (the pass-through pattern from Signal.history()). When historyWrite's
      // output is discarded (old API), BPTT doesn't apply.
      let blockHasPassThroughHistoryWrite = block.nodes.contains { nodeId in
        guard let node = g.nodes[nodeId] else { return false }
        guard case .historyWrite(let cellId) = node.op else { return false }
        guard g.gradCarryCells[cellId] != nil else { return false }
        // Check if any history node exists for this cellId (not tensor grad cells)
        let isHistoryCell = g.nodes.values.contains { n in
          if case .historyRead(let c) = n.op, c == cellId { return true }
          return false
        }
        guard isHistoryCell else { return false }
        // Check if historyWrite's output is consumed (pass-through)
        return g.nodes.values.contains { other in
          other.inputs.contains(nodeId)
        }
      }

      if blockHasPassThroughHistoryWrite {
        bodyUops = try wrapWithBPTTLoops(
          bodyUops: bodyUops,
          backwardStartIndex: backwardStart,
          block: block,
          g: g,
          ctx: ctx
        )
        ctx.lastBlockHasOwnFrameLoop = true
      }
    }
  }

  // Step 2: Analyze emitted UOps to determine if SIMD is safe
  // SIMD is safe if: tensor block + size divisible by 4 + no SIMD blockers + not frame-based
  // Note: frame-aware tensor blocks already handle parallelism via flat threading
  let hasSIMDBlockers = containsSIMDBlockers(bodyUops, backend: backend)
  let canUseSIMD: Bool
  let simdIncrement: Int

  if let shape = block.shape, block.tensorIndex != nil {
    let size = shape.reduce(1, *)
    // Frame-based tensor blocks must run element-by-element per frame
    // because their values change every frame (e.g., downstream of phasor(tensor))
    let isFrameBased = block.temporality == .frameBased
    canUseSIMD = !hasSIMDBlockers && !isFrameBased && (size % 4 == 0)
    simdIncrement = canUseSIMD ? 4 : 1
  } else {
    canUseSIMD = false
    simdIncrement = 1
  }

  // Step 3: Determine the effective kind for this block's ops
  let effectiveKind: Kind
  if block.tensorIndex != nil {
    effectiveKind = canUseSIMD ? .simd : .scalar
  } else {
    effectiveKind = block.kind
  }

  // Step 4: Apply the kind to all body UOps
  for i in 0..<bodyUops.count {
    bodyUops[i].kind = effectiveKind
  }

  // Step 5: Build final UOps array with parallelRange wrapper if needed
  // Note: frame-aware tensor blocks DON'T use parallelRange (no loop)
  var uops: [UOp] = []

  // C backend wraps tensor ops in a sequential loop
  // Skip if using shape-aware emission (it has its own loops)
  let needsTensorLoop = (backend == .c) && !useShapeAwareEmission
  if needsTensorLoop, let tensorIndex = block.tensorIndex,
    let shape = block.shape
  {
    let count = shape.reduce(1, *)
    var loopUOp = UOp(op: .beginParallelRange(count, simdIncrement), value: tensorIndex)
    loopUOp.kind = effectiveKind  // Match loop wrapper kind to body operations
    uops.append(loopUOp)
  }

  uops.append(contentsOf: bodyUops)

  // Handle cross-block dependencies using scratch buffers (for scalar values only)
  //
  // IMPORTANT: Tensor-valued outputs/inputs do NOT use scratch buffers.
  //
  // Why? Scratch buffers are indexed by frame (t<id>[i]), but tensor operations
  // run inside parallel loops where each tensor element has a different value.
  // If we wrote tensor results to scratch buffers inside a tensor loop:
  //   - Each iteration would overwrite the same t<id>[i] location
  //   - Only the LAST tensor element's value would survive
  //   - Reading it back and broadcasting would give wrong values (noise!)
  //
  // Instead, tensor data flows through memory cells which ARE properly indexed
  // by the tensor parallel range index (memory[cellId + tensorIndex]).

  let outbound = findNodesWithOutboundDependencies(blocks, g, block: block)
  for nodeId in outbound {
    if emittedNodes.contains(nodeId) {
      // Skip defineGlobal for tensor-valued outputs - they use memory cells, not scratch buffers
      if let node = g.nodes[nodeId], case .tensor = node.shape {
        continue
      }

      if let lz = ctx.values[nodeId] {
        switch lz {
        case .variable(let a, _):
          var defineGlobalUOp = UOp(op: .defineGlobal(a), value: .global(a))
          // Use block.kind (frame loop kind), not effectiveKind (tensor loop kind)
          // Globals are indexed by frame loop, not tensor loop
          defineGlobalUOp.kind = block.kind
          uops.insert(defineGlobalUOp, at: 0)
          // Only append if not already in globals to maintain stable ordering
          if !ctx.globals.contains(a) {
            ctx.globals.append(a)
          }
        default:
          break
        }
      }
    }
  }

  let inbound = findNodesAsInboundDependencies(blocks, g, block: block)

  for nodeId in inbound {
    if let lz = ctx.values[nodeId] {
      switch lz {
      case .variable(let a, _):
        // Skip variables without defineGlobal (tensor-valued nodes use memory cells instead)
        guard ctx.globals.contains(a) else { continue }

        var loadGlobalUOp = UOp(op: .loadGlobal(a), value: .variable(a, nil))
        // Globals are indexed by frame loop, not tensor loop
        loadGlobalUOp.kind = block.kind
        uops.insert(loadGlobalUOp, at: 0)
      default:
        break
      }
    }
  }

  // Close the tensor loop for C backend and hop-based Metal blocks
  if needsTensorLoop, block.tensorIndex != nil {
    uops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
  }
  return (uops: uops, effectiveKind: effectiveKind)
}

// MARK: - BPTT

/// Wraps emitted UOps with forward/reverse loops for correct BPTT.
///
/// Forward nodes run in a forward loop (0→N-1), backward nodes run in a reverse loop (N-1→0).
/// Forward values needed by backward ops are stored per-frame during the forward pass
/// and loaded during the backward pass.
///
/// - Parameters:
///   - bodyUops: All emitted UOps (forward + backward) in order
///   - backwardStartIndex: Index in bodyUops where backward UOps begin
///   - block: The block being emitted
///   - g: The computation graph
///   - ctx: IR context for allocating variables and cells
/// - Returns: Wrapped UOps with forward loop, per-frame storage, and reverse backward loop
private func wrapWithBPTTLoops(
  bodyUops: [UOp],
  backwardStartIndex: Int,
  block: Block,
  g: Graph,
  ctx: IRContext
) throws -> [UOp] {
  let lastForwardId = g.lastForwardNodeId!
  let forwardUops = Array(bodyUops[0..<backwardStartIndex])
  let backwardUops = Array(bodyUops[backwardStartIndex...])

  // Identify forward node values that backward nodes need.
  // These must be stored per-frame during the forward pass and loaded during the backward pass.
  let forwardNodeIds = Set(block.nodes.filter { $0 <= lastForwardId })
  let backwardNodeIds = Set(block.nodes.filter { $0 > lastForwardId })

  var forwardValuesNeeded = Set<NodeID>()
  for nodeId in backwardNodeIds {
    guard let node = g.nodes[nodeId] else { continue }
    for inputId in node.allDependencies {
      if forwardNodeIds.contains(inputId) {
        forwardValuesNeeded.insert(inputId)
      }
    }
  }

  // Allocate per-frame storage cells for forward values.
  // Each cell stores one float per frame.
  var perFrameCells: [NodeID: CellID] = [:]
  for nodeId in forwardValuesNeeded {
    guard let lz = ctx.values[nodeId] else { continue }
    // Only store variables (not constants or empty values)
    switch lz {
    case .variable:
      let cell = g.alloc()
      perFrameCells[nodeId] = cell
      // Register allocation size = maxFrameCount
      g.cellAllocationSizes[cell] = g.maxFrameCount
    default:
      break
    }
  }

  // BPTT carry cells: identify carry cell reads that backward UOps reference.
  // The carry cell memoryRead is emitted in a separate constants block (static, once),
  // but for BPTT it must be re-read each reverse iteration. We emit a fresh read
  // inside the reverse loop and remap backward UOps to use it.
  //
  // Also identify carry cell writes (gradient side effects from historyRead.backward)
  // that need to be emitted inside the reverse loop.
  var carryCellReads: [(cellId: CellID, originalLazy: Lazy)] = []
  var carryCellWriteNodes: [NodeID] = []

  // Find carry cell reads referenced by backward nodes but emitted outside this block
  let carryCellIds = Set(g.gradCarryCells.values)
  for nodeId in g.nodes.keys where !block.nodes.contains(nodeId) {
    guard let node = g.nodes[nodeId] else { continue }
    if case .memoryRead(let cell) = node.op, carryCellIds.contains(cell) {
      // This is a carry cell read node (outside this block)
      if let lz = ctx.values[nodeId] {
        // Check if any backward node in this block uses this value
        let usedInBlock = backwardNodeIds.contains { bNodeId in
          guard let bNode = g.nodes[bNodeId] else { return false }
          return bNode.allDependencies.contains(nodeId)
        }
        if usedInBlock {
          carryCellReads.append((cellId: cell, originalLazy: lz))
        }
      }
    }
  }

  // Find carry cell write nodes by scanning all backward nodes in the graph
  for (nodeId, node) in g.nodes {
    guard nodeId > lastForwardId else { continue }
    if case .memoryWrite(let cell) = node.op, carryCellIds.contains(cell) {
      carryCellWriteNodes.append(nodeId)
    }
  }

  // Build the wrapped UOp sequence
  var result: [UOp] = []

  // frameCount variable (sentinel -1, rendered as the frameCount parameter)
  let frameCountVar = Lazy.variable(-1, nil)

  // === Forward loop: for (uint i = 0; i < frameCount; i++) ===
  var beginFwd = UOp(op: .beginLoop(frameCountVar, 1), value: .empty)
  beginFwd.kind = .scalar
  result.append(beginFwd)

  // Forward UOps
  result.append(contentsOf: forwardUops)

  // Per-frame stores: save forward values for the backward pass
  for (nodeId, cell) in perFrameCells.sorted(by: { $0.key < $1.key }) {
    guard let lz = ctx.values[nodeId] else { continue }
    // Emit: memoryWrite(cell, frameIndex, value)
    let frameIdxVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .frameIndex, value: frameIdxVar, kind: .scalar, scalarType: .int))
    let storeVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .memoryWrite(cell, frameIdxVar, lz), value: storeVar, kind: .scalar))
  }

  // End forward loop
  var endFwd = UOp(op: .endLoop, value: .empty)
  endFwd.kind = .scalar
  result.append(endFwd)

  // === Backward (reverse) loop: for (int i = frameCount - 1; i >= 0; i--) ===
  var beginBwd = UOp(op: .beginReverseLoop(frameCountVar), value: .empty)
  beginBwd.kind = .scalar
  result.append(beginBwd)

  // Value remapping for backward UOps
  var valueRemapping: [Lazy: Lazy] = [:]

  // Per-frame loads: restore forward values for the backward pass
  for (nodeId, cell) in perFrameCells.sorted(by: { $0.key < $1.key }) {
    guard let originalLz = ctx.values[nodeId] else { continue }
    // Emit: loadedValue = memoryRead(cell, frameIndex)
    let frameIdxVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .frameIndex, value: frameIdxVar, kind: .scalar, scalarType: .int))
    let loadedVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .memoryRead(cell, frameIdxVar), value: loadedVar, kind: .scalar))
    valueRemapping[originalLz] = loadedVar
  }

  // BPTT carry cell reads: emit fresh reads inside the reverse loop
  // so carry values are re-read each iteration (not static from constants block)
  let zeroConst = ctx.useConstant(src: nil, value: 0.0)
  for carry in carryCellReads {
    let carryReadVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .memoryRead(carry.cellId, zeroConst), value: carryReadVar, kind: .scalar))
    valueRemapping[carry.originalLazy] = carryReadVar
  }

  // Backward UOps with remapped references.
  // Filter out carry cell writes — they're emitted explicitly below with proper remapping.
  let carryCellWriteSet = Set(
    carryCellWriteNodes.compactMap { g.nodes[$0] }.compactMap { node -> CellID? in
      if case .memoryWrite(let cell) = node.op, carryCellIds.contains(cell) { return cell }
      return nil
    })
  for uop in backwardUops {
    // Skip carry cell write UOps (will be re-emitted with BPTT remapping below)
    if case .memoryWrite(let cell, _, _) = uop.op, carryCellWriteSet.contains(cell) {
      continue
    }
    if valueRemapping.isEmpty {
      result.append(uop)
    } else {
      result.append(
        UOp(
          op: uop.op.remapLazyInputs(valueRemapping),
          value: uop.value,
          kind: uop.kind,
          scalarType: uop.scalarType
        ))
    }
  }

  // BPTT carry cell writes: emit inside the reverse loop
  // These store the temporal gradient for the next (earlier) frame to read
  for writeNodeId in carryCellWriteNodes {
    guard let writeNode = g.nodes[writeNodeId] else { continue }
    if case .memoryWrite(let cell) = writeNode.op {
      // The write value is the gradient that flows to historyRead (the second input)
      let gradInput = writeNode.inputs.count > 1 ? writeNode.inputs[1] : writeNode.inputs[0]
      if let gradLz = ctx.values[gradInput] {
        let remappedGradLz = valueRemapping[gradLz] ?? gradLz
        let writeVar = ctx.useVariable(src: nil)
        result.append(
          UOp(op: .memoryWrite(cell, zeroConst, remappedGradLz), value: writeVar, kind: .scalar))
      }
    }
  }

  // End backward loop
  var endBwd = UOp(op: .endLoop, value: .empty)
  endBwd.kind = .scalar
  result.append(endBwd)

  // Remove carry cell write side effects that we've handled in the BPTT loop
  // so they don't also get emitted in the output block via chainGradientSideEffects
  if !carryCellWriteNodes.isEmpty {
    let handledSet = Set(carryCellWriteNodes)
    g.gradientSideEffects.removeAll { handledSet.contains($0) }
  }

  return result
}

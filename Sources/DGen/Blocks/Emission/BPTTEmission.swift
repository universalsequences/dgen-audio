/// Backpropagation Through Time (BPTT) activation checks and loop-wrapping emission.
import Foundation

/// Check if this block contains a pass-through historyWrite with carry cells.
/// Only activate Backpropagation Through Time (BPTT) when historyWrite's output is consumed by other nodes
/// (the pass-through pattern from Signal.history()).
///
/// - Parameters:
///   - block: Candidate block.
///   - g: Graph containing carry-cell and node-consumer metadata.
/// - Returns: `true` when BPTT (Backpropagation Through Time) loop wrapping must be enabled for this block.
func blockHasPassThroughHistoryWriteWithCarry(block: Block, g: Graph) -> Bool {
  block.nodes.contains { nodeId in
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
}

/// Tuple alias for a carry-cell read value that must be reloaded per reverse iteration.
private typealias CarryCellRead = (cellId: CellID, originalLazy: Lazy)

/// Precomputed BPTT analysis artifacts used by loop emission.
private struct BPTTPlan {
  let forwardUops: [UOp]
  let backwardUops: [UOp]
  let perFrameCells: [NodeID: CellID]
  let carryCellReads: [CarryCellRead]
  let carryCellWriteNodes: [NodeID]
  let carryCellIds: Set<CellID>
}

/// Splits emitted block body UOps into forward and backward segments.
///
/// - Parameters:
///   - bodyUops: Emitted UOps in forward-then-backward order.
///   - backwardStartIndex: Boundary index where backward UOps begin.
/// - Returns: Forward and backward UOp arrays.
private func splitBodyUOpsForBPTT(
  bodyUops: [UOp], backwardStartIndex: Int
) -> (forward: [UOp], backward: [UOp]) {
  (
    forward: Array(bodyUops[0..<backwardStartIndex]),
    backward: Array(bodyUops[backwardStartIndex...])
  )
}

/// Finds forward nodes whose values are consumed by backward nodes.
///
/// - Parameters:
///   - block: Block being wrapped for BPTT.
///   - lastForwardId: Last forward node ID in the graph.
///   - g: Graph containing dependency metadata.
/// - Returns: Backward node set and forward value IDs needed for backward execution.
private func collectForwardBackwardDependencies(
  block: Block,
  lastForwardId: NodeID,
  g: Graph
) -> (backwardNodeIds: Set<NodeID>, forwardValuesNeeded: Set<NodeID>) {
  let forwardNodeIds = Set(block.nodes.filter { $0 <= lastForwardId })
  let backwardNodeIds = Set(block.nodes.filter { $0 > lastForwardId })

  var forwardValuesNeeded = Set<NodeID>()
  for nodeId in backwardNodeIds {
    guard let node = g.nodes[nodeId] else { continue }
    for inputId in node.allDependencies where forwardNodeIds.contains(inputId) {
      forwardValuesNeeded.insert(inputId)
    }
  }

  return (backwardNodeIds: backwardNodeIds, forwardValuesNeeded: forwardValuesNeeded)
}

/// Allocates per-frame storage for forward values required by backward computation.
///
/// - Parameters:
///   - forwardValuesNeeded: Forward node IDs needed by backward nodes.
///   - ctx: IR context used to inspect existing lazy values.
///   - g: Graph used for new cell allocation and size registration.
/// - Returns: Mapping from forward node ID to allocated per-frame cell ID.
private func allocatePerFrameStorageCells(
  forwardValuesNeeded: Set<NodeID>,
  ctx: IRContext,
  g: Graph
) -> [NodeID: CellID] {
  var perFrameCells: [NodeID: CellID] = [:]
  for nodeId in forwardValuesNeeded {
    guard let lz = ctx.values[nodeId] else { continue }
    switch lz {
    case .variable:
      let cell = g.alloc()
      perFrameCells[nodeId] = cell
      g.cellAllocationSizes[cell] = g.maxFrameCount
    default:
      break
    }
  }
  return perFrameCells
}

/// Collects carry-cell reads used by backward nodes but emitted outside this block.
///
/// - Parameters:
///   - block: Block being wrapped for BPTT.
///   - backwardNodeIds: Backward nodes in the current block.
///   - carryCellIds: Carry cell IDs tracked by the graph.
///   - g: Graph containing nodes and dependencies.
///   - ctx: IR context containing lazy values.
/// - Returns: Carry-cell reads that must be reloaded in the reverse loop.
private func collectCarryCellReads(
  block: Block,
  backwardNodeIds: Set<NodeID>,
  carryCellIds: Set<CellID>,
  g: Graph,
  ctx: IRContext
) -> [CarryCellRead] {
  let blockNodeSet = Set(block.nodes)
  var carryCellReads: [CarryCellRead] = []

  for nodeId in g.nodes.keys where !blockNodeSet.contains(nodeId) {
    guard let node = g.nodes[nodeId] else { continue }
    guard case .memoryRead(let cell) = node.op, carryCellIds.contains(cell) else { continue }
    guard let lz = ctx.values[nodeId] else { continue }

    let usedInBlock = backwardNodeIds.contains { bNodeId in
      guard let bNode = g.nodes[bNodeId] else { return false }
      return bNode.allDependencies.contains(nodeId)
    }
    if usedInBlock {
      carryCellReads.append((cellId: cell, originalLazy: lz))
    }
  }

  return carryCellReads
}

/// Finds carry-cell write nodes in the graph's backward partition.
///
/// - Parameters:
///   - lastForwardId: Last forward node ID in the graph.
///   - carryCellIds: Carry cell IDs tracked by the graph.
///   - g: Graph containing node definitions.
/// - Returns: Node IDs that write carry cells in backward execution.
private func collectCarryCellWriteNodes(
  lastForwardId: NodeID,
  carryCellIds: Set<CellID>,
  g: Graph
) -> [NodeID] {
  var carryCellWriteNodes: [NodeID] = []
  for (nodeId, node) in g.nodes {
    guard nodeId > lastForwardId else { continue }
    if case .memoryWrite(let cell) = node.op, carryCellIds.contains(cell) {
      carryCellWriteNodes.append(nodeId)
    }
  }
  return carryCellWriteNodes
}

/// Builds all analysis artifacts required for Backpropagation Through Time (BPTT) loop wrapping.
///
/// - Parameters:
///   - bodyUops: Emitted block UOps in forward/backward order.
///   - backwardStartIndex: Index where backward UOps begin.
///   - block: Block being wrapped.
///   - g: Graph with forward/backward metadata and carry-cell state.
///   - ctx: IR context containing emitted lazy values.
/// - Returns: A complete `BPTTPlan` consumed by emission helpers.
private func buildBPTTPlan(
  bodyUops: [UOp],
  backwardStartIndex: Int,
  block: Block,
  g: Graph,
  ctx: IRContext
) -> BPTTPlan {
  let lastForwardId = g.lastForwardNodeId!
  let splitUOps = splitBodyUOpsForBPTT(bodyUops: bodyUops, backwardStartIndex: backwardStartIndex)
  let deps = collectForwardBackwardDependencies(block: block, lastForwardId: lastForwardId, g: g)
  let perFrameCells = allocatePerFrameStorageCells(
    forwardValuesNeeded: deps.forwardValuesNeeded, ctx: ctx, g: g)

  let carryCellIds = Set(g.gradCarryCells.values)
  let carryCellReads = collectCarryCellReads(
    block: block, backwardNodeIds: deps.backwardNodeIds, carryCellIds: carryCellIds, g: g, ctx: ctx)
  let carryCellWriteNodes = collectCarryCellWriteNodes(
    lastForwardId: lastForwardId, carryCellIds: carryCellIds, g: g)

  return BPTTPlan(
    forwardUops: splitUOps.forward,
    backwardUops: splitUOps.backward,
    perFrameCells: perFrameCells,
    carryCellReads: carryCellReads,
    carryCellWriteNodes: carryCellWriteNodes,
    carryCellIds: carryCellIds
  )
}

/// Appends a scalar `beginLoop` UOp to start the forward loop.
///
/// - Parameters:
///   - result: UOp output buffer updated in place.
///   - frameCountVar: Lazy frame-count variable used by loop renderers.
private func appendForwardLoopStart(result: inout [UOp], frameCountVar: Lazy) {
  var beginFwd = UOp(op: .beginLoop(frameCountVar, 1), value: .empty)
  beginFwd.kind = .scalar
  result.append(beginFwd)
}

/// Appends per-frame stores for forward values needed by backward computation.
///
/// - Parameters:
///   - result: UOp output buffer updated in place.
///   - perFrameCells: Mapping of forward node IDs to per-frame cells.
///   - ctx: IR context used to fetch original lazy values and allocate temporaries.
private func appendPerFrameStores(
  result: inout [UOp],
  perFrameCells: [NodeID: CellID],
  ctx: IRContext
) {
  for (nodeId, cell) in perFrameCells.sorted(by: { $0.key < $1.key }) {
    guard let lz = ctx.values[nodeId] else { continue }
    let frameIdxVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .frameIndex, value: frameIdxVar, kind: .scalar, scalarType: .int))
    let storeVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .memoryWrite(cell, frameIdxVar, lz), value: storeVar, kind: .scalar))
  }
}

/// Appends a scalar `beginReverseLoop` UOp to start backward execution.
///
/// - Parameters:
///   - result: UOp output buffer updated in place.
///   - frameCountVar: Lazy frame-count variable used by loop renderers.
private func appendBackwardLoopStart(result: inout [UOp], frameCountVar: Lazy) {
  var beginBwd = UOp(op: .beginReverseLoop(frameCountVar), value: .empty)
  beginBwd.kind = .scalar
  result.append(beginBwd)
}

/// Appends per-frame loads and carry-cell rereads for backward execution.
///
/// - Parameters:
///   - result: UOp output buffer updated in place.
///   - perFrameCells: Forward values stored per frame.
///   - carryCellReads: External carry reads that must be reloaded each reverse iteration.
///   - ctx: IR context used to allocate temporaries.
/// - Returns: Value remapping table for backward UOps and the zero-offset constant.
private func appendBackwardReloadsAndBuildRemapping(
  result: inout [UOp],
  perFrameCells: [NodeID: CellID],
  carryCellReads: [CarryCellRead],
  ctx: IRContext
) -> (valueRemapping: [Lazy: Lazy], zeroConst: Lazy) {
  var valueRemapping: [Lazy: Lazy] = [:]

  for (nodeId, cell) in perFrameCells.sorted(by: { $0.key < $1.key }) {
    guard let originalLz = ctx.values[nodeId] else { continue }
    let frameIdxVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .frameIndex, value: frameIdxVar, kind: .scalar, scalarType: .int))
    let loadedVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .memoryRead(cell, frameIdxVar), value: loadedVar, kind: .scalar))
    valueRemapping[originalLz] = loadedVar
  }

  let zeroConst = ctx.useConstant(src: nil, value: 0.0)
  for carry in carryCellReads {
    let carryReadVar = ctx.useVariable(src: nil)
    result.append(UOp(op: .memoryRead(carry.cellId, zeroConst), value: carryReadVar, kind: .scalar))
    valueRemapping[carry.originalLazy] = carryReadVar
  }

  return (valueRemapping: valueRemapping, zeroConst: zeroConst)
}

/// Derives the set of carry-cell IDs already written by backward UOps.
///
/// - Parameters:
///   - carryCellWriteNodes: Node IDs known to write carry cells.
///   - carryCellIds: Valid carry cell IDs.
///   - g: Graph containing write-node definitions.
/// - Returns: Carry cell IDs whose writes should be skipped and re-emitted explicitly.
private func computeCarryCellWriteSet(
  carryCellWriteNodes: [NodeID],
  carryCellIds: Set<CellID>,
  g: Graph
) -> Set<CellID> {
  Set(
    carryCellWriteNodes.compactMap { g.nodes[$0] }.compactMap { node -> CellID? in
      if case .memoryWrite(let cell) = node.op, carryCellIds.contains(cell) { return cell }
      return nil
    })
}

/// Appends backward UOps, remapping lazy inputs and skipping carry writes.
///
/// - Parameters:
///   - result: UOp output buffer updated in place.
///   - backwardUops: Backward portion of the original block UOps.
///   - carryCellWriteSet: Carry cell IDs whose writes are re-emitted separately.
///   - valueRemapping: Lazy-value remapping built from per-frame reloads.
private func appendRemappedBackwardUOps(
  result: inout [UOp],
  backwardUops: [UOp],
  carryCellWriteSet: Set<CellID>,
  valueRemapping: [Lazy: Lazy]
) {
  for uop in backwardUops {
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
}

/// Appends explicit carry-cell writes inside the reverse loop.
///
/// - Parameters:
///   - result: UOp output buffer updated in place.
///   - carryCellWriteNodes: Node IDs that write carry-cell gradients.
///   - valueRemapping: Backward remapping from original to reloaded values.
///   - zeroConst: Zero offset constant for carry-cell addressing.
///   - g: Graph containing write-node definitions.
///   - ctx: IR context with input lazy values.
private func appendCarryCellWrites(
  result: inout [UOp],
  carryCellWriteNodes: [NodeID],
  valueRemapping: [Lazy: Lazy],
  zeroConst: Lazy,
  g: Graph,
  ctx: IRContext
) {
  for writeNodeId in carryCellWriteNodes {
    guard let writeNode = g.nodes[writeNodeId] else { continue }
    if case .memoryWrite(let cell) = writeNode.op {
      let gradInput = writeNode.inputs.count > 1 ? writeNode.inputs[1] : writeNode.inputs[0]
      if let gradLz = ctx.values[gradInput] {
        let remappedGradLz = valueRemapping[gradLz] ?? gradLz
        let writeVar = ctx.useVariable(src: nil)
        result.append(
          UOp(op: .memoryWrite(cell, zeroConst, remappedGradLz), value: writeVar, kind: .scalar))
      }
    }
  }
}

/// Appends a scalar `endLoop` UOp.
///
/// - Parameter result: UOp output buffer updated in place.
private func appendScalarLoopEnd(result: inout [UOp]) {
  var endLoop = UOp(op: .endLoop, value: .empty)
  endLoop.kind = .scalar
  result.append(endLoop)
}

/// Removes carry write side effects already handled inside the BPTT reverse loop.
///
/// - Parameters:
///   - carryCellWriteNodes: Node IDs whose writes were emitted manually.
///   - g: Graph whose `gradientSideEffects` list is updated.
private func removeHandledCarryCellSideEffects(
  carryCellWriteNodes: [NodeID],
  g: Graph
) {
  guard !carryCellWriteNodes.isEmpty else { return }
  let handledSet = Set(carryCellWriteNodes)
  g.gradientSideEffects.removeAll { handledSet.contains($0) }
}

/// Wraps emitted UOps with forward/reverse loops for correct Backpropagation Through Time (BPTT).
///
/// Forward nodes run in a forward loop (0->N-1), backward nodes run in a reverse loop (N-1->0).
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
/// - Throws: Propagates errors from downstream emission helpers.
func wrapWithBPTTLoops(
  bodyUops: [UOp],
  backwardStartIndex: Int,
  block: Block,
  g: Graph,
  ctx: IRContext
) throws -> [UOp] {
  let plan = buildBPTTPlan(
    bodyUops: bodyUops,
    backwardStartIndex: backwardStartIndex,
    block: block,
    g: g,
    ctx: ctx
  )

  var result: [UOp] = []
  let frameCountVar = Lazy.variable(-1, nil)

  appendForwardLoopStart(result: &result, frameCountVar: frameCountVar)
  result.append(contentsOf: plan.forwardUops)
  appendPerFrameStores(result: &result, perFrameCells: plan.perFrameCells, ctx: ctx)
  appendScalarLoopEnd(result: &result)

  appendBackwardLoopStart(result: &result, frameCountVar: frameCountVar)
  let remapState = appendBackwardReloadsAndBuildRemapping(
    result: &result, perFrameCells: plan.perFrameCells, carryCellReads: plan.carryCellReads, ctx: ctx)
  let carryCellWriteSet = computeCarryCellWriteSet(
    carryCellWriteNodes: plan.carryCellWriteNodes, carryCellIds: plan.carryCellIds, g: g)
  appendRemappedBackwardUOps(
    result: &result,
    backwardUops: plan.backwardUops,
    carryCellWriteSet: carryCellWriteSet,
    valueRemapping: remapState.valueRemapping
  )
  appendCarryCellWrites(
    result: &result,
    carryCellWriteNodes: plan.carryCellWriteNodes,
    valueRemapping: remapState.valueRemapping,
    zeroConst: remapState.zeroConst,
    g: g,
    ctx: ctx
  )
  appendScalarLoopEnd(result: &result)
  removeHandledCarryCellSideEffects(carryCellWriteNodes: plan.carryCellWriteNodes, g: g)

  return result
}

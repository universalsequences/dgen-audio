/// BPTT activation checks and loop-wrapping emission.
import Foundation

/// Check if this block contains a pass-through historyWrite with carry cells.
/// Only activate BPTT when historyWrite's output is consumed by other nodes
/// (the pass-through pattern from Signal.history()).
///
/// - Parameters:
///   - block: Candidate block.
///   - g: Graph containing carry-cell and node-consumer metadata.
/// - Returns: `true` when BPTT loop wrapping must be enabled for this block.
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

/// Wraps emitted UOps with forward/reverse loops for correct BPTT.
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

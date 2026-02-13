/// Block emission: UOp generation, dependency analysis, SIMD upgrade, shape transitions, BPTT.
import Foundation

// MARK: - SIMD Analysis

/// Check if any UOps contain patterns that prevent SIMD optimization:
/// - Inner loops (beginLoop, beginForLoop)
/// - View operations (reshape, transpose, shrink) that require complex index arithmetic (C only)
/// - Broadcast access (non-contiguous strides or shape mismatch) (C only)
///
/// - Parameters:
///   - uops: Candidate operations to inspect.
///   - backend: Active backend because SIMD blockers differ for C vs Metal.
/// - Returns: `true` when any operation pattern makes SIMD lowering unsafe.
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
///
/// - Parameter uops: Emitted operations to analyze and mutate in place.
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

// MARK: - Thread Count Scale

/// Emits setup UOps for flat thread decomposition in tensor blocks when required.
///
/// This initializes frame and element index variables used by frame-aware reads and
/// bounds logic in threaded tensor execution.
///
/// - Parameters:
///   - ctx: Emission context receiving per-node tensor indices and frame-aware flags.
///   - block: Block currently being emitted.
///   - g: Graph used to detect frame-aware tensor outputs.
/// - Returns: Setup UOps required before node emission, or an empty list for non-tensor blocks.
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

/// Clears per-block frame-aware state so each block starts with a clean context.
///
/// - Parameter ctx: Emission context to reset.
private func resetFrameAwareBlockContext(_ ctx: IRContext) {
  // Reset frame-aware tensor block context for each new block.
  // These flags are set per-block in emitThreadCountScaleOpIfNeeded.
  ctx.isInFrameAwareTensorBlock = false
  ctx.frameAwareTensorFrameIndex = nil
  ctx.frameAwareTensorElementIndex = nil
}

/// Ensures convolution input tensors are treated as outbound memory values.
///
/// Conv ops read inputs directly from memory (`memoryRead`) rather than register-backed
/// tensor loads, so their inputs must be materialized.
///
/// - Parameters:
///   - outboundCells: Outbound tensor cell set to update in place.
///   - block: Block whose nodes are being inspected.
///   - g: Graph containing node and tensor metadata.
private func markConvInputsAsOutbound(_ outboundCells: inout Set<CellID>, block: Block, g: Graph) {
  // Mark conv2d/conv1d input tensors as outbound - they use memoryRead() directly
  // instead of tload(), so the input MUST be in memory not just in registers.
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
}

/// Computes and writes the outbound tensor-cell policy for the block.
///
/// This merges cross-block dependencies, cross-region dependencies for shape-aware emission,
/// and special convolution input requirements, then resets register tracking.
///
/// - Parameters:
///   - ctx: Emission context updated with outbound cells and cleared tensor registers.
///   - block: Block being emitted.
///   - blocks: Full block list used for dependency analysis.
///   - g: Graph containing node/tensor metadata.
///   - shapeTransitions: Shape-transition boundaries for this block.
///   - hasMultipleShapes: Whether the block emits through multiple shape regions.
private func prepareOutboundTensorCells(
  ctx: IRContext, block: Block, blocks: [Block], g: Graph,
  shapeTransitions: [(nodeIndex: Int, shape: [Int])],
  hasMultipleShapes: Bool
) {
  // Tensor Register Optimization:
  // Compute which tensor cells need to be written to memory (used by later blocks)
  // and clear the register tracking for this new block.
  var outboundCells = findOutboundTensorCells(blocks, g, block: block)

  // For ALL backends: include cross-region outbound cells (tensor -> scalar reductions)
  // This ensures tensors are written to memory before scalar reductions read them.
  if hasMultipleShapes {
    let crossRegion = findCrossRegionOutboundCells(
      block: block, g: g, transitions: shapeTransitions)
    outboundCells.formUnion(crossRegion)
  }

  markConvInputsAsOutbound(&outboundCells, block: block, g: g)
  ctx.outboundTensorCells = outboundCells
  ctx.clearTensorRegisters()
}

/// Emits the standard (non shape-aware) body UOps for a block.
///
/// This path handles optional thread scaling, per-node emission, and BPTT wrapping when the
/// block contains a forward/backward split with carry-state history writes.
///
/// - Parameters:
///   - ctx: Emission context used by node emitters.
///   - block: Block currently being emitted.
///   - g: Graph containing node definitions and forward/backward partition metadata.
///   - backend: Target backend to gate backend-specific setup (e.g., thread scaling).
///   - emittedNodes: Accumulator of node IDs successfully emitted.
/// - Returns: Ordered block body UOps for downstream wrapping.
private func emitStandardBlockBodyUOps(
  ctx: IRContext, block: Block, g: Graph, backend: Backend,
  emittedNodes: inout Set<NodeID>
) throws -> [UOp] {
  var bodyUops: [UOp] = []

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
  // split into forward loop (0->N-1) and reverse backward loop (N-1->0).
  if let backwardStart = backwardUOpsStartIndex,
    blockHasPassThroughHistoryWriteWithCarry(block: block, g: g)
  {
    bodyUops = try wrapWithBPTTLoops(
      bodyUops: bodyUops,
      backwardStartIndex: backwardStart,
      block: block,
      g: g,
      ctx: ctx
    )
    ctx.lastBlockHasOwnFrameLoop = true
  }

  return bodyUops
}

/// Selects the body emission strategy for a block.
///
/// - Parameters:
///   - ctx: Emission context used by node emitters.
///   - block: Block currently being emitted.
///   - blocks: Full block list for shape-aware outbound analysis.
///   - g: Graph containing node metadata.
///   - backend: Target backend.
///   - useShapeAwareEmission: Whether to emit using per-region shape-aware loops.
///   - shapeTransitions: Shape-transition boundaries used by shape-aware emission.
///   - emittedNodes: Accumulator of emitted nodes for cross-block global wiring.
/// - Returns: Emitted body UOps for the selected strategy.
private func emitBlockBodyUOps(
  ctx: IRContext, block: Block, blocks: [Block], g: Graph, backend: Backend,
  useShapeAwareEmission: Bool,
  shapeTransitions: [(nodeIndex: Int, shape: [Int])],
  emittedNodes: inout Set<NodeID>
) throws -> [UOp] {
  if useShapeAwareEmission {
    // Use specialized emission with per-shape element loops.
    let shapeAwareUOps = try emitScalarBlockWithShapeTransitions(
      ctx: ctx, block: block, blocks: blocks, g: g, transitions: shapeTransitions
    )
    for nodeId in block.nodes {
      emittedNodes.insert(nodeId)
    }
    return shapeAwareUOps
  }
  return try emitStandardBlockBodyUOps(
    ctx: ctx, block: block, g: g, backend: backend, emittedNodes: &emittedNodes)
}

/// Computes final SIMD/scalar execution kind and loop increment for the block body.
///
/// - Parameters:
///   - bodyUops: Emitted body operations used to detect SIMD blockers.
///   - block: Block metadata containing shape, temporality, and tensor-loop context.
///   - backend: Target backend controlling SIMD constraints.
/// - Returns: Tuple with effective UOp kind and tensor-loop increment (`1` or `4`).
private func determineSIMDPlan(
  bodyUops: [UOp], block: Block, backend: Backend
) -> (effectiveKind: Kind, simdIncrement: Int) {
  // Analyze emitted UOps to determine if SIMD is safe.
  // SIMD is safe if: tensor block + size divisible by 4 + no SIMD blockers + not frame-based.
  let hasSIMDBlockers = containsSIMDBlockers(bodyUops, backend: backend)
  let canUseSIMD: Bool
  let simdIncrement: Int

  if let shape = block.shape, block.tensorIndex != nil {
    let size = shape.reduce(1, *)
    // Frame-based tensor blocks must run element-by-element per frame
    // because their values change every frame (e.g., downstream of phasor(tensor)).
    let isFrameBased = block.temporality == .frameBased
    canUseSIMD = !hasSIMDBlockers && !isFrameBased && (size % 4 == 0)
    simdIncrement = canUseSIMD ? 4 : 1
  } else {
    canUseSIMD = false
    simdIncrement = 1
  }

  let effectiveKind: Kind
  if block.tensorIndex != nil {
    effectiveKind = canUseSIMD ? .simd : .scalar
  } else {
    effectiveKind = block.kind
  }
  return (effectiveKind: effectiveKind, simdIncrement: simdIncrement)
}

/// Applies a uniform execution kind to every emitted body UOp.
///
/// - Parameters:
///   - kind: Kind to apply (`.scalar` or `.simd`).
///   - bodyUops: UOps updated in place.
private func applyKind(_ kind: Kind, to bodyUops: inout [UOp]) {
  for i in 0..<bodyUops.count {
    bodyUops[i].kind = kind
  }
}

/// Wraps body UOps with a tensor loop when backend/strategy requires it.
///
/// Shape-aware emission already emits its own element loops and therefore never receives an
/// outer tensor loop wrapper from this helper.
///
/// - Parameters:
///   - bodyUops: Emitted block body operations.
///   - block: Block metadata containing tensor index and shape.
///   - backend: Target backend.
///   - useShapeAwareEmission: Whether body UOps already include element loops.
///   - simdIncrement: Loop increment for tensor iteration.
///   - effectiveKind: Final operation kind applied to the loop wrapper.
/// - Returns: Tuple of final UOps and whether a tensor loop wrapper was added.
private func wrapBodyUOpsWithTensorLoopIfNeeded(
  bodyUops: [UOp], block: Block, backend: Backend,
  useShapeAwareEmission: Bool, simdIncrement: Int, effectiveKind: Kind
) -> (uops: [UOp], needsTensorLoop: Bool) {
  // Build final UOps array with parallelRange wrapper if needed.
  // Note: frame-aware tensor blocks DON'T use parallelRange (no loop).
  var uops: [UOp] = []

  // C backend wraps tensor ops in a sequential loop.
  // Skip if using shape-aware emission (it has its own loops).
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
  return (uops: uops, needsTensorLoop: needsTensorLoop)
}

/// Inserts cross-block global define/load operations for scalar dependencies.
///
/// Tensor-valued dependencies are handled through tensor memory cells and are intentionally
/// excluded from scratch-global wiring.
///
/// - Parameters:
///   - uops: Final block UOps mutated in place with prepended global operations.
///   - emittedNodes: Nodes emitted in this block.
///   - ctx: Emission context containing value and global-variable mappings.
///   - block: Block currently being emitted.
///   - blocks: Full block list for dependency queries.
///   - g: Graph containing node metadata.
private func wireCrossBlockGlobals(
  uops: inout [UOp], emittedNodes: Set<NodeID>, ctx: IRContext, block: Block,
  blocks: [Block], g: Graph
) {
  // Handle cross-block dependencies using scratch buffers (for scalar values only).
  // Tensor-valued outputs/inputs do NOT use scratch buffers.
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
}

/// Emits the full UOp sequence for a block, including analysis, body emission, and wrapping.
///
/// Pipeline steps:
/// 1. Reset frame-aware context.
/// 2. Detect shape transitions and choose emission strategy.
/// 3. Compute outbound tensor cells.
/// 4. Emit block body.
/// 5. Determine SIMD plan and apply final kind.
/// 6. Add tensor-loop wrapper if needed.
/// 7. Wire scalar cross-block globals.
///
/// - Parameters:
///   - ctx: Emission context shared across blocks.
///   - block: Block to emit.
///   - blocks: Full ordered block list for dependency analysis.
///   - g: Graph containing nodes, tensors, and metadata used by emitters.
///   - backend: Target backend (`.metal` by default).
///   - debug: Reserved debug toggle (currently ignored).
/// - Returns: Final UOps and the block effective kind used for rendering decisions.
public func emitBlockUOps(
  ctx: IRContext, block: Block, blocks: [Block], g: Graph, backend: Backend = .metal,
  debug: Bool = false
) throws -> (uops: [UOp], effectiveKind: Kind) {
  _ = debug
  resetFrameAwareBlockContext(ctx)

  // Check for scalar blocks with shape transitions (e.g., conv2d in feedback loops).
  // These need nested element loops instead of flat threading.
  let shapeTransitions = detectShapeTransitions(block: block, g: g)
  let hasMultipleShapes = shapeTransitions.count > 1
  let useShapeAwareEmission = hasMultipleShapes

  prepareOutboundTensorCells(
    ctx: ctx, block: block, blocks: blocks, g: g,
    shapeTransitions: shapeTransitions, hasMultipleShapes: hasMultipleShapes)

  var emittedNodes: Set<NodeID> = []
  var bodyUops = try emitBlockBodyUOps(
    ctx: ctx, block: block, blocks: blocks, g: g, backend: backend,
    useShapeAwareEmission: useShapeAwareEmission, shapeTransitions: shapeTransitions,
    emittedNodes: &emittedNodes)

  let simdPlan = determineSIMDPlan(bodyUops: bodyUops, block: block, backend: backend)
  applyKind(simdPlan.effectiveKind, to: &bodyUops)

  var (uops, needsTensorLoop) = wrapBodyUOpsWithTensorLoopIfNeeded(
    bodyUops: bodyUops, block: block, backend: backend,
    useShapeAwareEmission: useShapeAwareEmission,
    simdIncrement: simdPlan.simdIncrement, effectiveKind: simdPlan.effectiveKind)

  wireCrossBlockGlobals(
    uops: &uops, emittedNodes: emittedNodes, ctx: ctx, block: block, blocks: blocks, g: g)

  // Close the tensor loop for C backend and hop-based Metal blocks.
  if needsTensorLoop, block.tensorIndex != nil {
    uops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
  }

  return (uops: uops, effectiveKind: simdPlan.effectiveKind)
}

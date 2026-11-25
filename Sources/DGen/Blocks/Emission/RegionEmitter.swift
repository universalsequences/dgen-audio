/// Region-level UOp emission for shape-aware block emission.
import Foundation

/// Emit UOps for a single EmissionRegion.
///
/// Scalar nodes emit first, tensor nodes emit inside an optional element loop, and hop-gating
/// is inserted when the region carries a hop counter.
///
/// - Parameters:
///   - region: Pre-classified region to emit.
///   - ctx: Shared emission context.
///   - g: Graph containing node definitions.
/// - Returns: UOps for exactly one region.
private func emitRegion(
  _ region: EmissionRegion, ctx: IRContext, g: Graph,
  parallelElemIdx: Lazy? = nil, splittableRange: Bool = false, allowMatVec: Bool = false
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
      typedUop.vectorWidth = 1
      uops.append(typedUop)
    }
  }

  // Emit tensor nodes
  let activeTensorNodes =
    region.isSkipped
    ? []
    : region.tensorNodes.filter {
      !ctx.skippedTensorComputeNodes.contains($0)
        && !ctx.crossBlockSkippedTensorNodes.contains($0)
    }

  if !activeTensorNodes.isEmpty {
    // When parallelElemIdx is provided, each thread handles one element — no loop needed.
    if let elemIdx = parallelElemIdx, !region.isConvOnly {
      for nodeId in activeTensorNodes {
        ctx.tensorIndices[nodeId] = elemIdx
        if let node = g.nodes[nodeId] {
          for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
            var typedUop = uop
            typedUop.vectorWidth = 1
            uops.append(typedUop)
          }
        }
      }
    } else {
      let elemVar = ctx.useVariable(src: nil)
      let elementCount = region.shape.reduce(1, *)
      let savedFrame = ctx.frameAwareTensorFrameIndex
      ctx.frameAwareTensorFrameIndex = nil
      defer {
        ctx.frameAwareTensorFrameIndex = savedFrame
      }

      // Capture-frame gating: when the region's only compute node is a tensor
      // latch whose output tensor aliases its persistent state cell (consumers
      // read the cell directly — see TensorMemoryMaterializationPass), the
      // whole element loop runs only on frames where the scalar cond fires.
      // Between captures the cell simply holds, which is exactly the latch
      // semantics the per-element gswitch implemented every frame.
      let gatedLatch = gatableAliasedLatch(
        region: region, activeTensorNodes: activeTensorNodes, ctx: ctx, g: g)
      if let (latchNodeId, condLazy) = gatedLatch {
        ctx.gatedLatchNodes.insert(latchNodeId)
        let gate = IRBuilder(ctx: ctx, nodeId: latchNodeId)
        let cond = gate.value(condLazy) > gate.constant(0.0)
        gate.ops.append(UOp(op: .beginIf(cond.lazy), value: ctx.useVariable(src: nil)))
        uops.append(contentsOf: gate.ops)
      }

      // splittableRange: a parallel range instead of a plain loop, so the Metal renderer's
      // splitStaticParallelRanges can give the region its own thread-parallel kernel
      // (kernel boundaries order the regions). Unsplit, it still renders as a loop.
      let asRange = splittableRange && !region.isConvOnly && gatedLatch == nil

      // Matmul fast path (C backend): a `sumAxis(mul(x, W))` in matmul form is
      // computed for every output element by one dense kernel call ahead of the
      // element loop, which then only runs the region's other nodes.
      var precomputed = Set<NodeID>()
      if allowMatVec, !region.isConvOnly, gatedLatch == nil {
        for nodeId in activeTensorNodes {
          guard let plan = planMatVec(nodeId: nodeId, region: region, ctx: ctx, g: g) else {
            continue
          }
          uops.append(contentsOf: emitMatVec(plan, nodeId: nodeId, ctx: ctx, g: g))
          precomputed.insert(nodeId)
        }
      }

      if !region.isConvOnly {
        let beginLoop = UOp(
          op: asRange ? .beginParallelRange(elementCount, 1) : .beginForLoop(elemVar, .constant(0, Float(elementCount))),
          value: elemVar
        )
        uops.append(beginLoop)
      }

      ctx.matVecPrecomputedNodes.formUnion(precomputed)
      defer { ctx.matVecPrecomputedNodes.subtract(precomputed) }

      for nodeId in activeTensorNodes {
        ctx.tensorIndices[nodeId] = elemVar
        if let node = g.nodes[nodeId] {
          for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
            var typedUop = uop
            typedUop.vectorWidth = 1
            uops.append(typedUop)
          }
        }
      }

      if !region.isConvOnly {
        let endLoop = asRange ? UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)) : UOp(op: .endLoop, value: .empty)
        uops.append(endLoop)
      }

      if let (latchNodeId, _) = gatedLatch {
        uops.append(UOp(op: .endIf, value: ctx.useVariable(src: nil)))
        ctx.gatedLatchNodes.remove(latchNodeId)
      }
    }
  }

  if region.hopCounter != nil {
    uops.append(UOp(op: .endHopCheck, value: .empty))
  }

  ctx.clearTensorRegisters()
  return uops
}

/// A `sumAxis(mul(x, W), axis: 2)` in the exact form `Graph.matmul` builds:
/// x is a reshape of contiguous [M, K] storage, W a 2-D transpose (plus
/// reshapes) of contiguous [K, N] storage, the output plain [M, N] storage.
private struct MatVecPlan {
  let M: Int
  let K: Int
  let N: Int
  let x: Tensor
  let w: Tensor
  let outCell: CellID
}

private func planMatVec(
  nodeId: NodeID, region: EmissionRegion, ctx: IRContext, g: Graph
) -> MatVecPlan? {
  guard let node = g.nodes[nodeId], case .sumAxis(let axis) = node.op, axis == 2,
    let inId = node.inputs.first,
    case .tensor(let inShape)? = g.nodes[inId]?.shape, inShape.count == 3,
    case .tensor(let outShape) = node.shape, outShape == [inShape[0], inShape[1]],
    let inTensor = g.nodeToTensor[inId].flatMap({ g.tensors[$0] }),
    case .mul(let a, let b)? = ctx.inlineReduceSources[inTensor.cellId],
    let outTensor = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] }),
    outTensor.transforms.isEmpty
  else { return nil }
  let (M, N, K) = (inShape[0], inShape[1], inShape[2])
  // Small products stay on the generic path (no call overhead, nothing to gain).
  guard K >= 8, N >= 8 else { return nil }

  func onlyReshapes(_ transforms: ArraySlice<ViewTransform>) -> Bool {
    transforms.allSatisfy {
      if case .reshape = $0 { return true }
      return false
    }
  }
  func isX(_ t: Tensor) -> Bool {
    t.shape == [M, 1, K] && onlyReshapes(t.transforms[...])
      && t.baseSize == M * K
  }
  func isW(_ t: Tensor) -> Bool {
    guard t.shape == [1, N, K], t.baseSize == K * N,
      let tIdx = t.transforms.firstIndex(where: {
        if case .reshape = $0 { return false }
        return true
      }),
      case .transpose(let axes, let inputShape) = t.transforms[tIdx],
      axes == [1, 0], inputShape == [K, N]
    else { return false }
    return onlyReshapes(t.transforms[..<tIdx]) && onlyReshapes(t.transforms[(tIdx + 1)...])
  }
  let x: Tensor
  let w: Tensor
  if isX(a), isW(b) {
    (x, w) = (a, b)
  } else if isX(b), isW(a) {
    (x, w) = (b, a)
  } else {
    return nil
  }
  // Operands must be materialized storage: not themselves fused away, not
  // hop-scattered weights, and not produced inside this region's element loop
  // (the kernel runs before that loop).
  guard ctx.inlineReduceSources[x.cellId] == nil, ctx.inlineReduceSources[w.cellId] == nil,
    ctx.inlineReduceSources[outTensor.cellId] == nil,
    !g.frameAwareCellScatter.contains(w.cellId),
    x.cellId != outTensor.cellId, w.cellId != outTensor.cellId, x.cellId != w.cellId
  else { return nil }
  for other in region.tensorNodes where other != nodeId {
    guard let otherNode = g.nodes[other], !otherNode.op.isViewOnly,
      let cell = g.nodeToTensor[other].flatMap({ g.tensors[$0] })?.cellId
    else { continue }
    if cell == x.cellId || cell == w.cellId { return nil }
  }
  return MatVecPlan(M: M, K: K, N: N, x: x, w: w, outCell: outTensor.cellId)
}

private func emitMatVec(_ plan: MatVecPlan, nodeId: NodeID, ctx: IRContext, g: Graph) -> [UOp] {
  let b = IRBuilder(ctx: ctx, nodeId: nodeId)
  let frameIdx = b.currentFrameIndex()
  func base(_ cell: CellID, size: Int) -> Expr {
    guard ctx.frameAwareTensorCells.contains(cell) else { return b.intConstant(0) }
    let tensorSize = g.frameAwareCells[cell]?.tensorSize ?? size
    return b.cast(b.frameAwareBase(cellId: cell, frameIdx: frameIdx, tensorSize: tensorSize), to: .int)
  }
  let xOffset = base(plan.x.cellId, size: plan.M * plan.K)
  let wOffset = base(plan.w.cellId, size: plan.K * plan.N)
  let yOffset = base(plan.outCell, size: plan.M * plan.N)
  let gate =
    ctx.frameAwareTensorCells.contains(plan.x.cellId)
    ? b.frameAwareReadGate(cellId: plan.x.cellId, frameIdx: frameIdx)
    : b.constant(1.0)
  b.ops.append(
    UOp(
      op: .matVecCall(
        M: plan.M, K: plan.K, N: plan.N,
        xCell: plan.x.cellId, xOffset: xOffset.lazy,
        wCell: plan.w.cellId, wOffset: wOffset.lazy,
        yCell: plan.outCell, yOffset: yOffset.lazy,
        gate: gate.lazy),
      value: .empty))
  return b.ops.map {
    var typed = $0
    typed.vectorWidth = 1
    return typed
  }
}

/// Returns the (latch node, scalar cond Lazy) when a region's element loop can be
/// gated to capture frames: the region is a plain element loop whose only
/// non-view compute node is a tensor latch aliased to its state cell, with a
/// scalar cond whose value is already available. The aliasing guarantees no
/// per-frame output copy exists that a downstream frame would read back.
private func gatableAliasedLatch(
  region: EmissionRegion, activeTensorNodes: [NodeID], ctx: IRContext, g: Graph
) -> (NodeID, Lazy)? {
  guard !region.isConvOnly else { return nil }
  let computeNodes = activeTensorNodes.filter { !(g.nodes[$0]?.op.isViewOnly ?? false) }
  guard computeNodes.count == 1,
    let latchNodeId = computeNodes.first,
    let latchNode = g.nodes[latchNodeId],
    case .latch(let stateCell) = latchNode.op,
    latchNode.inputs.count == 2,
    g.nodeToTensor[latchNodeId].flatMap({ g.tensors[$0] })?.cellId == stateCell,
    let condLazy = ctx.values[latchNode.inputs[1]]
  else { return nil }
  return (latchNodeId, condLazy)
}

/// Emit UOps for a scalar block with shape transitions (e.g., conv2d in feedback loop).
/// Instead of one flat loop, emits nested loops: outer frame loop + inner element loops.
///
/// - Parameters:
///   - ctx: Shared emission context.
///   - block: Scalar block being emitted.
///   - blocks: Full block list used by outbound dependency analysis.
///   - g: Graph containing node/tensor metadata.
///   - transitions: Shape-region boundaries for this block.
/// - Returns: Ordered UOps spanning all planned regions for the block.
public func emitScalarBlockWithShapeTransitions(
  ctx: IRContext, block: Block, blocks: [Block], g: Graph,
  transitions: [(nodeIndex: Int, shape: [Int])],
  backend: Backend = .c,
  laneParallel: Bool = false,
  index: BlockDependencyIndex? = nil
) throws -> [UOp] {
  // Reset per-block sumAxis fusion metadata; detection repopulates it for this block.
  ctx.inlineReduceSources = [:]
  ctx.skippedTensorComputeNodes = []

  // Analysis: compute outbound cells and detect fusable reduces
  let blockOutbound = findOutboundTensorCells(blocks, g, block: block, index: index)
  var outbound = computeShapeAwareOutboundCells(
    block: block, blocks: blocks, g: g, transitions: transitions, index: index)
  let skipRegions = detectFusableReduces(
    block: block, g: g, transitions: transitions,
    blockOutbound: blockOutbound, outbound: &outbound, ctx: ctx)
  outbound.formUnion(detectInlineableMulReduceNodes(block: block, g: g, ctx: ctx))
  detectInlineableExpandAxisReduceNodes(block: block, g: g, ctx: ctx)
  outbound.formUnion(findInBlockViewReadCells(block: block, g: g))
  outbound.formUnion(ctx.fusedSumOperandCells)
  ctx.outboundTensorCells = outbound
  ctx.clearTensorRegisters()

  // Build structured regions from transitions
  let regions = buildRegions(
    block: block, g: g, ctx: ctx,
    transitions: transitions, skipRegions: skipRegions)

  // Lane-parallel sequential recurrence (detached BPTT backward): one thread
  // per tensor lane, each running the enclosing (reverse) frame loop privately.
  // The region element index is the thread id — no per-region element loops.
  // Scalar region nodes are pure value computations here (guaranteed by
  // StatefulTensorParallelPolicy.laneParallelizable) and run redundantly per
  // thread. The frame index intentionally stays un-overridden so frame-aware
  // tensor reads/writes resolve to the enclosing loop variable.
  if laneParallel, backend == .metal {
    let setup = IRBuilder(ctx: ctx, nodeId: block.nodes[block.nodes.count - 1])
    let laneIdx = setup.threadIndex()
    var uops: [UOp] = setup.ops
    for region in regions {
      uops += try emitRegion(region, ctx: ctx, g: g, parallelElemIdx: laneIdx.lazy)
    }
    return uops
  }

  // Metal: parallelize elements across threads when a single active region has uniform
  // element count. Guards: no scalar nodes (would execute per-thread), single active
  // region (cross-region deps would race without barriers).
  //
  // Conv regions are eligible: conv2d's inner parallelRange auto-consumes the outer
  // elemIdx via ctx.frameAwareTensorElementIndex, so one GPU thread handles one output pixel.
  if backend == .metal {
    var uniformCount: Int? = nil
    var canParallelize = true
    var activeRegionCount = 0
    for region in regions {
      // Scalar nodes are per-frame operations (atomic adds, gradient writes, etc.)
      // that must not be duplicated across element threads.
      if !region.scalarNodes.isEmpty && !region.isSkipped {
        canParallelize = false; break
      }
      let active = region.isSkipped ? [] : region.tensorNodes
      guard !active.isEmpty else { continue }
      activeRegionCount += 1
      let count = region.shape.reduce(1, *)
      if count <= 1 { continue }
      if let existing = uniformCount {
        if existing != count { canParallelize = false; break }
      } else {
        uniformCount = count
      }
    }
    // Multiple active regions may have cross-region data dependencies (e.g., sumAxis
    // reading from a preceding mul). Without inter-thread barriers within a kernel,
    // parallelizing would cause race conditions.
    if activeRegionCount > 1 { canParallelize = false }
    if canParallelize, let elemCount = uniformCount {
      let setup = IRBuilder(ctx: ctx, nodeId: block.nodes[block.nodes.count - 1])
      let (frameIdx, elemIdx) = setup.setupFlatThreading(tensorSize: elemCount)
      var uops: [UOp] = setup.ops
      // Expose the decomposed indices so conv2d/conv1d's inner parallelRange consumes
      // the outer elemIdx instead of emitting its own loop.
      let savedElem = ctx.frameAwareTensorElementIndex
      let savedFrame = ctx.frameAwareTensorFrameIndex
      let savedInBlock = ctx.isInFrameAwareTensorBlock
      ctx.frameAwareTensorElementIndex = elemIdx.lazy
      ctx.frameAwareTensorFrameIndex = frameIdx.lazy
      ctx.isInFrameAwareTensorBlock = true
      defer {
        ctx.frameAwareTensorElementIndex = savedElem
        ctx.frameAwareTensorFrameIndex = savedFrame
        ctx.isInFrameAwareTensorBlock = savedInBlock
      }
      for region in regions {
        uops += try emitRegion(region, ctx: ctx, g: g, parallelElemIdx: elemIdx.lazy)
      }
      return uops
    }
  }

  // Fallback: sequential element loops per region. A static Metal block runs as ONE thread
  // (staticThreads(1)), so its element loops would execute serially on the GPU (seconds for a
  // few million elements). Emit them as parallel ranges instead: splitStaticParallelRanges then
  // dispatches each region as its own N-thread kernel, and falls back to the loop form when the
  // block cannot be split safely.
  let splittable = backend == .metal && block.temporality == .static_
    && ProcessInfo.processInfo.environment["DGEN_NO_STATIC_REGION_SPLIT"] == nil
  let allowMatVec =
    backend == .c && ProcessInfo.processInfo.environment["DGEN_NO_MATVEC"] == nil
  var uops: [UOp] = []
  for region in regions {
    uops += try emitRegion(
      region, ctx: ctx, g: g, splittableRange: splittable, allowMatVec: allowMatVec)
  }
  return uops
}

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
  parallelElemIdx: Lazy? = nil
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
    : region.tensorNodes.filter { !ctx.skippedTensorComputeNodes.contains($0) }

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

      if !region.isConvOnly {
        let beginLoop = UOp(
          op: .beginForLoop(elemVar, .constant(0, Float(elementCount))),
          value: elemVar
        )
        uops.append(beginLoop)
      }

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
        let endLoop = UOp(op: .endLoop, value: .empty)
        uops.append(endLoop)
      }
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
  backend: Backend = .c
) throws -> [UOp] {
  // Reset per-block sumAxis fusion metadata; detection repopulates it for this block.
  ctx.inlineReduceSources = [:]
  ctx.skippedTensorComputeNodes = []

  // Analysis: compute outbound cells and detect fusable reduces
  let blockOutbound = findOutboundTensorCells(blocks, g, block: block)
  var outbound = computeShapeAwareOutboundCells(
    block: block, blocks: blocks, g: g, transitions: transitions)
  let skipRegions = detectFusableReduces(
    block: block, g: g, transitions: transitions,
    blockOutbound: blockOutbound, outbound: &outbound, ctx: ctx)
  detectInlineableMulReduceNodes(block: block, g: g, ctx: ctx)
  detectInlineableExpandAxisReduceNodes(block: block, g: g, ctx: ctx)
  ctx.outboundTensorCells = outbound
  ctx.clearTensorRegisters()

  // Build structured regions from transitions
  let regions = buildRegions(
    block: block, g: g, ctx: ctx,
    transitions: transitions, skipRegions: skipRegions)

  // Metal: parallelize elements across threads when a single active region has uniform
  // element count. Guards: no scalar nodes (would execute per-thread), no conv (own loops),
  // single active region (cross-region deps would race without barriers).
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
      if region.isConvOnly { canParallelize = false; break }
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
      let (_, elemIdx) = setup.setupFlatThreading(tensorSize: elemCount)
      var uops: [UOp] = setup.ops
      for region in regions {
        uops += try emitRegion(region, ctx: ctx, g: g, parallelElemIdx: elemIdx.lazy)
      }
      return uops
    }
  }

  // Fallback: sequential element loops per region.
  var uops: [UOp] = []
  for region in regions {
    uops += try emitRegion(region, ctx: ctx, g: g)
  }
  return uops
}

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

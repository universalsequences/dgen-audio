/// Core block types for the compilation pipeline.
import Foundation

/// Whether frames in this block depend on each other.
public enum FrameOrder { case sequential, parallel }

// corresponds to one kernel (metal backends) or for loop (in C backend)
public struct Block: Equatable {
  public var frameOrder: FrameOrder
  public var nodes: [NodeID] = []
  public var temporality: Temporality = .static_

  /// Logical element iterator for tensor blocks — identifies which tensor element the
  /// current execution is processing (0..<tensorSize). `nil` for pure scalar blocks.
  ///
  /// Lifecycle:
  ///
  ///     BlockFormation:  block.tensorIndex = ctx.useVariable()   // mint fresh VarID
  ///                                ↓
  ///     BlockEmission:   ctx.tensorIndices[nodeId] = tensorIndex // propagate to each node
  ///                                ↓
  ///          ┌─────────────────────┼─────────────────────┐
  ///          C backend             │               Metal backend
  ///          ↓                     │                     ↓
  ///     beginParallelRange    Node emitters         threadIndex or
  ///       (loop var)          read from             setupFlatThreading
  ///          ↓                ctx.tensorIndices           ↓
  ///     for(_pr=0; _pr<N)          ↓                if(id < N)
  ///                         memoryRead/Write
  ///                         at [cell + idx]
  ///
  /// Created in `determineTensorBlocks` (BlockFormation.swift) as a fresh VarID.
  /// Propagated to per-node `ctx.tensorIndices` during emission. Realized as a
  /// `for` loop variable (C) or thread ID (Metal). Node emitters use it to index
  /// into tensor memory cells for reads, writes, and per-element state.
  public var tensorIndex: Lazy?

  public var shape: Shape?

  public init(frameOrder: FrameOrder) {
    self.frameOrder = frameOrder
  }

  public static func == (lhs: Block, rhs: Block) -> Bool {
    return lhs.frameOrder == rhs.frameOrder && lhs.nodes == rhs.nodes
      && lhs.temporality == rhs.temporality && lhs.tensorIndex == rhs.tensorIndex
      && lhs.shape == rhs.shape
  }
}

extension LazyOp {
  var isOutput: Bool {
    if case .output = self { return true }
    return false
  }
}

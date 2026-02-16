import Foundation

/// Namespace for post-emission block finalization steps.
enum UOpBlockFinalization {}

extension UOpBlockFinalization {
  /// Converts emitted raw ops + block metadata into a finalized `BlockUOps` record.
  ///
  /// Finalization is where pipeline-level policies are applied:
  /// - strip `setThreadCountScale` directives into `threadCountScale` metadata
  /// - enforce scalar vectorWidth when backend legality requires it
  /// - run C-backend SIMD loop upgrades
  /// - infer launch policy and kernel split hints
  static func finalize(
    emittedOps: [UOp],
    block: Block,
    graph: Graph,
    backend: Backend,
    bodyFrameOrder: FrameOrder,
    bodyVectorWidth: Int,
    hasOwnFrameLoop: Bool
  ) -> BlockUOps {
    let statefulTensorDecision = StatefulTensorParallelPolicy.decide(
      block: block,
      graph: graph,
      backend: backend
    )
    let (threadCountScale, strippedOps) = extractThreadCountScale(from: emittedOps)
    var finalOps = strippedOps
    let (effectiveFrameOrder, effectiveVectorWidth) = resolveVectorWidth(
      backend: backend,
      blockFrameOrder: block.frameOrder,
      bodyFrameOrder: bodyFrameOrder,
      bodyVectorWidth: bodyVectorWidth,
      threadCountScale: threadCountScale,
      ops: &finalOps
    )

    if backend == .c {
      upgradeElementLoopsToSIMD(&finalOps)
    }

    let dispatchMode = computeDispatchMode(
      block: block,
      threadCountScale: threadCountScale,
      statefulTensorDecision: statefulTensorDecision,
      hasOwnFrameLoop: hasOwnFrameLoop,
      frameOrder: effectiveFrameOrder,
      ops: finalOps
    )

    return BlockUOps(
      ops: finalOps,
      frameOrder: effectiveFrameOrder,
      vectorWidth: effectiveVectorWidth,
      temporality: block.temporality,
      dispatchMode: dispatchMode
    )
  }

  /// Removes thread-scale directives from op list while keeping the latest scale value.
  private static func extractThreadCountScale(from ops: [UOp]) -> (Int?, [UOp]) {
    var scale: Int? = nil
    var filtered: [UOp] = []
    filtered.reserveCapacity(ops.count)

    for op in ops {
      if case .setThreadCountScale(let s) = op.op {
        scale = s
        continue
      }
      filtered.append(op)
    }

    return (scale, filtered)
  }

  /// Resolves final frame order and vector width after backend-specific scalar constraints.
  private static func resolveVectorWidth(
    backend: Backend,
    blockFrameOrder: FrameOrder,
    bodyFrameOrder: FrameOrder,
    bodyVectorWidth: Int,
    threadCountScale: Int?,
    ops: inout [UOp]
  ) -> (FrameOrder, Int) {
    let mustForceScalar =
      backend == .c && (threadCountScale != nil || bodyVectorWidth <= 1)
    guard !mustForceScalar else {
      for i in ops.indices {
        ops[i].vectorWidth = 1
      }
      return (blockFrameOrder, 1)
    }

    return (blockFrameOrder, bodyVectorWidth)
  }

  /// Computes the dispatch mode for a finalized block.
  private static func computeDispatchMode(
    block: Block,
    threadCountScale: Int?,
    statefulTensorDecision: StatefulTensorParallelPolicy.Decision,
    hasOwnFrameLoop: Bool,
    frameOrder: FrameOrder,
    ops: [UOp]
  ) -> DispatchMode {
    if hasOwnFrameLoop { return .selfManaged }

    if block.temporality == .static_ {
      if block.frameOrder.isParallel, let scale = threadCountScale {
        return .staticThreads(scale)
      }
      let hasParallelRange = ops.contains {
        if case .beginParallelRange = $0.op { return true }
        return false
      }
      if hasParallelRange { return .staticThreads(1) }  // split will adjust
      return .staticThreads(1)
    }

    if statefulTensorDecision.enabled {
      return .fixedWithFrameLoop(statefulTensorDecision.tensorSize)
    }

    if let scale = threadCountScale {
      return .perFrameScaled(scale)
    }

    return frameOrder.isSequential ? .singleThreaded : .perFrame
  }
}

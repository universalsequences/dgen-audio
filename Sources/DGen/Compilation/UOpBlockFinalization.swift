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
      graph: graph,
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
    graph: Graph,
    threadCountScale: Int?,
    statefulTensorDecision: StatefulTensorParallelPolicy.Decision,
    hasOwnFrameLoop: Bool,
    frameOrder: FrameOrder,
    ops: [UOp]
  ) -> DispatchMode {
    if hasOwnFrameLoop { return .selfManaged }

    // GEMM ops override dispatch: gemm/gemmChunkPartials use 2D threadgroup grids,
    // gemmSmall uses per-frame element-parallel dispatch.
    for nodeId in block.nodes {
      guard let node = graph.nodes[nodeId] else { continue }
      switch node.op {
      case .gemm(let M, let N, _, _, _):
        return .gemm(tilesM: M / 8, tilesN: N / 8)
      case .gemmChunkPartials(let M, let N, _, _, _, _, let chunkCount):
        return .gemm(tilesM: M / 8, tilesN: N / 8, depth: chunkCount)
      case .gemmSmall(let M, let N, _, _, _):
        return .perFrameScaled(M * N)
      default: continue
      }
    }

    if block.temporality == .static_ {
      if block.frameOrder.isParallel, let scale = threadCountScale {
        return .staticThreads(scale)
      }
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

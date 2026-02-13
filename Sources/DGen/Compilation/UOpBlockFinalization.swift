import Foundation

/// Namespace for post-emission block finalization steps.
enum UOpBlockFinalization {}

extension UOpBlockFinalization {
  /// Converts emitted raw ops + block metadata into a finalized `BlockUOps` record.
  ///
  /// Finalization is where pipeline-level policies are applied:
  /// - strip `setThreadCountScale` directives into `threadCountScale` metadata
  /// - enforce scalar kind when backend legality requires it
  /// - run C-backend SIMD loop upgrades
  /// - infer launch policy and kernel split hints
  static func finalize(
    emittedOps: [UOp],
    block: Block,
    graph: Graph,
    backend: Backend,
    bodyEffectiveKind: Kind,
    hasOwnFrameLoop: Bool
  ) -> BlockUOps {
    let statefulTensorDecision = StatefulTensorParallelPolicy.decide(
      block: block,
      graph: graph,
      backend: backend
    )
    let (threadCountScale, strippedOps) = extractThreadCountScale(from: emittedOps)
    var finalOps = strippedOps
    let effectiveKind = resolveEffectiveKind(
      backend: backend,
      blockKind: block.kind,
      bodyEffectiveKind: bodyEffectiveKind,
      threadCountScale: threadCountScale,
      ops: &finalOps
    )

    if backend == .c {
      upgradeElementLoopsToSIMD(&finalOps)
    }

    let parallelPolicy = inferParallelPolicy(
      kind: effectiveKind,
      temporality: block.temporality,
      ops: finalOps,
      statefulTensorDecision: statefulTensorDecision
    )
    let threadCountOverride = statefulTensorDecision.enabled ? statefulTensorDecision.tensorSize : nil

    let forceNewKernel = threadCountScale != nil || threadCountOverride != nil || hasOwnFrameLoop

    return BlockUOps(
      ops: finalOps,
      kind: effectiveKind,
      temporality: block.temporality,
      parallelPolicy: parallelPolicy,
      forceNewKernel: forceNewKernel,
      threadCountScale: threadCountScale,
      threadCountOverride: threadCountOverride,
      hasOwnFrameLoop: hasOwnFrameLoop
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

  /// Resolves final block kind after backend-specific scalar constraints.
  private static func resolveEffectiveKind(
    backend: Backend,
    blockKind: Kind,
    bodyEffectiveKind: Kind,
    threadCountScale: Int?,
    ops: inout [UOp]
  ) -> Kind {
    let mustForceScalar =
      backend == .c && (threadCountScale != nil || bodyEffectiveKind == .scalar)
    guard !mustForceScalar else {
      for i in ops.indices {
        ops[i].kind = .scalar
      }
      return .scalar
    }

    return blockKind
  }

  /// Infers whether finalized scalar static blocks can launch with thread-level parallel policy.
  private static func inferParallelPolicy(
    kind: Kind,
    temporality: Temporality,
    ops: [UOp],
    statefulTensorDecision: StatefulTensorParallelPolicy.Decision
  ) -> ParallelPolicy {
    if statefulTensorDecision.enabled {
      return .tensorElementParallel
    }

    guard temporality == .static_, kind == .scalar else { return .serial }

    let hasParallelRange = ops.contains {
      if case .beginParallelRange = $0.op { return true }
      return false
    }

    return hasParallelRange ? .threadParallel : .serial
  }
}

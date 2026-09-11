import Foundation

/// Temporality propagation result containing both frame-based and hop-based node sets.
public struct TemporalityResult {
  public let frameBasedNodes: Set<NodeID>
  public let hopBasedNodes: [NodeID: (Int, NodeID)]
}

/// Namespace for temporality analysis and block temporality assignment.
enum TemporalityPass {}

extension TemporalityPass {
  /// Returns true when an op is safe to schedule at hop rate for a node that
  /// carries explicit hop metadata.
  ///
  /// Most ops listed here already self-gate internally. Tensor-shaped
  /// `.accum` / `.latch` are a special case: when a caller explicitly tags the
  /// node with `graph.nodeHopRate`, running the enclosing block only on hop
  /// frames is equivalent to their frame-rate form with zero/no-op work
  /// between hops, and avoids a full per-sample `[N]` pass.
  static func opEmitsFullHopGate(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> Bool {
    switch op {
    case .hopTensorNoise, .spectrumDelay, .spectrumDelayMod:
      return true
    case .accum, .latch, .historyRead, .historyWrite, .historyReadWrite:
      // Tensor history read/write, like tensor accum/latch, only need hop-rate
      // scheduling when the caller explicitly tags them (TensorHistory(hop:)).
      // Running the read/write block only on hop frames is equivalent to its
      // frame-rate form with no-op work between hops — the state simply holds.
      guard graph.nodeHopRate[nodeId] != nil,
        let node = graph.nodes[nodeId],
        case .tensor = node.shape
      else {
        return false
      }
      return true
    default:
      return false
    }
  }

  /// Returns true if an op is intrinsically frame-based (its value changes per frame).
  static func isIntrinsicallyFrameBased(_ op: LazyOp) -> Bool {
    switch op {
    case .phasor(_), .deterministicPhasor, .output(_), .accum(_), .input(_),
      .historyRead(_), .historyWrite(_), .historyReadWrite(_), .latch(_), .click(_),
      .noise(_), .tensorNoise(_, _, _), .hopTensorNoise(_, _, _),
      .temporalGradStore, .temporalGradRead,
      .spectrumDelay(_, _, _, _, _),
      .spectrumDelayMod(_, _, _, _, _),
      .overlapAdd(_, _, _, _, _),
      // Per-sample scalar gradient output: sums hop-window tape entries for
      // every frame, so it must never be promoted to hop-rate scheduling.
      .bufferViewGradRead(_, _):
      return true
    default:
      return false
    }
  }

  /// Returns intrinsic hop rate for ops that are natively hop-based.
  ///
  /// Current behavior: no op is intrinsically hop-based from a scheduling perspective.
  /// FFT/IFFT emit internal hop logic but remain frame-based producers with hop-based outputs.
  static func intrinsicHopRate(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> (Int, NodeID)? {
    _ = (op, graph, nodeId)
    return nil
  }

  /// Returns hop rate for nodes explicitly marked as hop-output producers.
  static func producesHopBasedOutput(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> (Int, NodeID)? {
    _ = op
    return graph.nodeHopRate[nodeId]
  }

  /// Infers node temporality from intrinsic op properties and input propagation.
  static func inferTemporality(graph: Graph, sortedNodes: [NodeID]) throws -> TemporalityResult {
    var frameBasedNodes = Set<NodeID>()
    var hopBasedNodes: [NodeID: (Int, NodeID)] = [:]
    var hopProducingNodes: [NodeID: (Int, NodeID)] = [:]

    for nodeId in sortedNodes {
      guard let node = graph.nodes[nodeId] else { continue }

      if let hopRate = producesHopBasedOutput(node.op, graph: graph, nodeId: nodeId) {
        hopProducingNodes[nodeId] = hopRate
        // Explicitly-tagged hop outputs fall into two buckets:
        //
        // 1. Pure / derived ops (`add`, `mul`, `cos`, tensor latches already
        //    held at hop rate, etc.) whose value only changes when their
        //    hop-rate inputs change. These are safe to schedule hop-based.
        //
        // 2. Intrinsically frame-based stateful ops (`latch`, `accum`, ...)
        //    that still need a stricter check. Only promote those when their
        //    own emit semantics make hop-rate scheduling safe.
        let shouldPromoteToHopBased =
          !isIntrinsicallyFrameBased(node.op)
          || opEmitsFullHopGate(node.op, graph: graph, nodeId: nodeId)

        if shouldPromoteToHopBased {
          hopBasedNodes[nodeId] = hopRate
        } else {
          frameBasedNodes.insert(nodeId)
        }
        continue
      }

      if isIntrinsicallyFrameBased(node.op) || graph.isMutableTensorAccess(node) {
        frameBasedNodes.insert(nodeId)
        continue
      }

      if let hopRate = intrinsicHopRate(node.op, graph: graph, nodeId: nodeId) {
        hopBasedNodes[nodeId] = hopRate
        continue
      }

      // Global reduction ops (sampleGradReduce, peekGradReduce, etc.)
      // aggregate across ALL frames internally. Their output is static,
      // so they should not propagate frame-based temporality downstream.
      let hasFrameBasedInput = node.inputs.contains { inputId in
        guard let inputNode = graph.nodes[inputId],
              !isGlobalReductionOp(inputNode.op) else { return false }
        return frameBasedNodes.contains(inputId) && !hopProducingNodes.keys.contains(inputId)
      }
      if hasFrameBasedInput {
        frameBasedNodes.insert(nodeId)
        continue
      }

      var hopInputRates: [(Int, NodeID)] = []
      for inputId in node.inputs {
        if let rate = hopBasedNodes[inputId] {
          hopInputRates.append(rate)
        } else if let rate = hopProducingNodes[inputId] {
          hopInputRates.append(rate)
        }
      }

      if !hopInputRates.isEmpty {
        let clocks = Set(hopInputRates.map { $0.1 })
        if clocks.count > 1, !clocks.isDisjoint(with: graph.eventClockNodes) {
          throw DGenError.compilationFailed(
            "Independent event/hop clocks must be latched to frame rate before combining them (node \(nodeId))")
        }
        let fastestRate = hopInputRates.min(by: { $0.0 < $1.0 })!
        hopBasedNodes[nodeId] = fastestRate

        let counterNode = fastestRate.1
        if !node.inputs.contains(counterNode) {
          graph.nodes[nodeId]?.temporalDependencies.append(counterNode)
        }
      }
    }

    // A scalar produced only on event frames also needs its clock at a
    // frame-rate consumer. The value tape may contain an earlier call's data
    // on skipped frames, so the read must be masked there just like a tensor.
    for id in sortedNodes {
      guard let node = graph.nodes[id] else { continue }
      for input in node.inputs {
        guard let rate = hopBasedNodes[input], graph.eventClockNodes.contains(rate.1),
          hopBasedNodes[id]?.1 != rate.1,
          !node.allDependencies.contains(rate.1) else { continue }
        graph.nodes[id]?.temporalDependencies.append(rate.1)
      }
    }

    // Propagate buffer position dependencies for circular sliding-window modes.
    var positionDeps = graph.nodePositionDep
    for nodeId in sortedNodes {
      guard let node = graph.nodes[nodeId] else { continue }
      for inputId in node.inputs {
        if let posNode = positionDeps[inputId], posNode != nodeId {
          positionDeps[nodeId] = posNode
          if !node.inputs.contains(posNode) && !node.temporalDependencies.contains(posNode) {
            graph.nodes[nodeId]?.temporalDependencies.append(posNode)
          }
        }
      }
    }

    return TemporalityResult(frameBasedNodes: frameBasedNodes, hopBasedNodes: hopBasedNodes)
  }

  /// Splits blocks that mix hop-rate and frame-rate work into homogeneous runs.
  ///
  /// A frame-based block's tensor region is emitted under one `beginHopCheck`
  /// as soon as any node in it is hop-based (`ShapeTransitionPlanner`), because
  /// hop-rate producers must not recompute between hops. Frame-rate nodes in
  /// the same region are then gated too — and a frame-rate node is frame-rate
  /// precisely because skipping it is wrong. The canonical victim is
  /// `hop chain -> latch -> per-sample consumer`: the tensor `latch` stops
  /// re-emitting its held value between hops, so its consumer sees the value
  /// only on hop frames and the signal collapses to hop-rate impulses.
  ///
  /// Splitting here, before tensor memory materialization, keeps the boundary
  /// value materialized and lets `assignBlockTemporality` label each part.
  /// Each part keeps the original block's frame order, so a sequential block
  /// stays sequential and its interleaved frame loop is preserved.
  ///
  /// Returns true when any block was split.
  static func splitMixedRateBlocks(
    blocks: inout [Block],
    context: IRContext,
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) -> Bool {
    guard !hopBasedNodes.isEmpty else { return false }
    var result: [Block] = []
    var didSplit = false
    var nextGroup = (blocks.compactMap { $0.sequentialFrameGroup }.max() ?? -1) + 1
    for block in blocks {
      let firstPart = result.count
      guard block.nodes.contains(where: { hopBasedNodes[$0] != nil }),
        block.nodes.contains(where: { frameBasedNodes.contains($0) })
      else {
        result.append(block)
        continue
      }
      // A part's element loop is sized from its own leading tensor shape, not
      // the shape the original block started with.
      func retagShape(_ part: inout Block) {
        guard part.shape != nil else { return }
        for nodeId in part.nodes {
          if case .tensor(let shape)? = context.g.nodes[nodeId]?.shape {
            part.shape = shape
            return
          }
        }
        // A scalar-only fragment no longer owns an element loop. Retaining
        // the parent's shape advances scalar histories once per former lane.
        part.shape = nil
        part.tensorIndex = nil
      }
      var part = block
      part.nodes = []
      var hasHop = false
      var hasFrame = false
      for nodeId in block.nodes {
        let isHop = hopBasedNodes[nodeId] != nil
        let isFrame = frameBasedNodes.contains(nodeId)
        if !part.nodes.isEmpty, (isHop && hasFrame) || (isFrame && hasHop) {
          retagShape(&part)
          result.append(part)
          didSplit = true
          part = block
          part.nodes = []
          // Each emitted block owns its element loop, so the new part needs
          // its own iterator variable.
          if block.tensorIndex != nil { part.tensorIndex = context.useVariable(src: nil) }
          hasHop = false
          hasFrame = false
        }
        part.nodes.append(nodeId)
        hasHop = hasHop || isHop
        hasFrame = hasFrame || isFrame
      }
      if !part.nodes.isEmpty {
        retagShape(&part)
        result.append(part)
      }
      if block.frameOrder == .sequential, result.count - firstPart > 1 {
        let group = block.sequentialFrameGroup ?? nextGroup
        if block.sequentialFrameGroup == nil { nextGroup += 1 }
        for index in firstPart..<result.count { result[index].sequentialFrameGroup = group }
      }
    }
    if didSplit { blocks = result }
    return didSplit
  }

  /// Assigns block temporality from member node temporality.
  static func assignBlockTemporality(
    blocks: inout [Block],
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) {
    for i in 0..<blocks.count {
      blocks[i].temporality = determineBlockTemporality(
        block: blocks[i],
        frameBasedNodes: frameBasedNodes,
        hopBasedNodes: hopBasedNodes
      )
    }
  }

  /// Determines one block's temporality.
  private static func determineBlockTemporality(
    block: Block,
    frameBasedNodes: Set<NodeID>,
    hopBasedNodes: [NodeID: (Int, NodeID)]
  ) -> Temporality {
    if block.nodes.contains(where: { frameBasedNodes.contains($0) }) {
      return .frameBased
    }

    let hopRates = block.nodes.compactMap { hopBasedNodes[$0] }
    if let firstRate = hopRates.first {
      let allSameHopSize = hopRates.allSatisfy { $0.0 == firstRate.0 }
      if allSameHopSize {
        return .hopBased(hopSize: firstRate.0, counterNode: firstRate.1)
      }
      return .frameBased
    }

    return .static_
  }
}

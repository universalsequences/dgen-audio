import Foundation

/// Cross-block fusion of `(sum (* a b))` reductions on the C backend.
///
/// Block formation typically schedules the elementwise `mul` and the scalar
/// `sum` reduce in different (adjacent) blocks, so the product is materialized
/// into a frame-major scratch tensor that a second scalar pass re-reads and
/// accumulates. This pass detects sums whose product operands are safe to read
/// from the reduce's own block and records a plan; `.sum` emission then reads
/// both operands directly in one reduction loop. When every consumer of the
/// `mul` is a fused sum, the mul (and its whole product materialization) is
/// skipped entirely.
///
/// An operand is safe to read from a different block than the mul's when it is:
/// - a frame-aware cell (per-frame slots, indexed by the current frame), or
/// - a circular sliding-window view over a persistent ring buffer sized
///   `>= maxFrameCount + windowSize - 1` (the bufferView sizing contract:
///   within one kernel run, writes for later frames never overwrite the
///   window any earlier frame reads), or
/// - a static data tensor (immutable after init).
///
/// Runs after TensorMemoryMaterializationPass (frame-awareness is final) and
/// before emission. Circular-window operands add their position node to the
/// sum's temporal dependencies so cross-block defineGlobal/loadGlobal wiring
/// carries the per-frame write head into the sum's block.
enum SumOfMulFusionPass {

  static func run(
    graph: Graph,
    ctx: IRContext,
    backend: Backend
  ) {
    guard backend == .c else { return }
    // Gradient tapes differentiate through the materialized product; keep the
    // conservative layout when a backward pass exists.
    guard graph.lastForwardNodeId == nil else { return }

    var fusedSumsByMul: [NodeID: [NodeID]] = [:]

    for (sumId, sumNode) in graph.nodes.sorted(by: { $0.key < $1.key }) {
      guard case .sum = sumNode.op,
        let mulId = sumNode.inputs.first,
        let mulNode = graph.nodes[mulId],
        case .mul = mulNode.op,
        mulNode.inputs.count == 2,
        case .tensor(let shape)? = mulNode.shape,
        let mulTensor = graph.nodeToTensor[mulId].flatMap({ graph.tensors[$0] }),
        mulTensor.shape == shape,
        let aTensor = graph.nodeToTensor[mulNode.inputs[0]].flatMap({ graph.tensors[$0] }),
        let bTensor = graph.nodeToTensor[mulNode.inputs[1]].flatMap({ graph.tensors[$0] }),
        aTensor.shape == shape,
        bTensor.shape == shape
      else { continue }

      // The reduce must run under the same hop gate as the mul: moving the
      // operand reads into the reduce's block must not change which frames
      // they execute on (a per-frame read of hop-gated scratch observes
      // mid-hop clobber).
      guard ctx.hopBasedNodes[sumId]?.1 == ctx.hopBasedNodes[mulId]?.1 else { continue }

      guard let aPos = crossBlockReadSafety(aTensor, graph: graph, ctx: ctx),
        let bPos = crossBlockReadSafety(bTensor, graph: graph, ctx: ctx)
      else { continue }

      ctx.fusedSumOfMulPlans[sumId] = IRContext.FusedSumOfMulPlan(
        aTensor: aTensor, bTensor: bTensor, shape: shape)
      fusedSumsByMul[mulId, default: []].append(sumId)

      // Wire circular-window position values (per-frame scalars produced in
      // the ring's block) into the sum's block.
      for posNode in [aPos.positionNode, bPos.positionNode].compactMap({ $0 }) {
        if var node = graph.nodes[sumId], !node.temporalDependencies.contains(posNode) {
          node.temporalDependencies.append(posNode)
          graph.nodes[sumId] = node
        }
      }
    }

    // Skip the mul's materialization only when every consumer reads inline AND
    // the host never reads the product tensor directly (realize()/materialized
    // tensors are observed from memory without a graph edge).
    for (mulId, fusedSums) in fusedSumsByMul {
      guard !graph.materializeNodes.contains(mulId),
        graph.nodeToTensor[mulId].flatMap({ graph.tensors[$0] })?.materialize != true
      else { continue }
      let fused = Set(fusedSums)
      let consumers = graph.nodes.values.filter { $0.inputs.contains(mulId) }
      if !consumers.isEmpty, consumers.allSatisfy({ fused.contains($0.id) }) {
        ctx.crossBlockSkippedTensorNodes.insert(mulId)
      }
    }
  }

  /// Whether a fused sum in another block may read this operand tensor, and
  /// the sliding-window position node (if any) that must be wired across.
  private struct OperandSafety {
    let positionNode: NodeID?
  }

  private static func crossBlockReadSafety(
    _ tensor: Tensor, graph: Graph, ctx: IRContext
  ) -> OperandSafety? {
    if let window = circularWindowPosition(tensor, graph: graph) {
      return OperandSafety(positionNode: window)
    }
    if tensor.transforms.isEmpty, ctx.frameAwareTensorCells.contains(tensor.cellId) {
      return OperandSafety(positionNode: nil)
    }
    if tensor.transforms.isEmpty, tensor.data != nil {
      return OperandSafety(positionNode: nil)
    }
    return nil
  }

  /// Matches a circular sliding-window view over a persistent 1D ring buffer
  /// whose size honors the cross-frame lookback contract
  /// (`bufSize >= maxFrameCount + windowSize - 1`). Returns its position node.
  private static func circularWindowPosition(_ tensor: Tensor, graph: Graph) -> NodeID? {
    guard case .slidingWindow(let windowSize, let inputShape, let positionNode)? =
      tensor.transforms.first,
      let posNode = positionNode
    else { return nil }
    for transform in tensor.transforms.dropFirst() {
      guard case .reshape = transform else { return nil }
    }
    let count = tensor.shape.reduce(1, *)
    guard windowSize == count, count > 0 else { return nil }
    let bufSize = inputShape.last ?? 0
    guard inputShape.reduce(1, *) == bufSize,
      bufSize >= graph.maxFrameCount + windowSize - 1,
      tensor.offset == 0,
      graph.persistentCells.contains(tensor.cellId),
      graph.frameAwareCells[tensor.cellId] == nil
    else { return nil }
    return posNode
  }
}

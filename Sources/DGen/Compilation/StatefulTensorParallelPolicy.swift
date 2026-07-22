import Foundation

/// Shared eligibility checks for scalar frame-loop kernels that can safely parallelize
/// across tensor elements (`id < tensorSize`) while keeping a sequential frame loop.
enum StatefulTensorParallelPolicy {
  struct Decision {
    let enabled: Bool
    let tensorSize: Int
  }

  /// Strict predicate:
  /// - Metal backend only
  /// - scalar block with single known tensor shape
  /// - frameBased or hopBased temporality
  /// - contains stateful tensor-friendly ops (phasor/accum) or lane-safe tensor history
  ///
  /// Tensor-history blocks (the batched biquad recurrence) are eligible when the
  /// full `laneParallelizable` audit passes AND the block has a single shape
  /// region (multi-shape blocks are emitted with region element loops, which
  /// would double-iterate under `id < tensorSize` dispatch).
  static func decide(block: Block, graph: Graph, backend: Backend) -> Decision {
    guard backend == .metal else { return Decision(enabled: false, tensorSize: 0) }
    guard block.frameOrder == .sequential else { return Decision(enabled: false, tensorSize: 0) }
    guard let shape = block.shape else { return Decision(enabled: false, tensorSize: 0) }
    let tensorSize = shape.reduce(1, *)
    guard tensorSize > 1 else { return Decision(enabled: false, tensorSize: tensorSize) }

    switch block.temporality {
    case .frameBased, .hopBased:
      break
    case .static_:
      return Decision(enabled: false, tensorSize: tensorSize)
    }

    var hasCandidate = false
    var hasTensorHistory = false
    for nodeId in block.nodes {
      guard let node = graph.nodes[nodeId] else { continue }
      switch node.op {
      case .phasor(_), .accum(_):
        hasCandidate = true
      case .historyReadWrite(let cellId):
        // delay1-style read-modify-write cells stay on the strict path.
        if graph.cellToTensor[cellId] != nil {
          return Decision(enabled: false, tensorSize: tensorSize)
        }
      case .historyRead(let cellId), .historyWrite(let cellId):
        if graph.cellToTensor[cellId] != nil {
          hasTensorHistory = true
        }
      default:
        break
      }
    }

    if hasTensorHistory {
      // Per-lane tensor history (batched biquad forward): safe only when every
      // stateful/write op in the block is element-indexed and the block emits
      // as one shape region (no per-region element loops).
      guard block.temporality == .frameBased,
        detectShapeTransitions(block: block, g: graph).count <= 1,
        laneParallelizable(block: block, graph: graph, tensorSize: tensorSize)
      else {
        return Decision(enabled: false, tensorSize: tensorSize)
      }
      return Decision(enabled: true, tensorSize: tensorSize)
    }

    return Decision(enabled: hasCandidate, tensorSize: tensorSize)
  }

  /// Lane-parallel decision for the detached BPTT backward block (the
  /// consolidated tensor-history reverse recurrence).
  ///
  /// When enabled, the block is emitted with the region element index bound to
  /// the thread id (no per-region element loops) and dispatched as
  /// `.selfManagedThreads(W)`: W threads, each running the reverse frame loop
  /// over its own lane's carry/history state.
  ///
  /// Consulted identically by block emission and dispatch-mode finalization —
  /// both must agree or the kernel would run each element loop once per thread
  /// (or leave lanes 1..W-1 uncomputed).
  static func decideDetachedBPTTBackward(block: Block, graph: Graph, backend: Backend) -> Decision {
    guard backend == .metal else { return Decision(enabled: false, tensorSize: 0) }
    guard block.frameOrder == .sequential else { return Decision(enabled: false, tensorSize: 0) }
    guard block.temporality == .frameBased else { return Decision(enabled: false, tensorSize: 0) }
    guard let shape = block.shape else { return Decision(enabled: false, tensorSize: 0) }
    let tensorSize = shape.reduce(1, *)
    guard tensorSize > 1 else { return Decision(enabled: false, tensorSize: tensorSize) }

    // Must be the detached backward recurrence AND take the shape-aware
    // emission path (transitions > 1), where the lane-parallel region binding
    // is implemented.
    guard blockIsDetachedBPTTBackward(block: block, g: graph),
      detectShapeTransitions(block: block, g: graph).count > 1,
      laneParallelizable(block: block, graph: graph, tensorSize: tensorSize)
    else {
      return Decision(enabled: false, tensorSize: tensorSize)
    }
    return Decision(enabled: true, tensorSize: tensorSize)
  }

  /// Conservative audit: may this sequential frame-loop block run with one
  /// thread per tensor lane?
  ///
  /// Requirements (any failure degrades to today's correct single-threaded
  /// emission, never to wrong answers):
  /// - every stateful cell touched is tensor-registered with element-indexed
  ///   access (tensor history cells, tensor grad-carry cells);
  /// - no scalar-cell stateful ops (scalar accum/phasor/latch, click, noise,
  ///   delay1) — a shared PRNG or single-cell state cannot be duplicated
  ///   per thread;
  /// - no hop-gated nodes (hop feedback keeps the strict sequential path);
  /// - every memory write is element-indexed (tensor-shaped write node or a
  ///   tensor grad-carry cell);
  /// - no cross-lane addressing: no gather/peek/selectRow-style dynamic
  ///   element indices, no element-axis-permuting views, no reductions, and
  ///   no self-iterating ops (FFT, conv, overlapAdd, ...);
  /// - every tensor-shaped node has exactly `tensorSize` elements.
  ///
  /// Scalar VALUE nodes (constants, pure math, memory/tape reads) are allowed:
  /// each thread computes them redundantly, which is safe because they are
  /// read-only.
  static func laneParallelizable(block: Block, graph: Graph, tensorSize: Int) -> Bool {
    for nodeId in block.nodes {
      guard let node = graph.nodes[nodeId] else { continue }

      // Hop-gated nodes keep the strict frame-by-frame path.
      if graph.nodeHopRate[nodeId] != nil { return false }

      let isTensorShaped: Bool
      let nodeElementCount: Int
      if case .tensor(let nodeShape) = node.shape {
        isTensorShaped = true
        nodeElementCount = nodeShape.reduce(1, *)
      } else {
        isTensorShaped = false
        nodeElementCount = 1
      }

      // A tensor node with a different element count than the block width
      // would be mis-indexed by the per-lane thread id.
      if isTensorShaped && nodeElementCount != tensorSize { return false }

      switch node.op {
      // Per-lane tensor state is the whole point; scalar history is a race.
      case .historyRead(let cellId), .historyWrite(let cellId):
        if graph.cellToTensor[cellId] == nil { return false }
      case .historyReadWrite:
        return false

      // Scalar single-cell stateful ops cannot be duplicated per thread.
      // Tensor-shaped phasor/accum/latch keep per-element state and are safe.
      case .accum, .phasor, .latch:
        if !isTensorShaped { return false }
      case .click, .noise:
        return false

      // Writes must be element-indexed: either the write node itself is
      // tensor-shaped (region emission indexes it by the lane id) or it
      // targets a per-lane grad-carry cell.
      case .memoryWrite(let cellId), .memoryAccumulate(let cellId):
        if !isTensorShaped && !graph.tensorGradCarryCells.contains(cellId) { return false }
      case .memoryCellSum:
        return false

      // Reductions must live outside the recurrence block (they are closure
      // boundaries already — assert, don't assume).
      case .sum, .sumAxis, .sumMulAxis0, .maxAxis, .meanAxis, .mse:
        return false

      // Cross-lane / dynamic element addressing. (`.selector` is NOT listed:
      // it is a pure elementwise value select used for biquad mode switching.)
      case .peek, .selectRow, .sampleInline,
        .peekGradWrite, .peekGradReduce,
        .sampleGradWrite, .sampleGradReduce,
        .selectRowGradWrite, .selectRowGradReduce:
        return false

      // Element-axis-permuting or offset views break lane == element identity.
      case .transpose, .shrink, .pad, .asStrided, .repeatView:
        return false

      // Per-frame scalar output writes.
      case .output:
        return false

      case .temporalGradStore, .temporalGradScan, .temporalGradRead:
        return false

      default:
        // Self-iterating ops (FFT, conv, gemm, overlapAdd, tensorAccumulate,
        // spectral loss kernels, ...) own their dispatch/iteration and must
        // not run once per lane thread. tensorRef emits no code and seq just
        // forwards a value — both safe.
        switch node.op {
        case .tensorRef, .seq:
          break
        default:
          if node.op.emitsInternalIteration { return false }
        }
      }
    }
    return true
  }
}

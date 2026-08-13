/// Block partitioning, fusion, spectral isolation, tensor/reduce/memory splitting.
import Foundation

// Block partitioning that works with feedback group-aware sorted nodes
public func partitionIntoBlocks(
  sorted: [NodeID], scalar: Set<NodeID>, g: Graph,
  debug: Bool = false
) -> [Block] {
  var blocks: [Block] = []
  var currentBlock: Block? = nil

  for nodeId in sorted {
    let isScalar = scalar.contains(nodeId)
    let frameOrder: FrameOrder = isScalar ? .sequential : .parallel

    // Special handling for output nodes - they go in the same block as their dependencies
    if let node = g.nodes[nodeId], case .output = node.op {
      // Find the block containing the first dependency
      var targetBlockIdx = -1
      for inputID in node.allDependencies {
        for (blockIdx, block) in blocks.enumerated() {
          if block.nodes.contains(inputID) {
            targetBlockIdx = blockIdx
            break
          }
        }
        if targetBlockIdx != -1 { break }
      }

      if targetBlockIdx != -1 && targetBlockIdx < blocks.count {
        blocks[targetBlockIdx].nodes.append(nodeId)
        if debug {
          print(
            "📍 Placed output node \(nodeId) in block \(targetBlockIdx) with its dependency"
          )
        }
      } else if currentBlock != nil {
        currentBlock!.nodes.append(nodeId)
        if debug {
          print("📍 Placed output node \(nodeId) in current block")
        }
      } else {
        currentBlock = Block(frameOrder: frameOrder)
        currentBlock!.nodes.append(nodeId)
      }
      continue
    }

    // Regular node handling - group consecutive nodes of same kind together
    if let current = currentBlock {
      if current.frameOrder == frameOrder {
        currentBlock!.nodes.append(nodeId)
      } else {
        blocks.append(current)
        currentBlock = Block(frameOrder: frameOrder)
        currentBlock!.nodes.append(nodeId)
      }
    } else {
      currentBlock = Block(frameOrder: frameOrder)
      currentBlock!.nodes.append(nodeId)
    }
  }

  // Don't forget the last block
  if let current = currentBlock {
    blocks.append(current)
  }

  // Remove empty blocks (can arise from special placement of outputs or limits)
  blocks.removeAll { $0.nodes.isEmpty }

  if debug {
    print("📦 Created \(blocks.count) blocks")
    for (i, block) in blocks.enumerated() {
      print("  Block \(i) (\(block.frameOrder)): \(block.nodes)")
    }
  }

  return blocks
}

/// Merge consecutive blocks of the same kind into a single block.
public func fuseBlocks(_ blocks: [Block]) -> [Block] {
  var fused: [Block] = []
  for b in blocks {
    if b.nodes.isEmpty { continue }
    if let lastIdx = fused.indices.last, fused[lastIdx].frameOrder == b.frameOrder {
      fused[lastIdx].nodes.append(contentsOf: b.nodes)
    } else {
      fused.append(b)
    }
  }
  return fused
}

/// Isolate special passes into their own blocks to prevent unsafe fusion.
/// Includes:
/// - FFT-based spectral ops (shared scratch / race-avoidance)
/// - scalar side-effect grad-write ops that must not inherit tensor ThreadCountScale
/// Preserves ordering of other nodes.
public func isolateSpectralPasses(_ blocks: [Block], _ g: Graph) -> [Block] {
  var result: [Block] = []

  // Helper to create a new block with the same properties as the original
  func makeBlock(from original: Block, nodes: [NodeID]) -> Block {
    var newBlock = Block(frameOrder: original.frameOrder)
    newBlock.nodes = nodes
    newBlock.temporality = original.temporality
    newBlock.tensorIndex = original.tensorIndex
    newBlock.shape = original.shape
    return newBlock
  }

  for block in blocks {
    var currentNodes: [NodeID] = []

    for nodeId in block.nodes {
      let isIsolatedPass = { () -> Bool in
        guard let node = g.nodes[nodeId] else { return false }
        // FFT-based spectral loss ops need isolation because they use shared scratch
        // memory for FFT computation that can't be safely accessed by multiple SIMD threads
        if case .spectralLossFFT = node.op { return true }
        if case .spectralLossFFTGradInline = node.op { return true }
        if case .spectralLossFFTGradSpec = node.op { return true }
        if case .spectralLossFFTGradIFFT = node.op { return true }
        if case .spectralLossFFTGradRead = node.op { return true }
        if case .spectralLossFFTGradRead2 = node.op { return true }
        if case .spectralLossFFTBatched = node.op { return true }
        if case .spectralLossFFTBatchedReduce = node.op { return true }
        if case .spectralLossFFTBatchedGradSpec = node.op { return true }
        if case .spectralLossFFTBatchedGradIFFT = node.op { return true }
        if case .spectralLossFFTBatchedGradRead = node.op { return true }
        if case .spectralLossFFTBatchedGradRead2 = node.op { return true }
        // Scalar side-effect write passes must not inherit tensor ThreadCountScale.
        // If fused into a tensor-shaped region they can be massively over-dispatched.
        if case .sampleGradWrite = node.op { return true }
        if case .selectRowGradWrite = node.op { return true }
        if case .peekGradWrite = node.op { return true }
        if case .temporalGradStore = node.op { return true }
        return false
      }()

      if isIsolatedPass {
        // Flush any accumulated nodes before the spectral pass
        if !currentNodes.isEmpty {
          result.append(makeBlock(from: block, nodes: currentNodes))
          currentNodes = []
        }

        // Add isolated pass in its own SIMD block.
        // Clear shape/tensorIndex so it never inherits parent ThreadCountScale.
        var spectralBlock = makeBlock(from: block, nodes: [nodeId])
        spectralBlock.frameOrder = .parallel
        spectralBlock.shape = nil
        spectralBlock.tensorIndex = nil
        result.append(spectralBlock)
      } else {
        // Accumulate non-spectral nodes
        currentNodes.append(nodeId)
      }
    }

    // Flush any remaining nodes after the last spectral pass
    if !currentNodes.isEmpty {
      result.append(makeBlock(from: block, nodes: currentNodes))
    }
  }

  return result
}

public func isReductionOp(_ op: LazyOp) -> Bool {
  switch op {
  case .sum, .tensorAccumulate, .sampleGradReduce,
    .selectRowGradReduce, .peekGradReduce, .overlapAddGradGather, .bufferViewGradRead,
    .temporalGradScan,
    .sumMulAxis0, .gemmSmall, .chunkPartialsReduceToCell:
    return true
  default:
    return false
  }
}

public func isGlobalReductionOp(_ op: LazyOp) -> Bool {
  switch op {
  case .sampleGradReduce, .selectRowGradReduce, .peekGradReduce,
    .tensorAccumulate, .chunkPartialsReduceToCell, .temporalGradScan:
    return true
  default:
    return false
  }
}

/// Axis reduces (sumAxis, maxAxis, meanAxis) can be fused with their input
/// because each output element reduces over an independent slice — no cross-thread barrier needed.
public func isAxisReduceOp(_ op: LazyOp) -> Bool {
  switch op {
  case .sumAxis, .maxAxis, .meanAxis:
    return true
  default:
    return false
  }
}

/// Detect concat-by-padding joins represented as add trees over padded tensors:
/// `add(...add(pad(x0), pad(x1))..., pad(xN))`.
/// Keeping this shape transition inside the current block fuses butterfly math
/// with its pad+concat join into one kernel.
private func isConcatByPaddingFusionTransition(
  nodeId: NodeID,
  graph: Graph,
  currentShape: Shape
) -> Bool {
  guard let node = graph.nodes[nodeId] else { return false }
  guard case .add = node.op else { return false }
  guard case .tensor(let outputShape) = node.shape else { return false }
  guard outputShape.count == currentShape.count else { return false }

  func collectPadLeaves(_ currentNodeId: NodeID) -> [[(Int, Int)]]? {
    guard let currentNode = graph.nodes[currentNodeId] else { return nil }

    switch currentNode.op {
    case .add:
      guard currentNode.inputs.count == 2 else { return nil }
      guard let left = collectPadLeaves(currentNode.inputs[0]),
        let right = collectPadLeaves(currentNode.inputs[1])
      else { return nil }
      return left + right

    case .pad(let padding):
      guard currentNode.inputs.count == 1 else { return nil }
      guard padding.count == currentShape.count else { return nil }
      guard let padInputNode = graph.nodes[currentNode.inputs[0]],
        case .tensor(let padInputShape) = padInputNode.shape,
        padInputShape == currentShape
      else { return nil }
      guard case .tensor(let padOutputShape) = currentNode.shape,
        padOutputShape == outputShape
      else { return nil }
      return [padding]

    default:
      return nil
    }
  }

  guard let paddings = collectPadLeaves(nodeId), paddings.count >= 2 else { return false }

  var concatAxis: Int? = nil

  for axis in 0..<currentShape.count {
    let hasPaddingOnAxis = paddings.contains { padding in
      let p = padding[axis]
      return p != (0, 0)
    }
    if hasPaddingOnAxis {
      guard concatAxis == nil else { return false }
      concatAxis = axis
    }
  }

  guard let axis = concatAxis else { return false }

  for dim in 0..<outputShape.count where dim != axis {
    guard outputShape[dim] == currentShape[dim] else { return false }
  }

  let baseConcatDim = currentShape[axis]
  let outputConcatDim = outputShape[axis]
  guard outputConcatDim > baseConcatDim else { return false }

  struct Segment {
    let start: Int
    let end: Int
  }

  var segments: [Segment] = []
  segments.reserveCapacity(paddings.count)

  for padding in paddings {
    // Non-concat axes must be unchanged.
    for dim in 0..<padding.count where dim != axis {
      guard padding[dim] == (0, 0) else { return false }
    }

    let concatPadding = padding[axis]
    let start = concatPadding.0
    let end = start + baseConcatDim
    let expectedDim = start + baseConcatDim + concatPadding.1

    guard start >= 0, concatPadding.1 >= 0 else { return false }
    guard end <= outputConcatDim else { return false }
    guard expectedDim == outputConcatDim else { return false }

    segments.append(Segment(start: start, end: end))
  }

  // Require a full non-overlapping cover of the concat dimension.
  segments.sort { lhs, rhs in
    if lhs.start != rhs.start { return lhs.start < rhs.start }
    return lhs.end < rhs.end
  }

  var cursor = 0
  for segment in segments {
    guard segment.start == cursor else { return false }
    guard segment.end > segment.start else { return false }
    cursor = segment.end
  }

  return cursor == outputConcatDim
}

public func splitReduceBlocks(g: Graph, blocks: [Block]) -> [Block] {
  var splitBlocks: [Block] = []

  for block in blocks {
    let reductionOpIndex = block.nodes.firstIndex { nodeId in
      guard let node = g.nodes[nodeId] else { return false }
      if block.frameOrder == .sequential {
        // Scalar frame-loop blocks must stay intact for feedback/stateful ops,
        // except global reductions which must run once outside the frame loop.
        return isGlobalReductionOp(node.op)
      }
      return isReductionOp(node.op)
    }

    guard let reductionOpIndex else {
      splitBlocks.append(block)
      continue
    }

    // Pre-reduction block
    if reductionOpIndex > 0 {
      var preReductionBlock = Block(frameOrder: block.frameOrder)
      preReductionBlock.nodes = Array(block.nodes[0..<reductionOpIndex])
      preReductionBlock.shape = block.shape
      preReductionBlock.temporality = block.temporality
      splitBlocks.append(preReductionBlock)
    }

    // Reduction block
    // Global reduces run once total, not per-frame:
    // - sampleGradReduce/selectRowGradReduce/peekGradReduce: reduction ops with internal frame loops
    // - tensorAccumulate: atomic gradient accumulation, loops over frames internally
    let reductionNode = g.nodes[block.nodes[reductionOpIndex]]
    let isGlobalReduce = reductionNode.map { isGlobalReductionOp($0.op) } ?? false

    // Global reduces need frameOrder=.sequential AND temporality=.static_ to run once total
    var reductionBlock = Block(frameOrder: isGlobalReduce ? .sequential : block.frameOrder)
    reductionBlock.nodes = [block.nodes[reductionOpIndex]]
    reductionBlock.temporality = isGlobalReduce ? .static_ : block.temporality

    // Set output shape for tensor reductions (enables thread scaling)
    // Skip for global reduces - they loop internally over all frames
    if let reductionNode, case .tensor(let outputShape) = reductionNode.shape, !isGlobalReduce {
      reductionBlock.shape = outputShape
    }

    splitBlocks.append(reductionBlock)

    // Post-reduction block - recursively split if it contains more reductions
    if reductionOpIndex < block.nodes.count - 1 {
      let postNodes = Array(block.nodes[reductionOpIndex + 1..<block.nodes.count])
      let canPromotePostToSIMD = block.frameOrder == .sequential && isGlobalReduce

      var postReductionBlock = Block(frameOrder: canPromotePostToSIMD ? .parallel : block.frameOrder)
      postReductionBlock.nodes = postNodes
      postReductionBlock.shape = block.shape
      postReductionBlock.temporality = block.temporality
      // Recursively split the post-reduction block in case it has more reductions
      let furtherSplit = splitReduceBlocks(g: g, blocks: [postReductionBlock])
      splitBlocks.append(contentsOf: furtherSplit)
    }
  }

  return splitBlocks
}

/// The set of backward-partition nodes that must execute inside the single
/// reverse frame loop of a vector-width BPTT (tensor-history) backward pass.
///
/// Includes:
/// - every memoryWrite of a tensor grad carry cell, and its backward ancestors
///   (the grad arithmetic feeding the temporal recurrence);
/// - every memoryRead of a tensor grad carry cell, and its backward
///   descendants (per-frame gradients that combine the carry with the
///   downstream grad — reading the single-slot carry ring outside the reverse
///   loop would see only the final frame's value);
/// - backward ancestors of all of the above, so the block is closed over its
///   inputs.
///
/// The walk stops at reductions to scalar and grad accumulates: those are not
/// part of the recurrence (they reach carry writes only via seq side-effect
/// ordering chains) and read their tensor inputs from frame-aware cells in
/// later blocks.
public func tensorBPTTRecurrenceClosure(g: Graph, lastForwardId: NodeID) -> Set<NodeID> {
  func isBoundary(_ id: NodeID) -> Bool {
    guard let node = g.nodes[id] else { return true }
    switch node.op {
    case .sum, .sumAxis, .memoryAccumulate, .tensorAccumulate:
      return true
    // Isolated-pass ops (spectral loss forward/backward, grad tapes) run in
    // their own kernels with dedicated scratch and dispatch shapes. They
    // materialize the upstream per-frame gradient before the recurrence; the
    // consolidated reverse loop reads their outputs from frame-aware cells.
    case .spectralLossFFT, .spectralLossFFTGradSpec, .spectralLossFFTGradIFFT,
      .spectralLossFFTGradInline, .spectralLossFFTGradRead, .spectralLossFFTGradRead2,
      .spectralLossFFTBatched, .spectralLossFFTBatchedReduce,
      .spectralLossFFTBatchedGradSpec, .spectralLossFFTBatchedGradIFFT,
      .spectralLossFFTBatchedGradRead, .spectralLossFFTBatchedGradRead2,
      .sampleGradWrite, .selectRowGradWrite, .peekGradWrite,
      .temporalGradStore, .temporalGradRead:
      return true
    default:
      return false
    }
  }

  // seq nodes exist to ORDER gradient side effects; only their second input
  // is a value dependency. Walking the ordering edge would sweep unrelated
  // grad chains (whatever side effect happened to be chained first) into the
  // recurrence, nondeterministically per dictionary iteration order.
  func valueDeps(_ node: Node) -> [NodeID] {
    if case .seq = node.op, node.inputs.count == 2 {
      return [node.inputs[1]]
    }
    return node.allDependencies
  }

  var seeds: [NodeID] = []
  var carryReads: [NodeID] = []
  for (id, node) in g.nodes where id > lastForwardId {
    switch node.op {
    case .memoryWrite(let c) where g.tensorGradCarryCells.contains(c):
      seeds.append(id)
    case .memoryRead(let c) where g.tensorGradCarryCells.contains(c):
      seeds.append(id)
      carryReads.append(id)
    default:
      break
    }
  }
  guard !seeds.isEmpty else { return [] }

  // Consumer adjacency over the backward partition (value edges only).
  var consumers: [NodeID: [NodeID]] = [:]
  for (id, node) in g.nodes where id > lastForwardId {
    for dep in valueDeps(node) where dep > lastForwardId {
      consumers[dep, default: []].append(id)
    }
  }

  // Descendants of carry reads, stopping at (and excluding) boundary nodes.
  var descendantStack = carryReads
  var descendants = Set<NodeID>()
  while let id = descendantStack.popLast() {
    for consumer in consumers[id] ?? [] where !descendants.contains(consumer) {
      guard !isBoundary(consumer) else { continue }
      descendants.insert(consumer)
      descendantStack.append(consumer)
    }
  }

  // Ancestor closure over seeds + descendants, stopping at boundary nodes.
  var stack = seeds + Array(descendants)
  var closure = Set<NodeID>()
  while let id = stack.popLast() {
    guard closure.insert(id).inserted else { continue }
    guard let node = g.nodes[id] else { continue }
    for dep in valueDeps(node) where dep > lastForwardId && !isBoundary(dep) {
      stack.append(dep)
    }
  }
  return closure
}

/// Consolidate the vector-width BPTT recurrence (tensor-history biquad
/// backward) into one sequential block.
///
/// The reverse-time recurrence — carry reads, the grad arithmetic between
/// them, and the carry writes — must execute inside a single reverse frame
/// loop (the detached-BPTT wrap in BlockEmission). Default block formation
/// fragments these tensor-shaped backward nodes across several blocks, which
/// both breaks the recurrence and leaves backward UOps referencing forward
/// loop-scope variables. This pass extracts the closure of backward-partition
/// ancestors of every tensor carry write into one new sequential block,
/// inserted where the first of those nodes originally lived.
///
/// Safe by construction: closure nodes never depend on non-closure backward
/// nodes (the closure is ancestor-closed within the backward partition), and
/// consumers of closure outputs have higher node IDs so they sit in blocks at
/// or after the insertion point.
public func consolidateTensorBPTTBackwardBlocks(g: Graph, blocks: [Block], ctx: IRContext) -> [Block] {
  let debug = ProcessInfo.processInfo.environment["DGEN_DEBUG_BPTT_SPLIT"] != nil
  guard let lastForwardId = g.lastForwardNodeId, !g.tensorGradCarryCells.isEmpty else {
    return blocks
  }

  var stack: [NodeID] = []
  for (id, node) in g.nodes where id > lastForwardId {
    if case .memoryWrite(let c) = node.op, g.tensorGradCarryCells.contains(c) {
      stack.append(id)
    }
  }
  let closure = tensorBPTTRecurrenceClosure(g: g, lastForwardId: lastForwardId)
  guard !closure.isEmpty else { return blocks }

  // Forward nodes whose values the closure consumes: the consolidated block
  // must be inserted after the last block that produces any of them (block
  // fusion can reorder blocks relative to node-ID order).
  var forwardDeps = Set<NodeID>()
  for id in closure {
    guard let node = g.nodes[id] else { continue }
    for dep in node.allDependencies where dep <= lastForwardId {
      forwardDeps.insert(dep)
    }
  }

  // Blocks whose forward half contains tensor-registered history ops: any
  // backward leftovers here would trigger the scalar wrapWithBPTTLoops path,
  // which cannot re-emit tensor UOps in its reverse loop. Their backward
  // nodes are deferred to after the consolidated block instead.
  func hasTensorHistory(_ block: Block) -> Bool {
    block.nodes.contains { nodeId in
      guard let node = g.nodes[nodeId] else { return false }
      switch node.op {
      case .historyRead(let c), .historyWrite(let c):
        return g.cellToTensor[c] != nil
      default:
        return false
      }
    }
  }

  // Backward deps of the closure that were cut out of it (boundary nodes) must
  // run before the consolidated block, alongside its forward deps.
  var beforeDeps = forwardDeps
  for id in closure {
    guard let node = g.nodes[id] else { continue }
    for dep in node.allDependencies where dep > lastForwardId && !closure.contains(dep) {
      beforeDeps.insert(dep)
    }
  }

  // Pass 1: strip closure nodes; defer backward leftovers of tensor-history
  // blocks (they would otherwise trigger the scalar wrapWithBPTTLoops path).
  // Nodes the closure itself depends on are never deferred.
  var stripped: [Block] = []
  var deferredBackward: [Block] = []
  var deferredNodes = Set<NodeID>()
  var templateTemporality: Temporality? = nil
  var sawClosure = false
  for block in blocks {
    let hasClosure = block.nodes.contains { closure.contains($0) }
    if hasClosure {
      sawClosure = true
      if templateTemporality == nil, block.temporality == .frameBased {
        templateTemporality = block.temporality
      }
    }
    var rest = block.nodes.filter { !closure.contains($0) }
    if hasClosure || hasTensorHistory(block) {
      let deferred = rest.filter { $0 > lastForwardId && !beforeDeps.contains($0) }
      if !deferred.isEmpty {
        var deferredBlock = Block(frameOrder: block.frameOrder)
        deferredBlock.nodes = deferred
        deferredBlock.shape = block.shape
        deferredBlock.temporality = block.temporality
        deferredBlock.tensorIndex = block.tensorIndex
        deferredBackward.append(deferredBlock)
        deferredNodes.formUnion(deferred)
        rest = rest.filter { !deferredNodes.contains($0) }
      }
    }
    if !rest.isEmpty {
      var remainder = block
      remainder.nodes = rest
      stripped.append(remainder)
    }
  }
  guard sawClosure else { return blocks }

  // Transitive consumers of anything that now runs late (closure + deferred
  // leftovers) must also run after the consolidated block.
  var consumers: [NodeID: [NodeID]] = [:]
  for (id, node) in g.nodes where id > lastForwardId {
    for dep in node.allDependencies {
      consumers[dep, default: []].append(id)
    }
  }
  let lateSet = closure.union(deferredNodes)
  var descendants = Set<NodeID>()
  var descendantStack = Array(lateSet)
  while let id = descendantStack.popLast() {
    for consumer in consumers[id] ?? [] where !lateSet.contains(consumer) {
      if descendants.insert(consumer).inserted {
        descendantStack.append(consumer)
      }
    }
  }

  var bpttBlock = Block(frameOrder: .sequential)
  // Backward node IDs are created in dependency order, so ID order is a valid
  // topological order for the consolidated body.
  bpttBlock.nodes = closure.sorted()
  bpttBlock.temporality = templateTemporality ?? .frameBased
  assignTensorIndexFromFirstTensorNode(to: &bpttBlock, graph: g, ctx: ctx)

  // Pass 2: the consolidated block goes after every block containing one of
  // its dependencies; any earlier block holding closure descendants has those
  // nodes split out and moved after it.
  var insertAt = 0
  for (idx, block) in stripped.enumerated()
  where block.nodes.contains(where: { beforeDeps.contains($0) }) {
    insertAt = idx + 1
  }

  var before: [Block] = []
  var after: [Block] = []
  var movedConsumers: [Block] = []
  for (idx, block) in stripped.enumerated() {
    if idx >= insertAt {
      after.append(block)
      continue
    }
    let moved = block.nodes.filter { descendants.contains($0) }
    if moved.isEmpty {
      before.append(block)
      continue
    }
    let stay = block.nodes.filter { !descendants.contains($0) }
    if !stay.isEmpty {
      var stayBlock = block
      stayBlock.nodes = stay
      before.append(stayBlock)
    }
    var movedBlock = Block(frameOrder: block.frameOrder)
    movedBlock.nodes = moved
    movedBlock.shape = block.shape
    movedBlock.temporality = block.temporality
    movedBlock.tensorIndex = block.tensorIndex
    movedConsumers.append(movedBlock)
  }

  // Moved and deferred fragments are all backward nodes; order them by node ID
  // (creation order is dependency order for backward nodes).
  let tail = (movedConsumers + deferredBackward).sorted {
    ($0.nodes.min() ?? 0) < ($1.nodes.min() ?? 0)
  }
  let result = before + [bpttBlock] + tail + after
  if debug {
    print(
      "BPTT-CONSOLIDATE nodes=\(bpttBlock.nodes.count) at=\(before.count) "
        + "moved=\(movedConsumers.map { $0.nodes.count }) "
        + "deferred=\(deferredBackward.map { $0.nodes.count }) "
        + "shape=\(String(describing: bpttBlock.shape)) temporality=\(bpttBlock.temporality)")
  }
  return result
}

/// Split SIMD blocks where a memoryRead depends on a memoryWrite to the same
/// base cell. Without a kernel boundary, all frames execute simultaneously and
/// reads may see unwritten data. Follows the same pattern as splitReduceBlocks.
public func splitMemoryBlocks(g: Graph, blocks: [Block]) -> [Block] {
  var result: [Block] = []
  for block in blocks {
    if block.frameOrder == .sequential {
      result.append(block)
      continue
    }
    var writtenCells: Set<CellID> = []
    var splitIndex: Int? = nil
    for (i, nodeId) in block.nodes.enumerated() {
      guard let node = g.nodes[nodeId] else { continue }
      switch node.op {
      case .memoryWrite(let base): writtenCells.insert(base)
      case .memoryRead(let base):
        if writtenCells.contains(base) {
          splitIndex = i
          break
        }
      default: break
      }
      if splitIndex != nil { break }
    }
    guard let splitIndex else {
      result.append(block)
      continue
    }
    // Pre-read block (includes memoryWrite)
    if splitIndex > 0 {
      var pre = Block(frameOrder: .parallel)
      pre.nodes = Array(block.nodes[0..<splitIndex])
      pre.shape = block.shape
      pre.temporality = block.temporality
      result.append(pre)
    }
    // Post block (memoryRead onward) — recursively split if more conflicts
    var post = Block(frameOrder: .parallel)
    post.nodes = Array(block.nodes[splitIndex...])
    post.shape = block.shape
    post.temporality = block.temporality
    result.append(contentsOf: splitMemoryBlocks(g: g, blocks: [post]))
  }
  return result
}

/// Creates an empty derived block that preserves non-node metadata needed for grouping.
private func makeTensorGroupingBlock(from original: Block) -> Block {
  var newBlock = Block(frameOrder: original.frameOrder)
  newBlock.temporality = original.temporality
  return newBlock
}

/// Appends `currentBlock` to `grouped` when non-empty.
private func appendCurrentGroupingBlockIfNeeded(
  _ currentBlock: inout Block, grouped: inout [Block]
) {
  if !currentBlock.nodes.isEmpty {
    grouped.append(currentBlock)
  }
}

/// Assigns tensor loop metadata from the first tensor-shaped node in `block`.
private func assignTensorIndexFromFirstTensorNode(
  to block: inout Block, graph: Graph, ctx: IRContext
) {
  for nodeId in block.nodes {
    guard let node = graph.nodes[nodeId], case .tensor(let shape) = node.shape else { continue }
    // Self-iterating ops (delayLine, spectrumDelay, ...) emit their own lane
    // loops; a block whose only tensor work is self-iterating must not get a
    // block-level element loop wrapped around it.
    if node.op.emitsInternalIteration { continue }
    guard block.tensorIndex == nil else { break }
    block.tensorIndex = ctx.useVariable(src: nil)
    block.shape = shape
    break
  }
}

/// Returns the first tensor-shaped, non-view node offset in a scalar block.
private func firstNonViewTensorOffset(in block: Block, graph: Graph) -> Int? {
  block.nodes.enumerated().first { (_, nodeId) in
    guard let node = graph.nodes[nodeId], case .tensor = node.shape else { return false }
    return !node.op.isViewOnly
  }?.offset
}

/// Returns the first scalar-shaped, non-view node offset after tensor work begins.
///
/// This detects a tensor-to-scalar suffix such as `... -> overlapAdd -> output`.
/// Keeping that suffix in the same sequential block would force the tensor region
/// to inherit frame-based temporality from the scalar consumer.
private func firstScalarSuffixOffset(
  in block: Block, graph: Graph, after tensorOffset: Int
) -> Int? {
  guard tensorOffset + 1 < block.nodes.count else { return nil }
  return block.nodes[(tensorOffset + 1)...].enumerated().first { (relativeOffset, nodeId) in
    _ = relativeOffset
    guard let node = graph.nodes[nodeId] else { return false }
    if node.op.isViewOnly { return false }
    if case .tensor = node.shape { return false }
    return true
  }.map { tensorOffset + 1 + $0.offset }
}

/// Returns the first hop-producing tensor node after non-hop tensor work has begun.
///
/// This isolates patterns like:
/// `frame-rate tensor state update -> latch(triggered at hop) -> hop-rate FFT/IFFT chain`
/// so the hop-producing suffix keeps hop-based scheduling.
private func firstHopTensorSuffixOffset(
  in block: Block, graph: Graph, after tensorOffset: Int
) -> Int? {
  guard tensorOffset + 1 < block.nodes.count else { return nil }

  var sawNonHopTensor = false
  for offset in tensorOffset..<block.nodes.count {
    let nodeId = block.nodes[offset]
    guard let node = graph.nodes[nodeId], !node.op.isViewOnly else { continue }
    guard case .tensor = node.shape else { continue }

    // History feedback (read/update/write-back) is scheduled as one sequential
    // cluster. It must NOT be split here: this boundary targets a hop FFT/spectral
    // chain that follows a frame-rate tensor update, not self-contained history
    // feedback. A hop-gated historyWrite would otherwise be sundered from its
    // read+update, putting read and write-back in separate Metal kernels and
    // breaking the cross-hop dependency (every hop would read stale state).
    switch node.op {
    case .historyRead, .historyWrite, .historyReadWrite:
      continue
    default:
      break
    }

    if graph.nodeHopRate[nodeId] != nil {
      if sawNonHopTensor { return offset }
    } else {
      sawNonHopTensor = true
    }
  }

  return nil
}

/// Returns true when a scalar block prefix contains inherently-scalar stateful ops.
///
/// When true, the scalar prefix must be split out so tensor loop wrapping does not run
/// these stateful ops once per tensor element.
private func scalarPrefixNeedsSplit(
  block: Block, firstTensorOffset: Int, graph: Graph
) -> Bool {
  guard firstTensorOffset > 0 else { return false }
  return block.nodes[0..<firstTensorOffset].contains { nodeId in
    guard let op = graph.nodes[nodeId]?.op else { return false }
    if op.isInherentlyScalar { return true }
    // Tensor->scalar reduces (sum, …) emit their own internal element loop.
    // Left in the prefix they'd be wrapped by the tensor body's parallelRange
    // and re-run once per element (O(n²) per frame).
    if isReductionOp(op) { return true }
    switch op {
    case .tensorRef, .seq:
      return false
    default:
      return op.emitsInternalIteration
    }
  }
}

/// Splits scalar blocks only when needed to keep inherently scalar state ops out of tensor loops.
private func splitScalarBlockForTensorGrouping(
  _ block: Block, graph: Graph, ctx: IRContext
) -> [Block] {
  guard let firstTensorOffset = firstNonViewTensorOffset(in: block, graph: graph) else {
    var modified = block
    assignTensorIndexFromFirstTensorNode(to: &modified, graph: graph, ctx: ctx)
    return [modified]
  }

  let needsPrefixSplit = scalarPrefixNeedsSplit(
    block: block, firstTensorOffset: firstTensorOffset, graph: graph)
  let scalarSuffixOffset = firstScalarSuffixOffset(
    in: block, graph: graph, after: firstTensorOffset)
  let hopTensorSuffixOffset = firstHopTensorSuffixOffset(
    in: block, graph: graph, after: firstTensorOffset)
  let splitOffset = [scalarSuffixOffset, hopTensorSuffixOffset].compactMap { $0 }.min()

  if !needsPrefixSplit && splitOffset == nil {
    var modified = block
    assignTensorIndexFromFirstTensorNode(to: &modified, graph: graph, ctx: ctx)
    return [modified]
  }

  var result: [Block] = []
  let tensorStart = needsPrefixSplit ? firstTensorOffset : 0
  let tensorEndExclusive = splitOffset ?? block.nodes.count

  if needsPrefixSplit {
    var scalarPrefix = makeTensorGroupingBlock(from: block)
    scalarPrefix.nodes = Array(block.nodes[0..<firstTensorOffset])
    result.append(scalarPrefix)
  }

  if tensorStart < tensorEndExclusive {
    var tensorBody = makeTensorGroupingBlock(from: block)
    tensorBody.frameOrder = needsPrefixSplit ? .parallel : block.frameOrder
    tensorBody.nodes = Array(block.nodes[tensorStart..<tensorEndExclusive])
    assignTensorIndexFromFirstTensorNode(to: &tensorBody, graph: graph, ctx: ctx)
    result.append(tensorBody)
  }

  if let splitOffset {
    var scalarSuffix = makeTensorGroupingBlock(from: block)
    scalarSuffix.nodes = Array(block.nodes[splitOffset...])
    result.append(contentsOf: splitScalarBlockForTensorGrouping(scalarSuffix, graph: graph, ctx: ctx))
  }

  return result
}

/// Strip `block.shape` / `block.tensorIndex` when the block contains no node
/// that genuinely emits per-element work. Otherwise
/// `wrapBodyUOpsWithTensorLoopIfNeeded` would wrap the block body in a
/// `parallelRange(shape.size)` wrapper that forces scalar / self-iterating
/// ops to run `shape.size` times for no reason.
private func clearWastedTensorLoopMetadata(_ block: inout Block, graph: Graph) {
  if !blockHasPerElementComputeNode(block, graph: graph) {
    block.shape = nil
    block.tensorIndex = nil
  }
}

/// Groups non-scalar blocks by tensor shape while preserving special-case execution constraints.
private func groupRegularTensorBlock(
  _ block: Block, graph: Graph, ctx: IRContext
) -> [Block] {
  var grouped: [Block] = []
  var currentBlock = makeTensorGroupingBlock(from: block)
  var currentShape: Shape? = nil
  var currentHasNonHopTensor = false

  for nodeId in block.nodes {
    guard let node = graph.nodes[nodeId] else {
      currentBlock.nodes.append(nodeId)
      continue
    }

    // tensorRef only seeds tensor loop metadata; it should not force standalone compute blocks.
    if case .tensorRef = node.op {
      if case .tensor(let shape) = node.shape {
        let isHopTensorRef = graph.nodeHopRate[nodeId] != nil
        if currentShape != nil && isHopTensorRef && currentHasNonHopTensor {
          appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
          currentBlock = makeTensorGroupingBlock(from: block)
          currentHasNonHopTensor = false
        }
        if currentShape != nil && shape != currentShape {
          appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
          currentBlock = makeTensorGroupingBlock(from: block)
          currentHasNonHopTensor = false
        }
        currentBlock.shape = shape
        currentBlock.tensorIndex = ctx.useVariable(src: nil)
        currentShape = shape
        if !isHopTensorRef {
          currentHasNonHopTensor = true
        }
      }
      currentBlock.nodes.append(nodeId)
      continue
    }

    // View-only ops emit metadata markers and should not split tensor execution regions.
    if node.op.isViewOnly {
      currentBlock.nodes.append(nodeId)
      continue
    }

    if case .conv2d = node.op {
      if currentShape != nil {
        appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
        currentBlock = makeTensorGroupingBlock(from: block)
      }
      currentShape = nil
      currentHasNonHopTensor = false

    } else if node.op.isSelfDispatchedGemm {
      // GEMM variants manage their own dispatch — isolate into their own block.
      appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
      var gemmBlock = makeTensorGroupingBlock(from: block)
      gemmBlock.nodes.append(nodeId)
      gemmBlock.shape = nil
      grouped.append(gemmBlock)
      currentBlock = makeTensorGroupingBlock(from: block)
      currentShape = nil
      currentHasNonHopTensor = false
      continue

    } else if case .gemmSmall(let M, let N, _, _, _) = node.op {
      // gemmSmall uses perFrameScaled(M*N) dispatch — isolate into its own block
      appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
      var gemmSmallBlock = makeTensorGroupingBlock(from: block)
      gemmSmallBlock.nodes.append(nodeId)
      gemmSmallBlock.shape = [M, N]  // Shape drives tensorIndex assignment
      gemmSmallBlock.tensorIndex = ctx.useVariable(src: nil)
      grouped.append(gemmSmallBlock)
      currentBlock = makeTensorGroupingBlock(from: block)
      currentShape = nil
      currentHasNonHopTensor = false
      continue
    } else if case .constant = node.op {
      // Constants do not affect grouping state.
    } else if node.op.emitsInternalIteration {
      // Self-iterating ops like acceleratedFFT/IFFT, overlapAdd,
      // partitionedSpectralConvolve, tensorAccumulate, etc. emit their own
      // loops/vDSP calls and must not share a sibling tensor loop. Isolate the
      // op itself, then let any downstream tensorRef-backed consumers start a
      // fresh tensor region.
      appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)

      var selfIteratingBlock = makeTensorGroupingBlock(from: block)
      selfIteratingBlock.frameOrder = .sequential
      selfIteratingBlock.nodes.append(nodeId)
      grouped.append(selfIteratingBlock)

      currentBlock = makeTensorGroupingBlock(from: block)
      currentShape = nil
      currentHasNonHopTensor = false
      continue
    } else if case .tensor(let shape) = node.shape {
      let isHopTensor = graph.nodeHopRate[nodeId] != nil
      if currentShape != nil && isHopTensor && currentHasNonHopTensor {
        appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
        currentBlock = makeTensorGroupingBlock(from: block)
        currentBlock.tensorIndex = ctx.useVariable(src: nil)
        currentBlock.shape = shape
        currentShape = shape
        currentHasNonHopTensor = false
      }
      if shape != currentShape {
        // Axis reduces and concat-by-padding transitions stay in-region even when shape changes.
        if let previousShape = currentShape,
          isAxisReduceOp(node.op)
            || isConcatByPaddingFusionTransition(
              nodeId: nodeId, graph: graph, currentShape: previousShape)
        {
          currentShape = shape
        } else {
          appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
          currentBlock = makeTensorGroupingBlock(from: block)
          currentBlock.tensorIndex = ctx.useVariable(src: nil)
          currentBlock.shape = shape
          currentShape = shape
          currentHasNonHopTensor = false
        }
      }
      if !isHopTensor {
        currentHasNonHopTensor = true
      }
    } else {
      if currentShape != nil {
        appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
        currentBlock = makeTensorGroupingBlock(from: block)
      }
      currentShape = nil
      currentHasNonHopTensor = false
    }

    currentBlock.nodes.append(nodeId)
  }

  appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)

  // Post-pass: a `tensorRef` sets `block.shape` / `block.tensorIndex` up front
  // so axis-reduce and shape-transition grouping can track state. But if the
  // block ends up with NO node that genuinely emits per-element work driven by
  // the block tensorIndex (only tensorRefs, view-only ops, self-iterating ops
  // like FFT/overlapAdd/partitionedSpectralConvolve, and scalar-shape ops),
  // then `wrapBodyUOpsWithTensorLoopIfNeeded` would still wrap the body in a
  // `parallelRange(tensorRef.size)` — which turns a scalar per-frame input
  // read into `tensorRef.size` iterations of the same store. Strip the
  // metadata when it would only cause wasted iteration.
  for i in grouped.indices {
    clearWastedTensorLoopMetadata(&grouped[i], graph: graph)
  }

  return grouped
}

/// Returns true iff the block contains at least one node that emits per-element
/// tensor work driven by the block-level `tensorIndex`.
///
/// Skipped:
/// - tensorRef: just a cell pointer, emits no code.
/// - view-only ops (reshape/transpose/…): metadata markers, no code.
/// - ops marked `emitsInternalIteration` (bufferView's seq, FFT/IFFT, overlapAdd,
///   partitionedSpectralConvolve, gemm/conv self-isolated, spectral-loss
///   variants, …): these emit their own `b.loop`/`b.parallelRange`/vDSP calls.
/// - scalar-shape ops (e.g. `.input(0)`): per-frame, not per-element.
private func blockHasPerElementComputeNode(_ block: Block, graph: Graph) -> Bool {
  for nodeId in block.nodes {
    guard let node = graph.nodes[nodeId] else { continue }
    if node.op.isViewOnly { continue }
    if node.op.emitsInternalIteration { continue }
    if case .tensor = node.shape { return true }
  }
  return false
}

/// Annotates/splits blocks for tensor loop emission.
///
/// Scalar blocks are preserved unless a scalar prefix must be split out to protect stateful ops.
/// Non-scalar blocks are grouped by tensor shape with explicit handling for conv/overlap/view
/// semantics required by the emission backend.
private func isAcceleratedFFTOp(_ op: LazyOp) -> Bool {
  if case .acceleratedFFT = op { return true }
  if case .acceleratedIFFT = op { return true }
  return false
}

/// Splits a sequential block so each accelerated FFT/IFFT node sits alone in
/// its own block (shape/tensorIndex cleared — the op self-iterates). Non-FFT
/// runs keep the original block's properties and ordering.
private func splitOutAcceleratedFFTNodes(_ block: Block, graph: Graph) -> [Block] {
  guard block.nodes.count > 1 else { return [block] }
  var result: [Block] = []
  var run: [NodeID] = []
  func flushRun() {
    guard !run.isEmpty else { return }
    var b = block
    b.nodes = run
    result.append(b)
    run = []
  }
  for nodeId in block.nodes {
    if let node = graph.nodes[nodeId], isAcceleratedFFTOp(node.op) {
      flushRun()
      var fftBlock = block
      fftBlock.nodes = [nodeId]
      fftBlock.shape = nil
      fftBlock.tensorIndex = nil
      result.append(fftBlock)
    } else {
      run.append(nodeId)
    }
  }
  flushRun()
  return result
}

func determineTensorBlocks(_ blocks: [Block], _ graph: Graph, _ ctx: IRContext) -> [Block] {
  var determined: [Block] = []

  for block in blocks {
    if block.frameOrder == .sequential {
      // Accelerated FFT/IFFT nodes fused into a scalar block alongside
      // frame-rate neighbors (overlapAdd output taps, waveshapers, ...) would
      // inherit the block's frameBased temporality and run the transform once
      // per FRAME instead of once per hop — catastrophically expensive. The
      // parallel path already isolates self-iterating ops
      // (groupRegularTensorBlock); do the same minimal isolation here so
      // TemporalityPass can give the transform its own hop-rate block.
      let parts = splitOutAcceleratedFFTNodes(block, graph: graph)
      if parts.count > 1 {
        for part in parts {
          if part.nodes.count == 1, let node = graph.nodes[part.nodes[0]],
            isAcceleratedFFTOp(node.op)
          {
            determined.append(part)
          } else {
            var split = splitScalarBlockForTensorGrouping(part, graph: graph, ctx: ctx)
            for i in split.indices {
              clearWastedTensorLoopMetadata(&split[i], graph: graph)
            }
            determined.append(contentsOf: split)
          }
        }
        continue
      }
      var split = splitScalarBlockForTensorGrouping(block, graph: graph, ctx: ctx)
      // Scalar blocks that happen to "own" a tensorRef (e.g. bufferView's write
      // block, [memoryWrite, tensorRef]) inherit the tensorRef's shape here.
      // Without this sweep, `wrapBodyUOpsWithTensorLoopIfNeeded` wraps the
      // scalar memoryWrite in `parallelRange(tensorSize)` — a dead inner loop
      // that executes the same per-frame write hundreds of times.
      //
      // Scalar prefixes always strip when they contain no per-element work.
      // The parallel tensor-suffix also strips when every node there is
      // self-iterating (hopTensorNoise / FFT-family / overlapAdd) — the
      // combined `determineVectorPlan` + `hasSIMDBlockers` check keeps
      // SIMD-4 promotion from firing on blocks with internal scalar loops.
      for i in split.indices {
        clearWastedTensorLoopMetadata(&split[i], graph: graph)
      }
      determined.append(contentsOf: split)
      continue
    }
    determined.append(contentsOf: groupRegularTensorBlock(block, graph: graph, ctx: ctx))
  }

  return determined
}

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
    .sumMulAxis0, .gemmSmall, .chunkPartialsReduceToCell:
    return true
  default:
    return false
  }
}

public func isGlobalReductionOp(_ op: LazyOp) -> Bool {
  switch op {
  case .sampleGradReduce, .selectRowGradReduce, .peekGradReduce,
    .tensorAccumulate, .chunkPartialsReduceToCell:
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

/// Returns true when a scalar block prefix contains inherently-scalar stateful ops.
///
/// When true, the scalar prefix must be split out so tensor loop wrapping does not run
/// these stateful ops once per tensor element.
private func scalarPrefixNeedsSplit(
  block: Block, firstTensorOffset: Int, graph: Graph
) -> Bool {
  guard firstTensorOffset > 0 else { return false }
  return block.nodes[0..<firstTensorOffset].contains { nodeId in
    graph.nodes[nodeId]?.op.isInherentlyScalar ?? false
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

  guard scalarPrefixNeedsSplit(block: block, firstTensorOffset: firstTensorOffset, graph: graph)
  else {
    var modified = block
    assignTensorIndexFromFirstTensorNode(to: &modified, graph: graph, ctx: ctx)
    return [modified]
  }

  var scalarPrefix = makeTensorGroupingBlock(from: block)
  scalarPrefix.nodes = Array(block.nodes[0..<firstTensorOffset])

  var tensorSuffix = makeTensorGroupingBlock(from: block)
  tensorSuffix.frameOrder = .parallel
  tensorSuffix.nodes = Array(block.nodes[firstTensorOffset...])
  assignTensorIndexFromFirstTensorNode(to: &tensorSuffix, graph: graph, ctx: ctx)

  return [scalarPrefix, tensorSuffix]
}

/// Groups non-scalar blocks by tensor shape while preserving special-case execution constraints.
private func groupRegularTensorBlock(
  _ block: Block, graph: Graph, ctx: IRContext
) -> [Block] {
  var grouped: [Block] = []
  var currentBlock = makeTensorGroupingBlock(from: block)
  var currentShape: Shape? = nil

  for nodeId in block.nodes {
    guard let node = graph.nodes[nodeId] else {
      currentBlock.nodes.append(nodeId)
      continue
    }

    // tensorRef only seeds tensor loop metadata; it should not force standalone compute blocks.
    if case .tensorRef = node.op {
      if case .tensor(let shape) = node.shape {
        if currentShape != nil && shape != currentShape {
          appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
          currentBlock = makeTensorGroupingBlock(from: block)
        }
        currentBlock.shape = shape
        currentBlock.tensorIndex = ctx.useVariable(src: nil)
        currentShape = shape
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

    } else if case .gemm = node.op {
      // GEMM manages its own dispatch — isolate into its own block
      appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
      var gemmBlock = makeTensorGroupingBlock(from: block)
      gemmBlock.nodes.append(nodeId)
      gemmBlock.shape = nil  // GEMM has its own dispatch, no shape-based threading
      grouped.append(gemmBlock)
      currentBlock = makeTensorGroupingBlock(from: block)
      currentShape = nil
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
      continue
    } else if case .gemmChunkPartials = node.op {
      // Chunked GEMM partials use explicit 3D tiled dispatch.
      appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
      var gemmBlock = makeTensorGroupingBlock(from: block)
      gemmBlock.nodes.append(nodeId)
      gemmBlock.shape = nil
      grouped.append(gemmBlock)
      currentBlock = makeTensorGroupingBlock(from: block)
      currentShape = nil
      continue

    } else if case .constant = node.op {
      // Constants do not affect grouping state.
    } else if case .overlapAdd = node.op {
      // overlapAdd has its own internal frame/scatter loop and must be isolated.
      appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)

      var overlapAddBlock = makeTensorGroupingBlock(from: block)
      overlapAddBlock.frameOrder = .sequential
      overlapAddBlock.nodes.append(nodeId)
      grouped.append(overlapAddBlock)

      currentBlock = makeTensorGroupingBlock(from: block)
      currentShape = nil
      continue
    } else if case .tensor(let shape) = node.shape {
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
        }
      }
    } else {
      if currentShape != nil {
        appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
        currentBlock = makeTensorGroupingBlock(from: block)
      }
      currentShape = nil
    }

    currentBlock.nodes.append(nodeId)
  }

  appendCurrentGroupingBlockIfNeeded(&currentBlock, grouped: &grouped)
  return grouped
}

/// Annotates/splits blocks for tensor loop emission.
///
/// Scalar blocks are preserved unless a scalar prefix must be split out to protect stateful ops.
/// Non-scalar blocks are grouped by tensor shape with explicit handling for conv/overlap/view
/// semantics required by the emission backend.
func determineTensorBlocks(_ blocks: [Block], _ graph: Graph, _ ctx: IRContext) -> [Block] {
  var determined: [Block] = []

  for block in blocks {
    if block.frameOrder == .sequential {
      determined.append(contentsOf: splitScalarBlockForTensorGrouping(block, graph: graph, ctx: ctx))
      continue
    }
    determined.append(contentsOf: groupRegularTensorBlock(block, graph: graph, ctx: ctx))
  }

  return determined
}

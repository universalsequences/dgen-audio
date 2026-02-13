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
    let kind: Kind = isScalar ? .scalar : .simd

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
        currentBlock = Block(kind: kind)
        currentBlock!.nodes.append(nodeId)
      }
      continue
    }

    // Regular node handling - group consecutive nodes of same kind together
    if let current = currentBlock {
      if current.kind == kind {
        currentBlock!.nodes.append(nodeId)
      } else {
        blocks.append(current)
        currentBlock = Block(kind: kind)
        currentBlock!.nodes.append(nodeId)
      }
    } else {
      currentBlock = Block(kind: kind)
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
      print("  Block \(i) (\(block.kind)): \(block.nodes)")
    }
  }

  return blocks
}

/// Merge consecutive blocks of the same kind into a single block.
public func fuseBlocks(_ blocks: [Block]) -> [Block] {
  var fused: [Block] = []
  for b in blocks {
    if b.nodes.isEmpty { continue }
    if let lastIdx = fused.indices.last, fused[lastIdx].kind == b.kind {
      fused[lastIdx].nodes.append(contentsOf: b.nodes)
    } else {
      fused.append(b)
    }
  }
  return fused
}

/// Isolate spectralloss ops into their own blocks
/// to ensure they execute as separate kernels without any fused operations.
/// Also isolates FFT-based spectral loss gradient operations to avoid race conditions.
/// Preserves ordering of other nodes.
public func isolateSpectralPasses(_ blocks: [Block], _ g: Graph) -> [Block] {
  var result: [Block] = []

  // Helper to create a new block with the same properties as the original
  func makeBlock(from original: Block, nodes: [NodeID]) -> Block {
    var newBlock = Block(kind: original.kind)
    newBlock.nodes = nodes
    newBlock.temporality = original.temporality
    newBlock.tensorIndex = original.tensorIndex
    newBlock.shape = original.shape
    return newBlock
  }

  for block in blocks {
    var currentNodes: [NodeID] = []

    for nodeId in block.nodes {
      let isSpectralPass = { () -> Bool in
        guard let node = g.nodes[nodeId] else { return false }
        // FFT-based spectral loss ops need isolation because they use shared scratch
        // memory for FFT computation that can't be safely accessed by multiple SIMD threads
        if case .spectralLossFFT = node.op { return true }
        if case .spectralLossFFTGradInline = node.op { return true }
        if case .spectralLossFFTGradSpec = node.op { return true }
        if case .spectralLossFFTGradIFFT = node.op { return true }
        if case .spectralLossFFTGradRead = node.op { return true }
        if case .spectralLossFFTGradRead2 = node.op { return true }
        return false
      }()

      if isSpectralPass {
        // Flush any accumulated nodes before the spectral pass
        if !currentNodes.isEmpty {
          result.append(makeBlock(from: block, nodes: currentNodes))
          currentNodes = []
        }

        // Add spectral pass in its own SIMD block — spectral ops are always
        // parallel across frames, even when their inputs come from scalar blocks
        var spectralBlock = makeBlock(from: block, nodes: [nodeId])
        spectralBlock.kind = .simd
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
  case .sum, .tensorAccumulate, .peekRowGradReduce,
    .selectRowGradReduce, .overlapAddGradGather, .bufferViewGradRead:
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
    // Don't split scalar blocks - they run frame-by-frame serially and need
    // all operations to stay together for feedback cycles to work correctly
    if block.kind == .scalar {
      splitBlocks.append(block)
      continue
    }

    let reductionOpIndex = block.nodes.firstIndex { nodeId in
      guard let node = g.nodes[nodeId] else { return false }
      return isReductionOp(node.op)
    }

    guard let reductionOpIndex else {
      splitBlocks.append(block)
      continue
    }

    // Pre-reduction block
    if reductionOpIndex > 0 {
      var preReductionBlock = Block(kind: .simd)
      preReductionBlock.nodes = Array(block.nodes[0..<reductionOpIndex])
      preReductionBlock.shape = block.shape
      preReductionBlock.temporality = block.temporality
      splitBlocks.append(preReductionBlock)
    }

    // Reduction block
    // Global reduces run once total, not per-frame:
    // - peekRowGradReduce/selectRowGradReduce: reduction ops with internal frame loops
    // - tensorAccumulate: atomic gradient accumulation, loops over frames internally
    let reductionNode = g.nodes[block.nodes[reductionOpIndex]]
    let isGlobalReduce: Bool
    switch reductionNode?.op {
    case .peekRowGradReduce, .selectRowGradReduce, .tensorAccumulate:
      isGlobalReduce = true
    default:
      isGlobalReduce = false
    }

    // Global reduces need kind=.scalar AND temporality=.static_ to run once total
    var reductionBlock = Block(kind: isGlobalReduce ? .scalar : .simd)
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
      var postReductionBlock = Block(kind: .simd)
      postReductionBlock.nodes = Array(block.nodes[reductionOpIndex + 1..<block.nodes.count])
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
    if block.kind == .scalar {
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
      var pre = Block(kind: .simd)
      pre.nodes = Array(block.nodes[0..<splitIndex])
      pre.shape = block.shape
      pre.temporality = block.temporality
      result.append(pre)
    }
    // Post block (memoryRead onward) — recursively split if more conflicts
    var post = Block(kind: .simd)
    post.nodes = Array(block.nodes[splitIndex...])
    post.shape = block.shape
    post.temporality = block.temporality
    result.append(contentsOf: splitMemoryBlocks(g: g, blocks: [post]))
  }
  return result
}

func determineTensorBlocks(_ blocks: [Block], _ graph: Graph, _ ctx: IRContext) -> [Block] {
  var determined: [Block] = []

  // Helper to create a new block preserving original properties
  func makeBlock(from original: Block) -> Block {
    var newBlock = Block(kind: original.kind)
    newBlock.temporality = original.temporality
    return newBlock
  }

  // Assign tensorIndex from the first tensor-shaped node in the block
  func assignTensorIndex(to block: inout Block, graph: Graph, ctx: IRContext) {
    for nodeId in block.nodes {
      if let node = graph.nodes[nodeId], case .tensor(let shape) = node.shape {
        if block.tensorIndex == nil {
          block.tensorIndex = ctx.useVariable(src: nil)
          block.shape = shape
        }
        break
      }
    }
  }

  for block in blocks {
    // CRITICAL: Scalar blocks (feedback clusters) must NOT be split!
    // They contain history read/write ops that must execute sequentially together.
    // But we still need to set up tensorIndex for tensor operations inside.
    if block.kind == .scalar {
      // Find the first tensor-shaped node (excluding view-only ops)
      let firstTensorIdx = block.nodes.enumerated().first { (_, nodeId) in
        guard let node = graph.nodes[nodeId], case .tensor = node.shape else { return false }
        return !node.op.isViewOnly
      }?.offset

      // Split scalar blocks when the leading scalar nodes contain stateful ops
      // (accum, phasor, etc.) that must NOT execute inside the tensor element loop.
      // The C backend wraps blocks with tensorIndex in beginParallelRange, which would
      // cause these ops to execute N times per frame instead of once.
      let needsSplit: Bool
      if let splitIdx = firstTensorIdx, splitIdx > 0 {
        needsSplit = block.nodes[0..<splitIdx].contains { nodeId in
          graph.nodes[nodeId]?.op.isInherentlyScalar ?? false
        }
      } else {
        needsSplit = false
      }

      if let splitIdx = firstTensorIdx, needsSplit {
        // Scalar prefix as its own block (no tensorIndex)
        var scalarBlock = makeBlock(from: block)
        scalarBlock.nodes = Array(block.nodes[0..<splitIdx])
        determined.append(scalarBlock)

        // Tensor suffix with tensorIndex
        var tensorBlock = makeBlock(from: block)
        tensorBlock.nodes = Array(block.nodes[splitIdx...])
        assignTensorIndex(to: &tensorBlock, graph: graph, ctx: ctx)
        determined.append(tensorBlock)
      } else {
        // No split needed — assign tensorIndex to entire block
        var modifiedBlock = block
        assignTensorIndex(to: &modifiedBlock, graph: graph, ctx: ctx)
        determined.append(modifiedBlock)
      }
      continue
    }
    var innerBlocks: [Block] = []
    var currentBlock = makeBlock(from: block)
    var currentShape: Shape? = nil
    for nodeId in block.nodes {
      if let node = graph.nodes[nodeId] {
        // Skip tensorRef nodes for tensor block grouping decisions.
        //
        // tensorRef nodes are just data containers - they emit nothing (return []).
        // If we let them create tensor blocks, we'd get empty parallel loops:
        //   for (int simd1 = 0; simd1 < 16; simd1+=4) { }  // empty!
        //
        // Instead, tensorRef nodes stay in whatever block they're in but don't
        // trigger new tensor block creation. The actual tensor OPERATIONS (mul, add, etc.)
        // that process tensor data will create the tensor blocks.
        if case .tensorRef = node.op {
          if case .tensor(let shape) = node.shape {
            // If shape differs, split the block first
            if currentShape != nil && shape != currentShape {
              if currentBlock.nodes.count > 0 {
                innerBlocks.append(currentBlock)
              }
              currentBlock = makeBlock(from: block)
            }
            currentBlock.shape = shape
            currentBlock.tensorIndex = ctx.useVariable(src: nil)
            currentShape = shape
          }
          currentBlock.nodes.append(nodeId)
          continue
        }

        // Skip view-only ops for tensor block grouping.
        // These emit no compute code (just marker UOps). Letting them trigger
        // tensor block splits would create empty parallel loops.
        if node.op.isViewOnly {
          currentBlock.nodes.append(nodeId)
          continue
        }

        if case .conv2d = node.op {
          if currentShape != nil {
            if currentBlock.nodes.count > 0 {
              innerBlocks.append(currentBlock)
            }
            // regular node
            currentBlock = makeBlock(from: block)
          }
          currentShape = nil

        } else if case .constant = node.op {
          // do nothing
        } else if case .overlapAdd = node.op {
          // overlapAdd has internal loops (scatter-add) — needs its own scalar block
          if currentBlock.nodes.count > 0 {
            innerBlocks.append(currentBlock)
          }
          currentBlock = makeBlock(from: block)
          currentBlock.kind = .scalar
          currentBlock.nodes.append(nodeId)
          innerBlocks.append(currentBlock)
          currentBlock = makeBlock(from: block)
          currentShape = nil
          continue
        } else if case .tensor(let shape) = node.shape {
          if shape != currentShape {
            // Axis reduces (sumAxis, maxAxis, meanAxis) can stay in the same block
            // as their input — each output thread reduces over an independent slice.
            // Don't change block kind — shape-transition emission handles the different
            // element counts via per-region loops while keeping the block parallelizable.
            if let previousShape = currentShape,
              isAxisReduceOp(node.op)
                || isConcatByPaddingFusionTransition(
                  nodeId: nodeId,
                  graph: graph,
                  currentShape: previousShape)
            {
              currentShape = shape
            } else {
              if currentBlock.nodes.count > 0 {
                innerBlocks.append(currentBlock)
              }
              // tensor block
              currentBlock = makeBlock(from: block)
              currentBlock.tensorIndex = ctx.useVariable(src: nil)
              currentBlock.shape = shape
              currentShape = shape
            }
          }
        } else {
          if currentShape != nil {
            if currentBlock.nodes.count > 0 {
              innerBlocks.append(currentBlock)
            }
            // regular node
            currentBlock = makeBlock(from: block)
          }
          currentShape = nil
        }
      }
      currentBlock.nodes.append(nodeId)
    }
    if currentBlock.nodes.count > 0 {
      innerBlocks.append(currentBlock)
    }
    for block in innerBlocks {
      determined.append(block)
    }
  }
  return determined
}

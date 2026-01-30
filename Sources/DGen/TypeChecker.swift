/// Adapt strides when reshaping a tensor.
/// For contiguous tensors, computes fresh row-major strides.
/// For non-contiguous tensors (e.g., after transpose), preserves the stride pattern
/// when adding/removing dimensions of size 1.
///
/// Example: [2,3] strides [1,2] (transposed) -> [1,2,3] should give strides [6,1,2]
func adaptStridesForReshape(inputShape: [Int], inputStrides: [Int], newShape: [Int]) -> [Int] {
  // Check if input is contiguous (row-major)
  let expectedContiguousStrides = Tensor.computeRowMajorStrides(inputShape)
  let isContiguous = (inputStrides == expectedContiguousStrides)

  if isContiguous {
    // Input is contiguous, compute fresh row-major strides
    return Tensor.computeRowMajorStrides(newShape)
  }

  // Non-contiguous input - try to adapt strides
  // This works when we're only adding/removing dimensions of size 1

  let inputNonOnes = inputShape.filter { $0 != 1 }
  let newNonOnes = newShape.filter { $0 != 1 }

  // If the non-1 dimensions match, we can adapt strides
  if inputNonOnes == newNonOnes {
    var newStrides = [Int]()
    var inputIdx = 0

    for dim in newShape {
      if dim == 1 {
        // New dimension of size 1 - stride doesn't matter (use product of remaining)
        let remainingProduct = newShape.suffix(from: newStrides.count + 1).reduce(1, *)
        newStrides.append(remainingProduct)
      } else {
        // Find corresponding stride from input
        while inputIdx < inputShape.count && inputShape[inputIdx] == 1 {
          inputIdx += 1
        }
        if inputIdx < inputStrides.count {
          newStrides.append(inputStrides[inputIdx])
          inputIdx += 1
        }
      }
    }
    return newStrides
  }

  // Fallback: this reshape requires a copy (not supported as view)
  // For now, just compute row-major strides
  return Tensor.computeRowMajorStrides(newShape)
}

/// NumPy-style broadcasting: computes the output shape when two shapes are broadcast together.
/// Returns nil if the shapes are not broadcastable.
/// Example: [2, 1, 3] + [1, 2, 3] -> [2, 2, 3]
public func broadcastShapes(_ s1: [Int], _ s2: [Int]) -> [Int]? {
  // Pad shorter shape with 1s on the left
  let maxLen = max(s1.count, s2.count)
  let padded1 = Array(repeating: 1, count: maxLen - s1.count) + s1
  let padded2 = Array(repeating: 1, count: maxLen - s2.count) + s2

  var result = [Int]()
  for i in 0..<maxLen {
    let d1 = padded1[i]
    let d2 = padded2[i]
    if d1 == d2 {
      result.append(d1)
    } else if d1 == 1 {
      result.append(d2)
    } else if d2 == 1 {
      result.append(d1)
    } else {
      // Incompatible dimensions
      return nil
    }
  }
  return result
}

public func inferShape(op: LazyOp, inputs: [ValueShape], graph: Graph) throws -> ValueShape {
  switch op {
  case .tensorRef(let tid):
    guard let tensor = graph.tensors[tid] else {
      throw DGenError.missingTensorID
    }
    return .tensor(tensor.shape)

  // History read - returns scalar or tensor shape depending on cell
  case .historyRead(let cellId):
    // O(1) lookup using cellToTensor mapping
    if let tensorId = graph.cellToTensor[cellId], let tensor = graph.tensors[tensorId] {
      return .tensor(tensor.shape)
    }
    // Scalar cell
    return .scalar

  // History write - output shape same as input (passthrough)
  case .historyWrite(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "historyWrite", reason: "missing input")
    }
    return firstInput

  // Conv1d - output shape matches input shape (same padding)
  case .conv1d(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "conv1d", reason: "missing input tensor")
    }
    return firstInput

  // Conv2d - output shape matches input shape (same padding)
  case .conv2d(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "conv2d", reason: "missing input tensor")
    }
    return firstInput

  // Sum reduce - always outputs scalar
  case .sum:
    return .scalar

  // Sum along axis - reduces one dimension
  case .sumAxis(let axis):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "sumAxis", reason: "requires tensor input")
    }
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.shapeInferenceFailed(
        op: "sumAxis", reason: "axis \(axis) out of range for \(ndim)D tensor")
    }
    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      return .scalar
    }
    return .tensor(outputShape)

  // Reshape - changes shape, preserves total size
  case .reshape(let newShape):
    return .tensor(newShape)

  // Transpose - permutes axes
  case .transpose(let axes):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "transpose", reason: "requires tensor input")
    }
    let perm = axes.isEmpty ? Array((0..<shape.count).reversed()) : axes
    var newShape = [Int](repeating: 0, count: shape.count)
    for i in 0..<shape.count {
      newShape[i] = shape[perm[i]]
    }
    return .tensor(newShape)

  // Shrink - slices tensor along each axis
  case .shrink(let ranges):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "shrink", reason: "requires tensor input")
    }
    var newShape = [Int]()
    for (dim, range) in ranges.enumerated() {
      if let (start, end) = range {
        newShape.append(end - start)
      } else {
        newShape.append(shape[dim])
      }
    }
    return .tensor(newShape)

  // Pad - expands tensor with zeros along each axis
  case .pad(let padding):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "pad", reason: "requires tensor input")
    }
    let newShape = zip(shape, padding).map { dim, pad in
      dim + pad.0 + pad.1
    }
    return .tensor(newShape)

  // Peek - reads a scalar from a 2D tensor at (index, channel)
  case .peek:
    // Peek always outputs scalar - it reads one value from the tensor
    return .scalar

  // PeekRow - reads an entire row from a 2D tensor with interpolation
  case .peekRow:
    guard let firstInput = inputs.first,
      case .tensor(let shape) = firstInput,
      shape.count == 2
    else {
      throw DGenError.shapeInferenceFailed(op: "peekRow", reason: "requires 2D tensor input")
    }
    return .tensor([shape[1]])  // Output is [numCols]

  // FFT - outputs [numBins, 2] tensor where numBins = windowSize/2 + 1
  // Note: FFT is a bulk operation that handles all tensor writes internally,
  // so it should not be wrapped in parallelRange (handled in CompilationPipeline)
  case .fft(let windowSize, _, _, _, _, _):
    let numBins = windowSize / 2 + 1
    return .tensor([numBins, 2])

  // IFFT - outputs scalar (one sample per frame via overlap-add)
  // Takes spectrum tensor [numBins, 2] as input, reconstructs time-domain signal
  case .ifft(_, _, _, _, _, _):
    return .scalar

  // Inherited (elementwise) - includes all binary and unary math ops
  // Also includes stateful ops (phasor, accum, latch) that can operate element-wise on tensors
  case .add, .sub, .mul, .div, .sin, .cos, .exp, .sqrt, .tanh,
    .tan, .log, .log10, .abs, .sign, .floor, .ceil, .round,
    .pow, .mod, .min, .max, .atan2, .gt, .gte, .lt, .lte, .eq,
    .and, .or, .xor, .gswitch, .mix,
    .phasor(_), .accum(_), .latch(_), .deterministicPhasor:
    let tensors = inputs.filter { x in
      if case .tensor(_) = x { return true }
      return false
    }
    if tensors.count == 2 {
      if case .tensor(let s1) = tensors[0], case .tensor(let s2) = tensors[1] {
        // Try NumPy-style broadcasting
        if let broadcastShape = broadcastShapes(s1, s2) {
          return .tensor(broadcastShape)
        } else {
          throw DGenError.shapeMismatch(op: "\(op)", shape1: s1, shape2: s2)
        }
      }
    }
    if tensors.count > 0 {
      return tensors[0]  // return the tensor as the shape
    }
    return .scalar

  // Seq returns the shape of the last input (the value that's returned)
  case .seq:
    return inputs.last ?? .scalar

  // everything else is a scalar
  default: return .scalar
  }
}

public func inferShapes(graph: Graph, sortedNodes: [NodeID]) throws {
  for nodeId in sortedNodes {
    if var node = graph.nodes[nodeId] {
      node.shape = try inferShape(
        op: node.op, inputs: node.inputs.compactMap { graph.nodes[$0]?.shape }, graph: graph)
      graph.nodes[nodeId] = node
    }
  }
}

/// Allocate output tensors for nodes that produce tensor results.
/// This is called after shape inference, so all nodes have their shapes assigned.
/// Nodes that already have a tensor (via tensorRef) are skipped.
/// View operations (reshape, transpose, shrink, pad) create views of their input's tensor.
public func allocateTensorOutputs(graph: Graph, sortedNodes: [NodeID]) {
  for nodeId in sortedNodes {
    guard let node = graph.nodes[nodeId] else { continue }

    // Skip if this node already has a tensor (e.g., tensorRef nodes or already-created views)
    if graph.nodeToTensor[nodeId] != nil { continue }

    // Only allocate for tensor-shaped outputs
    guard case .tensor(let shape) = node.shape else { continue }

    // Handle stateful ops (phasor, accum, latch) - need to expand their state cell for tensor operations
    // These ops have a cellId for state that was allocated before shape inference.
    // Now that we know the tensor shape, we need to re-allocate with proper size.
    switch node.op {
    case .phasor(let originalCellId), .accum(let originalCellId), .latch(let originalCellId):
      let size = shape.reduce(1, *)
      // Only re-allocate if we need more than 1 cell
      if size > 1 {
        // Update the cell allocation size (the memory remapping will handle the actual layout)
        graph.cellAllocationSizes[originalCellId] = size
      }
    // Continue to allocate output tensor below

    default:
      break
    }

    // Handle view operations - create view of input tensor instead of allocating
    switch node.op {
    case .reshape(let newShape):
      guard let inputId = node.inputs.first,
        let inputTensorId = graph.nodeToTensor[inputId],
        let inputTensor = graph.tensors[inputTensorId]
      else { continue }

      let newStrides = adaptStridesForReshape(
        inputShape: inputTensor.shape,
        inputStrides: inputTensor.strides,
        newShape: newShape
      )

      let tensorId = graph.nextTensorId
      graph.nextTensorId += 1
      graph.tensors[tensorId] = Tensor(
        id: tensorId,
        shape: newShape,
        cellId: inputTensor.cellId,
        strides: newStrides,
        offset: inputTensor.offset,
        isView: true
      )
      graph.nodeToTensor[nodeId] = tensorId
      continue

    case .transpose(let axes):
      guard let inputId = node.inputs.first,
        let inputTensorId = graph.nodeToTensor[inputId],
        let inputTensor = graph.tensors[inputTensorId]
      else { continue }

      let perm = axes.isEmpty ? Array((0..<inputTensor.shape.count).reversed()) : axes
      let newStrides = perm.map { inputTensor.strides[$0] }

      let tensorId = graph.nextTensorId
      graph.nextTensorId += 1
      graph.tensors[tensorId] = Tensor(
        id: tensorId,
        shape: shape,
        cellId: inputTensor.cellId,
        strides: newStrides,
        offset: inputTensor.offset,
        isView: true
      )
      graph.nodeToTensor[nodeId] = tensorId
      continue

    case .shrink(let ranges):
      guard let inputId = node.inputs.first,
        let inputTensorId = graph.nodeToTensor[inputId],
        let inputTensor = graph.tensors[inputTensorId]
      else { continue }

      var offset = inputTensor.offset
      for (dim, range) in ranges.enumerated() {
        if let (start, _) = range {
          offset += start * inputTensor.strides[dim]
        }
      }

      let tensorId = graph.nextTensorId
      graph.nextTensorId += 1
      graph.tensors[tensorId] = Tensor(
        id: tensorId,
        shape: shape,
        cellId: inputTensor.cellId,
        strides: inputTensor.strides,
        offset: offset,
        isView: true
      )
      graph.nodeToTensor[nodeId] = tensorId
      continue

    case .pad(let padding):
      guard let inputId = node.inputs.first,
        let inputTensorId = graph.nodeToTensor[inputId],
        let inputTensor = graph.tensors[inputTensorId]
      else { continue }

      let tensorId = graph.nextTensorId
      graph.nextTensorId += 1
      graph.tensors[tensorId] = Tensor(
        id: tensorId,
        shape: shape,
        cellId: inputTensor.cellId,
        strides: inputTensor.strides,
        offset: inputTensor.offset,
        isView: true,
        padding: padding
      )
      graph.nodeToTensor[nodeId] = tensorId
      continue

    default:
      break
    }

    // Allocate new tensor for non-view ops
    let size = shape.reduce(1, *)
    let cellId = graph.alloc(vectorWidth: size)

    let tensorId = graph.nextTensorId
    graph.nextTensorId += 1
    graph.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId)
    graph.nodeToTensor[nodeId] = tensorId
  }
}

// MARK: - Temporality Inference

/// Returns true if the op is intrinsically frame-based (produces different values each frame)
public func isIntrinsicallyFrameBased(_ op: LazyOp) -> Bool {
  switch op {
  case .phasor(_): return true  // oscillator state changes each frame
  case .deterministicPhasor: return true  // parallelizable phasor, but output still varies per frame
  case .output(_): return true  // oscillator state changes each frame
  case .accum(_): return true  // accumulator state changes each frame
  case .input(_): return true  // audio input varies each frame
  case .historyRead(_): return true  // reads temporal state (scalar or tensor)
  case .historyWrite(_): return true  // writes temporal state (scalar or tensor)
  case .historyReadWrite(_): return true  // combined temporal operation
  case .latch(_): return true  // conditional state update
  case .click(_): return true  // trigger/event based
  default: return false
  }
}

/// Returns the hop rate if the op is intrinsically hop-based.
/// Note: FFT/IFFT are NOT intrinsically hop-based from a block perspective because
/// they handle their own internal hop logic (ring buffer writes run every frame,
/// FFT computation only runs when counter == 0). However, their OUTPUT is hop-based,
/// so downstream operations that consume FFT/IFFT output should inherit hop temporality.
public func intrinsicHopRate(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> (Int, CellID)? {
  // No ops are intrinsically hop-based from a rendering perspective
  // FFT/IFFT handle their own hop logic internally
  return nil
}

/// Returns the hop rate if the node produces hop-based output (FFT, IFFT)
/// This is different from intrinsicHopRate - these nodes handle hop logic internally
/// but their OUTPUT only changes every hopSize frames.
public func producesHopBasedOutput(_ op: LazyOp, graph: Graph, nodeId: NodeID) -> (Int, CellID)? {
  switch op {
  case .fft(_, _, _, _, _, _), .ifft(_, _, _, _, _, _):
    return graph.nodeHopRate[nodeId]
  default:
    return nil
  }
}

/// Temporality propagation result containing both frame-based and hop-based node sets
public struct TemporalityResult {
  public let frameBasedNodes: Set<NodeID>
  public let hopBasedNodes: [NodeID: (Int, CellID)]
}

/// Infer temporality for all nodes. Returns sets of frame-based and hop-based nodes.
///
/// Temporality Propagation Rules:
/// - All static inputs → static output
/// - All same hopBased(N, cell) inputs → hopBased(N, cell) output
/// - Mixed hopBased rates → use fastest rate (smallest hopSize)
/// - Any frameBased input → frameBased output
/// - hopBased + static inputs → hopBased output
///
/// Special handling for FFT/IFFT:
/// - FFT/IFFT are frame-based (they have ring buffer ops that run every frame)
/// - But their OUTPUT is hop-based (only changes every hopSize frames)
/// - Operations consuming FFT/IFFT output inherit hop-based temporality
public func inferTemporality(graph: Graph, sortedNodes: [NodeID]) -> TemporalityResult {
  var frameBasedNodes = Set<NodeID>()
  var hopBasedNodes: [NodeID: (Int, CellID)] = [:]
  // Track which nodes produce hop-based output (FFT/IFFT)
  var hopProducingNodes: [NodeID: (Int, CellID)] = [:]

  for nodeId in sortedNodes {
    guard let node = graph.nodes[nodeId] else { continue }

    // Check if this node produces hop-based output (FFT/IFFT)
    // These nodes are frame-based themselves but their output is hop-based
    if let hopRate = producesHopBasedOutput(node.op, graph: graph, nodeId: nodeId) {
      hopProducingNodes[nodeId] = hopRate
      // FFT/IFFT are frame-based (ring buffer ops run every frame)
      frameBasedNodes.insert(nodeId)
      continue
    }

    // Check if intrinsically frame-based
    if isIntrinsicallyFrameBased(node.op) {
      frameBasedNodes.insert(nodeId)
      continue
    }

    // Check if intrinsically hop-based
    if let hopRate = intrinsicHopRate(node.op, graph: graph, nodeId: nodeId) {
      hopBasedNodes[nodeId] = hopRate
      continue
    }

    // Check inputs for temporality propagation
    // First, check for frame-based inputs (but exclude hop-producing nodes like FFT)
    let hasFrameBasedInput = node.inputs.contains { inputId in
      frameBasedNodes.contains(inputId) && !hopProducingNodes.keys.contains(inputId)
    }
    if hasFrameBasedInput {
      // Frame-based input forces frame-based output
      frameBasedNodes.insert(nodeId)
      continue
    }

    // Check for hop-based inputs (either from hopBasedNodes or from hop-producing nodes like FFT)
    var hopInputRates: [(Int, CellID)] = []
    for inputId in node.inputs {
      if let rate = hopBasedNodes[inputId] {
        hopInputRates.append(rate)
      } else if let rate = hopProducingNodes[inputId] {
        // Input is from a hop-producing node (FFT/IFFT) - inherit its hop rate
        hopInputRates.append(rate)
      }
    }

    if !hopInputRates.isEmpty {
      // Find the fastest hop rate (smallest hopSize) among all inputs
      // This ensures we don't miss updates from faster-changing inputs
      let fastestRate = hopInputRates.min(by: { $0.0 < $1.0 })!
      hopBasedNodes[nodeId] = fastestRate
    }
    // If no frame-based or hop-based inputs, node remains static (not added to either set)
  }

  return TemporalityResult(frameBasedNodes: frameBasedNodes, hopBasedNodes: hopBasedNodes)
}

/// Assign temporality to blocks based on their nodes.
/// A block is:
/// - frameBased if ANY of its nodes is frame-based
/// - hopBased if ALL non-static nodes share the same hop rate
/// - static_ otherwise
public func assignBlockTemporality(
  blocks: inout [Block],
  frameBasedNodes: Set<NodeID>,
  hopBasedNodes: [NodeID: (Int, CellID)]
) {
  for i in 0..<blocks.count {
    // Check for frame-based nodes first (takes precedence)
    let hasFrameBasedNode = blocks[i].nodes.contains { frameBasedNodes.contains($0) }
    if hasFrameBasedNode {
      blocks[i].temporality = .frameBased
      continue
    }

    // Check for hop-based nodes
    let hopRates = blocks[i].nodes.compactMap { hopBasedNodes[$0] }
    if let firstRate = hopRates.first {
      // Check if all hop-based nodes have the same rate
      let allSameRate = hopRates.allSatisfy { $0.0 == firstRate.0 && $0.1 == firstRate.1 }
      if allSameRate {
        blocks[i].temporality = .hopBased(hopSize: firstRate.0, counter: firstRate.1)
        continue
      }
      // Mixed rates - fall back to frame-based
      blocks[i].temporality = .frameBased
      continue
    }

    // No frame-based or hop-based nodes - block is static
    blocks[i].temporality = .static_
  }
}

public func splitBlockByStaticIfPossible(
  block: Block,
  frameBasedNodes: Set<NodeID>,
  hopBasedNodes: [NodeID: (Int, CellID)],
  graph: Graph,
  fusableChains: [FrameDependentTensorChain] = []
) -> [Block] {
  func isStatic(_ node: NodeID) -> Bool {
    return hopBasedNodes[node] == nil && !frameBasedNodes.contains(node)
  }
  func initBlock(temporality: Temporality?) -> Block {
    var b = Block(kind: block.kind)
    if let temporality = temporality {
      b.temporality = temporality
    }
    b.shape = block.shape
    b.direction = block.direction
    b.tensorIndex = block.tensorIndex
    return b
  }

  // For backward blocks, use simple splitting to preserve gradient reductions
  // The complex merging logic breaks backward gradient flow
  if block.direction == .backwards {
    var blocks: [Block] = []
    var currentBlock = initBlock(temporality: nil)
    var currentTemporality: Temporality? = nil
    for node in block.nodes {
      if isStatic(node) {
        if let temporality = currentTemporality, case .static_ = temporality {
        } else if currentTemporality == nil {
          currentBlock.temporality = .static_
        } else {
          if currentBlock.nodes.count > 0 { blocks.append(currentBlock) }
          currentBlock = initBlock(temporality: .static_)
        }
        currentTemporality = .static_
      } else {
        if let temporality = currentTemporality, case .frameBased = temporality {
        } else if currentTemporality == nil {
          currentBlock.temporality = .frameBased
        } else {
          if currentBlock.nodes.count > 0 { blocks.append(currentBlock) }
          currentBlock = initBlock(temporality: .frameBased)
        }
        currentTemporality = .frameBased
      }
      currentBlock.nodes.append(node)
    }
    if currentBlock.nodes.count > 0 { blocks.append(currentBlock) }
    return blocks
  }

  // For forward blocks, use tensor-aware splitting:
  // - Static tensor ops get their own blocks (can be SIMD parallelized)
  // - Scalar-only blocks don't need splitting (no parallelization benefit)

  func hasTensorShapedOps(_ nodes: [NodeID]) -> Bool {
    for nodeId in nodes {
      if let node = graph.nodes[nodeId], case .tensor = node.shape {
        return true
      }
    }
    return false
  }

  // If block has no tensor ops at all, don't bother splitting - no parallelization benefit
  if !hasTensorShapedOps(block.nodes) && fusableChains.isEmpty {
    return [block]
  }

  // Build chain node set for fusable chains
  var nodeToChain: [NodeID: FrameDependentTensorChain] = [:]
  for chain in fusableChains {
    for nodeId in chain.chainNodes {
      nodeToChain[nodeId] = chain
    }
  }

  // First pass: identify segments
  var segments: [(nodes: [NodeID], isStatic: Bool, chain: FrameDependentTensorChain?)] = []
  var currentNodes: [NodeID] = []
  var currentIsStatic: Bool? = nil
  var currentChain: FrameDependentTensorChain? = nil

  for node in block.nodes {
    if let chain = nodeToChain[node] {
      // Chain nodes are frame-based
      let nodeIsStatic = false
      if currentIsStatic != nil
        && (currentIsStatic != nodeIsStatic
          || currentChain?.transitionNodeId != chain.transitionNodeId)
      {
        if !currentNodes.isEmpty {
          segments.append((nodes: currentNodes, isStatic: currentIsStatic!, chain: currentChain))
        }
        currentNodes = []
      }
      currentIsStatic = nodeIsStatic
      currentChain = chain
      currentNodes.append(node)
    } else {
      let nodeIsStatic = isStatic(node)
      if currentIsStatic == nil {
        currentIsStatic = nodeIsStatic
        currentChain = nil
      } else if currentIsStatic != nodeIsStatic || currentChain != nil {
        if !currentNodes.isEmpty {
          segments.append((nodes: currentNodes, isStatic: currentIsStatic!, chain: currentChain))
        }
        currentNodes = []
        currentIsStatic = nodeIsStatic
        currentChain = nil
      }
      currentNodes.append(node)
    }
  }
  if !currentNodes.isEmpty, let isStatic = currentIsStatic {
    segments.append((nodes: currentNodes, isStatic: isStatic, chain: currentChain))
  }

  // Second pass: just merge adjacent same-type segments
  // NOTE: We previously tried merging scalar static ops into frame-based to reduce kernel
  // overhead, but this breaks gradient flow. Only keep tensor static ops separate.
  var mergedSegments: [(nodes: [NodeID], isStatic: Bool, chain: FrameDependentTensorChain?)] = []
  for segment in segments {
    if segment.chain != nil {
      mergedSegments.append(segment)
      continue
    }

    // Merge adjacent same-type segments
    if let lastIdx = mergedSegments.indices.last,
      mergedSegments[lastIdx].isStatic == segment.isStatic && mergedSegments[lastIdx].chain == nil
    {
      mergedSegments[lastIdx].nodes.append(contentsOf: segment.nodes)
    } else {
      mergedSegments.append(segment)
    }
  }

  // Convert to blocks
  var blocks: [Block] = []
  for segment in mergedSegments {
    var b = initBlock(temporality: segment.isStatic ? .static_ : .frameBased)
    b.nodes = segment.nodes
    if let chain = segment.chain {
      b.frameTensorChain = chain
    }
    blocks.append(b)
  }

  return blocks.isEmpty ? [block] : blocks
}

public func extractStaticOpsIntoBlocks(
  blocks: [Block],
  frameBasedNodes: Set<NodeID>,
  hopBasedNodes: [NodeID: (Int, CellID)],
  graph: Graph,
  fusableChains: [FrameDependentTensorChain] = []
) -> [Block] {
  var extractedBlocks: [Block] = []

  for block in blocks {
    for b in splitBlockByStaticIfPossible(
      block: block, frameBasedNodes: frameBasedNodes, hopBasedNodes: hopBasedNodes, graph: graph,
      fusableChains: fusableChains)
    {
      extractedBlocks.append(b)
    }
  }
  return extractedBlocks
}

// MARK: - Frame-Dependent Tensor Chain Detection

/// Represents a fusable chain from a transition point (where static tensor becomes frame-dependent)
/// to a terminal scalar reduction. The entire chain can be SIMD-parallelized across frames.
public struct FrameDependentTensorChain {
  /// Node where static tensor becomes frame-dependent (e.g., peekRow)
  public let transitionNodeId: NodeID
  /// Terminal scalar reduction node (e.g., sum)
  public let reductionNodeId: NodeID
  /// All nodes in the chain (including transition and reduction)
  public let chainNodes: Set<NodeID>
  /// Shape of the tensor being processed
  public let tensorShape: [Int]
}

/// Detect frame-dependent tensor chains that can be fused for SIMD-across-frames execution.
///
/// A valid chain:
/// 1. Starts at a "transition point" - a node with tensor output, frame-based input, and static tensor input
///    (e.g., peekRow reading from static tensor with frame-dependent row index)
/// 2. Continues through element-wise tensor operations (mul, add, sin, cos, etc.)
/// 3. Ends at a scalar reduction (sum)
/// 4. Has no cross-element dependencies within the chain
///
/// When such a chain is detected, each frame can independently compute the entire tensor chain
/// in thread-local storage, then output the final scalar to a frame-indexed scratch buffer.
public func detectFrameDependentTensorChains(
  graph: Graph,
  sortedNodes: [NodeID],
  frameBasedNodes: Set<NodeID>
) -> [FrameDependentTensorChain] {
  var chains: [FrameDependentTensorChain] = []

  // Build consumer map for forward traversal
  var consumers: [NodeID: [NodeID]] = [:]
  for (nodeId, node) in graph.nodes {
    for inputId in node.inputs {
      consumers[inputId, default: []].append(nodeId)
    }
  }

  // Element-wise ops that can be part of a fusable chain
  let elementWiseOps: Set<String> = [
    "mul", "add", "sub", "div", "sin", "cos", "tan", "tanh", "exp", "log", "sqrt",
    "abs", "sign", "floor", "ceil", "round", "pow", "min", "max", "gswitch",
  ]

  func isElementWise(_ op: LazyOp) -> Bool {
    switch op {
    case .mul, .add, .sub, .div, .sin, .cos, .tan, .tanh, .exp, .log, .sqrt,
      .abs, .sign, .floor, .ceil, .round, .pow, .min, .max, .gswitch,
      .deterministicPhasor:
      return true
    default:
      return false
    }
  }

  func isTransitionPoint(_ node: Node) -> Bool {
    // A transition point has:
    // 1. Tensor output (checked via shape)
    // 2. At least one frame-based input (checked via frameBasedNodes)
    // 3. At least one static tensor input (not frame-based, has tensor shape)
    guard case .tensor = node.shape else { return false }

    let hasFrameBasedInput = node.inputs.contains { frameBasedNodes.contains($0) }
    let hasStaticTensorInput = node.inputs.contains { inputId in
      guard let inputNode = graph.nodes[inputId] else { return false }
      if frameBasedNodes.contains(inputId) { return false }
      if case .tensor = inputNode.shape { return true }
      return false
    }

    // peekRow is the canonical transition point
    if case .peekRow = node.op {
      return hasFrameBasedInput && hasStaticTensorInput
    }

    return false
  }

  // Find transition points
  for nodeId in sortedNodes {
    guard let node = graph.nodes[nodeId] else { continue }
    guard isTransitionPoint(node) else { continue }

    // Trace forward through element-wise ops to find terminal reduction
    var chainNodes: Set<NodeID> = [nodeId]
    var frontier: [NodeID] = [nodeId]
    var reductionNode: NodeID? = nil
    var tensorShape: [Int]? = nil

    if case .tensor(let shape) = node.shape {
      tensorShape = shape
    }

    while !frontier.isEmpty && reductionNode == nil {
      let current = frontier.removeFirst()

      for consumerId in consumers[current] ?? [] {
        guard let consumer = graph.nodes[consumerId] else { continue }

        // Check if this is the terminal reduction
        if case .sum = consumer.op {
          chainNodes.insert(consumerId)
          reductionNode = consumerId
          break
        }

        // Check if this is an element-wise op that can be part of the chain
        if isElementWise(consumer.op) {
          // Verify it has tensor shape matching our chain
          if case .tensor(let shape) = consumer.shape, shape == tensorShape {
            chainNodes.insert(consumerId)
            frontier.append(consumerId)
          }
        }
      }
    }

    // Valid chain found - must end in reduction
    if let reduction = reductionNode, let shape = tensorShape {
      chains.append(
        FrameDependentTensorChain(
          transitionNodeId: nodeId,
          reductionNodeId: reduction,
          chainNodes: chainNodes,
          tensorShape: shape
        ))
    }
  }

  return chains
}

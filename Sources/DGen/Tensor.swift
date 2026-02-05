/// Represents a single view transformation applied to a tensor.
/// Transforms are stored in order from base tensor to final view shape.
/// To read an element: walk backwards through transforms, mapping output indices to base indices.
public enum ViewTransform: Equatable {
  /// Virtual padding - indices outside inner region return 0
  /// padding: per-axis (left, right) padding amounts
  /// inputShape: shape before this transform (the inner, unpadded shape)
  case pad(padding: [(left: Int, right: Int)], inputShape: [Int])

  /// Reinterpret memory layout with custom strides
  /// outputShape: shape after this transform
  /// strides: strides for the output shape
  /// offset: additional offset from base
  /// inputShape: shape before this transform
  case asStrided(outputShape: [Int], strides: [Int], offset: Int, inputShape: [Int])

  /// Change shape preserving element order (row-major)
  /// outputShape: shape after reshape
  /// inputShape: shape before reshape
  case reshape(outputShape: [Int], inputShape: [Int])

  /// Permute dimensions
  /// axes: permutation of dimension indices
  /// inputShape: shape before transpose
  case transpose(axes: [Int], inputShape: [Int])

  /// Slice along dimensions - nil means keep all
  /// ranges: per-axis (start, end) or nil
  /// inputShape: shape before shrink
  case shrink(ranges: [(start: Int, end: Int)?], inputShape: [Int])

  /// Broadcast size-1 dims to target
  /// targetShape: shape after expand
  /// inputShape: shape before expand (has size-1 dims)
  case expand(targetShape: [Int], inputShape: [Int])

  /// Tile tensor via modular indexing
  /// innerShape: original shape before repeat (for modulo)
  /// outputShape: tiled shape after repeat
  case repeatTile(innerShape: [Int], outputShape: [Int])

  public static func == (lhs: ViewTransform, rhs: ViewTransform) -> Bool {
    switch (lhs, rhs) {
    case let (.pad(lp, lis), .pad(rp, ris)):
      return lis == ris && lp.count == rp.count && zip(lp, rp).allSatisfy { $0.left == $1.left && $0.right == $1.right }
    case let (.asStrided(lo, ls, loff, lis), .asStrided(ro, rs, roff, ris)):
      return lo == ro && ls == rs && loff == roff && lis == ris
    case let (.reshape(lo, lis), .reshape(ro, ris)):
      return lo == ro && lis == ris
    case let (.transpose(la, lis), .transpose(ra, ris)):
      return la == ra && lis == ris
    case let (.shrink(lr, lis), .shrink(rr, ris)):
      if lis != ris || lr.count != rr.count { return false }
      for (l, r) in zip(lr, rr) {
        switch (l, r) {
        case (nil, nil): continue
        case let (ls?, rs?): if ls.start != rs.start || ls.end != rs.end { return false }
        default: return false
        }
      }
      return true
    case let (.expand(lt, lis), .expand(rt, ris)):
      return lt == rt && lis == ris
    case let (.repeatTile(li, lo), .repeatTile(ri, ro)):
      return li == ri && lo == ro
    default:
      return false
    }
  }
}

public struct Tensor {
  public let id: TensorID
  public let shape: Shape              // Final shape after all transforms
  public var cellId: CellID            // Base memory cell
  public var data: [Float]?            // Initial data to be injected by runtime

  // Transform chain from base to final view
  public let baseShape: [Int]          // Shape of actual data in memory
  public let baseStrides: [Int]        // Strides of actual data (row-major)
  public let transforms: [ViewTransform]  // Applied in order from base → final

  // Computed property for quick view check
  public var isView: Bool { !transforms.isEmpty }

  // Allocation flags
  public var isLazy: Bool              // True if cellId is a lazy placeholder (not yet allocated)
  public var materialize: Bool         // True if this tensor should be stored in memory (for realize())

  public init(
    id: TensorID, shape: Shape, cellId: CellID, data: [Float]? = nil,
    baseShape: [Int]? = nil,
    baseStrides: [Int]? = nil,
    transforms: [ViewTransform] = [],
    isLazy: Bool = false,
    materialize: Bool = false
  ) {
    self.id = id
    self.shape = shape
    self.cellId = cellId
    self.data = data
    self.transforms = transforms
    self.isLazy = isLazy
    self.materialize = materialize
    // Base shape/strides default to final shape with row-major layout
    self.baseShape = baseShape ?? shape
    self.baseStrides = baseStrides ?? Tensor.computeRowMajorStrides(baseShape ?? shape)
  }

  /// Padding amounts - extracts from pad transform if present
  public var padding: [(left: Int, right: Int)]? {
    for transform in transforms {
      if case .pad(let padding, _) = transform {
        return padding
      }
    }
    return nil
  }

  /// Effective strides - from asStrided transform or baseStrides
  public var strides: [Int] {
    for transform in transforms.reversed() {
      if case .asStrided(_, let strides, _, _) = transform {
        return strides
      }
    }
    return baseStrides
  }

  /// Memory offset - from asStrided transform or 0
  public var offset: Int {
    for transform in transforms {
      if case .asStrided(_, _, let offset, _) = transform {
        return offset
      }
    }
    return 0
  }

  /// Total number of elements in this tensor (final shape)
  public var size: Int {
    shape.reduce(1, *)
  }

  /// Total number of elements in the base tensor (actual memory)
  public var baseSize: Int {
    baseShape.reduce(1, *)
  }

  /// Compute row-major strides for a given shape
  /// e.g. shape [M, N, K] -> strides [N*K, K, 1]
  public static func computeRowMajorStrides(_ shape: Shape) -> [Int] {
    guard !shape.isEmpty else { return [] }
    var strides = [Int](repeating: 1, count: shape.count)
    for i in stride(from: shape.count - 2, through: 0, by: -1) {
      strides[i] = strides[i + 1] * shape[i + 1]
    }
    return strides
  }

  /// Check if this tensor is contiguous (strides match row-major layout of base)
  public var isContiguous: Bool {
    transforms.isEmpty && baseStrides == Tensor.computeRowMajorStrides(baseShape)
  }
}

/// Inject tensor data from graph into runtime memory buffer
/// Call this after allocating memory but before running the kernel
public func injectTensorData(
  graph: Graph,
  cellAllocations: CellAllocations,
  memory: UnsafeMutablePointer<Float>
) {
  for (_, tensor) in graph.tensors {
    guard let data = tensor.data else { continue }

    // Get physical memory offset from cell mapping
    let physicalOffset = cellAllocations.cellMappings[tensor.cellId] ?? tensor.cellId

    // Copy data into memory
    for (i, value) in data.enumerated() {
      if i < tensor.size {
        memory[physicalOffset + i] = value
      }
    }
  }
}

/// Inject tensor data using CompilationResult (convenience overload)
public func injectTensorData(
  result: CompilationResult,
  memory: UnsafeMutablePointer<Float>
) {
  injectTensorData(
    graph: result.graph,
    cellAllocations: result.cellAllocations,
    memory: memory
  )
}

extension Graph {
  /// Create a tensor with given shape (uninitialized)
  public func tensor(shape: Shape) -> NodeID {
    return tensor(shape: shape, data: nil)
  }

  /// Create a tensor with given shape and initial data
  public func tensor(shape: Shape, data: [Float]?) -> NodeID {
    var size = 1
    for dim in shape {
      size *= dim
    }

    let cellId = alloc(vectorWidth: size)
    let tensorId = nextTensorId
    nextTensorId += 1
    self.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId, data: data)
    self.cellToTensor[cellId] = tensorId
    let nodeId = self.n(.tensorRef(tensorId), [], shape: .tensor(shape))
    self.nodeToTensor[nodeId] = tensorId
    return nodeId
  }

  /// Create a tensor from a flat array (infers 1D shape)
  public func tensor(_ data: [Float]) -> NodeID {
    return tensor(shape: [data.count], data: data)
  }

  /// Create a tensor from a 2D array
  public func tensor(_ data: [[Float]]) -> NodeID {
    let rows = data.count
    let cols = data.first?.count ?? 0
    let flat = data.flatMap { $0 }
    return tensor(shape: [rows, cols], data: flat)
  }

  /// Create a tensor filled with zeros
  public func zeros(shape: Shape) -> NodeID {
    let size = shape.reduce(1, *)
    return tensor(shape: shape, data: [Float](repeating: 0.0, count: size))
  }

  /// Create a tensor filled with ones
  public func ones(shape: Shape) -> NodeID {
    let size = shape.reduce(1, *)
    return tensor(shape: shape, data: [Float](repeating: 1.0, count: size))
  }

  /// Create a tensor filled with a constant value
  public func full(shape: Shape, value: Float) -> NodeID {
    let size = shape.reduce(1, *)
    return tensor(shape: shape, data: [Float](repeating: value, count: size))
  }

  /// Stack scalar nodes into a tensor (dynamic, per-frame values)
  /// At each frame, writes the current value of each scalar into the tensor
  /// Example: stack([phasor1, phasor2, phasor3]) creates a [3] tensor
  public func stack(_ scalars: [NodeID], shape: Shape? = nil) throws -> NodeID {
    let count = scalars.count
    guard count > 0 else {
      throw DGenError.tensorError(op: "stack", reason: "requires at least one scalar")
    }

    let finalShape = shape ?? [count]
    let size = finalShape.reduce(1, *)

    guard size == count else {
      throw DGenError.tensorError(
        op: "stack",
        reason: "shape \(finalShape) size \(size) doesn't match scalar count \(count)")
    }

    // Allocate tensor
    let cellId = alloc(vectorWidth: size)
    let tensorId = nextTensorId
    nextTensorId += 1
    tensors[tensorId] = Tensor(id: tensorId, shape: finalShape, cellId: cellId)
    cellToTensor[cellId] = tensorId

    // Create writes for each scalar
    var writes: [NodeID] = []
    for (i, scalar) in scalars.enumerated() {
      let indexNode = n(.constant(Float(i)))
      let writeNode = n(.memoryWrite(cellId), indexNode, scalar)
      writes.append(writeNode)
    }

    // Create tensorRef
    let tensorRefNode = n(.tensorRef(tensorId), [], shape: .tensor(finalShape))
    nodeToTensor[tensorRefNode] = tensorId

    // Chain: write0 -> write1 -> ... -> tensorRef using seq
    // This ensures all writes happen before the tensor is read
    var result = writes[0]
    for i in 1..<writes.count {
      result = n(.seq, result, writes[i])
    }
    result = n(.seq, result, tensorRefNode)

    // Map the final seq node to the same tensor so downstream ops can find it
    nodeToTensor[result] = tensorId

    return result
  }

  // MARK: - Tensor View Helpers

  /// Get the tensor associated with a node
  public func getTensor(_ nodeId: NodeID) throws -> Tensor {
    guard let tensorId = nodeToTensor[nodeId],
      let tensor = tensors[tensorId]
    else {
      throw DGenError.tensorError(op: "getTensor", reason: "tensor not found")
    }
    return tensor
  }

  /// Create a tensor view with a new transform appended to the chain.
  /// For derived ops without tensors, creates node only - tensor created during allocation.
  private func createViewWithTransform(
    input: NodeID,
    op: LazyOp,
    newShape: Shape,
    transform: ViewTransform
  ) throws -> NodeID {
    // Create the node first
    let nodeId = n(op, [input], shape: .tensor(newShape))

    // If input has a concrete tensor, create the view tensor now
    if let inputTensor = nodeToTensor[input].flatMap({ tensors[$0] }) {
      let tensorId = nextTensorId
      nextTensorId += 1

      // Append the new transform to the input tensor's chain
      var newTransforms = inputTensor.transforms
      newTransforms.append(transform)

      tensors[tensorId] = Tensor(
        id: tensorId,
        shape: newShape,
        cellId: inputTensor.cellId,
        baseShape: inputTensor.baseShape,
        baseStrides: inputTensor.baseStrides,
        transforms: newTransforms
      )

      nodeToTensor[nodeId] = tensorId
    }
    // If no concrete tensor, the view will be created during allocateTensorOutputs

    return nodeId
  }

  // MARK: - Tensor Views (Reshape/Transpose/Shrink)

  /// Reshape a tensor to a new shape (metadata only, no data movement)
  public func reshape(_ input: NodeID, to newShape: Shape) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "reshape", reason: "requires tensor input")
    }

    let inputSize = inputShape.reduce(1, *)
    guard inputSize == newShape.reduce(1, *) else {
      throw DGenError.tensorError(
        op: "reshape",
        reason: "size mismatch: \(inputSize) vs \(newShape.reduce(1, *))")
    }

    let transform = ViewTransform.reshape(outputShape: newShape, inputShape: inputShape)
    return try createViewWithTransform(input: input, op: .reshape(newShape), newShape: newShape, transform: transform)
  }

  /// Transpose a tensor by permuting axes
  public func transpose(_ input: NodeID, axes: [Int]? = nil) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "transpose", reason: "requires tensor input")
    }

    let ndim = inputShape.count
    let perm = axes ?? Array((0..<ndim).reversed())

    guard perm.count == ndim else {
      throw DGenError.tensorError(
        op: "transpose", reason: "axes must have \(ndim) elements, got \(perm.count)")
    }

    let newShape = perm.map { inputShape[$0] }
    let transform = ViewTransform.transpose(axes: perm, inputShape: inputShape)
    return try createViewWithTransform(input: input, op: .transpose(perm), newShape: newShape, transform: transform)
  }

  /// Shrink/slice a tensor along each axis (metadata only, no data movement)
  /// ranges: for each dimension, either nil (keep all) or (start, end) tuple
  public func shrink(_ input: NodeID, ranges: [(Int, Int)?]) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "shrink", reason: "requires tensor input")
    }

    guard ranges.count == inputShape.count else {
      throw DGenError.tensorError(
        op: "shrink",
        reason: "ranges count \(ranges.count) must match ndim \(inputShape.count)")
    }

    var newShape = [Int]()
    for (dim, range) in ranges.enumerated() {
      if let (start, end) = range {
        guard start >= 0 && end <= inputShape[dim] && start < end else {
          throw DGenError.tensorError(
            op: "shrink",
            reason: "invalid range (\(start), \(end)) for dimension \(dim) with size \(inputShape[dim])"
          )
        }
        newShape.append(end - start)
      } else {
        newShape.append(inputShape[dim])
      }
    }

    // Convert ranges to (start, end)? format for the transform
    let transformRanges: [(start: Int, end: Int)?] = ranges.enumerated().map { dim, range in
      if let (start, end) = range {
        return (start: start, end: end)
      }
      return nil
    }

    let transform = ViewTransform.shrink(ranges: transformRanges, inputShape: inputShape)
    return try createViewWithTransform(input: input, op: .shrink(ranges), newShape: newShape, transform: transform)
  }

  public func pad(_ input: NodeID, padding: [(Int, Int)]) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "pad", reason: "requires tensor input")
    }
    guard padding.count == inputShape.count else {
      throw DGenError.tensorError(
        op: "pad",
        reason: "padding count \(padding.count) must match ndim \(inputShape.count)")
    }

    let newShape = zip(inputShape, padding).map { dim, pad in dim + pad.0 + pad.1 }
    let paddingTuples = padding.map { (left: $0.0, right: $0.1) }
    let transform = ViewTransform.pad(padding: paddingTuples, inputShape: inputShape)
    return try createViewWithTransform(input: input, op: .pad(padding), newShape: newShape, transform: transform)
  }

  /// Expand a tensor by broadcasting size-1 dimensions to target shape (stride=0 view, no data copy)
  /// e.g. [2, 1, 3] -> expandView to [2, 4, 3] makes dim 1 appear repeated 4 times via stride=0
  public func expandView(_ input: NodeID, to targetShape: Shape) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "expandView", reason: "requires tensor input")
    }

    guard inputShape.count == targetShape.count else {
      throw DGenError.tensorError(
        op: "expandView",
        reason: "shape rank mismatch: \(inputShape.count) vs \(targetShape.count)")
    }

    // Validate: each dim must be same size OR input dim must be 1
    for (i, (inDim, targetDim)) in zip(inputShape, targetShape).enumerated() {
      guard inDim == targetDim || inDim == 1 else {
        throw DGenError.tensorError(
          op: "expandView",
          reason: "dim \(i): can only expand size-1 dims, got \(inDim) -> \(targetDim)")
      }
    }

    let transform = ViewTransform.expand(targetShape: targetShape, inputShape: inputShape)
    return try createViewWithTransform(input: input, op: .expandView(targetShape), newShape: targetShape, transform: transform)
  }

  /// Repeat/tile a tensor along each dimension (modular index view, no data copy)
  /// e.g. [2, 3] repeated by [2, 3] -> [4, 9] where each element appears multiple times
  /// Implemented via modular indexing: index[i] % originalShape[i]
  public func repeatView(_ input: NodeID, repeats: [Int]) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "repeatView", reason: "requires tensor input")
    }

    guard inputShape.count == repeats.count else {
      throw DGenError.tensorError(
        op: "repeatView",
        reason: "repeats count \(repeats.count) must match ndim \(inputShape.count)")
    }

    // Validate all repeats are positive
    for (i, r) in repeats.enumerated() {
      guard r >= 1 else {
        throw DGenError.tensorError(
          op: "repeatView",
          reason: "repeat count must be >= 1, got \(r) at dim \(i)")
      }
    }

    // Compute new shape: original * repeats
    let newShape = zip(inputShape, repeats).map { $0 * $1 }

    let transform = ViewTransform.repeatTile(innerShape: inputShape, outputShape: newShape)
    return try createViewWithTransform(input: input, op: .repeatView(repeats), newShape: newShape, transform: transform)
  }

  /// Create a strided view of a tensor with arbitrary shape and strides.
  /// This is the fundamental building block for view operations like pool.
  ///
  /// WARNING: This operation can create out-of-bounds accesses if strides/shape
  /// are not chosen carefully. The caller is responsible for ensuring the view
  /// stays within the underlying storage bounds.
  ///
  /// - Parameters:
  ///   - input: The source tensor
  ///   - shape: The new shape for the view
  ///   - strides: The strides for each dimension (how many elements to skip)
  ///   - offset: Additional offset from the tensor's current offset
  public func asStrided(
    _ input: NodeID,
    shape: [Int],
    strides: [Int],
    offset: Int = 0
  ) throws -> NodeID {
    // Get input shape - works for both concrete tensors and derived ops
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "asStrided", reason: "requires tensor input")
    }

    guard shape.count == strides.count else {
      throw DGenError.tensorError(
        op: "asStrided",
        reason: "shape and strides must have same count: \(shape.count) vs \(strides.count)")
    }

    let transform = ViewTransform.asStrided(outputShape: shape, strides: strides, offset: offset, inputShape: inputShape)
    return try createViewWithTransform(input: input, op: .asStrided(shape, strides), newShape: shape, transform: transform)
  }

  /// Pool operation: transforms a tensor to expose sliding windows as extra dimensions.
  /// Transforms [...batch, H, W] → [...batch, oH, oW, kH, kW] for 2D pooling.
  ///
  /// This is the "im2col" transformation done via view operations (no data copy)
  ///
  /// Supports both non-overlapping (stride >= kernel) and overlapping (stride < kernel) windows.
  public func pool(
    _ input: NodeID,
    kernelSize: [Int],
    stride: [Int]? = nil
  ) throws -> NodeID {
    // Get input shape - works for both concrete tensors and op-derived tensors
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(op: "pool", reason: "requires tensor input")
    }

    let spatialDims = kernelSize.count
    guard inputShape.count >= spatialDims else {
      throw DGenError.tensorError(
        op: "pool",
        reason: "input must have at least \(spatialDims) dims, got \(inputShape.count)")
    }

    let stride = stride ?? kernelSize  // Default: non-overlapping (stride == kernel)
    guard stride.count == spatialDims else {
      throw DGenError.tensorError(
        op: "pool",
        reason: "stride count \(stride.count) must match kernel count \(spatialDims)")
    }

    // Split into batch dims and spatial dims
    let batchDims = inputShape.count - spatialDims
    let batchShape = Array(inputShape.prefix(batchDims))
    let spatialShape = Array(inputShape.suffix(spatialDims))

    // Use row-major strides for the current shape
    // The transform chain will handle mapping to base memory correctly
    let inputStrides = Tensor.computeRowMajorStrides(inputShape)
    let batchStrides = Array(inputStrides.prefix(batchDims))
    let spatialStrides = Array(inputStrides.suffix(spatialDims))

    // Compute output spatial dims: o = (input - kernel) / stride + 1
    let outputSpatial = zip(zip(spatialShape, kernelSize), stride).map { dims, s in
      (dims.0 - dims.1) / s + 1
    }

    // Validate we have at least one output position per dimension
    for (i, o) in outputSpatial.enumerated() {
      guard o > 0 else {
        throw DGenError.tensorError(
          op: "pool",
          reason: "kernel \(kernelSize[i]) too large for spatial dim \(spatialShape[i])")
      }
    }

    let outputStrides: [Int] =
      batchStrides + zip(spatialStrides, stride).map { $0 * $1 } + spatialStrides

    // Build final shape: [...batch, o1, o2, ..., k1, k2, ...]
    let outputShape = batchShape + outputSpatial + kernelSize

    return try asStrided(input, shape: outputShape, strides: outputStrides)
  }

  /// 2D Convolution using the pool-based "im2col" approach (view operations, then multiply-sum)
  ///
  /// This implements conv2d by:
  /// 1. Using pool to extract sliding windows: [H, W] → [oH, oW, kH, kW]
  /// 2. Broadcasting kernel to match: [kH, kW] → [oH, oW, kH, kW]
  /// 3. Element-wise multiply
  /// 4. Sum over kernel dimensions → [oH, oW]
  ///
  /// This approach is zero-copy for the input (view operations only) and enables
  /// the same code path for both overlapping and non-overlapping strides.
  ///
  /// - Parameters:
  ///   - input: Input tensor of shape [H, W]
  ///   - kernel: Kernel tensor of shape [kH, kW]
  ///   - stride: Convolution stride (default [1, 1])
  /// - Returns: Output tensor of shape [oH, oW]
  public func conv2dView(
    _ input: NodeID,
    kernel: NodeID,
    stride: [Int] = [1, 1]
  ) throws -> NodeID {
    // Get input shape
    guard let inputNode = nodes[input], case .tensor(let inputShape) = inputNode.shape,
      inputShape.count == 2
    else {
      throw DGenError.tensorError(op: "conv2dView", reason: "input must be 2D tensor [H, W]")
    }

    // Get kernel shape
    guard let kernelNode = nodes[kernel], case .tensor(let kernelShape) = kernelNode.shape,
      kernelShape.count == 2
    else {
      throw DGenError.tensorError(op: "conv2dView", reason: "kernel must be 2D tensor [kH, kW]")
    }

    let (kH, kW) = (kernelShape[0], kernelShape[1])

    // Step 1: Pool input to get windows
    // [H, W] → [oH, oW, kH, kW]
    let pooled = try pool(input, kernelSize: [kH, kW], stride: stride)

    guard let pooledNode = nodes[pooled], case .tensor(let pooledShape) = pooledNode.shape else {
      throw DGenError.tensorError(op: "conv2dView", reason: "pool failed")
    }

    let (oH, oW) = (pooledShape[0], pooledShape[1])

    // Step 2: Reshape kernel [kH, kW] → [1, 1, kH, kW] and broadcast to [oH, oW, kH, kW]
    let reshapedKernel = try reshape(kernel, to: [1, 1, kH, kW])
    let broadcastedKernel = try expandView(reshapedKernel, to: [oH, oW, kH, kW])

    // Step 3: Element-wise multiply pooled windows with kernel
    let product = n(.mul, [pooled, broadcastedKernel])

    // Step 4: Sum over kernel dimensions to get [oH, oW]
    let sumKW = try sum(product, axis: -1)  // [oH, oW, kH, kW] → [oH, oW, kH]
    let sumKH = try sum(sumKW, axis: -1)  // [oH, oW, kH] → [oH, oW]

    return sumKH
  }

  /// Sum a tensor along a specific axis, reducing that dimension
  /// e.g. [M, N, K].sum(axis: -1) -> [M, N]
  public func sum(_ input: NodeID, axis: Int) throws -> NodeID {
    guard let inputNode = nodes[input] else {
      throw DGenError.tensorError(op: "sum", reason: "input node not found")
    }
    guard case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(
        op: "sum",
        reason: "requires tensor input, got \(String(describing: inputNode.shape))")
    }

    // Handle negative axis
    let ndim = inputShape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.tensorError(
        op: "sum", reason: "axis \(axis) out of range for tensor with \(ndim) dimensions")
    }

    // Compute output shape (remove the reduced axis)
    var outputShape = inputShape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      // Reducing to scalar
      return n(.sum, [input])
    }

    // Allocate output tensor using lazy cell for frame-aware allocation support
    // This is necessary because sumAxis output may be frame-based if input is frame-based
    let outputCellId = reserveLazyCellId()
    let outputTensorId = nextTensorId
    nextTensorId += 1
    tensors[outputTensorId] = Tensor(
      id: outputTensorId, shape: outputShape, cellId: outputCellId, isLazy: true)
    cellToTensor[outputCellId] = outputTensorId

    let nodeId = n(.sumAxis(normalizedAxis), [input], shape: .tensor(outputShape))
    nodeToTensor[nodeId] = outputTensorId
    return nodeId
  }

  /// Matrix multiply: A[M,K] @ B[K,N] -> C[M,N]
  /// Implemented as: reshape + broadcast multiply + sum
  public func matmul(_ a: NodeID, _ b: NodeID) throws -> NodeID {
    // Get shapes from nodes - works even for derived tensor ops like (tensor * scalar)
    guard let aNode = nodes[a], case .tensor(let aShape) = aNode.shape,
      let bNode = nodes[b], case .tensor(let bShape) = bNode.shape
    else {
      throw DGenError.tensorError(op: "matmul", reason: "requires tensor inputs")
    }

    guard aShape.count == 2, bShape.count == 2 else {
      throw DGenError.tensorError(
        op: "matmul",
        reason: "requires 2D tensors, got \(aShape.count)D and \(bShape.count)D")
    }

    let M = aShape[0]
    let K = aShape[1]
    let N = bShape[1]

    guard bShape[0] == K else {
      throw DGenError.tensorError(
        op: "matmul", reason: "dimension mismatch: [\(M),\(K)] @ [\(bShape[0]),\(N)]")
    }

    // A: [M, K] -> [M, 1, K]
    let aReshaped = try reshape(a, to: [M, 1, K])

    // B: [K, N] -> [1, N, K] (transpose then reshape)
    let bTransposed = try transpose(b, axes: [1, 0])  // [K, N] -> [N, K]
    let bReshaped = try reshape(bTransposed, to: [1, N, K])

    // Broadcast multiply: [M, 1, K] * [1, N, K] -> [M, N, K]
    let product = n(.mul, [aReshaped, bReshaped], shape: .tensor([M, N, K]))

    // Sum along last axis: [M, N, K] -> [M, N]
    return try sum(product, axis: -1)
  }

  // MARK: - Tensor History (State Buffers)

  /// A handle to a tensor history buffer (for state that persists across frames)
  public struct TensorHistoryBuffer {
    public let cellId: CellID
    public let shape: Shape
    public let tensorId: TensorID
  }

  /// Create a history buffer for tensor state that persists across frames
  /// Use with tensorHistoryRead/tensorHistoryWrite for membrane simulation etc.
  public func tensorHistoryBuffer(shape: Shape, data: [Float]? = nil) -> TensorHistoryBuffer {
    let size = shape.reduce(1, *)
    let cellId = alloc(vectorWidth: size)
    let tensorId = nextTensorId
    nextTensorId += 1
    tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId, data: data)
    cellToTensor[cellId] = tensorId
    return TensorHistoryBuffer(cellId: cellId, shape: shape, tensorId: tensorId)
  }

  /// Read the current state from a tensor history buffer
  @discardableResult
  public func tensorHistoryRead(_ buffer: TensorHistoryBuffer) -> NodeID {
    // Allocate a LAZY output cell for the read result
    // This is critical: the read copies from history cell to output cell
    // If we reuse the history cell, the read becomes a no-op!
    // Using lazy cell allows frame-aware allocation when output crosses block boundaries
    let outputCellId = reserveLazyCellId()
    let outputTensorId = nextTensorId
    nextTensorId += 1
    tensors[outputTensorId] = Tensor(
      id: outputTensorId, shape: buffer.shape, cellId: outputCellId, isLazy: true)
    cellToTensor[outputCellId] = outputTensorId

    // Use unified historyRead - checks cellToTensor to determine if tensor or scalar
    let nodeId = n(.historyRead(buffer.cellId), [], shape: .tensor(buffer.shape))
    nodeToTensor[nodeId] = outputTensorId
    return nodeId
  }

  /// Write new state to a tensor history buffer
  @discardableResult
  public func tensorHistoryWrite(_ buffer: TensorHistoryBuffer, _ value: NodeID) -> NodeID {
    // Use unified historyWrite - checks cellToTensor to determine if tensor or scalar
    return n(.historyWrite(buffer.cellId), [value], shape: .tensor(buffer.shape))
  }

  public func poke(tensor: NodeID, index: NodeID, channel: NodeID, value: NodeID) throws
    -> NodeID
  {
    guard let tensorId = nodeToTensor[tensor] else {
      throw DGenError.missingTensorID
    }
    guard let tensor = tensors[tensorId] else {
      throw DGenError.missingTensorID
    }

    let zero = n(.constant(0.0))
    let channelSizeFloat = n(.constant(Float(tensor.shape[0])))

    // Properly wrap the index within the channel using modulo for true wrapping
    // This handles cases where index might be very negative or very positive
    let wrappedIndex = n(.mod, index, channelSizeFloat)

    // Handle negative modulo results: if wrappedIndex < 0, add channelSize
    let isNegative = n(.lt, wrappedIndex, zero)
    let positiveIndex = n(
      .gswitch, isNegative,
      n(.add, wrappedIndex, channelSizeFloat),
      wrappedIndex)

    // Calculate channel offset: floor(channel) * channelSize
    // Clamp channel to valid range [0, numChannels-1]
    let clampedChannel = n(
      .floor,
      n(
        .max, zero,
        n(.min, channel, n(.constant(Float(tensor.shape[1] - 1))))))

    let channelOffset = n(.mul, channelSizeFloat, clampedChannel)

    // Calculate final write position within the tensor buffer
    let finalWritePos = n(.floor, n(.add, channelOffset, positiveIndex))

    let bufferBase = tensor.cellId
    return n(.memoryWrite(bufferBase), finalWritePos, value)
  }

  public func peek(tensor tensorNode: NodeID, index: NodeID, channel: NodeID) throws -> NodeID {
    // Check if input has tensor shape (works for both concrete and frame-based tensors)
    guard let inputNode = nodes[tensorNode],
      case .tensor(let originalShape) = inputNode.shape
    else {
      throw DGenError.tensorError(op: "peek", reason: "requires tensor input")
    }

    let shape = originalShape.count == 1 ? [originalShape[0], 1] : originalShape

    guard shape.count >= 2 else {
      throw DGenError.shapeMismatch(op: "peek", shape1: shape, shape2: shape)
    }

    // Always use lazy .peek node so backward pass can properly handle tensor gradients
    // The eager memoryRead path doesn't support tensor gradient accumulation
    return n(.peek, tensorNode, index, channel)
  }
}

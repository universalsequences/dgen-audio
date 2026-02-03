public struct Tensor {
  public let id: TensorID
  public let shape: Shape
  public let strides: [Int]  // How many elements to skip per dimension
  public let offset: Int  // Starting index in underlying storage (for views like shrink)
  public var cellId: CellID
  public var data: [Float]?  // Initial data to be injected by runtime
  public let isView: Bool  // True if this is a view of another tensor (reshape/transpose/shrink)
  public let padding: [(left: Int, right: Int)]?  // Per-axis padding for virtual pad views
  public let innerShapeForRepeat: [Int]?  // Original shape before repeat (for modular indexing)
  public let shrinkStart: [Int]?  // Logical start indices for shrink (used with repeat)
  public var isLazy: Bool  // True if cellId is a lazy placeholder, not yet allocated

  public init(
    id: TensorID, shape: Shape, cellId: CellID, data: [Float]? = nil, strides: [Int]? = nil,
    offset: Int = 0, isView: Bool = false, padding: [(left: Int, right: Int)]? = nil,
    innerShapeForRepeat: [Int]? = nil,
    shrinkStart: [Int]? = nil,
    isLazy: Bool = false
  ) {
    self.id = id
    self.shape = shape
    self.cellId = cellId
    self.data = data
    self.offset = offset
    self.isView = isView
    self.padding = padding
    self.innerShapeForRepeat = innerShapeForRepeat
    self.shrinkStart = shrinkStart
    self.isLazy = isLazy
    // Default to row-major (C-style) strides if not specified
    self.strides = strides ?? Tensor.computeRowMajorStrides(shape)
  }

  /// The inner (unpadded) shape - only valid when padding is set
  public var innerShape: [Int]? {
    guard let padding = padding else { return nil }
    return zip(shape, padding).map { dim, pad in
      dim - pad.left - pad.right
    }
  }

  /// Total number of elements in this tensor
  public var size: Int {
    shape.reduce(1, *)
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

  /// Check if this tensor is contiguous (strides match row-major layout)
  public var isContiguous: Bool {
    strides == Tensor.computeRowMajorStrides(shape)
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

  /// Create a tensor view with new shape/strides, sharing the same underlying data.
  /// For derived ops without tensors, creates node only - tensor created during allocation.
  private func createView(
    input: NodeID,
    op: LazyOp,
    newShape: Shape,
    offset: Int = 0,
    computeStrides: (Tensor) -> [Int]
  ) throws -> NodeID {
    // Create the node first
    let nodeId = n(op, [input], shape: .tensor(newShape))

    // If input has a concrete tensor, create the view tensor now
    if let inputTensor = nodeToTensor[input].flatMap({ tensors[$0] }) {
      let newStrides = computeStrides(inputTensor)
      let totalOffset = inputTensor.offset + offset

      let tensorId = nextTensorId
      nextTensorId += 1

      tensors[tensorId] = Tensor(
        id: tensorId,
        shape: newShape,
        cellId: inputTensor.cellId,
        strides: newStrides,
        offset: totalOffset,
        isView: true
      )

      nodeToTensor[nodeId] = tensorId
    }
    // If no concrete tensor, the view will be created during allocateTensorOutputs

    return nodeId
  }

  // MARK: - Tensor Views (Reshape/Transpose/Shrink)

  /// Reshape a tensor to a new shape (metadata only, no data movement)
  public func reshape(_ input: NodeID, to newShape: Shape) throws -> NodeID {
    /*
    let tensor = try getTensor(input)
    guard tensor.isContiguous else {
      throw DGenError.tensorError(
        op: "reshape",
        reason: "cannot reshape non-contiguous tensor - call contiguous() first")
    }
    */

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

    // If input has a concrete tensor, use its strides; otherwise assume row-major
    let inputStrides =
      nodeToTensor[input].flatMap { tensors[$0]?.strides }
      ?? Tensor.computeRowMajorStrides(inputShape)

    return try createView(input: input, op: .reshape(newShape), newShape: newShape) { _ in
      adaptStridesForReshape(
        inputShape: inputShape, inputStrides: inputStrides, newShape: newShape)
    }
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

    // If input has a concrete tensor, use its strides; otherwise assume row-major
    let inputStrides =
      nodeToTensor[input].flatMap { tensors[$0]?.strides }
      ?? Tensor.computeRowMajorStrides(inputShape)

    let newShape = perm.map { inputShape[$0] }
    let newStrides = perm.map { inputStrides[$0] }

    return try createView(input: input, op: .transpose(perm), newShape: newShape) { _ in
      newStrides
    }
  }

  /// Shrink/slice a tensor along each axis (metadata only, no data movement)
  /// ranges: for each dimension, either nil (keep all) or (start, end) tuple
  public func shrink(_ input: NodeID, ranges: [(Int, Int)?]) throws -> NodeID {
    let inputTensor = try getTensor(input)

    guard ranges.count == inputTensor.shape.count else {
      throw DGenError.tensorError(
        op: "shrink",
        reason: "ranges count \(ranges.count) must match ndim \(inputTensor.shape.count)")
    }

    var newShape = [Int]()
    var startIndices = [Int]()

    for (dim, range) in ranges.enumerated() {
      if let (start, end) = range {
        guard start >= 0 && end <= inputTensor.shape[dim] && start < end else {
          throw DGenError.tensorError(
            op: "shrink",
            reason:
              "invalid range (\(start), \(end)) for dimension \(dim) with size \(inputTensor.shape[dim])"
          )
        }
        startIndices.append(start)
        newShape.append(end - start)
      } else {
        startIndices.append(0)
        newShape.append(inputTensor.shape[dim])
      }
    }

    // If input has innerShapeForRepeat, use logical shrink (shrinkStart) instead of offset
    // This ensures modular indexing is applied correctly at read time
    if inputTensor.innerShapeForRepeat != nil {
      // Combine with any existing shrinkStart
      let combinedStart: [Int]
      if let existingStart = inputTensor.shrinkStart {
        combinedStart = zip(existingStart, startIndices).map { $0 + $1 }
      } else {
        combinedStart = startIndices
      }

      let tensorId = nextTensorId
      nextTensorId += 1

      tensors[tensorId] = Tensor(
        id: tensorId,
        shape: newShape,
        cellId: inputTensor.cellId,
        strides: inputTensor.strides,
        offset: inputTensor.offset,  // Don't change offset
        isView: true,
        innerShapeForRepeat: inputTensor.innerShapeForRepeat,  // Preserve repeat info
        shrinkStart: combinedStart  // Store logical start for read-time adjustment
      )

      let nodeId = n(.shrink(ranges), [input], shape: .tensor(newShape))
      nodeToTensor[nodeId] = tensorId
      return nodeId
    }

    // Standard case: no repeat, use offset-based shrink
    var offset = 0
    for (dim, start) in startIndices.enumerated() {
      offset += start * inputTensor.strides[dim]
    }

    return try createView(input: input, op: .shrink(ranges), newShape: newShape, offset: offset) {
      t in
      t.strides  // strides unchanged for shrink
    }
  }

  /// Pad a tensor with zeros along each axis (virtual view, no data copy)
  /// a virtual view requires conditional logic at read time (bounds check)
  /// padding: for each dimension, (left, right) padding amounts
  public func pad(_ input: NodeID, padding: [(Int, Int)]) throws -> NodeID {
    let inputTensor = try getTensor(input)

    guard padding.count == inputTensor.shape.count else {
      throw DGenError.tensorError(
        op: "pad",
        reason: "padding count \(padding.count) must match ndim \(inputTensor.shape.count)")
    }

    // Compute new padded shape
    let newShape = zip(inputTensor.shape, padding).map { dim, pad in
      dim + pad.0 + pad.1
    }

    // Create padded tensor view
    let tensorId = nextTensorId
    nextTensorId += 1

    // Strides are based on the INNER (unpadded) shape for memory access
    // The padded tensor shares the same underlying data
    tensors[tensorId] = Tensor(
      id: tensorId,
      shape: newShape,
      cellId: inputTensor.cellId,
      strides: inputTensor.strides,
      offset: inputTensor.offset,
      isView: true,
      padding: padding
    )

    let nodeId = n(.pad(padding), [input], shape: .tensor(newShape))
    nodeToTensor[nodeId] = tensorId
    return nodeId
  }

  /// Expand a tensor by broadcasting size-1 dimensions to target shape (stride=0 view, no data copy)
  /// e.g. [2, 1, 3] -> expandView to [2, 4, 3] makes dim 1 appear repeated 4 times via stride=0
  public func expandView(_ input: NodeID, to targetShape: Shape) throws -> NodeID {
    let inputTensor = try getTensor(input)

    guard inputTensor.shape.count == targetShape.count else {
      throw DGenError.tensorError(
        op: "expandView",
        reason: "shape rank mismatch: \(inputTensor.shape.count) vs \(targetShape.count)")
    }

    // Validate: each dim must be same size OR input dim must be 1
    for (i, (inDim, targetDim)) in zip(inputTensor.shape, targetShape).enumerated() {
      guard inDim == targetDim || inDim == 1 else {
        throw DGenError.tensorError(
          op: "expandView",
          reason: "dim \(i): can only expand size-1 dims, got \(inDim) -> \(targetDim)")
      }
    }

    // Compute new strides: set stride=0 for dimensions being expanded (broadcast)
    let newStrides = zip(zip(inputTensor.shape, targetShape), inputTensor.strides).map {
      dims, stride in
      dims.0 == 1 && dims.1 > 1 ? 0 : stride
    }

    let tensorId = nextTensorId
    nextTensorId += 1

    tensors[tensorId] = Tensor(
      id: tensorId,
      shape: targetShape,
      cellId: inputTensor.cellId,
      strides: newStrides,
      offset: inputTensor.offset,
      isView: true
    )

    let nodeId = n(.expandView(targetShape), [input], shape: .tensor(targetShape))
    nodeToTensor[nodeId] = tensorId
    return nodeId
  }

  /// Repeat/tile a tensor along each dimension (modular index view, no data copy)
  /// e.g. [2, 3] repeated by [2, 3] -> [4, 9] where each element appears multiple times
  /// Implemented via modular indexing: index[i] % originalShape[i]
  public func repeatView(_ input: NodeID, repeats: [Int]) throws -> NodeID {
    let inputTensor = try getTensor(input)

    guard inputTensor.shape.count == repeats.count else {
      throw DGenError.tensorError(
        op: "repeatView",
        reason: "repeats count \(repeats.count) must match ndim \(inputTensor.shape.count)")
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
    let newShape = zip(inputTensor.shape, repeats).map { $0 * $1 }

    // TODO(human): Create the repeated tensor view
    // The tensor needs:
    // - shape: newShape (the tiled shape)
    // - strides: same as input (we use modular indexing, not stride tricks)
    // - innerShapeForRepeat: inputTensor.shape (so tensorRead knows the modulo)
    // - cellId, offset: same as input (sharing memory)

    let tensorId = nextTensorId
    nextTensorId += 1

    tensors[tensorId] = Tensor(
      id: tensorId,
      shape: newShape,
      cellId: inputTensor.cellId,
      strides: inputTensor.strides,
      offset: inputTensor.offset,
      isView: true,
      innerShapeForRepeat: inputTensor.shape  // Store original shape for modular indexing
    )

    let nodeId = n(.repeatView(repeats), [input], shape: .tensor(newShape))
    nodeToTensor[nodeId] = tensorId
    return nodeId
  }

  /// Pool operation: transforms input for convolution by extracting kernel windows
  /// Input: [...batch, H, W] → Output: [...batch, oH, oW, kH, kW]
  /// Each (oH, oW) position contains its (kH, kW) kernel window
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
    guard shape.count == strides.count else {
      throw DGenError.tensorError(
        op: "asStrided",
        reason: "shape and strides must have same count: \(shape.count) vs \(strides.count)")
    }

    let inputTensor = try getTensor(input)

    let tensorId = nextTensorId
    nextTensorId += 1

    tensors[tensorId] = Tensor(
      id: tensorId,
      shape: shape,
      cellId: inputTensor.cellId,
      strides: strides,
      offset: inputTensor.offset + offset,
      isView: true
    )

    // Use reshape op as marker (the actual behavior is determined by tensor strides)
    let nodeId = n(.reshape(shape), [input], shape: .tensor(shape))
    nodeToTensor[nodeId] = tensorId
    return nodeId
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
    let inputTensor = try getTensor(input)
    let inputShape = inputTensor.shape
    let inputStrides = inputTensor.strides

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
    let batchStrides = Array(inputStrides.prefix(batchDims))
    let spatialShape = Array(inputShape.suffix(spatialDims))
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

    // Allocate output tensor
    let outputSize = outputShape.reduce(1, *)
    let outputCellId = alloc(vectorWidth: outputSize)
    let outputTensorId = nextTensorId
    nextTensorId += 1
    tensors[outputTensorId] = Tensor(
      id: outputTensorId, shape: outputShape, cellId: outputCellId)
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
    // Allocate a SEPARATE output cell for the read result
    // This is critical: the read copies from history cell to output cell
    // If we reuse the history cell, the read becomes a no-op!
    let size = buffer.shape.reduce(1, *)
    let outputCellId = alloc(vectorWidth: size)
    let outputTensorId = nextTensorId
    nextTensorId += 1
    tensors[outputTensorId] = Tensor(
      id: outputTensorId, shape: buffer.shape, cellId: outputCellId)
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

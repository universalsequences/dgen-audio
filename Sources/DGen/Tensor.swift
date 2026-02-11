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

  /// Sliding window into a flat history array, indexed by frame
  /// windowSize: number of elements visible in the window
  /// inputShape: shape of the base history array
  /// positionNode: if set, use this node's value as the write head position (circular buffer mode)
  ///              instead of currentFrameIndex(). Enables cross-block continuity for real-time streaming.
  case slidingWindow(windowSize: Int, inputShape: [Int], positionNode: NodeID? = nil)

  public static func == (lhs: ViewTransform, rhs: ViewTransform) -> Bool {
    switch (lhs, rhs) {
    case (.pad(let lp, let lis), .pad(let rp, let ris)):
      return lis == ris && lp.count == rp.count
        && zip(lp, rp).allSatisfy { $0.left == $1.left && $0.right == $1.right }
    case (
      .asStrided(let lo, let ls, let loff, let lis), .asStrided(let ro, let rs, let roff, let ris)
    ):
      return lo == ro && ls == rs && loff == roff && lis == ris
    case (.reshape(let lo, let lis), .reshape(let ro, let ris)):
      return lo == ro && lis == ris
    case (.transpose(let la, let lis), .transpose(let ra, let ris)):
      return la == ra && lis == ris
    case (.shrink(let lr, let lis), .shrink(let rr, let ris)):
      if lis != ris || lr.count != rr.count { return false }
      for (l, r) in zip(lr, rr) {
        switch (l, r) {
        case (nil, nil): continue
        case (let ls?, let rs?): if ls.start != rs.start || ls.end != rs.end { return false }
        default: return false
        }
      }
      return true
    case (.expand(let lt, let lis), .expand(let rt, let ris)):
      return lt == rt && lis == ris
    case (.repeatTile(let li, let lo), .repeatTile(let ri, let ro)):
      return li == ri && lo == ro
    case (.slidingWindow(let lw, let lis, let lp), .slidingWindow(let rw, let ris, let rp)):
      return lw == rw && lis == ris && lp == rp
    default:
      return false
    }
  }
}

/// A composed view represents a segment of consecutive composable transforms
/// collapsed into a single (shape, strides, offset) tuple.
/// This is the tinygrad ShapeTracker pattern: instead of replaying each transform
/// at IR time (generating divmod chains), we compose them at compile time.
public struct ComposedView {
  public let shape: [Int]
  public let strides: [Int]
  public let offset: Int

  /// A view is contiguous if its strides match row-major layout for its shape.
  /// This determines whether a subsequent reshape can be absorbed into this view
  /// (row-major reshape only works on contiguous data) or forces a new view boundary.
  public var isContiguous: Bool {
    let expectedContiguousStrides = Tensor.computeRowMajorStrides(shape)
    return (strides == expectedContiguousStrides)
  }
}

/// Result of composing a tensor's transform chain into ComposedView segments.
public struct ComposedTransformResult {
  public let views: [ComposedView]
  public let isFullyComposed: Bool
}

public struct Tensor {
  public let id: TensorID
  public let shape: Shape  // Final shape after all transforms
  public var cellId: CellID  // Base memory cell
  public var data: [Float]?  // Initial data to be injected by runtime

  // Transform chain from base to final view
  public let baseShape: [Int]  // Shape of actual data in memory
  public let baseStrides: [Int]  // Strides of actual data (row-major)
  public let transforms: [ViewTransform]  // Applied in order from base → final

  // Computed property for quick view check
  public var isView: Bool { !transforms.isEmpty }

  // Allocation flags
  public var isLazy: Bool  // True if cellId is a lazy placeholder (not yet allocated)
  public var materialize: Bool  // True if this tensor should be stored in memory (for realize())

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

  /// Memory offset - computed from transforms (asStrided or shrink)
  public var offset: Int {
    var totalOffset = 0
    var currentStrides = baseStrides

    for transform in transforms {
      switch transform {
      case .asStrided(_, let strides, let offset, _):
        // asStrided sets explicit offset and strides
        totalOffset = offset
        currentStrides = strides

      case .shrink(let ranges, _):
        // shrink adds offset based on start indices
        for (dim, range) in ranges.enumerated() {
          if let (start, _) = range, dim < currentStrides.count {
            totalOffset += start * currentStrides[dim]
          }
        }

      default:
        break
      }
    }
    return totalOffset
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

  /// Compose the transform chain into ComposedView segments.
  ///
  /// Walks transforms forward from base, accumulating shape/strides/offset.
  /// Compatible transforms (reshape on contiguous, transpose, shrink, expand, asStrided)
  /// are absorbed into the current view. Incompatible transforms or reshape on
  /// non-contiguous views push a view boundary.
  ///
  /// Non-composable transforms (pad, repeatTile, slidingWindow) cause an early
  /// return with isFullyComposed=false → caller falls back to the chain walker.
  public func composeTransforms() -> ComposedTransformResult {
    guard !transforms.isEmpty else {
      // No transforms: single contiguous view from base
      return ComposedTransformResult(
        views: [ComposedView(shape: baseShape, strides: baseStrides, offset: 0)],
        isFullyComposed: true
      )
    }

    var currentShape = baseShape
    var currentStrides = baseStrides
    var currentOffset = 0
    var views: [ComposedView] = []

    for transform in transforms {
      switch transform {

      case .reshape(let outputShape, _):
        let view = ComposedView(shape: currentShape, strides: currentStrides, offset: currentOffset)
        if view.isContiguous {
          // Contiguous: absorb reshape by computing new row-major strides
          currentShape = outputShape
          currentStrides = Tensor.computeRowMajorStrides(outputShape)
          // offset stays the same (contiguous → row-major reinterpretation)
        } else {
          // Non-contiguous: push current view as boundary, start fresh
          views.append(view)
          currentShape = outputShape
          currentStrides = Tensor.computeRowMajorStrides(outputShape)
          currentOffset = 0
        }

      case .transpose(let axes, _):
        // Permute shape and strides
        currentShape = axes.map { currentShape[$0] }
        currentStrides = axes.map { currentStrides[$0] }

      case .shrink(let ranges, _):
        // Add start offsets and update shape
        for (dim, range) in ranges.enumerated() {
          if let (start, end) = range {
            currentOffset += start * currentStrides[dim]
            currentShape[dim] = end - start
          }
        }

      case .expand(let targetShape, _):
        // Set strides to 0 for broadcast dims
        for i in 0..<currentShape.count {
          if currentShape[i] == 1 && targetShape[i] != 1 {
            currentStrides[i] = 0
          }
          currentShape[i] = targetShape[i]
        }

      case .asStrided(let outputShape, let strides, let offset, _):
        // Replace shape/strides/offset directly
        currentShape = outputShape
        currentStrides = strides
        currentOffset = offset

      case .pad, .repeatTile, .slidingWindow(_, _, _):
        // Non-composable: fall back to chain walker
        return ComposedTransformResult(views: [], isFullyComposed: false)
      }
    }

    // Push the final view
    views.append(ComposedView(shape: currentShape, strides: currentStrides, offset: currentOffset))
    return ComposedTransformResult(views: views, isFullyComposed: true)
  }
}

/// Inject tensor data from graph into runtime memory buffer
/// Call this after allocating memory but before running the kernel
public func injectTensorData(
  graph: Graph,
  cellAllocations: CellAllocations,
  memory: UnsafeMutablePointer<Float>,
  verbose: Bool = false
) {
  var maxIdx = 0
  var tensorCount = 0
  for (_, tensor) in graph.tensors {
    guard let data = tensor.data else { continue }

    // Get physical memory offset from cell mapping
    let physicalOffset = cellAllocations.cellMappings[tensor.cellId] ?? tensor.cellId

    // Copy data into memory
    for (i, value) in data.enumerated() {
      if i < tensor.size {
        memory[physicalOffset + i] = value
        maxIdx = max(maxIdx, physicalOffset + i)
      }
    }
    tensorCount += 1
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

/// Collect tensor initial data as (physicalOffset, values) pairs.
/// Used by the Engine to inject DGen-internal tensor data (e.g. twiddle factors)
/// through the AudioGraph param initialization path.
/// IMPORTANT: Returns physical memory offsets (after cell remapping), not raw cellIds.
public func collectTensorInitData(graph: Graph, cellAllocations: CellAllocations) -> [(
  Int, [Float]
)] {
  var result: [(Int, [Float])] = []
  for (_, tensor) in graph.tensors {
    guard let data = tensor.data else { continue }
    let physicalOffset = cellAllocations.cellMappings[tensor.cellId] ?? tensor.cellId
    result.append((physicalOffset, Array(data.prefix(tensor.size))))
  }
  return result
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

  // MARK: - Tensor View Operations
  //
  // View operations create new "views" of existing tensors without copying data.
  // They work by recording transforms that map output indices to base memory indices.
  // Views can be composed: reshape(pad(x)) creates a chain of transforms.

  /// Reshape a tensor to a new shape without copying data.
  ///
  /// The total number of elements must remain the same. Data is reinterpreted
  /// in row-major (C-style) order.
  ///
  /// ```swift
  /// // Flatten a 2D tensor to 1D
  /// let flat = try g.reshape(matrix, to: [rows * cols])
  ///
  /// // Reshape 1D to 2D
  /// let matrix = try g.reshape(vector, to: [4, 4])
  ///
  /// // Add a dimension (useful for broadcasting)
  /// let col = try g.reshape(vector, to: [n, 1])
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor
  ///   - newShape: Target shape (must have same total element count)
  /// - Returns: A view of the tensor with the new shape
  /// - Throws: If element counts don't match
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
    return try createViewWithTransform(
      input: input, op: .reshape(newShape), newShape: newShape, transform: transform)
  }

  /// Transpose a tensor by permuting its axes without copying data.
  ///
  /// Reorders the dimensions of a tensor. For 2D tensors (matrices), this swaps
  /// rows and columns. For higher dimensions, you specify which axis goes where.
  ///
  /// ```swift
  /// // 2D matrix transpose (swap rows and columns)
  /// let transposed = try g.transpose(matrix)  // [M, N] → [N, M]
  ///
  /// // Explicit axes for 2D (equivalent to above)
  /// let transposed = try g.transpose(matrix, axes: [1, 0])
  ///
  /// // 3D tensor: move last axis to front
  /// let reordered = try g.transpose(tensor3d, axes: [2, 0, 1])  // [A, B, C] → [C, A, B]
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor
  ///   - axes: Permutation of dimension indices. If nil, reverses all axes (standard transpose).
  ///           `axes[i]` specifies which input dimension becomes output dimension `i`.
  /// - Returns: A view of the tensor with reordered dimensions
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
    return try createViewWithTransform(
      input: input, op: .transpose(perm), newShape: newShape, transform: transform)
  }

  /// Slice a tensor along each axis without copying data.
  ///
  /// Extracts a contiguous sub-region of a tensor. Similar to Python's `tensor[start:end]`
  /// but generalized to multiple dimensions.
  ///
  /// ```swift
  /// // Extract rows 2-5 from a matrix (keeping all columns)
  /// let slice = try g.shrink(matrix, ranges: [(2, 5), nil])
  ///
  /// // Extract a 3x3 patch from position (10, 20)
  /// let patch = try g.shrink(image, ranges: [(10, 13), (20, 23)])
  ///
  /// // Keep first dimension, slice second
  /// let partial = try g.shrink(tensor, ranges: [nil, (0, 5)])
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor
  ///   - ranges: For each dimension, either:
  ///     - `nil`: Keep all elements in this dimension
  ///     - `(start, end)`: Keep elements from index `start` up to (but not including) `end`
  /// - Returns: A view of the sliced region
  /// - Throws: If ranges are out of bounds or start >= end
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
            reason:
              "invalid range (\(start), \(end)) for dimension \(dim) with size \(inputShape[dim])"
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
    return try createViewWithTransform(
      input: input, op: .shrink(ranges), newShape: newShape, transform: transform)
  }

  /// Pad a tensor with zeros along each axis without copying data.
  ///
  /// Adds virtual zero-padding around the tensor. The padded regions return 0 when
  /// read; the original data is not copied. This is essential for convolutions
  /// where you need to handle boundary conditions.
  ///
  /// ```swift
  /// // Pad a 1D signal with 2 zeros on each side
  /// let padded = try g.pad(signal, padding: [(2, 2)])  // [N] → [N+4]
  ///
  /// // Pad a 2D image: 1 pixel on top/bottom, 2 on left/right
  /// let padded = try g.pad(image, padding: [(1, 1), (2, 2)])  // [H, W] → [H+2, W+4]
  ///
  /// // Asymmetric padding (more on one side)
  /// let padded = try g.pad(tensor, padding: [(0, 3), (1, 0)])
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor
  ///   - padding: For each dimension, `(left, right)` specifying how many zeros to add
  ///              on each side. `left` is added before index 0, `right` after the last element.
  /// - Returns: A view of the padded tensor (reads in padded region return 0)
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
    return try createViewWithTransform(
      input: input, op: .pad(padding), newShape: newShape, transform: transform)
  }

  /// Broadcast a tensor by expanding size-1 dimensions without copying data.
  ///
  /// Expands dimensions that have size 1 to a larger size. The single value in that
  /// dimension is "broadcast" (repeated) to fill the new size. This is how numpy/PyTorch
  /// broadcasting works under the hood.
  ///
  /// ```swift
  /// // Broadcast a column vector to a matrix (for row-wise operations)
  /// // [N, 1] → [N, M]  (the single column is repeated M times)
  /// let expanded = try g.expandView(column, to: [n, m])
  ///
  /// // Broadcast a row vector to a matrix
  /// // [1, M] → [N, M]  (the single row is repeated N times)
  /// let expanded = try g.expandView(row, to: [n, m])
  ///
  /// // Broadcast a scalar-like tensor
  /// // [1, 1, 1] → [A, B, C]
  /// let expanded = try g.expandView(scalar3d, to: [a, b, c])
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor (must have size 1 in any dimension being expanded)
  ///   - targetShape: Target shape (must have same rank as input)
  /// - Returns: A view where size-1 dims appear expanded (reads return the same value)
  /// - Throws: If trying to expand a dimension that isn't size 1
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
    return try createViewWithTransform(
      input: input, op: .expandView(targetShape), newShape: targetShape, transform: transform)
  }

  /// Tile a tensor by repeating it along each dimension without copying data.
  ///
  /// Creates a larger tensor by conceptually tiling the input. Implemented via
  /// modular indexing: `output[i] = input[i % inputSize]`. Similar to `numpy.tile()`.
  ///
  /// ```swift
  /// // Repeat a [2, 3] tensor 2x along dim 0, 3x along dim 1 → [4, 9]
  /// let tiled = try g.repeatView(tensor, repeats: [2, 3])
  /// // Element at [3, 7] reads from [3 % 2, 7 % 3] = [1, 1]
  ///
  /// // Repeat only along one dimension
  /// let repeated = try g.repeatView(row, repeats: [4, 1])  // [1, N] → [4, N]
  ///
  /// // Create a checkerboard pattern from a 2x2 tile
  /// let board = try g.repeatView(tile2x2, repeats: [4, 4])  // [2, 2] → [8, 8]
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor to tile
  ///   - repeats: How many times to repeat along each dimension (must be >= 1)
  /// - Returns: A view of the tiled tensor (reads use modular indexing)
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
    return try createViewWithTransform(
      input: input, op: .repeatView(repeats), newShape: newShape, transform: transform)
  }

  /// Create a strided view with arbitrary shape and strides (advanced, low-level).
  ///
  /// This is the fundamental building block for sliding window operations like pooling
  /// and convolution. It allows you to create views where moving along each dimension
  /// skips a custom number of elements.
  ///
  /// **Warning**: This is a low-level operation. Incorrect strides can read out-of-bounds
  /// memory. Prefer higher-level operations like `pool()` when possible.
  ///
  /// ```swift
  /// // Extract 2x2 non-overlapping windows from a 4x4 image
  /// // Input [4, 4] → Output [2, 2, 2, 2] (2x2 grid of 2x2 windows)
  /// let windows = try g.asStrided(image,
  ///     shape: [2, 2, 2, 2],      // [outH, outW, kH, kW]
  ///     strides: [8, 2, 4, 1])    // [outH*W*stride, stride, W, 1]
  ///
  /// // Sliding window with overlap (stride < kernel)
  /// // Moving 1 element at a time instead of kernel-size
  /// ```
  ///
  /// - Parameters:
  ///   - input: Source tensor
  ///   - shape: Output shape for the view
  ///   - strides: Elements to skip when moving along each dimension.
  ///              `strides[i]` = how many elements to skip when incrementing index `i` by 1.
  ///   - offset: Starting offset from the tensor's base (default 0)
  /// - Returns: A strided view of the tensor
  /// - Throws: If shape and strides counts don't match
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

    let transform = ViewTransform.asStrided(
      outputShape: shape, strides: strides, offset: offset, inputShape: inputShape)
    return try createViewWithTransform(
      input: input, op: .asStrided(shape, strides), newShape: shape, transform: transform)
  }

  /// Extract sliding windows from a tensor as extra dimensions (im2col via views).
  ///
  /// Transforms a tensor to expose sliding kernel windows as additional dimensions,
  /// enabling convolutions and pooling via element-wise operations and reductions.
  /// This is the "im2col" transformation, but done as a zero-copy view operation.
  ///
  /// ```swift
  /// // 2D pooling: [H, W] → [outH, outW, kH, kW]
  /// // Each (outH, outW) position contains its (kH, kW) window
  /// let windows = try g.pool(image, kernelSize: [3, 3])
  ///
  /// // With stride (non-overlapping windows)
  /// let windows = try g.pool(image, kernelSize: [2, 2], stride: [2, 2])
  ///
  /// // 1D pooling: [N] → [outN, K]
  /// let windows = try g.pool(signal, kernelSize: [5])
  ///
  /// // Max pooling = pool + reduce max over kernel dims
  /// // Avg pooling = pool + reduce mean over kernel dims
  /// // Conv = pool + broadcast multiply kernel + sum over kernel dims
  /// ```
  ///
  /// **Output shape**: `[...batch, out₀, out₁, ..., k₀, k₁, ...]` where:
  /// - `outᵢ = (inputᵢ - kernelᵢ) / strideᵢ + 1`
  /// - Batch dimensions are preserved at the front
  /// - Kernel dimensions are appended at the end
  ///
  /// - Parameters:
  ///   - input: Source tensor with shape `[...batch, spatial...]`
  ///   - kernelSize: Size of the sliding window in each spatial dimension
  ///   - stride: Step size for the window (default = kernelSize for non-overlapping)
  /// - Returns: A view exposing all windows as extra dimensions
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

  /// 2D Convolution implemented via view operations (zero-copy im2col approach).
  ///
  /// Performs 2D convolution by extracting sliding windows as views, then using
  /// broadcasting and reduction. The input is never copied—only the final output
  /// is materialized.
  ///
  /// ```swift
  /// // Convolve an image with an edge detection kernel
  /// let edges = try g.conv2dView(image, kernel: sobelKernel)
  ///
  /// // With custom stride (downsampling)
  /// let downsampled = try g.conv2dView(image, kernel: kernel, stride: [2, 2])
  /// ```
  ///
  /// **How it works internally**:
  /// 1. `pool(input)` extracts sliding windows: `[H, W]` → `[oH, oW, kH, kW]`
  /// 2. Kernel is broadcast: `[kH, kW]` → `[oH, oW, kH, kW]`
  /// 3. Element-wise multiply windows with kernel
  /// 4. Sum over kernel dimensions → `[oH, oW]`
  ///
  /// **Output shape**: `[(H - kH) / stride + 1, (W - kW) / stride + 1]`
  ///
  /// - Parameters:
  ///   - input: Input tensor of shape `[H, W]`
  ///   - kernel: Convolution kernel of shape `[kH, kW]`
  ///   - stride: Step size for the convolution window (default `[1, 1]`)
  /// - Returns: Convolved output tensor of shape `[oH, oW]`
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

  /// Sum a tensor along a specific axis, removing that dimension from the result.
  ///
  /// Reduces one dimension by summing all elements along it. The output has one
  /// fewer dimension than the input.
  ///
  /// ```swift
  /// // Sum columns (reduce last dimension)
  /// let rowSums = try g.sum(matrix, axis: -1)    // [M, N] → [M]
  ///
  /// // Sum rows (reduce first dimension)
  /// let colSums = try g.sum(matrix, axis: 0)     // [M, N] → [N]
  ///
  /// // Sum over middle dimension of 3D tensor
  /// let reduced = try g.sum(tensor3d, axis: 1)   // [A, B, C] → [A, C]
  /// ```
  ///
  /// **Negative axis indexing**: Negative values count from the end.
  /// - `axis: -1` = last dimension
  /// - `axis: -2` = second-to-last dimension
  /// - This matches NumPy/PyTorch conventions
  ///
  /// For a tensor of shape `[A, B, C, D]`:
  /// - `axis: 0` or `axis: -4` → reduces dim A → `[B, C, D]`
  /// - `axis: 1` or `axis: -3` → reduces dim B → `[A, C, D]`
  /// - `axis: 2` or `axis: -2` → reduces dim C → `[A, B, D]`
  /// - `axis: 3` or `axis: -1` → reduces dim D → `[A, B, C]`
  ///
  /// - Parameters:
  ///   - input: Source tensor
  ///   - axis: Which dimension to sum over. Negative values count from the end.
  /// - Returns: Tensor with one fewer dimension (or scalar if input was 1D)
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

  /// Reduce along an axis keeping maximum values.
  /// Same structure as `sum(_:axis:)` but uses `.maxAxis`.
  public func max(_ input: NodeID, axis: Int) throws -> NodeID {
    guard let inputNode = nodes[input] else {
      throw DGenError.tensorError(op: "max", reason: "input node not found")
    }
    guard case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(
        op: "max",
        reason: "requires tensor input, got \(String(describing: inputNode.shape))")
    }

    let ndim = inputShape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.tensorError(
        op: "max", reason: "axis \(axis) out of range for tensor with \(ndim) dimensions")
    }

    var outputShape = inputShape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      // TODO: full max reduce (not needed yet)
      throw DGenError.tensorError(op: "max", reason: "reducing to scalar not yet supported")
    }

    let outputCellId = reserveLazyCellId()
    let outputTensorId = nextTensorId
    nextTensorId += 1
    tensors[outputTensorId] = Tensor(
      id: outputTensorId, shape: outputShape, cellId: outputCellId, isLazy: true)
    cellToTensor[outputCellId] = outputTensorId

    let nodeId = n(.maxAxis(normalizedAxis), [input], shape: .tensor(outputShape))
    nodeToTensor[nodeId] = outputTensorId
    return nodeId
  }

  /// Reduce along an axis computing the mean.
  /// Same structure as `sum(_:axis:)` but uses `.meanAxis`.
  public func mean(_ input: NodeID, axis: Int) throws -> NodeID {
    guard let inputNode = nodes[input] else {
      throw DGenError.tensorError(op: "mean", reason: "input node not found")
    }
    guard case .tensor(let inputShape) = inputNode.shape else {
      throw DGenError.tensorError(
        op: "mean",
        reason: "requires tensor input, got \(String(describing: inputNode.shape))")
    }

    let ndim = inputShape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.tensorError(
        op: "mean", reason: "axis \(axis) out of range for tensor with \(ndim) dimensions")
    }

    var outputShape = inputShape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      // TODO: full mean reduce (not needed yet)
      throw DGenError.tensorError(op: "mean", reason: "reducing to scalar not yet supported")
    }

    let outputCellId = reserveLazyCellId()
    let outputTensorId = nextTensorId
    nextTensorId += 1
    tensors[outputTensorId] = Tensor(
      id: outputTensorId, shape: outputShape, cellId: outputCellId, isLazy: true)
    cellToTensor[outputCellId] = outputTensorId

    let nodeId = n(.meanAxis(normalizedAxis), [input], shape: .tensor(outputShape))
    nodeToTensor[nodeId] = outputTensorId
    return nodeId
  }

  /// Matrix multiplication: `A[M, K] @ B[K, N] → C[M, N]`
  ///
  /// Performs standard matrix multiplication. The inner dimensions must match:
  /// A's last dimension (K) must equal B's first dimension (K).
  ///
  /// ```swift
  /// // Multiply two matrices
  /// let C = try g.matmul(A, B)  // [M, K] @ [K, N] → [M, N]
  ///
  /// // Linear layer: output = input @ weights
  /// let output = try g.matmul(input, weights)  // [batch, in] @ [in, out] → [batch, out]
  /// ```
  ///
  /// **Implementation**: Uses reshape + broadcast multiply + sum (no explicit GEMM kernel).
  /// This leverages the existing view operations and is differentiable.
  ///
  /// - Parameters:
  ///   - a: Left matrix of shape `[M, K]`
  ///   - b: Right matrix of shape `[K, N]`
  /// - Returns: Result matrix of shape `[M, N]`
  /// - Throws: If inputs aren't 2D or inner dimensions don't match
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

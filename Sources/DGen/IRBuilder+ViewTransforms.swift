/// IRBuilder extension for handling ViewTransform chains.
/// Walks transforms backwards to map output indices to base memory indices.

extension IRBuilder {

  // MARK: - Tensor Read (handles transform chain)

  /// Read from tensor by walking the transform chain backwards.
  /// Maps output indices to base memory indices, handling padding, striding, etc.
  /// For frame-aware tensors: uses frame-indexed addressing.
  public func tensorRead(_ tensor: Tensor, indices: [Expr]) -> Expr {
    // If no transforms, fast path: direct read from base memory
    if tensor.transforms.isEmpty {
      let memIdx = stridedIndex(indices: indices, strides: tensor.baseStrides)
      return tensorReadFromBase(tensor, elemIdx: memIdx)
    }

    // Walk transforms BACKWARDS to map output indices → base indices
    var currentIndices = indices
    var currentShape = tensor.shape
    var inBoundsCheck: Expr? = nil  // Accumulate bounds checks from padding

    for transform in tensor.transforms.reversed() {
      (currentIndices, currentShape, inBoundsCheck) = applyTransformBackward(
        transform: transform,
        indices: currentIndices,
        shape: currentShape,
        inBoundsCheck: inBoundsCheck
      )
    }

    // Now currentIndices are in base tensor coordinates
    let memOffset = stridedIndex(indices: currentIndices, strides: tensor.baseStrides)

    // Apply bounds check if any padding was encountered
    if let bounds = inBoundsCheck {
      let val = float(0.0)
      self.if(bounds) {
        mutate(val.value, to: tensorReadFromBase(tensor, elemIdx: memOffset))
      }
      return val.value
    }

    return tensorReadFromBase(tensor, elemIdx: memOffset)
  }

  /// Read from tensor using flat index with transform chain support.
  /// For frame-aware cells, uses frame-indexed addressing.
  public func tensorRead(_ tensor: Tensor, flatIdx: Expr, shape: [Int]) -> Expr {
    if !tensor.transforms.isEmpty {
      // Need multi-dim indices for transform chain
      let indices = flatToMultiIndex(flatIdx, shape)
      return tensorRead(tensor, indices: indices)
    } else if ctx.frameAwareTensorCells.contains(tensor.cellId),
      let (tensorSize, _) = ctx.g.frameAwareCells[tensor.cellId]
    {
      // Frame-aware tensor: use frame-indexed addressing
      return frameAwareTensorRead(cellId: tensor.cellId, tensorSize: tensorSize, elemIdx: flatIdx)
    } else {
      // Non-transformed: use fast path
      let memIdx = tensorMemoryIndex(tensor, flatIdx: flatIdx, shape: shape)
      return memoryRead(tensor.cellId, memIdx)
    }
  }

  // MARK: - Transform Backward Application

  /// Apply a single transform backwards, mapping output indices to input indices.
  /// Returns updated indices, shape, and accumulated bounds check.
  private func applyTransformBackward(
    transform: ViewTransform,
    indices: [Expr],
    shape: [Int],
    inBoundsCheck: Expr?
  ) -> (indices: [Expr], shape: [Int], inBoundsCheck: Expr?) {
    switch transform {
    case .pad(let padding, let inputShape):
      let (newIndices, boundsCheck) = applyPadBackward(
        indices: indices,
        padding: padding,
        paddedShape: shape
      )
      return (newIndices, inputShape, combineBoundsChecks(inBoundsCheck, boundsCheck))

    case .asStrided(_, let strides, let offset, let inputShape):
      // Convert strided indices to flat offset, then to input indices
      let flatOffset = stridedIndex(indices: indices, strides: strides, offset: offset)
      let newIndices = flatToMultiIndex(flatOffset, inputShape)
      return (newIndices, inputShape, inBoundsCheck)

    case .reshape(_, let inputShape):
      // Linearize then un-linearize
      let flat = multiIndexToFlat(indices, shape)
      let newIndices = flatToMultiIndex(flat, inputShape)
      return (newIndices, inputShape, inBoundsCheck)

    case .transpose(let axes, let inputShape):
      // Inverse permutation
      let inverseAxes = invertPermutation(axes)
      let newIndices = permuteIndices(indices, inverseAxes)
      return (newIndices, inputShape, inBoundsCheck)

    case .shrink(let ranges, let inputShape):
      // Add start offsets
      let newIndices = applyShrinkBackward(indices, ranges)
      return (newIndices, inputShape, inBoundsCheck)

    case .expand(_, let inputShape):
      // Clamp broadcast dims to 0
      let newIndices = applyExpandBackward(indices, inputShape)
      return (newIndices, inputShape, inBoundsCheck)

    case .repeatTile(let innerShape, _):
      // Modular indexing
      let newIndices = indices.enumerated().map { i, idx in
        mod(idx, constant(Float(innerShape[i])))
      }
      return (newIndices, innerShape, inBoundsCheck)

    case .circularOffset(let offsetCellId, let bufferSize, let inputShape):
      // Dynamic circular buffer offset: index i → buffer[(writePos + 1 + i) % bufferSize]
      let offset = load(offsetCellId, nil)
      let one = constant(1.0)
      let size = constant(Float(bufferSize))
      var newIndices = indices
      let lastDim = newIndices.count - 1
      let shifted = newIndices[lastDim] + offset + one
      let q = floor(shifted / size)
      newIndices[lastDim] = shifted - q * size
      return (newIndices, inputShape, inBoundsCheck)
    }
  }

  // MARK: - Transform Helpers

  /// Read from base tensor memory, handling frame-aware tensors
  private func tensorReadFromBase(_ tensor: Tensor, elemIdx: Expr) -> Expr {
    if ctx.frameAwareTensorCells.contains(tensor.cellId),
      let (tensorSize, _) = ctx.g.frameAwareCells[tensor.cellId]
    {
      return frameAwareTensorRead(cellId: tensor.cellId, tensorSize: tensorSize, elemIdx: elemIdx)
    } else {
      return memoryRead(tensor.cellId, cast(elemIdx, to: .int))
    }
  }

  /// Apply pad transform backwards: output indices → input indices
  /// Returns adjusted indices and a bounds check expression
  private func applyPadBackward(
    indices: [Expr],
    padding: [(left: Int, right: Int)],
    paddedShape: [Int]
  ) -> (indices: [Expr], inBounds: Expr) {
    var newIndices: [Expr] = []
    var boundsCheck: Expr = constant(1.0)

    for (i, (idx, pad)) in zip(indices, padding).enumerated() {
      let innerSize = paddedShape[i] - pad.left - pad.right

      // Check: idx >= pad.left AND idx < pad.left + innerSize
      let leftCheck = idx >= constant(Float(pad.left))
      let rightCheck = idx < constant(Float(pad.left + innerSize))
      boundsCheck = boundsCheck * leftCheck * rightCheck

      // Adjust index: subtract left padding
      newIndices.append(idx - constant(Float(pad.left)))
    }

    return (newIndices, boundsCheck)
  }

  /// Combine two optional bounds checks with AND
  private func combineBoundsChecks(_ a: Expr?, _ b: Expr) -> Expr {
    if let existing = a {
      return existing * b
    }
    return b
  }

  /// Invert a permutation: inverseAxes[axes[i]] = i
  private func invertPermutation(_ axes: [Int]) -> [Int] {
    var inverse = [Int](repeating: 0, count: axes.count)
    for (i, axis) in axes.enumerated() {
      inverse[axis] = i
    }
    return inverse
  }

  /// Permute indices according to axes
  private func permuteIndices(_ indices: [Expr], _ axes: [Int]) -> [Expr] {
    return axes.map { indices[$0] }
  }

  /// Apply shrink transform backwards: add start offsets
  private func applyShrinkBackward(_ indices: [Expr], _ ranges: [(start: Int, end: Int)?]) -> [Expr]
  {
    return indices.enumerated().map { i, idx in
      if let range = ranges[i] {
        return idx + constant(Float(range.start))
      }
      return idx
    }
  }

  /// Apply expand transform backwards: clamp broadcast dims to 0
  private func applyExpandBackward(_ indices: [Expr], _ inputShape: [Int]) -> [Expr] {
    return indices.enumerated().map { i, idx in
      inputShape[i] == 1 ? constant(0.0) : idx
    }
  }
}

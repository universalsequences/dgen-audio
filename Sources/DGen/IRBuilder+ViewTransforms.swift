/// IRBuilder extension for handling ViewTransform chains.
/// Walks transforms backwards to map output indices to base memory indices.

extension IRBuilder {

  // MARK: - Tensor Read (handles transform chain)

  /// Read from tensor by walking the transform chain backwards.
  /// Maps output indices to base memory indices, handling padding, striding, etc.
  /// For frame-aware tensors: uses frame-indexed addressing.
  ///
  /// Tries composed path first: if all transforms compose into (shape, strides, offset)
  /// views, emits a single stridedIndex per view instead of divmod chains.
  public func tensorRead(_ tensor: Tensor, indices: [Expr]) -> Expr {
    // If no transforms, fast path: direct read from base memory
    if tensor.transforms.isEmpty {
      let memIdx = stridedIndex(indices: indices, strides: tensor.baseStrides)
      return tensorReadFromBase(tensor, elemIdx: memIdx)
    }

    // Try composed path: collapse transform chain into stride tuples
    let composed = tensor.composeTransforms()
    if composed.isFullyComposed && !composed.views.isEmpty {
      if composed.views.count == 1 {
        return tensorReadSingleView(tensor, indices: indices, view: composed.views[0])
      }
      return tensorReadComposed(tensor, indices: indices, views: composed.views)
    }

    // Fall back: walk transforms BACKWARDS to map output indices → base indices
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
  ///
  /// Tries composed path: if the outermost view is contiguous with offset 0,
  /// the flat index maps directly through without flatToMultiIndex.
  public func tensorRead(_ tensor: Tensor, flatIdx: Expr, shape: [Int]) -> Expr {
    if !tensor.transforms.isEmpty {
      // Try composed path to avoid unnecessary flatToMultiIndex
      let composed = tensor.composeTransforms()
      if composed.isFullyComposed && !composed.views.isEmpty {
        let outerView = composed.views.last!
        if composed.views.count == 1 && outerView.isContiguous && outerView.offset == 0 {
          // Single contiguous view: flat index maps directly to base memory
          return tensorReadFromBase(tensor, elemIdx: flatIdx)
        }
        // Multi-view or non-contiguous: decompose flat index into outer view indices
        let indices = flatToMultiIndex(flatIdx, shape)
        if composed.views.count == 1 {
          return tensorReadSingleView(tensor, indices: indices, view: outerView)
        }
        return tensorReadComposed(tensor, indices: indices, views: composed.views)
      }
      // Fall back: need multi-dim indices for transform chain
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

  // MARK: - Composed View Read Paths

  /// Fast path for a single composed view: one stridedIndex call → base memory read.
  /// Squeezes out size-1 dims to avoid redundant IR ops.
  private func tensorReadSingleView(_ tensor: Tensor, indices: [Expr], view: ComposedView) -> Expr {
    let (sqIdx, sqStr) = squeezeView(indices: indices, view: view)
    let memIdx = stridedIndex(indices: sqIdx, strides: sqStr, offset: view.offset)
    return tensorReadFromBase(tensor, elemIdx: memIdx)
  }

  /// Multi-view composed path: walk from outermost view to innermost.
  /// Each view boundary requires: stridedIndex → flatToMultiIndex → next view's stridedIndex.
  /// Squeezes size-1 dims at each boundary to minimize divmod chains in generated IR.
  ///
  /// Views are ordered innermost-first (index 0 = closest to base memory).
  /// We process from the last view (outermost, matching the tensor's final shape) inward.
  private func tensorReadComposed(
    _ tensor: Tensor, indices: [Expr], views: [ComposedView]
  ) -> Expr {
    // Fast path: 2-view chain where the outer view is contiguous (row-major strides, zero offset)
    // and shapes match. The outer stridedIndex→flatToMultiIndex round-trip is a no-op — skip it
    // and pass indices directly to the inner view's stridedIndex.
    let outerView = views.last!
    let outerRowMajor = Tensor.computeRowMajorStrides(outerView.shape)
    if views.count == 2 && outerView.strides == outerRowMajor && outerView.offset == 0
      && outerView.shape == views[0].shape
    {
      let (sqIdx, sqStr) = squeezeView(indices: indices, view: views[0])
      let memIdx = stridedIndex(indices: sqIdx, strides: sqStr, offset: views[0].offset)
      return tensorReadFromBase(tensor, elemIdx: memIdx)
    }

    var currentIndices = indices

    // Walk from outermost view (last) down to innermost (first)
    for i in stride(from: views.count - 1, through: 0, by: -1) {
      let view = views[i]

      // Squeeze size-1 dims to avoid generating divmod for dead dimensions
      let (sqIdx, sqStr) = squeezeView(indices: currentIndices, view: view)
      let flatIdx = stridedIndex(indices: sqIdx, strides: sqStr, offset: view.offset)

      if i == 0 {
        // Innermost view: flat index maps directly to base memory
        return tensorReadFromBase(tensor, elemIdx: flatIdx)
      }

      // Decompose flat index into the next inner view's full shape.
      // Must use unsqueezed shape so indices stay positionally aligned with view dims
      // (squeezeView picks indices[i] where i is the dim position).
      let innerView = views[i - 1]
      currentIndices = flatToMultiIndex(flatIdx, innerView.shape)
    }

    // Should not reach here (loop always returns at i==0)
    let memIdx = stridedIndex(
      indices: currentIndices, strides: views[0].strides, offset: views[0].offset)
    return tensorReadFromBase(tensor, elemIdx: memIdx)
  }

  /// Remove size-1 dimensions from indices and strides to avoid useless IR.
  /// Size-1 dims always index at 0, contributing nothing to the strided offset.
  private func squeezeView(indices: [Expr], view: ComposedView) -> (indices: [Expr], strides: [Int])
  {
    var sqIdx: [Expr] = []
    var sqStr: [Int] = []
    for i in 0..<view.shape.count where view.shape[i] != 1 {
      sqIdx.append(i < indices.count ? indices[i] : intConstant(0))
      sqStr.append(view.strides[i])
    }
    if sqIdx.isEmpty {
      return ([intConstant(0)], [0])
    }
    return (sqIdx, sqStr)
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
        idx % intConstant(innerShape[i])
      }
      return (newIndices, innerShape, inBoundsCheck)

    case .slidingWindow(let windowSize, let inputShape, let positionNode):
      let wSize = intConstant(windowSize)
      let one = intConstant(1)
      var newIndices = indices
      let lastDim = newIndices.count - 1

      if let posNode = positionNode {
        // Circular buffer mode: position comes from accum node (persists across runs)
        guard let posLazy = ctx.values[posNode] else {
          fatalError("slidingWindow positionNode \(posNode) not available in ctx.values")
        }
        let pos = cast(value(posLazy, scalarType: .float), to: .int)
        let bufSize = intConstant(inputShape[lastDim])
        // Window element j maps to (pos - windowSize + 1 + j) mod bufferSize
        // Since pos >= 0 and windowSize <= bufferSize, (raw + bufSize) >= 1, so single add handles negative
        let raw = pos - wSize + one + newIndices[lastDim]
        let baseIdx = (raw + bufSize) % bufSize
        newIndices[lastDim] = baseIdx
        return (newIndices, inputShape, inBoundsCheck)
      } else {
        // Original mode: window indexed by frame position
        let fi = currentFrameIndex()
        let baseIdx = fi - wSize + one + newIndices[lastDim]
        newIndices[lastDim] = baseIdx
        // Early frames: out-of-bounds → 0 (reuse padding bounds-check infrastructure)
        let boundsCheck = baseIdx >= intConstant(0)
        return (newIndices, inputShape, combineBoundsChecks(inBoundsCheck, boundsCheck))
      }
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
      let leftCheck = idx >= intConstant(pad.left)
      let rightCheck = idx < intConstant(pad.left + innerSize)
      boundsCheck = boundsCheck * leftCheck * rightCheck

      // Adjust index: subtract left padding
      newIndices.append(idx - intConstant(pad.left))
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
        return idx + intConstant(range.start)
      }
      return idx
    }
  }

  /// Apply expand transform backwards: clamp broadcast dims to 0
  private func applyExpandBackward(_ indices: [Expr], _ inputShape: [Int]) -> [Expr] {
    return indices.enumerated().map { i, idx in
      inputShape[i] == 1 ? intConstant(0) : idx
    }
  }
}

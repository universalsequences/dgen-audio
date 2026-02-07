/// IRBuilder extension for tensor indexing and high-level tensor I/O.
/// High-level: readInput, writeOutput, broadcastIndices.
/// Indexing: stridedIndex, tensorMemoryIndex, flatToMultiIndex, multiIndexToFlat, broadcastIndex.

extension IRBuilder {

  // MARK: - High-level tensor I/O

  /// Read a tensor or scalar input for a given node.
  /// For scalars, returns the lazy value directly.
  /// For tensors with view transforms (reshape, transpose, etc.), walks the transform chain
  /// via `tensorRead` in IRBuilder+ViewTransforms. For plain tensors, computes a broadcast-aware
  /// memory offset and loads via `tload`.
  public func readInput(_ node: Node, _ inputs: [Lazy], at idx: Int) throws -> Expr {
    let inputId = node.inputs[idx]
    guard let inputNode = ctx.g.nodes[inputId] else { throw DGenError.missingTensorID }

    // scalar
    if case .scalar = inputNode.shape ?? .scalar {
      return value(inputs[idx])
    }

    // tensor
    guard case .tensor(let outShape) = node.shape,
      let tensor = ctx.g.nodeToTensor[inputId].flatMap({ ctx.g.tensors[$0] }),
      let loopIdx = ctx.tensorIndices[node.id]
    else {
      throw DGenError.missingTensorID
    }

    // For tensors with transforms, use tensorRead which handles the transform chain
    if !tensor.transforms.isEmpty {
      let indices = flatToMultiIndex(value(loopIdx, scalarType: .int), outShape)
      let broadcastedIndices = broadcastIndices(
        outputIndices: indices, outputShape: outShape, inputTensor: tensor)
      return tensorRead(tensor, indices: broadcastedIndices)
    }

    let memOffset = broadcastIndex(
      outputIdx: value(loopIdx, scalarType: .int), outputShape: outShape,
      inputTensor: tensor
    )
    return tload(tensor.cellId, memOffset)
  }

  /// Adjust multi-dimensional indices for broadcasting between output and input shapes.
  /// Right-aligns the shapes and clamps broadcast dimensions (size 1) to index 0.
  func broadcastIndices(outputIndices: [Expr], outputShape: [Int], inputTensor: Tensor) -> [Expr] {
    let inputShape = inputTensor.shape
    let rankDiff = outputShape.count - inputShape.count

    // right-align shapes, clamp broadcast dims to 0
    return inputShape.enumerated().map { i, dim in
      dim == 1 ? constant(0) : outputIndices[i + rankDiff]
    }
  }

  /// Write a computed result to the node's output tensor.
  /// Registers the value in the context via `use(val:)` so downstream nodes can reference it.
  /// For tensor nodes, also stores the value to the tensor cell via `tstore`.
  /// For scalar nodes, only registers the value (no memory write needed).
  public func writeOutput(_ node: Node, _ result: Expr) throws {
    use(val: result)
    guard case .tensor = node.shape,
      let tensor = ctx.g.nodeToTensor[node.id].flatMap({ ctx.g.tensors[$0] }),
      let loopIdx = ctx.tensorIndices[node.id]
    else {
      // scalar case, no need to store in tensor
      return
    }
    _ = tstore(tensor.cellId, value(loopIdx, scalarType: .int), result)
  }

  // MARK: - Indexing Helpers

  /// Compute a linear memory offset from multi-dimensional indices and strides.
  /// Skips zero-stride dimensions and optimizes stride-1 dimensions.
  /// The optional `offset` adds a constant base offset (for view slicing).
  public func stridedIndex(indices: [Expr], strides: [Int], offset: Int = 0) -> Expr {
    assert(indices.count == strides.count)
    var acc: Expr? = offset != 0 ? intConstant(offset) : nil
    for (idx, s) in zip(indices, strides) where s != 0 {
      let term = s == 1 ? idx : idx * intConstant(s)
      acc = acc.map { $0 + term } ?? term
    }
    return acc ?? intConstant(0)
  }

  // MARK: - Tensor Memory Indexing

  /// Compute a memory index for tensor access from a flat iteration index.
  /// Fast path: returns `flatIdx` directly for contiguous tensors with no offset.
  /// Slow path: decomposes into multi-dimensional indices, then applies strides and offset.
  public func tensorMemoryIndex(_ tensor: Tensor, flatIdx: Expr, shape: [Int]) -> Expr {
    if tensor.isContiguous && tensor.offset == 0 {
      return flatIdx
    }
    let multiIdx = flatToMultiIndex(flatIdx, shape)
    return stridedIndex(indices: multiIdx, strides: tensor.strides, offset: tensor.offset)
  }

  /// Compute a memory index for tensor access from pre-computed multi-dimensional indices.
  /// Applies the tensor's strides and offset to produce the final linear address.
  public func tensorMemoryIndex(_ tensor: Tensor, indices: [Expr]) -> Expr {
    return stridedIndex(indices: indices, strides: tensor.strides, offset: tensor.offset)
  }

  // MARK: - Broadcast Indexing

  /// Decompose a flat (linear) index into multi-dimensional indices for a row-major shape.
  /// Uses repeated division by trailing-dimension strides.
  /// When the input is int-typed, integer division truncates automatically (no `floor` needed).
  /// The last dimension uses the remainder directly since its stride is always 1.
  func flatToMultiIndex(_ flat: Expr, _ shape: [Int]) -> [Expr] {
    var indices: [Expr] = []
    var rem = flat
    let useIntMath = flat.scalarType == .int
    let lastDim = shape.count - 1
    for i in 0..<shape.count {
      if i == lastDim {
        indices.append(rem)
      } else {
        let stride = intConstant(shape[(i + 1)...].reduce(1, *))
        let quotient = rem / stride
        let idx = useIntMath ? quotient : floor(quotient)
        indices.append(idx)
        rem = rem - idx * stride
      }
    }
    return indices
  }

  /// Convert multi-dimensional indices back to a flat (linear) index for a row-major shape.
  /// Inverse of `flatToMultiIndex`.
  func multiIndexToFlat(_ indices: [Expr], _ shape: [Int]) -> Expr {
    var flat: Expr = intConstant(0)
    for i in 0..<shape.count {
      let stride = shape[(i + 1)...].reduce(1, *)
      flat = flat + indices[i] * intConstant(stride)
    }
    return flat
  }

  /// Map an output flat index to the corresponding input memory offset,
  /// handling shape broadcasting, view strides, and offsets.
  ///
  /// Fast paths:
  /// - Shapes match + contiguous + no offset → pass through the flat index.
  /// - Shapes match + contiguous + has offset → add the offset.
  ///
  /// Slow path: decomposes to multi-dim indices, right-aligns shapes, clamps broadcast
  /// dimensions to 0, and computes a strided index. Emits `.broadcastAccess` to disable SIMD.
  func broadcastIndex(
    outputIdx: Expr, outputShape: [Int],
    inputTensor: Tensor
  ) -> Expr {
    let inputShape = inputTensor.shape

    // fast path: shapes match + contiguous + no offset → just use flat idx
    if inputShape == outputShape && inputTensor.isContiguous && inputTensor.offset == 0 {
      return outputIdx
    }

    // fast path: shapes match + contiguous + has offset → just add offset
    if inputShape == outputShape && inputTensor.isContiguous {
      return outputIdx + intConstant(inputTensor.offset)
    }

    ops.append(UOp(op: .broadcastAccess, value: .empty))  // disable SIMD

    let multiIdx = flatToMultiIndex(outputIdx, outputShape)
    let rankDiff = outputShape.count - inputShape.count

    // right-align shapes, clamp broadcast dims to 0
    let indices = inputShape.enumerated().map { i, dim in
      dim == 1 ? intConstant(0) : multiIdx[i + rankDiff]
    }
    return tensorMemoryIndex(inputTensor, indices: indices)
  }
}

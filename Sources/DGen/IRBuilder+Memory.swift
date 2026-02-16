/// IRBuilder extension for memory and I/O operations.
/// Cell ops: load, store, delay1.
/// Raw memory: memoryRead, memoryWrite, memoryAccumulate.
/// Tensor registers: tload, tstore.
/// Frame-aware tensor ops: frameAwareOffset, frameAwareTensorRead, frameAwareTensorWrite.
/// I/O: output, input, tapeLoad, pi.

extension IRBuilder {

  // MARK: - Cell Ops

  /// Load the current value of a scalar cell. Emits a `.load` UOp.
  /// The optional `nodeId` overrides the source node for variable tracking.
  public func load(_ cell: CellID, _ nodeId: NodeID? = nil) -> Expr {
    let thunk = u_load(cell)
    let uop = thunk(ctx, nodeId)
    ops.append(uop)
    return value(uop.value)
  }

  /// Store a value into a scalar cell. Returns the stored value for chaining.
  public func store(_ cell: CellID, _ val: Expr) -> Expr {
    let thunk = u_store(cell, val.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  /// One-sample delay: reads the previous value of `cell`, then writes `a` as the new value.
  /// Used for feedback loops (e.g., IIR filters, recurrences).
  public func delay1(_ cell: CellID, _ a: Expr) -> Expr {
    let thunk = u_delay1(cell, a.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  // MARK: - Raw Memory

  /// Read from the global memory buffer at `memory[cellId + offset]`.
  /// Used for direct tensor element access and frame-aware storage.
  public func memoryRead(_ cellId: CellID, _ offset: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .memoryRead(cellId, offset.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Write a value to the global memory buffer at `memory[cellId + offset]`.
  /// Overwrites the existing value at that location.
  public func memoryWrite(_ cellId: CellID, _ offset: Expr, _ value: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .memoryWrite(cellId, offset.lazy, value.lazy), value: dest)
    ops.append(uop)
    return self.value(dest)
  }

  /// Atomically add `value` to `memory[cellId + offset]`.
  /// Used for gradient accumulation where multiple threads contribute to the same location.
  public func memoryAccumulate(_ cellId: CellID, _ offset: Expr, _ value: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .memoryAccumulate(cellId, offset.lazy, value.lazy), value: dest)
    ops.append(uop)
    return self.value(dest)
  }

  // MARK: - Tensor Ops (register-cached)

  /// Load from a tensor cell. Returns the cached register value if available,
  /// avoiding a redundant memory read. For frame-aware cells, uses frame-indexed addressing.
  /// Falls back to raw `memoryRead` for non-cached, non-frame-aware cells.
  public func tload(_ cell: CellID, _ idx: Expr) -> Expr {
    if let v = ctx.tensorCellToVar[cell] { return value(v) }
    // Frame-aware cells use frame-indexed storage
    if ctx.frameAwareTensorCells.contains(cell),
      let (tensorSize, _) = ctx.g.frameAwareCells[cell]
    {
      return frameAwareTensorRead(cellId: cell, tensorSize: tensorSize, elemIdx: idx)
    }
    return memoryRead(cell, cast(idx, to: .int))
  }

  /// Store a value to a tensor cell. Caches the value in a register for future `tload` calls.
  /// Only writes to memory if the cell is marked as outbound (needed by a later kernel).
  /// For frame-aware cells, uses frame-indexed addressing.
  public func tstore(_ cell: CellID, _ idx: Expr, _ val: Expr) -> Expr {
    ctx.tensorCellToVar[cell] = val.lazy
    guard ctx.outboundTensorCells.contains(cell) else { return val }
    // Frame-aware cells use frame-indexed storage
    if ctx.frameAwareTensorCells.contains(cell),
      let (tensorSize, _) = ctx.g.frameAwareCells[cell]
    {
      return frameAwareTensorWrite(cellId: cell, tensorSize: tensorSize, elemIdx: idx, value: val)
    }
    return memoryWrite(cell, cast(idx, to: .int), val)
  }

  // MARK: - Frame-Aware Tensor Ops

  /// Compute the linear memory offset for frame-aware storage: `frameIdx * tensorSize + elemIdx`.
  /// Result is cast to int for use as a memory index.
  private func frameAwareOffset(frameIdx: Expr, tensorSize: Int, elemIdx: Expr) -> Expr {
    return cast(frameIdx * intConstant(tensorSize) + elemIdx, to: .int)
  }

  /// Read from a frame-aware tensor using the current thread's frame index.
  /// Accesses `memory[cellId + frameIdx * tensorSize + elemIdx]`.
  /// Used for tensors that need per-frame storage to enable cross-block parallelism.
  public func frameAwareTensorRead(cellId: CellID, tensorSize: Int, elemIdx: Expr) -> Expr {
    return memoryRead(cellId, frameAwareOffset(frameIdx: frameIndex(nodeId), tensorSize: tensorSize, elemIdx: elemIdx))
  }

  /// Read from a frame-aware tensor with an explicit frame index.
  /// Used when the caller has already computed or overridden the frame index.
  public func frameAwareTensorRead(cellId: CellID, tensorSize: Int, frameIdx: Expr, elemIdx: Expr)
    -> Expr
  {
    return memoryRead(cellId, frameAwareOffset(frameIdx: frameIdx, tensorSize: tensorSize, elemIdx: elemIdx))
  }

  /// Write to a frame-aware tensor using the current thread's frame index.
  /// Stores to `memory[cellId + frameIdx * tensorSize + elemIdx]`.
  public func frameAwareTensorWrite(cellId: CellID, tensorSize: Int, elemIdx: Expr, value: Expr)
    -> Expr
  {
    return memoryWrite(cellId, frameAwareOffset(frameIdx: frameIndex(nodeId), tensorSize: tensorSize, elemIdx: elemIdx), value)
  }

  /// Write to a frame-aware tensor with an explicit frame index.
  /// Used when the caller has already computed or overridden the frame index.
  public func frameAwareTensorWrite(
    cellId: CellID, tensorSize: Int, frameIdx: Expr, elemIdx: Expr, value: Expr
  ) -> Expr {
    return memoryWrite(cellId, frameAwareOffset(frameIdx: frameIdx, tensorSize: tensorSize, elemIdx: elemIdx), value)
  }

  // MARK: - I/O

  /// Emit a write to the output buffer for the given audio channel.
  public func output(_ channelNumber: ChannelNumber, _ val: Expr) -> Expr {
    let thunk = u_output(channelNumber, val.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  /// Emit a read from the input buffer for the given audio channel.
  public func input(_ channelNumber: ChannelNumber) -> Expr {
    let thunk = u_input(channelNumber)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  /// Load a value from the tape (intermediates buffer) at a dynamic offset from `signal`.
  /// Used for delay lines and history access in audio processing.
  public func tapeLoad(_ signal: Expr, at offset: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .loadTape(signal.lazy, offset.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// The constant pi (3.14159...) as a float `Expr`.
  public var pi: Expr {
    return constant(Float.pi)
  }

  /// Retrieve a previously computed value for a node from the context's value cache.
  /// Used during backward pass to access forward-pass intermediates.
  /// Fatal errors if the value is not found.
  func tapeValue(_ nodeId: NodeID) -> Expr {
    if let lazy = ctx.values[nodeId] {
      return value(lazy)
    }

    fatalError("no tape")
  }

  // MARK: - Threadgroup Position

  /// The threadgroup's X position in the dispatch grid (column tile index).
  public func threadgroupPositionX() -> Expr {
    return emitIntOp(.threadgroupPositionX)
  }

  /// The threadgroup's Y position in the dispatch grid (row tile index).
  public func threadgroupPositionY() -> Expr {
    return emitIntOp(.threadgroupPositionY)
  }

  /// The threadgroup's Z position in the dispatch grid (frame index for per-frame GEMM).
  public func threadgroupPositionZ() -> Expr {
    return emitIntOp(.threadgroupPositionZ)
  }

  // MARK: - SIMD Group Matrix Ops

  /// Declare a zero-initialized simdgroup_float8x8 matrix variable.
  public func simdgroupMatrixZero() -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    ops.append(UOp(op: .simdgroupMatrixZero, value: dest))
    return value(dest)
  }

  /// Cooperatively load an 8x8 tile from memory[cellId + offset] with the given stride.
  public func simdgroupLoad(_ cellId: CellID, offset: Expr, stride: Int) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    ops.append(UOp(op: .simdgroupLoad(cellId, offset.lazy, stride), value: dest))
    return value(dest)
  }

  /// Cooperatively store an 8x8 tile to memory[cellId + offset] with the given stride.
  @discardableResult
  public func simdgroupStore(_ src: Expr, cellId: CellID, offset: Expr, stride: Int) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    ops.append(UOp(op: .simdgroupStore(src.lazy, cellId, offset.lazy, stride), value: dest))
    return value(dest)
  }

  /// Multiply-accumulate: acc = a * b + acc. In-place on the accumulator.
  /// Uses acc's variable as the result — the renderer emits
  /// `simdgroup_multiply_accumulate(acc, a, b, acc)` which modifies acc in place.
  @discardableResult
  public func simdgroupMultiplyAccumulate(_ a: Expr, _ b: Expr, _ acc: Expr) -> Expr {
    ops.append(UOp(op: .simdgroupMultiplyAccumulate(a.lazy, b.lazy, acc.lazy), value: acc.lazy))
    return acc
  }
}

import Darwin

public final class IRBuilder {
  public let ctx: IRContext
  public let nodeId: NodeID
  public var ops: [UOp] = []

  public init(ctx: IRContext, nodeId: NodeID) {
    self.ctx = ctx
    self.nodeId = nodeId
  }

  public func value(_ lazy: Lazy) -> Expr {
    return Expr(lazy, ctx: ctx, nodeId: nodeId, builder: self)
  }

  func values(_ inputs: [Lazy], count: Int) -> (Expr, Expr) {
    guard inputs.count == count else { fatalError("expected \(count) values") }
    return (value(inputs[0]), value(inputs[1]))
  }

  func values(_ inputs: [Lazy], count: Int) -> (Expr, Expr, Expr) {
    guard inputs.count == count else { fatalError("expected \(count) values") }
    return (value(inputs[0]), value(inputs[1]), value(inputs[2]))
  }

  func values(_ inputs: [Lazy], count: Int) -> (Expr, Expr, Expr, Expr) {
    guard inputs.count == count else { fatalError("expected \(count) values") }
    return (value(inputs[0]), value(inputs[1]), value(inputs[2]), value(inputs[3]))
  }

  public func constant(_ v: Float) -> Expr {
    let l = ctx.useConstant(src: nodeId, value: v)
    return value(l)
  }

  public func load(_ cell: CellID, _ nodeId: NodeID? = nil) -> Expr {
    let thunk = u_load(cell)
    let uop = thunk(ctx, nodeId)
    ops.append(uop)
    return value(uop.value)
  }

  public func use(val: Expr) {
    ctx.values[nodeId] = val.lazy
  }

  public func store(_ cell: CellID, _ val: Expr) -> Expr {
    let thunk = u_store(cell, val.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  public func delay1(_ cell: CellID, _ a: Expr) -> Expr {
    let thunk = u_delay1(cell, a.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  public func concatShift(_ a: Expr, _ b: Expr, _ shift: Int) -> Expr {
    let thunk = u_concatShift(a.lazy, b.lazy, shift)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  public func output(_ channelNumber: ChannelNumber, _ val: Expr) -> Expr {
    let thunk = u_output(channelNumber, val.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  public func input(_ channelNumber: ChannelNumber) -> Expr {
    let thunk = u_input(channelNumber)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  public func gswitch(_ cond: Expr, _ then: Expr, _ els: Expr) -> Expr {
    let thunk = u_switch(cond.lazy, then.lazy, els.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  func frameIndex(_ nodeId: NodeID) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .frameIndex, value: dest)
    ops.append(uop)
    return value(uop.value)
  }

  func mutate(_ target: Expr, to newValue: Expr) {
    guard case .variable(_, _) = target.lazy else {
      fatalError("Can only mutate variables")
    }
    let uop = UOp(op: .mutate(target.lazy, newValue.lazy), value: target.lazy)
    ops.append(uop)
  }

  public func `if`(_ cond: Expr, then: () -> Void) {
    ops.append(u_begin_if(cond.lazy)(ctx, nodeId))
    then()
    ops.append(u_end_if()(ctx, nil))
  }

  // Read a forward intermediate value for a given nodeId from the tape (intermediates buffer)
  // Uses ctx.tapeIndex to find the per-node offset; returns an Expr referencing that value.
  func tapeValue(_ nodeId: NodeID) -> Expr {
    if let lazy = ctx.values[nodeId] {
      return value(lazy)
    }

    fatalError("no tape")
  }

  public func abs(_ val: Expr) -> Expr {
    if case .constant(_, let absConst) = val.lazy {
      return self.constant(fabs(absConst))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .abs(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func sign(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sign(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func sin(_ val: Expr) -> Expr {
    if case .constant(_, let sinConst) = val.lazy {
      return self.constant(sinf(sinConst))
    }

    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sin(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func neg(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let negativeConstant = constant(-1)
    let uop = UOp(op: .mul(val.lazy, negativeConstant.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func cos(_ val: Expr) -> Expr {
    if case .constant(_, let cosConst) = val.lazy {
      return self.constant(cosf(cosConst))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .cos(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func tan(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .tan(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func tanh(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .tanh(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func exp(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .exp(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func applyUOp(op: Op) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: op, value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func log(_ val: Expr) -> Expr {
    return applyUOp(op: .log(val.lazy))
  }

  public func log10(_ val: Expr) -> Expr {
    return applyUOp(op: .log10(val.lazy))
  }

  public func and(_ a: Expr, _ b: Expr) -> Expr {
    return applyUOp(op: .and(a.lazy, b.lazy))
  }

  public func or(_ a: Expr, _ b: Expr) -> Expr {
    return applyUOp(op: .or(a.lazy, b.lazy))
  }

  public func xor(_ a: Expr, _ b: Expr) -> Expr {
    return applyUOp(op: .xor(a.lazy, b.lazy))
  }

  public func sqrt(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sqrt(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func pow(_ base: Expr, _ exponent: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .pow(base.lazy, exponent.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func atan2(_ y: Expr, _ x: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .atan2(y.lazy, x.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func floor(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .floor(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func ceil(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .ceil(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func round(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .round(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func noise(_ cellId: CellID) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .noise(cellId), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func memoryRead(_ cellId: CellID, _ offset: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .memoryRead(cellId, offset.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func memoryWrite(_ cellId: CellID, _ offset: Expr, _ value: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .memoryWrite(cellId, offset.lazy, value.lazy), value: dest)
    ops.append(uop)
    return self.value(dest)
  }

  public func memoryAccumulate(_ cellId: CellID, _ offset: Expr, _ value: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .memoryAccumulate(cellId, offset.lazy, value.lazy), value: dest)
    ops.append(uop)
    return self.value(dest)
  }

  // MARK: - Tensor Ops (register-cached)

  /// Load from tensor cell - uses cached register if available
  /// For frame-aware cells, uses frame-indexed access pattern
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

  /// Store to tensor cell - caches in register, only writes memory if outbound
  /// For frame-aware cells, uses frame-indexed access pattern
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

  /// Read from frame-aware tensor: memory[cellId + frameIdx * tensorSize + elemIdx]
  /// Used for tensors that need per-frame storage to enable parallelism
  public func frameAwareTensorRead(cellId: CellID, tensorSize: Int, elemIdx: Expr) -> Expr {
    let frameIdx = frameIndex(nodeId)
    let offset = frameIdx * constant(Float(tensorSize)) + elemIdx
    return memoryRead(cellId, cast(offset, to: .int))
  }

  /// Read from frame-aware tensor with explicit frame index
  public func frameAwareTensorRead(cellId: CellID, tensorSize: Int, frameIdx: Expr, elemIdx: Expr)
    -> Expr
  {
    let offset = frameIdx * constant(Float(tensorSize)) + elemIdx
    return memoryRead(cellId, cast(offset, to: .int))
  }

  /// Write to frame-aware tensor: memory[cellId + frameIdx * tensorSize + elemIdx]
  public func frameAwareTensorWrite(cellId: CellID, tensorSize: Int, elemIdx: Expr, value: Expr)
    -> Expr
  {
    let frameIdx = frameIndex(nodeId)
    let offset = frameIdx * constant(Float(tensorSize)) + elemIdx
    return memoryWrite(cellId, cast(offset, to: .int), value)
  }

  /// Write to frame-aware tensor with explicit frame index
  public func frameAwareTensorWrite(
    cellId: CellID, tensorSize: Int, frameIdx: Expr, elemIdx: Expr, value: Expr
  ) -> Expr {
    let offset = frameIdx * constant(Float(tensorSize)) + elemIdx
    return memoryWrite(cellId, cast(offset, to: .int), value)
  }

  // MARK: - High-level tensor I/O

  public func readInput(_ node: Node, _ inputs: [Lazy], at idx: Int) throws -> Expr {
    let inputId = node.inputs[idx]
    guard let inputNode = ctx.g.nodes[inputId] else { throw DGenError.missingTensorID }

    // scalar
    if case .scalar = inputNode.shape ?? .scalar { return value(inputs[idx]) }

    // tensor
    guard case .tensor(let outShape) = node.shape,
      let tensor = ctx.g.nodeToTensor[inputId].flatMap({ ctx.g.tensors[$0] }),
      let loopIdx = ctx.tensorIndices[node.id]
    else {
      throw DGenError.missingTensorID
    }

    // For padded or repeated tensors, use tensorRead which handles bounds checking and modular indexing
    if tensor.padding != nil || tensor.innerShapeForRepeat != nil {
      let indices = flatToMultiIndex(value(loopIdx), outShape)
      return tensorRead(
        tensor,
        indices: broadcastIndices(
          outputIndices: indices, outputShape: outShape, inputTensor: tensor))
    }

    let memOffset = broadcastIndex(
      outputIdx: value(loopIdx), outputShape: outShape,
      inputTensor: tensor
    )
    return tload(tensor.cellId, memOffset)
  }

  /// Compute broadcast-adjusted indices (handles shape broadcasting)
  func broadcastIndices(outputIndices: [Expr], outputShape: [Int], inputTensor: Tensor) -> [Expr] {
    let inputShape = inputTensor.shape
    let rankDiff = outputShape.count - inputShape.count

    // right-align shapes, clamp broadcast dims to 0
    return inputShape.enumerated().map { i, dim in
      dim == 1 ? constant(0) : outputIndices[i + rankDiff]
    }
  }

  public func writeOutput(_ node: Node, _ result: Expr) throws {
    use(val: result)
    guard case .tensor = node.shape,
      let tensor = ctx.g.nodeToTensor[node.id].flatMap({ ctx.g.tensors[$0] }),
      let loopIdx = ctx.tensorIndices[node.id]
    else {
      // scalar case, no need to store in tensor
      return
    }
    _ = tstore(tensor.cellId, value(loopIdx), result)
  }

  /// Multi-dim indices + strides + offset → linear memory offset
  public func stridedIndex(indices: [Expr], strides: [Int], offset: Int = 0) -> Expr {
    assert(indices.count == strides.count)
    var acc: Expr? = offset != 0 ? constant(Float(offset)) : nil
    for (idx, s) in zip(indices, strides) where s != 0 {
      let term = s == 1 ? idx : idx * constant(Float(s))
      acc = acc.map { $0 + term } ?? term
    }
    return acc ?? constant(0)
  }

  // MARK: - Tensor Memory Indexing (encapsulates fast path vs strided path)

  /// Compute memory index for tensor access from flat iteration index.
  /// Fast path avoids multi-index conversion for contiguous tensors with no offset.
  public func tensorMemoryIndex(_ tensor: Tensor, flatIdx: Expr, shape: [Int]) -> Expr {
    if tensor.isContiguous && tensor.offset == 0 {
      return flatIdx
    }
    let multiIdx = flatToMultiIndex(flatIdx, shape)
    return stridedIndex(indices: multiIdx, strides: tensor.strides, offset: tensor.offset)
  }

  /// Compute memory index for tensor access from multi-dimensional indices.
  /// Uses strides and offset to support views (shrink, transpose, etc.)
  public func tensorMemoryIndex(_ tensor: Tensor, indices: [Expr]) -> Expr {
    return stridedIndex(indices: indices, strides: tensor.strides, offset: tensor.offset)
  }

  // MARK: - Tensor Read (handles padding)

  /// Read from tensor with padding support.
  /// For non-padded tensors: direct memory read.
  /// For padded tensors: returns 0 for padded regions, real data otherwise.
  public func tensorRead(_ tensor: Tensor, indices: [Expr]) -> Expr {
    var adjustedIndices = indices

    // Step 1: Apply shrinkStart if present (for shrunk repeated tensors)
    // This converts from shrunk tensor indices to full repeated tensor indices
    if let shrinkStart = tensor.shrinkStart {
      adjustedIndices = indices.enumerated().map { i, idx in
        idx + constant(Float(shrinkStart[i]))
      }
    }

    // Step 2: Apply modular indexing for repeated tensor
    // This wraps indices to the original tensor size
    if let innerShape = tensor.innerShapeForRepeat {
      adjustedIndices = adjustedIndices.enumerated().map { i, idx in
        mod(idx, constant(Float(innerShape[i])))
      }
    }

    if let padding = tensor.padding {
      // Padded tensor: default to 0, conditionally read real data
      let val = float(0.0)

      // Build inBounds check for all axes
      var inBounds: Expr = constant(1.0)
      for (i, (left, right)) in padding.enumerated() {
        let idx = adjustedIndices[i]
        let innerEnd = constant(Float(tensor.shape[i] - right))
        // idx >= left AND idx < innerEnd
        inBounds = inBounds * (idx >= constant(Float(left))) * (idx < innerEnd)
      }

      self.if(inBounds) {
        // Adjust indices by subtracting padLeft
        let adjusted = adjustedIndices.enumerated().map { i, idx in
          idx - constant(Float(padding[i].left))
        }
        let memIdx = tensorMemoryIndex(tensor, indices: adjusted)
        mutate(val.value, to: memoryRead(tensor.cellId, memIdx))
      }

      return val.value
    } else {
      // Non-padded: direct read using (possibly modulo-adjusted) indices
      let memIdx = tensorMemoryIndex(tensor, indices: adjustedIndices)
      return memoryRead(tensor.cellId, memIdx)
    }
  }

  /// Read from tensor using flat index with padding support.
  /// For frame-aware cells, uses frame-indexed addressing.
  public func tensorRead(_ tensor: Tensor, flatIdx: Expr, shape: [Int]) -> Expr {
    if tensor.padding != nil || tensor.innerShapeForRepeat != nil || tensor.shrinkStart != nil {
      // Need multi-dim indices for padding check, modular repeat indexing, or shrink offset
      let indices = flatToMultiIndex(flatIdx, shape)
      return tensorRead(tensor, indices: indices)
    } else if ctx.frameAwareTensorCells.contains(tensor.cellId),
      let (tensorSize, _) = ctx.g.frameAwareCells[tensor.cellId]
    {
      // Frame-aware tensor: use frame-indexed addressing
      return frameAwareTensorRead(cellId: tensor.cellId, tensorSize: tensorSize, elemIdx: flatIdx)
    } else {
      // Non-padded: use fast path
      let memIdx = tensorMemoryIndex(tensor, flatIdx: flatIdx, shape: shape)
      return memoryRead(tensor.cellId, memIdx)
    }
  }

  // MARK: - Broadcast Indexing

  /// Flat index → multi-dim indices for shape (row-major)
  func flatToMultiIndex(_ flat: Expr, _ shape: [Int]) -> [Expr] {
    var indices: [Expr] = []
    var rem = flat
    for i in 0..<shape.count {
      let stride = shape[(i + 1)...].reduce(1, *)
      if stride == 1 {
        indices.append(rem)
      } else {
        let s = constant(Float(stride))
        let idx = floor(rem / s)
        indices.append(idx)
        rem = rem - idx * s
      }
    }
    return indices
  }

  /// Output flat idx → input memory offset (handles broadcasting and view offsets)
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
      return outputIdx + constant(Float(inputTensor.offset))
    }

    ops.append(UOp(op: .broadcastAccess, value: .empty))  // disable SIMD

    let multiIdx = flatToMultiIndex(outputIdx, outputShape)
    let rankDiff = outputShape.count - inputShape.count

    // right-align shapes, clamp broadcast dims to 0
    let indices = inputShape.enumerated().map { i, dim in
      dim == 1 ? constant(0) : multiIdx[i + rankDiff]
    }
    return tensorMemoryIndex(inputTensor, indices: indices)
  }

  public func min(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .min(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func max(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .max(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func mod(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .mod(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public func mix(_ a: Expr, _ b: Expr, _ t: Expr) -> Expr {
    // mix(a, b, t) = a * (1 - t) + b * t
    return u_mix(a, b, lerp: t)(self)
  }

  public func selector(_ mode: Expr, _ options: [Expr]) -> Expr {
    if case .constant(let constId, let constMode) = mode.lazy {
      if constMode <= 0 {
        return self.constant(0)
      }
      for (i, option) in options.enumerated() {
        if constMode <= Float(i + 1) {
          return option
        }
      }
      return self.constant(0)
    }
    let dest = ctx.useVariable(src: nodeId)
    let optionLazys = options.map { $0.lazy }
    let uop = UOp(op: .selector(mode.lazy, optionLazys), value: dest)

    ops.append(uop)
    return value(dest)
  }

  // MARK: - New IRBuilder Abstractions

  public func float(_ value: Float) -> MutableVar {
    let constLazy = ctx.useConstant(src: nodeId, value: value)
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .declareVar(constLazy), value: dest)
    ops.append(uop)
    return MutableVar(dest, ctx: ctx, nodeId: nodeId, builder: self)
  }

  public func integer(_ value: Int) -> MutableVar {
    let constLazy = ctx.useConstant(src: nodeId, value: Float(value))
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .declareVar(constLazy), value: dest)
    ops.append(uop)
    return MutableVar(dest, ctx: ctx, nodeId: nodeId, builder: self)
  }

  /// Create an integer constant expression (for index calculations)
  public func int(_ value: Int) -> Expr {
    let constLazy = ctx.useConstant(src: nodeId, value: Float(value))
    return Expr(constLazy, ctx: ctx, nodeId: nodeId, builder: self)
  }

  /// Add two expressions
  public func add(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .add(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Multiply two expressions
  public func mul(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .mul(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Divide two expressions
  public func div(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .div(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Floor division: floor(a / b)
  /// Use this for integer index calculations where we need truncation toward negative infinity.
  public func floorDiv(_ a: Expr, _ b: Expr) -> Expr {
    return floor(div(a, b))
  }

  public func loop(_ count: Int, body: (Expr) -> Void) {
    let loopVar = ctx.useVariable(src: nodeId)
    let countLazy = ctx.useConstant(src: nodeId, value: Float(count))
    ops.append(UOp(op: .beginForLoop(loopVar, countLazy), value: loopVar))
    body(value(loopVar))
    ops.append(UOp(op: .endLoop, value: ctx.useVariable(src: nil)))
  }

  /// Conditional execution block.
  /// The body is only executed when condition is true (non-zero).
  public func if_(_ condition: Expr, body: () -> Void) {
    ops.append(UOp(op: .beginIf(condition.lazy), value: ctx.useVariable(src: nil)))
    body()
    ops.append(UOp(op: .endIf, value: ctx.useVariable(src: nil)))
  }

  /// Parallel range for tensor operations.
  /// Iterations are independent and can be parallelized.
  /// Renderer decides: C emits a loop, Metal static emits thread-parallel.
  ///
  /// When in a frame-aware tensor block (ctx.isInFrameAwareTensorBlock == true),
  /// the block already handles parallelism via flat threading (frameCount × tensorSize threads).
  /// In this case, we skip emitting a loop and use the pre-computed element index directly.
  public func parallelRange(_ count: Int, body: (Expr) -> Void, kind: Kind? = .scalar) {
    // In frame-aware tensor blocks, the parallelism is already handled by flat threading.
    // Use the pre-computed element index instead of creating a loop.
    if ctx.isInFrameAwareTensorBlock, let elemIdx = ctx.frameAwareTensorElementIndex {
      body(value(elemIdx))
      return
    }

    let indexVar = ctx.useVariable(src: nodeId)

    var incr = 1
    if case .simd = kind {
      incr = 4
    }

    ops.append(UOp(op: .beginParallelRange(count, incr), value: indexVar))
    body(value(indexVar))
    ops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
    if case .simd = kind {
      ops = ops.map { UOp(op: $0.op, value: $0.value, kind: $0.kind, kindOverride: .simd) }
    }
  }

  /// Parallel map over (frame, bin) where total threads = frameCount * bins.
  /// The body receives (frameIndex, binIndex) as floats.
  public func parallelMap2D(bins: Int, body: (Expr, Expr) -> Void) {
    setThreadCountScale(bins)
    let flatIdx = threadIndex()
    let binsExpr = constant(Float(bins))
    let frameIdx = floor(flatIdx / binsExpr)
    setFrameIndex(frameIdx)
    let binIdx = flatIdx - frameIdx * binsExpr
    body(frameIdx, binIdx)
  }

  public func threadIndex() -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .threadIndex, value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Returns the current frame index. In normal blocks, this is the thread index.
  /// In frame-aware tensor blocks, this returns the pre-computed frame index from
  /// the flat thread decomposition (flat index / tensor size).
  public func currentFrameIndex() -> Expr {
    if ctx.isInFrameAwareTensorBlock, let frameIdx = ctx.frameAwareTensorFrameIndex {
      return value(frameIdx)
    }
    // Fall back to thread index for normal blocks
    return threadIndex()
  }

  /// Returns the frame index using the .frameIndex UOp.
  /// This returns _frameIndex if setFrameIndex was called, otherwise falls back to thread index.
  public func frameIndex() -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .frameIndex, value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Frame count (runtime parameter)
  public func frameCount() -> Expr {
    return value(.variable(-1, nil))
  }

  /// Override the frame index used for outputs/gradients in this kernel
  public func setFrameIndex(_ expr: Expr) {
    let uop = UOp(op: .setFrameIndex(expr.lazy), value: expr.lazy)
    ops.append(uop)
  }

  /// Dispatch threads as frameCount * scale for this kernel
  public func setThreadCountScale(_ scale: Int) {
    let uop = UOp(op: .setThreadCountScale(scale), value: .empty)
    ops.append(uop)
  }

  public func tapeLoad(_ signal: Expr, at offset: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .loadTape(signal.lazy, offset.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  public var pi: Expr {
    return constant(Float.pi)
  }

  public func cast(_ expr: Expr, to type: CastType) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .cast(expr.lazy, type), value: dest)
    ops.append(uop)
    return value(dest)
  }
}

public struct Expr {
  public let lazy: Lazy
  private let ctx: IRContext
  private let nodeId: NodeID
  private unowned let builder: IRBuilder

  init(_ lazy: Lazy, ctx: IRContext, nodeId: NodeID, builder: IRBuilder) {
    self.lazy = lazy
    self.ctx = ctx
    self.nodeId = nodeId
    self.builder = builder
  }

  // Operators emit automatically
  static func + (lhs: Expr, rhs: Expr) -> Expr {
    if case .constant(let lconst) = lhs.lazy {
      if case .constant(let hconst) = rhs.lazy {
        let sum = lconst.1 + hconst.1
        let ctx = lhs.ctx
        return Expr(
          ctx.useConstant(src: lhs.nodeId, value: sum), ctx: lhs.ctx, nodeId: lhs.nodeId,
          builder: lhs.builder)
      }
    }
    let thunk = u_add(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func / (lhs: Expr, rhs: Expr) -> Expr {
    if case .constant(let lconst) = lhs.lazy {
      if case .constant(let hconst) = rhs.lazy {
        let sum = lconst.1 / hconst.1
        let ctx = lhs.ctx
        return Expr(
          ctx.useConstant(src: lhs.nodeId, value: sum), ctx: lhs.ctx, nodeId: lhs.nodeId,
          builder: lhs.builder)
      }
    }

    let thunk = u_div(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func * (lhs: Expr, rhs: Expr) -> Expr {
    if case .constant(_, let lconst) = lhs.lazy {
      if case .constant(_, let hconst) = rhs.lazy {
        let sum = lconst * hconst
        let ctx = lhs.ctx
        return Expr(
          ctx.useConstant(src: lhs.nodeId, value: sum), ctx: lhs.ctx, nodeId: lhs.nodeId,
          builder: lhs.builder)
      }
    }

    let thunk = u_mul(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func - (lhs: Expr, rhs: Expr) -> Expr {
    if case .constant(_, let lconst) = lhs.lazy {
      if case .constant(_, let hconst) = rhs.lazy {
        let sum = lconst - hconst
        let ctx = lhs.ctx
        return Expr(
          ctx.useConstant(src: lhs.nodeId, value: sum), ctx: lhs.ctx, nodeId: lhs.nodeId,
          builder: lhs.builder)
      }
    }

    let thunk = u_sub(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func > (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_gt(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func >= (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_gte(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func <= (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_lte(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func < (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_lt(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func == (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_eq(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func % (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_mod(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

}

public struct MutableVar {
  public let lazy: Lazy
  private unowned let builder: IRBuilder
  private let ctx: IRContext
  private let nodeId: NodeID

  init(_ lazy: Lazy, ctx: IRContext, nodeId: NodeID, builder: IRBuilder) {
    self.lazy = lazy
    self.ctx = ctx
    self.nodeId = nodeId
    self.builder = builder
  }

  public var value: Expr {
    return Expr(lazy, ctx: ctx, nodeId: nodeId, builder: builder)
  }

  public func accumulate(_ expr: Expr) {
    // Emit: self = self + expr
    let sum = value + expr
    let uop = UOp(op: .mutate(lazy, sum.lazy), value: lazy)
    builder.ops.append(uop)
  }
}

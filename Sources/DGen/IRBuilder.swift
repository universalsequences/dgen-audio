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

  public func grad(_ nodeId: NodeID, value: Lazy) {
    let gradId = ctx.useGradient(src: nodeId)
    let uop = UOp(op: .accumulateGrad(gradId, value), value: value)
    ops.append(uop)
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

  func loadGradMemory(_ cellId: CellID) -> Expr {
    let dest = ctx.useVariable(src: nil)
    let uop = UOp(op: .loadGradMemory(cellId), value: dest)
    ops.append(uop)
    return value(dest)
  }

  func loadGrad(_ gradId: GradID) -> Expr {
    let dest = ctx.useVariable(src: nil)
    let uop = UOp(op: .loadGrad(gradId), value: dest)
    ops.append(uop)
    return value(dest)
  }

  func storeGradMemory(_ cellId: CellID, _ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nil)
    let uop = UOp(op: .storeGradMemory(cellId, val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
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

  // MARK: - Tensor Register Optimization
  //
  // These methods optimize tensor operations by keeping intermediate values in registers
  // instead of going through memory for every operation.
  //
  // tensorMemoryRead: Check if value is already in a register from a previous computation
  //                   in this block. If so, return that register. Otherwise emit memoryRead.
  //
  // tensorMemoryWrite: Record the computed value in a register. Only emit actual memoryWrite
  //                    if this cell is needed by later blocks (outbound).

  /// Read tensor cell, preferring register if available
  public func tensorMemoryRead(_ cellId: CellID, _ offset: Expr) -> Expr {
    // Check if we already have this cell's value in a register from earlier in this block
    if let existingVar = ctx.tensorCellToVar[cellId] {
      return value(existingVar)
    }
    // Not in register, need to read from memory
    return memoryRead(cellId, offset)
  }

  /// Write tensor cell, only emitting memory write if needed for cross-block transfer
  public func tensorMemoryWrite(_ cellId: CellID, _ offset: Expr, _ val: Expr) -> Expr {
    // Always record in tensor register map so subsequent reads can use the register
    ctx.tensorCellToVar[cellId] = val.lazy

    // Only emit actual memory write if this cell is needed by later blocks
    if ctx.outboundTensorCells.contains(cellId) {
      return memoryWrite(cellId, offset, val)
    }

    // Cell is only used within this block - skip memory write, value stays in register
    return val
  }

  // MARK: - Strided Tensor Indexing

  /// Compute linear memory index from multi-dimensional indices using tensor strides.
  /// Optimizes for the trivial case (contiguous row-major) to avoid unnecessary operations.
  ///
  /// For a tensor with shape [M, N] and strides [N, 1]:
  ///   linearIndex([row, col]) = row * N + col  (trivial: last stride is 1)
  ///
  /// For a transposed tensor with shape [N, M] and strides [1, N]:
  ///   linearIndex([row, col]) = row * 1 + col * N  (non-trivial: requires actual stride math)
  ///
  public func stridedIndex(indices: [Expr], strides: [Int]) -> Expr {
    guard indices.count == strides.count else {
      fatalError("stridedIndex: indices count (\(indices.count)) must match strides count (\(strides.count))")
    }

    guard !indices.isEmpty else {
      return constant(0)
    }

    var result: Expr? = nil

    for i in 0..<indices.count {
      let stride = strides[i]

      // Skip stride-0 dimensions (broadcast dimensions)
      if stride == 0 { continue }

      let idx = indices[i]

      // Optimization: stride of 1 means just use the index directly
      let term: Expr
      if stride == 1 {
        term = idx
      } else {
        // Use Expr operators (* and +) which emit proper IR
        term = idx * constant(Float(stride))
      }

      // Accumulate terms
      if let r = result {
        result = r + term
      } else {
        result = term
      }
    }

    return result ?? constant(0)
  }

  /// Read from tensor using multi-dimensional indices (handles strides correctly)
  /// This is the preferred way to read tensor elements when you have logical indices.
  public func tensorRead(cellId: CellID, indices: [Expr], strides: [Int]) -> Expr {
    // Check register cache first
    if let existingVar = ctx.tensorCellToVar[cellId] {
      return value(existingVar)
    }

    // Compute linear index using strides
    let linearIdx = stridedIndex(indices: indices, strides: strides)
    return memoryRead(cellId, cast(linearIdx, to: .int))
  }

  // MARK: - Broadcast Indexing

  /// Convert a flat output index to multi-dimensional indices for the given shape.
  /// e.g., flatIdx=7 with shape [2,2,3] -> [1,0,1] (row-major)
  public func flatToMultiIndex(flatIdx: Expr, shape: [Int]) -> [Expr] {
    guard !shape.isEmpty else { return [] }

    var indices: [Expr] = []
    var remaining = flatIdx

    for i in 0..<shape.count {
      // Compute stride for this dimension (product of all subsequent dims)
      let stride = shape[(i+1)...].reduce(1, *)

      if stride == 1 {
        // Last dimension - just use remaining
        indices.append(remaining)
      } else {
        // idx[i] = floor(remaining / stride)
        let strideConst = constant(Float(stride))
        let rawIdx = remaining / strideConst
        let flooredIdx = floor(rawIdx)
        indices.append(flooredIdx)  // Use floored index!
        // remaining = remaining % stride (using: a - floor(a/b)*b)
        remaining = remaining - (flooredIdx * strideConst)
      }
    }

    return indices
  }

  /// Compute broadcast-aware memory index for reading from an input tensor.
  /// Takes the output's flat index and output shape, and computes the correct
  /// memory offset for an input tensor that may have different (broadcastable) shape.
  ///
  /// Broadcasting rules:
  /// - Shapes are right-aligned (pad shorter shape with 1s on the left)
  /// - Where input dim == 1, that index is clamped to 0 (broadcast)
  /// - Uses input tensor's strides for the final memory offset
  ///
  /// Example:
  ///   outputIdx=7, outputShape=[2,2,3], inputShape=[2,1,3], inputStrides=[3,3,1]
  ///   -> multiIdx = [1,0,1]
  ///   -> broadcastIdx = [1,0,1] (middle dim clamped to 0 since input dim is 1)
  ///   -> memoryOffset = 1*3 + 0*3 + 1*1 = 4
  public func broadcastIndex(
    outputIdx: Expr,
    outputShape: [Int],
    inputShape: [Int],
    inputStrides: [Int]
  ) -> Expr {
    // Fast path: if shapes match exactly and strides are contiguous (row-major),
    // just use the flat index directly - no complex arithmetic needed.
    // This is critical for SIMD optimization to work.
    if inputShape == outputShape {
      let expectedStrides = Tensor.computeRowMajorStrides(inputShape)
      if inputStrides == expectedStrides {
        return outputIdx
      }
    }

    // Emit marker to signal SIMD should be disabled for this block
    ops.append(UOp(op: .broadcastAccess, value: .empty))

    // Convert flat index to multi-dimensional indices for output shape
    let multiIdx = flatToMultiIndex(flatIdx: outputIdx, shape: outputShape)

    // Right-align shapes (pad input with 1s on left if needed)
    let outputRank = outputShape.count
    let inputRank = inputShape.count
    let rankDiff = outputRank - inputRank

    var broadcastIndices: [Expr] = []
    var broadcastStrides: [Int] = []

    for i in 0..<inputRank {
      let outputDimIdx = i + rankDiff
      let inputDim = inputShape[i]
      let outputIdx = multiIdx[outputDimIdx]

      if inputDim == 1 {
        // Broadcast dimension - always index 0
        broadcastIndices.append(constant(0))
      } else {
        // Normal dimension - use the output index
        broadcastIndices.append(outputIdx)
      }
      broadcastStrides.append(inputStrides[i])
    }

    // Compute final memory offset using input strides
    return stridedIndex(indices: broadcastIndices, strides: broadcastStrides)
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

  public func loop(_ count: Int, body: (Expr) -> Void) {
    let loopVar = ctx.useVariable(src: nodeId)
    let countLazy = ctx.useConstant(src: nodeId, value: Float(count))
    ops.append(UOp(op: .beginForLoop(loopVar, countLazy), value: loopVar))
    body(value(loopVar))
    ops.append(UOp(op: .endLoop, value: ctx.useVariable(src: nil)))
  }

  /// Parallel range for tensor operations.
  /// Iterations are independent and can be parallelized.
  /// Renderer decides: C emits a loop, Metal static emits thread-parallel.
  public func parallelRange(_ count: Int, body: (Expr) -> Void, kind: Kind? = .scalar) {
    let indexVar = ctx.useVariable(src: nodeId)

    var incr = 1
    if case .simd = kind {
      incr = 4
    }

    ops.append(UOp(op: .beginParallelRange(count, incr), value: indexVar))

    // Emit parallelIndex - use the SAME variable as beginParallelRange
    // so the renderer can match them up
    //ops.append(UOp(op: .parallelIndex, value: indexVar))
    body(value(indexVar))
    ops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
    if case .simd = kind {
      ops = ops.map { UOp(op: $0.op, value: $0.value, kind: $0.kind, kindOverride: .simd) }
    }
  }

  public func threadIndex() -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .threadIndex, value: dest)
    ops.append(uop)
    return value(dest)
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

/// IRBuilder extension for control flow and threading.
/// Loops: loop, parallelRange, parallelMap2D.
/// Conditionals: gswitch, if, if_.
/// Threading: threadIndex, currentFrameIndex, frameIndex, frameCount, setFrameIndex,
///            setThreadCountScale, setupFlatThreading.
/// Other: mutate.

extension IRBuilder {

  // MARK: - Conditionals

  /// Emit a ternary select: returns `then` when `cond` is non-zero, `els` otherwise.
  /// Rendered as `select(els, then, cond)` in Metal or `cond ? then : els` in C.
  public func gswitch(_ cond: Expr, _ then: Expr, _ els: Expr) -> Expr {
    let thunk = u_switch(cond.lazy, then.lazy, els.lazy)
    let uop = thunk(ctx, nil)
    ops.append(uop)
    return value(uop.value)
  }

  /// Emit a conditional block using the legacy `beginIf`/`endIf` UOp pair.
  /// Prefer `if_` for new code — it uses the same mechanism but with a cleaner name.
  public func `if`(_ cond: Expr, then: () -> Void) {
    ops.append(u_begin_if(cond.lazy)(ctx, nodeId))
    then()
    ops.append(u_end_if()(ctx, nil))
  }

  /// Emit a conditional execution block. The body executes only when `condition` is non-zero.
  /// Emits `beginIf`/`endIf` UOps that the renderer translates to `if (cond) { ... }`.
  public func if_(_ condition: Expr, body: () -> Void) {
    ops.append(UOp(op: .beginIf(condition.lazy), value: ctx.useVariable(src: nil)))
    body()
    ops.append(UOp(op: .endIf, value: ctx.useVariable(src: nil)))
  }

  // MARK: - Loops

  /// Emit a sequential for-loop from 0 to `count - 1`.
  /// The body receives the loop index as an int-typed `Expr`.
  public func loop(_ count: Int, body: (Expr) -> Void) {
    let loopVar = ctx.useVariable(src: nodeId)
    let countLazy = ctx.useConstant(src: nodeId, value: Float(count))
    ops.append(UOp(op: .beginForLoop(loopVar, countLazy), value: loopVar))
    body(value(loopVar, scalarType: .int))
    ops.append(UOp(op: .endLoop, value: ctx.useVariable(src: nil)))
  }

  /// Emit a parallel range for tensor element iteration.
  /// All iterations are independent and the renderer decides execution strategy:
  /// C emits a sequential loop, Metal maps iterations to GPU threads.
  ///
  /// When in a frame-aware tensor block, the parallelism is already handled by flat threading
  /// (frameCount × tensorSize threads). In this case, skips emitting a loop and uses the
  /// pre-computed element index from the context.
  ///
  /// The `kind` parameter can be `.simd` to mark all emitted UOps for SIMD vectorization
  /// (4-wide increment, NEON intrinsics in C renderer).
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
    body(value(indexVar, scalarType: .int))
    ops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
    if case .simd = kind {
      ops = ops.map { UOp(op: $0.op, value: $0.value, kind: $0.kind, kindOverride: .simd) }
    }
  }

  /// Emit a 2D parallel map over (frame, bin) pairs.
  /// Total thread count = frameCount × `bins`. Decomposes the flat thread index into
  /// a frame index and bin index, overriding the kernel's frame index accordingly.
  /// The body receives (frameIndex, binIndex) as float-typed expressions.
  public func parallelMap2D(bins: Int, body: (Expr, Expr) -> Void) {
    setThreadCountScale(bins)
    let flatIdx = threadIndex()
    let binsExpr = constant(Float(bins))
    let frameIdx = floor(flatIdx / binsExpr)
    setFrameIndex(frameIdx)
    let binIdx = flatIdx - frameIdx * binsExpr
    body(frameIdx, binIdx)
  }

  // MARK: - Threading

  /// Emit the current GPU thread index (or loop iteration in the C renderer).
  /// Returns an int-typed expression.
  public func threadIndex() -> Expr {
    return emitIntOp(.threadIndex)
  }

  /// Return the current frame index, respecting frame-aware tensor block context.
  /// In frame-aware blocks, returns the pre-computed frame index from flat thread decomposition
  /// (flat index / tensor size). In normal blocks, falls back to the raw thread index.
  public func currentFrameIndex() -> Expr {
    if ctx.isInFrameAwareTensorBlock, let frameIdx = ctx.frameAwareTensorFrameIndex {
      return value(frameIdx, scalarType: .int)
    }
    // Fall back to thread index for normal blocks
    return threadIndex()
  }

  /// Emit the frame index UOp. If `setFrameIndex` was called earlier in this kernel,
  /// returns the overridden `_frameIndex` variable; otherwise falls back to the thread index.
  public func frameIndex() -> Expr {
    return emitIntOp(.frameIndex)
  }

  /// Return the runtime frame count as an expression. Uses the sentinel variable ID -1,
  /// which the renderer replaces with the actual frame count parameter.
  public func frameCount() -> Expr {
    return value(.variable(-1, nil))
  }

  /// Override the frame index used for output writes and gradient indexing in this kernel.
  /// Typically called after decomposing a flat thread index in scaled-thread kernels.
  public func setFrameIndex(_ expr: Expr) {
    let uop = UOp(op: .setFrameIndex(expr.lazy), value: expr.lazy)
    ops.append(uop)
  }

  /// Set the thread count scale factor for this kernel.
  /// The runtime dispatches `frameCount * scale` threads instead of just `frameCount`.
  /// Used for kernels that process multiple elements per frame (e.g., tensor ops, FFT bins).
  public func setThreadCountScale(_ scale: Int) {
    let uop = UOp(op: .setThreadCountScale(scale), value: .empty)
    ops.append(uop)
  }

  /// Decompose a flat thread index into (frameIndex, elementIndex) using integer arithmetic.
  /// Sets up thread count scaling and overrides the kernel's frame index.
  /// Returns both indices as int-typed expressions.
  ///
  /// This is the standard pattern for kernels that process `tensorSize` elements per frame:
  /// ```
  /// let (frameIdx, elemIdx) = b.setupFlatThreading(tensorSize: 64)
  /// // frameIdx = flatThreadIdx / 64
  /// // elemIdx  = flatThreadIdx % 64  (via subtraction)
  /// ```
  public func setupFlatThreading(tensorSize: Int) -> (frameIdx: Expr, elemIdx: Expr) {
    setThreadCountScale(tensorSize)
    let flatIdx = threadIndex()
    let sizeExpr = intConstant(tensorSize)
    let frameIdx = flatIdx / sizeExpr
    setFrameIndex(frameIdx)
    let elemIdx = flatIdx - frameIdx * sizeExpr
    return (frameIdx, elemIdx)
  }

  // MARK: - Mutation

  /// Reassign an existing mutable variable to a new value.
  /// The target must be a `.variable` — fatal errors otherwise.
  /// Emits a `.mutate` UOp that the renderer translates to a simple assignment.
  func mutate(_ target: Expr, to newValue: Expr) {
    guard case .variable(_, _) = target.lazy else {
      fatalError("Can only mutate variables")
    }
    let uop = UOp(op: .mutate(target.lazy, newValue.lazy), value: target.lazy)
    ops.append(uop)
  }
}

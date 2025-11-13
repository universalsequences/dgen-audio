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

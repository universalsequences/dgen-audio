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
    let uop = thunk(ctx,nil )
    ops.append(uop)
    return value(uop.value)
  }

  public func gswitch(_ cond: Expr, _ then: Expr, _ els: Expr) -> Expr {
    let thunk = u_switch(cond.lazy
                        , then.lazy, els.lazy);
    let uop = thunk(ctx,nil )
    ops.append(uop)
    return value(uop.value)
  }

  func mutate(_ target: Expr, to newValue: Expr) {
    guard case .variable(let id, _) = target.lazy else {
      fatalError("Can only mutate variables")
    }
    let uop = UOp(op: .mutate(target.lazy, newValue.lazy), value: target.lazy)
    ops.append(uop)
  }

  public func `if`(_ cond: Expr, then: () -> Void) {
    ops.append(u_begin_if(cond.lazy)(ctx, nodeId))
    then()
    ops.append(u_end_if()(ctx,nil ))
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
    let thunk = u_add(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func / (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_div(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func * (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_mul(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  static func - (lhs: Expr, rhs: Expr) -> Expr {
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

  static func < (lhs: Expr, rhs: Expr) -> Expr {
    let thunk = u_lt(lhs.lazy, rhs.lazy)
    let uop = thunk(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }
}

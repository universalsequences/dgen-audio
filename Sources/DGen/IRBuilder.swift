import Darwin

public final class IRBuilder {
  public let ctx: IRContext
  public let nodeId: NodeID
  public var ops: [UOp] = []

  public init(ctx: IRContext, nodeId: NodeID) {
    self.ctx = ctx
    self.nodeId = nodeId
  }

  public func value(_ lazy: Lazy, scalarType: CastType = .float) -> Expr {
    return Expr(lazy, ctx: ctx, nodeId: nodeId, builder: self, scalarType: scalarType)
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
    return value(l, scalarType: .float)
  }

  /// Alias for `int(_:)` with a more descriptive name for use in operator emit code.
  public func intConstant(_ v: Int) -> Expr {
    return int(v)
  }

  /// Create an integer constant expression (for index calculations)
  public func int(_ value: Int) -> Expr {
    let constLazy = ctx.useConstant(src: nodeId, value: Float(value))
    return Expr(constLazy, ctx: ctx, nodeId: nodeId, builder: self, scalarType: .int)
  }

  public func use(val: Expr) {
    ctx.values[nodeId] = val.lazy
  }

  public func cast(_ expr: Expr, to type: CastType) -> Expr {
    // If already the right type, skip the cast
    if expr.scalarType == type { return expr }
    let dest = ctx.useVariable(src: nodeId)
    var uop = UOp(op: .cast(expr.lazy, type), value: dest)
    uop.scalarType = type
    ops.append(uop)
    return value(dest, scalarType: type)
  }

  // MARK: - Type Helpers

  /// Emit an int-typed UOp (frameIndex, threadIndex, etc.) and return an int Expr
  func emitIntOp(_ op: Op, src: NodeID? = nil) -> Expr {
    let dest = ctx.useVariable(src: src ?? nodeId)
    var uop = UOp(op: op, value: dest)
    uop.scalarType = .int
    ops.append(uop)
    return value(dest, scalarType: .int)
  }

  /// Emit a binary UOp with int/float type propagation: int op int -> int, otherwise float.
  func emitTypedBinaryOp(_ op: Op, _ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let resultType: CastType = (a.scalarType == .int && b.scalarType == .int) ? .int : .float
    var uop = UOp(op: op, value: dest)
    uop.scalarType = resultType
    ops.append(uop)
    return value(dest, scalarType: resultType)
  }

  public func add(_ a: Expr, _ b: Expr) -> Expr {
    return emitTypedBinaryOp(.add(a.lazy, b.lazy), a, b)
  }

  public func mul(_ a: Expr, _ b: Expr) -> Expr {
    return emitTypedBinaryOp(.mul(a.lazy, b.lazy), a, b)
  }

  public func div(_ a: Expr, _ b: Expr) -> Expr {
    return emitTypedBinaryOp(.div(a.lazy, b.lazy), a, b)
  }

  /// Floor division: floor(a / b)
  /// Use this for integer index calculations where we need truncation toward negative infinity.
  public func floorDiv(_ a: Expr, _ b: Expr) -> Expr {
    let quotient = div(a, b)
    // Int division truncates automatically — floor is redundant
    if a.scalarType == .int && b.scalarType == .int { return quotient }
    return floor(quotient)
  }

  // MARK: - Mutable Variable Declarations

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

  func frameIndex(_ nodeId: NodeID) -> Expr {
    return emitIntOp(.frameIndex, src: nodeId)
  }
}

public struct Expr {
  public let lazy: Lazy
  public let scalarType: CastType
  private let ctx: IRContext
  private let nodeId: NodeID
  private unowned let builder: IRBuilder

  init(
    _ lazy: Lazy, ctx: IRContext, nodeId: NodeID, builder: IRBuilder, scalarType: CastType = .float
  ) {
    self.lazy = lazy
    self.ctx = ctx
    self.nodeId = nodeId
    self.builder = builder
    self.scalarType = scalarType
  }

  /// Compute the result scalarType: int op int -> int, otherwise float
  private static func promotedType(_ lhs: Expr, _ rhs: Expr) -> CastType {
    return (lhs.scalarType == .int && rhs.scalarType == .int) ? .int : .float
  }

  /// Emit a binary UOp with type promotion, returning a typed Expr
  private static func emitBinaryOp(
    _ lhs: Expr, _ rhs: Expr,
    thunk: (Lazy, Lazy) -> (IRContext, NodeID?) -> UOp
  ) -> Expr {
    let resultType = promotedType(lhs, rhs)
    var uop = thunk(lhs.lazy, rhs.lazy)(lhs.ctx, nil)
    uop.scalarType = resultType
    lhs.builder.ops.append(uop)
    return Expr(
      uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder, scalarType: resultType)
  }

  /// Emit a comparison UOp (always returns float, no type promotion)
  private static func emitComparisonOp(
    _ lhs: Expr, _ rhs: Expr,
    thunk: (Lazy, Lazy) -> (IRContext, NodeID?) -> UOp
  ) -> Expr {
    let uop = thunk(lhs.lazy, rhs.lazy)(lhs.ctx, nil)
    lhs.builder.ops.append(uop)
    return Expr(uop.value, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  /// The constant value of this expression, or nil if not a compile-time constant.
  private var constantValue: Float? {
    if case .constant(_, let v) = lazy { return v }
    return nil
  }

  /// Try constant folding for a binary op; returns nil if either operand is not a constant.
  /// When both operands are int-typed, the result is truncated to match GPU int arithmetic.
  private static func foldConstants(
    _ lhs: Expr, _ rhs: Expr, op: (Float, Float) -> Float
  ) -> Expr? {
    guard case .constant(_, let lval) = lhs.lazy,
      case .constant(_, let rval) = rhs.lazy
    else { return nil }
    let resultType = promotedType(lhs, rhs)
    var result = op(lval, rval)
    if resultType == .int { result = Float(Int(result)) }
    let folded = lhs.ctx.useConstant(src: lhs.nodeId, value: result)
    return Expr(
      folded, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder, scalarType: resultType)
  }

  /// Try constant folding for a comparison op; returns nil if either is not a constant.
  /// Evaluates to 1.0 (true) or 0.0 (false) as a float-typed constant.
  private static func foldComparison(
    _ lhs: Expr, _ rhs: Expr, op: (Float, Float) -> Bool
  ) -> Expr? {
    guard let lval = lhs.constantValue, let rval = rhs.constantValue else { return nil }
    let result: Float = op(lval, rval) ? 1.0 : 0.0
    let folded = lhs.ctx.useConstant(src: lhs.nodeId, value: result)
    return Expr(folded, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder)
  }

  /// Create a typed constant using the promoted type of two operands.
  private static func makeTypedConstant(_ value: Float, _ lhs: Expr, _ rhs: Expr) -> Expr {
    let resultType = promotedType(lhs, rhs)
    let c = lhs.ctx.useConstant(src: lhs.nodeId, value: value)
    return Expr(c, ctx: lhs.ctx, nodeId: lhs.nodeId, builder: lhs.builder, scalarType: resultType)
  }

  // MARK: - Arithmetic Operators (with constant folding and identity elimination)

  /// Add. Folds constants, eliminates `x + 0` and `0 + x`.
  static func + (lhs: Expr, rhs: Expr) -> Expr {
    if let folded = foldConstants(lhs, rhs, op: +) { return folded }
    if rhs.constantValue == 0, promotedType(lhs, rhs) == lhs.scalarType { return lhs }
    if lhs.constantValue == 0, promotedType(lhs, rhs) == rhs.scalarType { return rhs }
    return emitBinaryOp(lhs, rhs, thunk: u_add)
  }

  /// Subtract. Folds constants, eliminates `x - 0`.
  static func - (lhs: Expr, rhs: Expr) -> Expr {
    if let folded = foldConstants(lhs, rhs, op: -) { return folded }
    if rhs.constantValue == 0, promotedType(lhs, rhs) == lhs.scalarType { return lhs }
    return emitBinaryOp(lhs, rhs, thunk: u_sub)
  }

  /// Multiply. Folds constants, eliminates `x * 1` / `1 * x`, and `x * 0` / `0 * x`.
  static func * (lhs: Expr, rhs: Expr) -> Expr {
    if let folded = foldConstants(lhs, rhs, op: *) { return folded }
    if rhs.constantValue == 1, promotedType(lhs, rhs) == lhs.scalarType { return lhs }
    if lhs.constantValue == 1, promotedType(lhs, rhs) == rhs.scalarType { return rhs }
    if rhs.constantValue == 0 || lhs.constantValue == 0 {
      return makeTypedConstant(0, lhs, rhs)
    }
    return emitBinaryOp(lhs, rhs, thunk: u_mul)
  }

  /// Divide. Folds constants, eliminates `x / 1`.
  static func / (lhs: Expr, rhs: Expr) -> Expr {
    if let folded = foldConstants(lhs, rhs, op: /) { return folded }
    if rhs.constantValue == 1, promotedType(lhs, rhs) == lhs.scalarType { return lhs }
    return emitBinaryOp(lhs, rhs, thunk: u_div)
  }

  // MARK: - Comparison Operators (with constant folding)

  static func > (lhs: Expr, rhs: Expr) -> Expr {
    return foldComparison(lhs, rhs, op: >) ?? emitComparisonOp(lhs, rhs, thunk: u_gt)
  }

  static func >= (lhs: Expr, rhs: Expr) -> Expr {
    return foldComparison(lhs, rhs, op: >=) ?? emitComparisonOp(lhs, rhs, thunk: u_gte)
  }

  static func <= (lhs: Expr, rhs: Expr) -> Expr {
    return foldComparison(lhs, rhs, op: <=) ?? emitComparisonOp(lhs, rhs, thunk: u_lte)
  }

  static func < (lhs: Expr, rhs: Expr) -> Expr {
    return foldComparison(lhs, rhs, op: <) ?? emitComparisonOp(lhs, rhs, thunk: u_lt)
  }

  static func == (lhs: Expr, rhs: Expr) -> Expr {
    return foldComparison(lhs, rhs, op: ==) ?? emitComparisonOp(lhs, rhs, thunk: u_eq)
  }

  static func % (lhs: Expr, rhs: Expr) -> Expr {
    return foldConstants(lhs, rhs, op: { fmodf($0, $1) }) ?? emitBinaryOp(lhs, rhs, thunk: u_mod)
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

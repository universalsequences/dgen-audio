import Darwin

/// IRBuilder extension for math operations.
/// Unary: abs, sign, sin, cos, tan, tanh, exp, log, log10, sqrt, neg, floor, ceil, round.
/// Binary: pow, atan2, and, or, xor, min, max, mod, mix.
/// Other: selector, noise, applyUOp.
///
/// Most math ops constant-fold: when the input(s) are compile-time constants,
/// the result is computed at compile time and no UOp is emitted.

extension IRBuilder {

  // MARK: - Unary Math

  /// Emit absolute value. Constant-folds when the input is a known constant.
  public func abs(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(fabs(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .abs(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit sign function: returns -1, 0, or 1. Constant-folds.
  public func sign(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy {
      return constant(c > 0 ? 1 : c < 0 ? -1 : 0)
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sign(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit sine. Constant-folds when the input is a known constant.
  public func sin(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(sinf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sin(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit cosine. Constant-folds when the input is a known constant.
  public func cos(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(cosf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .cos(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit tangent. Constant-folds.
  public func tan(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(tanf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .tan(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit hyperbolic tangent. Constant-folds.
  public func tanh(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(tanhf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .tanh(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit natural exponential (e^x). Constant-folds.
  public func exp(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(expf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .exp(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit an arbitrary `Op`, allocating a fresh destination variable.
  /// Used by `log`, `log10`, and the bitwise ops to avoid boilerplate.
  public func applyUOp(op: Op) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: op, value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit natural logarithm (ln). Constant-folds.
  public func log(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(logf(c)) }
    return applyUOp(op: .log(val.lazy))
  }

  /// Emit base-10 logarithm. Constant-folds.
  public func log10(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(log10f(c)) }
    return applyUOp(op: .log10(val.lazy))
  }

  /// Emit square root. Constant-folds.
  public func sqrt(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(sqrtf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sqrt(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit negation (val × -1). Constant-folds.
  public func neg(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(-c) }
    let dest = ctx.useVariable(src: nodeId)
    let negativeConstant = constant(-1)
    let uop = UOp(op: .mul(val.lazy, negativeConstant.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit floor (round toward negative infinity). Constant-folds.
  /// Not needed for int-typed expressions — use `floorDiv` for integer index math.
  public func floor(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(floorf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .floor(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit ceiling (round toward positive infinity). Constant-folds.
  public func ceil(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(ceilf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .ceil(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit rounding to nearest integer (half-away-from-zero). Constant-folds.
  public func round(_ val: Expr) -> Expr {
    if case .constant(_, let c) = val.lazy { return constant(roundf(c)) }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .round(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  // MARK: - Binary Math

  /// Emit exponentiation: `base` raised to `exponent`. Constant-folds.
  public func pow(_ base: Expr, _ exponent: Expr) -> Expr {
    if case .constant(_, let b) = base.lazy, case .constant(_, let e) = exponent.lazy {
      return constant(powf(b, e))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .pow(base.lazy, exponent.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit two-argument arctangent: angle from x-axis to point (x, y). Constant-folds.
  public func atan2(_ y: Expr, _ x: Expr) -> Expr {
    if case .constant(_, let yc) = y.lazy, case .constant(_, let xc) = x.lazy {
      return constant(atan2f(yc, xc))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .atan2(y.lazy, x.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit bitwise AND.
  public func and(_ a: Expr, _ b: Expr) -> Expr {
    return applyUOp(op: .and(a.lazy, b.lazy))
  }

  /// Emit bitwise OR.
  public func or(_ a: Expr, _ b: Expr) -> Expr {
    return applyUOp(op: .or(a.lazy, b.lazy))
  }

  /// Emit bitwise XOR.
  public func xor(_ a: Expr, _ b: Expr) -> Expr {
    return applyUOp(op: .xor(a.lazy, b.lazy))
  }

  /// Emit component-wise minimum. Constant-folds.
  public func min(_ a: Expr, _ b: Expr) -> Expr {
    if case .constant(_, let av) = a.lazy, case .constant(_, let bv) = b.lazy {
      return constant(fminf(av, bv))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .min(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit component-wise maximum. Constant-folds.
  public func max(_ a: Expr, _ b: Expr) -> Expr {
    if case .constant(_, let av) = a.lazy, case .constant(_, let bv) = b.lazy {
      return constant(fmaxf(av, bv))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .max(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit modulo (remainder after division). Propagates int/float types via `emitTypedBinaryOp`.
  public func mod(_ a: Expr, _ b: Expr) -> Expr {
    return emitTypedBinaryOp(.mod(a.lazy, b.lazy), a, b)
  }

  /// Emit linear interpolation: `mix(a, b, t) = a * (1 - t) + b * t`.
  public func mix(_ a: Expr, _ b: Expr, _ t: Expr) -> Expr {
    return u_mix(a, b, lerp: t)(self)
  }

  // MARK: - Selector & Noise

  /// Emit a multi-way selector indexed by `mode`.
  /// When `mode` is a known constant, statically resolves to the matching option (1-indexed).
  /// Returns 0 if `mode <= 0` or exceeds the number of options.
  public func selector(_ mode: Expr, _ options: [Expr]) -> Expr {
    if case .constant(_, let constMode) = mode.lazy {
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

  /// Emit a pseudo-random noise value. The `cellId` provides per-cell state for the RNG.
  public func noise(_ cellId: CellID) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .noise(cellId), value: dest)
    ops.append(uop)
    return value(dest)
  }
}

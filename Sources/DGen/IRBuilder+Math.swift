import Darwin

/// IRBuilder extension for math operations.
/// Unary: abs, sign, sin, cos, tan, tanh, exp, log, log10, sqrt, neg, floor, ceil, round.
/// Binary: pow, atan2, and, or, xor, min, max, mod, mix.
/// Other: selector, noise, applyUOp.

extension IRBuilder {

  // MARK: - Unary Math

  /// Emit absolute value. Constant-folds when the input is a known constant.
  public func abs(_ val: Expr) -> Expr {
    if case .constant(_, let absConst) = val.lazy {
      return self.constant(fabs(absConst))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .abs(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit sign function: returns -1, 0, or 1 depending on the sign of `val`.
  public func sign(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sign(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit sine. Constant-folds when the input is a known constant.
  public func sin(_ val: Expr) -> Expr {
    if case .constant(_, let sinConst) = val.lazy {
      return self.constant(sinf(sinConst))
    }

    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sin(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit cosine. Constant-folds when the input is a known constant.
  public func cos(_ val: Expr) -> Expr {
    if case .constant(_, let cosConst) = val.lazy {
      return self.constant(cosf(cosConst))
    }
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .cos(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit tangent.
  public func tan(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .tan(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit hyperbolic tangent. Commonly used as an activation function in neural networks.
  public func tanh(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .tanh(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit natural exponential (e^x).
  public func exp(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .exp(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit an arbitrary unary/binary `Op`, allocating a fresh destination variable.
  /// Used by `log`, `log10`, and the bitwise ops to avoid boilerplate.
  public func applyUOp(op: Op) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: op, value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit natural logarithm (ln).
  public func log(_ val: Expr) -> Expr {
    return applyUOp(op: .log(val.lazy))
  }

  /// Emit base-10 logarithm.
  public func log10(_ val: Expr) -> Expr {
    return applyUOp(op: .log10(val.lazy))
  }

  /// Emit square root.
  public func sqrt(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .sqrt(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit negation. Implemented as multiplication by -1.
  public func neg(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let negativeConstant = constant(-1)
    let uop = UOp(op: .mul(val.lazy, negativeConstant.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit floor (round toward negative infinity).
  /// Not needed for int-typed expressions — use `floorDiv` for integer index math.
  public func floor(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .floor(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit ceiling (round toward positive infinity).
  public func ceil(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .ceil(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit rounding to nearest integer (half-away-from-zero).
  public func round(_ val: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .round(val.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  // MARK: - Binary Math

  /// Emit exponentiation: `base` raised to `exponent`.
  public func pow(_ base: Expr, _ exponent: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .pow(base.lazy, exponent.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit two-argument arctangent: angle in radians from the x-axis to the point (x, y).
  public func atan2(_ y: Expr, _ x: Expr) -> Expr {
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

  /// Emit component-wise minimum of two values.
  public func min(_ a: Expr, _ b: Expr) -> Expr {
    let dest = ctx.useVariable(src: nodeId)
    let uop = UOp(op: .min(a.lazy, b.lazy), value: dest)
    ops.append(uop)
    return value(dest)
  }

  /// Emit component-wise maximum of two values.
  public func max(_ a: Expr, _ b: Expr) -> Expr {
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

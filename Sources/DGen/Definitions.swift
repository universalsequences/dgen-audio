func binaryOp(
  _ opConstructor: @escaping (Lazy, Lazy) -> Op
) -> (Lazy, Lazy) -> (IRContext, NodeID?) -> UOp {
  return { a, b in
    return { ctx, nodeId in
      let dest = ctx.useVariable(src: nodeId)
      let op = opConstructor(a, b)
      return UOp(op: op, value: dest)
    }
  }
}

func u_begin_if(_ cond: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, nodeId in
    return UOp(op: .beginIf(cond), value: ctx.useVariable(src: nil))
  }
}

func u_end_if() -> (IRContext, NodeID?) -> UOp {
  return { ctx, nodeId in
    return UOp(op: .endIf, value: ctx.useVariable(src: nil))
  }
}

func u_switch(_ cond: Lazy, _ then: Lazy, _ els: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, nodeId in
    return UOp(op: .gswitch(cond, then, els), value: ctx.useVariable(src: nodeId))
  }
}

func u_load(_ cellId: CellID) -> (IRContext, NodeID?) -> UOp {
  return { ctx, nodeId in
    let dest = ctx.useVariable(src: nodeId)
    return UOp(op: .load(cellId), value: dest)
  }
}

func u_store(_ cellId: CellID, _ value: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, _ in
    return UOp(op: .store(cellId, value), value: ctx.useVariable(src: nil))
  }
}

func u_output(_ channelNumber: ChannelNumber, _ value: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, _ in
    return UOp(op: .output(channelNumber, value), value: ctx.useVariable(src: nil))
  }
}

func u_input(_ channelNumber: ChannelNumber) -> (IRContext, NodeID?) -> UOp {
  return { ctx, _ in
    return UOp(op: .input(channelNumber), value: ctx.useVariable(src: nil))
  }
}

func u_delay1(_ cell: CellID, _ a: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, _ in
    return UOp(op: .delay1(cell, a), value: ctx.useVariable(src: nil))
  }
}

func u_concatShift(_ a: Lazy, _ b: Lazy, _ shift: Int) -> (IRContext, NodeID?) -> UOp {
  return { ctx, _ in
    return UOp(op: .concatShift(a, b, shift), value: ctx.useVariable(src: nil))
  }
}

func u_floor(_ value: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, nodeId in
    let dest = ctx.useVariable(src: nodeId)
    return UOp(op: .floor(value), value: dest)
  }
}

func u_memoryRead(_ cellId: CellID, _ offset: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, nodeId in
    let dest = ctx.useVariable(src: nodeId)
    return UOp(op: .memoryRead(cellId, offset), value: dest)
  }
}

func u_memoryWrite(_ cellId: CellID, _ offset: Lazy, _ value: Lazy) -> (IRContext, NodeID?) -> UOp {
  return { ctx, _ in
    return UOp(op: .memoryWrite(cellId, offset, value), value: ctx.useVariable(src: nil))
  }
}

let u_gt = binaryOp(Op.gt)
let u_gte = binaryOp(Op.gte)
let u_lte = binaryOp(Op.lte)
let u_lt = binaryOp(Op.lt)
let u_mod = binaryOp(Op.mod)
let u_eq = binaryOp(Op.eq)
let u_add = binaryOp(Op.add)
let u_div = binaryOp(Op.div)
let u_mul = binaryOp(Op.mul)
let u_sub = binaryOp(Op.sub)
let u_pow = binaryOp(Op.pow)
let u_min = binaryOp(Op.min)
let u_max = binaryOp(Op.max)

func u_phasor(_ cellId: CellID, freq: Expr, reset: Expr) -> (IRBuilder) -> Expr {
  return { b in
    let b_sr = b.constant(b.ctx.g.sampleRate)
    return u_accum(
      cellId, incr: freq / b_sr, reset: reset, min: b.constant(0), max: b.constant(1))(b)
  }
}

func u_latch(_ cellId: CellID, value: Expr, cond: Expr) -> (IRBuilder) -> Expr {
  return { b in
    let latched = b.load(cellId)
    b.if(cond > b.constant(0)) {
      _ = b.store(cellId, value)
      b.mutate(latched, to: value)
    }
    return latched
  }
}

func u_mix(_ x: Expr, _ y: Expr, lerp: Expr) -> (IRBuilder) -> Expr {
  return { b in
    let oneMinusT = b.constant(1.0) - lerp
    let xScaled = x * oneMinusT
    let yScaled = y * lerp
    let mixed = xScaled + yScaled
    return mixed
  }
}

/// Computes spectral loss: measures how different two signals are in frequency space.
///
/// Uses a sliding window DFT (Discrete Fourier Transform) to convert time-domain samples into
/// frequency magnitudes. For each frequency bin k, computes real/imaginary components via:
///   real_k = Σ sample[n] * cos(-2π*k*n/windowSize)
///   imag_k = Σ sample[n] * sin(-2π*k*n/windowSize)
/// Then computes per-bin squared error (mag1_k - mag2_k)² and writes to scratch.
/// Pass2 reduces those per-bin errors to a scalar loss.
///
/// Better than MSE for audio: invariant to small time shifts, captures perceptual differences.
func u_spectralLoss(sig1: Expr, sig2: Expr, windowSize: Int, scratchCell: CellID) -> (IRBuilder) -> Expr {
  return { b in
    let numBins = windowSize / 2 + 1
    let winSize = b.constant(Float(windowSize))
    let frameIdx = b.threadIndex()
    let baseOffset = frameIdx * winSize * b.constant(2.0)

    // For each frequency bin
    b.parallelRange(numBins) { binIndex in
      let binIndexFloat = b.cast(binIndex, to: .float)

      // DFT accumulators (real and imaginary parts)
      let real1 = b.float(0.0)
      let real2 = b.float(0.0)
      let imag1 = b.float(0.0)
      let imag2 = b.float(0.0)

      // Sum over window samples
      b.loop(windowSize) { n in
        let j = frameIdx - (winSize - b.constant(1.0)) + b.cast(n, to: .float)

        // Load samples from tape with bounds checking
        let s1 = b.tapeLoad(sig1, at: j)
        let s2 = b.tapeLoad(sig2, at: j)

        // DFT basis: e^(-2πi*k*n/N) = cos(angle) - i*sin(angle)
        let nFloat = b.cast(n, to: .float)
        let angle = b.constant(-2.0) * b.pi * binIndexFloat * nFloat / winSize

        let c = b.cos(angle)
        let s = b.sin(angle)

        // Accumulate DFT: Real(X[k]) += x[n]*cos, Imag(X[k]) += x[n]*sin
        real1.accumulate(s1 * c)
        imag1.accumulate(s1 * s)
        real2.accumulate(s2 * c)
        imag2.accumulate(s2 * s)
      }

      // Magnitude: |X[k]| = sqrt(Real² + Imag²)
      let mag1 = b.sqrt(real1.value * real1.value + imag1.value * imag1.value)
      let mag2 = b.sqrt(real2.value * real2.value + imag2.value * imag2.value)

      // Accumulate squared error for this bin
      let diff = mag1 - mag2
      let binError = diff * diff
      let offset = baseOffset + binIndexFloat
      _ = b.memoryWrite(scratchCell, b.cast(offset, to: .int), binError)
    }
    // Pass2 will reduce from scratch to a scalar loss.
    return b.constant(0.0)
  }
}

func u_historyWrite(cellId: CellID, _ curr: Expr) -> (IRBuilder) -> Expr {
  return { b in
    return b.delay1(cellId, curr)
  }
}

func u_click(_ cellId: CellID) -> (IRBuilder) -> Expr {
  return { b in
    let trig = b.load(cellId, b.nodeId)
    let zero = b.constant(0.0)
    _ = b.store(cellId, zero)
    return trig
  }
}

func u_noise(_ cellId: CellID) -> (IRBuilder) -> Expr {
  return { b in
    b.noise(cellId)
  }
}

func u_mse(_ a: Expr, _ b: Expr) -> (IRBuilder) -> Expr {
  return { builder in
    // MSE = (a - b)^2
    let diff = a - b
    let squared = diff * diff
    return squared
  }
}

func u_accum(_ cellId: CellID, incr: Expr, reset: Expr, min: Expr, max: Expr) -> (IRBuilder) -> Expr
{
  return { b in
    let acc = b.load(cellId, b.nodeId)
    let zero = b.constant(0.0)
    let span = max - min
    let incrAbs = incr

    let nextCand = acc + incrAbs
    let next = b.gswitch(reset > zero, min, nextCand)

    // Base modulo (pure expr)
    let rel = next - min
    let k = b.floor(rel / span)
    let wBase = next - (k * span)

    // 1) store base
    _ = b.store(cellId, wBase)

    // 2) correct rounding: if (wBase >= max) store(wBase - span)
    b.if(wBase >= max) {
      _ = b.store(cellId, wBase - span)
    }

    // 3) reset override last: if (reset > 0) store(min)
    b.if(reset > zero) {
      _ = b.store(cellId, min)
    }

    return acc
  }
}

/// Pass B: Reduce from memory to gradients.
/// Each thread j (sample index) gathers contributions from all windows that include sample j.
func u_spectralLossBackwardPass2(
  _ windowSize: Int,
  _ scratchCell: CellID,
  _ sig1: Expr,
  _ sig2: Expr,
  _ upstreamGrad: Expr,
  _ gradId1: GradID,
  _ gradId2: GradID
) -> (IRBuilder) -> (Expr, Expr) {
  return { b in
    let j = b.threadIndex()  // Sample index
    let grad1 = b.float(0.0)
    let grad2 = b.float(0.0)
    let frameCount = b.value(.variable(-1, nil))

    // Gather from all windows that include sample j
    // Windows ending at i ∈ [j .. j+(windowSize-1)]
    b.loop(windowSize) { offsetFromJ in
      let windowEnd = j + b.cast(offsetFromJ, to: .float)  // i

      // Window offset: n = j - i + (windowSize - 1)
      let winSize = b.constant(Float(windowSize))
      let n = j - windowEnd + (winSize - b.constant(1.0))

      // Only read scratch for valid window ends
      b.if(windowEnd < frameCount) {
        // Memory offset: memory[scratchCell + (i * windowSize * 2) + (n * 2) + component]
        let offset1 = windowEnd * winSize * b.constant(2.0) + n * b.constant(2.0)
        let offset2 = offset1 + b.constant(1.0)

        // Read contributions from memory
        let contrib1 = b.memoryRead(scratchCell, b.cast(offset1, to: .int))
        let contrib2 = b.memoryRead(scratchCell, b.cast(offset2, to: .int))

        grad1.accumulate(contrib1)
        grad2.accumulate(contrib2)
      }
    }

    return (grad1.value, grad2.value)
  }
}

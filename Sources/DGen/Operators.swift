public typealias NodeID = Int
public typealias VarID = Int
public typealias ConstantID = Int
public typealias CellID = Int
public typealias GradID = Int
public typealias ChannelNumber = Int

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

func u_mse(_ a: Expr, _ b: Expr) -> (IRBuilder) -> Expr {
    return { builder in
        // MSE = (a - b)^2
        let diff = a - b
        let squared = diff * diff
        return squared
    }
}

/// Compute gradient of DFT magnitude with respect to a sample at given position
/// Formula: ∂mag/∂sample[n] = (real * cos(angle) + imag * sin(angle)) / mag
/// where angle = 2π * binIndex * samplePos / windowSize
func u_dftMagnitudeGradient(
    _ real: Expr, _ imag: Expr, _ mag: Expr,
    _ windowSize: Int, _ binIndex: Int, _ samplePos: Int
) -> (IRBuilder) -> Expr {
    return { b in
        // Compute angle = 2π * binIndex * samplePos / windowSize
        let pi = b.constant(Float.pi)
        let two = b.constant(2.0)
        let k = b.constant(Float(binIndex))
        let n = b.constant(Float(samplePos))
        let N = b.constant(Float(windowSize))

        let angle = (two * pi * k * n) / N

        // cos and sin of angle (negated for DFT convention)
        let negAngle = b.constant(0.0) - angle
        let cosAngle = b.cos(negAngle)
        let sinAngle = b.sin(negAngle)

        // Gradient: (real * cos + imag * sin) / mag
        // Note: real, imag, mag are already Expr, not Lazy
        let realPart = real * cosAngle
        let imagPart = imag * sinAngle
        let numerator = realPart + imagPart

        // Avoid division by zero
        let epsilon = b.constant(1e-8)
        let safeMag = mag + epsilon

        return numerator / safeMag
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

func u_phasor(_ cellId: CellID, freq: Expr, reset: Expr) -> (IRBuilder) -> Expr {
    return { b in
        let b_sr = b.constant(44100)
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
/// Then returns Σ(mag1_k - mag2_k)² across all bins, where mag = sqrt(real² + imag²).
///
/// Better than MSE for audio: invariant to small time shifts, captures perceptual differences.
func u_spectralLoss(sig1: Expr, sig2: Expr, windowSize: Int) -> (IRBuilder) -> Expr {
    return { b in
        let numBins = windowSize / 2 + 1
        let totalError = b.float(0.0)

        // For each frequency bin
        b.loop(numBins) { binIndex in
            // DFT accumulators (real and imaginary parts)
            let real1 = b.float(0.0)
            let real2 = b.float(0.0)
            let imag1 = b.float(0.0)
            let imag2 = b.float(0.0)

            // Sum over window samples
            b.loop(windowSize) { n in
                let idx = b.threadIndex()
                let winSize = b.constant(Float(windowSize))
                let j = idx - (winSize - b.constant(1.0)) + b.cast(n, to: .float)

                // Load samples from tape with bounds checking
                let s1 = b.tapeLoad(sig1, at: j)
                let s2 = b.tapeLoad(sig2, at: j)

                // DFT basis: e^(-2πi*k*n/N) = cos(angle) - i*sin(angle)
                let binIndexFloat = b.cast(binIndex, to: .float)
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
            totalError.accumulate(diff * diff)
        }

        return totalError.value
    }
}

public struct BackwardsEmitResult {
    let ops: [UOp]
    let dependencies: [NodeID]
}

// frontend
public enum LazyOp {
    case add, sub, div, mul, abs, sign, sin, cos, tan, tanh, exp, log, log10, sqrt, atan2, gt, gte,
        lte,
        lt, eq,
        gswitch, mix, pow, floor, ceil, round, mod, min, max
    case mse  // mean squared error per-sample: (a-b)^2
    case spectralLossTape(Int)  // DEPRECATED: old single-pass spectral loss
    case spectralLossPass1(Int, CellID)  // Pass 1: compute loss & store DFT contributions
    case spectralLossPass2(Int, CellID)  // Pass 2: reduce contributions to gradients (no-op in forward)
    case selector  // selector(mode, options[])
    case memoryRead(CellID)
    case memoryWrite(CellID)
    case scalarMemoryWrite(CellID)
    case historyWrite(CellID)
    case historyReadWrite(CellID)
    case param(CellID)
    case latch(CellID)
    case click(CellID)
    case historyRead(CellID)
    case phasor(CellID)
    case accum(CellID)
    case constant(Float)
    case output(Int)
    case input(Int)
    case seq  // Sequential execution - returns value of last input

    public func emit(ctx: IRContext, g: Graph, nodeId: NodeID) throws -> [UOp] {
        guard let node = g.nodes[nodeId] else { return [] }

        // collect operands
        let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
        var ops: [UOp] = []
        let b = IRBuilder(ctx: ctx, nodeId: nodeId)

        switch self {
        case .constant(let value):
            _ = ctx.useConstant(src: nodeId, value: value)
            return []

        case .add:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "add", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) + b.value(inputs[1]))
        case .sub:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "sub", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) - b.value(inputs[1]))
        case .mul:
            guard inputs.count == 2 else {
                print("mul failing \(node.id)")
                throw DGenError.insufficientInputs(
                    operator: "mul", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) * b.value(inputs[1]))
        case .div:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "div", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) / b.value(inputs[1]))
        case .mod:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "mod", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) % b.value(inputs[1]))
        case .min:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "min", expected: 2, actual: inputs.count)
            }
            b.use(val: b.min(b.value(inputs[0]), b.value(inputs[1])))
        case .max:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "max", expected: 2, actual: inputs.count)
            }
            b.use(val: b.max(b.value(inputs[0]), b.value(inputs[1])))
        case .abs:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "abs", expected: 1, actual: inputs.count)
            }
            b.use(val: b.abs(b.value(inputs[0])))
        case .sign:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "sign", expected: 1, actual: inputs.count)
            }
            b.use(val: b.sign(b.value(inputs[0])))
        case .sin:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "sin", expected: 1, actual: inputs.count)
            }
            b.use(val: b.sin(b.value(inputs[0])))
        case .cos:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "cos", expected: 1, actual: inputs.count)
            }
            b.use(val: b.cos(b.value(inputs[0])))
        case .tan:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "tan", expected: 1, actual: inputs.count)
            }
            b.use(val: b.tan(b.value(inputs[0])))
        case .tanh:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "tanh", expected: 1, actual: inputs.count)
            }
            b.use(val: b.tanh(b.value(inputs[0])))
        case .exp:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "exp", expected: 1, actual: inputs.count)
            }
            b.use(val: b.exp(b.value(inputs[0])))
        case .log:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "log", expected: 1, actual: inputs.count)
            }
            b.use(val: b.log(b.value(inputs[0])))
        case .log10:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "log10", expected: 1, actual: inputs.count)
            }
            b.use(val: b.log10(b.value(inputs[0])))
        case .sqrt:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "sqrt", expected: 1, actual: inputs.count)
            }
            b.use(val: b.sqrt(b.value(inputs[0])))
        case .pow:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "pow", expected: 2, actual: inputs.count)
            }
            b.use(val: b.pow(b.value(inputs[0]), b.value(inputs[1])))
        case .floor:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "floor", expected: 1, actual: inputs.count)
            }
            b.use(val: b.floor(b.value(inputs[0])))
        case .round:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "floor", expected: 1, actual: inputs.count)
            }
            b.use(val: b.round(b.value(inputs[0])))
        case .ceil:
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "ceil", expected: 1, actual: inputs.count)
            }
            b.use(val: b.ceil(b.value(inputs[0])))
        case .memoryRead(let cellId):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "memoryRead", expected: 1, actual: inputs.count)
            }
            b.use(val: b.memoryRead(cellId, b.value(inputs[0])))
        case .memoryWrite(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "memoryWrite", expected: 2, actual: inputs.count)
            }
            b.use(val: b.memoryWrite(cellId, b.value(inputs[0]), b.value(inputs[1])))
        case .scalarMemoryWrite(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "scalarMemoryWrite", expected: 2, actual: inputs.count)
            }
            // Force a scalar memory write by emitting a dedicated UOp
            let dest = ctx.useVariable(src: node.id)
            let uop = UOp(
                op: .scalarMemoryWrite(cellId, b.value(inputs[0]).lazy, b.value(inputs[1]).lazy),
                value: dest)
            b.ops.append(uop)
            b.use(val: b.value(dest))
        case .atan2:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "atan2", expected: 2, actual: inputs.count)
            }
            b.use(val: b.atan2(b.value(inputs[0]), b.value(inputs[1])))
        case .mse:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "mse", expected: 2, actual: inputs.count)
            }
            let (a, b2) = b.values(inputs, count: 2)
            b.use(val: u_mse(a, b2)(b))

        case let .spectralLossTape(windowSize):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossTape", expected: 2, actual: inputs.count)
            }
            let (sig1, sig2) = b.values(inputs, count: 2)
            b.use(val: u_spectralLoss(sig1: sig1, sig2: sig2, windowSize: windowSize)(b))

        case let .spectralLossPass1(windowSize, scratchCell):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossPass1", expected: 2, actual: inputs.count)
            }
            let (sig1, sig2) = b.values(inputs, count: 2)
            // Forward: compute spectral loss normally (Pass1 does the actual work)
            b.use(val: u_spectralLoss(sig1: sig1, sig2: sig2, windowSize: windowSize)(b))

        case let .spectralLossPass2(windowSize, scratchCell):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "spectralLossPass2", expected: 1, actual: inputs.count)
            }
            // Forward: no-op, just forward the value from Pass1
            let pass1Result = b.value(inputs[0])
            b.use(val: pass1Result)

        case .gt:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "gt", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) > b.value(inputs[1]))
        case .gte:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "gte", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) >= b.value(inputs[1]))
        case .lte:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "lte", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) <= b.value(inputs[1]))
        case .lt:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "lt", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) < b.value(inputs[1]))
        case .eq:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "eq", expected: 2, actual: inputs.count)
            }
            b.use(val: b.value(inputs[0]) == b.value(inputs[1]))
        case .gswitch:
            guard inputs.count == 3 else {
                throw DGenError.insufficientInputs(
                    operator: "gswitch", expected: 3, actual: inputs.count)
            }
            b.use(val: b.gswitch(b.value(inputs[0]), b.value(inputs[1]), b.value(inputs[2])))
        case .selector:
            guard inputs.count >= 2 else {
                throw DGenError.insufficientInputs(
                    operator: "selector", expected: 2, actual: inputs.count)
            }
            let mode = inputs[0]
            let options = Array(inputs.dropFirst())
            b.use(val: b.selector(b.value(mode), options.map { b.value($0) }))
        case .historyWrite(let cellId):
            // for simd its beyond just this -- we need to ensure that we shift the results 1
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "history write", expected: 1, actual: inputs.count)
            }
            b.use(val: b.store(cellId, b.value(inputs[0])))
        case .param(let cellId):
            b.use(val: b.load(cellId))
        case .historyReadWrite(let cellId):
            // for simd its beyond just this -- we need to ensure that we shift the results 1
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "history write", expected: 1, actual: inputs.count)
            }
            b.use(val: u_historyWrite(cellId: cellId, b.value(inputs[0]))(b))
        case .historyRead(let cellId):
            // no longer doing anything here TODO - remove
            guard inputs.count == 0 else {
                throw DGenError.insufficientInputs(
                    operator: "history read", expected: 0, actual: inputs.count)
            }
            b.use(val: b.load(cellId))
        case .latch(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "latch", expected: 2, actual: inputs.count)
            }
            let value = b.value(inputs[0])
            let cond = b.value(inputs[1])
            b.use(val: u_latch(cellId, value: value, cond: cond)(b))
        case .mix:
            guard inputs.count == 3 else {
                throw DGenError.insufficientInputs(
                    operator: "mix", expected: 3, actual: inputs.count)
            }
            let (x, y, t) = b.values(inputs, count: 3)
            b.use(val: u_mix(x, y, lerp: t)(b))
        case .accum(let cellId):
            guard inputs.count == 4 else {
                throw DGenError.insufficientInputs(
                    operator: "accum", expected: 4, actual: inputs.count)
            }
            let (incr, reset, min, max) = b.values(inputs, count: 4)
            b.use(val: u_accum(cellId, incr: incr, reset: reset, min: min, max: max)(b))
        case .click(let cellId):
            guard inputs.count == 0 else {
                throw DGenError.insufficientInputs(
                    operator: "click", expected: 4, actual: inputs.count)
            }
            b.use(val: u_click(cellId)(b))
        case .phasor(let cellId):
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "phasor", expected: 2, actual: inputs.count)
            }
            let (freq, reset) = b.values(inputs, count: 2)
            b.use(val: u_phasor(cellId, freq: freq, reset: reset)(b))
        case .output(let outputNumber):
            guard inputs.count == 1 else {
                throw DGenError.insufficientInputs(
                    operator: "output", expected: 1, actual: inputs.count)
            }
            b.use(val: b.output(outputNumber, b.value(inputs[0])))
        case .input(let inputNumber):
            b.use(val: b.input(inputNumber))
        case .seq:
            // Sequential execution - assumes all input nodes have been emitted in topological order
            // This operator simply returns the value of the last input
            guard node.inputs.count >= 2 else {
                throw DGenError.insufficientInputs(
                    operator: "seq", expected: 2, actual: node.inputs.count)
            }

            // Use the value of the last input node (all inputs should already be emitted)
            if let lastInputId = node.inputs.last,
                let lastValue = ctx.values[lastInputId]
            {
                b.use(val: b.value(lastValue))
            } else {
                throw DGenError.insufficientInputs(
                    operator: "seq", expected: node.inputs.count,
                    actual: node.inputs.compactMap { ctx.values[$0] }.count)
            }
        }
        ops.append(contentsOf: b.ops)
        return ops
    }

    func emitBackward(ctx: IRContext, g: Graph, nodeId: NodeID)
        -> [UOp]
    {
        guard let node = g.nodes[nodeId] else {
            return []
        }

        var ops: [UOp] = []
        let b = IRBuilder(ctx: ctx, nodeId: nodeId)

        // we'll assume that gradient seeds are set to 1 before running an epoch, and everything else is 0
        var gradOutput = ctx.useConstant(src: nil, value: 1.0)
        if let gradCellId = ctx.gradients[nodeId] {
            gradOutput = b.loadGrad(gradCellId).lazy
        } else {
            let gradCellId = ctx.useGradient(src: nodeId, seed: true)
            gradOutput = b.loadGrad(gradCellId).lazy
        }

        // Collect operands
        let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }

        switch self {
        case .constant(_):
            // Constants have no gradients to propagate
            // should we return 0 so that it can actually propagate?
            // constant is a leaf so it'd be at the very end anyway so probably fine
            return []
        case .click:
            // TODO - implement backprop for click
            break
        case .add:
            // d(x+y)/dx = 1, d(x+y)/dy = 1
            guard node.inputs.count == 2 else { fatalError("add requires 2 inputs") }
            b.grad(node.inputs[0], value: gradOutput)
            b.grad(node.inputs[1], value: gradOutput)
        case .sub:
            // z = x - y  =>  dz/dx = 1, dz/dy = -1
            guard node.inputs.count == 2 else { fatalError("sub requires 2 inputs") }
            b.grad(node.inputs[0], value: gradOutput)
            // negate grad for second input
            let negGrad = (b.constant(0.0) - b.value(gradOutput)).lazy
            b.grad(node.inputs[1], value: negGrad)
        case .mod:
            // z = fmod(a, b) ≈ a - b*trunc(a/b). Treat trunc grad as 0 almost everywhere.
            // -> dz/da ≈ 1, dz/db ≈ 0 (stable choice for DSP wrapping)
            guard node.inputs.count == 2 else { fatalError("mod requires 2 inputs") }
            b.grad(node.inputs[0], value: gradOutput)
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))
        case .param(_):
            // the canonical gradient for the param already lives in gradients, in the row for that param’s gradId.
            // There’s nothing left to push. It’s a leaf.
            break
        case .min:
            // d(min(x,y))/dx = (x <= y) ? 1 : 0, d(min(x,y))/dy = (y < x) ? 1 : 0
            guard inputs.count == 2 else { fatalError("min requires 2 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let xIsMin = x <= y
            let yIsMin = y < x
            let gradX = b.gswitch(xIsMin, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(yIsMin, b.value(gradOutput), b.constant(0.0))
            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
        case .max:
            // d(max(x,y))/dx = (x >= y) ? 1 : 0, d(max(x,y))/dy = (y > x) ? 1 : 0
            guard inputs.count == 2 else { fatalError("max requires 2 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let xIsMax = x >= y
            let yIsMax = y > x
            let gradX = b.gswitch(xIsMax, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(yIsMax, b.value(gradOutput), b.constant(0.0))
            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
        case .mul:
            // d(x*y)/dx = y, d(x*y)/dy = x
            guard inputs.count == 2 else { fatalError("mul \(node.id) requires 2 inputs") }
            let rhs = b.tapeValue(node.inputs[1])
            let lhs = b.tapeValue(node.inputs[0])
            let gradX = b.value(gradOutput) * rhs
            let gradY = b.value(gradOutput) * lhs
            print(
                "GRAD CALLED FOR MUL with inputs[0]=\(node.inputs[0]) inputs[1]=\(node.inputs[1])")
            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
        case .div:
            // z = x / y  =>  dz/dx = 1/y, dz/dy = -x / y^2
            guard inputs.count == 2 else { fatalError("div \(node.id) requires 2 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let gradX = b.value(gradOutput) / y
            let gradY = (b.constant(0.0) - b.value(gradOutput)) * (x / (y * y))
            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
        case .abs:
            // d(abs(x))/dx = sign(x), but zero at x=0
            guard inputs.count == 1 else { fatalError("abs requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * b.sign(input)
            b.grad(node.inputs[0], value: grad.lazy)
        case .sign:
            // d(sign(x))/dx = 0 everywhere except at x=0 where it's undefined
            guard inputs.count == 1 else { fatalError("sign requires 1 input") }
            let zero = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zero)
        case .sin:
            // d(sin(x))/dx = cos(x)
            guard inputs.count == 1 else { fatalError("sin requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * b.cos(input)
            b.grad(node.inputs[0], value: grad.lazy)
        case .cos:
            // d(cos(x))/dx = -sin(x)
            guard inputs.count == 1 else { fatalError("cos requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * (b.constant(0.0) - b.sin(input))
            b.grad(node.inputs[0], value: grad.lazy)
        case .tan:
            // d(tan(x))/dx = sec²(x) = 1/cos²(x)
            guard inputs.count == 1 else { fatalError("tan requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let cosInput = b.cos(input)
            let sec2 = b.constant(1.0) / (cosInput * cosInput)
            let grad = b.value(gradOutput) * sec2
            b.grad(node.inputs[0], value: grad.lazy)
        case .tanh:
            // d(tanh(x))/dx = 1 - tanh(x)^2
            guard inputs.count == 1 else { fatalError("tanh requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let t = b.tanh(input)
            let grad = b.value(gradOutput) * (b.constant(1.0) - t * t)
            b.grad(node.inputs[0], value: grad.lazy)
        case .exp:
            // d(exp(x))/dx = exp(x)
            guard inputs.count == 1 else { fatalError("exp requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) * b.exp(input)
            b.grad(node.inputs[0], value: grad.lazy)
        case .log:
            // d(log(x))/dx = 1/x
            guard inputs.count == 1 else { fatalError("log requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) / input
            b.grad(node.inputs[0], value: grad.lazy)
        case .log10:
            // d(log10(x))/dx = 1 / (x * ln(10))
            guard inputs.count == 1 else { fatalError("log10 requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let ln10 = b.constant(2.302585092994046)  // natural log of 10
            let grad = b.value(gradOutput) / (input * ln10)
            b.grad(node.inputs[0], value: grad.lazy)
        case .sqrt:
            // d(sqrt(x))/dx = 1/(2*sqrt(x))
            guard inputs.count == 1 else { fatalError("sqrt requires 1 input") }
            let input = b.tapeValue(node.inputs[0])
            let grad = b.value(gradOutput) / (b.constant(2.0) * b.sqrt(input))
            b.grad(node.inputs[0], value: grad.lazy)
        case .pow:
            // d(x^y)/dx = y * x^(y-1), d(x^y)/dy = x^y * ln(x)
            guard inputs.count == 2 else { fatalError("pow requires 2 inputs") }
            let base = b.tapeValue(node.inputs[0])
            let exponent = b.tapeValue(node.inputs[1])
            let result = b.pow(base, exponent)

            // Gradient w.r.t. base: y * x^(y-1)
            let baseGrad = b.value(gradOutput) * exponent * b.pow(base, exponent - b.constant(1.0))
            b.grad(node.inputs[0], value: baseGrad.lazy)

            // Gradient w.r.t. exponent: x^y * ln(x)
            let expGrad = b.value(gradOutput) * result * b.log(base)
            b.grad(node.inputs[1], value: expGrad.lazy)
        case .atan2:
            // d(atan2(y,x))/dy = x/(x²+y²), d(atan2(y,x))/dx = -y/(x²+y²)
            guard inputs.count == 2 else { fatalError("atan2 requires 2 inputs") }
            let y = b.tapeValue(node.inputs[0])
            let x = b.tapeValue(node.inputs[1])
            let denom = x * x + y * y
            let gradY = b.value(gradOutput) * (x / denom)
            let gradX = b.value(gradOutput) * (b.constant(0.0) - y / denom)
            b.grad(node.inputs[0], value: gradY.lazy)
            b.grad(node.inputs[1], value: gradX.lazy)
        case .mse:
            // loss = (a - b)^2; grads: d/da = 2*(a-b)*go, d/db = -2*(a-b)*go
            guard inputs.count == 2 else { fatalError("mse requires 2 inputs") }
            let a = b.tapeValue(node.inputs[0])
            let c = b.tapeValue(node.inputs[1])
            let diff = a - c
            let two = b.constant(2.0)
            let gradA = b.value(gradOutput) * two * diff
            let gradB = b.value(gradOutput) * (b.constant(0.0) - two * diff)
            b.grad(node.inputs[0], value: gradA.lazy)
            b.grad(node.inputs[1], value: gradB.lazy)

        case let .spectralLossTape(windowSize):
            guard inputs.count == 2 else { fatalError("spectralLossTape requires 2 inputs") }
            // We distribute the gradient across the entire analysis window by
            // writing directly into the gradients buffer for each sample j in
            // the window. To do that, we pass the gradIds for both inputs into
            // the specialized backward op so the renderer can index gradients.
            let sig1 = b.tapeValue(node.inputs[0])
            let sig2 = b.tapeValue(node.inputs[1])
            // Pass gradIds (not used by the non-scatter path, but available for future)
            let gradId1 = b.ctx.useGradient(src: node.inputs[0])
            let gradId2 = b.ctx.useGradient(src: node.inputs[1])
            let (gradExpr1, gradExpr2) = u_spectralLossTapeBackward(
                windowSize, sig1, sig2, b.value(gradOutput), gradId1, gradId2
            )(b)
            b.grad(node.inputs[0], value: gradExpr1.lazy)
            b.grad(node.inputs[1], value: gradExpr2.lazy)

        case let .spectralLossPass1(windowSize, scratchCell):
            guard inputs.count == 2 else { fatalError("spectralLossPass1 requires 2 inputs") }
            let sig1 = b.tapeValue(node.inputs[0])
            let sig2 = b.tapeValue(node.inputs[1])
            let gradId1 = b.ctx.useGradient(src: node.inputs[0])
            let gradId2 = b.ctx.useGradient(src: node.inputs[1])

            // Pass B: Reduce from memory to gradients (READ)
            // This runs SECOND in backward (after Pass2), so memory has been written
            let (grad1, grad2) = u_spectralLossBackwardPass2(
                windowSize, scratchCell, sig1, sig2, b.value(gradOutput), gradId1, gradId2
            )(b)

            // Propagate gradients to original signals
            b.grad(node.inputs[0], value: grad1.lazy)
            b.grad(node.inputs[1], value: grad2.lazy)

        case let .spectralLossPass2(windowSize, scratchCell):
            guard inputs.count == 1 else { fatalError("spectralLossPass2 requires 1 input") }

            // Get the original signal inputs from Pass1's node
            let pass1Node = g.nodes[node.inputs[0]]!
            guard pass1Node.inputs.count == 2 else { fatalError("Expected Pass1 to have 2 inputs") }

            let sig1 = b.tapeValue(pass1Node.inputs[0])
            let sig2 = b.tapeValue(pass1Node.inputs[1])

            // Pass A: Accumulate per-window gradient contributions to memory (WRITE)
            // This runs FIRST in backward (before Pass1), writing data that Pass1 will read
            u_spectralLossBackwardPass1(
                windowSize, scratchCell, sig1, sig2, b.value(gradOutput)
            )(b)

        // Don't propagate gradients - Pass1 will handle that

        case .floor:
            // d(floor(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .ceil:
            // d(ceil(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .round:
            // d(ceil(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .memoryRead(_):
            // For memoryRead, gradient flows through to the values written to memory
            // This is complex and depends on the memory write operations
            guard inputs.count == 1 else { fatalError("memoryRead requires 1 input") }
            // For now, treat as zero gradient for the offset
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
        case .memoryWrite(_):
            // For memoryWrite, gradient flows through to both offset and value inputs
            guard inputs.count == 2 else { fatalError("memoryWrite requires 2 inputs") }
            // Gradient for offset is typically zero (address computation)
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
            // Gradient for value flows through
            b.grad(node.inputs[1], value: gradOutput)
        case .scalarMemoryWrite(_):
            // Same gradient semantics as memoryWrite
            guard inputs.count == 2 else { fatalError("scalarMemoryWrite requires 2 inputs") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zeroGrad)
            b.grad(node.inputs[1], value: gradOutput)
        case .gt, .gte, .lte, .lt, .eq:
            // Comparisons have zero gradient (non-differentiable)
            guard node.inputs.count == 2 else { fatalError("comparison requires 2 inputs") }
            let zero = ctx.useConstant(src: nil, value: 0.0)
            b.grad(node.inputs[0], value: zero)
            b.grad(node.inputs[1], value: zero)

        case .gswitch:
            // gswitch(cond, x, y) = cond ? x : y
            guard inputs.count == 3 else { fatalError("gswitch requires 3 inputs") }
            let cond = b.tapeValue(node.inputs[0])
            let gradX = b.gswitch(cond, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(cond, b.constant(0.0), b.value(gradOutput))
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[1], value: gradX.lazy)
            b.grad(node.inputs[2], value: gradY.lazy)

        case .selector:
            // selector(mode, options[]) -> gradient flows only to the selected option
            guard inputs.count >= 2 else { fatalError("selector requires at least 2 inputs") }

            // Gradient for mode is always zero (index is non-differentiable)
            b.grad(node.inputs[0], value: ctx.useConstant(src: nil, value: 0.0))

            // For each option, gradient is non-zero only if it was selected
            let mode = b.tapeValue(node.inputs[0])
            for i in 1..<node.inputs.count {
                let optionIndex = b.constant(Float(i - 1))
                let isSelected = mode == optionIndex
                let gradOption = b.gswitch(isSelected, b.value(gradOutput), b.constant(0.0))
                b.grad(node.inputs[i], value: gradOption.lazy)
            }

        case .mix:
            // mix(x, y, t) = x * (1-t) + y * t
            guard inputs.count == 3 else { fatalError("mix requires 3 inputs") }
            let x = b.tapeValue(node.inputs[0])
            let y = b.tapeValue(node.inputs[1])
            let t = b.tapeValue(node.inputs[2])

            // d/dx = (1-t)
            let gradX = b.value(gradOutput) * (b.constant(1.0) - t)
            // d/dy = t
            let gradY = b.value(gradOutput) * t
            // d/dt = y - x
            let gradT = b.value(gradOutput) * (y - x)

            b.grad(node.inputs[0], value: gradX.lazy)
            b.grad(node.inputs[1], value: gradY.lazy)
            b.grad(node.inputs[2], value: gradT.lazy)

        case .historyReadWrite(let cellId):
            print(
                "HISTORY WRITE CALEDDEDEDEDEDEDEDEDEDEEEEEEEEEEE\(String(repeating:"*", count:256))"
            )

            // Combined history read/write (not in feedback loop):
            // Forward returns prev and persists curr. Backward must:
            //   1) send gradOutput to previous timestep via storeGrad(cellId, gradOutput)
            //   2) pass grad to input as gradOutput + gradFromFuture (like historyWrite)
            guard inputs.count == 1 else { fatalError("historyReadWrite requires 1 input") }
            // Whatever hits y[i-1] becomes part of the carry for the previous timestep.
            // IMPORTANT: don't add loadGradMemory again here; you'll double-count.
            let inc = b.value(gradOutput)
            _ = b.storeGradMemory(cellId, inc)
            let total = b.value(gradOutput) + b.loadGradMemory(cellId)
            b.grad(node.inputs[0], value: total.lazy)

        case .historyWrite(let cellId):
            guard inputs.count == 1 else { fatalError("history write requires 1 input") }
            // total grad on y[i] = local seed + future carry (slot already holds i+1 in reverse-time)
            let total = b.value(gradOutput) + b.loadGradMemory(cellId)
            b.grad(node.inputs[0], value: total.lazy)
        case .historyRead(let cellId):
            _ = b.storeGradMemory(cellId, b.value(gradOutput))
        case .latch(_):
            // Gradient flows through value input only when condition was true
            guard inputs.count == 2 else { fatalError("latch requires 2 inputs") }
            let cond = b.tapeValue(node.inputs[1])
            let gradValue = b.gswitch(cond > b.constant(0), b.value(gradOutput), b.constant(0.0))
            b.grad(node.inputs[0], value: gradValue.lazy)
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))

        case .accum(let cellId):
            let reset = b.tapeValue(node.inputs[1])

            // Gradient for increment (blocked by reset)
            let gradIncr = b.gswitch(reset > b.constant(0), b.constant(0.0), b.value(gradOutput))

            // handle temporal gradient flow:
            // The gradient also flows to the previous accumulated value
            let gradFromFuture = b.loadGradMemory(cellId)

            // The total gradient flowing backward through time
            let gradToPrev = b.value(gradOutput) + gradFromFuture

            // Store for previous timestep's accum
            _ = b.storeGradMemory(cellId, gradToPrev)

            b.grad(node.inputs[0], value: gradIncr.lazy)
            b.grad(node.inputs[1], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[2], value: ctx.useConstant(src: nil, value: 0.0))
            b.grad(node.inputs[3], value: ctx.useConstant(src: nil, value: 0.0))

        case .phasor(let cellId):
            let sampleRate = b.constant(44100.0)
            let currentTime = b.frameIndex(nodeId)

            // Compute this timestep's frequency gradient
            // d(phase)/d(freq) = time / sampleRate (since phase accumulates as: phase += freq/sampleRate)
            let gradFreq = b.value(gradOutput) * currentTime / sampleRate

            // Write gradient directly (no accumulation needed - test will sum across frames)
            b.grad(node.inputs[0], value: gradFreq.lazy)
        // NOTE ON GRADIENT SCALE VS frameCount
        // ------------------------------------------------------------
        // The phasor’s backward pass produces a gradient proportional
        // to the current frame index (time). Even though Training.swift
        // averages gradients across frames, the mean of i/sampleRate over
        // i = 0..(frameCount-1) grows with the time horizon (~O(frameCount)).
        // As a result, increasing frameCount increases the effective
        // gradient scale for frequency parameters, requiring a smaller
        // learning rate to maintain stability. This is independent of the
        // particular loss (e.g., spectral loss) and comes from the time-
        // weighted nature of d(phase)/d(freq).
        case .output(_):
            // Output just passes gradient through to its input
            guard inputs.count == 1 else { fatalError("output requires 1 input") }
            b.grad(node.inputs[0], value: gradOutput)
        case .input(_):
            break
        case .seq:
            // TODO: take another look at this, all the computation in each input is executed, even though the final one is returned
            // Gradient flows only to the last input (the one whose value is returned)
            guard node.inputs.count >= 2 else { fatalError("seq requires at least 2 inputs") }
            // Zero gradients for all inputs except the last
            for i in 0..<(node.inputs.count - 1) {
                b.grad(node.inputs[i], value: ctx.useConstant(src: nil, value: 0.0))
            }
            // Pass gradient to the last input
            if let lastInput = node.inputs.last {
                b.grad(lastInput, value: gradOutput)
            }
        }

        ops.append(contentsOf: b.ops)
        return ops
    }
}

// Tape-based spectral loss backward op
/// Backward pass for spectral loss: computes gradients w.r.t. both input signals.
///
/// Uses window-averaging to avoid race conditions: instead of scattering gradient
/// contributions to all samples j in the window, averages contributions and assigns
/// to the current sample. This is safe but creates late-frame bias.
func u_spectralLossTapeBackward(
    _ windowSize: Int,
    _ sig1: Expr,
    _ sig2: Expr,
    _ upstreamGrad: Expr,
    _ gradId1: GradID,
    _ gradId2: GradID
) -> (IRBuilder) -> (Expr, Expr) {
    return { b in
        let numBins = windowSize / 2 + 1

        // Accumulate total gradients for current sample
        let totalGrad1 = b.float(0.0)
        let totalGrad2 = b.float(0.0)

        // For each frequency bin
        b.loop(numBins) { binIndex in
            // DFT accumulators (real and imaginary parts)
            let real1 = b.float(0.0)
            let imag1 = b.float(0.0)
            let real2 = b.float(0.0)
            let imag2 = b.float(0.0)

            // Compute DFT over window samples
            b.loop(windowSize) { n in
                let idx = b.threadIndex()
                let winSize = b.constant(Float(windowSize))
                let j = idx - (winSize - b.constant(1.0)) + b.cast(n, to: .float)

                // Load samples from tape with bounds checking
                let s1 = b.tapeLoad(sig1, at: j)
                let s2 = b.tapeLoad(sig2, at: j)

                // DFT basis: e^(-2πi*k*n/N) = cos(angle) - i*sin(angle)
                let binIndexFloat = b.cast(binIndex, to: .float)
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

            // Loss gradient for this bin: d/d(mag1) of (mag1-mag2)²
            let magDiff = mag1 - mag2
            let lossGrad = b.constant(2.0) * magDiff

            // Average gradient contributions across window (to avoid scatter races)
            let accum1 = b.float(0.0)
            let accum2 = b.float(0.0)

            b.loop(windowSize) { n in
                let binIndexFloat = b.cast(binIndex, to: .float)
                let nFloat = b.cast(n, to: .float)
                let winSize = b.constant(Float(windowSize))
                let angle_n = b.constant(-2.0) * b.pi * binIndexFloat * nFloat / winSize
                let c_n = b.cos(angle_n)
                let s_n = b.sin(angle_n)

                // ∂mag/∂s[n] = (real*cos + imag*sin) / mag
                let eps = b.constant(1e-8)
                let sampleGrad1 = (real1.value * c_n + imag1.value * s_n) / (mag1 + eps)
                let sampleGrad2 = (real2.value * c_n + imag2.value * s_n) / (mag2 + eps)

                // Chain rule: ∂L/∂s = (∂L/∂mag) * (∂mag/∂s)
                accum1.accumulate(lossGrad * sampleGrad1)
                accum2.accumulate((b.constant(0.0) - lossGrad) * sampleGrad2)
            }

            // Normalize by window size to keep scale consistent
            let winSizeConst = b.constant(Float(windowSize))
            totalGrad1.accumulate(accum1.value / winSizeConst)
            totalGrad2.accumulate(accum2.value / winSizeConst)
        }

        // Multiply by upstream gradient from loss
        let grad1 = totalGrad1.value * upstreamGrad
        let grad2 = totalGrad2.value * upstreamGrad

        return (grad1, grad2)
    }
}

// Two-pass spectral loss backward functions

/// Pass A: Accumulate per-window gradient contributions to memory.
/// Each thread i (window end) computes DFT for its window and stores per-sample contributions.
func u_spectralLossBackwardPass1(
    _ windowSize: Int,
    _ scratchCell: CellID,
    _ sig1: Expr,
    _ sig2: Expr,
    _ upstreamGrad: Expr
) -> (IRBuilder) -> Void {
    return { b in
        let numBins = windowSize / 2 + 1
        let i = b.threadIndex()  // Current frame (window end)

        // For each frequency bin
        b.loop(numBins) { binIndex in
            // DFT accumulators (real and imaginary parts)
            let real1 = b.float(0.0)
            let imag1 = b.float(0.0)
            let real2 = b.float(0.0)
            let imag2 = b.float(0.0)

            // Compute DFT over window samples
            b.loop(windowSize) { n in
                let winSize = b.constant(Float(windowSize))
                let j = i - (winSize - b.constant(1.0)) + b.cast(n, to: .float)

                // Load samples from tape with bounds checking
                let s1 = b.tapeLoad(sig1, at: j)
                let s2 = b.tapeLoad(sig2, at: j)

                // DFT basis: e^(-2πi*k*n/N) = cos(angle) - i*sin(angle)
                let binIndexFloat = b.cast(binIndex, to: .float)
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

            // Loss gradient for this bin: d/d(mag1) of (mag1-mag2)²
            let magDiff = mag1 - mag2
            let lossGrad = b.constant(2.0) * magDiff

            // For each window offset, compute and store contribution
            b.loop(windowSize) { n in
                let binIndexFloat = b.cast(binIndex, to: .float)
                let nFloat = b.cast(n, to: .float)
                let winSize = b.constant(Float(windowSize))
                let angle_n = b.constant(-2.0) * b.pi * binIndexFloat * nFloat / winSize
                let c_n = b.cos(angle_n)
                let s_n = b.sin(angle_n)

                // ∂mag/∂s[n] = (real*cos + imag*sin) / mag
                let eps = b.constant(1e-8)
                let sampleGrad1 = (real1.value * c_n + imag1.value * s_n) / (mag1 + eps)
                let sampleGrad2 = (real2.value * c_n + imag2.value * s_n) / (mag2 + eps)

                // Chain rule: ∂L/∂s = (∂L/∂mag) * (∂mag/∂s) * upstreamGrad
                let contrib1 = lossGrad * sampleGrad1 * upstreamGrad
                let contrib2 = (b.constant(0.0) - lossGrad) * sampleGrad2 * upstreamGrad

                // Write to memory: memory[scratchCell + (i * windowSize * 2) + (n * 2) + component]
                let winSizeConst = b.constant(Float(windowSize))
                let offset1 =
                    i * winSizeConst * b.constant(2.0) + b.cast(n, to: .float) * b.constant(2.0)
                let offset2 = offset1 + b.constant(1.0)

                _ = b.memoryWrite(scratchCell, b.cast(offset1, to: .int), contrib1)
                _ = b.memoryWrite(scratchCell, b.cast(offset2, to: .int), contrib2)
            }
        }
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

        // Gather from all windows that include sample j
        // Windows ending at i ∈ [j .. j+(windowSize-1)]
        b.loop(windowSize) { offsetFromJ in
            let windowEnd = j + b.cast(offsetFromJ, to: .float)  // i

            // Window offset: n = j - i + (windowSize - 1)
            let winSize = b.constant(Float(windowSize))
            let n = j - windowEnd + (winSize - b.constant(1.0))

            // Memory offset: memory[scratchCell + (i * windowSize * 2) + (n * 2) + component]
            let offset1 = windowEnd * winSize * b.constant(2.0) + n * b.constant(2.0)
            let offset2 = offset1 + b.constant(1.0)

            // Read contributions from memory
            let contrib1 = b.memoryRead(scratchCell, b.cast(offset1, to: .int))
            let contrib2 = b.memoryRead(scratchCell, b.cast(offset2, to: .int))

            grad1.accumulate(contrib1)
            grad2.accumulate(contrib2)
        }

        return (grad1.value, grad2.value)
    }
}

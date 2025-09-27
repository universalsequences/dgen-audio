public typealias NodeID = Int
public typealias VarID = Int
public typealias ConstantID = Int
public typealias CellID = Int
public typealias ChannelNumber = Int

public enum Lazy {
    case constant(ConstantID, Float)
    case global(VarID)
    case variable(VarID, NodeID?)
    case empty
}

// IR (intermediate representation) is called UOp and consists of an
// operator (Op) and value (the variable it's result is bound to)
public enum Op {
    case load(CellID)
    case store(CellID, Lazy)
    case delay1(CellID, Lazy)
    case loadGrad(CellID)
    case storeGrad(CellID, Lazy)
    case mutate(Lazy, Lazy)
    case add(Lazy, Lazy)
    case sub(Lazy, Lazy)
    case mul(Lazy, Lazy)
    case div(Lazy, Lazy)
    case abs(Lazy)
    case sign(Lazy)
    case sin(Lazy)
    case cos(Lazy)
    case tan(Lazy)
    case tanh(Lazy)
    case exp(Lazy)
    case log(Lazy)
    case log10(Lazy)
    case sqrt(Lazy)
    case pow(Lazy, Lazy)
    case atan2(Lazy, Lazy)
    case mod(Lazy, Lazy)
    case gt(Lazy, Lazy)
    case gte(Lazy, Lazy)
    case lte(Lazy, Lazy)
    case lt(Lazy, Lazy)
    case eq(Lazy, Lazy)
    case min(Lazy, Lazy)
    case max(Lazy, Lazy)
    case floor(Lazy)
    case ceil(Lazy)
    case round(Lazy)
    case memoryRead(CellID, Lazy)
    case memoryWrite(CellID, Lazy, Lazy)
    case latch(Lazy, Lazy)
    case beginIf(Lazy)
    case gswitch(Lazy, Lazy, Lazy)
    case selector(Lazy, [Lazy])  // selector(mode, options[])
    case endIf
    case defineGlobal(VarID)
    case defineConstant(ConstantID, Float)
    case defineMemory(Int)
    case loadGlobal(VarID)
    case beginLoop(Lazy, Int)
    case endLoop
    case beginRange(Lazy, Lazy)
    case endRange
    case output(ChannelNumber, Lazy)
    case input(ChannelNumber)
    case frameCount
    case frameIndex
}

public struct UOp {
    public let op: Op
    public let value: Lazy
    public var kind: Kind? = nil
}

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

// writes current value to cell and returns delayed by 1 sample
func u_historyWrite(cellId: CellID, _ curr: Expr) -> (IRBuilder) -> Expr {
    return { b in
        // delay1 now encapsulates both: returns x[i-1] and persists current to cell
        let prev = b.delay1(cellId, curr)
        return prev
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
    case selector  // selector(mode, options[])
    case memoryRead(CellID)
    case memoryWrite(CellID)
    case historyWrite(CellID)
    case historyReadWrite(CellID)
    case param(CellID)
    case latch(CellID)
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
        case .atan2:
            guard inputs.count == 2 else {
                throw DGenError.insufficientInputs(
                    operator: "atan2", expected: 2, actual: inputs.count)
            }
            b.use(val: b.atan2(b.value(inputs[0]), b.value(inputs[1])))
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

    func emitBackward(ctx: IRContext, g: Graph, nodeId: NodeID, gradOutput: Lazy)
        -> BackwardsEmitResult
    {
        guard let node = g.nodes[nodeId] else {
            return BackwardsEmitResult(ops: [], dependencies: [])
        }

        // Collect operands
        let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
        var ops: [UOp] = []
        let b = IRBuilder(ctx: ctx, nodeId: nodeId)

        var deps: [NodeID] = []

        switch self {
        case .constant(_):
            // Constants have no gradients to propagate
            return BackwardsEmitResult(ops: [], dependencies: deps)
        case .add:
            // d(x+y)/dx = 1, d(x+y)/dy = 1
            guard node.inputs.count == 2 else { fatalError("add requires 2 inputs") }
            ctx.gradients[node.inputs[0]] = gradOutput
            ctx.gradients[node.inputs[1]] = gradOutput
        case .sub:
            // d(x+y)/dx = 1, d(x+y)/dy = 1
            guard node.inputs.count == 2 else { fatalError("add requires 2 inputs") }
            ctx.gradients[node.inputs[0]] = gradOutput
            ctx.gradients[node.inputs[1]] = gradOutput
        case .mod:
            // TODO  - implement
            print("need back")
        case .param(_):
            // TODO  - implement
            print("need back")
        case .min:
            // d(min(x,y))/dx = (x <= y) ? 1 : 0, d(min(x,y))/dy = (y < x) ? 1 : 0
            guard inputs.count == 2 else { fatalError("min requires 2 inputs") }
            let x = b.value(inputs[0])
            let y = b.value(inputs[1])
            let xIsMin = b.value(inputs[0]) <= b.value(inputs[1])
            let yIsMin = b.value(inputs[1]) < b.value(inputs[0])
            let gradX = b.gswitch(xIsMin, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(yIsMin, b.value(gradOutput), b.constant(0.0))
            ctx.gradients[node.inputs[0]] = gradX.lazy
            ctx.gradients[node.inputs[1]] = gradY.lazy
            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
        case .max:
            // d(max(x,y))/dx = (x >= y) ? 1 : 0, d(max(x,y))/dy = (y > x) ? 1 : 0
            guard inputs.count == 2 else { fatalError("max requires 2 inputs") }
            let x = b.value(inputs[0])
            let y = b.value(inputs[1])
            let xIsMax = b.value(inputs[0]) >= b.value(inputs[1])
            let yIsMax = b.value(inputs[1]) > b.value(inputs[0])
            let gradX = b.gswitch(xIsMax, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(yIsMax, b.value(gradOutput), b.constant(0.0))
            ctx.gradients[node.inputs[0]] = gradX.lazy
            ctx.gradients[node.inputs[1]] = gradY.lazy
            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
        case .mul:
            // d(x*y)/dx = y, d(x*y)/dy = x
            guard inputs.count == 2 else { fatalError("mul \(node.id) requires 2 inputs") }
            let gradX = b.value(gradOutput) * b.value(inputs[1])
            let gradY = b.value(gradOutput) * b.value(inputs[0])
            ctx.gradients[node.inputs[0]] = gradX.lazy
            ctx.gradients[node.inputs[1]] = gradY.lazy
            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
        case .div:
            // TODO - this is copied from div but needs to be implemented correctly
            // d(x*y)/dx = y, d(x*y)/dy = x
            guard inputs.count == 2 else { fatalError("div \(node.id) requires 2 inputs") }
            let gradX = b.value(gradOutput) * b.value(inputs[1])
            let gradY = b.value(gradOutput) * b.value(inputs[0])
            ctx.gradients[node.inputs[0]] = gradX.lazy
            ctx.gradients[node.inputs[1]] = gradY.lazy
            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
        case .abs:
            // d(abs(x))/dx = sign(x), but zero at x=0
            guard inputs.count == 1 else { fatalError("abs requires 1 input") }
            let input = b.value(inputs[0])
            let grad = b.value(gradOutput) * b.sign(input)
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .sign:
            // d(sign(x))/dx = 0 everywhere except at x=0 where it's undefined
            guard inputs.count == 1 else { fatalError("sign requires 1 input") }
            let zero = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zero
        case .sin:
            // d(sin(x))/dx = cos(x)
            guard inputs.count == 1 else { fatalError("sin requires 1 input") }
            let input = b.value(inputs[0])
            let grad = b.value(gradOutput) * b.cos(input)
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .cos:
            // d(cos(x))/dx = -sin(x)
            guard inputs.count == 1 else { fatalError("cos requires 1 input") }
            let input = b.value(inputs[0])
            let grad = b.value(gradOutput) * (b.constant(0.0) - b.sin(input))
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .tan:
            // d(tan(x))/dx = sec²(x) = 1/cos²(x)
            guard inputs.count == 1 else { fatalError("tan requires 1 input") }
            let input = b.value(inputs[0])
            let cosInput = b.cos(input)
            let sec2 = b.constant(1.0) / (cosInput * cosInput)
            let grad = b.value(gradOutput) * sec2
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .tanh:
            // TODO - implement
            //
            break
        case .exp:
            // d(exp(x))/dx = exp(x)
            guard inputs.count == 1 else { fatalError("exp requires 1 input") }
            let input = b.value(inputs[0])
            let grad = b.value(gradOutput) * b.exp(input)
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .log:
            // d(log(x))/dx = 1/x
            guard inputs.count == 1 else { fatalError("log requires 1 input") }
            let input = b.value(inputs[0])
            let grad = b.value(gradOutput) / input
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .log10:
            // TODO: implement diff
            break
        case .sqrt:
            // d(sqrt(x))/dx = 1/(2*sqrt(x))
            guard inputs.count == 1 else { fatalError("sqrt requires 1 input") }
            let input = b.value(inputs[0])
            let grad = b.value(gradOutput) / (b.constant(2.0) * b.sqrt(input))
            ctx.gradients[node.inputs[0]] = grad.lazy
            deps.append(node.inputs[0])
        case .pow:
            // d(x^y)/dx = y * x^(y-1), d(x^y)/dy = x^y * ln(x)
            guard inputs.count == 2 else { fatalError("pow requires 2 inputs") }
            let base = b.value(inputs[0])
            let exponent = b.value(inputs[1])
            let result = b.pow(base, exponent)

            // Gradient w.r.t. base: y * x^(y-1)
            let baseGrad = b.value(gradOutput) * exponent * b.pow(base, exponent - b.constant(1.0))
            ctx.gradients[node.inputs[0]] = baseGrad.lazy
            deps.append(node.inputs[0])

            // Gradient w.r.t. exponent: x^y * ln(x)
            let expGrad = b.value(gradOutput) * result * b.log(base)
            ctx.gradients[node.inputs[1]] = expGrad.lazy
            deps.append(node.inputs[1])
        case .atan2:
            // d(atan2(y,x))/dy = x/(x²+y²), d(atan2(y,x))/dx = -y/(x²+y²)
            guard inputs.count == 2 else { fatalError("atan2 requires 2 inputs") }
            let y = b.value(inputs[0])
            let x = b.value(inputs[1])
            let denom = x * x + y * y
            let gradY = b.value(gradOutput) * (x / denom)
            let gradX = b.value(gradOutput) * (b.constant(0.0) - y / denom)
            ctx.gradients[node.inputs[0]] = gradY.lazy
            ctx.gradients[node.inputs[1]] = gradX.lazy
            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
        case .floor:
            // d(floor(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zeroGrad
            deps.append(node.inputs[0])
        case .ceil:
            // d(ceil(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zeroGrad
            deps.append(node.inputs[0])
        case .round:
            // d(ceil(x))/dx = 0 (floor is not differentiable, but we set to 0)
            guard inputs.count == 1 else { fatalError("floor requires 1 input") }
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zeroGrad
            deps.append(node.inputs[0])
        case .memoryRead(_):
            // For memoryRead, gradient flows through to the values written to memory
            // This is complex and depends on the memory write operations
            guard inputs.count == 1 else { fatalError("memoryRead requires 1 input") }
            // For now, treat as zero gradient for the offset
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zeroGrad
            deps.append(node.inputs[0])
        case .memoryWrite(_):
            // For memoryWrite, gradient flows through to both offset and value inputs
            guard inputs.count == 2 else { fatalError("memoryWrite requires 2 inputs") }
            // Gradient for offset is typically zero (address computation)
            let zeroGrad = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zeroGrad
            // Gradient for value flows through
            ctx.gradients[node.inputs[1]] = gradOutput
            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
        case .gt, .gte, .lte, .lt, .eq:
            // Comparisons have zero gradient (non-differentiable)
            guard node.inputs.count == 2 else { fatalError("comparison requires 2 inputs") }
            let zero = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[0]] = zero
            ctx.gradients[node.inputs[1]] = zero

        case .gswitch:
            // gswitch(cond, x, y) = cond ? x : y
            guard inputs.count == 3 else { fatalError("gswitch requires 3 inputs") }
            let cond = b.value(inputs[0])
            let gradX = b.gswitch(cond, b.value(gradOutput), b.constant(0.0))
            let gradY = b.gswitch(cond, b.constant(0.0), b.value(gradOutput))
            ctx.gradients[node.inputs[0]] = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[1]] = gradX.lazy
            ctx.gradients[node.inputs[2]] = gradY.lazy
            deps.append(node.inputs[0])

        case .selector:
            // selector(mode, options[]) -> gradient flows only to the selected option
            guard inputs.count >= 2 else { fatalError("selector requires at least 2 inputs") }

            // Gradient for mode is always zero (index is non-differentiable)
            ctx.gradients[node.inputs[0]] = ctx.useConstant(src: nil, value: 0.0)

            // For each option, gradient is non-zero only if it was selected
            let mode = b.value(inputs[0])
            for i in 1..<node.inputs.count {
                let optionIndex = b.constant(Float(i - 1))
                let isSelected = b.value(inputs[0]) == optionIndex
                let gradOption = b.gswitch(isSelected, b.value(gradOutput), b.constant(0.0))
                ctx.gradients[node.inputs[i]] = gradOption.lazy
                deps.append(node.inputs[i])
            }

        case .mix:
            // mix(x, y, t) = x * (1-t) + y * t
            guard inputs.count == 3 else { fatalError("mix requires 3 inputs") }
            let x = b.value(inputs[0])
            let y = b.value(inputs[1])
            let t = b.value(inputs[2])

            // d/dx = (1-t)
            let gradX = b.value(gradOutput) * (b.constant(1.0) - t)
            // d/dy = t
            let gradY = b.value(gradOutput) * t
            // d/dt = y - x
            let gradT = b.value(gradOutput) * (y - x)

            ctx.gradients[node.inputs[0]] = gradX.lazy
            ctx.gradients[node.inputs[1]] = gradY.lazy
            ctx.gradients[node.inputs[2]] = gradT.lazy

            deps.append(node.inputs[0])
            deps.append(node.inputs[1])
            deps.append(node.inputs[2])
        case .historyReadWrite(let cellId):
            print("Yo")
        case .historyWrite(let cellId):
            // Pass gradient through to input, plus any gradient from future reads
            guard inputs.count == 1 else { fatalError("history write requires 1 input") }

            // Load gradient that was stored by historyRead in the future
            let gradFromFuture = b.loadGrad(cellId)
            let totalGrad = b.value(gradOutput) + gradFromFuture
            ctx.gradients[node.inputs[0]] = totalGrad.lazy

        case .historyRead(let cellId):
            // Store gradient for the corresponding historyWrite in the past
            guard inputs.count == 0 else { fatalError("history read requires 0 inputs") }
            _ = b.storeGrad(cellId, b.value(gradOutput))

        case .latch(_):
            // Gradient flows through value input only when condition was true
            guard inputs.count == 2 else { fatalError("latch requires 2 inputs") }
            let cond = b.value(inputs[1])
            let gradValue = b.gswitch(cond > b.constant(0), b.value(gradOutput), b.constant(0.0))
            ctx.gradients[node.inputs[0]] = gradValue.lazy
            ctx.gradients[node.inputs[1]] = ctx.useConstant(src: nil, value: 0.0)

            deps.append(node.inputs[1])

        case .accum(let cellId):
            let b = IRBuilder(ctx: ctx, nodeId: nodeId)
            let reset = b.value(inputs[1])

            // Gradient for increment (blocked by reset)
            let gradIncr = b.gswitch(reset > b.constant(0), b.constant(0.0), b.value(gradOutput))

            // handle temporal gradient flow:
            // The gradient also flows to the previous accumulated value
            let gradFromFuture = b.loadGrad(cellId)

            // The total gradient flowing backward through time
            let gradToPrev = b.value(gradOutput) + gradFromFuture

            // Store for previous timestep's accum
            _ = b.storeGrad(cellId, gradToPrev)

            ctx.gradients[node.inputs[0]] = gradIncr.lazy
            ctx.gradients[node.inputs[1]] = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[2]] = ctx.useConstant(src: nil, value: 0.0)
            ctx.gradients[node.inputs[3]] = ctx.useConstant(src: nil, value: 0.0)

            deps.append(node.inputs[1])

        case .phasor(let cellId):
            let b = IRBuilder(ctx: ctx, nodeId: nodeId)
            let sampleRate = b.constant(44100.0)
            let currentTime = b.frameIndex(nodeId)

            // Compute this timestep's frequency gradient
            let gradFreq = b.value(gradOutput) * currentTime / sampleRate

            // Accumulate in gradient memory at cellId (cell is not used for anything else)
            let prevGradFreq = b.loadGrad(cellId)  // First time: 0
            let totalGradFreq = prevGradFreq + gradFreq
            _ = b.storeGrad(cellId, totalGradFreq)

            ctx.gradients[node.inputs[0]] = totalGradFreq.lazy
        case .output(_):
            // Output just passes gradient through to its input
            guard inputs.count == 1 else { fatalError("output requires 1 input") }
            ctx.gradients[node.inputs[0]] = gradOutput
        case .input(_):
            break
        case .seq:
            // Gradient flows only to the last input (the one whose value is returned)
            guard node.inputs.count >= 2 else { fatalError("seq requires at least 2 inputs") }
            // Zero gradients for all inputs except the last
            for i in 0..<(node.inputs.count - 1) {
                ctx.gradients[node.inputs[i]] = ctx.useConstant(src: nil, value: 0.0)
            }
            // Pass gradient to the last input
            if let lastInput = node.inputs.last {
                ctx.gradients[lastInput] = gradOutput
            }
        }

        ops.append(contentsOf: b.ops)
        return BackwardsEmitResult(ops: ops, dependencies: deps)
    }
}

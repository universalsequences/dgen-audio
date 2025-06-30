public typealias NodeID = Int;
public typealias VarID = Int;
public typealias ConstantID = Int;
public typealias CellID = Int;
public typealias ChannelNumber = Int

public enum Lazy: Hashable {
    case constant(ConstantID, Float)
    case global(VarID)
    case variable(VarID, NodeID?)
    case empty

     public static func == (lhs: Lazy, rhs: Lazy) -> Bool {
        switch (lhs, rhs) {
        case let (.constant(v1, a), .constant(v2, b)):
            return a.bitPattern == b.bitPattern
        case let (.variable(a1, b1), .variable(a2, b2)):
            return a1 == a2 && b1 == b2
        default:
            return false
        }
    }

    public func hash(into hasher: inout Hasher) {
        switch self {
        case .empty:
            hasher.combine(-1)
        case let .constant(v,a):
            hasher.combine(0)
            hasher.combine(a.bitPattern)  // Avoid precision-based float issues
        case let .variable(v, node):
            hasher.combine(1)
            hasher.combine(v)
            hasher.combine(node)
        case let .global(v):
            hasher.combine(v)
        }
    }
}

// IR
public enum Op {
    case load(CellID)
    case store(CellID, Lazy)
    case mutate(Lazy, Lazy)
    case add(Lazy, Lazy)
    case sub(Lazy, Lazy)
    case mul(Lazy, Lazy)
    case div(Lazy, Lazy)
    case gt(Lazy, Lazy)
    case lt(Lazy, Lazy)
    case latch(Lazy, Lazy)
    case beginIf(Lazy)
    case gswitch(Lazy, Lazy, Lazy)
    case endIf
    case defineGlobal(VarID)
    case defineConstant(ConstantID, Float)
    case defineMemory(Int)
    case loadGlobal(VarID)
    case beginLoop(Lazy, Int)
    case endLoop
    case beginRange(Int,Int)
    case endRange
    case output(ChannelNumber, Lazy)
    case frameCount
}

public struct UOp {
    public let op: Op;
    public let value: Lazy;
    public var kind: Kind? = nil;
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
    return {ctx, nodeId in
        return UOp(op: .beginIf(cond), value: ctx.useVariable(src: nil))
    }
}

func u_end_if() -> (IRContext, NodeID?) -> UOp {
    return {ctx, nodeId in
        return UOp(op: .endIf, value: ctx.useVariable(src: nil))
    }
}

func u_switch(_ cond: Lazy, _ then: Lazy, _ els: Lazy) -> (IRContext, NodeID?) -> UOp {
    return {ctx, nodeId in
        return UOp(op: .gswitch(cond,then,els), value: ctx.useVariable(src: nodeId))
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

let u_gt = binaryOp(Op.gt)
let u_lt = binaryOp(Op.lt)
let u_add = binaryOp(Op.add)
let u_div = binaryOp(Op.div)
let u_mul = binaryOp(Op.mul)
let u_sub = binaryOp(Op.sub)

func u_accum(_ cellId: CellID, incr: Expr, reset: Expr, min: Expr, max: Expr) -> (IRBuilder) -> Expr {
    return {b in
        let acc = b.load(cellId, b.nodeId)
        let newVal = acc + incr
        var skip = true
        switch reset.lazy {
        case let .constant(_, value):
            if (value == 0) {
                skip = true
            }
        default:
            break
        }
        if (!skip) {
            b.if(reset > b.constant(0)) {
                _ = b.store(cellId, min)
                b.mutate(newVal, to: min)
            }
        }

        _ = b.store(cellId, newVal)

        b.if(newVal > max) {
            let wrapped = newVal - (max - min)
            _ = b.store(cellId, wrapped)
        }
        
        return acc;
    } 
}

func u_phasor(_ cellId: CellID, freq: Expr, reset: Expr) -> (IRBuilder) -> Expr {
    return { b in
        let b_sr = b.constant(44100)
        return u_accum(cellId, incr: freq / b_sr, reset: reset, min: b.constant(0), max: b.constant(1))(b)
    } 
}

func u_latch(_ cellId: CellID, value: Expr, cond: Expr) -> (IRBuilder) -> Expr {
    return { b in
        let latched = b.load(cellId);
        b.if(cond > b.constant(0)) {
            _ = b.store(cellId, value)
            b.mutate(latched, to: value)
        }
        return latched; 
    }
}

func u_mix(_ x: Expr, _ y: Expr, lerp: Expr) -> (IRBuilder) -> Expr {
    return {b in
        let oneMinusT = b.constant(1.0) - lerp
        let xScaled = x * oneMinusT
        let yScaled = y * lerp
        let mixed = xScaled + yScaled
        return mixed;
    }
}
 
// frontend
public enum LazyOp {
    case add, mul, gt, lt, gswitch, mix
    case historyWrite(CellID)
    case latch(CellID)
    case historyRead(CellID)
    case phasor(CellID)
    case accum(CellID)
    case constant(Float)
    case output(Int)

    func emit(_ thunk: (IRContext, NodeID) -> UOp, into ops: inout [UOp], ctx: IRContext, nodeId: NodeID) -> Lazy {
        let uop = thunk(ctx, nodeId)
        ops.append(uop)
        return uop.value
    }

    func emitAll(_ thunks: ((IRContext, NodeID) -> UOp)..., into ops: inout [UOp], ctx: IRContext, nodeId: NodeID) {
        for thunk in thunks {
            _ = emit(thunk, into: &ops, ctx: ctx, nodeId: nodeId)
        }
    }

    public func emit(ctx: IRContext, g: Graph, nodeId: NodeID) -> [UOp] {
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
               guard inputs.count == 2 else { fatalError("add requires 2 inputs") }
               b.use(val: b.value(inputs[0]) + b.value(inputs[1]))
           case .mul:
               guard inputs.count == 2 else { fatalError("mul requires 2 inputs") }
               b.use(val: b.value(inputs[0]) * b.value(inputs[1]))
           case .gt:
               guard inputs.count == 2 else { fatalError("gt requires 2 inputs") }
               b.use(val: b.value(inputs[0]) > b.value(inputs[1]))
           case .lt:
               guard inputs.count == 2 else { fatalError("gt requires 2 inputs") }
               b.use(val: b.value(inputs[0]) < b.value(inputs[1]))
           case .gswitch:
               guard inputs.count == 3 else { fatalError("gswitch rquires 3 inputs") }
               b.use(val: b.gswitch(b.value(inputs[0]), b.value(inputs[1]), b.value(inputs[2])))
           case .historyWrite(let cellId):
               guard inputs.count == 1 else { fatalError("history write requires 1 inputs") }
               b.use(val: b.store(cellId, b.value(inputs[0])))
           case .historyRead(let cellId):
               guard inputs.count == 0 else { fatalError("history read requires 0 inputs") }
               b.use(val: b.load(cellId))
           case .latch(let cellId):
               guard inputs.count == 2 else { fatalError("latch requires 2 inputs") }
               let value = b.value(inputs[0])
               let cond = b.value(inputs[1])
               b.use(val: u_latch(cellId, value: value, cond: cond)(b))
           case .mix:
               guard inputs.count == 3 else { fatalError("mix requires 3 inputs") }
               let (x,y,t) = b.values(inputs, count: 3)
               b.use(val: u_mix(x, y, lerp: t)(b))
           case .accum(let cellId):
               guard inputs.count == 4 else { fatalError("accum requires 4 inputs") }
               let (incr, reset, min, max) = b.values(inputs, count: 4)
               b.use(val: u_accum(cellId, incr: incr, reset: reset, min: min, max: max)(b))
           case .phasor(let cellId):
               guard inputs.count == 2 else { fatalError("phasor requires 2 inputs") }
               let (freq, reset) = b.values(inputs, count: 2)
               b.use(val: u_phasor(cellId, freq: freq, reset: reset)(b))
           case .output(let outputNumber):
               guard inputs.count == 1 else { fatalError("output requires 1 inputs") }
               b.use(val: b.output(outputNumber, b.value(inputs[0])))
        }
        ops.append(contentsOf: b.ops)
        return ops
    }
}


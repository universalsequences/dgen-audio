
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
}

public struct UOp {
    public let op: Op;
    public let value: Lazy;
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

let u_gt = binaryOp(Op.gt)
let u_lt = binaryOp(Op.lt)
let u_add = binaryOp(Op.add)
let u_div = binaryOp(Op.div)
let u_mul = binaryOp(Op.mul)
let u_sub = binaryOp(Op.sub)

func u_accum(_ cellId: CellID, incr: Expr, reset: Expr, min: Expr, max: Expr) -> (IRBuilder) -> [UOp] {
    return {b in
        let acc = b.load(cellId, b.nodeId)
        let newVal = acc + incr
        b.if(reset > b.constant(0)) {
            b.store(cellId, min)
            b.mutate(newVal, to: min)
        }

        b.store(cellId, newVal)

        b.if(newVal > max) {
            let wrapped = newVal - (max - min)
            b.store(cellId, wrapped)
        }
        
        b.use(val: acc)
        return b.ops
    } 
}

func u_phasor(_ cellId: CellID, freq: Expr, reset: Expr) -> (IRBuilder) -> [UOp] {
    return { b in
        let b_sr = b.constant(44100)
        return u_accum(cellId, incr: freq / b_sr, reset: reset, min: b.constant(0), max: b.constant(1))(b)
    } 
}

func u_latch(_ cellId: CellID, value: Expr, cond: Expr) -> (IRBuilder) -> Expr {
    return { b in
        let latched = b.load(cellId);
        b.if(cond > b.constant(0)) {
            b.store(cellId, value)
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
    case add, mul, load, gt, lt, gswitch, mix
    case historyWrite(CellID)
    case latch(CellID)
    case historyRead(CellID)
    case phasor(CellID)
    case accum(CellID)
    case constant(Float)

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
        let children: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
        var ops: [UOp] = []

        switch self {
        case .constant(let value):
            _ = ctx.useConstant(src: nodeId, value: value)
            return []

        case .load:
            guard node.inputs.count == 1,
                  case let .constant(cellIDFloat) = ctx.values[node.inputs[0]],
                  let cellID = CellID(exactly: cellIDFloat) else {
                fatalError("load expects constant cell ID")
            }
            let dest = ctx.useVariable(src: nodeId)
            let uop = UOp(op: .load(cellID), value: dest)
            ops.append(uop)
           case .add:
               guard children.count == 2 else { fatalError("add requires 2 inputs") }
               ops.append(u_add(children[0], children[1])(ctx, nodeId))
           case .mul:
               guard children.count == 2 else { fatalError("mul requires 2 inputs") }
               ops.append(u_mul(children[0], children[1])(ctx, nodeId))
           case .gt:
               guard children.count == 2 else { fatalError("gt requires 2 inputs") }
               ops.append(u_gt(children[0], children[1])(ctx, nodeId))
           case .lt:
               guard children.count == 2 else { fatalError("gt requires 2 inputs") }
               ops.append(u_lt(children[0], children[1])(ctx, nodeId))
           case .gswitch:
               guard children.count == 3 else { fatalError("gswitch rquires 3 inputs") }
               ops.append(u_switch(children[0], children[1], children[2])(ctx, nodeId))
           case .historyWrite(let cellId):
               guard children.count == 1 else { fatalError("history write requires 1 inputs") }
               let b = IRBuilder(ctx: ctx, nodeId: nodeId)
               b.store(cellId, b.value(children[0]))
               ops.append(contentsOf: b.ops)
           case .historyRead(let cellId):
               guard children.count == 0 else { fatalError("history read requires 0 inputs") }
               let dest = ctx.useVariable(src: nodeId)
               let uop = UOp(op: .load(cellId), value: dest)
               ops.append(uop)
           case .latch(let cellId):
               guard children.count == 2 else { fatalError("latch requires 2 inputs") }
               let builder = IRBuilder(ctx: ctx, nodeId: nodeId)
               let value = builder.value(children[0])
               let cond = builder.value(children[1])
               let latched = u_latch(cellId, value: value, cond: cond)(builder)
               builder.use(val: latched)
               ops.append(contentsOf: builder.ops)
           case .mix:
               guard children.count == 3 else { fatalError("mix requires 3 inputs") }
               let builder = IRBuilder(ctx: ctx, nodeId: nodeId)
               let a = builder.value(children[0])
               let b = builder.value(children[1])
               let t = builder.value(children[2])
               let out = u_mix(a, b, lerp: t)(builder)
               builder.use(val: out)
               ops.append(contentsOf: builder.ops)
           case .accum(let cellId):
               guard children.count == 4 else { fatalError("accum requires 4 inputs") }
               let b = IRBuilder(ctx: ctx, nodeId: nodeId)
               let incr  = b.value(children[0])
               let reset = b.value(children[1])
               let min   = b.value(children[2])
               let max   = b.value(children[3])

               ops.append(contentsOf: u_accum(cellId, incr: incr, reset: reset, min: min, max: max)(b))
           case .phasor(let cellId):
               guard children.count == 2 else { fatalError("phasor requires 2 inputs") }
               let b = IRBuilder(ctx: ctx, nodeId: nodeId)
               let freq = b.value(children[0])
               let reset = b.value(children[1])
               ops.append(contentsOf: u_phasor(cellId, freq: freq, reset: reset)(b))
        }
        return ops
    }
}


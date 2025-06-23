// the beauty is this doesn't need to even know if its forward or backward

public enum Device {
    case C
    case Metal
}

public class ScheduleItem {
    public var ops: [UOp] = []

    init() {}
}

func lowerUOpBlocks(_ uopBlocks: [BlockUOps], target: Device, ctx: IRContext) {
    var scheduleItems: [ScheduleItem] = []
    if (target == Device.C) {
        let scheduleItem = ScheduleItem();
        scheduleItems.append(scheduleItem);
        for (constantId, constant) in ctx.constants {
            scheduleItem.ops.append(UOp(op: .defineConstant(constantId, constant), value: .empty))
        }
        for globalId in ctx.globals {
            scheduleItem.ops.append(UOp(op: .defineGlobal(globalId), value: .global(globalId)))
        }
        scheduleItem.ops.append(UOp(op: .defineMemory(512), value: .empty))
    }
    for block in uopBlocks {
        if (target == Device.Metal) {
            scheduleItems.append(ScheduleItem())
        }
        let scheduleItem = scheduleItems[scheduleItems.count - 1]
        if (target == Device.C) {
            scheduleItem.ops.append(UOp(op: .begin_loop(block.kind == Kind.scalar ? 1 : 4), value: .empty))
        }
        for uop in block.ops {
            scheduleItem.ops.append(UOp(op: uop.op, value: uop.value, kind: block.kind))
        }
        // now lower each op into target op (should be about the same)
        if (target == Device.C) {
            scheduleItem.ops.append(UOp(op: .end_loop, value: .empty))
        }
    }

    var indent = 0;
    for scheduleItem in scheduleItems {
        for uop in scheduleItem.ops {
            var diff = 0
            switch uop.op {
            case .beginIf, .begin_loop:
                diff = 1
            case .endIf, .end_loop:
                indent -= 1
            default:
                break
            }
            print("\(String(repeating: "  ", count: indent))\(uop.prettyDescription())")
            indent += diff
        }
    }

    var body = ""

    for scheduleItem in scheduleItems {
        var indent = 0
        for uop in scheduleItem.ops {
            var diff = 0
            switch uop.op {
            case .beginIf, .begin_loop:
                diff = 1
            case .endIf, .end_loop:
                indent -= 1
            default:
                break
            }

            let cg = cg_C;
            body += "\(String(repeating: "  ", count: indent))\(cg(uop, ctx))\n"
            indent += diff
        }
    }
    print("\(String(repeating: "####\n", count: 10))Generated C Code:\(String(repeating:"\n", count: 4))")
    print(body)
}

func cg(_ lazy: Lazy, _ ctx: IRContext, _ kind: Kind? = nil, isOut: Bool) -> String {
    switch lazy {
    case let .constant(constantId, x):
        return kind == Kind.simd ? "c\(constantId)" : "\(x)";
    case let .variable(id,_):
        // if load simd then do
        return ctx.globals.contains(id) ? "\(!isOut && kind == .simd ? "simd\(id)" : kind == .simd ? " t\(id) + i" : "t\(id)[i]")" : "t\(id)";
    default:
        return "";
    }
}

func cgAssign(_ uop: UOp, _ gen: String, _ ctx: IRContext) -> String {
    let variable = cg(uop.value, ctx, uop.kind, isOut: true);
    switch (uop.kind) {
    case .simd:
        if (ctx.globals.contains(varId(uop.value))) {
            return "vst1q_f32(\(variable), \(gen));" 
        }
        // todo: is it a buffer? then we need to handle it
        return "float32x4_t \(variable) = \(gen);"
    default:
        if (ctx.globals.contains(varId(uop.value))) {
            return "\(variable) = \(gen);" 
        }
        // todo: is it a buffer? then we need to handle it
        return "float \(variable) = \(gen);"
    }
}

func cg_C(_ uop: UOp, _ ctx: IRContext) -> String {
    let assign: (String) -> String = {gen in
        return cgAssign(uop, gen, ctx);
    }
    let g: (Lazy) -> String = {a in
        return cg(a,ctx, uop.kind, isOut: false)
    }

    var body = ""
    switch uop.op {
    case let .defineConstant(constantId, value):
        return "float32x4_t c\(constantId )= vdupq_n_f32(\(value)f); "
    case let .defineGlobal(varId):
        return "float t\(varId)[128] __attribute__((aligned(16)));"
    case let .defineMemory(length):
        return "float memory[\(length)] __attribute__((aligned(16)));"
    case let .add(a,b):
        body = uop.kind == .scalar ? "\(g(a)) + \(g(b))" :
          "vaddq_f32(\(g(a)), \(g(b)))"
    case let .mul(a,b):
        body = uop.kind == .scalar ? "\(g(a)) * \(g(b))" :
          "vmulq_f32(\(g(a)), \(g(b)))"
    case let .sub(a,b):
        body = "\(g(a)) - \(g(b))"
    case let .div(a,b):
        body = "\(g(a)) / \(g(b))"
    case let .gt(a,b):
        body = "\(g(a)) > \(g(b))"
    case let .lt(a,b):
        body = uop.kind == .simd ?
          "vcvtq_f32_u32(vcltq_f32(\(g(a)), \(g(b))))" :
          "\(g(a)) < \(g(b))"
    case let .store(cell,val):
        return "memory[\(cell)] = \(g(val));"
    case let .load(cell):
        body = "memory[\(cell)]"
    case let .beginIf(cond):
        return "if (\(g(cond))) {"
    case let .mutate(a,b):
        return "\(g(a)) = \(g(b));"
    case .endIf:
        return "}"
    case .end_loop:
        return "}"
    case let .begin_loop(iter):
        return "for (int i=0; i < 128; i+=\(iter)) {"
    case let .loadGlobal(id):
        if uop.kind == .simd {
            return "float32x4_t simd\(id) = vld1q_f32(t\(id) + i);  "
        }
        return "/* \(uop.prettyDescription()) */";
    default:
        return "/* \(uop.prettyDescription()) */";
    }
    return assign(body)
}

func varId(_ lazy: Lazy) -> VarID {
    switch lazy {
    case let .variable(varid,_):
          return varid 
    default:
        fatalError("var id missing")
    }
}

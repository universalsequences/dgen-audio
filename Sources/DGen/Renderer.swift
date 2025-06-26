// the beauty is this doesn't need to even know if its forward or backward

public enum Device {
    case C
    case Metal
}

public class ScheduleItem {
    public var ops: [UOp] = []
    public let kind: Kind

    init(kind: Kind) {
        self.kind = kind
    }
}

func lowerUOpBlocks(_ uopBlocks: inout [BlockUOps], renderer: Renderer, ctx: IRContext, frameCount: Int) {
    var scheduleItems: [ScheduleItem] = []
    if renderer is CRenderer {
        let scheduleItem = ScheduleItem(kind: .scalar); // C renderer is always scalar
        scheduleItems.append(scheduleItem);
        for (constantId, constant) in ctx.constants {
            scheduleItem.ops.append(UOp(op: .defineConstant(constantId, constant), value: .empty))
        }
        scheduleItem.ops.append(UOp(op: .defineMemory(512), value: .empty))

        var defineGlobals: [UOp] = []
        for i in 0..<uopBlocks.count {
            uopBlocks[i].ops.removeAll(where: { uop in
                                                if case .defineGlobal = uop.op {
                                                    defineGlobals.append(uop)
                                                    return true
                                                }
                                                return false
                                            })
        }
        scheduleItem.ops.append(contentsOf: defineGlobals)
    }

    for block in uopBlocks {
        if renderer is MetalRenderer {
            let scheduleItem = ScheduleItem(kind: block.kind)
            scheduleItems.append(scheduleItem)
        }
        let scheduleItem = scheduleItems[scheduleItems.count - 1]
        if renderer is CRenderer {
            scheduleItem.ops.append(UOp(op: .beginLoop(128, block.kind == Kind.scalar ? 1 : 4), value: .empty))
        } else if renderer is MetalRenderer {
            switch block.kind {
            case .scalar:
                scheduleItem.ops.append(UOp(op: .beginRange(0, 1), value: .empty))
                scheduleItem.ops.append(UOp(op: .beginLoop(frameCount, 1), value: .empty))
            default:
                break
            }           
        }
        for uop in block.ops {
            scheduleItem.ops.append(UOp(op: uop.op, value: uop.value, kind: block.kind))
        }
        if renderer is MetalRenderer {
            if (block.kind == .scalar) {
                scheduleItem.ops.append(UOp(op: .endRange, value: .empty))
                scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
            }
        } else {
            scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
        }
    }

    for i in 0..<scheduleItems.count {
        let body = renderer.render(name: "kernel_\(i)", scheduleItem: scheduleItems[i], ctx: ctx)
        print("Generated \(renderer is CRenderer ? "C" : "Metal") (\(scheduleItems[i].kind)) Code:\(String(repeating:"\n", count: 1))")
        print(body)
    }
}

func varId(_ lazy: Lazy) -> VarID {
    switch lazy {
    case let .variable(varid,_):
        return varid
    default:
        fatalError("var id missing")
    }
}

open class Renderer {
    open func prepareSchedule(_ scheduleItems: inout [ScheduleItem], _ blocks: [BlockUOps], _ ctx: IRContext, _ frameCount: Int) {}

    open func render(name: String, scheduleItem: ScheduleItem, ctx: IRContext) -> String {
        fatalError("must be implemented by subclass")
    }
}

class CRenderer: Renderer {
    override func render(name: String, scheduleItem: ScheduleItem, ctx: IRContext) -> String {
        var body = ""
        var indent = 0
        for uop in scheduleItem.ops {
            var diff = 0
            switch uop.op {
            case .beginIf, .beginLoop:
                diff = 1
            case .endIf, .endLoop:
                indent -= 1
            default:
                break
            }

            body += "\(String(repeating: "  ", count: indent))\(cg(uop, ctx))\n"
            indent += diff
        }
        return body
    }

    func cg(_ uop: UOp, _ ctx: IRContext) -> String {
        let assign: (String) -> String = {gen in
            return self.cgAssign(uop, gen, ctx);
        }
        let g: (Lazy) -> String = {a in
            return self.cg(a,ctx, uop.kind, isOut: false)
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
        case .endLoop:
            return "}"
        case let .beginLoop(iters,step):
            return "for (int i=0; i < \(iters); i+=\(step)) {"
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

    func cg(_ lazy: Lazy, _ ctx: IRContext, _ kind: Kind? = nil, isOut: Bool) -> String {
        switch lazy {
        case let .constant(constantId, x):
            return kind == Kind.simd ? "c\(constantId)" : "\(x)";
        case let .variable(id,_):
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
            return "float32x4_t \(variable) = \(gen);"
        default:
            if (ctx.globals.contains(varId(uop.value))) {
                return "\(variable) = \(gen);"
            }
            return "float \(variable) = \(gen);"
        }
    }
}

class MetalRenderer: Renderer {
    let memoryVarID = -1 // Virtual ID for the global memory buffer

    override func render(name: String, scheduleItem: ScheduleItem, ctx: IRContext) -> String {
        var kernels = ""
        var (inputs, outputs) = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)

        let hasMemoryOps = scheduleItem.ops.contains { uop in
            if case .load = uop.op { return true }
            if case .store = uop.op { return true }
            return false
        }

        if hasMemoryOps {
            if !inputs.contains(memoryVarID) { inputs.append(memoryVarID) }
            if !outputs.contains(memoryVarID) { outputs.append(memoryVarID) }
        }

        let allBuffers = Set(inputs + outputs)

        var parameters: [String] = []
        for (i, bufferId) in allBuffers.sorted().enumerated() {
            let bufferName = bufferId == memoryVarID ? "memory" : "t\(bufferId)"
            parameters.append("    device float *\(bufferName) [[buffer(\(i))]]")
        }

        parameters.append("    uint id [[thread_position_in_grid]]")

        kernels += "kernel void \(name)(\n"
        kernels += parameters.joined(separator: ",\n")
        kernels += "\n) {\n"
        
        var indent = 1

       
        for uop in scheduleItem.ops {
            var diff = 0
            switch uop.op {
            case .beginIf, .beginLoop, .beginRange:
                diff = 1
            case .endIf, .endLoop, .endRange:
                indent -= 1
            default:
                break
            }

            kernels += "\(String(repeating: "  ", count: indent))\(cg(uop, ctx, scheduleItem.kind))\n"
            indent += diff
        }

        kernels += "}\n\n"
        return kernels
    }

    func analyzeDependencies(scheduleItem: ScheduleItem, ctx: IRContext) -> (inputs: [VarID], outputs: [VarID]) {
        var inputs: Set<VarID> = []
        var outputs: Set<VarID> = []
        var defined: Set<VarID> = []

        for uop in scheduleItem.ops {
            switch uop.op {
            case let .defineGlobal(varId):
                outputs.insert(varId)
            case let .loadGlobal(varId):
                inputs.insert(varId)
            default:
                break
            }
        }

        return (inputs: Array(inputs), outputs: Array(outputs))
    }

    func cg(_ uop: UOp, _ ctx: IRContext, _ scheduleKind: Kind) -> String {
        let assign: (String) -> String = {gen in
            return self.cgAssign(uop, gen, ctx, scheduleKind);
        }
        let g: (Lazy) -> String = {a in
            return self.cg(a,ctx, uop.kind, isOut: false, scheduleKind: scheduleKind)
        }

        var body = ""
        switch uop.op {
        case .defineMemory(_):
            return ""
        case let .add(a,b):
            body = "\(g(a)) + \(g(b))"
        case let .mul(a,b):
            body = "\(g(a)) * \(g(b))"
        case let .sub(a,b):
            body = "\(g(a)) - \(g(b))"
        case let .div(a,b):
            body = "\(g(a)) / \(g(b))"
        case let .gt(a,b):
            body = "\(g(a)) > \(g(b))"
        case let .lt(a,b):
            body = "\(g(a)) < \(g(b))"
        case let .store(cell,val):
            return "memory[\(cell)] = \(g(val));"
        case let .load(cell):
            body = "memory[\(cell)]"
        case let .beginIf(cond):
            return "if (\(g(cond))) {"
        case let .mutate(a,b):
            return "\(g(a)) = \(g(b));"
        case let .beginLoop(iters,step):
            return "for (int i=0; i < \(iters); i+=\(step)) {"
        case .endLoop:
            return "}"
        case let .beginRange(start,end):
            return "if (id >= \(start) && id < \(end)) {"
        case .endRange:
            return "}"
        case .endIf:
            return "}"
        default:
            return "/* \(uop.prettyDescription()) */";
        }
        return assign(body)
    }

    func cg(_ lazy: Lazy, _ ctx: IRContext, _ kind: Kind? = nil, isOut: Bool, scheduleKind: Kind) -> String {
        switch lazy {
        case let .constant(_, x):
            return "\(x)"
        case let .variable(id,_):
            if ctx.globals.contains(id) {
                let index = scheduleKind == .simd ? "id" : "i"
                return "t\(id)[\(index)]"
            }
            return "t\(id)"
        default:
            return "";
        }
    }

    func cgAssign(_ uop: UOp, _ gen: String, _ ctx: IRContext, _ scheduleKind: Kind) -> String {
        let variable = cg(uop.value, ctx, uop.kind, isOut: true, scheduleKind: scheduleKind);
        let type = "float"

        if ctx.globals.contains(varId(uop.value)) {
            return "\(variable) = \(gen);"
        }
        
        return "\(type) \(variable) = \(gen);"
    }
}

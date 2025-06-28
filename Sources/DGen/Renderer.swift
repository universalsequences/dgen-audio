// the beauty is this doesn't need to even know if its forward or backward

public struct BlockUOps {
    public var ops: [UOp]
    public let kind: Kind
    
    public init(ops: [UOp], kind: Kind) {
        self.ops = ops
        self.kind = kind
    }
}

public enum Device {
    case C
    case Metal
}

public struct CompiledKernel {
    public let name: String
    public let source: String
    public let kind: Kind
    public let buffers: [String]  // names of inputs/outputs
    public let threadGroupSize: Int  // for Metal
}

public class ScheduleItem {
    public var ops: [UOp] = []
    public let kind: Kind

    init(kind: Kind) {
        self.kind = kind
    }
}

public func lowerUOpBlocks(
  _ uopBlocks: inout [BlockUOps],
  renderer: Renderer,
  ctx: IRContext,
  frameCount: Int
) -> [CompiledKernel] {
    var scheduleItems: [ScheduleItem] = []
    renderer.prepareSchedule(&scheduleItems, uopBlocks, ctx, frameCount)
    return renderer.compile(scheduleItems: scheduleItems, ctx: ctx)
}

func extractVarId(_ lazy: Lazy) -> VarID {
    switch lazy {
    case let .variable(varid,_):
        return varid
    default:
        fatalError("var id missing")
    }
}

protocol UOpEmitter {
    func emit(_ uop: UOp, ctx: IRContext) -> String
    func emitLazy(_ lazy: Lazy, ctx: IRContext, kind: Kind?, isOut: Bool) -> String
}

open class Renderer {
    open func prepareSchedule(_ scheduleItems: inout [ScheduleItem], _ blocks: [BlockUOps], _ ctx: IRContext, _ frameCount: Int) {}

    open func compile(
      scheduleItems: [ScheduleItem],
      ctx: IRContext
    ) -> [CompiledKernel] {
        fatalError("must implement")
    }
    
    open func render(name: String, scheduleItem: ScheduleItem, ctx: IRContext) -> String {
        fatalError("must be implemented by subclass")
    }
}

public class CRenderer: Renderer {
    
    public override init() {}

    public override func compile(
      scheduleItems: [ScheduleItem],
      ctx: IRContext
    ) -> [CompiledKernel] {
        return scheduleItems.enumerated().map { (i, scheduleItem) in
            let source = render(name: "kernel_\(i)", scheduleItem: scheduleItem, ctx: ctx)

            // For C, the "buffers" are just the global arrays used
            var buffers: [String] = []

            for uop in scheduleItem.ops {
                switch uop.op {
                case let .defineGlobal(varId):
                    buffers.append("t\(varId)")
                case .defineMemory:
                    buffers.append("memory")
                default:
                    break
                }
            }

            return CompiledKernel(
       name: "kernel_\(i)",
       source: source,
       kind: scheduleItem.kind,
       buffers: buffers,
       threadGroupSize: 1 // C execution is scalar for now
     )
        }
    }
    
    public override func prepareSchedule(
      _ scheduleItems: inout [ScheduleItem],
      _ uopBlocks: [BlockUOps],
      _ ctx: IRContext,
      _ frameCount: Int
    ) {
        let scheduleItem = ScheduleItem(kind: .scalar)
        scheduleItems.append(scheduleItem)

        // Extract defineGlobal from all blocks
        for block in uopBlocks {
            scheduleItem.ops.append(UOp(op: .beginLoop(128, block.kind == .scalar ? 1 : 4), value: .empty))
            for uop in block.ops {
                switch uop.op {
                case .defineGlobal:
                    break
                default: 
                    scheduleItem.ops.append(UOp(op: uop.op, value: uop.value, kind: block.kind))
                }
            }
            scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
        }

        // Append all UOps inside a single loop
        scheduleItem.ops.append(UOp(op: .beginLoop(128, 1), value: .empty))

        scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
    }
    
    public override func render(name: String, scheduleItem: ScheduleItem, ctx: IRContext) -> String {
        var code: [String] = []

        // C includes and function signature
        code.append("""
                      #include <arm_neon.h>
                      #include <stdint.h>

                      """)

        // Declare globals
        let sortedGlobals = ctx.globals.sorted()
        for varId in sortedGlobals {
            code.append("float t\(varId)[128] __attribute__((aligned(16)));")
        }

     

        // Declare memory if needed
        code.append("float memory[512] __attribute__((aligned(16)));")

        code.append("void process(float *outputs, float *inputs, int frameCount) {")

   // Define constants and memory
        for (constantId, constant) in ctx.constants {
            let uop = UOp(op: .defineConstant(constantId, constant), value: .empty)
            code.append("  \(emit(uop, ctx: ctx))")
        }

        // Emit ops
        var indent = 1
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

            let line = emit(uop, ctx: ctx)
            let indentStr = String(repeating: "  ", count: indent)
            code.append("\(indentStr)\(line)")
            indent += diff
        }

        // Close process function
        code.append("}")

        return code.joined(separator: "\n")
    }


    func emit(_ uop: UOp, ctx: IRContext) -> String {
        let g = { self.emitLazy($0, ctx: ctx, kind: uop.kind, isOut: false) }

        switch uop.op {
        case let .defineConstant(constantId, val):
            return "float32x4_t c\(constantId) = vdupq_n_f32(\(val)f);"
        case let .defineGlobal(varId):
            return "float t\(varId)[128] __attribute__((aligned(16)));"
        case let .defineMemory(length):
            return "float memory[\(length)] __attribute__((aligned(16)));"

        case let .add(a, b):
            let expr = uop.kind == .simd ? "vaddq_f32(\(g(a)), \(g(b)))" : "\(g(a)) + \(g(b))"
            return emitAssign(uop, expr, ctx)

        case let .mul(a, b):
            let expr = uop.kind == .simd ? "vmulq_f32(\(g(a)), \(g(b)))" : "\(g(a)) * \(g(b))"
            return emitAssign(uop, expr, ctx)

        case let .sub(a, b): return emitAssign(uop, "\(g(a)) - \(g(b))", ctx)
        case let .div(a, b): return emitAssign(uop, "\(g(a)) / \(g(b))", ctx)

        case let .gt(a, b): return emitAssign(uop, "\(g(a)) > \(g(b))", ctx)
        case let .lt(a, b):
            let expr = uop.kind == .simd ?
              "vcvtq_f32_u32(vcltq_f32(\(g(a)), \(g(b))))" :
              "\(g(a)) < \(g(b))"
            return emitAssign(uop, expr, ctx)

        case let .store(cell, val): return "memory[\(cell)] = \(g(val));"
        case let .load(cell): return emitAssign(uop, "memory[\(cell)]", ctx)

        case let .beginIf(cond): return "if (\(g(cond))) {"
        case .endIf: return "}"

        case let .mutate(a, b): return "\(emitLazy(a, ctx: ctx, kind: uop.kind, isOut: true)) = \(g(b));"

        case let .output(channel, val):
            print("value to output = \(val)")
                if uop.kind == .simd {
                    let ptr = "outputs + \(channel) * 128 + i"
                    return "vst1q_f32(\(ptr), \(g(val)));"
                } else {
                    let addr = "outputs[\(channel) * 128 + i]"
                    return "\(addr) = \(g(val));"
                }
            

        case let .beginLoop(iters, step): return "for (int i = 0; i < \(iters); i += \(step)) {"
        case .endLoop: return "}"

        case let .loadGlobal(id):
            return uop.kind == .simd
              ? "float32x4_t simd\(id) = vld1q_f32(t\(id) + i);"
              : "/* \(uop.prettyDescription()) */"

        default:
            return "/* \(uop.prettyDescription()) */"
        }
    }

    func emitLazy(_ lazy: Lazy, ctx: IRContext, kind: Kind?, isOut: Bool) -> String {
        switch lazy {
        case let .constant(constantId, val):
            return kind == .simd ? "c\(constantId)" : "\(val)"
        case let .variable(id, _):
            if ctx.globals.contains(id) {
                if kind == .simd {
                    return isOut ? "t\(id) + i" : "simd\(id)"
                } else {
                    return "t\(id)[i]"
                }
            } else {
                return "t\(id)"
            }
        default:
            return "/* unknown lazy */"
        }
    }

    func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext) -> String {
        let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
        let isGlobal = ctx.globals.contains(extractVarId(uop.value))

        if uop.kind == .simd {
            return isGlobal
              ? "vst1q_f32(\(lhs), \(expr));"
              : "float32x4_t \(lhs) = \(expr);"
        } else {
            return isGlobal
              ? "\(lhs) = \(expr);"
              : "float \(lhs) = \(expr);"
        }
    }
}

public class MetalRenderer: Renderer, UOpEmitter {
    let memoryVarID = -1 // Virtual ID for the global memory buffer

    public override init() {}

    public override func compile(
      scheduleItems: [ScheduleItem],
      ctx: IRContext
    ) -> [CompiledKernel] {
        return scheduleItems.enumerated().map { (i, scheduleItem) in
            let source = render(name: "kernel_\(i)", scheduleItem: scheduleItem, ctx: ctx)
            let deps = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)
            var buffers = deps.inputs + deps.outputs
            
            // Check if this kernel has output operations
            let hasOutputOps = scheduleItem.ops.contains { uop in
                if case .output = uop.op { return true }
                return false
            }
            
            var bufferNames: [String] = []
            if hasOutputOps {
                bufferNames.append("outputs")
            }
            bufferNames.append(contentsOf: buffers.map { $0 == -1 ? "memory" : "t\($0)" })
            
            return CompiledKernel(
              name: "kernel_\(i)",
              source: source,
              kind: scheduleItem.kind,
              buffers: bufferNames,
              threadGroupSize: scheduleItem.kind == .simd ? 128 : 1
            )
        }
    }
    
    public override func prepareSchedule(
      _ scheduleItems: inout [ScheduleItem],
      _ uopBlocks: [BlockUOps],
      _ ctx: IRContext,
      _ frameCount: Int
    ) {
        for block in uopBlocks {
            let scheduleItem = ScheduleItem(kind: block.kind)

            // Optional: extract defineGlobals early (or leave in place if your IR handles it)
            for uop in block.ops {
                if case .defineGlobal = uop.op {
                    scheduleItem.ops.append(uop)
                }
            }

            if block.kind == .scalar {
                var beginRange = UOp(op: .beginRange(0, 1), value: .empty)
                beginRange.kind = block.kind
                scheduleItem.ops.append(beginRange)
                
                var beginLoop = UOp(op: .beginLoop(frameCount, 1), value: .empty)
                beginLoop.kind = block.kind
                scheduleItem.ops.append(beginLoop)
            }

            for uop in block.ops {
                if case .defineGlobal = uop.op { continue }
                var typedUOp = uop
                typedUOp.kind = block.kind
                scheduleItem.ops.append(typedUOp)
            }

            if block.kind == .scalar {
                var endLoop = UOp(op: .endLoop, value: .empty)
                endLoop.kind = block.kind
                scheduleItem.ops.append(endLoop)
                
                var endRange = UOp(op: .endRange, value: .empty)
                endRange.kind = block.kind
                scheduleItem.ops.append(endRange)
            }

            scheduleItems.append(scheduleItem)
        }
    }

    public override func render(name: String, scheduleItem: ScheduleItem, ctx: IRContext) -> String {
        var kernels = ""
        var (inputs, outputs) = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)

        let hasOutputOps = scheduleItem.ops.contains { uop in
            if case .output = uop.op { return true }
            return false
        }

        let allBuffers = Set(inputs + outputs)

        var parameters: [String] = []
        var bufferIndex = 0
        
        // Add outputs buffer first if needed
        if hasOutputOps {
            parameters.append("    device float *outputs [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }
        
        // Add other buffers
        for bufferId in allBuffers.sorted() {
            let bufferName = bufferId == memoryVarID ? "memory" : "t\(bufferId)"
            parameters.append("    device float *\(bufferName) [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
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

            kernels += "\(String(repeating: "  ", count: indent))\(emit(uop, ctx: ctx))\n"
            indent += diff
        }

        kernels += "}\n\n"
        return kernels
    }

    func analyzeDependencies(scheduleItem: ScheduleItem, ctx: IRContext) -> (inputs: [VarID], outputs: [VarID]) {
        var inputs: Set<VarID> = []
        var outputs: Set<VarID> = []
        var needsMemory = false

        for uop in scheduleItem.ops {
            switch uop.op {
            case let .defineGlobal(varId): 
                outputs.insert(varId)
            case let .loadGlobal(varId): 
                inputs.insert(varId)
            case .load, .store:
                needsMemory = true
            case .defineMemory:
                needsMemory = true
            default: 
                break
            }
        }

        // Add memory buffer if needed
        if needsMemory {
            inputs.insert(memoryVarID)
            outputs.insert(memoryVarID)
        }

        return (inputs: Array(inputs), outputs: Array(outputs))
    }

    func emit(_ uop: UOp, ctx: IRContext) -> String {
        let g = { self.emitLazy($0, ctx: ctx, kind: uop.kind, isOut: false) }

        switch uop.op {
        case .defineMemory: return ""

        case let .add(a, b): return emitAssign(uop, "\(g(a)) + \(g(b))", ctx)
        case let .mul(a, b): return emitAssign(uop, "\(g(a)) * \(g(b))", ctx)
        case let .sub(a, b): return emitAssign(uop, "\(g(a)) - \(g(b))", ctx)
        case let .div(a, b): return emitAssign(uop, "\(g(a)) / \(g(b))", ctx)

        case let .gt(a, b): return emitAssign(uop, "\(g(a)) > \(g(b))", ctx)
        case let .lt(a, b): return emitAssign(uop, "\(g(a)) < \(g(b))", ctx)

        case let .load(cell): return emitAssign(uop, "memory[\(cell)]", ctx)
        case let .store(cell, val): return "memory[\(cell)] = \(g(val));"

        case let .mutate(a, b): return "\(emitLazy(a, ctx: ctx, kind: uop.kind, isOut: true)) = \(g(b));"
        case let .beginIf(cond): return "if (\(g(cond))) {"
        case .endIf: return "}"

        case let .beginLoop(iters, step): return "for (int i = 0; i < \(iters); i += \(step)) {"
        case .endLoop: return "}"

        case let .beginRange(start, end): return "if (id >= \(start) && id < \(end)) {"
        case .endRange: return "}"

        case let .output(channel, val):
            // Store output value to a device buffer that can be read back
            let idx = (uop.kind == .simd) ? "id" : "i"
            return "outputs[\(channel) * 128 + \(idx)] = \(g(val));"

        default:
            return "/* \(uop.prettyDescription()) */"
        }
    }

    func emitLazy(_ lazy: Lazy, ctx: IRContext, kind: Kind?, isOut: Bool) -> String {
        switch lazy {
        case let .constant(_, val): return "\(val)"
        case let .variable(id, _):
            if ctx.globals.contains(id) {
                let idx = (kind == .simd) ? "id" : "i"
                return "t\(id)[\(idx)]"
            } else {
                return "t\(id)"
            }
        default: return "/* unknown lazy */"
        }
    }

    func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext) -> String {
        let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
        if ctx.globals.contains(extractVarId(uop.value)) {
            return "\(lhs) = \(expr);"
        }
        return "float \(lhs) = \(expr);"
    }
}

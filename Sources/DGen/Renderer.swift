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
    public let threadGroupSize: Int?  // for Metal: nil means runtime-determined, 1 for scalar
    public let memorySize: Int  // Required memory allocation size in floats
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
    frameCount: Int,
    graph: Graph,
    totalMemorySlots: Int
) -> [CompiledKernel] {
    var scheduleItems: [ScheduleItem] = []
    renderer.prepareSchedule(&scheduleItems, uopBlocks, ctx, frameCount)
    return renderer.compile(
        scheduleItems: scheduleItems, ctx: ctx, graph: graph, totalMemorySlots: totalMemorySlots)
}

func extractVarId(_ lazy: Lazy) -> VarID {
    switch lazy {
    case let .variable(varid, _):
        return varid
    case let .global(varid):
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
    open func prepareSchedule(
        _ scheduleItems: inout [ScheduleItem], _ blocks: [BlockUOps], _ ctx: IRContext,
        _ frameCount: Int
    ) {}

    open func compile(
        scheduleItems: [ScheduleItem],
        ctx: IRContext,
        graph: Graph,
        totalMemorySlots: Int
    ) -> [CompiledKernel] {
        fatalError("must implement")
    }

    open func render(
        name: String, scheduleItem: ScheduleItem, ctx: IRContext, graph: Graph,
        totalMemorySlots: Int
    ) -> String {
        fatalError("must be implemented by subclass")
    }
}

public class CRenderer: Renderer {

    public override init() {}

    public override func compile(
        scheduleItems: [ScheduleItem],
        ctx: IRContext,
        graph: Graph,
        totalMemorySlots: Int
    ) -> [CompiledKernel] {
        return scheduleItems.enumerated().map { (i, scheduleItem) in
            let source = render(
                name: "kernel_\(i)", scheduleItem: scheduleItem, ctx: ctx, graph: graph,
                totalMemorySlots: totalMemorySlots)

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
                threadGroupSize: 1,  // C execution is scalar for now
                memorySize: max(totalMemorySlots, 1024)  // Match memory size calculation from render method
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

        // Add frameCount UOp that will render to function parameter
        scheduleItem.ops.append(UOp(op: .frameCount, value: .empty))
        let frameCountUOp = Lazy.variable(-1, nil)  // Special variable ID for frameCount

        // Extract defineGlobal from all blocks
        for block in uopBlocks {
            scheduleItem.ops.append(
                UOp(op: .beginLoop(frameCountUOp, block.kind == .scalar ? 1 : 4), value: .empty))
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
        scheduleItem.ops.append(UOp(op: .beginLoop(frameCountUOp, 1), value: .empty))

        scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
    }

    public override func render(
        name: String, scheduleItem: ScheduleItem, ctx: IRContext, graph: Graph,
        totalMemorySlots: Int
    ) -> String {
        var code: [String] = []

        // C includes and function signature
        code.append(
            """
            #include <arm_neon.h>
            #include <stdint.h>
            #include <stdio.h>
            #include <math.h>
            #include <Accelerate/Accelerate.h>

            float32x4_t vfmodq_f32(float32x4_t a, float32x4_t b) {
              float32x4_t div = vdivq_f32(a, b);
              float32x4_t div_trunc = vrndq_f32(div);  // truncates toward zero
              return vsubq_f32(a, vmulq_f32(b, div_trunc));
            }

            """)

        // Declare globals
        let sortedGlobals = ctx.globals.sorted()
        for varId in sortedGlobals {
            code.append("float t\(varId)[128] __attribute__((aligned(16))) = {0};")
        }

        // Memory is now passed as parameter - calculate required size for external allocation
        let memorySize = max(totalMemorySlots, 1024)  // Minimum of 1024 for safety
        code.append("// Memory size required: \(memorySize) floats")

        // Entry point for setting parameter
        code.append(
            """

            void setParamValue(int cellId, float val) {
              //memory[cellId] = val;
            }

            """)

        code.append(
            "void process(float *const *in, float *const *out, int nframes, void *state) {")

        // Use audiograph parameters directly - no mapping needed
        code.append("  int frameCount = nframes;  // Use audiograph frame count parameter")

        // Define constants and memory
        for (constantId, constant) in ctx.constants {
            let uop = UOp(op: .defineConstant(constantId, constant), value: .empty)
            code.append("  \(emit(uop, ctx: ctx))")
        }

        // Cast state parameter to float pointer for use in function
        code.append("  float *memory = (float*)state;")

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

        case let .sub(a, b):
            let expr = uop.kind == .simd ? "vsubq_f32(\(g(a)), \(g(b)))" : "\(g(a)) - \(g(b))"
            return emitAssign(uop, expr, ctx)

        case let .div(a, b):
            let expr = uop.kind == .simd ? "vdivq_f32(\(g(a)), \(g(b)))" : "\(g(a)) / \(g(b))"
            return emitAssign(uop, expr, ctx)

        case let .mod(a, b):
            let expr =
                uop.kind == .simd ? "vfmodq_f32(\(g(a)), \(g(b)))" : "fmodf(\(g(a)), \(g(b)))"
            return emitAssign(uop, expr, ctx)

        case let .pow(a, b):
            let expr = uop.kind == .simd ? "vpowf(\(g(a)), \(g(b)))" : "powf(\(g(a)), \(g(b)))"
            return emitAssign(uop, expr, ctx)

        case let .min(a, b):
            let expr = uop.kind == .simd ? "vminq_f32(\(g(a)), \(g(b)))" : "fminf(\(g(a)), \(g(b)))"
            return emitAssign(uop, expr, ctx)

        case let .max(a, b):
            let expr = uop.kind == .simd ? "vmaxq_f32(\(g(a)), \(g(b)))" : "fmaxf(\(g(a)), \(g(b)))"
            return emitAssign(uop, expr, ctx)

        case let .abs(a):
            let expr = uop.kind == .simd ? "vabsq_f32(\(g(a)))" : "fabs(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .sign(a):
            if uop.kind == .simd {
                // For SIMD: return -1.0 for negative, 1.0 for positive, 0.0 for zero
                let expr =
                    "vbslq_f32(vcltq_f32(\(g(a)), vdupq_n_f32(0.0f)), vdupq_n_f32(-1.0f), vbslq_f32(vcgtq_f32(\(g(a)), vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)))"
                return emitAssign(uop, expr, ctx)
            } else {
                // For scalar: use copysignf for efficient sign extraction
                return emitAssign(uop, "(\(g(a)) == 0.0f) ? 0.0f : copysignf(1.0f, \(g(a)))", ctx)
            }

        case let .floor(a):
            let expr = uop.kind == .simd ? "vrndmq_f32(\(g(a)))" : "floorf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .ceil(a):
            let expr = uop.kind == .simd ? "vrndpq_f32(\(g(a)))" : "ceilf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .round(a):
            let expr =
                uop.kind == .simd
                ? "vrndaq_f32(\(g(a)))"
                : "roundf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .memoryRead(base, offset):
            if uop.kind == .simd {
                // For SIMD: gather 4 values from potentially different memory locations
                let offsetExpr = g(offset)
                let gatherExpr = """
                    (float32x4_t){
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 0)],
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 1)],
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 2)],
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 3)]
                    }
                    """.trimmingCharacters(in: .whitespacesAndNewlines)
                return emitAssign(uop, gatherExpr, ctx)
            } else {
                return emitAssign(uop, "memory[\(base) + (int)\(g(offset))]", ctx)
            }

        case let .memoryWrite(base, offset, value):
            if uop.kind == .simd {
                // For SIMD: scatter 4 values to potentially different memory locations
                let offsetExpr = g(offset)
                let valueExpr = g(value)
                return """
                    memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 0)] = vgetq_lane_f32(\(valueExpr), 0);
                    memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 1)] = vgetq_lane_f32(\(valueExpr), 1);
                    memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 2)] = vgetq_lane_f32(\(valueExpr), 2);
                    memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 3)] = vgetq_lane_f32(\(valueExpr), 3);
                    """.trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                return "memory[\(base) + (int)\(g(offset))] = \(g(value));"
            }

        case let .sin(a):
            let expr = uop.kind == .simd ? "vsinf(\(g(a)))" : "sinf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .cos(a):
            let expr = uop.kind == .simd ? "vcosf(\(g(a)))" : "cosf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .tan(a):
            let expr = uop.kind == .simd ? "vtanf(\(g(a)))" : "tanf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .tanh(a):
            let expr = uop.kind == .simd ? "vtanhf(\(g(a)))" : "tanhf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .exp(a):
            let expr = uop.kind == .simd ? "vexpf(\(g(a)))" : "expf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .log(a):
            let expr = uop.kind == .simd ? "vlogf(\(g(a)))" : "logf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .log10(a):
            let expr =
                uop.kind == .simd
                ? "vmulq_f32(vlogf(\(g(a))), vdupq_n_f32((float)M_LOG10E))"  // log10(x) = ln(x) * log10(e)
                : "log10f(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .sqrt(a):
            let expr = uop.kind == .simd ? "vsqrtf(\(g(a)))" : "sqrtf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case let .atan2(y, x):
            let expr = uop.kind == .simd ? "vatan2f(\(g(y)), \(g(x)))" : "atan2f(\(g(y)), \(g(x)))"
            return emitAssign(uop, expr, ctx)

        case let .gt(a, b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcgtq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) > \(g(b))"
            return emitAssign(uop, expr, ctx)
        case let .gte(a, b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcgeq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) >= \(g(b))"
            return emitAssign(uop, expr, ctx)
        case let .lte(a, b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcleq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) <= \(g(b))"
            return emitAssign(uop, expr, ctx)
        case let .lt(a, b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcltq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) < \(g(b))"
            return emitAssign(uop, expr, ctx)
        case let .eq(a, b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vceqq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) == \(g(b))"
            return emitAssign(uop, expr, ctx)

        case let .gswitch(cond, a, b):
            if uop.kind == .simd {
                // For SIMD: use vbslq_f32 to select between a and b based on condition > 0
                let mask = "vcgtq_f32(\(g(cond)), vdupq_n_f32(0.0f))"
                let expr = "vbslq_f32(\(mask), \(g(a)), \(g(b)))"
                return emitAssign(uop, expr, ctx)
            } else {
                let expr = "\(g(cond)) > 0.0f ? \(g(a)) : \(g(b))"
                return emitAssign(uop, expr, ctx)
            }
        case let .delay1(cell, curr):
            if uop.kind == .simd {
                // Return delayed value, and also persist current vector into memory for next chunk
                let expr = "vextq_f32(vld1q_f32(&memory[\(cell)]), \(g(curr)), 3)"
                let assign = emitAssign(uop, expr, ctx)
                return "\(assign) vst1q_f32(&memory[\(cell)], \(g(curr)));"
            } else {
                // Scalar: read previous then write current
                let assign = emitAssign(uop, "memory[\(cell)]", ctx)
                return "\(assign) memory[\(cell)] = \(g(curr));"
            }
        case let .selector(mode, options):
            if uop.kind == .simd {
                // For SIMD: if mode <= 0 return 0, if mode <= 1 return options[0], etc.
                var expr = "vdupq_n_f32(0.0f)"  // Default to 0 if mode <= 0

                // Build the selector from the end backwards
                for (i, option) in options.enumerated().reversed() {
                    let threshold = "vdupq_n_f32(\(Float(i + 1))f)"
                    let cond = "vcleq_f32(\(g(mode)), \(threshold))"
                    expr = "vbslq_f32(\(cond), \(g(option)), \(expr))"
                }

                // Final check for mode <= 0
                let zeroVec = "vdupq_n_f32(0.0f)"
                let condZero = "vcleq_f32(\(g(mode)), \(zeroVec))"
                expr = "vbslq_f32(\(condZero), \(zeroVec), \(expr))"

                return emitAssign(uop, expr, ctx)
            } else {
                // For scalar: if mode <= 0 return 0, if mode <= 1 return options[0], etc.
                var expr = "(\(g(mode)) <= 0.0f ? 0.0f"
                for (i, option) in options.enumerated() {
                    expr += " : \(g(mode)) <= \(Float(i + 1))f ? \(g(option))"
                }
                expr += " : 0.0f)"
                return emitAssign(uop, expr, ctx)
            }

        case let .store(cell, val):
            if uop.kind == .simd {
                // For SIMD: store all 4 vector elements to consecutive memory slots
                return "vst1q_f32(&memory[\(cell)], \(g(val)));"
            } else {
                return "memory[\(cell)] = \(g(val));"
            }
        case let .load(cell):
            if uop.kind == .simd {
                // For SIMD: load 4 consecutive memory slots into vector
                return emitAssign(uop, "vld1q_f32(&memory[\(cell)])", ctx)
            } else {
                return emitAssign(uop, "memory[\(cell)]", ctx)
            }

        case let .beginIf(cond): return "if (\(g(cond))) {"
        case .endIf: return "}"

        case let .mutate(a, b):
            return "\(emitLazy(a, ctx: ctx, kind: uop.kind, isOut: true)) = \(g(b));"

        case let .input(channel):
            if uop.kind == .simd {
                // For SIMD: load 4 consecutive memory slots into vector
                let ptr = "in[\(channel)] + i"
                return emitAssign(uop, "vld1q_f32(\(ptr))", ctx)
            } else {
                let addr = "in[\(channel)][i]"
                return emitAssign(uop, "\(addr)", ctx)
            }
        case let .output(channel, val):
            if uop.kind == .simd {
                // For audiograph compatibility: use out[channel] directly
                let ptr = "out[\(channel)] + i"
                return "vst1q_f32(\(ptr), \(g(val)));"
            } else {
                // For audiograph compatibility: use out[channel][i] directly
                let addr = "out[\(channel)][i]"
                return "\(addr) = \(g(val));"
            }

        case let .beginLoop(iters, step): return "for (int i = 0; i < \(g(iters)); i += \(step)) {"
        case .endLoop: return "}"

        case .frameCount: return "/* frameCount available as function parameter */"

        case let .loadGlobal(id):
            if uop.kind == .simd {
                // Create a proper SIMD variable declaration for loadGlobal
                return "float32x4_t simd\(id) = vld1q_f32(t\(id) + i);"
            } else {
                return emitAssign(uop, "t\(id)[i]", ctx)
            }

        default:
            return "/* \(uop.prettyDescription()) */"
        }
    }

    func emitLazy(_ lazy: Lazy, ctx: IRContext, kind: Kind?, isOut: Bool) -> String {
        switch lazy {
        case let .constant(constantId, val):
            return kind == .simd ? "c\(constantId)" : "\(val)"
        case let .variable(id, _):
            if id == -1 {  // Special case for frameCount
                return "frameCount"
            } else if ctx.globals.contains(id) {
                if kind == .simd {
                    return isOut ? "t\(id) + i" : "simd\(id)"
                } else {
                    return "t\(id)[i]"
                }
            } else {
                if kind == .simd {
                    return "simd\(id)"
                } else {
                    return "t\(id)"
                }
            }
        case let .global(id):
            // Global variables are always accessed through global buffers
            if kind == .simd {
                return isOut ? "t\(id) + i" : "simd\(id)"
            } else {
                return "t\(id)[i]"
            }
        default:
            return "/* unknown lazy */"
        }
    }

    func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext) -> String {
        let varId = extractVarId(uop.value)
        let isGlobal = ctx.globals.contains(varId)

        if uop.kind == .simd {
            if isGlobal {
                // For global variables, we need both a local variable declaration
                // AND a store to the global buffer for cross-block transfer
                let localVar = "simd\(varId)"
                let globalStore = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
                return "float32x4_t \(localVar) = \(expr); vst1q_f32(\(globalStore), \(localVar));"
            } else {
                let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
                return "float32x4_t \(lhs) = \(expr);"
            }
        } else {
            let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
            return isGlobal
                ? "\(lhs) = \(expr);"
                : "float \(lhs) = \(expr);"
        }
    }
}

public class MetalRenderer: Renderer, UOpEmitter {
    let memoryVarID = -1  // Virtual ID for the global memory buffer
    private var loadVarToCell: [VarID: Int] = [:]
    // Track the order of vector history store bases within a kernel
    private var vectorStoreOrder: [Int] = []

    public override init() {}

    public override func compile(
        scheduleItems: [ScheduleItem],
        ctx: IRContext,
        graph: Graph,
        totalMemorySlots: Int
    ) -> [CompiledKernel] {
        return scheduleItems.enumerated().map { (i, scheduleItem) in
            let source = render(
                name: "kernel_\(i)", scheduleItem: scheduleItem, ctx: ctx, graph: graph,
                totalMemorySlots: totalMemorySlots)
            let deps = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)
            let allBuffers = Set(deps.inputs + deps.outputs)

            // Check if this kernel has output operations
            let hasOutputOps = scheduleItem.ops.contains { uop in
                if case .output = uop.op { return true }
                return false
            }

            // Detect whether this kernel needs segmented dispatch (for delay/barrier semantics)
            let needsSegmenting: Bool = scheduleItem.ops.contains { uop in
                if case .delay1 = uop.op { return true }
                return false
            }

            var bufferNames: [String] = []
            if hasOutputOps {
                bufferNames.append("outputs")
            }
            // Use same sorted order as render() method
            for bufferId in allBuffers.sorted() {
                let bufferName = bufferId == memoryVarID ? "memory" : "t\(bufferId)"
                bufferNames.append(bufferName)
            }

            // Add frameCount buffer for all Metal kernels (needed for output operations)
            bufferNames.append("frameCount")

            // Add segment buffers if this kernel needs segmented execution
            if needsSegmenting {
                bufferNames.append("segmentLen")
                bufferNames.append("segmentBase")
            }

            return CompiledKernel(
                name: "kernel_\(i)",
                source: source,
                kind: scheduleItem.kind,
                buffers: bufferNames,
                threadGroupSize: scheduleItem.kind == .scalar ? 1 : nil,
                memorySize: max(totalMemorySlots, 1024)  // Match memory size calculation from render method
            )
        }
    }

    public override func prepareSchedule(
        _ scheduleItems: inout [ScheduleItem],
        _ uopBlocks: [BlockUOps],
        _ ctx: IRContext,
        _ frameCount: Int
    ) {
        // Define frameCount UOp for Metal kernels
        let frameCountUOp = Lazy.variable(-1, nil)  // Special variable ID for frameCount

        for block in uopBlocks {
            let scheduleItem = ScheduleItem(kind: block.kind)

            // Add frameCount UOp for all Metal kernels (needed for output operations)
            scheduleItem.ops.append(UOp(op: .frameCount, value: .empty))

            // Optional: extract defineGlobals early (or leave in place if your IR handles it)
            for uop in block.ops {
                if case .defineGlobal = uop.op {
                    scheduleItem.ops.append(uop)
                }
            }

            if block.kind == .scalar {
                // Scalar kernels: only thread 0 executes, then loops through frameCount
                var beginRange = UOp(
                    op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty)
                beginRange.kind = block.kind
                scheduleItem.ops.append(beginRange)

                var beginLoop = UOp(op: .beginLoop(frameCountUOp, 1), value: .empty)
                beginLoop.kind = block.kind
                scheduleItem.ops.append(beginLoop)
            } else {
                // SIMD kernels: each thread processes one frame, bound by frameCount
                var beginRange = UOp(op: .beginRange(.constant(0, 0), frameCountUOp), value: .empty)
                beginRange.kind = block.kind
                scheduleItem.ops.append(beginRange)
            }

            for uop in block.ops {
                if case .defineGlobal = uop.op { continue }
                var typedUOp = uop
                typedUOp.kind = block.kind
                scheduleItem.ops.append(typedUOp)
            }

            // Close the range/loop blocks
            if block.kind == .scalar {
                scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
                scheduleItem.ops.append(UOp(op: .endRange, value: .empty))
            } else {
                scheduleItem.ops.append(UOp(op: .endRange, value: .empty))
            }

            scheduleItems.append(scheduleItem)
        }
    }

    public override func render(
        name: String, scheduleItem: ScheduleItem, ctx: IRContext, graph: Graph,
        totalMemorySlots: Int
    ) -> String {
        var kernels = ""
        let (inputs, outputs) = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)

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

        // Add frameCount parameter for all Metal kernels (needed for output operations)
        parameters.append("    constant int &frameCount [[buffer(\(bufferIndex))]]")
        bufferIndex += 1

        // Detect whether this kernel needs segmented dispatch (for delay/barrier semantics)
        let needsSegmenting: Bool = scheduleItem.ops.contains { uop in
            if case .delay1 = uop.op { return true }
            return false
        }

        // If segmented, add segmentLen and segmentBase buffers
        if needsSegmenting {
            parameters.append("    constant int &segmentLen [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
            parameters.append("    constant int &segmentBase [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        // Thread indices
        parameters.append("    uint id [[thread_position_in_grid]]")
        if needsSegmenting {
            parameters.append("    uint tid [[thread_index_in_threadgroup]]")
        }

        kernels += "kernel void \(name)(\n"
        kernels += parameters.joined(separator: ",\n")
        kernels += "\n) {\n"

        // Mark kernel context for emit()
        self.loadVarToCell.removeAll()
        self.vectorStoreOrder.removeAll()

        var indent = 1

        // If segmented, declare threadgroup scratch buffers for delay/store helpers
        if needsSegmenting {
            kernels += "  threadgroup float __dgen_delay_tmp[128];\n"
            kernels += "  threadgroup float __dgen_store_tmp[128];\n"
        }

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
        // Reset kernel context
        return kernels
    }

    func analyzeDependencies(scheduleItem: ScheduleItem, ctx: IRContext) -> (
        inputs: [VarID], outputs: [VarID]
    ) {
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
            case .delay1:
                // delay1 reads (and now also persists) memory state
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
        case let .mod(a, b): return emitAssign(uop, "\(g(a)) % \(g(b))", ctx)
        case let .pow(a, b): return emitAssign(uop, "metal::pow(\(g(a)), \(g(b)))", ctx)
        case let .min(a, b): return emitAssign(uop, "metal::min(\(g(a)), \(g(b)))", ctx)
        case let .max(a, b): return emitAssign(uop, "metal::max(\(g(a)), \(g(b)))", ctx)

        case let .abs(a): return emitAssign(uop, "metal::abs(\(g(a)))", ctx)
        case let .sign(a): return emitAssign(uop, "metal::sign(\(g(a)))", ctx)
        case let .floor(a): return emitAssign(uop, "metal::floor(\(g(a)))", ctx)
        case let .ceil(a): return emitAssign(uop, "metal::ceil(\(g(a)))", ctx)
        case let .round(a): return emitAssign(uop, "metal::round\(g(a)))", ctx)
        case let .memoryRead(base, offset):
            return emitAssign(uop, "memory[\(base) + (int)\(g(offset))]", ctx)
        case let .memoryWrite(base, offset, value):
            return "memory[\(base) + (int)\(g(offset))] = \(g(value));"

        case let .sin(a): return emitAssign(uop, "metal::sin(\(g(a)))", ctx)
        case let .cos(a): return emitAssign(uop, "metal::cos(\(g(a)))", ctx)
        case let .tan(a): return emitAssign(uop, "metal::tan(\(g(a)))", ctx)
        case let .tanh(a): return emitAssign(uop, "metal::tanh(\(g(a)))", ctx)
        case let .round(a): return emitAssign(uop, "metal::round\(g(a)))", ctx)
        case let .exp(a): return emitAssign(uop, "metal::exp(\(g(a)))", ctx)
        case let .log(a): return emitAssign(uop, "metal::log(\(g(a)))", ctx)
        case let .log10(a): return emitAssign(uop, "metal::log10(\(g(a)))", ctx)
        case let .sqrt(a): return emitAssign(uop, "sqrt(\(g(a)))", ctx)
        case let .atan2(y, x): return emitAssign(uop, "atan2(\(g(y)), \(g(x)))", ctx)

        case let .gt(a, b): return emitAssign(uop, "\(g(a)) > \(g(b))", ctx)
        case let .gte(a, b): return emitAssign(uop, "\(g(a)) >= \(g(b))", ctx)
        case let .lte(a, b): return emitAssign(uop, "\(g(a)) <= \(g(b))", ctx)
        case let .lt(a, b): return emitAssign(uop, "\(g(a)) < \(g(b))", ctx)
        case let .eq(a, b): return emitAssign(uop, "\(g(a)) == \(g(b))", ctx)
        case let .gswitch(cond, a, b):
            let expr = "metal::select(\(g(b)), \(g(a)), \(g(cond)) > 0.0)"
            return emitAssign(uop, expr, ctx)

        case let .delay1(cell, a):
            // Metal thread-per-sample delay-by-1 using threadgroup neighbor exchange.
            // Also persists the last 4 current values to memory[cell..cell+3] at the end of the segment.
            // Relies on segmented dispatch so all threads in a group hit barriers.
            let writeTmp = "__dgen_delay_tmp[tid] = \(g(a));"
            let barrier1 = "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            let expr = "(tid > 0u ? __dgen_delay_tmp[tid - 1] : memory[\(cell) + 3])"
            let assign = emitAssign(uop, expr, ctx)
            // Persist state for next segment
            let writeStore = "__dgen_store_tmp[tid] = \(g(a));"
            let barrier2 = "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            let persist = """
                if ((int)tid == segmentLen - 1) {
                    memory[\(cell) + 0] = __dgen_store_tmp[segmentLen - 4];
                    memory[\(cell) + 1] = __dgen_store_tmp[segmentLen - 3];
                    memory[\(cell) + 2] = __dgen_store_tmp[segmentLen - 2];
                    memory[\(cell) + 3] = __dgen_store_tmp[segmentLen - 1];
                }
                """.trimmingCharacters(in: .whitespacesAndNewlines)
            return "\(writeTmp) \(barrier1) \(assign) \(writeStore) \(barrier2) \(persist)"
        /**
         delay1 implemented via threadgroup neighbor exchange
         */
        case let .selector(mode, options):
            // Metal: if mode <= 0 return 0, if mode <= 1 return options[0], etc.
            var expr = "0.0f"  // Default value

            // Build from the end backwards to match the priority order
            for (i, option) in options.enumerated().reversed() {
                expr = "metal::select(\(expr), \(g(option)), \(g(mode)) <= \(Float(i + 1)))"
            }

            // Final check for mode <= 0
            expr = "metal::select(\(expr), 0.0f, \(g(mode)) <= 0.0f)"

            return emitAssign(uop, expr, ctx)

        case let .load(cell): return emitAssign(uop, "memory[\(cell)]", ctx)
        case let .store(cell, val):
            // Regular store for Metal (scalar or per-thread value)
            return "memory[\(cell)] = \(g(val));"

        case let .mutate(a, b):
            return "\(emitLazy(a, ctx: ctx, kind: uop.kind, isOut: true)) = \(g(b));"
        case let .beginIf(cond): return "if (\(g(cond))) {"
        case .endIf: return "}"

        case let .beginLoop(iters, step): return "for (int i = 0; i < \(g(iters)); i += \(step)) {"
        case .endLoop: return "}"

        case .frameCount: return "/* frameCount available as function parameter */"

        case let .beginRange(_, end):
            let endS = g(end)
            return "if (id < uint(\(endS))) {"
        case .endRange: return "}"
        case let .output(channel, val):
            // Store output value to a device buffer that can be read back
            let idx = (uop.kind == .simd) ? "id" : "i"
            return "outputs[\(channel) * frameCount + \(idx)] = \(g(val));"

        case .input(_):
            return ""
        case let .loadGlobal(id):
            // For Metal, loadGlobal is handled transparently through direct buffer access
            // The actual variable access happens in emitLazy
            return "/* loadGlobal(\(id)) - handled in variable access */"

        default:
            return "/* \(uop.prettyDescription()) */"
        }
    }

    func emitLazy(_ lazy: Lazy, ctx: IRContext, kind: Kind?, isOut: Bool) -> String {
        switch lazy {
        case let .constant(_, val): return "\(val)"
        case let .variable(id, _):
            if id == -1 {  // Special case for frameCount
                return "frameCount"
            } else if ctx.globals.contains(id) {
                let idx = (kind == .simd) ? "id" : "i"
                return "t\(id)[\(idx)]"
            } else {
                return "t\(id)"
            }
        case let .global(id):
            // Global variables are accessed through global buffers
            let idx = (kind == .simd) ? "id" : "i"
            return "t\(id)[\(idx)]"
        default: return "/* unknown lazy */"
        }
    }

    func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext) -> String {
        let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
        let isGlobal = ctx.globals.contains(extractVarId(uop.value))

        if isGlobal {
            return "\(lhs) = \(expr);"
        }
        return "float \(lhs) = \(expr);"
    }
}

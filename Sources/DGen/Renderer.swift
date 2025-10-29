// the beauty is this doesn't need to even know if its forward or backward

import Foundation

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
    totalMemorySlots: Int,
    name: String = "kernel"
) throws -> [CompiledKernel] {
    var scheduleItems: [ScheduleItem] = []
    renderer.prepareSchedule(&scheduleItems, uopBlocks, ctx, frameCount)
    // For C backend, ensure we have at least one output op. If not, fail compilation.
    if renderer is CRenderer {
        let hasOutput = scheduleItems.contains { schedule in
            schedule.ops.contains { uop in
                if case .output = uop.op { return true }
                return false
            }
        }
        if !hasOutput {
            throw DGenError.compilationFailed("no output node")
        }
    }
    return renderer.compile(
        scheduleItems: scheduleItems, ctx: ctx, graph: graph, totalMemorySlots: totalMemorySlots,
        name: name)
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
        totalMemorySlots: Int,
        name: String = "kernel"
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

    // MC support parameters supplied by CompilationPipeline.Options
    public var voiceCount: Int = 1
    public var voiceCellIdOpt: Int? = nil

    public override init() {
    }

    public override func compile(
        scheduleItems: [ScheduleItem],
        ctx: IRContext,
        graph: Graph,
        totalMemorySlots: Int,
        name: String = "kernel"
    ) -> [CompiledKernel] {
        return scheduleItems.enumerated().map { (i, scheduleItem) in
            let kernelName = scheduleItems.count > 1 ? "\(name)_\(i)" : name
            let source = render(
                name: kernelName, scheduleItem: scheduleItem, ctx: ctx, graph: graph,
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

            // Ensure memory size covers special cells (e.g., voice cell) as well
            let extra = (self.voiceCellIdOpt ?? -1) + 1
            let computedMem = max(totalMemorySlots, max(1024, extra))
            return CompiledKernel(
                name: kernelName,
                source: source,
                kind: scheduleItem.kind,
                buffers: buffers,
                threadGroupSize: 1,  // C execution is scalar for now
                memorySize: computedMem  // Ensure at least enough for voiceCellId
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

        // Merge adjacent blocks of the same kind into a single loop to reduce passes
        var currentKind: Kind? = nil
        for block in uopBlocks {
            if currentKind != block.kind {
                // Close previous loop if open
                if currentKind != nil {
                    scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
                }
                // Open new loop for this kind
                scheduleItem.ops.append(
                    UOp(op: .beginLoop(frameCountUOp, block.kind == .scalar ? 1 : 4), value: .empty)
                )
                currentKind = block.kind
            }

            for uop in block.ops {
                if case .defineGlobal = uop.op { continue }
                scheduleItem.ops.append(UOp(op: uop.op, value: uop.value, kind: block.kind))
            }
        }

        // Close any open loop
        if currentKind != nil {
            scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
        }
    }

    public override func render(
        name: String, scheduleItem: ScheduleItem, ctx: IRContext, graph: Graph,
        totalMemorySlots: Int
    ) -> String {
        var code: [String] = []

        // Generate a unique UUID for this kernel
        let kernelUUID = UUID().uuidString

        // Sanitize kernel name to be a valid C identifier
        // Replace invalid characters with underscores
        let sanitizedName = name.replacingOccurrences(
            of: "[^a-zA-Z0-9_]", with: "_", options: .regularExpression)

        // C includes and function signature
        code.append(
            """
            #include <arm_neon.h>
            #include <stdint.h>
            #include <stdio.h>
            #include <math.h>
            #include <Accelerate/Accelerate.h>
            #include <mach/mach_time.h>

            // Enable profiling only when DGEN_PROFILE is defined by build flags

            float32x4_t vfmodq_f32(float32x4_t a, float32x4_t b) {
              // a - floor(a / b) * b  (faster and correct for positive ranges)
              float32x4_t q = vdivq_f32(a, b);
              float32x4_t q_floor = vrndmq_f32(q);  // floor
              return vsubq_f32(a, vmulq_f32(b, q_floor));
            }

            // Timing instrumentation for \(name) [\(kernelUUID)]
            #ifdef DGEN_PROFILE
            static uint64_t kernel_invocation_count_\(sanitizedName) = 0;
            static uint64_t kernel_max_time_nanos_\(sanitizedName) = 0;
            static mach_timebase_info_data_t timebase_info_\(sanitizedName);
            static int timebase_initialized_\(sanitizedName) = 0;
            #endif

            """)

        // Declare globals
        let sortedGlobals = ctx.globals.sorted()
        code.append("const int VOICE_COUNT = \(voiceCount);")
        code.append("const int SCRATCH_STRIDE = 512;")
        for varId in sortedGlobals {
            code.append(
                "float t\(varId)_g[VOICE_COUNT * SCRATCH_STRIDE] __attribute__((aligned(64))) = {0};"
            )
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
            "void process(float * restrict const *in, float * restrict const *out, int nframes, void * restrict state) {"
        )

        // Add timing instrumentation at start of function
        code.append(
            """
              #ifdef DGEN_PROFILE
              // Initialize timebase info once
              if (!timebase_initialized_\(sanitizedName)) {
                mach_timebase_info(&timebase_info_\(sanitizedName));
                timebase_initialized_\(sanitizedName) = 1;
              }

              // Start timing
              uint64_t start_time = mach_absolute_time();
              #endif

            """)

        // Use audiograph parameters directly - no mapping needed
        code.append("  int frameCount = nframes;  // Use audiograph frame count parameter")

        // Define constants and memory
        for (constantId, constant) in ctx.constants {
            let uop = UOp(op: .defineConstant(constantId, constant), value: .empty)
            code.append("  \(emit(uop, ctx: ctx))")
        }

        // Cast state parameter to float pointer for use in function
        code.append("  float *memory = (float*)state;")
        // Determine voice index and compute base offset for scratch arrays
        code.append("  int voiceIndex = 0;")
        if let voiceCellId = voiceCellIdOpt {
            code.append("  voiceIndex = (int)memory[\(voiceCellId)];")
        }
        code.append("  if (voiceIndex < 0) voiceIndex = 0;")
        code.append("  if (voiceIndex >= VOICE_COUNT) voiceIndex = VOICE_COUNT - 1;")
        code.append("  int _scratchBase = voiceIndex * SCRATCH_STRIDE;")
        for varId in sortedGlobals {
            code.append("  float *t\(varId) = t\(varId)_g + _scratchBase;")
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

        // Add timing instrumentation before closing function
        code.append(
            """
              #ifdef DGEN_PROFILE
              // End timing and update stats
              uint64_t end_time = mach_absolute_time();
              uint64_t elapsed_abs = end_time - start_time;
              uint64_t elapsed_nanos = elapsed_abs * timebase_info_\(sanitizedName).numer / timebase_info_\(sanitizedName).denom;

              if (elapsed_nanos > kernel_max_time_nanos_\(sanitizedName)) {
                kernel_max_time_nanos_\(sanitizedName) = elapsed_nanos;
              }

              kernel_invocation_count_\(sanitizedName)++;

              // Print every 2048 invocations
              if (kernel_invocation_count_\(sanitizedName) % 2048 == 0) {
                double max_time_ms = kernel_max_time_nanos_\(sanitizedName) / 1000000.0;
                printf("⏱️  \(name) [\(kernelUUID)] invocation %llu - Max time: %.3f ms\\n",
                       kernel_invocation_count_\(sanitizedName), max_time_ms);
                kernel_max_time_nanos_\(sanitizedName) = 0;  // Reset for next interval
              }
              #endif
            """)

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
            return "/* t\(varId) declared globally */"
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
            // Strength-reduce division by constant to multiply by reciprocal
            switch b {
            case let .constant(_, val):
                if uop.kind == .simd {
                    let expr = "vmulq_f32(\(g(a)), vdupq_n_f32(\(1.0/val)f))"
                    return emitAssign(uop, expr, ctx)
                } else {
                    let expr = "(\(g(a)) * \(1.0/val)f)"
                    return emitAssign(uop, expr, ctx)
                }
            default:
                let expr = uop.kind == .simd ? "vdivq_f32(\(g(a)), \(g(b)))" : "\(g(a)) / \(g(b))"
                return emitAssign(uop, expr, ctx)
            }

        case let .mod(a, b):
            // Fast modulo for constant denominator: a - floor(a / b) * b
            switch b {
            case let .constant(_, val):
                if uop.kind == .simd {
                    if val == 1.0 {
                        let expr = "vsubq_f32(\(g(a)), vrndmq_f32(\(g(a))))"
                        return emitAssign(uop, expr, ctx)
                    } else {
                        let expr =
                            "vsubq_f32(\(g(a)), vmulq_f32(vdupq_n_f32(\(val)f), vrndmq_f32(vmulq_f32(\(g(a)), vdupq_n_f32(\(1.0/val)f)))))"
                        return emitAssign(uop, expr, ctx)
                    }
                } else {
                    if val == 1.0 {
                        let expr = "(\(g(a)) - floorf(\(g(a))))"
                        return emitAssign(uop, expr, ctx)
                    } else {
                        let expr = "(\(g(a)) - floorf(\(g(a)) / \(val)f) * \(val)f)"
                        return emitAssign(uop, expr, ctx)
                    }
                }
            default:
                let expr =
                    uop.kind == .simd ? "vfmodq_f32(\(g(a)), \(g(b)))" : "fmodf(\(g(a)), \(g(b)))"
                return emitAssign(uop, expr, ctx)
            }

        case let .pow(a, b):
            // Specialize common exponents to avoid expensive powf
            switch b {
            case let .constant(_, val):
                if val == 1.0 {
                    return emitAssign(uop, uop.kind == .simd ? "\(g(a))" : "\(g(a))", ctx)
                } else if val == 2.0 {
                    let expr =
                        uop.kind == .simd ? "vmulq_f32(\(g(a)), \(g(a)))" : "(\(g(a)) * \(g(a)))"
                    return emitAssign(uop, expr, ctx)
                } else if val == 3.0 {
                    if uop.kind == .simd {
                        let expr = "vmulq_f32(vmulq_f32(\(g(a)), \(g(a))), \(g(a)))"
                        return emitAssign(uop, expr, ctx)
                    } else {
                        let expr = "(\(g(a)) * \(g(a)) * \(g(a)))"
                        return emitAssign(uop, expr, ctx)
                    }
                } else if val == 4.0 {
                    if uop.kind == .simd {
                        let t2 = "vmulq_f32(\(g(a)), \(g(a)))"
                        let expr = "vmulq_f32(\(t2), \(t2))"
                        return emitAssign(uop, expr, ctx)
                    } else {
                        let expr = "({ float _t=\(g(a))*\(g(a)); _t*_t; })"
                        return emitAssign(uop, expr, ctx)
                    }
                } else if val == 0.5 {
                    let expr = uop.kind == .simd ? "vsqrtf(\(g(a)))" : "sqrtf(\(g(a)))"
                    return emitAssign(uop, expr, ctx)
                } else if val == 0.0 {
                    let expr = uop.kind == .simd ? "vdupq_n_f32(1.0f)" : "1.0f"
                    return emitAssign(uop, expr, ctx)
                }
                // Fallback
                let expr = uop.kind == .simd ? "vpowf(\(g(a)), \(g(b)))" : "powf(\(g(a)), \(g(b)))"
                return emitAssign(uop, expr, ctx)
            default:
                // If base is constant, emit exp(b * log(base)) which is faster than generic pow
                if case let .constant(_, baseVal) = a {
                    if uop.kind == .simd {
                        let expr = "vexpf(vmulq_f32(\(g(b)), vdupq_n_f32(logf(\(baseVal)f))))"
                        return emitAssign(uop, expr, ctx)
                    } else {
                        let expr = "expf((\(g(b))) * logf(\(baseVal)f))"
                        return emitAssign(uop, expr, ctx)
                    }
                }
                let expr = uop.kind == .simd ? "vpowf(\(g(a)), \(g(b)))" : "powf(\(g(a)), \(g(b)))"
                return emitAssign(uop, expr, ctx)
            }

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
            // Constant-fold equality when both operands are constants
            switch (a, b) {
            case let (.constant(_, av), .constant(_, bv)):
                if uop.kind == .simd {
                    let expr = (av == bv) ? "vdupq_n_f32(1.0f)" : "vdupq_n_f32(0.0f)"
                    return emitAssign(uop, expr, ctx)
                } else {
                    return emitAssign(uop, (av == bv) ? "1.0f" : "0.0f", ctx)
                }
            default:
                let expr =
                    uop.kind == .simd
                    ? "vbslq_f32(vceqq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                    : "\(g(a)) == \(g(b))"
                return emitAssign(uop, expr, ctx)
            }

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
        case let .concatShift(a, b, shift):
            if uop.kind == .simd {
                // this op is only relevant in vectorized block
                let expr = "vextq_f32(\(g(a)), \(g(b)), \(shift))"
                return emitAssign(uop, expr, ctx)
            } else {
                // scalar version is simple identity (i.e. a no-op).
                let expr = "\(g(a))"
                return emitAssign(uop, expr, ctx)
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
        case let .loadTape(offset):
            if uop.kind == .simd {
                // SIMD: load 4 consecutive frames from tape starting at [offset + i]
                return emitAssign(uop, "vld1q_f32(&tape[\(offset) + i])", ctx)
            } else {
                return emitAssign(uop, "tape[\(offset) + i]", ctx)
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
    var needsSegmenting = false

    public override init() {
    }

    public override func compile(
        scheduleItems: [ScheduleItem],
        ctx: IRContext,
        graph: Graph,
        totalMemorySlots: Int,
        name: String = "kernel"
    ) -> [CompiledKernel] {
        return scheduleItems.enumerated().map { (i, scheduleItem) in
            let kernelName = scheduleItems.count > 1 ? "\(name)_\(i)" : name
            let source = render(
                name: kernelName, scheduleItem: scheduleItem, ctx: ctx, graph: graph,
                totalMemorySlots: totalMemorySlots)
            let deps = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)
            let allBuffers = Set(deps.inputs + deps.outputs)

            let bufferRequirements = analyzeRequiredBuffers(scheduleItem: scheduleItem)

            var bufferNames: [String] = []
            if bufferRequirements.hasOutputOps {
                bufferNames.append("outputs")
            }

            var hasMemory = false
            var hasCrossKernelBuffers = false
            // Use same sorted order as render() method
            for bufferId in allBuffers.sorted() {
                if bufferId == memoryVarID {
                    hasMemory = true
                } else {
                    hasCrossKernelBuffers = true
                }
            }

            // ordering must match
            if hasMemory {
                bufferNames.append("memory")
            }

            if hasCrossKernelBuffers {
                bufferNames.append("t")
            }

            print("ORDERED BUFFER NAMES=\(bufferNames)")

            // Add frameCount buffer for all Metal kernels (needed for output operations)
            bufferNames.append("frameCount")

            // Add segment buffers if this kernel needs segmented execution
            if bufferRequirements.needsSegmenting {
                bufferNames.append("segmentLen")
                bufferNames.append("segmentBase")
            }

            if bufferRequirements.needsGradMemory {
                bufferNames.append("grad_memory")
            }

            if bufferRequirements.needsGrad {
                bufferNames.append("gradients")
            }

            return CompiledKernel(
                name: kernelName,
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
        var (inputs, outputs) = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)

        let hasOutputOps = scheduleItem.ops.contains { uop in
            if case .output = uop.op { return true }
            return false
        }

        let allBuffers = Set(inputs + outputs)

        var parameters: [String] = []
        var bufferIndex = 0

        let bufferRequirements = analyzeRequiredBuffers(scheduleItem: scheduleItem)
        // Add outputs buffer first if needed
        if bufferRequirements.hasOutputOps {
            parameters.append("    device float *outputs [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        var hasMemory = false
        var hasCrossKernelBuffers = false
        // Add other buffers
        for bufferId in allBuffers.sorted() {
            if bufferId == memoryVarID {
                hasMemory = true
            } else {
                hasCrossKernelBuffers = true
            }
        }

        if hasMemory {
            parameters.append("    device float *memory [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        if hasCrossKernelBuffers {
            parameters.append("    device float *t [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        if bufferRequirements.needsGrad {
            parameters.append("    device float *gradients [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        if bufferRequirements.needsGradMemory {
            parameters.append("    device float *grad_memory [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        // Add frameCount parameter for all Metal kernels (needed for output operations)
        parameters.append("    constant int &frameCount [[buffer(\(bufferIndex))]]")
        bufferIndex += 1

        // If segmented, add segmentLen and segmentBase buffers
        if bufferRequirements.needsSegmenting {
            parameters.append("    constant int &segmentLen [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
            parameters.append("    constant int &segmentBase [[buffer(\(bufferIndex))]]")
            bufferIndex += 1
        }

        parameters.append("    uint id [[thread_position_in_grid]]")
        if bufferRequirements.needsSegmenting {
            parameters.append("    uint tid [[thread_index_in_threadgroup]]")
        }

        kernels += "kernel void \(name)(\n"
        kernels += parameters.joined(separator: ",\n")
        kernels += "\n) {\n"

        var indent = 1

        // If segmented, declare threadgroup scratch buffers for delay/store helpers
        if bufferRequirements.needsSegmenting {
            kernels += "  threadgroup float __dgen_delay_tmp[128];\n"
            kernels += "  threadgroup float __dgen_store_tmp[128];\n"
        }

        self.needsSegmenting = bufferRequirements.needsSegmenting

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

            kernels +=
                "\(String(repeating: "  ", count: indent))\(emit(uop, ctx: ctx))\n"
            indent += diff
        }

        kernels += "}\n\n"
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
            case .load, .store, .delay1:
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
        case let .div(a, b):
            // Strength-reduce division by constant to multiply by reciprocal
            switch b {
            case let .constant(_, val):
                return emitAssign(uop, "(\(g(a)) * \(1.0/val))", ctx)
            default:
                return emitAssign(uop, "\(g(a)) / \(g(b))", ctx)
            }
        case let .mod(a, b):
            // Fast modulo for constant denominator: a - floor(a / b) * b
            switch b {
            case let .constant(_, val):
                if val == 1.0 {
                    return emitAssign(uop, "(\(g(a)) - floor(\(g(a))))", ctx)
                } else {
                    return emitAssign(uop, "(\(g(a)) - floor(\(g(a)) / \(val)) * \(val))", ctx)
                }
            default:
                return emitAssign(uop, "fmod(\(g(a)), \(g(b)))", ctx)
            }
        case let .pow(a, b):
            // Specialize common exponents to avoid expensive pow
            switch b {
            case let .constant(_, val):
                if val == 1.0 { return emitAssign(uop, "\(g(a))", ctx) }
                if val == 2.0 { return emitAssign(uop, "(\(g(a)) * \(g(a)))", ctx) }
                if val == 3.0 { return emitAssign(uop, "(\(g(a)) * \(g(a)) * \(g(a)))", ctx) }
                if val == 4.0 {
                    return emitAssign(uop, "({ float _t=\(g(a))*\(g(a)); _t*_t; })", ctx)
                }
                if val == 0.5 { return emitAssign(uop, "metal::sqrt(\(g(a)))", ctx) }
                if val == 0.0 { return emitAssign(uop, "1.0", ctx) }
                return emitAssign(uop, "metal::pow(\(g(a)), \(g(b)))", ctx)
            default:
                // If base is constant: exp(b * log(base))
                if case let .constant(_, baseVal) = a {
                    return emitAssign(uop, "metal::exp(\(g(b)) * metal::log(\(baseVal)))", ctx)
                }
                return emitAssign(uop, "metal::pow(\(g(a)), \(g(b)))", ctx)
            }
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
        case let .exp(a): return emitAssign(uop, "metal::exp(\(g(a)))", ctx)
        case let .log(a): return emitAssign(uop, "metal::log(\(g(a)))", ctx)
        case let .log10(a): return emitAssign(uop, "metal::log10(\(g(a)))", ctx)
        case let .sqrt(a): return emitAssign(uop, "metal::sqrt(\(g(a)))", ctx)
        case let .atan2(y, x): return emitAssign(uop, "metal::atan2(\(g(y)), \(g(x)))", ctx)

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
        case let .loadGradMemory(cellId): return emitAssign(uop, "grad_memory[\(cellId)]", ctx)
        case let .frameIndex: return emitAssign(uop, "i", ctx)
        case let .loadTape(offset):
            let idx = (uop.kind == .simd) ? "id" : "i"
            return emitAssign(uop, "tape[\(offset) + \(idx)]", ctx)
        case let .store(cell, val): return "memory[\(cell)] = \(g(val));"
        case let .storeGradMemory(cell, val): return "grad_memory[\(cell)] = \(g(val));"
        case let .loadGrad(gradId):
            return emitAssign(
                uop, "gradients[frameCount * \(gradId) + \(uop.kind == .scalar ? "i" : "id")]", ctx)
        case let .accumulateGrad(gradId, val):
            return
                "gradients[frameCount*\(gradId)+\(uop.kind == .scalar ? "i" : "id")] += \(g(val));"
        case let .mutate(a, b):
            return "\(emitLazy(a, ctx: ctx, kind: uop.kind, isOut: true)) = \(g(b));"
        case let .beginIf(cond): return "if (\(g(cond))) {"
        case .endIf: return "}"

        case let .beginLoop(iters, step): return "for (int i = 0; i < \(g(iters)); i += \(step)) {"
        case .endLoop: return "}"

        case let .beginRange(start, end): return "if (id >= \(g(start)) && id < \(g(end))) {"
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
                return
                    "t[\(ctx.getGlobalId(id))*frameCount + \(needsSegmenting ? "segmentBase + " : "") \(idx)]"
            } else {
                return "t\(id)"
            }
        case let .global(id):
            // Global variables are accessed through global buffers
            let idx = (kind == .simd) ? "id" : "i"
            return
                "t[\(ctx.getGlobalId(id))*frameCount + \(needsSegmenting ? "segmentBase + " : "")\(idx)]"
        default: return "/* unknown lazy */"
        }
    }

    func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext) -> String {
        // TODO (backpropagation) - if this value is needed in the tape of the backprop and we're in forward pass we must store it

        let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
        let isGlobal = ctx.globals.contains(extractVarId(uop.value))

        if isGlobal {
            return "\(lhs) = \(expr);"
        }
        return "float \(lhs) = \(expr);"
    }
}

struct RequiredBuffers {
    let hasOutputOps: Bool
    let needsSegmenting: Bool
    let needsGradMemory: Bool
    let needsGrad: Bool
}

func analyzeRequiredBuffers(scheduleItem: ScheduleItem) -> RequiredBuffers {
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

    let needsGradMemory: Bool = scheduleItem.ops.contains { uop in
        if case .loadGradMemory = uop.op {
            return true
        } else if case .storeGradMemory = uop.op {
            return true
        }
        return false
    }

    let needsGrad: Bool = scheduleItem.ops.contains { uop in
        if case .loadGrad = uop.op {
            return true
        } else if case .accumulateGrad = uop.op {
            return true
        }
        return false
    }

    return RequiredBuffers(
        hasOutputOps: hasOutputOps, needsSegmenting: needsSegmenting,
        needsGradMemory: needsGradMemory, needsGrad: needsGrad
    )
}

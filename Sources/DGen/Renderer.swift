// the beauty is this doesn't need to even know if its forward or backward

import Foundation

public struct BlockUOps {
    public var ops: [UOp]
    public let kind: Kind
    public let temporality: Temporality

    public init(ops: [UOp], kind: Kind, temporality: Temporality = .static_) {
        self.ops = ops
        self.kind = kind
        self.temporality = temporality
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
    public let threadCount: Int?  // for Metal: override total threads (non-frame dispatch)
    public let needsReducedGradsSum: Bool
    public let memorySize: Int  // Required memory allocation size in floats
}

public class ScheduleItem {
    public var ops: [UOp] = []
    public let kind: Kind
    public var temporality: Temporality = .frameBased

    init(kind: Kind, temporality: Temporality = .frameBased) {
        self.kind = kind
        self.temporality = temporality
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
    case .variable(let varid, _):
        return varid
    case .global(let varid):
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
    public var loadedGlobal: [Int: Bool] = [:]

    // Track the emitted type of each variable in current scope
    public enum EmittedType {
        case int_
        case float_
        case float32x4
    }
    public var varEmittedTypes: [VarID: EmittedType] = [:]

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
                case .defineGlobal(let varId):
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
                threadCount: nil,
                needsReducedGradsSum: false,
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
        var currentTemporality: Temporality? = nil
        var hopCheckOpen = false  // Track if we have an open hop check conditional

        for block in uopBlocks {
            let needsNewLoop = currentKind != block.kind || currentTemporality != block.temporality

            if needsNewLoop {
                // Close previous hop check if open
                if hopCheckOpen {
                    scheduleItem.ops.append(UOp(op: .endHopCheck, value: .empty))
                    hopCheckOpen = false
                }

                // Close previous loop if open
                if currentKind != nil {
                    scheduleItem.ops.append(UOp(op: .endLoop, value: .empty))
                }

                // Open new loop based on temporality
                switch block.temporality {
                case .frameBased, .static_:
                    // Frame-based or static: standard frame loop
                    scheduleItem.ops.append(
                        UOp(
                            op: .beginLoop(frameCountUOp, block.kind == .scalar ? 1 : 4),
                            value: .empty)
                    )

                case .hopBased(_, let counterCell):
                    // Hop-based: frame loop with conditional check inside
                    // The block only executes when counter == 0
                    scheduleItem.ops.append(
                        UOp(
                            op: .beginLoop(frameCountUOp, block.kind == .scalar ? 1 : 4),
                            value: .empty)
                    )
                    scheduleItem.ops.append(
                        UOp(op: .beginHopCheck(counterCell), value: .empty)
                    )
                    hopCheckOpen = true
                }

                currentKind = block.kind
                currentTemporality = block.temporality
            }

            for uop in block.ops {
                // Don't skip defineGlobal - it needs to run through emit to mark loadedGlobal
                scheduleItem.ops.append(UOp(op: uop.op, value: uop.value, kind: uop.kind))
            }
        }

        // Close any open hop check
        if hopCheckOpen {
            scheduleItem.ops.append(UOp(op: .endHopCheck, value: .empty))
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

            static inline uint32x4_t mask_nz_f32(float32x4_t x) {
                float32x4_t zero = vdupq_n_f32(0.0f);
                // eq0 = (x == 0.0f)
                uint32x4_t eq0  = vceqq_f32(x, zero);
                // non-zero mask = bitwise NOT of eq0
                return vmvnq_u32(eq0);
            }

            static inline float32x4_t boolmask_to_float(uint32x4_t m) {
                float32x4_t ones  = vdupq_n_f32(1.0f);
                float32x4_t zeros = vdupq_n_f32(0.0f);
                // Select 1.0f where mask bits are 1, else 0.0f
                return vbslq_f32(m, ones, zeros);
            }

            static inline float32x4_t simd_and_f32(float32x4_t a, float32x4_t b) {
                uint32x4_t a_nz = mask_nz_f32(a);
                uint32x4_t b_nz = mask_nz_f32(b);
                uint32x4_t m    = vandq_u32(a_nz, b_nz);
                return boolmask_to_float(m);
            }

            static inline float32x4_t simd_or_f32(float32x4_t a, float32x4_t b) {
                uint32x4_t a_nz = mask_nz_f32(a);
                uint32x4_t b_nz = mask_nz_f32(b);
                uint32x4_t m    = vorrq_u32(a_nz, b_nz);
                return boolmask_to_float(m);
            }

            static inline float32x4_t simd_xor_f32(float32x4_t a, float32x4_t b) {
                uint32x4_t a_nz = mask_nz_f32(a);
                uint32x4_t b_nz = mask_nz_f32(b);
                uint32x4_t m    = veorq_u32(a_nz, b_nz);
                return boolmask_to_float(m);
            }

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
            "void process(float * restrict const *in, float * restrict const *out, int nframes, void * restrict state, void * restrict buffers) {"
        )

        // Use audiograph parameters directly - no mapping needed
        code.append("  int frameCount = nframes;  // Use audiograph frame count parameter")

        loadedGlobal = [:]
        varEmittedTypes = [:]

        // Define constants and memory (sorted by constantId for deterministic output)
        for (constantId, constant) in ctx.constants.sorted(by: { $0.key < $1.key }) {
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
            case .beginIf, .beginForLoop, .beginHopCheck:
                diff = 1
            case .beginLoop:
                diff = 1
                // Reset for each new loop scope - variables need to be redeclared
                loadedGlobal = [:]
                varEmittedTypes = [:]
            case .beginParallelRange:
                diff = 1
            case .endIf, .endLoop, .endParallelRange, .endHopCheck:
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
        case .defineConstant(let constantId, let val):
            return "float32x4_t c\(constantId) = vdupq_n_f32(\(val)f);"
        case .defineGlobal(let varId):
            // Mark as loaded so subsequent loadGlobal is skipped (variable will be defined by actual computation)
            loadedGlobal[varId] = true
            return "/* t\(varId) declared globally */"
        case .defineMemory(let length):
            return "float memory[\(length)] __attribute__((aligned(16)));"

        case .add(let a, let b):
            let expr = uop.kind == .simd ? "vaddq_f32(\(g(a)), \(g(b)))" : "\(g(a)) + \(g(b))"
            return emitAssign(uop, expr, ctx)

        case .mul(let a, let b):
            let expr = uop.kind == .simd ? "vmulq_f32(\(g(a)), \(g(b)))" : "\(g(a)) * \(g(b))"
            return emitAssign(uop, expr, ctx)

        case .sub(let a, let b):
            let expr = uop.kind == .simd ? "vsubq_f32(\(g(a)), \(g(b)))" : "\(g(a)) - \(g(b))"
            return emitAssign(uop, expr, ctx)

        case .div(let a, let b):
            // Strength-reduce division by constant to multiply by reciprocal
            switch b {
            case .constant(_, let val):
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

        case .mod(let a, let b):
            // Fast modulo for constant denominator: a - floor(a / b) * b
            switch b {
            case .constant(_, let val):
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

        case .pow(let a, let b):
            // Specialize common exponents to avoid expensive powf
            switch b {
            case .constant(_, let val):
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
                if case .constant(_, let baseVal) = a {
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

        case .min(let a, let b):
            let expr = uop.kind == .simd ? "vminq_f32(\(g(a)), \(g(b)))" : "fminf(\(g(a)), \(g(b)))"
            return emitAssign(uop, expr, ctx)

        case .max(let a, let b):
            let expr = uop.kind == .simd ? "vmaxq_f32(\(g(a)), \(g(b)))" : "fmaxf(\(g(a)), \(g(b)))"
            return emitAssign(uop, expr, ctx)

        case .abs(let a):
            let expr = uop.kind == .simd ? "vabsq_f32(\(g(a)))" : "fabs(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .sign(let a):
            if uop.kind == .simd {
                // For SIMD: return -1.0 for negative, 1.0 for positive, 0.0 for zero
                let expr =
                    "vbslq_f32(vcltq_f32(\(g(a)), vdupq_n_f32(0.0f)), vdupq_n_f32(-1.0f), vbslq_f32(vcgtq_f32(\(g(a)), vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)))"
                return emitAssign(uop, expr, ctx)
            } else {
                // For scalar: use copysignf for efficient sign extraction
                return emitAssign(uop, "(\(g(a)) == 0.0f) ? 0.0f : copysignf(1.0f, \(g(a)))", ctx)
            }

        case .floor(let a):
            let expr = uop.kind == .simd ? "vrndmq_f32(\(g(a)))" : "floorf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .ceil(let a):
            let expr = uop.kind == .simd ? "vrndpq_f32(\(g(a)))" : "ceilf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .round(let a):
            let expr =
                uop.kind == .simd
                ? "vrndaq_f32(\(g(a)))"
                : "roundf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .noise(let cellId):
            // Xorshift32 PRNG - better spectral properties than LCG
            if uop.kind == .simd {
                // For SIMD, generate 4 random values using 4 sequential xorshift updates
                let expr = """
                    ({
                        uint32_t s = (uint32_t)memory[\(cellId)];
                        if (s == 0u) s = 1u;
                        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                        float r0 = (float)s / 4294967296.0f;
                        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                        float r1 = (float)s / 4294967296.0f;
                        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                        float r2 = (float)s / 4294967296.0f;
                        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                        float r3 = (float)s / 4294967296.0f;
                        memory[\(cellId)] = (float)s;
                        (float32x4_t){r0, r1, r2, r3};
                    })
                    """
                return emitAssign(uop, expr, ctx)
            } else {
                let expr = """
                    ({
                        uint32_t s = (uint32_t)memory[\(cellId)];
                        if (s == 0u) s = 1u;
                        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                        memory[\(cellId)] = (float)s;
                        (float)s / 4294967296.0f;
                    })
                    """
                return emitAssign(uop, expr, ctx)
            }

        case .memoryRead(let base, let offset):
            if uop.kind == .simd {
                let offsetExpr = g(offset)
                // Check offset type to determine how to handle it
                let offsetType: EmittedType
                if case .variable(let varId, _) = offset {
                    offsetType = varEmittedTypes[varId] ?? .float32x4
                } else {
                    offsetType = .float32x4
                }

                switch offsetType {
                case .int_, .float_:
                    // Offset is scalar (int or float) - use direct SIMD load
                    return emitAssign(uop, "vld1q_f32(&memory[\(base) + (int)\(offsetExpr)])", ctx)
                case .float32x4:
                    // Offset is a SIMD vector - gather 4 values from different locations
                    let gatherExpr = """
                        (float32x4_t){
                            memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 0)],
                            memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 1)],
                            memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 2)],
                            memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 3)]
                        }
                        """.trimmingCharacters(in: .whitespacesAndNewlines)
                    return emitAssign(uop, gatherExpr, ctx)
                }
            } else {
                let safeOffset = "(isfinite((int) \(g(offset))) ? (int) \(g(offset)) : 0)"
                return emitAssign(
                    uop,
                    "memory[\(base) + \(safeOffset)]",
                    ctx)
            }

        case .memoryWrite(let base, let offset, let value):
            if uop.kind == .simd {
                let offsetExpr = g(offset)
                let valueExpr = g(value)
                // Check offset type to determine how to handle it
                let offsetType: EmittedType
                if case .variable(let varId, _) = offset {
                    offsetType = varEmittedTypes[varId] ?? .float32x4
                } else {
                    offsetType = .float32x4
                }

                switch offsetType {
                case .int_, .float_:
                    // Offset is scalar (int or float) - use direct SIMD store
                    return "vst1q_f32(&memory[\(base) + (int)\(offsetExpr)], \(valueExpr));"
                case .float32x4:
                    // Offset is a SIMD vector - scatter 4 values to different locations
                    return """
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 0)] = vgetq_lane_f32(\(valueExpr), 0);
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 1)] = vgetq_lane_f32(\(valueExpr), 1);
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 2)] = vgetq_lane_f32(\(valueExpr), 2);
                        memory[\(base) + (int)vgetq_lane_f32(\(offsetExpr), 3)] = vgetq_lane_f32(\(valueExpr), 3);
                        """.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            } else {
                return "memory[\(base) + (int)\(g(offset))] = \(g(value));"
            }
        case .sin(let a):
            let expr = uop.kind == .simd ? "vsinf(\(g(a)))" : "sinf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .cos(let a):
            let expr = uop.kind == .simd ? "vcosf(\(g(a)))" : "cosf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .tan(let a):
            let expr = uop.kind == .simd ? "vtanf(\(g(a)))" : "tanf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .tanh(let a):
            let expr = uop.kind == .simd ? "vtanhf(\(g(a)))" : "tanhf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .exp(let a):
            let expr = uop.kind == .simd ? "vexpf(\(g(a)))" : "expf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .log(let a):
            let expr = uop.kind == .simd ? "vlogf(\(g(a)))" : "logf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .log10(let a):
            let expr =
                uop.kind == .simd
                ? "vmulq_f32(vlogf(\(g(a))), vdupq_n_f32((float)M_LOG10E))"  // log10(x) = ln(x) * log10(e)
                : "log10f(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .sqrt(let a):
            let expr = uop.kind == .simd ? "vsqrtf(\(g(a)))" : "sqrtf(\(g(a)))"
            return emitAssign(uop, expr, ctx)

        case .and(let a, let b):
            let expr =
                uop.kind == .simd
                ? "simd_and_f32(\(g(a)), \(g(b)))"
                : "(((\(g(a)) != 0.0f) && (\(g(b)) != 0.0f)) ? 1.0f : 0.0f)"
            return emitAssign(uop, expr, ctx)

        case .or(let a, let b):
            let expr =
                uop.kind == .simd
                ? "simd_or_f32(\(g(a)), \(g(b)))"
                : "(((\(g(a)) != 0.0f) || (\(g(b)) != 0.0f)) ? 1.0f : 0.0f)"
            return emitAssign(uop, expr, ctx)

        case .xor(let a, let b):
            // XOR: true iff exactly one is non-zero
            let expr =
                uop.kind == .simd
                ? "simd_xor_f32(\(g(a)), \(g(b)))"
                : "((((\(g(a)) != 0.0f) ^ ((\(g(b)) != 0.0f))) ? 1.0f : 0.0f))"
            return emitAssign(uop, expr, ctx)
        case .atan2(let y, let x):
            let expr = uop.kind == .simd ? "vatan2f(\(g(y)), \(g(x)))" : "atan2f(\(g(y)), \(g(x)))"
            return emitAssign(uop, expr, ctx)

        case .gt(let a, let b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcgtq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) > \(g(b))"
            return emitAssign(uop, expr, ctx)
        case .gte(let a, let b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcgeq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) >= \(g(b))"
            return emitAssign(uop, expr, ctx)
        case .lte(let a, let b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcleq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) <= \(g(b))"
            return emitAssign(uop, expr, ctx)
        case .lt(let a, let b):
            let expr =
                uop.kind == .simd
                ? "vbslq_f32(vcltq_f32(\(g(a)), \(g(b))), vdupq_n_f32(1.0f), vdupq_n_f32(0.0f))"
                : "\(g(a)) < \(g(b))"
            return emitAssign(uop, expr, ctx)
        case .eq(let a, let b):
            // Constant-fold equality when both operands are constants
            switch (a, b) {
            case (.constant(_, let av), .constant(_, let bv)):
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

        case .gswitch(let cond, let a, let b):
            if uop.kind == .simd {
                // For SIMD: use vbslq_f32 to select between a and b based on condition > 0
                let mask = "vcgtq_f32(\(g(cond)), vdupq_n_f32(0.0f))"
                let expr = "vbslq_f32(\(mask), \(g(a)), \(g(b)))"
                return emitAssign(uop, expr, ctx)
            } else {
                let expr = "\(g(cond)) > 0.0f ? \(g(a)) : \(g(b))"
                return emitAssign(uop, expr, ctx)
            }
        case .delay1(let cell, let curr):
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
        case .concatShift(let a, let b, let shift):
            if uop.kind == .simd {
                // this op is only relevant in vectorized block
                let expr = "vextq_f32(\(g(a)), \(g(b)), \(shift))"
                return emitAssign(uop, expr, ctx)
            } else {
                // scalar version is simple identity (i.e. a no-op).
                let expr = "\(g(a))"
                return emitAssign(uop, expr, ctx)
            }
        case .selector(let mode, let options):
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

        case .store(let cell, let val):
            if uop.kind == .simd {
                // For SIMD: store all 4 vector elements to consecutive memory slots
                return "vst1q_f32(&memory[\(cell)], \(g(val)));"
            } else {
                return "memory[\(cell)] = \(g(val));"
            }
        case .load(let cell):
            if uop.kind == .simd {
                // For SIMD: load 4 consecutive memory slots into vector
                return emitAssign(uop, "vld1q_f32(&memory[\(cell)])", ctx)
            } else {
                return emitAssign(uop, "memory[\(cell)]", ctx)
            }
        case .loadTape(let val, let offset):
            // loadTape reads from global tape buffer with bounds checking
            // In C, this is handled at compile-time (no runtime bounds check for now)
            let varId = g(val)  // The signal/variable to load from
            if uop.kind == .simd {
                // SIMD: load 4 consecutive frames from tape starting at [offset + i]
                return emitAssign(uop, "vld1q_f32(&tape[\(varId)][\(g(offset))])", ctx)
            } else {
                return emitAssign(uop, "tape[\(varId)][\(g(offset))]", ctx)
            }
        case .beginIf(let cond): return "if (\(g(cond))) {"
        case .endIf: return "}"

        case .mutate(let a, let b):
            return "\(emitLazy(a, ctx: ctx, kind: uop.kind, isOut: true)) = \(g(b));"

        case .input(let channel):
            if uop.kind == .simd {
                // For SIMD: load 4 consecutive memory slots into vector
                let ptr = "in[\(channel)] + i"
                return emitAssign(uop, "vld1q_f32(\(ptr))", ctx)
            } else {
                let addr = "in[\(channel)][i]"
                return emitAssign(uop, "\(addr)", ctx)
            }
        case .output(let channel, let val):
            if uop.kind == .simd {
                // For audiograph compatibility: use out[channel] directly
                let ptr = "out[\(channel)] + i"
                return "vst1q_f32(\(ptr), \(g(val)));"
            } else {
                // For audiograph compatibility: use out[channel][i] directly
                let addr = "out[\(channel)][i]"
                return "\(addr) = \(g(val));"
            }

        case .beginLoop(let iters, let step):
            return "for (int i = 0; i < \(g(iters)); i += \(step)) {"
        case .beginForLoop(let loopVar, let count):
            guard case .variable(let varId, _) = loopVar else {
                fatalError("beginForLoop requires variable")
            }
            // Emit count as integer to avoid "t < 33.0" in loop bounds
            let countStr: String
            if case .constant(_, let val) = count {
                countStr = "\(Int(val))"
            } else {
                countStr = "(int)\(g(count))"
            }
            return "for (int t\(varId) = 0; t\(varId) < \(countStr); t\(varId)++) {"
        case .endLoop: return "}"

        case .threadIndex:
            // In C, threadIndex maps to the loop variable 'i'
            // For SIMD, create a vector of sequential indices: {i, i+1, i+2, i+3}
            if uop.kind == .simd {
                let varId = extractVarId(uop.value)
                varEmittedTypes[varId] = .float32x4
                let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
                return "float32x4_t \(lhs) = (float32x4_t){(float)i, (float)(i+1), (float)(i+2), (float)(i+3)};"
            } else {
                return emitAssign(uop, "i", ctx)
            }

        case .cast(let expr, let castType):
            let typeStr = castType == .int ? "int" : "float"
            return emitAssign(
                uop, "(\(typeStr))\(g(expr))", ctx,
                forceFloatType: true)
        case .declareVar(let value):
            // Declares and initializes a variable: float t = value;
            return emitAssign(uop, g(value), ctx)

        case .frameCount: return "/* frameCount available as function parameter */"

        case .loadGlobal(let id):
            if loadedGlobal[id] != nil {
                return "/* skip load */"
            }
            loadedGlobal[id] = true
            if uop.kind == .simd {
                // Create a proper SIMD variable declaration for loadGlobal
                return "float32x4_t simd\(id) = vld1q_f32(t\(id) + i);"
            } else {
                // if this global is required by a beginParallelRange element then we need
                // a simd version where we
                let simdVersion =
                    "float32x4_t simd\(id) = vdupq_n_f32(t\(id)[i]); /* extra */"
                let scalarVersion = emitAssign(uop, "t\(id)[i]", ctx)
                return simdVersion + "\n    " + scalarVersion
            }

        // Parallel range - for C, render as a simple for loop
        // For static tensor ops, this could be outside frame loop (future optimization)
        case .beginParallelRange(let count, let incr):
            let pre = incr == 4 ? "simd" : "t"
            guard case .variable(let varId, _) = uop.value else {
                fatalError("beginParallelRange requires variable")
            }
            // Track this as an int loop counter
            varEmittedTypes[varId] = .int_
            return
                "for (int \(pre)\(varId) = 0; \(pre)\(varId) < \(count); \(pre)\(varId)+=\(incr)) {"
        case .endParallelRange:
            return "}"
        case .parallelIndex:
            // Return the loop variable from the enclosing parallelRange
            // We use the value's varId to match the beginParallelRange
            guard case .variable(let varId, _) = uop.value else {
                fatalError("parallelIndex requires variable")
            }
            return emitAssign(uop, "_pr\(varId)", ctx, forceFloatType: true)

        case .loadGrad(let gradId):
            return emitAssign(uop, "gradients[\(gradId) * frameCount + i]", ctx)

        case .accumulateGrad(let gradId, let val):
            return "gradients[\(gradId) * frameCount + i] += \(g(val));"

        case .loadTensorGrad(let baseGradId, let indexLazy):
            return emitAssign(
                uop, "gradients[(\(baseGradId)+(int)(\(g(indexLazy)))) * frameCount + i]", ctx)

        case .accumulateTensorGrad(let baseGradId, let indexLazy, let valueLazy):
            return
                "gradients[(\(baseGradId)+(int)(\(g(indexLazy)))) * frameCount + i] += \(g(valueLazy));"

        // Hop-based execution: only run block when counter == 0
        case .beginHopCheck(let counterCell):
            return "if (memory[\(counterCell)] == 0.0f) {"
        case .endHopCheck:
            return "}"

        default:
            return "/* \(uop.prettyDescription()) */"
        }
    }

    func emitLazy(_ lazy: Lazy, ctx: IRContext, kind: Kind?, isOut: Bool) -> String {
        switch lazy {
        case .constant(let constantId, let val):
            return kind == .simd ? "c\(constantId)" : "\(val)"
        case .variable(let id, _):
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
        case .global(let id):
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

    func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext, forceFloatType: Bool = false)
        -> String
    {
        let varId = extractVarId(uop.value)
        let isGlobal = ctx.globals.contains(varId)

        if uop.kind == .simd {
            // Track type as float32x4 (or float if forced)
            varEmittedTypes[varId] = forceFloatType ? .float_ : .float32x4

            if isGlobal {
                // For global variables, we need both a local variable declaration
                // AND a store to the global buffer for cross-block transfer
                let localVar = "simd\(varId)"
                let globalStore = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
                return "float32x4_t \(localVar) = \(expr); vst1q_f32(\(globalStore), \(localVar));"
            } else {
                let lhs = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
                let type = forceFloatType ? "float" : "float32x4_t"
                return "\(type) \(lhs) = \(expr);"
            }
        } else {
            // Track type as float
            varEmittedTypes[varId] = .float_

            let lhs = emitLazy(
                uop.value, ctx: ctx, kind: uop.kind, isOut: true)
            return isGlobal
                ? "\(lhs) = \(expr);"
                : "float \(lhs) = \(expr);"
        }
    }
}

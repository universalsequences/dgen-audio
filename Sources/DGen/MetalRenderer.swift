import Foundation

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

        var hasGradMemoryCalls = false
        for n in block.ops {
          if case .storeGradMemory(_, _) = n.op {
            hasGradMemoryCalls = true
            break
          } else if case .loadGradMemory(_) = n.op {
            hasGradMemoryCalls = true
            break
          }
        }
        let incr = hasGradMemoryCalls ? -1 : 1
        var beginLoop = UOp(op: .beginLoop(frameCountUOp, incr), value: .empty)
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

    // Add frameCount parameter for all Metal kernels (needed for output operations)
    parameters.append("    constant uint &frameCount [[buffer(\(bufferIndex))]]")
    bufferIndex += 1

    // If segmented, add segmentLen and segmentBase buffers
    if bufferRequirements.needsSegmenting {
      parameters.append("    constant uint &segmentLen [[buffer(\(bufferIndex))]]")
      bufferIndex += 1
      parameters.append("    constant uint &segmentBase [[buffer(\(bufferIndex))]]")
      bufferIndex += 1
    }

    if bufferRequirements.needsGradMemory {
      parameters.append("    device float *grad_memory [[buffer(\(bufferIndex))]]")
      bufferIndex += 1
    }

    if bufferRequirements.needsGrad {
      parameters.append("    device float *gradients [[buffer(\(bufferIndex))]]")
      bufferIndex += 1
    }

    parameters.append("    uint id [[thread_position_in_grid]]")
    if bufferRequirements.needsSegmenting {
      parameters.append("    uint tid [[thread_index_in_threadgroup]]")
    }

    kernels += "kernel void \(name)(\n"
    kernels += parameters.joined(separator: ",\n")
    kernels += "\n) {\n"

    // Suppress unused variable warnings (common in backward pass code generation)
    kernels += "  #pragma clang diagnostic push\n"
    kernels += "  #pragma clang diagnostic ignored \"-Wunused-variable\"\n"

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
      case .beginIf, .beginLoop, .beginRange, .beginForLoop:
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

    // Restore warning settings
    kernels += "  #pragma clang diagnostic pop\n"
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
      case .load, .store, .delay1, .memoryRead, .memoryWrite, .scalarMemoryWrite:
        needsMemory = true
      case let .spectralLossTape(sig1, sig2, _):
        inputs.insert(extractVarId(sig1))
        inputs.insert(extractVarId(sig2))
      case let .spectralLossTapeBackward(_, sig1, sig2, _, _, _, _, _):
        inputs.insert(extractVarId(sig1))
        inputs.insert(extractVarId(sig2))
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
    case let .scalarMemoryWrite(base, offset, value):
      return "memory[\(base) + (int)\(g(offset))] = \(g(value));"

    // Removed ring-only spectral helpers

    // Removed ring-only DFT compute

    // Removed ring-only spectral DFT compute branches

    // Removed ring-only spectralLoss

    case let .spectralLossTape(sig1, sig2, windowSize):
      // Deprecated in favor of IR abstractions (see .spectralLoss)
      // Forward pass: compute spectral loss using tape windows ending at current frame
      let numBins = windowSize / 2 + 1
      let slot1 = ctx.getGlobalId(extractVarId(sig1))
      let slot2 = ctx.getGlobalId(extractVarId(sig2))
      let idx = (uop.kind == .simd) ? "id" : "i"

      let code = """
        ({
            const int WIN_SIZE = \(windowSize);
            const int BASE1 = \(slot1) * frameCount;
            const int BASE2 = \(slot2) * frameCount;

            float totalError = 0.0;
            const int NUM_BINS = \(numBins);

            for (int binIndex = 0; binIndex < NUM_BINS; binIndex++) {
                float real1 = 0.0, imag1 = 0.0;
                float real2 = 0.0, imag2 = 0.0;

                for (int n = 0; n < WIN_SIZE; n++) {
                    int j = (int)\(idx) - (WIN_SIZE - 1) + n;
                    float s1 = (j < 0 || j >= frameCount) ? 0.0 : t[BASE1 + j];
                    float s2 = (j < 0 || j >= frameCount) ? 0.0 : t[BASE2 + j];
                    float angle = -2.0 * M_PI_F * (float)binIndex * (float)n / (float)WIN_SIZE;
                    float c = metal::cos(angle);
                    float s = metal::sin(angle);
                    real1 += s1 * c; imag1 += s1 * s;
                    real2 += s2 * c; imag2 += s2 * s;
                }
                float mag1 = metal::sqrt(real1 * real1 + imag1 * imag1);
                float mag2 = metal::sqrt(real2 * real2 + imag2 * imag2);
                float diff = mag1 - mag2;
                totalError += diff * diff;
            }

            totalError;
        })
        """
      return emitAssign(uop, code, ctx)

    // Removed ring-only spectralLossBackward

    case let .spectralLossTapeBackward(
      windowSize, sig1, sig2, upstreamGrad, grad1Dest, grad2Dest, _, _):
      // DEPRECATED: This direct Metal emission is no longer used.
      // The abstraction-based implementation in u_spectralLossTapeBackward
      // (Operators.swift) now compiles via normal UOp emission instead.
      // Kept here for reference and potential rollback.
      //
      // Original approach: compute gradients using all samples in the window.
      // To avoid cross-thread races, we average per-window per-bin
      // sample contributions and emit a single gradient for the current
      // sample id (instead of scattering to all j). This reduces the
      // late-frame bias of using only the last sample.
      //
      // IMPORTANT: A true "distributed across j" implementation that
      // scatters into gradients[... + j] for every j in the window would
      // require atomic adds or a multipass reduction (e.g., per-bin
      // accumulation into a scratch buffer, then a second pass to sum
      // into gradients). Without atomics/multipass, concurrent writes
      // from many threads are unsafe and can corrupt the gradients.
      // If you want that behavior, we need to add an atomic or
      // two-pass variant explicitly.
      let grad1Var = "t\(grad1Dest)"
      let grad2Var = "t\(grad2Dest)"
      let numBins = windowSize / 2 + 1
      let slot1 = ctx.getGlobalId(extractVarId(sig1))
      let slot2 = ctx.getGlobalId(extractVarId(sig2))
      let idx2 = (uop.kind == .simd) ? "id" : "i"

      let code = """
            const int WIN_SIZE = \(windowSize);
            const int BASE1 = \(slot1) * frameCount;
            const int BASE2 = \(slot2) * frameCount;

            float \(grad1Var) = 0.0;
            float \(grad2Var) = 0.0;
            const int NUM_BINS = \(numBins);

            for (int binIndex = 0; binIndex < NUM_BINS; binIndex++) {
                float real1 = 0.0, imag1 = 0.0;
                float real2 = 0.0, imag2 = 0.0;
                for (int n = 0; n < WIN_SIZE; n++) {
                    int j = (int)\(idx2) - (WIN_SIZE - 1) + n;
                    float s1 = (j < 0 || j >= frameCount) ? 0.0 : t[BASE1 + j];
                    float s2 = (j < 0 || j >= frameCount) ? 0.0 : t[BASE2 + j];
                    float angle = -2.0 * M_PI_F * (float)binIndex * (float)n / (float)WIN_SIZE;
                    float c = metal::cos(angle);
                    float s = metal::sin(angle);
                    real1 += s1 * c; imag1 += s1 * s;
                    real2 += s2 * c; imag2 += s2 * s;
                }
                float mag1 = metal::sqrt(real1 * real1 + imag1 * imag1);
                float mag2 = metal::sqrt(real2 * real2 + imag2 * imag2);

                float magDiff = mag1 - mag2;
                float lossGrad = 2.0 * magDiff;

                // Average contributions across the full window instead of last sample only
                float accum1 = 0.0;
                float accum2 = 0.0;
                for (int n = 0; n < WIN_SIZE; n++) {
                    float angle_n = -2.0 * M_PI_F * (float)binIndex * (float)n / (float)WIN_SIZE;
                    float c_n = metal::cos(angle_n);
                    float s_n = metal::sin(angle_n);
                    float sampleGrad1 = (real1 * c_n + imag1 * s_n) / (mag1 + 1e-8);
                    float sampleGrad2 = (real2 * c_n + imag2 * s_n) / (mag2 + 1e-8);
                    accum1 += (lossGrad * sampleGrad1);
                    accum2 += ((-lossGrad) * sampleGrad2);
                }
                // Normalize by window size to keep scale consistent
                \(grad1Var) += accum1 / (float)WIN_SIZE;
                \(grad2Var) += accum2 / (float)WIN_SIZE;
            }

            \(grad1Var) *= \(g(upstreamGrad));
            \(grad2Var) *= \(g(upstreamGrad));
        """
      return code

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
    case let .loadTape(val, offset):
      let varId = ctx.getGlobalId(extractVarId(val))
      let boundedFetch =
        "(\(g(offset)) < 0 || \(g(offset)) >= frameCount) ? 0.0 : t[\(varId) * frameCount + (int)\(g(offset))]"
      return emitAssign(uop, boundedFetch, ctx)
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

    case let .beginLoop(iters, step):
      if step < 0 {
        return "for (int i = \(g(iters)) - 1; i >= 0; i += \(step)) {"
      } else {
        return "for (uint i = 0; i < \(g(iters)); i += \(step)) {"
      }
    case let .beginForLoop(loopVar, count):
      guard case .variable(let varId, _) = loopVar else {
        fatalError("beginForLoop requires variable")
      }
      // Emit count as integer to avoid "t < 33.0" in loop bounds
      let countStr: String
      if case .constant(_, let val) = count {
        countStr = "\(UInt(val))"
      } else {
        countStr = "(uint)\(g(count))"
      }
      return "for (uint t\(varId) = 0; t\(varId) < \(countStr); t\(varId)++) {"
    case .endLoop: return "}"

    case .threadIndex:
      // In Metal, threadIndex maps to 'id' (thread_position_in_grid)
      let idx = (uop.kind == .simd) ? "id" : "i"
      return emitAssign(uop, idx, ctx)

    case let .cast(expr, castType):
      let typeStr = castType == .int ? "int" : "float"
      return emitAssign(uop, "(\(typeStr))\(g(expr))", ctx)

    case let .declareVar(value):
      // Declares and initializes a variable: float t = value;
      return emitAssign(uop, g(value), ctx)

    case let .beginRange(start, end):

      var startInt: Int = 0
      if case let .constant(_, val) = start {
        startInt = Int(val)
      }
      var endInt: Int = 0
      if case let .constant(_, val) = end {
        endInt = Int(val)
      } else if case let .variable(id, _) = end {
        if id == -1 {  // Special case for frameCount
          return "if (id >= \(startInt) && id < frameCount) {"
        }
      }
      return "if (id >= \(startInt) && id < \(endInt)) {"
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
        let tapeSlot = ctx.getGlobalId(id)
        let idx = (kind == .simd) ? "id" : "i"
        return
          "t[\(tapeSlot)*frameCount + \(needsSegmenting ? "segmentBase + " : "") \(idx)]"
      } else {
        return "t\(id)"
      }
    case let .global(id):
      // Global variables are accessed through global buffers
      let tapeSlot = ctx.getGlobalId(id)
      let idx = (kind == .simd) ? "id" : "i"
      return
        "t[\(tapeSlot)*frameCount + \(needsSegmenting ? "segmentBase + " : "")\(idx)]"
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
    if case .loadGradMemory = uop.op { return true }
    if case .storeGradMemory = uop.op { return true }
    return false
  }

  let needsGrad: Bool = scheduleItem.ops.contains { uop in
    if case .loadGrad = uop.op {
      return true
    } else if case .accumulateGrad = uop.op {
      return true
    } else if case .spectralLossTapeBackward = uop.op {
      // The spectral backward op writes directly to gradients[] now
      return true
    }
    return false
  }

  return RequiredBuffers(
    hasOutputOps: hasOutputOps, needsSegmenting: needsSegmenting,
    needsGradMemory: needsGradMemory, needsGrad: needsGrad
  )
}

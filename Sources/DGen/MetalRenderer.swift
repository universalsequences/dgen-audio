import Foundation

public class MetalRenderer: Renderer, UOpEmitter {
  let memoryVarID = -1  // Virtual ID for the global memory buffer
  var needsSegmenting = false
  var parallelRangeVars: Set<VarID> = []  // Track parallel range loop variable IDs
  var currentTemporality: Temporality = .frameBased  // Track temporality for gradient indexing
  private enum ParallelRangeMode {
    case loop      // Render as a for-loop (default)
    case thread    // Render as a single-thread index (thread-parallel kernel)
  }
  private var parallelRangeMode: ParallelRangeMode = .loop
  private var useReducedGradsSum = false
  private var staticGlobalVars: Set<VarID> = []

  public override init() {
  }

  public override func compile(
    scheduleItems: [ScheduleItem],
    ctx: IRContext,
    graph: Graph,
    totalMemorySlots: Int,
    name: String = "kernel"
  ) -> [CompiledKernel] {
    staticGlobalVars.removeAll()
    for item in scheduleItems {
      guard item.temporality == .static_ else { continue }
      for uop in item.ops {
        if case let .defineGlobal(varId) = uop.op {
          staticGlobalVars.insert(varId)
        }
      }
    }

    let expanded = scheduleItems.flatMap { item in
      if item.parallelPolicy == .threadParallel {
        return splitStaticParallelRanges(item)
      }
      return [SplitScheduleItem(item: item, threadCount: nil)]
    }
    let filtered = expanded.filter { !isNoOpStaticKernel($0.item, ctx: ctx) }
    let fused = fuseStaticThreadParallel(filtered)

    return fused.enumerated().map { (i, entry) in
      let scheduleItem = entry.item
      let parallelCount = entry.threadCount

      // Always use _0, _1 suffix to avoid 'kernel' being a reserved word in Metal
      let kernelName = "\(name)_\(i)"

      let bufferRequirements = analyzeRequiredBuffers(scheduleItem: scheduleItem)
      let hasGradWrites = scheduleItem.ops.contains { uop in
        if case .accumulateGrad = uop.op { return true }
        if case .accumulateTensorGrad = uop.op { return true }
        return false
      }
      let useReduced = bufferRequirements.needsReducedGradsSum && !hasGradWrites
      useReducedGradsSum = useReduced

      // Render parallelRange loops as thread-parallel only for split static kernels
      parallelRangeMode = parallelCount == nil ? .loop : .thread
      let source = render(
        name: kernelName, scheduleItem: scheduleItem, ctx: ctx, graph: graph,
        totalMemorySlots: totalMemorySlots)
      parallelRangeMode = .loop
      useReducedGradsSum = false

      let deps = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)
      let allBuffers = Set(deps.inputs + deps.outputs)

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

      if bufferRequirements.needsReducedGradsSum {
        if useReduced {
          bufferNames.append("reducedGradsSum")
        }
      }

      let threadGroupSize: Int? =
        (scheduleItem.kind == .scalar && parallelCount == nil) ? 1 : nil

      return CompiledKernel(
        name: kernelName,
        source: source,
        kind: scheduleItem.kind,
        buffers: bufferNames,
        threadGroupSize: threadGroupSize,
        threadCount: parallelCount,
        needsReducedGradsSum: useReduced,
        memorySize: max(totalMemorySlots, 1024)  // Match memory size calculation from render method
      )
    }
  }

  private struct SplitScheduleItem {
    let item: ScheduleItem
    let threadCount: Int?
  }

  private func splitStaticParallelRanges(_ scheduleItem: ScheduleItem) -> [SplitScheduleItem] {
    guard scheduleItem.temporality == .static_, scheduleItem.kind == .scalar else {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }

    let ops = scheduleItem.ops
    guard let beginRangeIdx = ops.firstIndex(where: {
      if case .beginRange = $0.op { return true }
      return false
    }),
      let endRangeIdx = ops.lastIndex(where: {
        if case .endRange = $0.op { return true }
        return false
      }),
      beginRangeIdx < endRangeIdx
    else {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }

    let prefixOps = Array(ops[0..<beginRangeIdx])
    let coreOps = Array(ops[(beginRangeIdx + 1)..<endRangeIdx])

    var segments: [(ops: [UOp], parallelCount: Int?)] = []
    var current: [UOp] = []
    var parallelDepth = 0
    var currentParallelCount: Int? = nil

    for op in coreOps {
      switch op.op {
      case let .beginParallelRange(count, _):
        if parallelDepth == 0 {
          if !current.isEmpty {
            segments.append((ops: current, parallelCount: nil))
            current.removeAll(keepingCapacity: true)
          }
          currentParallelCount = count
        }
        parallelDepth += 1
        current.append(op)
      case .endParallelRange:
        current.append(op)
        parallelDepth -= 1
        if parallelDepth == 0 {
          segments.append((ops: current, parallelCount: currentParallelCount))
          current.removeAll(keepingCapacity: true)
          currentParallelCount = nil
        }
      default:
        current.append(op)
      }
    }

    // Unbalanced parallel range - fall back to original kernel
    if parallelDepth != 0 {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }
    if !current.isEmpty {
      segments.append((ops: current, parallelCount: nil))
    }

    let hasParallel = segments.contains { $0.parallelCount != nil }
    guard hasParallel else {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }

    let beginRangeOp = ops[beginRangeIdx]
    let endRangeOp = ops[endRangeIdx]

    var splitItems: [SplitScheduleItem] = []
    for segment in segments {
      if segment.ops.isEmpty { continue }
      let item = ScheduleItem(kind: scheduleItem.kind, temporality: scheduleItem.temporality)
      item.ops.append(contentsOf: prefixOps)
      if segment.parallelCount == nil {
        item.ops.append(beginRangeOp)
      }
      item.ops.append(contentsOf: segment.ops)
      if segment.parallelCount == nil {
        item.ops.append(endRangeOp)
      }
      splitItems.append(SplitScheduleItem(item: item, threadCount: segment.parallelCount))
    }

    return splitItems.isEmpty
      ? [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
      : splitItems
  }

  private func isNoOpStaticKernel(_ scheduleItem: ScheduleItem, ctx: IRContext) -> Bool {
    guard scheduleItem.temporality == .static_ else { return false }
    return !kernelHasSideEffects(scheduleItem, ctx: ctx)
  }

  private func kernelHasSideEffects(_ scheduleItem: ScheduleItem, ctx: IRContext) -> Bool {
    for uop in scheduleItem.ops {
      switch uop.op {
      case .memoryWrite, .store, .delay1, .output,
        .accumulateGrad, .accumulateTensorGrad, .storeGradMemory:
        return true
      case .beginRange, .endRange, .beginLoop, .endLoop,
        .beginForLoop, .beginParallelRange, .endParallelRange,
        .beginIf, .endIf, .frameCount:
        break
      case .defineGlobal:
        // Only a definition, not a write
        break
      default:
        break
      }

      switch uop.value {
      case .variable(let varId, _):
        if ctx.globals.contains(varId) {
          return true
        }
      case .global(let varId):
        if ctx.globals.contains(varId) {
          return true
        }
      default:
        break
      }
    }
    return false
  }

  private struct KernelAccessInfo {
    var reads: Set<CellID> = []
    var writes: Set<CellID> = []
    var readsGlobals: Bool = false
    var writesGlobals: Bool = false
    var usesGradMemory: Bool = false
  }

  private func kernelAccessInfo(_ scheduleItem: ScheduleItem) -> KernelAccessInfo {
    var info = KernelAccessInfo()
    for uop in scheduleItem.ops {
      switch uop.op {
      case let .memoryRead(base, _):
        info.reads.insert(base)
      case let .memoryWrite(base, _, _):
        info.writes.insert(base)
      case let .load(cellId):
        info.reads.insert(cellId)
      case let .store(cellId, _):
        info.writes.insert(cellId)
      case let .delay1(cellId, _):
        info.reads.insert(cellId)
        info.writes.insert(cellId)
      case .loadGradMemory:
        info.usesGradMemory = true
      case .storeGradMemory:
        info.usesGradMemory = true
      case .defineGlobal:
        info.writesGlobals = true
      case .loadGlobal:
        info.readsGlobals = true
      default:
        break
      }
    }
    return info
  }

  private func canFuseStaticThreadParallel(_ a: SplitScheduleItem, _ b: SplitScheduleItem)
    -> Bool
  {
    guard let countA = a.threadCount, let countB = b.threadCount, countA == countB else {
      return false
    }
    guard a.item.temporality == .static_, b.item.temporality == .static_ else { return false }
    guard a.item.kind == .scalar, b.item.kind == .scalar else { return false }

    let accA = kernelAccessInfo(a.item)
    let accB = kernelAccessInfo(b.item)

    if accA.usesGradMemory || accB.usesGradMemory { return false }

    // If the first kernel writes globals, don't fuse if the next reads/writes globals.
    if accA.writesGlobals && (accB.readsGlobals || accB.writesGlobals) { return false }

    // Conservative: avoid fusing if the first kernel writes to any base
    // that the second kernel reads or writes.
    if !accA.writes.isDisjoint(with: accB.reads) { return false }
    if !accA.writes.isDisjoint(with: accB.writes) { return false }

    return true
  }

  private func fuseStaticThreadParallel(_ items: [SplitScheduleItem]) -> [SplitScheduleItem] {
    guard !items.isEmpty else { return [] }
    var fused: [SplitScheduleItem] = []
    var current = items[0]

    for next in items.dropFirst() {
      if canFuseStaticThreadParallel(current, next) {
        let merged = ScheduleItem(kind: current.item.kind, temporality: current.item.temporality)
        merged.ops.append(contentsOf: current.item.ops)
        merged.ops.append(contentsOf: next.item.ops)
        current = SplitScheduleItem(item: merged, threadCount: current.threadCount)
      } else {
        fused.append(current)
        current = next
      }
    }
    fused.append(current)
    return fused
  }

  public override func prepareSchedule(
    _ scheduleItems: inout [ScheduleItem],
    _ uopBlocks: [BlockUOps],
    _ ctx: IRContext,
    _ frameCount: Int
  ) {
    let frameCountUOp = Lazy.variable(-1, nil)

    var currentSchedule: ScheduleItem? = nil
    var currentKind: Kind? = nil
    var currentTemporality: Temporality? = nil
    var currentPolicy: ParallelPolicy? = nil
    var loopOpened = false
    var hasFrameLoop = false

    func closeCurrentKernel() {
      guard let schedule = currentSchedule, loopOpened else { return }
      if hasFrameLoop && currentKind == .scalar {
        schedule.ops.append(UOp(op: .endLoop, value: .empty))
      }
      schedule.ops.append(UOp(op: .endRange, value: .empty))
      scheduleItems.append(schedule)
    }

    func blockHasGradMemoryCalls(_ block: BlockUOps) -> Bool {
      return block.ops.contains { uop in
        switch uop.op {
        case .storeGradMemory, .loadGradMemory:
          return true
        default:
          return false
        }
      }
    }

    for block in uopBlocks {
      let needsNewKernel = currentKind != block.kind
        || currentTemporality != block.temporality
        || currentPolicy != block.parallelPolicy
        || block.forceNewKernel

      if needsNewKernel {
        closeCurrentKernel()

        let scheduleItem = ScheduleItem(kind: block.kind, temporality: block.temporality)
        scheduleItem.parallelPolicy = block.parallelPolicy
        scheduleItem.ops.append(UOp(op: .frameCount, value: .empty))

        for uop in block.ops {
          if case .defineGlobal = uop.op {
            scheduleItem.ops.append(uop)
          }
        }

        let isStaticScalar = block.temporality == .static_ && block.kind == .scalar
        let isScalar = block.kind == .scalar

        if isStaticScalar {
          // Static scalar blocks: run once, no frame loop needed
          var beginRange = UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty)
          beginRange.kind = block.kind
          scheduleItem.ops.append(beginRange)
          hasFrameLoop = false
        } else if isScalar {
          // Scalar kernels: thread 0 loops through frameCount
          var beginRange = UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty)
          beginRange.kind = block.kind
          scheduleItem.ops.append(beginRange)

          let incr = blockHasGradMemoryCalls(block) ? -1 : 1
          var beginLoop = UOp(op: .beginLoop(frameCountUOp, incr), value: .empty)
          beginLoop.kind = block.kind
          scheduleItem.ops.append(beginLoop)
          hasFrameLoop = true
        } else {
          // SIMD kernels: each thread processes one frame
          var beginRange = UOp(op: .beginRange(.constant(0, 0), frameCountUOp), value: .empty)
          beginRange.kind = block.kind
          scheduleItem.ops.append(beginRange)
          hasFrameLoop = true
        }

        currentSchedule = scheduleItem
        currentKind = block.kind
        currentTemporality = block.temporality
        currentPolicy = block.parallelPolicy
        loopOpened = true
      }

      if let schedule = currentSchedule {
        for uop in block.ops {
          if case .defineGlobal = uop.op { continue }
          var typedUOp = uop
          typedUOp.kind = block.kind
          schedule.ops.append(typedUOp)
        }
      }
    }

    closeCurrentKernel()
  }

  public override func render(
    name: String, scheduleItem: ScheduleItem, ctx: IRContext, graph: Graph,
    totalMemorySlots: Int
  ) -> String {
    var kernels = ""
    parallelRangeVars.removeAll()  // Reset parallel range tracking for new kernel
    currentTemporality = scheduleItem.temporality  // Set temporality for gradient indexing
    let (inputs, outputs) = analyzeDependencies(scheduleItem: scheduleItem, ctx: ctx)

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

    if bufferRequirements.needsReducedGradsSum {
      parameters.append("    device float *reducedGradsSum [[buffer(\(bufferIndex))]]")
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

    // For static blocks, define i=0 since there's no frame loop
    if case .static_ = currentTemporality {
      kernels += "  uint i = 0; // Static block - no frame loop\n"
    }

    for uop in scheduleItem.ops {
      var diff = 0
      switch uop.op {
      case .beginIf, .beginLoop, .beginRange, .beginForLoop, .beginParallelRange:
        diff = 1
      case .endIf, .endLoop, .endRange, .endParallelRange:
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

    // Helper to check if a Lazy value contains a global reference
    func checkLazyForGlobal(_ lazy: Lazy) {
      switch lazy {
      case let .global(id):
        // Global values need the tape buffer
        inputs.insert(id)
        outputs.insert(id)
      case let .variable(id, _):
        // Check if this variable is a global in the context
        if ctx.globals.contains(id) {
          inputs.insert(id)
          outputs.insert(id)
        }
      default:
        break
      }
    }

    for uop in scheduleItem.ops {
      // Check the uop's value for global references
      checkLazyForGlobal(uop.value)

      switch uop.op {
      case let .defineGlobal(varId):
        outputs.insert(varId)
      case let .loadGlobal(varId):
        inputs.insert(varId)
      case .load, .store, .delay1, .memoryRead, .memoryWrite:
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
          return emitAssign(uop, "(\(g(a)) - metal::floor(\(g(a))))", ctx)
        } else {
          return emitAssign(uop, "(\(g(a)) - metal::floor(\(g(a)) / \(val)) * \(val))", ctx)
        }
      default:
        return emitAssign(uop, "metal::fmod(\(g(a)), \(g(b)))", ctx)
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
    case let .noise(cellId):
      // Xorshift32 PRNG - better spectral properties than LCG
      let expr = """
        ({
          uint s = as_type<uint>(memory[\(cellId)]);
          if (s == 0u) s = 1u;
          s ^= s << 13; s ^= s >> 17; s ^= s << 5;
          memory[\(cellId)] = as_type<float>(s);
          float(s) / 4294967296.0f;
        })
        """
      return emitAssign(uop, expr, ctx)
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
    case .frameIndex: return emitAssign(uop, "i", ctx)
    case let .loadTape(val, offset):
      let varId = ctx.getGlobalId(extractVarId(val))
      let boundedFetch =
        "(\(g(offset)) < 0 || \(g(offset)) >= frameCount) ? 0.0 : t[\(varId) * frameCount + (int)\(g(offset))]"
      return emitAssign(uop, boundedFetch, ctx)
    case let .store(cell, val): return "memory[\(cell)] = \(g(val));"
    case let .storeGradMemory(cell, val): return "grad_memory[\(cell)] = \(g(val));"
    case let .loadGrad(gradId):
      if case .static_ = currentTemporality {
        if useReducedGradsSum {
          return emitAssign(uop, "reducedGradsSum[\(gradId)]", ctx)
        } else {
          // Static blocks must sum gradients from all frames
          let varName = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
          return """
            float \(varName) = 0.0;
            for (uint _gfi = 0; _gfi < frameCount; _gfi++) {
              \(varName) += gradients[frameCount * \(gradId) + _gfi];
            }
          """
        }
      } else {
        let idx = uop.kind == .scalar ? "i" : "id"
        return emitAssign(uop, "gradients[frameCount * \(gradId) + \(idx)]", ctx)
      }
    case let .accumulateGrad(gradId, val):
      let idx: String
      if case .static_ = currentTemporality {
        idx = "0"
      } else {
        idx = uop.kind == .scalar ? "i" : "id"
      }
      return "gradients[frameCount*\(gradId)+\(idx)] += \(g(val));"
    case let .loadTensorGrad(baseGradId, indexLazy):
      if case .static_ = currentTemporality {
        if useReducedGradsSum {
          return emitAssign(
            uop,
            "reducedGradsSum[(\(baseGradId)+(int)(\(g(indexLazy))))]",
            ctx)
        } else {
          // Static blocks must sum gradients from all frames
          let varName = emitLazy(uop.value, ctx: ctx, kind: uop.kind, isOut: true)
          return """
            float \(varName) = 0.0;
            for (uint _gfi = 0; _gfi < frameCount; _gfi++) {
              \(varName) += gradients[frameCount*(\(baseGradId)+(int)(\(g(indexLazy))))+_gfi];
            }
          """
        }
      } else {
        let idx = uop.kind == .scalar ? "i" : "id"
        return emitAssign(uop, "gradients[frameCount*(\(baseGradId)+(int)(\(g(indexLazy))))+\(idx)]", ctx)
      }
    case let .accumulateTensorGrad(baseGradId, indexLazy, valueLazy):
      let idx: String
      if case .static_ = currentTemporality {
        idx = "0"
      } else {
        idx = uop.kind == .scalar ? "i" : "id"
      }
      return "gradients[frameCount*(\(baseGradId)+(int)(\(g(indexLazy))))+\(idx)] += \(g(valueLazy));"
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

    // Parallel range - for Metal, could be thread-parallel for static tensors
    // For now, render as a loop (future: check block.temporality and use thread-parallel for static)
    case let .beginParallelRange(count, _):
      guard case .variable(let varId, _) = uop.value else {
        fatalError("beginParallelRange requires variable")
      }
      parallelRangeVars.insert(varId)  // Track this as a parallel range loop variable
      if parallelRangeMode == .thread {
        return "if (id < \(count)) { uint _pr\(varId) = id;"
      }
      return "for (uint _pr\(varId) = 0; _pr\(varId) < \(count); _pr\(varId)++) {"
    case .endParallelRange:
      // IMPORTANT: Memory fence required for correctness on Metal.
      //
      // Without this fence, the Metal compiler may hoist memory reads outside the
      // enclosing frame loop, causing all frames to read the initial state instead
      // of seeing updates from previous frames. For example:
      //
      //   for (frame = 0; frame < 4; frame++) {
      //     for (i = 0; i < 4; i++) {
      //       val = memory[history + i];   // Metal may hoist this read
      //       memory[history + i] = val + 1;
      //     }
      //   }
      //
      // The fence ensures writes complete and are visible before the next frame's reads.
      // This matches how the C backend behaves (sequential consistency on CPU).
      if parallelRangeMode == .thread {
        return "}"
      }
      return "} atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);"
    case .parallelIndex:
      // Return the loop variable from the enclosing parallelRange
      guard case .variable(let varId, _) = uop.value else {
        fatalError("parallelIndex requires variable")
      }
      return emitAssign(uop, "_pr\(varId)", ctx)

    // Local tensor operations for SIMD-across-frames
    case .declareLocalTensor(let varId, let size):
      return "float localT\(varId)[\(size)];"

    case .localTensorRead(let varId, let idx):
      return emitAssign(uop, "localT\(varId)[(int)\(g(idx))]", ctx)

    case .localTensorWrite(let varId, let idx, let val):
      return "localT\(varId)[(int)\(g(idx))] = \(g(val));"

    case .beginInlineLoop(let count, _):
      guard case .variable(let varId, _) = uop.value else {
        fatalError("beginInlineLoop requires variable")
      }
      let countStr: String
      if case .constant(_, let val) = count {
        countStr = "\(Int(val))"
      } else {
        countStr = "(int)\(g(count))"
      }
      return "for (int t\(varId) = 0; t\(varId) < \(countStr); t\(varId)++) {"

    case .endInlineLoop:
      return "}"

    case .frameTensorChainMarker(let shape):
      return "/* ====== SIMD-ACROSS-FRAMES: Frame-Tensor Chain Block (shape: \(shape)) ====== */"

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
      } else if parallelRangeVars.contains(id) {
        // This is a parallel range loop variable - use _pr prefix
        return "_pr\(id)"
      } else if ctx.globals.contains(id) {
        let tapeSlot = ctx.getGlobalId(id)
        let idx = staticGlobalVars.contains(id) ? "0" : ((kind == .simd) ? "id" : "i")
        return
          "t[\(tapeSlot)*frameCount + \(needsSegmenting ? "segmentBase + " : "") \(idx)]"
      } else {
        return "t\(id)"
      }
    case let .global(id):
      // Global variables are accessed through global buffers
      let tapeSlot = ctx.getGlobalId(id)
      let idx = staticGlobalVars.contains(id) ? "0" : ((kind == .simd) ? "id" : "i")
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
  let needsReducedGradsSum: Bool
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

  let hasGradReads: Bool = scheduleItem.ops.contains { uop in
    if case .loadGrad = uop.op { return true }
    if case .loadTensorGrad = uop.op { return true }
    return false
  }

  let hasGradWrites: Bool = scheduleItem.ops.contains { uop in
    if case .accumulateGrad = uop.op { return true }
    if case .accumulateTensorGrad = uop.op { return true }
    return false
  }

  let needsGrad: Bool = hasGradReads || hasGradWrites

  let needsReducedGradsSum: Bool = false

  return RequiredBuffers(
    hasOutputOps: hasOutputOps, needsSegmenting: needsSegmenting,
    needsGradMemory: needsGradMemory, needsGrad: needsGrad,
    needsReducedGradsSum: needsReducedGradsSum
  )
}

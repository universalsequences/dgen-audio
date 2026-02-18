import Foundation

public class MetalRenderer: Renderer, UOpEmitter {
  let memoryVarID = -1  // Virtual ID for the global memory buffer
  var parallelRangeVars: Set<VarID> = []  // Track parallel range loop variable IDs
  var currentTemporality: Temporality = .frameBased  // Track temporality for gradient indexing
  private var currentFrameOrder: FrameOrder = .parallel
  var frameIndexOverride: String? = nil
  private enum ParallelRangeMode {
    case loop  // Render as a for-loop (default)
    case thread  // Render as a single-thread index (thread-parallel kernel)
  }
  private var parallelRangeMode: ParallelRangeMode = .loop
  private var useReducedGradsSum = false
  private var staticGlobalVars: Set<VarID> = []
  private var currentThreadCountScale: Int? = nil
  private var currentThreadCountOverride: Int? = nil
  private var isGemmKernel: Bool = false
  /// Kernel uses threadgroup scratch (threadGroupSize=1) — skip device memory fences
  private var usesThreadgroupScratch: Bool = false
  /// Track scalarType of emitted UOps by their VarID for offset type lookups
  private var varScalarTypes: [VarID: CastType] = [:]

  public override init() {
  }

  public override func compile(
    scheduleItems: [ScheduleItem],
    ctx: IRContext,
    graph: Graph,
    totalMemorySlots: Int,
    name: String = "kernel"
  ) -> [CompiledKernel] {
    repairCrossKernelVarDependencies(scheduleItems, ctx: ctx)

    staticGlobalVars.removeAll()
    for item in scheduleItems {
      guard item.temporality == .static_ else { continue }
      for uop in item.ops {
        if case .defineGlobal(let varId) = uop.op {
          staticGlobalVars.insert(varId)
        }
      }
    }

    let expanded = scheduleItems.flatMap { item in
      // Static blocks dispatched as staticThreads(1) may contain beginParallelRange
      // that needs to be split into separate thread-parallel kernels
      if case .staticThreads(1) = item.dispatchMode {
        let hasParallelRange = item.ops.contains {
          if case .beginParallelRange = $0.op { return true }
          return false
        }
        if hasParallelRange {
          return splitStaticParallelRanges(item, ctx: ctx)
        }
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
      let useReduced = bufferRequirements.needsReducedGradsSum
      useReducedGradsSum = useReduced

      // Render parallelRange loops as thread-parallel only for split static kernels.
      // When `parallelCount` is set, beginParallelRange is lowered as `_pr = id`.
      parallelRangeMode = (parallelCount == nil) ? .loop : .thread
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

      if bufferRequirements.needsReducedGradsSum {
        bufferNames.append("reducedGradsSum")
      }

      // If splitStaticParallelRanges produced a thread count, override to staticThreads(N).
      var finalDispatchMode =
        parallelCount.map { DispatchMode.staticThreads($0) }
        ?? scheduleItem.dispatchMode

      // FFT kernels using threadgroup scratch need threadGroupSize=1 so each thread
      // gets its own on-chip scratch arrays (no sharing between threads).
      let usesThreadgroupScratch = scheduleItem.ops.contains {
        if case .threadgroupArrayDecl = $0.op { return true }
        return false
      }
      if usesThreadgroupScratch {
        finalDispatchMode = .perFrameThreadgroup1
      }

      return CompiledKernel(
        name: kernelName,
        source: source,
        frameOrder: scheduleItem.frameOrder,
        temporality: scheduleItem.temporality,
        buffers: bufferNames,
        dispatchMode: finalDispatchMode,
        needsReducedGradsSum: useReduced,
        memorySize: max(totalMemorySlots, 1024)  // Match memory size calculation from render method
      )
    }
  }

  private func repairCrossKernelVarDependencies(_ scheduleItems: [ScheduleItem], ctx: IRContext) {
    guard scheduleItems.count > 1 else { return }

    let excluded: Set<VarID> = [-1]  // -1 is frameCount sentinel
    let existingGlobals = Set(ctx.globals)

    let defsByKernel = scheduleItems.map { variableIdsDefined(in: $0.ops).subtracting(excluded) }
    let usesByKernel = scheduleItems.map { variableIdsUsed(in: $0.ops).subtracting(excluded) }

    var producerToVars: [Int: Set<VarID>] = [:]
    var consumerToVars: [Int: Set<VarID>] = [:]
    var newlyPromoted = Set<VarID>()

    for consumerIdx in scheduleItems.indices {
      let unresolved = usesByKernel[consumerIdx]
        .subtracting(defsByKernel[consumerIdx])
        .subtracting(existingGlobals)
      guard !unresolved.isEmpty else { continue }

      for varId in unresolved {
        var producerIdx: Int? = nil
        var scan = consumerIdx - 1
        while scan >= 0 {
          if defsByKernel[scan].contains(varId) {
            producerIdx = scan
            break
          }
          scan -= 1
        }
        guard let producerIdx else { continue }

        producerToVars[producerIdx, default: []].insert(varId)
        consumerToVars[consumerIdx, default: []].insert(varId)
        newlyPromoted.insert(varId)
      }
    }

    guard !newlyPromoted.isEmpty else { return }
    for varId in newlyPromoted where !ctx.globals.contains(varId) {
      ctx.globals.append(varId)
    }

    for (kernelIdx, vars) in producerToVars {
      for varId in vars {
        insertDefineGlobalIfMissing(varId, in: scheduleItems[kernelIdx])
      }
    }
    for (kernelIdx, vars) in consumerToVars {
      for varId in vars {
        insertLoadGlobalIfMissing(varId, in: scheduleItems[kernelIdx])
      }
    }
  }

  private func insertDefineGlobalIfMissing(_ varId: VarID, in scheduleItem: ScheduleItem) {
    let hasDefine = scheduleItem.ops.contains {
      if case .defineGlobal(let existing) = $0.op { return existing == varId }
      return false
    }
    guard !hasDefine else { return }

    let define = UOp(op: .defineGlobal(varId), value: .global(varId))
    scheduleItem.ops.insert(define, at: globalPrologueInsertionIndex(in: scheduleItem))
  }

  private func insertLoadGlobalIfMissing(_ varId: VarID, in scheduleItem: ScheduleItem) {
    let hasLoad = scheduleItem.ops.contains {
      if case .loadGlobal(let existing) = $0.op { return existing == varId }
      return false
    }
    guard !hasLoad else { return }

    let load = UOp(op: .loadGlobal(varId), value: .variable(varId, nil))
    scheduleItem.ops.insert(load, at: globalPrologueInsertionIndex(in: scheduleItem))
  }

  private func globalPrologueInsertionIndex(in scheduleItem: ScheduleItem) -> Int {
    var index = 0
    while index < scheduleItem.ops.count {
      switch scheduleItem.ops[index].op {
      case .frameCount, .defineGlobal, .loadGlobal:
        index += 1
      default:
        return index
      }
    }
    return index
  }

  private struct SplitScheduleItem {
    let item: ScheduleItem
    let threadCount: Int?
  }

  private func splitStaticParallelRanges(_ scheduleItem: ScheduleItem, ctx: IRContext)
    -> [SplitScheduleItem]
  {
    guard scheduleItem.temporality == .static_, scheduleItem.frameOrder == .sequential else {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }

    let ops = scheduleItem.ops
    guard
      let beginRangeIdx = ops.firstIndex(where: {
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
    var maxParallelDepth = 0
    var currentParallelCount: Int? = nil

    for op in coreOps {
      switch op.op {
      case .beginParallelRange(let count, _):
        if parallelDepth == 0 {
          if !current.isEmpty {
            segments.append((ops: current, parallelCount: nil))
            current.removeAll(keepingCapacity: true)
          }
          currentParallelCount = count
        }
        parallelDepth += 1
        if parallelDepth > maxParallelDepth { maxParallelDepth = parallelDepth }
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
    // Thread-mode lowering maps `beginParallelRange` to `_pr = id`.
    // Nested parallel ranges are not representable in that mode yet.
    if maxParallelDepth > 1 {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }
    if !current.isEmpty {
      segments.append((ops: current, parallelCount: nil))
    }

    let hasParallel = segments.contains { $0.parallelCount != nil }
    guard hasParallel else {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }

    // Safety guard: do not split if a later segment reads a non-global temp
    // that was defined in an earlier segment. Splitting would move the use into
    // a separate kernel where that local variable is out of scope.
    if hasCrossSegmentLocalVariableDependency(
      prefixOps: prefixOps, segments: segments, globalVarIds: Set(ctx.globals))
    {
      return [SplitScheduleItem(item: scheduleItem, threadCount: nil)]
    }

    let beginRangeOp = ops[beginRangeIdx]
    let endRangeOp = ops[endRangeIdx]

    var splitItems: [SplitScheduleItem] = []
    for segment in segments {
      if segment.ops.isEmpty { continue }
      let item = ScheduleItem(
        frameOrder: scheduleItem.frameOrder, vectorWidth: scheduleItem.vectorWidth,
        temporality: scheduleItem.temporality)
      item.dispatchMode = scheduleItem.dispatchMode
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

  private func variableIdsDefined(in ops: [UOp]) -> Set<VarID> {
    var defs = Set<VarID>()
    for op in ops {
      if case .variable(let varId, _) = op.value {
        defs.insert(varId)
      }
    }
    return defs
  }

  private func variableIdsUsed(in lazy: Lazy) -> Set<VarID> {
    if case .variable(let varId, _) = lazy { return [varId] }
    return []
  }

  private func variableIdsUsed(in op: Op) -> Set<VarID> {
    switch op {
    case .store(_, let a), .delay1(_, let a), .abs(let a), .sign(let a), .sin(let a), .cos(let a),
      .tan(let a), .tanh(let a), .exp(let a), .log(let a), .log10(let a), .sqrt(let a),
      .floor(let a), .ceil(let a), .round(let a), .beginIf(let a), .beginReverseLoop(let a),
      .beginHopCheck(let a), .setFrameIndex(let a), .identity(let a), .declareVar(let a),
      .cast(let a, _), .loadTape(let a, _), .output(_, let a):
      return variableIdsUsed(in: a)

    case .mutate(let a, let b), .add(let a, let b), .sub(let a, let b), .mul(let a, let b),
      .div(let a, let b), .pow(let a, let b), .atan2(let a, let b), .mod(let a, let b),
      .gt(let a, let b), .gte(let a, let b), .lte(let a, let b), .lt(let a, let b),
      .eq(let a, let b), .min(let a, let b), .max(let a, let b), .and(let a, let b),
      .or(let a, let b), .xor(let a, let b), .beginRange(let a, let b), .beginForLoop(let a, let b):
      return variableIdsUsed(in: a).union(variableIdsUsed(in: b))

    case .memoryRead(_, let a), .beginLoop(let a, _), .simdgroupLoad(_, let a, _, _):
      return variableIdsUsed(in: a)

    case .memoryWrite(_, let a, let b), .memoryAccumulate(_, let a, let b):
      return variableIdsUsed(in: a).union(variableIdsUsed(in: b))

    case .simdgroupStore(let src, _, let off, _):
      return variableIdsUsed(in: src).union(variableIdsUsed(in: off))

    case .simdgroupMultiplyAccumulate(let a, let b, let acc):
      return variableIdsUsed(in: a).union(variableIdsUsed(in: b)).union(variableIdsUsed(in: acc))

    case .latch(let a, let b):
      return variableIdsUsed(in: a).union(variableIdsUsed(in: b))

    case .gswitch(let c, let a, let b):
      return variableIdsUsed(in: c).union(variableIdsUsed(in: a)).union(variableIdsUsed(in: b))

    case .selector(let m, let opts):
      return opts.reduce(variableIdsUsed(in: m)) { acc, lazy in
        acc.union(variableIdsUsed(in: lazy))
      }

    case .mse(let a, let b):
      return variableIdsUsed(in: a).union(variableIdsUsed(in: b))

    default:
      return []
    }
  }

  private func variableIdsUsed(in ops: [UOp]) -> Set<VarID> {
    ops.reduce(into: Set<VarID>()) { acc, uop in
      acc.formUnion(variableIdsUsed(in: uop.op))
      if let tensorIndex = uop.tensorIndex {
        acc.formUnion(variableIdsUsed(in: tensorIndex))
      }
    }
  }

  private func hasCrossSegmentLocalVariableDependency(
    prefixOps: [UOp],
    segments: [(ops: [UOp], parallelCount: Int?)],
    globalVarIds: Set<VarID>
  ) -> Bool {
    let excluded: Set<VarID> = globalVarIds.union([-1])  // -1 is frameCount sentinel
    let prefixDefs = variableIdsDefined(in: prefixOps)
    var priorSegmentDefs = Set<VarID>()

    for segment in segments {
      let uses = variableIdsUsed(in: segment.ops).subtracting(excluded)
      if !uses.intersection(priorSegmentDefs).isEmpty {
        return true
      }

      let defs = variableIdsDefined(in: segment.ops).subtracting(excluded).subtracting(prefixDefs)
      priorSegmentDefs.formUnion(defs)
    }

    return false
  }

  private func isNoOpStaticKernel(_ scheduleItem: ScheduleItem, ctx: IRContext) -> Bool {
    guard scheduleItem.temporality == .static_ else { return false }
    return !kernelHasSideEffects(scheduleItem, ctx: ctx)
  }

  private func kernelHasSideEffects(_ scheduleItem: ScheduleItem, ctx: IRContext) -> Bool {
    for uop in scheduleItem.ops {
      switch uop.op {
      case .memoryWrite, .memoryAccumulate, .store, .delay1, .output, .simdgroupStore:
        return true
      case .beginRange, .endRange, .beginLoop, .beginReverseLoop, .endLoop,
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
  }

  private func kernelAccessInfo(_ scheduleItem: ScheduleItem) -> KernelAccessInfo {
    var info = KernelAccessInfo()
    for uop in scheduleItem.ops {
      switch uop.op {
      case .memoryRead(let base, _):
        info.reads.insert(base)
      case .memoryWrite(let base, _, _):
        info.writes.insert(base)
      case .load(let cellId):
        info.reads.insert(cellId)
      case .store(let cellId, _):
        info.writes.insert(cellId)
      case .delay1(let cellId, _):
        info.reads.insert(cellId)
        info.writes.insert(cellId)
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
    if a.item.dispatchMode != b.item.dispatchMode { return false }
    guard a.item.temporality == .static_, b.item.temporality == .static_ else { return false }
    guard a.item.frameOrder == .sequential, b.item.frameOrder == .sequential else { return false }

    let accA = kernelAccessInfo(a.item)
    let accB = kernelAccessInfo(b.item)

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
        let merged = ScheduleItem(
          frameOrder: current.item.frameOrder, vectorWidth: current.item.vectorWidth,
          temporality: current.item.temporality)
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
    var currentFrameOrderForLoop: FrameOrder? = nil
    var loopOpened = false
    var hasFrameLoop = false
    var hopCheckOpen = false

    func closeCurrentKernel() {
      guard let schedule = currentSchedule, loopOpened else { return }
      if hopCheckOpen {
        schedule.ops.append(UOp(op: .endHopCheck, value: .empty))
        hopCheckOpen = false
      }
      if hasFrameLoop && currentFrameOrderForLoop == .sequential {
        schedule.ops.append(UOp(op: .endLoop, value: .empty))
      }
      schedule.ops.append(UOp(op: .endRange, value: .empty))
      scheduleItems.append(schedule)
    }

    func scaledFrameCount(_ scale: Int?, _ schedule: ScheduleItem) -> Lazy {
      guard let scale, scale != 1 else { return frameCountUOp }
      let scaleConst = ctx.useConstant(src: nil, value: Float(scale))
      let dest = ctx.useVariable(src: nil, trackInValues: false)
      schedule.ops.append(UOp(op: .mul(frameCountUOp, scaleConst), value: dest))
      return dest
    }

    for block in uopBlocks {
      closeCurrentKernel()

      let scheduleItem = ScheduleItem(
        frameOrder: block.frameOrder, vectorWidth: block.vectorWidth, temporality: block.temporality
      )
      scheduleItem.dispatchMode = block.dispatchMode
      scheduleItem.ops.append(UOp(op: .frameCount, value: .empty))

      for uop in block.ops {
        if case .defineGlobal = uop.op {
          scheduleItem.ops.append(uop)
        }
      }

      switch block.dispatchMode {
      case .selfManaged:
        // Block contains its own frame loops (BPTT) -- dispatch 1 thread, no wrapping.
        let beginRange = UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty)
        scheduleItem.ops.append(beginRange)
        hasFrameLoop = false

      case .staticThreads(let n):
        // Static blocks: dispatch N threads with no frame loop.
        let beginRange = UOp(
          op: .beginRange(.constant(0, 0), .constant(0, Float(n))), value: .empty)
        scheduleItem.ops.append(beginRange)
        hasFrameLoop = false

      case .fixedWithFrameLoop(let tensorThreads):
        // Scalar frame loop, but one GPU thread per tensor element.
        let beginRange = UOp(
          op: .beginRange(.constant(0, 0), .constant(0, Float(tensorThreads))),
          value: .empty
        )
        scheduleItem.ops.append(beginRange)
        let beginLoop = UOp(op: .beginLoop(frameCountUOp, 1), value: .empty)
        scheduleItem.ops.append(beginLoop)
        hasFrameLoop = true

      case .singleThreaded:
        // Scalar kernels: thread 0 loops through frameCount
        let beginRange = UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty)
        scheduleItem.ops.append(beginRange)
        let beginLoop = UOp(op: .beginLoop(frameCountUOp, 1), value: .empty)
        scheduleItem.ops.append(beginLoop)
        hasFrameLoop = true

      case .perFrame, .perFrameThreadgroup1:
        // SIMD kernels: each thread processes one frame
        let beginRange = UOp(op: .beginRange(.constant(0, 0), frameCountUOp), value: .empty)
        scheduleItem.ops.append(beginRange)
        hasFrameLoop = true

      case .perFrameScaled(let n):
        if block.frameOrder == .sequential {
          // Sequential with scale: 1 thread loops over frameCount*scale iterations
          let beginRange = UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty)
          scheduleItem.ops.append(beginRange)
          let loopCount = scaledFrameCount(n, scheduleItem)
          let beginLoop = UOp(op: .beginLoop(loopCount, 1), value: .empty)
          scheduleItem.ops.append(beginLoop)
        } else {
          // SIMD kernels: frameCount * N threads
          let rangeEnd = scaledFrameCount(n, scheduleItem)
          let beginRange = UOp(op: .beginRange(.constant(0, 0), rangeEnd), value: .empty)
          scheduleItem.ops.append(beginRange)
        }
        hasFrameLoop = true

      case .gemm(let tilesM, let tilesN):
        // GEMM: dispatch tilesM × tilesN threadgroups of 32 threads each.
        // No frame wrapping — GEMM is a static-like operation.
        let totalThreads = tilesM * tilesN * 32
        let beginRange = UOp(
          op: .beginRange(.constant(0, 0), .constant(0, Float(totalThreads))), value: .empty)
        scheduleItem.ops.append(beginRange)
        hasFrameLoop = false
      }

      // Hop-based temporality: wrap body in hop check conditional.
      // Applies to all dispatch modes that produce a frame loop.
      if hasFrameLoop, case .hopBased(_, let counterNodeId) = block.temporality {
        guard let counterLazy = ctx.values[counterNodeId] else {
          fatalError("Hop counter node \(counterNodeId) not found in ctx.values")
        }
        scheduleItem.ops.append(UOp(op: .beginHopCheck(counterLazy), value: .empty))
        hopCheckOpen = true
      }

      currentSchedule = scheduleItem
      currentFrameOrderForLoop = block.frameOrder
      loopOpened = true

      for uop in block.ops {
        if case .defineGlobal = uop.op { continue }
        scheduleItem.ops.append(uop)
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
    varScalarTypes.removeAll()  // Reset type tracking for new kernel
    frameIndexOverride = nil
    currentThreadCountScale = scheduleItem.dispatchMode.threadCountScale
    currentThreadCountOverride = scheduleItem.dispatchMode.fixedThreadCount
    currentTemporality = scheduleItem.temporality  // Set temporality for gradient indexing
    currentFrameOrder = scheduleItem.frameOrder
    isGemmKernel = {
      if case .gemm = scheduleItem.dispatchMode { return true }
      return false
    }()
    usesThreadgroupScratch = scheduleItem.ops.contains {
      if case .threadgroupArrayDecl = $0.op { return true }
      return false
    }
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

    if bufferRequirements.needsReducedGradsSum {
      parameters.append("    device float *reducedGradsSum [[buffer(\(bufferIndex))]]")
      bufferIndex += 1
    }

    if isGemmKernel {
      parameters.append("    uint3 gid [[threadgroup_position_in_grid]]")
    } else {
      parameters.append("    uint id [[thread_position_in_grid]]")
    }

    if isGemmKernel {
      kernels += "#include <metal_simdgroup_matrix>\n"
    }
    kernels += "kernel void \(name)(\n"
    kernels += parameters.joined(separator: ",\n")
    kernels += "\n) {\n"

    // Suppress unused variable warnings (common in backward pass code generation)
    kernels += "  #pragma clang diagnostic push\n"
    kernels += "  #pragma clang diagnostic ignored \"-Wunused-variable\"\n"

    var indent = 1

    // For static blocks, define i=0 since there's no frame loop
    if case .static_ = currentTemporality {
      kernels += "  uint i = 0; // Static block - no frame loop\n"
    }

    for uop in scheduleItem.ops {
      var diff = 0
      switch uop.op {
      case .beginIf, .beginLoop, .beginReverseLoop, .beginRange, .beginForLoop, .beginParallelRange,
        .beginHopCheck:
        diff = 1
      case .endIf, .endLoop, .endRange, .endParallelRange, .endHopCheck:
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
      case .global(let id):
        // Global values need the tape buffer
        inputs.insert(id)
        outputs.insert(id)
      case .variable(let id, _):
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
      case .defineGlobal(let varId):
        outputs.insert(varId)
      case .loadGlobal(let varId):
        inputs.insert(varId)
      case .loadTape(let val, _):
        checkLazyForGlobal(val)
      case .load, .store, .delay1, .memoryRead, .memoryWrite, .memoryAccumulate, .noise,
        .simdgroupLoad, .simdgroupStore:
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

  /// Check if a Lazy value was emitted as an int-typed variable
  private func isIntTyped(_ lazy: Lazy) -> Bool {
    guard case .variable(let varId, _) = lazy else { return false }
    return varScalarTypes[varId] == .int
  }

  /// Render a Lazy value, optionally as an integer literal for int-typed constants
  private func emitLazyTyped(_ lazy: Lazy, ctx: IRContext, asInt: Bool) -> String {
    if asInt, case .constant(_, let val) = lazy {
      return "\(Int(val))"
    }
    return emitLazy(lazy, ctx: ctx, isOut: false)
  }

  /// Returns "(int)" cast prefix if offset is not already int-typed, empty string otherwise
  private func intCastPrefix(for offset: Lazy) -> String {
    return isIntTyped(offset) ? "" : "(int)"
  }

  func emit(_ uop: UOp, ctx: IRContext) -> String {
    // Track scalarType for this UOp's destination variable
    if case .variable(let varId, _) = uop.value {
      varScalarTypes[varId] = uop.scalarType
    }

    let g = { self.emitLazy($0, ctx: ctx, isOut: false) }
    // Integer-aware emitter: renders constants as int literals when the UOp is int-typed
    let gi = { self.emitLazyTyped($0, ctx: ctx, asInt: uop.scalarType == .int) }

    switch uop.op {
    case .add(let a, let b): return emitAssign(uop, "\(gi(a)) + \(gi(b))", ctx)
    case .mul(let a, let b): return emitAssign(uop, "\(gi(a)) * \(gi(b))", ctx)
    case .sub(let a, let b): return emitAssign(uop, "\(gi(a)) - \(gi(b))", ctx)
    case .div(let a, let b):
      if uop.scalarType == .int {
        // Integer division — no reciprocal optimization
        return emitAssign(uop, "\(gi(a)) / \(gi(b))", ctx)
      }
      // Strength-reduce division by constant to multiply by reciprocal
      switch b {
      case .constant(_, let val):
        return emitAssign(uop, "(\(g(a)) * \(1.0/val))", ctx)
      default:
        return emitAssign(uop, "\(g(a)) / \(g(b))", ctx)
      }
    case .mod(let a, let b):
      if uop.scalarType == .int {
        return emitAssign(uop, "\(gi(a)) % \(gi(b))", ctx)
      }
      // Fast modulo for constant denominator: a - floor(a / b) * b
      switch b {
      case .constant(_, let val):
        if val == 1.0 {
          return emitAssign(uop, "(\(g(a)) - metal::floor(\(g(a))))", ctx)
        } else {
          return emitAssign(uop, "(\(g(a)) - metal::floor(\(g(a)) / \(val)) * \(val))", ctx)
        }
      default:
        return emitAssign(uop, "metal::fmod(\(g(a)), \(g(b)))", ctx)
      }
    case .pow(let a, let b):
      // Specialize common exponents to avoid expensive pow
      switch b {
      case .constant(_, let val):
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
        if case .constant(_, let baseVal) = a {
          return emitAssign(uop, "metal::exp(\(g(b)) * metal::log(\(baseVal)))", ctx)
        }
        return emitAssign(uop, "metal::pow(\(g(a)), \(g(b)))", ctx)
      }
    case .min(let a, let b): return emitAssign(uop, "metal::min(\(g(a)), \(g(b)))", ctx)
    case .max(let a, let b): return emitAssign(uop, "metal::max(\(g(a)), \(g(b)))", ctx)

    case .abs(let a): return emitAssign(uop, "metal::abs(\(g(a)))", ctx)
    case .sign(let a): return emitAssign(uop, "metal::sign(\(g(a)))", ctx)
    case .floor(let a): return emitAssign(uop, "metal::floor(\(g(a)))", ctx)
    case .ceil(let a): return emitAssign(uop, "metal::ceil(\(g(a)))", ctx)
    case .round(let a): return emitAssign(uop, "metal::round\(g(a)))", ctx)
    case .noise(let cellId):
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
    case .memoryRead(let base, let offset):
      let cast = intCastPrefix(for: offset)
      return emitAssign(uop, "memory[\(base) + \(cast)\(g(offset))]", ctx)
    case .memoryWrite(let base, let offset, let value):
      let cast = intCastPrefix(for: offset)
      return "memory[\(base) + \(cast)\(g(offset))] = \(g(value));"
    case .memoryAccumulate(let base, let offset, let value):
      // Atomic add to memory cell - safe for concurrent accumulation from SIMD threads
      let cast = intCastPrefix(for: offset)
      return
        "atomic_fetch_add_explicit((device metal::atomic<float>*)&memory[\(base) + \(cast)\(g(offset))], \(g(value)), metal::memory_order_relaxed);"
    case .sin(let a): return emitAssign(uop, "metal::sin(\(g(a)))", ctx)
    case .cos(let a): return emitAssign(uop, "metal::cos(\(g(a)))", ctx)
    case .tan(let a): return emitAssign(uop, "metal::tan(\(g(a)))", ctx)
    case .tanh(let a): return emitAssign(uop, "metal::tanh(\(g(a)))", ctx)
    case .exp(let a): return emitAssign(uop, "metal::exp(\(g(a)))", ctx)
    case .log(let a): return emitAssign(uop, "metal::log(\(g(a)))", ctx)
    case .log10(let a): return emitAssign(uop, "metal::log10(\(g(a)))", ctx)
    case .sqrt(let a): return emitAssign(uop, "metal::sqrt(\(g(a)))", ctx)
    case .atan2(let y, let x): return emitAssign(uop, "metal::atan2(\(g(y)), \(g(x)))", ctx)

    case .gt(let a, let b): return emitAssign(uop, "\(g(a)) > \(g(b))", ctx)
    case .gte(let a, let b): return emitAssign(uop, "\(g(a)) >= \(g(b))", ctx)
    case .lte(let a, let b): return emitAssign(uop, "\(g(a)) <= \(g(b))", ctx)
    case .lt(let a, let b): return emitAssign(uop, "\(g(a)) < \(g(b))", ctx)
    case .eq(let a, let b): return emitAssign(uop, "\(g(a)) == \(g(b))", ctx)
    case .gswitch(let cond, let a, let b):
      // Cast both value args to float to avoid ambiguous select overload (int vs float)
      let bStr = isIntTyped(b) ? "(float)\(gi(b))" : g(b)
      let aStr = isIntTyped(a) ? "(float)\(gi(a))" : g(a)
      let expr = "metal::select(\(bStr), \(aStr), \(g(cond)) > 0.0)"
      return emitAssign(uop, expr, ctx)
    case .delay1(let cell, let a):
      // Scalar: read previous value, then write current value
      let assign = emitAssign(uop, "memory[\(cell)]", ctx)
      return "\(assign) memory[\(cell)] = \(g(a));"
    case .selector(let mode, let options):
      // Metal: if mode <= 0 return 0, if mode <= 1 return options[0], etc.
      var expr = "0.0f"  // Default value

      // Build from the end backwards to match the priority order
      for (i, option) in options.enumerated().reversed() {
        expr = "metal::select(\(expr), \(g(option)), \(g(mode)) <= \(Float(i + 1)))"
      }

      // Final check for mode <= 0
      expr = "metal::select(\(expr), 0.0f, \(g(mode)) <= 0.0f)"

      return emitAssign(uop, expr, ctx)

    case .load(let cell): return emitAssign(uop, "memory[\(cell)]", ctx)
    case .frameIndex:
      // Use id for SIMD blocks, i for scalar blocks
      let baseIdx = (currentFrameOrder == .parallel) ? "id" : "i"
      return emitAssign(uop, frameIndexOverride ?? baseIdx, ctx)
    case .loadTape(let val, let offset):
      let varId = ctx.getGlobalId(extractVarId(val))
      let boundedFetch =
        "(\(g(offset)) < 0 || \(g(offset)) >= frameCount) ? 0.0 : t[\(varId) * frameCount + (int)\(g(offset))]"
      return emitAssign(uop, boundedFetch, ctx)
    case .store(let cell, let val): return "memory[\(cell)] = \(g(val));"
    case .mutate(let a, let b):
      return "\(emitLazy(a, ctx: ctx, isOut: true)) = \(g(b));"
    case .beginIf(let cond): return "if (\(g(cond))) {"
    case .endIf: return "}"

    case .setThreadCountScale:
      return "/* setThreadCountScale - handled in scheduler */"

    case .setFrameIndex(let idx):
      frameIndexOverride = "_frameIndex"
      return "uint _frameIndex = (uint)(\(g(idx)));"

    case .beginLoop(let iters, let step):
      if step < 0 {
        return "for (int i = \(g(iters)) - 1; i >= 0; i += \(step)) {"
      } else {
        return "for (uint i = 0; i < \(g(iters)); i += \(step)) {"
      }
    case .beginReverseLoop(let iters):
      return "for (int i = \(g(iters)) - 1; i >= 0; i--) {"
    case .beginForLoop(let loopVar, let count):
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
    case .endLoop:
      // Skip device memory fence for threadgroup scratch kernels (threadGroupSize=1,
      // single thread per threadgroup — no cross-thread synchronization needed)
      if usesThreadgroupScratch {
        return "}"
      }
      return "} atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst);"

    case .threadIndex:
      // In Metal, threadIndex maps to 'id' (thread_position_in_grid)
      let idx =
        (currentFrameOrder == .parallel || currentThreadCountOverride != nil)
        ? "id"
        : "i"
      return emitAssign(uop, idx, ctx)

    case .identity(let a):
      return emitAssign(uop, "\(gi(a))", ctx)

    case .cast(let expr, let castType):
      let typeStr = castType == .int ? "int" : "float"
      return emitAssign(uop, "(\(typeStr))\(g(expr))", ctx)

    case .declareVar(let value):
      // Declares and initializes a variable: float t = value;
      return emitAssign(uop, g(value), ctx)

    case .beginRange(let start, let end):
      // GEMM kernels use 2D threadgroup dispatch — no linear id range guard needed
      if isGemmKernel { return "{" }

      let startExpr: String
      if case .constant(_, let val) = start {
        startExpr = "\(Int(val))"
      } else {
        startExpr = "\(g(start))"
      }

      let endExpr: String
      if case .constant(_, let val) = end {
        endExpr = "\(Int(val))"
      } else if case .variable(let id, _) = end, id == -1 {
        endExpr = "frameCount"  // Special case for frameCount
      } else {
        endExpr = "\(g(end))"
      }

      return "if (id >= \(startExpr) && id < (uint)(\(endExpr))) {"
    case .endRange: return "}"

    case .output(let channel, let val):
      // Store output value to a device buffer that can be read back
      let baseIdx = (currentFrameOrder == .parallel) ? "id" : "i"
      let idx =
        frameIndexOverride
        ?? (currentThreadCountScale == nil
          ? baseIdx
          : "(\(baseIdx) / \(currentThreadCountScale!))")
      return "outputs[\(channel) * frameCount + \(idx)] = \(g(val));"
    case .input(_):
      return ""
    case .loadGlobal(let id):
      // For Metal, loadGlobal is handled transparently through direct buffer access
      // The actual variable access happens in emitLazy
      return "/* loadGlobal(\(id)) - handled in variable access */"

    // Parallel range - for Metal, could be thread-parallel for static tensors
    // For now, render as a loop (future: check block.temporality and use thread-parallel for static)
    case .beginParallelRange(let count, _):
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

    // Hop-based execution: only run block when counter == 0
    case .beginHopCheck(let cond):
      return "if (\(g(cond)) == 0.0) {"
    case .endHopCheck:
      return "}"

    // GEMM threadgroup position
    case .threadgroupPositionX:
      return emitAssign(uop, "gid.x", ctx)
    case .threadgroupPositionY:
      return emitAssign(uop, "gid.y", ctx)
    case .threadgroupPositionZ:
      return emitAssign(uop, "gid.z", ctx)

    // GEMM simdgroup matrix operations
    case .simdgroupMatrixZero:
      let lhs = emitLazy(uop.value, ctx: ctx, isOut: true)
      return "metal::simdgroup_float8x8 \(lhs) = metal::simdgroup_float8x8(0);"
    case .simdgroupLoad(let cellId, let offset, let stride, let transpose):
      let lhs = emitLazy(uop.value, ctx: ctx, isOut: true)
      let cast = intCastPrefix(for: offset)
      let transposeArgs = transpose ? ", ulong2(0, 0), true" : ""
      return
        "metal::simdgroup_float8x8 \(lhs) = metal::simdgroup_float8x8(0); metal::simdgroup_load(\(lhs), &memory[\(cellId) + \(cast)\(g(offset))], \(stride)\(transposeArgs));"
    case .simdgroupStore(let src, let cellId, let offset, let stride):
      let cast = intCastPrefix(for: offset)
      return
        "metal::simdgroup_store(\(g(src)), &memory[\(cellId) + \(cast)\(g(offset))], \(stride));"
    case .simdgroupMultiplyAccumulate(let a, let b, let acc):
      return "metal::simdgroup_multiply_accumulate(\(g(acc)), \(g(a)), \(g(b)), \(g(acc)));"

    // Threadgroup shared memory (on-chip SRAM for FFT scratch)
    case .threadgroupArrayDecl(let scratchId, let size):
      return "threadgroup float scratch_\(scratchId)[\(size)];"
    case .threadgroupRead(let scratchId, let offset):
      let cast = intCastPrefix(for: offset)
      return emitAssign(uop, "scratch_\(scratchId)[\(cast)\(g(offset))]", ctx)
    case .threadgroupWrite(let scratchId, let offset, let value):
      let cast = intCastPrefix(for: offset)
      return "scratch_\(scratchId)[\(cast)\(g(offset))] = \(g(value));"

    default:
      return "/* \(uop.prettyDescription()) */"
    }
  }

  func emitLazy(_ lazy: Lazy, ctx: IRContext, isOut: Bool) -> String {
    switch lazy {
    case .constant(_, let val): return "\(val)"
    case .variable(let id, _):
      if id == -1 {  // Special case for frameCount
        return "frameCount"
      } else if parallelRangeVars.contains(id) {
        // This is a parallel range loop variable - use _pr prefix
        return "_pr\(id)"
      } else if ctx.globals.contains(id) {
        let tapeSlot = ctx.getGlobalId(id)
        let baseIdx = (currentFrameOrder == .parallel) ? "id" : "i"
        let idx =
          staticGlobalVars.contains(id)
          ? "0"
          : (frameIndexOverride
            ?? (currentThreadCountScale == nil
              ? baseIdx : "(\(baseIdx) / \(currentThreadCountScale!))"))
        return
          "t[\(tapeSlot)*frameCount + \(idx)]"
      } else {
        return "t\(id)"
      }
    case .global(let id):
      // Global variables are accessed through global buffers
      let tapeSlot = ctx.getGlobalId(id)
      let baseIdx = (currentFrameOrder == .parallel) ? "id" : "i"
      let idx =
        staticGlobalVars.contains(id)
        ? "0"
        : (frameIndexOverride
          ?? (currentThreadCountScale == nil
            ? baseIdx : "(\(baseIdx) / \(currentThreadCountScale!))"))
      return
        "t[\(tapeSlot)*frameCount + \(idx)]"
    default: return "/* unknown lazy */"
    }
  }

  func emitAssign(_ uop: UOp, _ expr: String, _ ctx: IRContext) -> String {
    let lhs = emitLazy(uop.value, ctx: ctx, isOut: true)
    let isGlobal = ctx.globals.contains(extractVarId(uop.value))

    if isGlobal {
      return "\(lhs) = \(expr);"
    }
    let typeStr = uop.scalarType == .int ? "int" : "float"
    return "\(typeStr) \(lhs) = \(expr);"
  }
}

struct RequiredBuffers {
  let hasOutputOps: Bool
  let needsReducedGradsSum: Bool
}

func analyzeRequiredBuffers(scheduleItem: ScheduleItem) -> RequiredBuffers {
  // Check if this kernel has output operations
  let hasOutputOps = scheduleItem.ops.contains { uop in
    if case .output = uop.op { return true }
    return false
  }

  return RequiredBuffers(
    hasOutputOps: hasOutputOps,
    needsReducedGradsSum: false
  )
}

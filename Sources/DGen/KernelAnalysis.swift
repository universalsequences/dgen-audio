// KernelAnalysis - Static analysis of compiled kernel IR
//
// Analyzes ScheduleItems (post-prepareSchedule Metal IR) to compute
// work (total FLOPs), span (sequential depth per thread), memory usage,
// and kernel count — without executing anything.

import Foundation

// MARK: - Per-Kernel Stats

public struct KernelStats {
  public let name: String
  public let frameOrder: FrameOrder
  public let dispatchMode: DispatchMode
  public let temporality: Temporality
  public let work: Int
  public let span: Int
  public let arithmeticOps: Int
  public let transcendentalOps: Int
  public let memoryOps: Int
}

// MARK: - Aggregate Analysis

public struct KernelAnalysis {
  public let kernelCount: Int
  public let work: Int
  public let span: Int
  public let memoryBytes: Int
  public let memorySlots: Int
  public let kernels: [KernelStats]
  public let includesBackward: Bool
}

// MARK: - Op Classification

private enum OpClass {
  case arithmetic
  case transcendental
  case memory
  case control
}

private func classify(_ op: Op) -> OpClass {
  switch op {
  // Transcendental ops
  case .sin, .cos, .tan, .tanh, .exp, .log, .log10, .sqrt, .pow, .atan2:
    return .transcendental

  // Memory ops
  case .load, .store, .memoryRead, .memoryWrite, .memoryAccumulate,
       .delay1, .loadGlobal, .loadTape:
    return .memory

  // Arithmetic ops
  case .add, .sub, .mul, .div, .mod, .abs, .sign, .floor, .ceil, .round,
       .min, .max, .gt, .gte, .lt, .lte, .eq, .gswitch, .noise, .latch,
       .and, .or, .xor, .cast, .identity, .declareVar, .mutate, .selector:
    return .arithmetic

  // MSE is sub + square + add = 3 arithmetic ops, but we count it as 1 here
  // since it's a single UOp. The plan says 3 FLOPs but treating it as
  // arithmetic keeps the counting simple per-UOp.
  case .mse:
    return .arithmetic

  // Control flow / markers — 0 FLOPs
  case .beginIf, .endIf, .beginLoop, .endLoop, .beginRange, .endRange,
       .beginForLoop, .beginParallelRange, .endParallelRange,
       .setThreadCountScale, .setFrameIndex, .frameCount, .frameIndex,
       .threadIndex, .output, .input, .defineGlobal, .defineConstant,
       .reshape, .transpose, .shrink, .pad, .expandView, .repeatView,
       .broadcastAccess, .sumAxisMarker, .maxAxisMarker, .meanAxisMarker,
       .expandAxisMarker, .beginHopCheck, .endHopCheck,
       .beginReverseLoop:
    return .control
  }
}

private func flopCount(_ op: Op) -> Int {
  switch op {
  case .mse: return 3
  default: return 1
  }
}

// MARK: - Schedule Item Analysis

public func analyzeScheduleItems(
  _ scheduleItems: [ScheduleItem],
  frameCount: Int,
  totalMemorySlots: Int,
  includesBackward: Bool
) -> KernelAnalysis {
  var kernelStats: [KernelStats] = []

  for (index, item) in scheduleItems.enumerated() {
    let stats = analyzeOneKernel(
      item, name: "kernel_\(index)", frameCount: frameCount
    )
    kernelStats.append(stats)
  }

  let totalWork = kernelStats.reduce(0) { $0 + $1.work }
  let totalSpan = kernelStats.reduce(0) { $0 + $1.span }

  return KernelAnalysis(
    kernelCount: kernelStats.count,
    work: totalWork,
    span: totalSpan,
    memoryBytes: totalMemorySlots * 4,
    memorySlots: totalMemorySlots,
    kernels: kernelStats,
    includesBackward: includesBackward
  )
}

// MARK: - Single Kernel Analysis

/// Entry in the loop stack tracking both iteration count and whether
/// the loop is executed in parallel across GPU threads.
private struct LoopEntry {
  let iterations: Int
  let isParallel: Bool
}

private func analyzeOneKernel(
  _ item: ScheduleItem, name: String, frameCount: Int
) -> KernelStats {
  var threadCount = item.dispatchMode.threadCount(frameCount: frameCount)

  // Static blocks dispatched as staticThreads(1) may contain beginParallelRange(N)
  // that gets split into separate kernels dispatching N threads by
  // splitStaticParallelRanges (which runs after prepareSchedule in
  // lowerUOpBlocks). We model this by treating the parallelRange body as
  // thread-parallel: N multiplies work but NOT span.
  let isStaticParallel: Bool
  if case .staticThreads(1) = item.dispatchMode {
    isStaticParallel = item.ops.contains {
      if case .beginParallelRange = $0.op { return true }
      return false
    }
  } else {
    isStaticParallel = false
  }

  // Walk UOps with a loop stack
  var loopStack: [LoopEntry] = []
  var arithmeticOps = 0
  var transcendentalOps = 0
  var memoryOps = 0
  var span = 0

  for uop in item.ops {
    switch uop.op {
    case .beginLoop(_, let step):
      let iters = resolveLoopCount(uop.op, item: item, frameCount: frameCount)
      loopStack.append(LoopEntry(iterations: iters / max(step, 1), isParallel: false))

    case .beginForLoop(_, let countLazy):
      let count = resolveLazy(countLazy, frameCount: frameCount)
      loopStack.append(LoopEntry(iterations: count, isParallel: false))

    case .beginParallelRange(let count, _):
      if isStaticParallel {
        // Thread-parallel: body runs once per thread, N threads dispatched.
        // Multiply work by N (via threadCount), span stays the same.
        loopStack.append(LoopEntry(iterations: count, isParallel: true))
        threadCount = max(threadCount, count)
      } else {
        // Sequential loop in non-static or non-parallel blocks
        loopStack.append(LoopEntry(iterations: count, isParallel: false))
      }

    case .beginReverseLoop(let countLazy):
      let count = resolveLazy(countLazy, frameCount: frameCount)
      loopStack.append(LoopEntry(iterations: count, isParallel: false))

    case .endLoop, .endParallelRange:
      if !loopStack.isEmpty { loopStack.removeLast() }

    case .beginRange, .endRange:
      // Thread dispatch — not a loop for span. Work uses threadCount separately.
      break

    default:
      let cls = classify(uop.op)
      guard cls != .control else { continue }

      // Sequential multiplier: product of all NON-parallel loop iterations
      let seqMultiplier = loopStack.filter { !$0.isParallel }.reduce(1) { $0 * $1.iterations }
      let flops = flopCount(uop.op)

      switch cls {
      case .arithmetic:
        arithmeticOps += seqMultiplier * flops
      case .transcendental:
        transcendentalOps += seqMultiplier * flops
      case .memory:
        memoryOps += seqMultiplier
      case .control:
        break
      }

      span += seqMultiplier * flops
    }
  }

  let work = span * threadCount

  return KernelStats(
    name: name,
    frameOrder: item.frameOrder,
    dispatchMode: item.dispatchMode,
    temporality: item.temporality,
    work: work,
    span: span,
    arithmeticOps: arithmeticOps,
    transcendentalOps: transcendentalOps,
    memoryOps: memoryOps
  )
}

// MARK: - Helpers

/// Resolve iteration count from a beginLoop op.
private func resolveLoopCount(
  _ op: Op, item: ScheduleItem, frameCount: Int
) -> Int {
  if case .beginLoop(let countLazy, _) = op {
    return resolveLazy(countLazy, frameCount: frameCount)
  }
  return 1
}

/// Best-effort resolution of a Lazy value to an integer.
/// For variables that represent frameCount (variable -1), returns frameCount.
/// For constants, returns the value. Otherwise returns 1 as fallback.
private func resolveLazy(_ lazy: Lazy, frameCount: Int) -> Int {
  switch lazy {
  case .constant(_, let val):
    return max(Int(val), 1)
  case .variable(let varId, _):
    // variable(-1) is the canonical frameCount variable in MetalRenderer
    if varId == -1 { return frameCount }
    return frameCount  // conservative: assume it's frame-count-derived
  case .global:
    return frameCount
  case .gradient, .empty:
    return 1
  }
}

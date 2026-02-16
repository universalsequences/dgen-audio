import Foundation

public struct BlockUOps {
  public var ops: [UOp]
  public let frameOrder: FrameOrder
  public let vectorWidth: Int
  public let temporality: Temporality
  public var dispatchMode: DispatchMode

  public init(
    ops: [UOp],
    frameOrder: FrameOrder,
    vectorWidth: Int = 1,
    temporality: Temporality = .static_,
    dispatchMode: DispatchMode = .singleThreaded
  ) {
    self.ops = ops
    self.frameOrder = frameOrder
    self.vectorWidth = vectorWidth
    self.temporality = temporality
    self.dispatchMode = dispatchMode
  }
}

/// How a kernel distributes work across GPU threads.
public enum DispatchMode: Equatable {
  /// 1 thread, renderer wraps body in frame loop (sequential blocks)
  case singleThreaded

  /// frameCount threads, one frame per thread (parallel blocks)
  case perFrame

  /// frameCount × N threads (parallel blocks with tensor elements)
  case perFrameScaled(Int)

  /// N fixed threads, each loops over all frames (tensor element parallel)
  case fixedWithFrameLoop(Int)

  /// N threads, no frame loop (static blocks)
  case staticThreads(Int)

  /// 1 thread, no frame loop — block has its own loops (BPTT)
  case selfManaged

  /// GEMM: 2D threadgroup grid, 32 threads per group (one SIMD group).
  /// tilesM × tilesN threadgroups, each handling one 8×8 output tile.
  case gemm(tilesM: Int, tilesN: Int)
}

extension DispatchMode {
  /// Total GPU threads to dispatch
  func threadCount(frameCount: Int) -> Int {
    switch self {
    case .singleThreaded, .selfManaged: return 1
    case .perFrame: return frameCount
    case .perFrameScaled(let n): return frameCount * max(1, n)
    case .fixedWithFrameLoop(let n), .staticThreads(let n): return max(1, n)
    case .gemm(let tilesM, let tilesN): return tilesM * tilesN * 32
    }
  }

  /// Whether the renderer should wrap the body in a frame loop
  var hasRendererFrameLoop: Bool {
    switch self {
    case .singleThreaded, .fixedWithFrameLoop: return true
    case .perFrame, .perFrameScaled, .staticThreads, .selfManaged, .gemm: return false
    }
  }

  /// Thread group size hint (1 for single-thread kernels, nil for runtime-determined)
  var threadGroupSize: Int? {
    switch self {
    case .singleThreaded, .selfManaged: return 1
    case .gemm: return 32
    default: return nil
    }
  }

  /// The thread count scale factor, if any (for perFrameScaled or staticThreads > 1)
  var threadCountScale: Int? {
    switch self {
    case .perFrameScaled(let n): return n
    case .staticThreads(let n) where n > 1: return n
    default: return nil
    }
  }

  /// Fixed thread count for modes that dispatch a constant number of threads with a frame loop.
  var fixedThreadCount: Int? {
    if case .fixedWithFrameLoop(let n) = self { return n }
    return nil
  }
}

public enum Device {
  case C
  case Metal
}

public struct CompiledKernel {
  public let name: String
  public let source: String
  public let frameOrder: FrameOrder
  public let temporality: Temporality
  public let buffers: [String]  // names of inputs/outputs
  public let dispatchMode: DispatchMode
  public let needsReducedGradsSum: Bool
  public let memorySize: Int  // Required memory allocation size in floats
}

public class ScheduleItem {
  public var ops: [UOp] = []
  public let frameOrder: FrameOrder
  public let vectorWidth: Int
  public var temporality: Temporality = .frameBased
  public var dispatchMode: DispatchMode = .perFrame

  init(frameOrder: FrameOrder, vectorWidth: Int = 1, temporality: Temporality = .frameBased) {
    self.frameOrder = frameOrder
    self.vectorWidth = vectorWidth
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
  func emitLazy(_ lazy: Lazy, ctx: IRContext, isOut: Bool) -> String
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

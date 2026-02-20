import Foundation

/// Result of the compilation pipeline containing all intermediate steps
public struct CompilationResult {
  /// Original graph
  public let graph: Graph

  /// Topologically sorted node IDs
  public let sortedNodes: [NodeID]

  /// Set of nodes that must execute sequentially (not SIMD-parallelizable)
  public let sequentialNodes: Set<NodeID>

  /// Blocks before sorting by dependencies
  public let blocks: [Block]

  /// Block indices sorted by execution order
  public let sortedBlockIndices: [Int]

  /// Blocks sorted by execution order
  public let sortedBlocks: [Block]

  /// IR context with values and constants
  public let context: IRContext

  /// UOp blocks generated from the sorted blocks
  public let uopBlocks: [BlockUOps]

  /// Final compiled kernels
  public let kernels: [CompiledKernel]

  /// Backend used for compilation
  public let backend: Backend

  /// Total memory slots needed after vector remapping
  public let totalMemorySlots: Int

  public let cellAllocations: CellAllocations
  /// Logical cell ID containing runtime voice index for MC execution.
  ///
  /// This is the graph-level ID. The physical slot used by generated code is derived through
  /// `cellAllocations.cellMappings` during compilation.
  public let voiceCellId: Int?
}

/// Backend type for compilation
public enum Backend {
  case c
  case metal
}

/// Main compilation pipeline that converts a Graph into compiled kernels
public struct CompilationPipeline {

  /// Options for the compilation pipeline
  public struct Options {
    public let frameCount: Int
    public let debug: Bool
    public let printBlockStructure: Bool
    public let forceScalar: Bool
    /// Number of MC voices that will run the same compiled kernel.
    public let voiceCount: Int
    /// Optional logical graph cell that stores runtime voice index.
    ///
    /// If omitted while `voiceCount > 1`, compilation allocates a dedicated cell.
    public let voiceCellId: Int?
    public let enableBufferReuse: Bool

    public init(
      frameCount: Int = 128,
      debug: Bool = false,
      printBlockStructure: Bool = false,
      forceScalar: Bool = false,
      voiceCount: Int = 1,
      voiceCellId: Int? = nil,
      enableBufferReuse: Bool = false
    ) {
      self.frameCount = frameCount
      self.debug = debug
      self.printBlockStructure = printBlockStructure
      self.forceScalar = forceScalar
      self.voiceCount = voiceCount
      self.voiceCellId = voiceCellId
      self.enableBufferReuse = enableBufferReuse
    }
  }

  /// Aggregates per-pass timing so compile flow can stay linear and readable.
  private struct PipelineTimings {
    let startedAt: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()
    var entries: [(label: String, elapsedMs: Double)] = []

    /// Runs a pass, records elapsed milliseconds, and returns the pass result.
    mutating func measure<T>(_ label: String, _ pass: () throws -> T) rethrows -> T {
      let passStart = CFAbsoluteTimeGetCurrent()
      let result = try pass()
      let elapsedMs = (CFAbsoluteTimeGetCurrent() - passStart) * 1000
      entries.append((label: label, elapsedMs: elapsedMs))
      return result
    }

    /// Prints the heaviest pass timings when debug mode is enabled.
    func printSummaryIfNeeded(enabled: Bool, nodeCount: Int) {
      guard enabled else { return }
      let totalMs = (CFAbsoluteTimeGetCurrent() - startedAt) * 1000
      print(
        "⏱️ [DGen Pipeline] Total: \(String(format: "%.1f", totalMs))ms | nodes: \(nodeCount)"
      )
      let sorted = entries.sorted { $0.elapsedMs > $1.elapsedMs }
      for (label, ms) in sorted.prefix(10) {
        let pct = totalMs > 0 ? (ms / totalMs) * 100 : 0
        print(
          "   \(String(format: "%6.1f", ms))ms (\(String(format: "%4.1f", pct))%) - \(label)")
      }
    }
  }

  /// Output of graph analysis passes that feed block partitioning.
  private struct GraphPreparationResult {
    let feedbackClusters: [[NodeID]]
    let scalarNodeSet: Set<NodeID>
    let sortedNodes: [NodeID]
  }

  /// Temporality analysis artifacts used by later block/layout passes.
  private struct BlockTemporalityResult {
    let frameBasedNodes: Set<NodeID>
    let hopBasedNodes: [NodeID: (Int, NodeID)]
  }

  /// Compile a graph with the specified backend and options
  public static func compile(
    graph: Graph,
    backend: Backend,
    options: Options = Options(),
    name: String = "kernel"
  ) throws -> CompilationResult {
    validateFrameCount(options, graph: graph)
    var timings = PipelineTimings()

    let prep = try runGraphPreparationPasses(
      graph: graph, backend: backend, options: options, timings: &timings)
    let finalScalarSet = timings.measure("seqScalarPropagate") {
      GraphPrepPasses.propagateSeqScalarInputs(
        graph: graph, initialScalarSet: prep.scalarNodeSet)
    }

    let context = IRContext(g: graph)
    var finalBlocks = buildInitialBlocks(
      graph: graph, sortedNodes: prep.sortedNodes, scalarNodeSet: finalScalarSet, context: context,
      timings: &timings)

    let temporalityResult = inferAndAssignTemporality(
      graph: graph,
      sortedNodes: prep.sortedNodes,
      blocks: &finalBlocks,
      context: context,
      timings: &timings
    )
    applyBackendBlockSafetySplitsIfNeeded(
      graph: graph,
      backend: backend,
      blocks: &finalBlocks
    )
    materializeTensorMemory(
      graph: graph,
      blocks: finalBlocks,
      temporality: temporalityResult,
      feedbackClusters: prep.feedbackClusters,
      context: context,
      backend: backend,
      frameCount: options.frameCount,
      timings: &timings
    )

    let finalBlockIndices = Array(0..<finalBlocks.count)
    if options.printBlockStructure {
      printBlockStructure(blocks: finalBlocks, sortedIndices: finalBlockIndices)
    }

    var uopBlocks = try emitBlocksToUOpBlocks(
      graph: graph, blocks: finalBlocks, context: context, backend: backend, options: options,
      timings: &timings)
    uopBlocks.removeAll { $0.ops.isEmpty }

    if options.debug {
      printUOpBlocks(uopBlocks, blocks: finalBlocks)
    }

    let voiceState = VoiceStateCompilation.buildPlan(graph: graph, options: options)
    var cellAllocations = timings.measure("remapMemorySlots") {
      remapVectorMemorySlots(
        &uopBlocks,
        cellSizes: graph.cellAllocationSizes,
        voiceCellId: voiceState.voiceCellIdForMemoryRemap,
        graph: graph,
        enableBufferReuse: options.enableBufferReuse)
    }
    // Collect precomputed tensor data (e.g. FFT twiddle factors, Hann window) for injection
    cellAllocations.tensorInitData = collectTensorInitData(
      graph: graph, cellAllocations: cellAllocations)

    let renderer = createRenderer(for: backend)
    VoiceStateCompilation.configureRenderer(
      renderer, plan: voiceState, cellAllocations: cellAllocations)

    let kernels = try timings.measure("lowerUOpBlocks") {
      try lowerUOpBlocks(
        &uopBlocks,
        renderer: renderer,
        ctx: context,
        frameCount: options.frameCount,
        graph: graph,
        totalMemorySlots: cellAllocations.totalMemorySlots,
        name: name
      )
    }

    timings.printSummaryIfNeeded(enabled: options.debug, nodeCount: graph.nodes.count)

    return CompilationResult(
      graph: graph,
      sortedNodes: prep.sortedNodes,
      sequentialNodes: finalScalarSet,
      blocks: finalBlocks,
      sortedBlockIndices: finalBlockIndices,
      sortedBlocks: finalBlocks,
      context: context,
      uopBlocks: uopBlocks,
      kernels: kernels,
      backend: backend,
      totalMemorySlots: cellAllocations.totalMemorySlots,
      cellAllocations: cellAllocations,
      voiceCellId: voiceState.logicalVoiceCellId,
    )
  }

  /// Enforces frame-count bounds before any mutating compile pass runs.
  private static func validateFrameCount(_ options: Options, graph: Graph) {
    precondition(
      options.frameCount <= graph.maxFrameCount,
      "frameCount (\(options.frameCount)) exceeds graph.maxFrameCount (\(graph.maxFrameCount)). "
        + "Set graph.maxFrameCount to at least \(options.frameCount) before compilation."
    )
  }

  /// Runs graph-level analysis passes that prepare sorting and scalar execution decisions.
  private static func runGraphPreparationPasses(
    graph: Graph, backend: Backend, options: Options, timings: inout PipelineTimings
  ) throws -> GraphPreparationResult {
    let feedbackClusters = timings.measure("findFeedbackLoops") {
      findFeedbackLoops(graph)
    }

    timings.measure("combineHistoryOps") {
      GraphPrepPasses.combineHistoryOpsNotInFeedback(
        graph, feedbackClusters: feedbackClusters, options: options)
    }

    timings.measure("foldConstants") {
      GraphPrepPasses.foldConstants(graph, options: options)
    }

    let scalarNodeSet = timings.measure("findSequentialNodes") {
      options.forceScalar
        ? Set(graph.nodes.keys)
        : findSequentialNodes(graph, feedbackClusters: feedbackClusters, backend: backend)
    }

    if backend == .metal {
      timings.measure("gemmPass") {
        GraphPrepPasses.gemmPass(graph: graph)
      }
    }

    let sortedNodes = timings.measure("topologicalSort") {
      topologicalSort(
        graph, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet,
        debug: false)
    }

    try timings.measure("shapeInference") {
      try ShapeInferencePass.inferNodeShapes(graph: graph, sortedNodes: sortedNodes)
    }

    timings.measure("allocateTensorOutputs") {
      TensorOutputBindingPass.bindTensorOutputsAndReserveLazyCells(
        graph: graph, sortedNodes: sortedNodes)
    }

    return GraphPreparationResult(
      feedbackClusters: feedbackClusters,
      scalarNodeSet: scalarNodeSet,
      sortedNodes: sortedNodes)
  }

  /// Builds executable blocks before temporality-aware rewrites.
  private static func buildInitialBlocks(
    graph: Graph, sortedNodes: [NodeID], scalarNodeSet: Set<NodeID>, context: IRContext,
    timings: inout PipelineTimings
  ) -> [Block] {
    let blocks = timings.measure("determineBlocks") {
      partitionIntoBlocks(
        sorted: sortedNodes,
        scalar: scalarNodeSet,
        g: graph,
        debug: false
      )
    }

    let fusedBlocks = timings.measure("fuseBlocks1") {
      fuseBlocks(blocks)
    }

    let separatedBlocks = timings.measure("tensorBlocks") {
      determineTensorBlocks(fusedBlocks, graph, context)
    }

    var finalBlocks = separatedBlocks.compactMap { $0 }
    if graphContainsSpectralLossOps(graph) {
      finalBlocks = isolateSpectralPasses(finalBlocks, graph)
    }
    return finalBlocks
  }

  /// Checks whether the graph needs spectral-pass isolation to prevent pass overlap.
  private static func graphContainsSpectralLossOps(_ graph: Graph) -> Bool {
    graph.nodes.values.contains { node in
      switch node.op {
      case .spectralLossFFT, .spectralLossFFTGradInline, .spectralLossFFTGradSpec,
        .spectralLossFFTGradIFFT, .spectralLossFFTGradRead, .spectralLossFFTGradRead2,
        .spectralLossFFTBatched, .spectralLossFFTBatchedReduce,
        .spectralLossFFTBatchedGradSpec, .spectralLossFFTBatchedGradIFFT,
        .spectralLossFFTBatchedGradRead, .spectralLossFFTBatchedGradRead2:
        return true
      default:
        return false
      }
    }
  }

  /// Infers temporality and assigns block temporality metadata.
  private static func inferAndAssignTemporality(
    graph: Graph, sortedNodes: [NodeID], blocks: inout [Block], context: IRContext,
    timings: inout PipelineTimings
  ) -> BlockTemporalityResult {
    let temporalityResult = timings.measure("inferTemporality") {
      TemporalityPass.inferTemporality(graph: graph, sortedNodes: sortedNodes)
    }

    timings.measure("assignTemporality") {
      TemporalityPass.assignBlockTemporality(
        blocks: &blocks,
        frameBasedNodes: temporalityResult.frameBasedNodes,
        hopBasedNodes: temporalityResult.hopBasedNodes
      )
      context.hopBasedNodes = temporalityResult.hopBasedNodes
    }
    return BlockTemporalityResult(
      frameBasedNodes: temporalityResult.frameBasedNodes,
      hopBasedNodes: temporalityResult.hopBasedNodes
    )
  }

  /// Applies backend-specific block splits that require hard kernel boundaries.
  private static func applyBackendBlockSafetySplitsIfNeeded(
    graph: Graph, backend: Backend, blocks: inout [Block]
  ) {
    blocks = BackendBlockSafetySplitPass.applyIfNeeded(
      graph: graph, blocks: blocks, backend: backend)
  }

  /// Materializes lazy tensor cells after temporality and backend block layout are finalized.
  private static func materializeTensorMemory(
    graph: Graph,
    blocks: [Block],
    temporality: BlockTemporalityResult,
    feedbackClusters: [[NodeID]],
    context: IRContext,
    backend: Backend,
    frameCount: Int,
    timings: inout PipelineTimings
  ) {
    timings.measure("allocateTensorMemory") {
      let feedbackClusterNodes = Set(feedbackClusters.flatMap { $0 })
      TensorMemoryMaterializationPass.allocateTensorMemory(
        graph: graph,
        blocks: blocks,
        frameBasedNodes: temporality.frameBasedNodes,
        hopBasedNodes: temporality.hopBasedNodes,
        feedbackClusterNodes: feedbackClusterNodes,
        backend: backend,
        frameCount: frameCount
      )
      for cellId in graph.frameAwareCells.keys {
        context.frameAwareTensorCells.insert(cellId)
      }
    }
  }

  /// Emits UOp blocks and applies backend-specific post-processing on emitted ops.
  private static func emitBlocksToUOpBlocks(
    graph: Graph, blocks: [Block], context: IRContext, backend: Backend, options: Options,
    timings: inout PipelineTimings
  ) throws -> [BlockUOps] {
    var uopBlocks = [BlockUOps]()

    try timings.measure("emitBlockUOps") {
      for block in blocks {
        let emission = try emitBlockUOps(
          ctx: context,
          block: block,
          blocks: blocks,
          g: graph,
          backend: backend,
          debug: options.debug
        )

        uopBlocks.append(
          UOpBlockFinalization.finalize(
            emittedOps: emission.uops,
            block: block,
            graph: graph,
            backend: backend,
            bodyFrameOrder: emission.frameOrder,
            bodyVectorWidth: emission.vectorWidth,
            hasOwnFrameLoop: emission.hasOwnFrameLoop
          )
        )
      }
    }

    return uopBlocks
  }

  /// Create a renderer for the specified backend
  private static func createRenderer(for backend: Backend) -> Renderer {
    switch backend {
    case .c:
      return CRenderer()
    case .metal:
      return MetalRenderer()
    }
  }

  /// Print the block structure in a human-readable format
  private static func printBlockStructure(blocks: [Block], sortedIndices: [Int]) {
    print("Block structure:")
    for (i, blockIdx) in sortedIndices.enumerated() {
      let block = blocks[blockIdx]
      print("  Block \(i) (orig \(blockIdx), \(block.frameOrder)): \(block.nodes)")
    }
  }
}

// MARK: - Convenience Extensions

extension CompilationResult {
  /// Get the generated source code for all kernels
  public var source: String {
    kernels.map { $0.source }.joined(separator: "\n\n")
  }
}

// MARK: - Debug Helpers

private func printUOpBlocks(_ uopBlocks: [BlockUOps], blocks: [Block]) {
  for (i, uopBlock) in uopBlocks.enumerated() {
    let block = i < blocks.count ? blocks[i] : nil
    let shape = String(describing: block?.shape)
    let tensorIndex = String(describing: block?.tensorIndex)
    let nodes = block?.nodes ?? []
    print(
      "block #\(i+1) frameOrder=\(uopBlock.frameOrder) vectorWidth=\(uopBlock.vectorWidth) dispatchMode=\(uopBlock.dispatchMode) shape=\(shape) tensorIndex=\(tensorIndex) nodes=\(nodes)"
    )
    var indentLevel = 0
    for uop in uopBlock.ops {
      switch uop.op {
      case .beginIf, .beginForLoop, .beginParallelRange, .beginLoop, .beginRange:
        print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
        indentLevel += 1
      case .endIf, .endLoop, .endParallelRange, .endRange:
        indentLevel = max(0, indentLevel - 1)
        print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
      default:
        print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
      }
    }
  }
}

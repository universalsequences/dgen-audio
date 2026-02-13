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
    public let voiceCount: Int
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
      propagateSeqScalarInputs(graph: graph, initialScalarSet: prep.scalarNodeSet)
    }

    let context = IRContext(g: graph)
    var finalBlocks = buildInitialBlocks(
      graph: graph, sortedNodes: prep.sortedNodes, scalarNodeSet: finalScalarSet, context: context,
      timings: &timings)

    assignTemporalityAndAllocateTensorMemory(
      graph: graph,
      sortedNodes: prep.sortedNodes,
      feedbackClusters: prep.feedbackClusters,
      blocks: &finalBlocks,
      context: context,
      backend: backend,
      frameCount: options.frameCount,
      timings: &timings)

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

    let voiceCellState = ensureVoiceCell(graph: graph, options: options)
    let cellAllocations = timings.measure("remapMemorySlots") {
      remapVectorMemorySlots(
        &uopBlocks,
        cellSizes: graph.cellAllocationSizes,
        voiceCellId: voiceCellState.generated ? voiceCellState.id : nil,
        graph: graph,
        enableBufferReuse: options.enableBufferReuse)
    }

    let renderer = createRenderer(for: backend, options: options)
    configureRendererVoiceState(
      renderer: renderer,
      voiceCount: options.voiceCount,
      voiceCellId: voiceCellState.id,
      cellAllocations: cellAllocations)

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
      voiceCellId: voiceCellState.id,
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
      combineHistoryOpsNotInFeedback(
        graph, feedbackClusters: feedbackClusters, options: options)
    }

    timings.measure("foldConstants") {
      foldConstants(graph, options: options)
    }

    let scalarNodeSet = timings.measure("findSequentialNodes") {
      options.forceScalar
        ? Set(graph.nodes.keys)
        : findSequentialNodes(graph, feedbackClusters: feedbackClusters, backend: backend)
    }

    let sortedNodes = timings.measure("topologicalSort") {
      topologicalSort(
        graph, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet,
        debug: false)
    }

    try timings.measure("inferShapes") {
      try inferShapes(graph: graph, sortedNodes: sortedNodes)
    }

    timings.measure("allocateTensorOutputs") {
      allocateTensorOutputs(graph: graph, sortedNodes: sortedNodes)
    }

    return GraphPreparationResult(
      feedbackClusters: feedbackClusters,
      scalarNodeSet: scalarNodeSet,
      sortedNodes: sortedNodes)
  }

  /// Propagates scalar requirements through `seq` inputs while preserving SIMD-safe atomics.
  private static func propagateSeqScalarInputs(
    graph: Graph, initialScalarSet: Set<NodeID>
  ) -> Set<NodeID> {
    let simdSafeNodes = findSIMDSafeAtomicNodes(graph: graph)
    var scalarSet = initialScalarSet

    for node in graph.nodes.values {
      guard case .seq = node.op else { continue }
      let hasScalarInput = node.inputs.contains { scalarSet.contains($0) }
      guard hasScalarInput else { continue }
      for inputId in node.inputs where !simdSafeNodes.contains(inputId) {
        scalarSet.insert(inputId)
      }
    }

    return scalarSet
  }

  /// Finds nodes that intentionally stay SIMD-safe even when traversed by scalar propagation.
  private static func findSIMDSafeAtomicNodes(graph: Graph) -> Set<NodeID> {
    var simdSafe = Set<NodeID>()
    for (nodeId, node) in graph.nodes {
      switch node.op {
      case .memoryAccumulate(_), .tensorAccumulate(_):
        simdSafe.insert(nodeId)
      default:
        break
      }
    }
    return simdSafe
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
        .spectralLossFFTGradIFFT, .spectralLossFFTGradRead, .spectralLossFFTGradRead2:
        return true
      default:
        return false
      }
    }
  }

  /// Applies temporality metadata, backend-specific block splitting, and tensor memory allocation.
  private static func assignTemporalityAndAllocateTensorMemory(
    graph: Graph, sortedNodes: [NodeID], feedbackClusters: [[NodeID]], blocks: inout [Block],
    context: IRContext, backend: Backend, frameCount: Int, timings: inout PipelineTimings
  ) {
    let temporalityResult = timings.measure("inferTemporality") {
      inferTemporality(graph: graph, sortedNodes: sortedNodes)
    }

    timings.measure("assignTemporality") {
      assignBlockTemporality(
        blocks: &blocks,
        frameBasedNodes: temporalityResult.frameBasedNodes,
        hopBasedNodes: temporalityResult.hopBasedNodes
      )
      context.hopBasedNodes = temporalityResult.hopBasedNodes
    }

    if backend == .metal {
      blocks = splitReduceBlocks(g: graph, blocks: blocks)
      blocks = splitMemoryBlocks(g: graph, blocks: blocks)
    }

    timings.measure("allocateTensorMemory") {
      let feedbackClusterNodes = Set(feedbackClusters.flatMap { $0 })
      allocateTensorMemory(
        graph: graph,
        blocks: blocks,
        frameBasedNodes: temporalityResult.frameBasedNodes,
        hopBasedNodes: temporalityResult.hopBasedNodes,
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
        context.lastBlockHasOwnFrameLoop = false
        let (ops, bodyEffectiveKind) = try emitBlockUOps(
          ctx: context,
          block: block,
          blocks: blocks,
          g: graph,
          backend: backend,
          debug: options.debug
        )

        let hasOwnFrameLoop = context.lastBlockHasOwnFrameLoop
        let (threadCountScale, strippedOps) = extractThreadCountScale(from: ops)
        var finalOps = strippedOps
        let effectiveKind = resolveEffectiveKind(
          backend: backend,
          blockKind: block.kind,
          bodyEffectiveKind: bodyEffectiveKind,
          threadCountScale: threadCountScale,
          ops: &finalOps)

        if backend == .c {
          upgradeElementLoopsToSIMD(&finalOps)
        }

        let parallelPolicy = inferParallelPolicy(
          kind: effectiveKind, temporality: block.temporality, ops: finalOps)
        let forceNewKernel = threadCountScale != nil || hasOwnFrameLoop

        uopBlocks.append(
          BlockUOps(
            ops: finalOps,
            kind: effectiveKind,
            temporality: block.temporality,
            parallelPolicy: parallelPolicy,
            forceNewKernel: forceNewKernel,
            threadCountScale: threadCountScale,
            hasOwnFrameLoop: hasOwnFrameLoop))
      }
    }

    return uopBlocks
  }

  /// Removes thread-scale directives from ops and returns the last scale encountered.
  private static func extractThreadCountScale(from ops: [UOp]) -> (Int?, [UOp]) {
    var scale: Int? = nil
    var filtered: [UOp] = []
    filtered.reserveCapacity(ops.count)
    for op in ops {
      if case .setThreadCountScale(let s) = op.op {
        scale = s
        continue
      }
      filtered.append(op)
    }
    return (scale, filtered)
  }

  /// Resolves final block kind after backend-specific scalar constraints.
  private static func resolveEffectiveKind(
    backend: Backend, blockKind: Kind, bodyEffectiveKind: Kind, threadCountScale: Int?,
    ops: inout [UOp]
  ) -> Kind {
    let mustForceScalar =
      backend == .c && (threadCountScale != nil || bodyEffectiveKind == .scalar)
    guard mustForceScalar else { return blockKind }

    for i in ops.indices {
      ops[i].kind = .scalar
    }
    return .scalar
  }

  /// Infers whether a block can launch with thread-level parallel policy.
  private static func inferParallelPolicy(
    kind: Kind, temporality: Temporality, ops: [UOp]
  ) -> ParallelPolicy {
    guard temporality == .static_, kind == .scalar else { return .serial }
    let hasParallelRange = ops.contains {
      if case .beginParallelRange = $0.op { return true }
      return false
    }
    return hasParallelRange ? .threadParallel : .serial
  }

  /// Ensures a voice-index cell exists when multi-voice rendering is requested.
  private static func ensureVoiceCell(
    graph: Graph, options: Options
  ) -> (id: Int?, generated: Bool) {
    var voiceCellId = options.voiceCellId
    var generated = false
    if options.voiceCount > 1 && voiceCellId == nil {
      voiceCellId = graph.alloc()
      generated = true
    }
    return (voiceCellId, generated)
  }

  /// Applies mapped voice-cell state to renderers that expose voice configuration.
  private static func configureRendererVoiceState(
    renderer: Renderer, voiceCount: Int, voiceCellId: Int?, cellAllocations: CellAllocations
  ) {
    guard let cRenderer = renderer as? CRenderer else { return }
    cRenderer.voiceCount = voiceCount
    if let voiceCellId {
      cRenderer.voiceCellIdOpt = cellAllocations.cellMappings[voiceCellId]
    }
  }

  /// Create a renderer for the specified backend
  private static func createRenderer(for backend: Backend, options: Options) -> Renderer {
    switch backend {
    case .c:
      let r = CRenderer()
      r.voiceCount = options.voiceCount
      r.voiceCellIdOpt = options.voiceCellId
      return r
    case .metal:
      return MetalRenderer()
    }
  }

  /// Print the block structure in a human-readable format
  private static func printBlockStructure(blocks: [Block], sortedIndices: [Int]) {
    print("Block structure:")
    for (i, blockIdx) in sortedIndices.enumerated() {
      let block = blocks[blockIdx]
      print("  Block \(i) (orig \(blockIdx), \(block.kind)): \(block.nodes)")
    }
  }
}

// MARK: - Convenience Extensions

extension CompilationResult {
  /// Get the generated source code for all kernels
  public var source: String {
    kernels.map { $0.source }.joined(separator: "\n\n")
  }

  /// Get the first kernel's source (useful for single-kernel results)
  public var firstKernelSource: String? {
    kernels.first?.source
  }

  /// Check if compilation produced any kernels
  public var hasKernels: Bool {
    !kernels.isEmpty
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
      "block #\(i+1) kind=\(uopBlock.kind) threadCountScale=\(String(describing: uopBlock.threadCountScale)) shape=\(shape) tensorIndex=\(tensorIndex) nodes=\(nodes)"
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

// MARK: - History Operation Combining Pass

/// Combines historyRead and historyWrite operations that are not in feedback loops
/// into a single historyReadWrite operation
func combineHistoryOpsNotInFeedback(
  _ graph: Graph, feedbackClusters: [[NodeID]], options: CompilationPipeline.Options
) {
  // Create a set of all nodes that are in feedback loops
  var nodesInFeedback = Set<NodeID>()
  for cluster in feedbackClusters {
    for nodeId in cluster {
      nodesInFeedback.insert(nodeId)
    }
  }

  // Find all historyRead and historyWrite nodes grouped by cellId
  var historyReads: [CellID: NodeID] = [:]
  var historyWrites: [CellID: (nodeId: NodeID, inputs: [NodeID])] = [:]

  for (nodeId, node) in graph.nodes {
    switch node.op {
    case .historyRead(let cellId):
      historyReads[cellId] = nodeId
    case .historyWrite(let cellId):
      historyWrites[cellId] = (nodeId: nodeId, inputs: node.inputs)
    default:
      break
    }
  }

  // For each cellId that has both read and write, check if they're not in feedback loops
  for (cellId, readNodeId) in historyReads {
    if let writeInfo = historyWrites[cellId] {
      // Check if neither the read nor write node is in a feedback loop
      if !nodesInFeedback.contains(readNodeId) && !nodesInFeedback.contains(writeInfo.nodeId) {
        // Replace the historyRead node with historyReadWrite using the write's inputs
        if graph.nodes[readNodeId] != nil {
          // Create new node with historyReadWrite operation at the read node's ID
          let newNode = Node(
            id: readNodeId,
            op: .historyReadWrite(cellId),
            inputs: writeInfo.inputs
          )
          graph.nodes[readNodeId] = newNode

          // Remove the historyWrite node
          graph.nodes.removeValue(forKey: writeInfo.nodeId)

          if options.debug {
            print("   - Converted read node \(readNodeId) to historyReadWrite")
            print("   - Removed historyWrite node \(writeInfo.nodeId)")
            print("   - Inputs: \(writeInfo.inputs)")
          }
        }
      } else if options.debug {
        print("⚠️  Skipping combination for cell \(cellId) - nodes are in feedback loop")
      }
    }
  }
}

// MARK: - Constant Folding Pass

/// Folds constant expressions at the graph level before topological sort.
func foldConstants(_ graph: Graph, options: CompilationPipeline.Options) {
  // Track constant values: NodeID -> Float
  var constantValues: [NodeID: Float] = [:]

  // Initialize with existing constants
  for (nodeId, node) in graph.nodes {
    if case .constant(let value) = node.op {
      constantValues[nodeId] = value
    }
  }

  // Build consumer map: input -> [consumers]
  var consumers: [NodeID: [NodeID]] = [:]
  for (nodeId, node) in graph.nodes {
    for input in node.inputs {
      consumers[input, default: []].append(nodeId)
    }
  }

  // Initialize worklist with foldable nodes that have all-constant inputs
  var worklist = Set<NodeID>()
  for (nodeId, node) in graph.nodes {
    if canFoldOp(node.op) && !node.inputs.isEmpty
      && node.inputs.allSatisfy({ constantValues[$0] != nil })
    {
      worklist.insert(nodeId)
    }
  }

  var foldedCount = 0

  // Process worklist
  while let nodeId = worklist.popFirst() {
    guard let node = graph.nodes[nodeId] else { continue }

    // Get input values
    let inputValues = node.inputs.compactMap { constantValues[$0] }
    guard inputValues.count == node.inputs.count else { continue }

    // Evaluate the constant expression
    guard let result = evaluateConstantOp(node.op, inputValues),
      result.isFinite
    else { continue }

    // Replace node with constant (preserves NodeID, no rewiring needed)
    constantValues[nodeId] = result
    graph.nodes[nodeId] = Node(id: nodeId, op: .constant(result), inputs: [])
    foldedCount += 1

    // Add newly-eligible consumers to worklist
    for consumer in consumers[nodeId] ?? [] {
      if let consumerNode = graph.nodes[consumer],
        canFoldOp(consumerNode.op),
        consumerNode.inputs.allSatisfy({ constantValues[$0] != nil })
      {
        worklist.insert(consumer)
      }
    }
  }

  if options.debug && foldedCount > 0 {
    print("Constant folding: folded \(foldedCount) nodes")
  }
}

private func canFoldOp(_ op: LazyOp) -> Bool {
  switch op {
  // Arithmetic
  case .add, .sub, .mul, .div, .pow, .mod, .min, .max:
    return true
  // Comparisons
  case .gt, .gte, .lt, .lte, .eq:
    return true
  // Logical
  case .and, .or, .xor:
    return true
  // Unary math
  case .abs, .sign, .sin, .cos, .tan, .tanh, .exp, .log, .log10, .sqrt,
    .floor, .ceil, .round, .atan2:
    return true
  // Control flow (key for biquad)
  case .gswitch, .mix, .selector:
    return true
  default:
    return false
  }
}

private func evaluateConstantOp(_ op: LazyOp, _ inputs: [Float]) -> Float? {
  switch op {
  // Unary
  case .abs: return inputs.count == 1 ? Swift.abs(inputs[0]) : nil
  case .sign: return inputs.count == 1 ? (inputs[0] > 0 ? 1 : (inputs[0] < 0 ? -1 : 0)) : nil
  case .sin: return inputs.count == 1 ? sin(inputs[0]) : nil
  case .cos: return inputs.count == 1 ? cos(inputs[0]) : nil
  case .tan: return inputs.count == 1 ? tan(inputs[0]) : nil
  case .tanh: return inputs.count == 1 ? tanh(inputs[0]) : nil
  case .exp: return inputs.count == 1 ? exp(inputs[0]) : nil
  case .log: return inputs.count == 1 && inputs[0] > 0 ? log(inputs[0]) : nil
  case .log10: return inputs.count == 1 && inputs[0] > 0 ? log10(inputs[0]) : nil
  case .sqrt: return inputs.count == 1 && inputs[0] >= 0 ? sqrt(inputs[0]) : nil
  case .floor: return inputs.count == 1 ? floor(inputs[0]) : nil
  case .ceil: return inputs.count == 1 ? ceil(inputs[0]) : nil
  case .round: return inputs.count == 1 ? round(inputs[0]) : nil

  // Binary
  case .add: return inputs.count == 2 ? inputs[0] + inputs[1] : nil
  case .sub: return inputs.count == 2 ? inputs[0] - inputs[1] : nil
  case .mul: return inputs.count == 2 ? inputs[0] * inputs[1] : nil
  case .div: return inputs.count == 2 && inputs[1] != 0 ? inputs[0] / inputs[1] : nil
  case .pow: return inputs.count == 2 ? pow(inputs[0], inputs[1]) : nil
  case .mod:
    return inputs.count == 2 && inputs[1] != 0
      ? inputs[0].truncatingRemainder(dividingBy: inputs[1]) : nil
  case .min: return inputs.count == 2 ? Swift.min(inputs[0], inputs[1]) : nil
  case .max: return inputs.count == 2 ? Swift.max(inputs[0], inputs[1]) : nil
  case .atan2: return inputs.count == 2 ? atan2(inputs[0], inputs[1]) : nil

  // Comparisons (return 1.0 for true, 0.0 for false)
  case .gt: return inputs.count == 2 ? (inputs[0] > inputs[1] ? 1 : 0) : nil
  case .gte: return inputs.count == 2 ? (inputs[0] >= inputs[1] ? 1 : 0) : nil
  case .lt: return inputs.count == 2 ? (inputs[0] < inputs[1] ? 1 : 0) : nil
  case .lte: return inputs.count == 2 ? (inputs[0] <= inputs[1] ? 1 : 0) : nil
  case .eq: return inputs.count == 2 ? (inputs[0] == inputs[1] ? 1 : 0) : nil

  // Logical
  case .and: return inputs.count == 2 ? ((inputs[0] != 0 && inputs[1] != 0) ? 1 : 0) : nil
  case .or: return inputs.count == 2 ? ((inputs[0] != 0 || inputs[1] != 0) ? 1 : 0) : nil
  case .xor: return inputs.count == 2 ? (((inputs[0] != 0) != (inputs[1] != 0)) ? 1 : 0) : nil

  // Ternary (key for biquad mode selection)
  case .gswitch:
    // gswitch(cond, ifTrue, ifFalse): returns ifTrue if cond > 0
    return inputs.count == 3 ? (inputs[0] > 0 ? inputs[1] : inputs[2]) : nil
  case .mix:
    // mix(a, b, t) = a * (1-t) + b * t
    return inputs.count == 3 ? inputs[0] * (1 - inputs[2]) + inputs[1] * inputs[2] : nil

  // N-ary (key for biquad mode selection)
  case .selector:
    // selector(mode, options...): 1-indexed, mode<=0 returns 0
    guard inputs.count >= 2 else { return nil }
    let mode = Int(inputs[0])
    if mode <= 0 { return 0.0 }
    if mode <= inputs.count - 1 {
      return inputs[mode]
    }
    return 0.0  // Out of range

  default:
    return nil
  }
}

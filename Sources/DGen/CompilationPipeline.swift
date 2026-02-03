import Foundation

/// Result of the compilation pipeline containing all intermediate steps
public struct CompilationResult {
  /// Original graph
  public let graph: Graph

  /// Topologically sorted node IDs
  public let sortedNodes: [NodeID]

  /// Set of nodes that must execute in scalar mode
  public let scalarNodes: Set<NodeID>

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

    public init(
      frameCount: Int = 128,
      debug: Bool = false,
      printBlockStructure: Bool = false,
      forceScalar: Bool = false,
      voiceCount: Int = 1,
      voiceCellId: Int? = nil
    ) {
      self.frameCount = frameCount
      self.debug = debug
      self.printBlockStructure = printBlockStructure
      self.forceScalar = forceScalar
      self.voiceCount = voiceCount
      self.voiceCellId = voiceCellId
    }
  }

  /// Compile a graph with the specified backend and options
  public static func compile(
    graph: Graph,
    backend: Backend,
    options: Options = Options(),
    name: String = "kernel"
  ) throws -> CompilationResult {
    precondition(
      options.frameCount <= graph.maxFrameCount,
      "frameCount (\(options.frameCount)) exceeds graph.maxFrameCount (\(graph.maxFrameCount)). "
        + "Set graph.maxFrameCount to at least \(options.frameCount) before compilation."
    )
    let pipelineStart = CFAbsoluteTimeGetCurrent()
    var timings: [(String, Double)] = []

    func time<T>(_ label: String, _ block: () throws -> T) rethrows -> T {
      let start = CFAbsoluteTimeGetCurrent()
      let result = try block()
      let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
      timings.append((label, elapsed))
      return result
    }

    // Step 1: Topological sort that respects scalar corridors
    let feedbackClusters = time("findFeedbackLoops") {
      findFeedbackLoops(graph)
    }

    // Step 1.5: Combine history operations that are not in feedback loops
    time("combineHistoryOps") {
      combineHistoryOpsNotInFeedback(
        graph, feedbackClusters: feedbackClusters, options: options)
    }

    // Step 1.6: Fold constant expressions
    time("foldConstants") {
      foldConstants(graph, options: options)
    }

    let scalarNodeSet = time("scalarNodes") {
      options.forceScalar
        ? Set(graph.nodes.keys)
        : scalarNodes(graph, feedbackClusters: feedbackClusters, backend: backend)
    }

    let sortedNodes = time("topoWithCorridors") {
      topoWithCorridors(
        graph, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet,
        debug: false)
    }

    try time("inferShapes") {
      try inferShapes(graph: graph, sortedNodes: sortedNodes)
    }

    time("allocateTensorOutputs") {
      allocateTensorOutputs(graph: graph, sortedNodes: sortedNodes)
    }

    // Step 2: Determine scalar nodes and create blocks

    // Step 2.5: Handle seq operators - if any input to seq is scalar, make all inputs scalar
    // But don't re-add SIMD-safe operations (they use atomics and can run in parallel)
    var finalScalarSet = scalarNodeSet
    time("seqScalarPropagate") {
      // Build set of SIMD-safe nodes that should never be marked scalar
      var simdSafe = Set<NodeID>()
      for (nodeId, node) in graph.nodes {
        switch node.op {
        case .memoryAccumulate(_), .tensorAccumulate(_):
          simdSafe.insert(nodeId)
        default: break
        }
      }

      for (_, node) in graph.nodes {
        if case .seq = node.op {
          let hasScalarInput = node.inputs.contains { finalScalarSet.contains($0) }
          if hasScalarInput {
            for inputId in node.inputs {
              // Don't add SIMD-safe nodes to scalar set
              if !simdSafe.contains(inputId) {
                finalScalarSet.insert(inputId)
              }
            }
          }
        }
      }
    }

    // Step 3: Determine blocks (simplified since corridors are already grouped)
    let blocks = time("determineBlocks") {
      determineBlocksSimple(
        sorted: sortedNodes,
        scalar: finalScalarSet,
        g: graph,
        debug: false
      )
    }

    // Since we're using corridor-aware topological sort, blocks are already properly ordered
    // Fuse adjacent blocks of the same kind to reduce cross-block communication

    // rather than having a different buffer for each value we could have one giant array and significantly reduce the number of cross-chain-blocks needed
    let fusedBlocks = time("fuseBlocks1") {
      fuseBlocks(blocks, graph)
    }

    let context = IRContext(g: graph)

    // finally separate tensor blocks of shared size into their own blocks
    let seperatedBlocks = time("tensorBlocks") {
      determineTensorBlocks(fusedBlocks, graph, context)
    }

    var finalBlocks = seperatedBlocks.compactMap { $0 }

    // Isolate spectral loss passes into their own blocks to prevent race conditions
    // This ensures FFT forward pass completes before gradient pass runs
    // Only do this if the graph has spectral loss ops (to avoid modifying blocks unnecessarily)
    let hasSpectralLossOps = graph.nodes.values.contains { node in
      switch node.op {
      case .spectralLossFFT, .spectralLossFFTGradInline, .spectralLossFFTGradRead,
        .spectralLossFFTGradRead2:
        return true
      default:
        return false
      }
    }
    if hasSpectralLossOps {
      finalBlocks = isolateSpectralPasses(finalBlocks, graph)
    }

    // Step 4: Infer temporality and assign to blocks
    // This now includes hop-based temporality for FFT/IFFT and downstream operations
    let temporalityResult = time("inferTemporality") {
      inferTemporality(graph: graph, sortedNodes: sortedNodes)
    }

    time("assignTemporality") {
      assignBlockTemporality(
        blocks: &finalBlocks,
        frameBasedNodes: temporalityResult.frameBasedNodes,
        hopBasedNodes: temporalityResult.hopBasedNodes
      )
    }

    // Split reduce blocks AFTER temporality assignment so we can override
    // temporality for global reduces (peekRowGradReduce/selectRowGradReduce)
    if backend == .metal {
      finalBlocks = splitReduceBlocks(g: graph, blocks: finalBlocks)
    }

    // Step 4.6: Allocate real memory for lazy tensor cells
    // Now that we know temporality, we can allocate the right sizes:
    // - Frame-based outbound tensors: tensorSize * frameCount
    // - Static outbound tensors: tensorSize
    // - Non-outbound tensors: no allocation (register-only)
    time("allocateTensorMemory") {
      // Convert feedback clusters to a flat set of nodes
      let feedbackClusterNodes = Set(feedbackClusters.flatMap { $0 })
      allocateTensorMemory(
        graph: graph,
        blocks: finalBlocks,
        frameBasedNodes: temporalityResult.frameBasedNodes,
        feedbackClusterNodes: feedbackClusterNodes,
        backend: backend,
        frameCount: options.frameCount
      )
      // Populate IRContext's frameAwareTensorCells from graph.frameAwareCells
      // This is needed for tstore/tload to use frame-indexed addressing
      for cellId in graph.frameAwareCells.keys {
        context.frameAwareTensorCells.insert(cellId)
      }
    }

    let finalBlockIndices = Array(0..<finalBlocks.count)

    // Step 5: Convert blocks to UOp blocks
    var uopBlocks = [BlockUOps]()

    func inferParallelPolicy(
      kind: Kind, temporality: Temporality, ops: [UOp]
    ) -> ParallelPolicy {
      guard temporality == .static_, kind == .scalar else { return .serial }

      var hasParallelRange = false

      for uop in ops {
        switch uop.op {
        case .beginParallelRange:
          hasParallelRange = true
        default:
          break
        }
      }
      if hasParallelRange { return .threadParallel }
      return .serial
    }

    func extractThreadCountScale(_ ops: [UOp]) -> (Int?, [UOp]) {
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

    try time("emitBlockUOps") {
      for blockIdx in finalBlockIndices {
        let block = finalBlocks[blockIdx]
        let ops = try emitBlockUOps(
          ctx: context,
          block: block,
          blocks: finalBlocks,
          g: graph,
          backend: backend,
          debug: options.debug
        )
        let (threadCountScale, filteredOps) = extractThreadCountScale(ops)
        var finalOps = filteredOps
        let effectiveKind: Kind
        if backend == .c, threadCountScale != nil {
          for i in 0..<finalOps.count {
            finalOps[i].kind = .scalar
          }
          effectiveKind = .scalar
        } else {
          effectiveKind = block.kind
        }
        let parallelPolicy = inferParallelPolicy(
          kind: effectiveKind, temporality: block.temporality, ops: finalOps)
        // Force new kernel for spectral passes (forward + backward) and scaled thread blocks
        let forceNew = threadCountScale != nil
        uopBlocks.append(
          BlockUOps(
            ops: finalOps, kind: effectiveKind, temporality: block.temporality,
            parallelPolicy: parallelPolicy, forceNewKernel: forceNew,
            threadCountScale: threadCountScale))
      }
    }

    uopBlocks.removeAll { $0.ops.isEmpty }

    if options.debug {
      printUOpBlocks(uopBlocks, blocks: finalBlocks)
    }

    // Step 7: Lower UOp blocks to compiled kernels
    // Ensure a dedicated voice cell exists when voiceCount > 1
    var voiceCellIdFinal: Int? = options.voiceCellId
    var generatedVoiceCell = false
    if options.voiceCount > 1 && voiceCellIdFinal == nil {
      voiceCellIdFinal = graph.alloc()  // Reserve a cell for voice index
      generatedVoiceCell = true
    }

    // Step 6: Fix memory slot conflicts for vector operations
    let cellAllocations = time("remapMemorySlots") {
      remapVectorMemorySlots(
        &uopBlocks, cellSizes: graph.cellAllocationSizes,
        voiceCellId: generatedVoiceCell ? voiceCellIdFinal : nil)
    }

    let renderer: Renderer = createRenderer(for: backend, options: options)
    if let cr = renderer as? CRenderer {
      cr.voiceCount = options.voiceCount
      if let voiceCellId = voiceCellIdFinal {
        cr.voiceCellIdOpt = cellAllocations.cellMappings[voiceCellId]
      }
    }
    let kernels = try time("lowerUOpBlocks") {
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

    // Print timing summary
    let pipelineTotal = (CFAbsoluteTimeGetCurrent() - pipelineStart) * 1000
    /*
    print(
      "⏱️ [DGen Pipeline] Total: \(String(format: "%.1f", pipelineTotal))ms | nodes: \(graph.nodes.count)"
    )
    let sortedTimings = timings.sorted { $0.1 > $1.1 }
    for (label, ms) in sortedTimings.prefix(10) {
      let pct = (ms / pipelineTotal) * 100
      print(
        "   \(String(format: "%6.1f", ms))ms (\(String(format: "%4.1f", pct))%) - \(label)")
    }
    */

    return CompilationResult(
      graph: graph,
      sortedNodes: sortedNodes,
      scalarNodes: finalScalarSet,
      blocks: finalBlocks,
      sortedBlockIndices: finalBlockIndices,
      sortedBlocks: finalBlocks,
      context: context,
      uopBlocks: uopBlocks,
      kernels: kernels,
      backend: backend,
      totalMemorySlots: cellAllocations.totalMemorySlots,
      cellAllocations: cellAllocations,
      voiceCellId: voiceCellIdFinal,
    )
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
    print(
      "block #\(i+1) kind=\(uopBlock.kind) threadCountScale\(uopBlock.threadCountScale) shape=\(blocks[i].shape) tensorIndex=\(blocks[i].tensorIndex)"
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

// MARK: - Memory Layout Remapping

/// Remap memory slots to avoid conflicts between scalar and vector operations
/// Returns the total number of memory slots needed after remapping
func remapVectorMemorySlots(
  _ uopBlocks: inout [BlockUOps], cellSizes: [CellID: Int], voiceCellId: CellID?
)
  -> CellAllocations
{
  // Collect all memory operations and their execution modes
  var memoryUsage: [CellID: Kind] = [:]
  var allCellIds: Set<CellID> = []
  var cellUsedInMultipleModes: Set<CellID> = []

  // Helper to register a cell's usage and detect multi-mode access
  func registerCell(_ cellId: CellID, kind: Kind) {
    allCellIds.insert(cellId)

    if let existingKind = memoryUsage[cellId] {
      if existingKind != kind {
        // Cell used in multiple execution modes
        cellUsedInMultipleModes.insert(cellId)
        // Upgrade to SIMD if any block uses it as SIMD
        // (SIMD needs 4x space, scalar can still access first element)
        if kind == .simd || existingKind == .simd {
          memoryUsage[cellId] = .simd
        }
      }
    } else {
      memoryUsage[cellId] = kind
    }
  }

  // First pass: identify which memory cells are used in which execution modes
  for block in uopBlocks {
    for uop in block.ops {
      if let cellId = uop.op.memoryCellId {
        registerCell(cellId, kind: block.kind)
      }
    }
  }

  if let voiceCellId = voiceCellId {
    registerCell(voiceCellId, kind: .simd)
  }

  // Second pass: create a remapping for vector cells
  var cellRemapping: [CellID: CellID] = [:]
  var nextAvailableSlot = (allCellIds.max() ?? -1) + 1

  for (cellId, kind) in memoryUsage.sorted(by: { $0.key < $1.key }) {
    let allocSize = cellSizes[cellId] ?? 1

    if kind == .simd {
      let alignedSlot = ((nextAvailableSlot + 3) / 4) * 4
      cellRemapping[cellId] = alignedSlot
      nextAvailableSlot = alignedSlot + max(4, allocSize)
    } else if allocSize > 1 {
      cellRemapping[cellId] = nextAvailableSlot
      nextAvailableSlot += allocSize
    } else {
      cellRemapping[cellId] = cellId
    }
  }

  // Third pass: apply the remapping to all UOps
  for blockIndex in 0..<uopBlocks.count {
    for uopIndex in 0..<uopBlocks[blockIndex].ops.count {
      let uop = uopBlocks[blockIndex].ops[uopIndex]
      if let remappedOp = uop.op.withRemappedCellId(cellRemapping) {
        uopBlocks[blockIndex].ops[uopIndex] = UOp(
          op: remappedOp,
          value: uop.value,
          kind: uop.kind
        )
      }
    }
  }

  return CellAllocations(
    totalMemorySlots: nextAvailableSlot,
    cellMappings: cellRemapping,
    cellKinds: memoryUsage
  )
}

public struct CellAllocations {
  public let cellKinds: [CellID: Kind]
  public let cellMappings: [CellID: CellID]
  public let totalMemorySlots: Int

  public init(totalMemorySlots: Int, cellMappings: [CellID: CellID], cellKinds: [CellID: Kind]) {
    self.cellKinds = cellKinds
    self.cellMappings = cellMappings
    self.totalMemorySlots = totalMemorySlots
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

func determineTensorBlocks(_ blocks: [Block], _ graph: Graph, _ ctx: IRContext) -> [Block] {
  var determined: [Block] = []

  // Helper to create a new block preserving original properties
  func makeBlock(from original: Block) -> Block {
    var newBlock = Block(kind: original.kind)
    newBlock.temporality = original.temporality
    return newBlock
  }

  for block in blocks {
    var innerBlocks: [Block] = []
    var currentBlock = makeBlock(from: block)
    var currentShape: Shape? = nil
    for nodeId in block.nodes {
      if let node = graph.nodes[nodeId] {
        // Skip tensorRef nodes for tensor block grouping decisions.
        //
        // tensorRef nodes are just data containers - they emit nothing (return []).
        // If we let them create tensor blocks, we'd get empty parallel loops:
        //   for (int simd1 = 0; simd1 < 16; simd1+=4) { }  // empty!
        //
        // Instead, tensorRef nodes stay in whatever block they're in but don't
        // trigger new tensor block creation. The actual tensor OPERATIONS (mul, add, etc.)
        // that process tensor data will create the tensor blocks.
        if case .tensorRef = node.op {
          if case .tensor(let shape) = node.shape {
            currentBlock.shape = shape
            currentBlock.tensorIndex = ctx.useVariable(src: nil)
            currentShape = shape
          }
          currentBlock.nodes.append(nodeId)
          continue
        }

        // Skip view operations (reshape, transpose) for tensor block grouping.
        //
        // These are metadata-only changes - they share the input tensor's cellId
        // and don't emit any actual code. If we let them create tensor blocks,
        // we'd get empty parallel loops. The operations that USE these reshaped
        // tensors (mul, add, etc.) will create the actual tensor blocks.
        if case .reshape = node.op {
          currentBlock.nodes.append(nodeId)
          continue
        }
        if case .transpose = node.op {
          currentBlock.nodes.append(nodeId)
          continue
        }

        if case .conv2d = node.op {
          if currentShape != nil {
            if currentBlock.nodes.count > 0 {
              innerBlocks.append(currentBlock)
            }
            // regular node
            currentBlock = makeBlock(from: block)
          }
          currentShape = nil

        } else if case .fft = node.op {
          // FFT is a bulk tensor operation that handles all tensor writes internally
          // It needs its own scalar block - don't mix with SIMD ops
          if currentBlock.nodes.count > 0 {
            innerBlocks.append(currentBlock)
          }
          // Create isolated block for FFT
          currentBlock = makeBlock(from: block)
          currentBlock.kind = .scalar  // Force scalar execution
          currentBlock.nodes.append(nodeId)
          innerBlocks.append(currentBlock)
          // Start fresh block for nodes after FFT
          currentBlock = makeBlock(from: block)
          currentShape = nil
          continue  // Skip the append at end of loop
        } else if case .constant = node.op {
          // do nothing
        } else if case .ifft = node.op {
          // IFFT is a bulk operation that handles spectrum-to-time conversion
          // It needs its own scalar block - don't mix with SIMD ops
          if currentBlock.nodes.count > 0 {
            innerBlocks.append(currentBlock)
          }
          // Create isolated block for IFFT
          currentBlock = makeBlock(from: block)
          currentBlock.kind = .scalar  // Force scalar execution
          currentBlock.nodes.append(nodeId)
          innerBlocks.append(currentBlock)
          // Start fresh block for nodes after IFFT
          currentBlock = makeBlock(from: block)
          currentShape = nil
          continue  // Skip the append at end of loop
        } else if case .tensor(let shape) = node.shape {
          if shape != currentShape {
            if currentBlock.nodes.count > 0 {
              innerBlocks.append(currentBlock)
            }
            // tensor block
            currentBlock = makeBlock(from: block)
            currentBlock.tensorIndex = ctx.useVariable(src: nil)
            currentBlock.shape = shape
            currentShape = shape
          }
        } else {
          if currentShape != nil {
            if currentBlock.nodes.count > 0 {
              innerBlocks.append(currentBlock)
            }
            // regular node
            currentBlock = makeBlock(from: block)
          }
          currentShape = nil
        }
      }
      currentBlock.nodes.append(nodeId)
    }
    if currentBlock.nodes.count > 0 {
      innerBlocks.append(currentBlock)
    }
    for block in innerBlocks {
      determined.append(block)
    }
  }
  return determined
}

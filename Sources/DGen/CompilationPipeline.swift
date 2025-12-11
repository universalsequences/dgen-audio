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
        public let backwards: Bool

        public init(
            frameCount: Int = 128,
            debug: Bool = false,
            printBlockStructure: Bool = false,
            forceScalar: Bool = false,
            backwards: Bool = false,
            voiceCount: Int = 1,
            voiceCellId: Int? = nil
        ) {
            self.frameCount = frameCount
            self.debug = debug
            self.printBlockStructure = printBlockStructure
            self.forceScalar = forceScalar
            self.voiceCount = voiceCount
            self.voiceCellId = voiceCellId
            self.backwards = backwards
        }
    }

    /// Compile a graph with the specified backend and options
    public static func compile(
        graph: Graph,
        backend: Backend,
        options: Options = Options(),
        name: String = "kernel"
    ) throws -> CompilationResult {
        // Step 1: Topological sort that respects scalar corridors
        let feedbackClusters = findFeedbackLoops(graph)

        // Step 1.5: Combine history operations that are not in feedback loops
        combineHistoryOpsNotInFeedback(graph, feedbackClusters: feedbackClusters, options: options)

        let scalarNodeSet =
            options.forceScalar
            ? Set(graph.nodes.keys)
            : scalarNodes(graph, feedbackClusters: feedbackClusters)

        let sortedNodes = topoWithCorridors(
            graph, feedbackClusters: feedbackClusters, scalarNodeSet: scalarNodeSet, debug: false)

        try inferShapes(graph: graph, sortedNodes: sortedNodes)

        allocateTensorOutputs(graph: graph, sortedNodes: sortedNodes)

        // Step 2: Determine scalar nodes and create blocks

        // Step 2.5: Handle seq operators - if any input to seq is scalar, make all inputs scalar
        var finalScalarSet = scalarNodeSet
        for (_, node) in graph.nodes {
            if case .seq = node.op {
                let hasScalarInput = node.inputs.contains { finalScalarSet.contains($0) }
                if hasScalarInput {
                    for inputId in node.inputs {
                        finalScalarSet.insert(inputId)
                    }
                }
            }
        }

        // Step 3: Determine blocks (simplified since corridors are already grouped)
        let blocks = determineBlocksSimple(
            sorted: sortedNodes,
            scalar: finalScalarSet,
            g: graph,
            debug: false
        )

        // Since we're using corridor-aware topological sort, blocks are already properly ordered
        // Fuse adjacent blocks of the same kind to reduce cross-block communication

        // rather than having a different buffer for each value we could have one giant array and significantly reduce the number of cross-chain-blocks needed
        let fusedBlocks = fuseBlocks(blocks, graph)

        // Isolate spectralLossPass1 and Pass2 into their own kernels to avoid dependency issues
        let isolatedBlocks = isolateSpectralPasses(fusedBlocks, graph)

        // Re-run fusion after isolation, to merge any adjacent same-kind blocks that were split
        // by isolation but do not straddle Pass1/Pass2 boundaries.
        let reFusedBlocks = fuseBlocks(isolatedBlocks, graph)

        let context = IRContext(g: graph)

        // finally separate tensor blocks of shared size into their own blocks
        let seperatedBlocks = determineTensorBlocks(reFusedBlocks, graph, context)

        var finalBlocks = seperatedBlocks.compactMap { $0 }

        // Re-check block kinds after tensor block splitting.
        // When determineTensorBlocks splits a block, the new blocks inherit the original's kind,
        // but if all nodes in the new block are scalar, the block should be scalar.
        // This is critical for frame-based tensor ops (cos(phasor(tensor))) to get stride 1.
        for i in 0..<finalBlocks.count {
            let allNodesScalar = finalBlocks[i].nodes.allSatisfy { finalScalarSet.contains($0) }
            if allNodesScalar && finalBlocks[i].kind == .simd {
                finalBlocks[i].kind = .scalar
            }
        }

        if options.backwards {
            var backwardsBlocks: [Block] = []
            for block in finalBlocks.reversed() {
                var backwardsBlock = Block(kind: block.kind)
                backwardsBlock.nodes = block.nodes.reversed()
                backwardsBlock.direction = .backwards
                backwardsBlocks.append(backwardsBlock)

            }
            finalBlocks += fuseBlocks(backwardsBlocks, graph)
        }

        // Step 4: Infer temporality and assign to blocks
        let frameBasedNodes = inferTemporality(graph: graph, sortedNodes: sortedNodes)
        assignBlockTemporality(blocks: &finalBlocks, frameBasedNodes: frameBasedNodes)

        let finalBlockIndices = Array(0..<finalBlocks.count)

        var blockId: Int = 0
        for block in finalBlocks {
            print("block \(blockId) kind=\(block.kind) temporality=\(block.temporality)")
            blockId += 1
        }

        // Step 5: Convert blocks to UOp blocks
        var uopBlocks = [BlockUOps]()

        for blockIdx in finalBlockIndices {
            let block = finalBlocks[blockIdx]
            let ops = try emitBlockUOps(
                ctx: context,
                block: block,
                blocks: finalBlocks,
                g: graph,
                debug: options.debug
            )
            uopBlocks.append(BlockUOps(ops: ops, kind: block.kind, temporality: block.temporality))
        }

        // Remove empty UOp blocks prior to vector memory remap and lowering
        uopBlocks.removeAll { $0.ops.isEmpty }

        // Step 5.5: Fuse consecutive parallelRange loops with producer-consumer relationships.
        // This reduces redundant memory traffic and makes the generated code cleaner.
        for i in 0..<uopBlocks.count {
            //uopBlocks[i].ops = fuseParallelRanges(uopBlocks[i].ops)
        }

        // Debug: print UOps after fusion
        if options.debug {
            var i = 1
            for uopBlock in uopBlocks {
                print("Block \(i) temporality=\(uopBlock.temporality)")
                i += 1
                var indentLevel = 0
                for uop in uopBlock.ops {
                    switch uop.op {
                    case .beginIf, .beginForLoop, .beginParallelRange, .beginLoop, .beginRange:
                        print(
                            "\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())"
                        )
                        indentLevel += 1
                    case .endIf, .endLoop, .endParallelRange, .endRange:
                        indentLevel = max(0, indentLevel - 1)
                        print(
                            "\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())"
                        )
                    default:
                        print(
                            "\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())"
                        )
                    }
                }
            }
        }

        // Step 7: Lower UOp blocks to compiled kernels
        // Ensure a dedicated voice cell exists when voiceCount > 1
        var voiceCellIdFinal: Int? = options.voiceCellId
        print("\(ANSI.green)VOICE COUNT=\(options.voiceCount)\(ANSI.reset)")
        var generatedVoiceCell = false
        if options.voiceCount > 1 && voiceCellIdFinal == nil {
            voiceCellIdFinal = graph.alloc()  // Reserve a cell for voice index
            generatedVoiceCell = true
            print("\(ANSI.green)ALLOCATING VOICE CELL \(voiceCellIdFinal) \(ANSI.reset)")
        }

        // Step 6: Fix memory slot conflicts for vector operations
        let cellAllocations = remapVectorMemorySlots(
            &uopBlocks, cellSizes: graph.cellAllocationSizes,
            voiceCellId: generatedVoiceCell ? voiceCellIdFinal : nil)

        let renderer: Renderer = createRenderer(for: backend, options: options)
        if let cr = renderer as? CRenderer {
            print("we got a c renderer! voiceCellIdFinal=\(voiceCellIdFinal)")
            cr.voiceCount = options.voiceCount
            if let voiceCellId = voiceCellIdFinal {
                cr.voiceCellIdOpt = cellAllocations.cellMappings[voiceCellId]
                print("Cell mapped voice id for voiceId=\(voiceCellId) -> \(cr.voiceCellIdOpt)")
            }
        }
        let kernels = try lowerUOpBlocks(
            &uopBlocks,
            renderer: renderer,
            ctx: context,
            frameCount: options.frameCount,
            graph: graph,
            totalMemorySlots: cellAllocations.totalMemorySlots,
            name: name
        )

        // Use the scalar node set we calculated earlier

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
            switch uop.op {
            case let .load(cellId):
                registerCell(cellId, kind: block.kind)
            case let .store(cellId, _):
                registerCell(cellId, kind: block.kind)
            case let .delay1(cellId, _):
                registerCell(cellId, kind: block.kind)
            case let .memoryRead(cellId, _):
                registerCell(cellId, kind: block.kind)
            case let .memoryWrite(cellId, _, _):
                registerCell(cellId, kind: block.kind)
            default:
                break
            }
        }
    }

    if let voiceCellId = voiceCellId {
        registerCell(voiceCellId, kind: .simd)
    }

    // Log cells used in multiple modes
    if !cellUsedInMultipleModes.isEmpty {
        print(
            "[REMAP DEBUG] Cells used in multiple execution modes (upgraded to SIMD): \(cellUsedInMultipleModes)"
        )
    }

    // Second pass: create a remapping for vector cells
    var cellRemapping: [CellID: CellID] = [:]
    var nextAvailableSlot = (allCellIds.max() ?? -1) + 1

    // Reserve space for vector operations and large buffers (sorted for deterministic allocation)
    for (cellId, kind) in memoryUsage.sorted(by: { $0.key < $1.key }) {
        // Check if this cell has a custom allocation size (like spectral scratch buffer)
        let allocSize = cellSizes[cellId] ?? 1

        if kind == .simd {
            // Find a safe starting position (aligned to 4 and not conflicting)
            let alignedSlot = ((nextAvailableSlot + 3) / 4) * 4  // Align to 4-byte boundary
            cellRemapping[cellId] = alignedSlot
            // Reserve space for the full allocation size (e.g., spectral scratch needs windowSize * frameCount * 2)
            nextAvailableSlot = alignedSlot + max(4, allocSize)
        } else {
            // Scalar operations: use allocation size if specified, otherwise keep original slot
            if allocSize > 1 {
                // Large scalar buffer (shouldn't happen for spectral, but handle it)
                cellRemapping[cellId] = nextAvailableSlot
                nextAvailableSlot += allocSize
            } else {
                // Single scalar cell keeps its original slot
                cellRemapping[cellId] = cellId
            }
        }
    }

    // Third pass: apply the remapping to all UOps
    for blockIndex in 0..<uopBlocks.count {
        for uopIndex in 0..<uopBlocks[blockIndex].ops.count {
            let uop = uopBlocks[blockIndex].ops[uopIndex]

            switch uop.op {
            case let .load(cellId):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .load(newCellId),
                        value: uop.value,
                        kind: uop.kind
                    )
                }
            case let .store(cellId, val):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .store(newCellId, val),
                        value: uop.value,
                        kind: uop.kind
                    )
                }
            case let .delay1(cellId, a):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .delay1(newCellId, a),
                        value: uop.value,
                        kind: uop.kind
                    )
                }
            case let .memoryRead(cellId, offset):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .memoryRead(newCellId, offset),
                        value: uop.value,
                        kind: uop.kind
                    )
                }
            case let .memoryWrite(cellId, offset, value):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .memoryWrite(newCellId, offset, value),
                        value: uop.value,
                        kind: uop.kind
                    )
                }
            // No remapping needed for tape-based spectral ops
            default:
                break
            }
        }
    }

    print("[REMAP DEBUG] Cell allocation sizes: \(cellSizes)")
    print(
        "[REMAP DEBUG] Cell remapping for cells 3, 68: \(cellRemapping[3] ?? -1), \(cellRemapping[68] ?? -1)"
    )
    print("[REMAP DEBUG] Total memory slots needed: \(nextAvailableSlot)")

    let cellAllocations = CellAllocations(
        totalMemorySlots: nextAvailableSlot, cellMappings: cellRemapping, cellKinds: memoryUsage)
    return cellAllocations
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
            if !nodesInFeedback.contains(readNodeId) && !nodesInFeedback.contains(writeInfo.nodeId)
            {
                // Replace the historyRead node with historyReadWrite using the write's inputs
                if let readNode = graph.nodes[readNodeId] {
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

func determineTensorBlocks(_ blocks: [Block], _ graph: Graph, _ ctx: IRContext) -> [Block] {
    var determined: [Block] = []

    // Helper to create a new block preserving original properties
    func makeBlock(from original: Block) -> Block {
        var newBlock = Block(kind: original.kind)
        newBlock.direction = original.direction
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

                } else if case let .tensor(shape) = node.shape {
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

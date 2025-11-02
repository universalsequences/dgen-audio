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

        // Step 2: Determine scalar nodes and create blocks

        // Step 2.5: Handle seq operators - if any input to seq is scalar, make all inputs scalar
        var finalScalarSet = scalarNodeSet
        for node in graph.nodes.values {
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

        //let splitBlocks = splitBlocksIfNeeded(blocks, backend)
        var finalBlocks = isolatedBlocks.compactMap { $0 }

        if options.backwards {
            var backwardsBlocks: [Block] = []
            for block in finalBlocks.reversed() {
                var kind: Kind = block.kind

                /*
                for nodeId in block.nodes {
                    if let node = graph.nodes[nodeId] {
                        if case .historyRead(_) = node.op {
                            kind = .scalar
                        } else if case .historyReadWrite(_) = node.op {
                            kind = .scalar
                        } else if case .historyWrite(_) = node.op {
                            kind = .scalar
                        }
                    }
                }

                 */
                var backwardsBlock = Block(kind: kind)
                backwardsBlock.nodes = block.nodes.reversed()
                backwardsBlock.direction = .backwards
                backwardsBlocks.append(backwardsBlock)

            }
            finalBlocks += backwardsBlocks
        }

        let finalBlockIndices = Array(0..<finalBlocks.count)

        // Step 5: Convert blocks to UOp blocks
        let context = IRContext()
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
            uopBlocks.append(BlockUOps(ops: ops, kind: block.kind))
        }

        // Remove empty UOp blocks prior to vector memory remap and lowering
        uopBlocks.removeAll { $0.ops.isEmpty }

        // Step 7: Lower UOp blocks to compiled kernels
        // Ensure a dedicated voice cell exists when voiceCount > 1
        var voiceCellIdFinal: Int? = options.voiceCellId
        if options.voiceCount > 1 && voiceCellIdFinal == nil {
            voiceCellIdFinal = graph.alloc()  // Reserve a cell for voice index
        }

        // Step 6: Fix memory slot conflicts for vector operations
        let cellAllocations = remapVectorMemorySlots(
            &uopBlocks, cellSizes: graph.cellAllocationSizes)

        let renderer: Renderer = createRenderer(for: backend, options: options)
        if let cr = renderer as? CRenderer {
            cr.voiceCount = options.voiceCount
            cr.voiceCellIdOpt = voiceCellIdFinal
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
func remapVectorMemorySlots(_ uopBlocks: inout [BlockUOps], cellSizes: [CellID: Int])
    -> CellAllocations
{
    // Collect all memory operations and their execution modes
    var memoryUsage: [CellID: Kind] = [:]
    var allCellIds: Set<CellID> = []
    var cellUsedInMultipleModes: Set<CellID> = []

    // Helper to register a cell's usage and detect multi-mode access
    func registerCell(_ cellId: CellID, kind: Kind, forceScalar: Bool = false) {
        allCellIds.insert(cellId)

        if forceScalar {
            // scalarMemoryWrite forces scalar mode
            memoryUsage[cellId] = .scalar
            return
        }

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
            case let .scalarMemoryWrite(cellId, _, _):
                registerCell(cellId, kind: block.kind, forceScalar: true)
            default:
                break
            }
        }
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

    // Reserve space for vector operations and large buffers
    for (cellId, kind) in memoryUsage {
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
            case let .scalarMemoryWrite(cellId, offset, value):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .scalarMemoryWrite(newCellId, offset, value),
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

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

        public init(
            frameCount: Int = 128,
            debug: Bool = false,
            printBlockStructure: Bool = false,
            forceScalar: Bool = false
        ) {
            self.frameCount = frameCount
            self.debug = debug
            self.printBlockStructure = printBlockStructure
            self.forceScalar = forceScalar
        }
    }

    /// Compile a graph with the specified backend and options
    public static func compile(
        graph: Graph,
        backend: Backend,
        options: Options = Options()
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
                    print(
                        "🔗 Seq node \(node.id) has scalar input, marking all inputs as scalar: \(node.inputs)"
                    )
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
        // and nodes within blocks are already in correct topological order
        let finalBlocks = blocks
        let finalBlockIndices = Array(0..<finalBlocks.count)

        // Step 5: Convert blocks to UOp blocks
        let context = IRContext()
        var uopBlocks = [BlockUOps]()

        for blockIdx in finalBlockIndices {
            if options.debug {
                print("block \(blockIdx)")
            }
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

        // Step 6: Fix memory slot conflicts for vector operations
        let cellAllocations = remapVectorMemorySlots(&uopBlocks)

        // Step 7: Lower UOp blocks to compiled kernels
        let renderer: Renderer = createRenderer(for: backend)
        let kernels = lowerUOpBlocks(
            &uopBlocks,
            renderer: renderer,
            ctx: context,
            frameCount: options.frameCount,
            graph: graph,
            totalMemorySlots: cellAllocations.totalMemorySlots
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
            cellAllocations: cellAllocations
        )
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

// MARK: - Simplified API

extension CompilationPipeline {
    /// Compile a graph and return just the kernels (for backward compatibility)
    public static func compileToKernels(
        graph: Graph,
        backend: Backend,
        frameCount: Int = 128,
        debug: Bool = false,
        forceScalar: Bool = false
    ) throws -> [CompiledKernel] {
        let result = try compile(
            graph: graph,
            backend: backend,
            options: Options(frameCount: frameCount, debug: debug, forceScalar: forceScalar)
        )
        return result.kernels
    }

    /// Compile a graph and return just the source code
    public static func compileToSource(
        graph: Graph,
        backend: Backend,
        frameCount: Int = 128,
        forceScalar: Bool = false
    ) throws -> String {
        let result = try compile(
            graph: graph,
            backend: backend,
            options: Options(frameCount: frameCount, forceScalar: forceScalar)
        )
        return result.source
    }
}

// MARK: - Memory Layout Remapping

/// Remap memory slots to avoid conflicts between scalar and vector operations
/// Returns the total number of memory slots needed after remapping
func remapVectorMemorySlots(_ uopBlocks: inout [BlockUOps]) -> CellAllocations {
    // Collect all memory operations and their execution modes
    var memoryUsage: [CellID: Kind] = [:]
    var allCellIds: Set<CellID> = []

    // First pass: identify which memory cells are used in which execution modes
    for block in uopBlocks {
        for uop in block.ops {
            switch uop.op {
            case let .load(cellId):
                allCellIds.insert(cellId)
                if memoryUsage[cellId] == nil {
                    memoryUsage[cellId] = block.kind
                } else if memoryUsage[cellId] != block.kind {
                    // Cell used in both scalar and vector - this is a problem
                    print("⚠️  Memory cell \(cellId) used in both scalar and vector modes")
                }
            case let .store(cellId, _):
                allCellIds.insert(cellId)
                if memoryUsage[cellId] == nil {
                    memoryUsage[cellId] = block.kind
                } else if memoryUsage[cellId] != block.kind {
                    print("⚠️  Memory cell \(cellId) used in both scalar and vector modes")
                }
            case let .delay1(cellId, _):
                allCellIds.insert(cellId)
                if memoryUsage[cellId] == nil {
                    memoryUsage[cellId] = block.kind
                } else if memoryUsage[cellId] != block.kind {
                    print("⚠️  Memory cell \(cellId) used in both scalar and vector modes")
                }
            default:
                break
            }
        }
    }

    // Second pass: create a remapping for vector cells
    var cellRemapping: [CellID: CellID] = [:]
    var nextAvailableSlot = (allCellIds.max() ?? -1) + 1

    // Reserve space for vector operations (each vector cell needs 4 slots)
    // Deterministic remapping order: sort by original cellId so C and Metal builds match
    for cellId in Array(memoryUsage.keys).sorted() {
        let kind = memoryUsage[cellId]!
        if kind == .simd {
            // Find a safe starting position (aligned to 4 and not conflicting)
            let alignedSlot = ((nextAvailableSlot + 3) / 4) * 4  // Align to 4-slot boundary
            cellRemapping[cellId] = alignedSlot
            nextAvailableSlot = alignedSlot + 4
            print(
                "🔧 Remapping vector cell \(cellId) -> \(alignedSlot) (uses slots \(alignedSlot)-\(alignedSlot+3))"
            )
        } else {
            // Scalar operations keep their original slots
            cellRemapping[cellId] = cellId
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
            case let .delay1(cellId, val):
                if let newCellId = cellRemapping[cellId] {
                    uopBlocks[blockIndex].ops[uopIndex] = UOp(
                        op: .delay1(newCellId, val),
                        value: uop.value,
                        kind: uop.kind
                    )
                }
            default:
                break
            }
        }
    }

    print("✅ Memory remapping complete. Total memory slots needed: \(nextAvailableSlot)")

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
                    print(
                        "🔄 Combining history pair for cell \(cellId): read node \(readNodeId) + write node \(writeInfo.nodeId) -> historyReadWrite at node \(readNodeId)"
                    )

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

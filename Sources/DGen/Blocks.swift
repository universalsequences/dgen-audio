/// Fuse adjacent blocks of the same kind to reduce cross-block traffic
/// and improve loop fusion opportunities in later stages.
import Foundation

public enum Kind { case simd, scalar }

// corresponds to one kernel (metal backends) or for loop (in C backend)
public struct Block: Equatable {
    public var kind: Kind
    public var nodes: [NodeID] = []
    public var temporality: Temporality = .static_
    public var tensorIndex: Lazy?
    public var shape: Shape?
    /// If set, this block contains a frame-dependent tensor chain that can be
    /// SIMD-parallelized across frames with thread-local tensor storage.
    public var frameTensorChain: FrameDependentTensorChain? = nil

    public init(kind: Kind) {
        self.kind = kind
    }

    public static func == (lhs: Block, rhs: Block) -> Bool {
        // Exclude frameTensorChain from equality to avoid issues with Equatable
        return lhs.kind == rhs.kind && lhs.nodes == rhs.nodes
            && lhs.temporality == rhs.temporality && lhs.tensorIndex == rhs.tensorIndex
            && lhs.shape == rhs.shape
    }
}

// Find all nodes that participate in feedback loops (not just minimal cycles)
public func findFeedbackLoops(_ g: Graph) -> [[NodeID]] {
    // Build maps of history cells to their read/write nodes
    // historyRead/historyWrite now handle both scalar and tensor cases
    var cellReads: [Int: NodeID] = [:]
    var cellWrites: [Int: NodeID] = [:]

    for (nodeId, node) in g.nodes {
        switch node.op {
        case .historyRead(let cell):
            cellReads[cell] = nodeId
        case .historyWrite(let cell):
            cellWrites[cell] = nodeId
        default:
            break
        }
    }

    // Build consumer map for efficient traversal
    var consumers = [NodeID: [NodeID]]()
    for (nodeId, node) in g.nodes {
        for dep in node.allDependencies {
            consumers[dep, default: []].append(nodeId)
        }
    }

    // Helper to find all nodes reachable forward from a set of nodes
    func reachForward(from nodes: Set<NodeID>) -> Set<NodeID> {
        var reached = nodes
        var queue = Array(nodes)

        while let node = queue.popLast() {
            // Add all regular consumers
            for consumer in consumers[node] ?? [] {
                if reached.insert(consumer).inserted {
                    queue.append(consumer)
                }
            }

            // Handle implicit historyWrite -> historyRead connection
            if let n = g.nodes[node] {
                if case .historyWrite(let cellId) = n.op, let readNode = cellReads[cellId] {
                    if reached.insert(readNode).inserted {
                        queue.append(readNode)
                    }
                }
            }
        }

        return reached
    }

    // Helper to find all nodes reachable backward from a set of nodes
    func reachBackward(from nodes: Set<NodeID>) -> Set<NodeID> {
        var reached = nodes
        var queue = Array(nodes)

        while let node = queue.popLast() {
            // Add all regular dependencies
            if let n = g.nodes[node] {
                for dep in n.allDependencies {
                    if reached.insert(dep).inserted {
                        queue.append(dep)
                    }
                }
            }

            // Handle implicit historyRead -> historyWrite connection
            if let n = g.nodes[node] {
                if case .historyRead(let cellId) = n.op, let writeNode = cellWrites[cellId] {
                    if reached.insert(writeNode).inserted {
                        queue.append(writeNode)
                    }
                }
            }
        }

        return reached
    }

    // Find feedback clusters - a cluster exists when any history write depends on any history read
    var clusters: [Set<NodeID>] = []
    var processedWrites = Set<NodeID>()

    for (cellId, writeNode) in cellWrites {
        if processedWrites.contains(writeNode) { continue }

        // Find all nodes this write depends on
        let writeDeps = reachBackward(from: [writeNode])

        // Find all history reads (for this cell) that this write depends on
        let dependentReads = writeDeps.intersection(Set([cellReads[cellId]!]))

        if !dependentReads.isEmpty {
            // This write creates feedback - find all writes reachable from the reads it depends on
            var allReadsInCluster = dependentReads
            var allWritesInCluster = Set([writeNode])

            // OPTIMIZATION: Precompute which nodes can reach writeNode (done once)
            let canReachWriteNode = reachBackward(from: [writeNode])
            let allWriteNodes = Set(cellWrites.values)

            // Keep expanding until we find all connected reads and writes
            var changed = true
            while changed {
                changed = false

                // From all reads in cluster, find what writes they can reach
                let forwardFromReads = reachForward(from: allReadsInCluster)
                let newWrites = forwardFromReads.intersection(allWriteNodes)

                // OPTIMIZATION: Instead of reachForward per write, just check precomputed set
                for newWrite in newWrites {
                    if canReachWriteNode.contains(newWrite) {
                        if allWritesInCluster.insert(newWrite).inserted {
                            changed = true
                        }
                    }
                }
            }

            // Mark all writes in this cluster as processed
            processedWrites.formUnion(allWritesInCluster)

            // Find ALL nodes that participate in the feedback computation
            // These are all nodes that are on any path from any read to any write in the cluster
            var clusterNodes = Set<NodeID>()

            // Add all reads and writes in the cluster
            clusterNodes.formUnion(allReadsInCluster)
            clusterNodes.formUnion(allWritesInCluster)

            // CRITICAL: Ensure read/write pairs for the same cell are both included
            // If a read is in the cluster, its corresponding write must also be included
            for readNode in allReadsInCluster {
                if let node = g.nodes[readNode] {
                    if case .historyRead(let cell) = node.op, let writeNode = cellWrites[cell] {
                        clusterNodes.insert(writeNode)
                        allWritesInCluster.insert(writeNode)
                    }
                }
            }

            // If a write is in the cluster, its corresponding read must also be included
            for writeNode in allWritesInCluster {
                if let node = g.nodes[writeNode] {
                    if case .historyWrite(let cell) = node.op, let readNode = cellReads[cell] {
                        clusterNodes.insert(readNode)
                        allReadsInCluster.insert(readNode)
                    }
                }
            }

            // For each read in the cluster, find all nodes that can reach any write in the cluster
            // OPTIMIZATION: Instead of checking reachability per-node (O(N²)), compute:
            // 1. All nodes reachable forward from reads (done once)
            // 2. All nodes that can reach writes backward (done once)
            // 3. Intersection gives nodes on paths from reads to writes
            let forwardFromReadsAll = reachForward(from: allReadsInCluster)
            let backwardToWrites = reachBackward(from: allWritesInCluster)

            // Nodes on feedback paths = reachable from reads AND can reach writes
            let nodesOnFeedbackPaths = forwardFromReadsAll.intersection(backwardToWrites)
            clusterNodes.formUnion(nodesOnFeedbackPaths)

            clusters.append(clusterNodes)
        }
    }

    // Convert sets to arrays for compatibility
    return clusters.map { Array($0).sorted() }
}

// Determines scalar corridors - groups of scalar nodes that must execute together
// Returns a mapping from nodeId to corridor index (-1 means not in a corridor)
public func determineScalarCorridors(_ g: Graph, feedbackClusters: [[NodeID]]) -> [NodeID: Int] {
    var nodeToCorridorId: [NodeID: Int] = [:]
    var nextCorridorId = 0

    // Use findFeedbackLoops to identify all nodes in feedback loops
    let feedbackLoops = feedbackClusters

    // Assign corridor IDs to nodes in feedback loops
    for loop in feedbackLoops {
        let corridorId = nextCorridorId
        nextCorridorId += 1

        for nodeId in loop {
            nodeToCorridorId[nodeId] = corridorId
        }
    }

    // Also handle other memory operations that aren't in feedback loops
    var reads = [Int: [NodeID]]()
    var writes = [Int: [NodeID]]()

    g.nodes.values.forEach {
        switch $0.op {
        case .historyRead(let c):
            reads[c, default: []].append($0.id)
        case .historyWrite(let c):
            writes[c, default: []].append($0.id)
        case .memoryRead(let c):
            reads[c, default: []].append($0.id)
        case .memoryWrite(let c):
            writes[c, default: []].append($0.id)
        case .accum(let c):
            writes[c, default: []].append($0.id)
        case .latch(let c):
            writes[c, default: []].append($0.id)
        case .phasor(let c):
            writes[c, default: []].append($0.id)
        default: break
        }
    }

    // Assign -1 to nodes not in any corridor
    for nodeId in g.nodes.keys {
        if nodeToCorridorId[nodeId] == nil {
            nodeToCorridorId[nodeId] = -1
        }
    }

    return nodeToCorridorId
}

public func scalarNodes(_ g: Graph, feedbackClusters: [[NodeID]], backend: Backend = .metal) -> Set<
    NodeID
> {
    var scalar: Set<NodeID> = []

    // Track nodes that are SIMD-safe due to using atomics (should never be scalar)
    var simdSafe: Set<NodeID> = []
    g.nodes.values.forEach {
        switch $0.op {
        case .memoryAccumulate(_):
            // memoryAccumulate uses atomics - safe for SIMD execution
            simdSafe.insert($0.id)
        case .tensorAccumulate(_):
            // tensorAccumulate uses atomics internally - safe for SIMD execution
            simdSafe.insert($0.id)
        default: break
        }
    }

    // First, mark inherently scalar operations (stateful ops with frame-to-frame dependencies)
    g.nodes.values.forEach {
        switch $0.op {
        case .accum(_):
            scalar.insert($0.id)  // Accum operations need to be scalar (stateful)
        case .latch(_):
            scalar.insert($0.id)  // Latch operations need to be scalar (stateful)
        case .phasor(_):
            scalar.insert($0.id)  // Phasor operations need to be scalar (stateful)
        default: break
        }
    }

    // Identify tensor source nodes for tracking
    var tensorNodes: Set<NodeID> = []
    g.nodes.values.forEach {
        switch $0.op {
        case .tensorRef(_):
            tensorNodes.insert($0.id)
        case .historyRead(let cellId):
            if g.cellToTensor[cellId] != nil {
                tensorNodes.insert($0.id)
            }
        default: break
        }
    }

    // Propagate tensor status through the graph (needed for both backends)
    var changed = true
    while changed {
        changed = false
        g.nodes.values.forEach { node in
            for inputId in node.inputs {
                if tensorNodes.contains(inputId) && !tensorNodes.contains(node.id) {
                    tensorNodes.insert(node.id)
                    changed = true
                }
            }
        }
    }

    // For C backend: Mark tensor ops as scalar (they need sequential element loops)
    // For Metal: Tensor ops can run SIMD across frames with internal parallelRange loops
    if case .c = backend {
        // C backend: conservative approach - tensor ops are scalar
        g.nodes.values.forEach {
            switch $0.op {
            case .historyRead(let cellId):
                if g.cellToTensor[cellId] != nil {
                    scalar.insert($0.id)
                }
            case .historyWrite(let cellId):
                if g.cellToTensor[cellId] != nil {
                    scalar.insert($0.id)
                }
            case .conv2d(_), .sum, .peek:
                scalar.insert($0.id)
            default: break
            }
        }

        // Mark nodes with tensor inputs as scalar for C
        g.nodes.values.forEach {
            if simdSafe.contains($0.id) { return }
            for inputId in $0.inputs {
                if tensorNodes.contains(inputId) {
                    scalar.insert($0.id)
                }
            }
        }

        // Propagate scalar status through tensor chains for C
        changed = true
        while changed {
            changed = false
            g.nodes.values.forEach { node in
                if simdSafe.contains(node.id) { return }
                for inputId in node.inputs {
                    if tensorNodes.contains(inputId) && !scalar.contains(node.id) {
                        scalar.insert(node.id)
                        changed = true
                    }
                }
            }
        }
    }

    // Use feedback loop detection to mark all nodes in feedback loops as scalar
    // This is the core reason for scalar execution - frame-to-frame state dependencies
    for loop in feedbackClusters {
        for nodeId in loop {
            scalar.insert(nodeId)
        }
    }

    // Remove SIMD-safe nodes from scalar set - these use atomics and are safe for parallel execution
    scalar.subtract(simdSafe)

    return scalar
}

// Simplified corridor-aware topological sort
public func topoWithCorridors(
    _ g: Graph, feedbackClusters: [[NodeID]], scalarNodeSet: Set<NodeID>, debug: Bool = false
) -> [NodeID] {
    let nodeToCorridorId = determineScalarCorridors(g, feedbackClusters: feedbackClusters)

    // Handle seq operators - if any input to seq is scalar, make all inputs scalar
    var finalScalarSet = scalarNodeSet

    // Group nodes by corridor (much simpler grouping)
    var corridorToNodes: [Int: [NodeID]] = [:]
    var nodeToFinalCorridorId: [NodeID: Int] = [:]
    var nextCorridorId = 1000

    for nodeId in g.nodes.keys.sorted() {
        let corridorId = nodeToCorridorId[nodeId] ?? -1
        if corridorId != -1 {
            // Node is in a real corridor
            corridorToNodes[corridorId, default: []].append(nodeId)
            nodeToFinalCorridorId[nodeId] = corridorId
        } else {
            // Node gets its own corridor
            corridorToNodes[nextCorridorId] = [nodeId]
            nodeToFinalCorridorId[nodeId] = nextCorridorId
            nextCorridorId += 1
        }
    }

    if debug {
        print("🏗️ Simplified corridors:")
        for (corridorId, nodes) in corridorToNodes.sorted(by: { $0.key < $1.key })
        where corridorId < 1000 {
            print("  Corridor \(corridorId): \(nodes)")
        }
    }

    // Build corridor-level dependencies (simplified)
    var corridorDeps: [Int: Set<Int>] = [:]
    var corridorIndegree: [Int: Int] = [:]

    // Initialize
    for corridorId in corridorToNodes.keys {
        corridorDeps[corridorId] = []
        corridorIndegree[corridorId] = 0
    }

    // Calculate dependencies between corridors
    for (corridorId, nodes) in corridorToNodes {
        var depCorridors = Set<Int>()
        for nodeId in nodes {
            if let node = g.nodes[nodeId] {
                for dep in node.allDependencies {
                    if let depCorridorId = nodeToFinalCorridorId[dep] {
                        if depCorridorId != corridorId {
                            depCorridors.insert(depCorridorId)
                        }
                    } else {
                    }
                }
            }
        }
        corridorDeps[corridorId] = depCorridors
        corridorIndegree[corridorId] = depCorridors.count
    }

    // Simple topological sort on corridors (Kahn's algorithm)
    var queue = corridorIndegree.filter { $0.value == 0 }.map { $0.key }.sorted()
    var sortedCorridors: [Int] = []

    while let corridorId = queue.first {
        queue.removeFirst()
        sortedCorridors.append(corridorId)

        // Update consumers
        for (otherCorridorId, deps) in corridorDeps {
            if deps.contains(corridorId) {
                corridorIndegree[otherCorridorId]! -= 1
                if corridorIndegree[otherCorridorId] == 0 {
                    // Insert in sorted order for determinism
                    let insertIndex = queue.firstIndex { $0 > otherCorridorId } ?? queue.count
                    queue.insert(otherCorridorId, at: insertIndex)
                }
            }
        }
    }

    // Handle cycles (simple fallback: add remaining in sorted order)
    let remaining = Set(corridorToNodes.keys).subtracting(sortedCorridors)
    if !remaining.isEmpty {
        if debug {
            print(
                "⚠️ Cycle detected, adding remaining corridors in sorted order: \(remaining.sorted())"
            )
        }
        sortedCorridors.append(contentsOf: remaining.sorted())
    }

    // Expand corridors back to nodes
    var result: [NodeID] = []
    for corridorId in sortedCorridors {
        let nodes = corridorToNodes[corridorId]!

        if nodes.count == 1 {
            result.append(nodes[0])
        } else {
            // Simple topological sort within corridor
            result.append(contentsOf: simpleTopoSortWithinCorridor(nodes: nodes, g: g))
        }
    }

    if debug {
        let nodeDescriptions = result.map { nodeId in
            let op = g.nodes[nodeId]?.op ?? .add
            let isScalar = finalScalarSet.contains(nodeId)
            let corridorId = nodeToFinalCorridorId[nodeId] ?? -1
            let color = isScalar ? ANSI.red : ""
            let reset = isScalar ? ANSI.reset : ""
            let corridorMarker = corridorId < 1000 ? "~\(corridorId)" : ""
            return "\(color)(\(nodeId),\(op)\(corridorMarker))\(reset)"
        }
        print("📋 Simplified topo sort: [\(nodeDescriptions.joined(separator: " | "))]")
    }

    return result
}

/// Topological sort within a corridor that separates forward and gradient nodes.
/// If lastForwardNodeId is set, forward nodes (id <= lastForwardNodeId) are processed
/// before gradient nodes (id > lastForwardNodeId) to prevent gradient additions from
/// affecting forward node ordering.
private func simpleTopoSortWithinCorridor(nodes: [NodeID], g: Graph) -> [NodeID] {
    let lastForwardId = g.lastForwardNodeId

    // Separate forward and gradient nodes if applicable
    let forwardNodes: [NodeID]
    let gradientNodes: [NodeID]
    if let lastFwd = lastForwardId {
        forwardNodes = nodes.filter { $0 <= lastFwd }.sorted()
        gradientNodes = nodes.filter { $0 > lastFwd }.sorted()
    } else {
        forwardNodes = nodes.sorted()
        gradientNodes = []
    }

    // Topological sort a subset of nodes using Kahn's algorithm
    func topoSort(_ subsetNodes: [NodeID]) -> [NodeID] {
        guard !subsetNodes.isEmpty else { return [] }
        let subsetSet = Set(subsetNodes)
        var indegree: [NodeID: Int] = [:]

        // Calculate in-degrees (only counting deps within the subset)
        for nodeId in subsetNodes {
            var count = 0
            if let node = g.nodes[nodeId] {
                for dep in node.allDependencies where subsetSet.contains(dep) {
                    count += 1
                }
            }
            indegree[nodeId] = count
        }

        // Kahn's algorithm
        var queue = indegree.filter { $0.value == 0 }.map { $0.key }.sorted()
        var result: [NodeID] = []

        while let nodeId = queue.first {
            queue.removeFirst()
            result.append(nodeId)

            // Update consumers within subset
            for otherNodeId in subsetNodes {
                if let otherNode = g.nodes[otherNodeId],
                    otherNode.allDependencies.contains(nodeId)
                {
                    indegree[otherNodeId]! -= 1
                    if indegree[otherNodeId] == 0 {
                        let insertIndex = queue.firstIndex { $0 > otherNodeId } ?? queue.count
                        queue.insert(otherNodeId, at: insertIndex)
                    }
                }
            }
        }

        // Add any remaining nodes (cycles)
        let remaining = subsetSet.subtracting(result)
        result.append(contentsOf: remaining.sorted())

        return result
    }

    // Sort forward nodes first, then gradient nodes
    var result = topoSort(forwardNodes)
    result.append(contentsOf: topoSort(gradientNodes))

    return result
}

// Helper function to check if adding a node to a block would exceed the node limit
func wouldExceedNodeLimit(_ block: Block, maxNodesPerBlock: Int) -> Bool {
    return block.nodes.count >= maxNodesPerBlock
}

// Simplified block determination that works with corridor-aware sorted nodes
public func determineBlocksSimple(
    sorted: [NodeID], scalar: Set<NodeID>, g: Graph, maxNodesPerBlock: Int = Int.max,
    debug: Bool = false
) -> [Block] {
    var blocks: [Block] = []
    var currentBlock: Block? = nil
    var currentBlockHasPass1 = false  // Track spectral loss passes in current block
    var currentBlockHasPass2 = false

    for nodeId in sorted {
        let isScalar = scalar.contains(nodeId)
        let kind: Kind = isScalar ? .scalar : .simd

        // Special handling for output nodes - they go in the same block as their dependencies
        if let node = g.nodes[nodeId], case .output = node.op {
            // Find the block containing the first dependency
            var targetBlockIdx = -1
            for inputID in node.allDependencies {
                for (blockIdx, block) in blocks.enumerated() {
                    if block.nodes.contains(inputID) {
                        targetBlockIdx = blockIdx
                        break
                    }
                }
                if targetBlockIdx != -1 { break }
            }

            // Check if we can add to an existing block that contains the dependency
            var addedToExistingBlock = false
            if targetBlockIdx != -1 && targetBlockIdx < blocks.count {
                if !wouldExceedNodeLimit(blocks[targetBlockIdx], maxNodesPerBlock: maxNodesPerBlock)
                {
                    blocks[targetBlockIdx].nodes.append(nodeId)
                    addedToExistingBlock = true
                    if debug {
                        print(
                            "📍 Placed output node \(nodeId) in block \(targetBlockIdx) with its dependency"
                        )
                    }
                }
            }

            if !addedToExistingBlock {
                // If we have a current block and the dependency isn't found yet,
                // the dependency is likely in the current block
                if let current = currentBlock,
                    !wouldExceedNodeLimit(current, maxNodesPerBlock: maxNodesPerBlock)
                {
                    currentBlock!.nodes.append(nodeId)
                    if debug {
                        print("📍 Placed output node \(nodeId) in current block")
                    }
                } else {
                    // Last resort - finish current block and start new one
                    if let current = currentBlock {
                        blocks.append(current)
                    }
                    currentBlock = Block(kind: kind)
                    currentBlock!.nodes.append(nodeId)
                }
            }
            continue
        }

        // Regular node handling - group consecutive nodes of same kind together
        if let current = currentBlock {
            // Check if this node must be separated from nodes in the current block
            var mustSeparate = false

            // Spectral loss two-pass: Pass1 and Pass2 must be in different blocks
            // Check using tracked flags instead of scanning all nodes (O(1) instead of O(n))
            var nodeIsPass1 = false
            var nodeIsPass2 = false
            if let node = g.nodes[nodeId] {
                if case .spectralLossPass1 = node.op { nodeIsPass1 = true }
                if case .spectralLossPass2 = node.op { nodeIsPass2 = true }
            }

            // Don't allow Pass1 and Pass2 in the same block
            if (nodeIsPass1 && currentBlockHasPass2) || (nodeIsPass2 && currentBlockHasPass1) {
                mustSeparate = true
            }

            if current.kind == kind
                && !wouldExceedNodeLimit(current, maxNodesPerBlock: maxNodesPerBlock)
                && !mustSeparate
            {
                // Add to current block and update flags
                currentBlock!.nodes.append(nodeId)
                if nodeIsPass1 { currentBlockHasPass1 = true }
                if nodeIsPass2 { currentBlockHasPass2 = true }
            } else {
                // Finish current block and start new one
                blocks.append(current)
                currentBlock = Block(kind: kind)
                currentBlock!.nodes.append(nodeId)
                // Reset flags for new block
                currentBlockHasPass1 = nodeIsPass1
                currentBlockHasPass2 = nodeIsPass2
            }
        } else {
            // Start first block
            currentBlock = Block(kind: kind)
            currentBlock!.nodes.append(nodeId)
            // Set flags for first node
            if let node = g.nodes[nodeId] {
                if case .spectralLossPass1 = node.op { currentBlockHasPass1 = true }
                if case .spectralLossPass2 = node.op { currentBlockHasPass2 = true }
            }
        }
    }

    // Don't forget the last block
    if let current = currentBlock {
        blocks.append(current)
    }

    // Remove empty blocks (can arise from special placement of outputs or limits)
    blocks.removeAll { $0.nodes.isEmpty }

    if debug {
        print("📦 Created \(blocks.count) blocks with simplified logic")
        for (i, block) in blocks.enumerated() {
            print("  Block \(i) (\(block.kind)): \(block.nodes)")
        }
    }

    return blocks
}

public func fuseBlocks(_ blocks: [Block], _ g: Graph) -> [Block] {
    let debugFuse = (ProcessInfo.processInfo.environment["DGEN_DEBUG_FUSE"] == "1")
    if debugFuse {
        print("[FUSE] Input blocks: \(blocks.count)")
        let signature = blocks.map { $0.kind == .simd ? "S" : "C" }.joined()
        print("[FUSE] Kinds: \(signature)")
    }
    // Pre-compute which blocks contain Pass1/Pass2 and which scratch cells they touch
    var blockHasPass1: [Bool] = []
    var blockHasPass2: [Bool] = []
    var blockHasScaledThreads: [Bool] = []
    var blockPass1Cells: [Set<CellID>] = []
    var blockPass2Cells: [Set<CellID>] = []

    for b in blocks {
        var hasPass1 = false
        var hasPass2 = false
        var hasScaled = false
        var p1Cells = Set<CellID>()
        var p2Cells = Set<CellID>()
        for nodeId in b.nodes {
            if let node = g.nodes[nodeId] {
                switch node.op {
                case .spectralLossPass1(_, let scratchCell):
                    hasPass1 = true
                    p1Cells.insert(scratchCell)
                    hasScaled = true
                case .spectralLossPass2(_, let scratchCell):
                    hasPass2 = true
                    p2Cells.insert(scratchCell)
                case .parallelMap2DTestPass1(_, _):
                    hasScaled = true
                case .parallelMap2DTestPass2(_, _):
                    // no-op for scaled flag
                    break
                default:
                    break
                }
            }
        }
        blockHasPass1.append(hasPass1)
        blockHasPass2.append(hasPass2)
        blockHasScaledThreads.append(hasScaled)
        blockPass1Cells.append(p1Cells)
        blockPass2Cells.append(p2Cells)
    }

    var fused: [Block] = []
    var fusedHasPass1: [Bool] = []
    var fusedHasPass2: [Bool] = []
    var fusedHasScaledThreads: [Bool] = []
    var fusedPass1Cells: [Set<CellID>] = []
    var fusedPass2Cells: [Set<CellID>] = []

    for (idx, b) in blocks.enumerated() {
        if b.nodes.isEmpty { continue }
        if let lastIdx = fused.indices.last, fused[lastIdx].kind == b.kind {
            // Check if we should prevent fusion due to spectral loss two-pass
            // Only prevent fusion if Pass1 and Pass2 touch the SAME scratch cell across the boundary
            var conflict = false
            if fusedHasPass1[lastIdx] && blockHasPass2[idx] {
                conflict = !fusedPass1Cells[lastIdx].intersection(blockPass2Cells[idx]).isEmpty
            }
            if !conflict && fusedHasPass2[lastIdx] && blockHasPass1[idx] {
                conflict = !fusedPass2Cells[lastIdx].intersection(blockPass1Cells[idx]).isEmpty
            }
            if !conflict && (fusedHasScaledThreads[lastIdx] || blockHasScaledThreads[idx]) {
                conflict = true
            }
            let canFuse = !conflict

            if canFuse {
                if debugFuse {
                    print("[FUSE] merge idx=\(idx) into last=\(lastIdx) kind=\(b.kind)")
                }
                fused[lastIdx].nodes.append(contentsOf: b.nodes)
                // Update flags
                fusedHasPass1[lastIdx] = fusedHasPass1[lastIdx] || blockHasPass1[idx]
                fusedHasPass2[lastIdx] = fusedHasPass2[lastIdx] || blockHasPass2[idx]
                fusedHasScaledThreads[lastIdx] =
                    fusedHasScaledThreads[lastIdx] || blockHasScaledThreads[idx]
                fusedPass1Cells[lastIdx].formUnion(blockPass1Cells[idx])
                fusedPass2Cells[lastIdx].formUnion(blockPass2Cells[idx])
            } else {
                if debugFuse {
                    let common12 = fusedPass1Cells[lastIdx].intersection(blockPass2Cells[idx])
                    let common21 = fusedPass2Cells[lastIdx].intersection(blockPass1Cells[idx])
                    print(
                        "[FUSE] prevent (spectral cell conflict) last=\(lastIdx) P1cells=\(fusedPass1Cells[lastIdx]) P2cells=\(fusedPass2Cells[lastIdx]) | idx=\(idx) P1cells=\(blockPass1Cells[idx]) P2cells=\(blockPass2Cells[idx]) common12=\(common12) common21=\(common21)"
                    )
                }
                fused.append(b)
                fusedHasPass1.append(blockHasPass1[idx])
                fusedHasPass2.append(blockHasPass2[idx])
                fusedHasScaledThreads.append(blockHasScaledThreads[idx])
                fusedPass1Cells.append(blockPass1Cells[idx])
                fusedPass2Cells.append(blockPass2Cells[idx])
            }
        } else {
            if debugFuse {
                if let last = fused.last {
                    print("[FUSE] new run at idx=\(idx) prevKind=\(last.kind) newKind=\(b.kind)")
                } else {
                    print("[FUSE] start with idx=\(idx) kind=\(b.kind)")
                }
            }
            fused.append(b)
            fusedHasPass1.append(blockHasPass1[idx])
            fusedHasPass2.append(blockHasPass2[idx])
            fusedHasScaledThreads.append(blockHasScaledThreads[idx])
            fusedPass1Cells.append(blockPass1Cells[idx])
            fusedPass2Cells.append(blockPass2Cells[idx])
        }
    }
    if debugFuse {
        print("[FUSE] Output blocks: \(fused.count)")
        let signature = fused.map { $0.kind == .simd ? "S" : "C" }.joined()
        print("[FUSE] Fused kinds: \(signature)")
    }
    return fused
}

/// Isolate spectralLossPass1 and spectralLossPass2 into their own blocks
/// to ensure they execute as separate kernels without any fused operations.
/// Preserves ordering of other nodes.
public func isolateSpectralPasses(_ blocks: [Block], _ g: Graph) -> [Block] {
    var result: [Block] = []

    for block in blocks {
        var currentNodes: [NodeID] = []

        for nodeId in block.nodes {
            let isSpectralPass = { () -> Bool in
                guard let node = g.nodes[nodeId] else { return false }
                if case .spectralLossPass1 = node.op { return true }
                if case .spectralLossPass2 = node.op { return true }
                if case .parallelMap2DTestPass1 = node.op { return true }
                if case .parallelMap2DTestPass2 = node.op { return true }
                return false
            }()

            if isSpectralPass {
                // Flush any accumulated nodes before the spectral pass
                if !currentNodes.isEmpty {
                    var newBlock = Block(kind: block.kind)
                    newBlock.nodes = currentNodes
                    result.append(newBlock)
                    currentNodes = []
                }

                // Add spectral pass in its own block
                var spectralBlock = Block(kind: block.kind)
                spectralBlock.nodes = [nodeId]
                result.append(spectralBlock)
            } else {
                // Accumulate non-spectral nodes
                currentNodes.append(nodeId)
            }
        }

        // Flush any remaining nodes after the last spectral pass
        if !currentNodes.isEmpty {
            var newBlock = Block(kind: block.kind)
            newBlock.nodes = currentNodes
            result.append(newBlock)
        }
    }

    return result
}

public func splitBlocksIfNeeded(_ blocks: [Block], backend: Backend) -> [Block] {
    if case .c = backend {
        return blocks
    }

    var split: [Block] = []
    for b in blocks {
        if b.kind == .simd {
            // only split simd since scalar blocks must be executed together
        } else {
            split.append(b)
        }
    }
    return split
}

extension LazyOp {
    var isOutput: Bool {
        if case .output = self { return true }
        return false
    }
}

public func findOutputNodeNeeds(_ b: Block, _ g: Graph) -> Set<NodeID> {
    var outputNodeNeeds: Set<NodeID> = []
    for nID in b.nodes {
        if let n = g.nodes[nID] {
            switch n.op {
            case .output:
                n.inputs.forEach {
                    outputNodeNeeds.insert($0)
                }
            default:
                break
            }
        }
    }
    return outputNodeNeeds
}

/// Find tensor nodes with outbound dependencies in frame-based blocks.
/// These tensors need frame-aware allocation (tensorSize × frameCount cells)
/// to enable parallelization across frames without race conditions.
public func findFrameAwareTensors(_ blocks: [Block], _ g: Graph) -> Set<NodeID> {
    var result = Set<NodeID>()
    for block in blocks where block.temporality == .frameBased {
        // Only consider blocks with tensor shapes (tensor SIMD blocks)
        guard block.shape != nil else { continue }

        for nodeId in findNodesWithOutboundDependencies(blocks, g, block: block) {
            if let node = g.nodes[nodeId], case .tensor = node.shape {
                result.insert(nodeId)
            }
        }
    }
    return result
}

// ─── 3. decide which nodes need cross-block scratch buffers ─────
// in the case of metal, these are transmitted via buffers
public func findNodesWithOutboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> [NodeID]
{
    // Map node -> block index
    var nodeBlock = [NodeID: Int]()
    for (bidx, b) in blks.enumerated() {
        b.nodes.forEach { nid in
            if nodeBlock[nid] == nil {
                nodeBlock[nid] = bidx
            }
        }
    }

    guard let thisIdx = blks.firstIndex(of: block) else { return [] }

    // Compute contiguous-kind groups to enable within-group fusion
    /*
    var groupForBlock = Array(repeating: 0, count: blks.count)
    var group = 0
    for i in 0..<blks.count {
        if i > 0 && blks[i].kind != blks[i - 1].kind { group += 1 }
        groupForBlock[i] = group
    }
    let thisGroup = groupForBlock[thisIdx]
     */

    var need: Set<NodeID> = []
    for (consumerIdx, b) in blks.enumerated() {
        if thisIdx == consumerIdx { continue }
        // Only consider consumers in other groups; within-group we can keep values in registers
        //if groupForBlock[consumerIdx] == thisGroup { continue }
        for nID in b.nodes {
            g.nodes[nID]!.allDependencies.forEach { dep in
                if let producerIdx = nodeBlock[dep], producerIdx == thisIdx {
                    if let depNode = g.nodes[dep], case .seq = depNode.op {
                        if let lastInput = depNode.inputs.last { need.insert(lastInput) }
                    } else {
                        need.insert(dep)
                    }
                }
            }
        }
    }
    return need.sorted()  // Return sorted array for stable ordering
}

public func findNodesAsInboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> [NodeID] {
    guard let thisIdx = blks.firstIndex(of: block) else { return [] }

    // Map node -> block index
    var nodeBlock = [NodeID: Int]()
    for (bidx, b) in blks.enumerated() {
        b.nodes.forEach { nid in
            if nodeBlock[nid] == nil {
                nodeBlock[nid] = bidx
            }
        }
    }
    //for (bidx, b) in blks.enumerated() { b.nodes.forEach { nodeBlock[$0] = bidx } }

    // Compute contiguous-kind groups
    /*
    var groupForBlock = Array(repeating: 0, count: blks.count)
    var group = 0
    for i in 0..<blks.count {
        if i > 0 && blks[i].kind != blks[i - 1].kind { group += 1 }
        groupForBlock[i] = group
    }
    let thisGroup = groupForBlock[thisIdx]
     */

    var need: Set<NodeID> = []
    // Collect only dependencies produced in a different group
    for nID in block.nodes {
        g.nodes[nID]!.allDependencies.forEach { dep in
            if let prodIdx = nodeBlock[dep] {
                if prodIdx != thisIdx { need.insert(dep) }
            }
        }
    }
    return need.sorted()  // Return sorted array for stable ordering
}

/// Compute which tensor cells in this block need to be written to memory because
/// they're used by later blocks. Cells only used within this block stay in registers.
public func findOutboundTensorCells(_ blks: [Block], _ g: Graph, block: Block) -> Set<CellID> {
    guard let thisIdx = blks.firstIndex(of: block) else { return [] }

    // Collect all tensor cells produced by nodes in this block
    var producedCells: Set<CellID> = []
    for nodeId in block.nodes {
        if let node = g.nodes[nodeId], case .tensor = node.shape {
            if let tensorId = g.nodeToTensor[nodeId], let tensor = g.tensors[tensorId] {
                producedCells.insert(tensor.cellId)
            }
        }
    }

    // Check which cells are consumed by later blocks
    var outboundCells: Set<CellID> = []
    for (blockIdx, b) in blks.enumerated() {
        if blockIdx <= thisIdx { continue }  // Only look at later blocks

        for nodeId in b.nodes {
            guard let node = g.nodes[nodeId] else { continue }

            // Check if this node reads from any of our produced cells
            for inputId in node.inputs {
                if let inputNode = g.nodes[inputId], case .tensor = inputNode.shape {
                    if let tensorId = g.nodeToTensor[inputId], let tensor = g.tensors[tensorId] {
                        if producedCells.contains(tensor.cellId) {
                            outboundCells.insert(tensor.cellId)
                        }
                    }
                }
            }

            // Also check historyRead/historyWrite operations that reference tensor cells
            switch node.op {
            case .historyRead(let cellId), .historyWrite(let cellId):
                if producedCells.contains(cellId) {
                    outboundCells.insert(cellId)
                }
            default:
                break
            }
        }
    }

    return outboundCells
}

/// Check if any UOps contain patterns that prevent SIMD optimization:
/// - Inner loops (beginLoop, beginForLoop)
/// - View operations (reshape, transpose, shrink) that require complex index arithmetic (C only)
/// - Broadcast access (non-contiguous strides or shape mismatch) (C only)
private func containsSIMDBlockers(_ uops: [UOp], backend: Backend) -> Bool {
    for uop in uops {
        switch uop.op {
        case .beginLoop, .beginForLoop:
            return true
        case .reshape, .transpose, .shrink, .pad:
            // Metal handles these fine with per-thread execution
            if case .c = backend { return true }
        case .broadcastAccess:
            // Metal handles broadcast access fine with per-thread execution
            if case .c = backend { return true }
        case .requiresScalar:
            return true  // Stateful ops (phasor, accum, latch) need sample-by-sample execution
        default:
            break
        }
    }
    return false
}

public func emitThreadCountScaleOpIfNeeded(ctx: IRContext, block: Block, g: Graph) -> [UOp] {
    guard let shape = block.shape else { return [] }

    // todo temporality check (are we frameCount*size or size)

    let tensorSize = shape.reduce(1, *)

    var uops: [UOp] = []
    // Flattened (frame, bin) threading
    let setup = IRBuilder(ctx: ctx, nodeId: block.nodes[block.nodes.count - 1])
    setup.setThreadCountScale(tensorSize)
    let flatIdx = setup.threadIndex()
    let sizeExpr = setup.constant(Float(tensorSize))
    let frameIdx = setup.floor(flatIdx / sizeExpr)
    setup.setFrameIndex(frameIdx)
    let binIdx = flatIdx - frameIdx * sizeExpr
    uops.append(contentsOf: setup.ops)

    // Use binIdx as the tensor index for all nodes in the chain
    for nodeId in block.nodes {
        ctx.tensorIndices[nodeId] = binIdx.lazy
    }
    return uops
}

/// Emit UOps for a frame-dependent tensor chain block.
/// This generates SIMD-across-frames code where each frame/thread:
/// 1. Declares thread-local tensor storage
/// 2. Computes the entire tensor chain using inline loops
/// 3. Outputs the final scalar to frame-indexed scratch buffer
public func emitFrameTensorChainBlock(
    ctx: IRContext, chain: FrameDependentTensorChain, block: Block, g: Graph
) throws -> [UOp] {
    var uops: [UOp] = []

    guard let scratch = ctx.frameTensorChainScratch[chain.reductionNodeId] else {
        return uops
    }

    let tensorSize = scratch.tensorSize

    // Flattened (frame, bin) threading
    let setup = IRBuilder(ctx: ctx, nodeId: chain.reductionNodeId)
    setup.setThreadCountScale(tensorSize)
    let flatIdx = setup.threadIndex()
    let sizeExpr = setup.constant(Float(tensorSize))
    let frameIdx = setup.floor(flatIdx / sizeExpr)
    setup.setFrameIndex(frameIdx)
    let binIdx = flatIdx - frameIdx * sizeExpr
    uops.append(contentsOf: setup.ops)

    // Use binIdx as the tensor index for all nodes in the chain
    for nodeId in block.nodes {
        ctx.tensorIndices[nodeId] = binIdx.lazy
    }

    // Keep chain tensors thread-local (avoid global memory writes)
    let savedOutbound = ctx.outboundTensorCells
    ctx.outboundTensorCells = []
    ctx.clearTensorRegisters()

    let reductionInputId = g.nodes[chain.reductionNodeId]?.inputs.first

    // Emit chain nodes (map step)
    for nodeId in block.nodes {
        if let node = g.nodes[nodeId] {
            if case .selectRow = node.op {
                // Specialized per-bin selectRow: compute only the current column (binIdx)
                guard node.inputs.count == 2 else {
                    throw DGenError.insufficientInputs(
                        operator: "selectRow", expected: 2, actual: node.inputs.count)
                }
                let tensorInput = node.inputs[0]
                guard let inputNode = g.nodes[tensorInput],
                    case .tensor(let shape) = inputNode.shape,
                    shape.count == 2
                else {
                    throw DGenError.tensorError(op: "selectRow", reason: "requires 2D tensor input")
                }
                guard let inTensorId = g.nodeToTensor[tensorInput],
                    let inTensor = g.tensors[inTensorId],
                    let outTensorId = g.nodeToTensor[node.id],
                    let outTensor = g.tensors[outTensorId]
                else {
                    throw DGenError.tensorError(op: "selectRow", reason: "missing tensor")
                }

                let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
                let b = IRBuilder(ctx: ctx, nodeId: nodeId)
                let rowIndex = try b.readInput(node, inputs, at: 1)

                let numRows = shape[0]
                let numRowsFloat = b.constant(Float(numRows))
                let zero = b.constant(0.0)

                // Wrap rowIndex using modulo for wrapping behavior, then floor
                let wrappedIndex = b.mod(rowIndex, numRowsFloat)
                let isNegative = wrappedIndex < zero
                let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)
                let floorIndex = b.floor(positiveIndex)

                // Read the selected row element for this column
                let colOffset = b.value(binIdx.lazy) * numRowsFloat
                let readPos = colOffset + floorIndex
                let value = b.memoryRead(inTensor.cellId, b.cast(readPos, to: .int))

                // Register output tensor element in a register (avoid shared memory writes)
                _ = b.tstore(outTensor.cellId, b.value(binIdx.lazy), value)
                ctx.values[nodeId] = value.lazy
                uops.append(contentsOf: b.ops)
            } else if case .deterministicPhasor = node.op {
                guard node.inputs.count == 1 else {
                    throw DGenError.insufficientInputs(
                        operator: "deterministicPhasor", expected: 1, actual: node.inputs.count)
                }

                let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
                let b = IRBuilder(ctx: ctx, nodeId: nodeId)
                let freq = try b.readInput(node, inputs, at: 0)
                let sampleRate = b.constant(g.sampleRate)
                let frameIdxExpr = b.value(frameIdx.lazy)

                let phaseIncrement = freq / sampleRate
                let rawPhase = phaseIncrement * frameIdxExpr
                let phase = rawPhase - b.floor(rawPhase)

                try b.writeOutput(node, phase)
                ctx.values[nodeId] = phase.lazy
                uops.append(contentsOf: b.ops)
            } else {
                for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
                    uops.append(uop)
                }
            }

            // If this node feeds the reduction, write its value to scratch immediately
            if let reductionInputId, nodeId == reductionInputId,
                let scratch = ctx.frameTensorChainScratch[chain.reductionNodeId],
                let tensorId = g.nodeToTensor[reductionInputId],
                let tensor = g.tensors[tensorId]
            {
                let writer = IRBuilder(ctx: ctx, nodeId: chain.reductionNodeId)
                let frameIdxExpr = writer.value(frameIdx.lazy)
                let binIdxExpr = writer.value(binIdx.lazy)
                let offset =
                    frameIdxExpr * writer.constant(Float(scratch.tensorSize)) + binIdxExpr
                let val = writer.tload(tensor.cellId, binIdxExpr)
                _ = writer.memoryWrite(scratch.cellId, writer.cast(offset, to: .int), val)
                uops.append(contentsOf: writer.ops)
            }
        }
    }

    // Restore outbound tensor tracking
    ctx.outboundTensorCells = savedOutbound

    return uops
}

public func emitBlockUOps(
    ctx: IRContext, block: Block, blocks: [Block], g: Graph, backend: Backend = .metal,
    debug: Bool = false
) throws -> [UOp] {
    var emittedNodes: Set<NodeID> = []
    var bodyUops: [UOp] = []

    // Tensor Register Optimization:
    // Compute which tensor cells need to be written to memory (used by later blocks)
    // and clear the register tracking for this new block.
    ctx.outboundTensorCells = findOutboundTensorCells(blocks, g, block: block)
    ctx.clearTensorRegisters()

    var emitted = false
    for uop in emitThreadCountScaleOpIfNeeded(ctx: ctx, block: block, g: g) {
        emitted = true
        bodyUops.append(uop)
    }
    for nodeId in block.nodes {
        if !emitted {
            if let tensorIndex = block.tensorIndex {
                ctx.tensorIndices[nodeId] = tensorIndex
            }
        }

        if let node = g.nodes[nodeId] {
            for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
                emittedNodes.insert(nodeId)
                bodyUops.append(uop)
            }
        }
    }

    // Step 2: Analyze emitted UOps to determine if SIMD is safe
    // SIMD is safe if: tensor block + size divisible by 4 + no SIMD blockers + not frame-based
    // Note: frame-aware tensor blocks already handle parallelism via flat threading
    let hasSIMDBlockers = containsSIMDBlockers(bodyUops, backend: backend)
    let canUseSIMD: Bool
    let simdIncrement: Int

    let isFrameAwareTensorBlock = false

    if isFrameAwareTensorBlock {
        // Frame-aware blocks use flat threading, no loop needed
        canUseSIMD = false
        simdIncrement = 1
    } else if let shape = block.shape, block.tensorIndex != nil {
        let size = shape.reduce(1, *)
        // Frame-based tensor blocks must run element-by-element per frame
        // because their values change every frame (e.g., downstream of phasor(tensor))
        let isFrameBased = block.temporality == .frameBased
        canUseSIMD = !hasSIMDBlockers && !isFrameBased && (size % 4 == 0)
        simdIncrement = canUseSIMD ? 4 : 1
    } else {
        canUseSIMD = false
        simdIncrement = 1
    }

    // Step 3: Determine the effective kind for this block's ops
    let effectiveKind: Kind
    if isFrameAwareTensorBlock {
        // Frame-aware tensor blocks are always SIMD (flat threading)
        effectiveKind = .simd
    } else if block.tensorIndex != nil {
        effectiveKind = canUseSIMD ? .simd : .scalar
    } else {
        effectiveKind = block.kind
    }

    // Step 4: Apply the kind to all body UOps
    for i in 0..<bodyUops.count {
        bodyUops[i].kind = effectiveKind
    }

    // Step 5: Build final UOps array with parallelRange wrapper if needed
    // Note: frame-aware tensor blocks DON'T use parallelRange (no loop)
    var uops: [UOp] = []

    if backend == .c {
        if !isFrameAwareTensorBlock,
            let tensorIndex = block.tensorIndex,
            let shape = block.shape
        {
            let count = shape.reduce(1, *)
            uops.append(UOp(op: .beginParallelRange(count, simdIncrement), value: tensorIndex))
        }
    }

    uops.append(contentsOf: bodyUops)

    // Handle cross-block dependencies using scratch buffers (for scalar values only)
    //
    // IMPORTANT: Tensor-valued outputs/inputs do NOT use scratch buffers.
    //
    // Why? Scratch buffers are indexed by frame (t<id>[i]), but tensor operations
    // run inside parallel loops where each tensor element has a different value.
    // If we wrote tensor results to scratch buffers inside a tensor loop:
    //   - Each iteration would overwrite the same t<id>[i] location
    //   - Only the LAST tensor element's value would survive
    //   - Reading it back and broadcasting would give wrong values (noise!)
    //
    // Instead, tensor data flows through memory cells which ARE properly indexed
    // by the tensor parallel range index (memory[cellId + tensorIndex]).

    let outbound = findNodesWithOutboundDependencies(blocks, g, block: block)
    for nodeId in outbound {
        if emittedNodes.contains(nodeId) {
            // Skip defineGlobal for tensor-valued outputs - they use memory cells, not scratch buffers
            if let node = g.nodes[nodeId], case .tensor = node.shape {
                continue
            }

            if let lz = ctx.values[nodeId] {
                switch lz {
                case .variable(let a, _):
                    var defineGlobalUOp = UOp(op: .defineGlobal(a), value: .global(a))
                    // Use block.kind (frame loop kind), not effectiveKind (tensor loop kind)
                    // Globals are indexed by frame loop, not tensor loop
                    defineGlobalUOp.kind = block.kind
                    uops.insert(defineGlobalUOp, at: 0)
                    // Only append if not already in globals to maintain stable ordering
                    if !ctx.globals.contains(a) {
                        ctx.globals.append(a)
                    }
                default:
                    break
                }
            }
        }
    }

    let inbound = findNodesAsInboundDependencies(blocks, g, block: block)

    for nodeId in inbound {
        // Skip loadGlobal for tensor-valued inputs - they use memory cells, not scratch buffers
        if let node = g.nodes[nodeId], case .tensor = node.shape {
            continue
        }

        if let lz = ctx.values[nodeId] {
            switch lz {
            case .variable(let a, _):
                var loadGlobalUOp = UOp(op: .loadGlobal(a), value: .variable(a, nil))
                // Use block.kind (frame loop kind), not effectiveKind (tensor loop kind)
                // Globals are indexed by frame loop, not tensor loop
                loadGlobalUOp.kind = block.kind
                uops.insert(loadGlobalUOp, at: 0)
            default:
                break
            }
        }
    }

    // Only emit endParallelRange for non-frame-aware tensor blocks
    if backend == .c {
        if !isFrameAwareTensorBlock && block.tensorIndex != nil {
            uops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
        }
    }

    return uops
}

public func isReductionOp(_ op: LazyOp) -> Bool {
    switch op {
    case .sum:
        return true
    case .sumAxis(_):
        return true
    case .tensorAccumulate(_):
        return true
    default:
        return false
    }
}

public func splitReduceBlocks(g: Graph, blocks: [Block]) -> [Block] {
    var splitBlocks: [Block] = []
    for block in blocks {
        if let reductionOpIndex = block.nodes.firstIndex { nodeId in
            guard let node = g.nodes[nodeId] else { return false }
            return isReductionOp(node.op)
        } {
            if reductionOpIndex > 0 {
                var preReductionBlock = Block(kind: .simd)
                preReductionBlock.nodes = Array(block.nodes[0..<reductionOpIndex])
                preReductionBlock.shape = block.shape
                preReductionBlock.temporality = block.temporality
                splitBlocks.append(preReductionBlock)
            }
            var reductionBlock = Block(kind: .simd)  // still SIMD w.r.t frame count
            reductionBlock.nodes = [block.nodes[reductionOpIndex]]
            reductionBlock.temporality = block.temporality
            splitBlocks.append(reductionBlock)
            if reductionOpIndex < block.nodes.count - 1 {
                var postReductionBlock = Block(kind: .simd)
                postReductionBlock.nodes = Array(
                    block.nodes[reductionOpIndex + 1..<block.nodes.count])
                postReductionBlock.shape = block.shape
                postReductionBlock.temporality = block.temporality
                splitBlocks.append(postReductionBlock)
            }
        } else {
            splitBlocks.append(block)
        }
    }
    return splitBlocks
}

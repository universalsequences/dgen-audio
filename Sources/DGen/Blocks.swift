/// Fuse adjacent blocks of the same kind to reduce cross-block traffic
/// and improve loop fusion opportunities in later stages.
import Foundation

public enum Kind { case simd, scalar }
public enum Direction { case forward, backwards }

// corresponds to one kernel (metal backends) or for loop (in C backend)
public struct Block: Equatable {
    public var kind: Kind
    public var nodes: [NodeID] = []
    public var direction: Direction = .forward
    public var temporality: Temporality = .static_
    public var tensorIndex: Lazy?
    public var shape: Shape?

    public init(kind: Kind) {
        self.kind = kind
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

            // Keep expanding until we find all connected reads and writes
            var changed = true
            while changed {
                changed = false

                // From all reads in cluster, find what writes they can reach
                let forwardFromReads = reachForward(from: allReadsInCluster)
                // NOTE ->>> when i change this to forwardFromReads.intersection(Set(cellWrites.values)) biquad works correctly
                // However in that case, it its too aggressive with how
                let newWrites = forwardFromReads.intersection(Set(cellWrites.values))

                //let newWrites = forwardFromReads.intersection(Set([cellWrites[cellId]!]))
                for newWrite in newWrites {
                    let forwards = reachForward(from: [newWrite])
                    if forwards.contains(writeNode) {
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
            for read in allReadsInCluster {
                let fromRead = reachForward(from: [read])
                // Only include nodes that can reach at least one write in the cluster
                for node in fromRead {
                    let fromNode = reachForward(from: [node])
                    if !fromNode.isDisjoint(with: allWritesInCluster) {
                        clusterNodes.insert(node)
                    }
                }
            }

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

public func scalarNodes(_ g: Graph, feedbackClusters: [[NodeID]]) -> Set<NodeID> {
    var scalar: Set<NodeID> = []

    //return Set(g.nodes.keys)

    // First, mark inherently scalar operations
    g.nodes.values.forEach {
        switch $0.op {
        case .accum(_):
            scalar.insert($0.id)  // Accum operations need to be scalar (stateful)
        case .latch(_):
            scalar.insert($0.id)  // Latch operations need to be scalar (stateful)
        case .phasor(_):
            scalar.insert($0.id)  // Phasor operations need to be scalar (stateful)
        //case .seq:
        //    scalar.insert($0.id)  // Seq operations need to be scalar (ordering dependent)

        // Tensor operations must be scalar because they have internal loops
        case .historyRead(let cellId):
            // Tensor history reads are scalar (they use parallelRange internally)
            if g.cellToTensor[cellId] != nil {
                scalar.insert($0.id)
            }
        case .historyWrite(let cellId):
            // Tensor history writes are scalar (they use parallelRange internally)
            if g.cellToTensor[cellId] != nil {
                scalar.insert($0.id)
            }
        case .conv2d(_):
            scalar.insert($0.id)
        case .sum:
            scalar.insert($0.id)
        default: break
        }
    }

    // Also mark nodes with tensor inputs as scalar (they use parallelRange internally)
    // First, identify all tensor source nodes (tensorRef produces tensors, historyRead with tensor cell)
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

    // Mark nodes with tensor inputs as scalar (they need element-wise loops)
    g.nodes.values.forEach {
        for inputId in $0.inputs {
            if tensorNodes.contains(inputId) {
                scalar.insert($0.id)
                // If this node consumes a tensor and produces something, it's also a tensor producer
                tensorNodes.insert($0.id)
            }
        }
    }

    // Repeat to propagate through chains of tensor ops
    var changed = true
    while changed {
        changed = false
        g.nodes.values.forEach { node in
            for inputId in node.inputs {
                if tensorNodes.contains(inputId) && !scalar.contains(node.id) {
                    scalar.insert(node.id)
                    if !tensorNodes.contains(node.id) {
                        tensorNodes.insert(node.id)
                        changed = true
                    }
                }
            }
        }
    }

    // Use the new feedback loop detection to mark all nodes in feedback loops as scalar
    for loop in feedbackClusters {
        for nodeId in loop {
            scalar.insert(nodeId)
        }
    }

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

// Simple topological sort within a corridor
private func simpleTopoSortWithinCorridor(nodes: [NodeID], g: Graph) -> [NodeID] {
    let nodeSet = Set(nodes)
    var indegree: [NodeID: Int] = [:]

    // Calculate in-degrees within corridor
    for nodeId in nodes {
        indegree[nodeId] = 0
    }

    for nodeId in nodes {
        if let node = g.nodes[nodeId] {
            for dep in node.allDependencies {
                if nodeSet.contains(dep) {
                    indegree[nodeId]! += 1
                }
            }
        }
    }

    // Kahn's algorithm
    var queue = indegree.filter { $0.value == 0 }.map { $0.key }.sorted()
    var result: [NodeID] = []

    while let nodeId = queue.first {
        queue.removeFirst()
        result.append(nodeId)

        // Update consumers within corridor
        for otherNodeId in nodes {
            if let otherNode = g.nodes[otherNodeId] {
                if otherNode.allDependencies.contains(nodeId) {
                    indegree[otherNodeId]! -= 1
                    if indegree[otherNodeId] == 0 {
                        let insertIndex = queue.firstIndex { $0 > otherNodeId } ?? queue.count
                        queue.insert(otherNodeId, at: insertIndex)
                    }
                }
            }
        }
    }

    // Add any remaining nodes (cycles within corridor)
    let remaining = Set(nodes).subtracting(result)
    result.append(contentsOf: remaining.sorted())

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
    var blockPass1Cells: [Set<CellID>] = []
    var blockPass2Cells: [Set<CellID>] = []

    for b in blocks {
        var hasPass1 = false
        var hasPass2 = false
        var p1Cells = Set<CellID>()
        var p2Cells = Set<CellID>()
        for nodeId in b.nodes {
            if let node = g.nodes[nodeId] {
                switch node.op {
                case let .spectralLossPass1(_, scratchCell):
                    hasPass1 = true
                    p1Cells.insert(scratchCell)
                case let .spectralLossPass2(_, scratchCell):
                    hasPass2 = true
                    p2Cells.insert(scratchCell)
                default:
                    break
                }
            }
        }
        blockHasPass1.append(hasPass1)
        blockHasPass2.append(hasPass2)
        blockPass1Cells.append(p1Cells)
        blockPass2Cells.append(p2Cells)
    }

    var fused: [Block] = []
    var fusedHasPass1: [Bool] = []
    var fusedHasPass2: [Bool] = []
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
            let canFuse = !conflict

            if canFuse {
                if debugFuse {
                    print("[FUSE] merge idx=\(idx) into last=\(lastIdx) kind=\(b.kind)")
                }
                fused[lastIdx].nodes.append(contentsOf: b.nodes)
                // Update flags
                fusedHasPass1[lastIdx] = fusedHasPass1[lastIdx] || blockHasPass1[idx]
                fusedHasPass2[lastIdx] = fusedHasPass2[lastIdx] || blockHasPass2[idx]
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

public func emitBlockUOps(
    ctx: IRContext, block: Block, blocks: [Block], g: Graph, debug: Bool = false
) throws -> [UOp] {
    var emittedNodes: Set<NodeID> = []

    var uops: [UOp] = []

    // Tensor Register Optimization:
    // Compute which tensor cells need to be written to memory (used by later blocks)
    // and clear the register tracking for this new block.
    ctx.outboundTensorCells = findOutboundTensorCells(blocks, g, block: block)
    ctx.clearTensorRegisters()

    if let tensorIndex = block.tensorIndex,
        let shape = block.shape
    {
        // Use reduce to compute count for any dimensionality
        let count = shape.reduce(1, *)
        let incr = 1  //count % 4 == 0 ? 4 : 1
        uops.append(UOp(op: .beginParallelRange(count, incr), value: tensorIndex))
    }

    for nodeId in block.nodes {
        if let tensorIndex = block.tensorIndex {
            ctx.tensorIndices[nodeId] = tensorIndex
        }

        if let node = g.nodes[nodeId] {
            if case .forward = block.direction {
                for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
                    emittedNodes.insert(nodeId)

                    var typedUOp = uop
                    /*
                    if block.tensorIndex != nil,
                        let shape = block.shape
                    {
                        let size = shape.reduce(1, *)
                        typedUOp.kind = size % 4 == 0 ? .simd : .scalar

                    } else {
                        typedUOp.kind = block.kind
                    }
                     */
                    typedUOp.kind = block.kind
                    uops.append(typedUOp)
                }
            } else {
                let back = node.op.emitBackward(ctx: ctx, g: g, nodeId: nodeId)
                // should be even better
                // ideally we could just ask inside emitBackward (is there a grad for nodeId if not use 1 cuz its the start)
                for uop in back {
                    emittedNodes.insert(nodeId)

                    var typedUOp = uop
                    typedUOp.kind = block.kind
                    uops.append(typedUOp)
                }
            }
        }
    }

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
                loadGlobalUOp.kind = block.kind
                uops.insert(loadGlobalUOp, at: 0)
            default:
                break
            }
        }
    }

    if block.tensorIndex != nil {
        uops.append(UOp(op: .endParallelRange, value: ctx.useVariable(src: nil)))
    }

    return uops
}

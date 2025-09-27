public enum Kind { case simd, scalar }

// corresponds to one kernel (metal backends) or for loop (in C backend)
public struct Block: Equatable {
    public var kind: Kind
    public var nodes: [NodeID] = []

    public init(kind: Kind) {
        self.kind = kind
    }
}

// Find all nodes that participate in feedback loops (not just minimal cycles)
public func findFeedbackLoops(_ g: Graph) -> [[NodeID]] {
    // Build maps of history cells to their read/write nodes
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
            if let n = g.nodes[node], case .historyWrite(let cellId) = n.op {
                if let readNode = cellReads[cellId] {
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
            if let n = g.nodes[node], case .historyRead(let cellId) = n.op {
                if let writeNode = cellWrites[cellId] {
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
                if let node = g.nodes[readNode], case .historyRead(let cell) = node.op {
                    if let writeNode = cellWrites[cell] {
                        clusterNodes.insert(writeNode)
                        allWritesInCluster.insert(writeNode)
                    }
                }
            }

            // If a write is in the cluster, its corresponding read must also be included
            for writeNode in allWritesInCluster {
                if let node = g.nodes[writeNode], case .historyWrite(let cell) = node.op {
                    if let readNode = cellReads[cell] {
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
        default: break
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

            if targetBlockIdx != -1
                && !wouldExceedNodeLimit(blocks[targetBlockIdx], maxNodesPerBlock: maxNodesPerBlock)
            {
                blocks[targetBlockIdx].nodes.append(nodeId)
                if debug {
                    print(
                        "📍 Placed output node \(nodeId) in block \(targetBlockIdx) with its dependency"
                    )
                }
            } else {
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
            if current.kind == kind
                && !wouldExceedNodeLimit(current, maxNodesPerBlock: maxNodesPerBlock)
            {
                // Add to current block
                currentBlock!.nodes.append(nodeId)
            } else {
                // Finish current block and start new one
                blocks.append(current)
                currentBlock = Block(kind: kind)
                currentBlock!.nodes.append(nodeId)
            }
        } else {
            // Start first block
            currentBlock = Block(kind: kind)
            currentBlock!.nodes.append(nodeId)
        }
    }

    // Don't forget the last block
    if let current = currentBlock {
        blocks.append(current)
    }

    if debug {
        print("📦 Created \(blocks.count) blocks with simplified logic")
        for (i, block) in blocks.enumerated() {
            print("  Block \(i) (\(block.kind)): \(block.nodes)")
        }
    }

    return blocks
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
public func findNodesWithOutboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> Set<
    NodeID
> {
    // node -> block idx map
    var nodeBlock = [NodeID: Int]()
    let idx = blks.firstIndex { b in return b == block }
    //block.nodes.forEach { nodeBlock[$0] = idx }
    for (idx, b) in blks.enumerated() { b.nodes.forEach { nodeBlock[$0] = idx } }

    let outputNodeNeeds = findOutputNodeNeeds(block, g)

    var need: Set<NodeID> = []
    for (i, b) in blks.enumerated() {
        if i == idx {
            continue
        }
        for nID in b.nodes {
            g.nodes[nID]!.allDependencies.forEach {
                if let nodeBlockIdx = nodeBlock[$0] {
                    if nodeBlockIdx == idx && !outputNodeNeeds.contains($0) {
                        // Check if this is a seq node - if so, we need to make its last input available as global
                        if let depNode = g.nodes[$0], case .seq = depNode.op {
                            // For seq nodes, the actual value comes from the last input
                            if let lastInput = depNode.inputs.last {
                                need.insert(lastInput)
                            }
                        } else {
                            need.insert($0)
                        }
                    }  // producer in diff block
                }
            }
        }
    }
    return need
}

public func findNodesAsInboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> Set<NodeID>
{
    let outputNodeNeeds = findOutputNodeNeeds(block, g)

    let blockNumber = blks.firstIndex(of: block)

    // find which block each node is defined in
    var nodeBlock = [NodeID: Int]()
    for (idx, b) in blks.enumerated() { b.nodes.forEach { nodeBlock[$0] = idx } }

    var need: Set<NodeID> = []
    // go thru each node in this block and if its defined in some other block, add it here
    for nID in block.nodes {
        // for each input of a node from **this** block, check if it was produced somewhere else
        g.nodes[nID]!.allDependencies.forEach {
            if nodeBlock[$0]! != blockNumber && !outputNodeNeeds.contains($0) {
                need.insert($0)
            }  // producer in diff block
        }
    }
    return need
}

// ─── decide which forward values from other blocks are needed for backward pass ─────

// TODO - implement back prop emitBlocks
// Back props blocks needs to understand its dependencies, i.e. what values that come from other blocks' buffers
// are required
// Ideally this would be emitted directly from the emitBackward() so that we get a list of uops along with
// what their dependencies are
//

public func emitBlockUOps(
    ctx: IRContext, block: Block, blocks: [Block], g: Graph, debug: Bool = false
) throws -> [UOp] {
    var emittedNodes: Set<NodeID> = []

    var uops: [UOp] = []
    for nodeId in block.nodes {
        if var node = g.nodes[nodeId] {
            var indentLevel = 0
            for uop in try node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
                emittedNodes.insert(nodeId)

                var typedUOp = uop
                typedUOp.kind = block.kind
                uops.append(typedUOp)
            }
        }
    }

    let outbound = findNodesWithOutboundDependencies(blocks, g, block: block)
    for nodeId in outbound {
        if emittedNodes.contains(nodeId) {
            if let lz = ctx.values[nodeId] {
                switch lz {
                case .variable(let a, _):
                    var defineGlobalUOp = UOp(op: .defineGlobal(a), value: .global(a))
                    defineGlobalUOp.kind = block.kind
                    uops.insert(defineGlobalUOp, at: 0)
                    ctx.globals.insert(a)
                default:
                    break
                }
            }
        }
    }

    let inbound = findNodesAsInboundDependencies(blocks, g, block: block)

    for nodeId in inbound {
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

    if debug {
        var indentLevel = 0
        for uop in uops {
            switch uop.op {
            case .beginIf:
                print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
                indentLevel += 1
            case .endIf:
                indentLevel = max(0, indentLevel - 1)
                print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
            default:
                print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
            }
        }
    }

    return uops
}

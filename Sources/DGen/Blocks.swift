
public enum Kind { case simd, scalar }

// corresponds to one kernel (metal backends) or for loop (in C backend)
public struct Block: Equatable {
    public var kind: Kind
    public var nodes: [NodeID] = []

    public init(kind: Kind) {
        self.kind = kind
    }
}

public func scalarNodes(_ g: Graph) -> Set<NodeID> {
    var reads = [Int: [NodeID]](), writes = [Int: [NodeID]]()
    var scalar: Set<NodeID> = []

    // First, mark all history operations as scalar
    g.nodes.values.forEach {
        switch $0.op {
        case .historyRead(let c):
            reads[c, default: []].append($0.id)
            scalar.insert($0.id)
        case .historyWrite(let c):
            writes[c, default: []].append($0.id)
            scalar.insert($0.id)
        case .accum(let c):
            writes[c, default: []].append($0.id)
            scalar.insert($0.id)
        case .latch(let c):
            writes[c, default: []].append($0.id)
            scalar.insert($0.id)
        case .phasor(let c):
            writes[c, default: []].append($0.id)
            scalar.insert($0.id)
        default: break
        }
    }

    // forward & backward reach helpers
    var consumers = [NodeID: [NodeID]]()
    g.nodes.values.forEach { n in n.inputs.forEach { consumers[$0, default: []].append(n.id) } }
    func reach(_ roots: [NodeID], fwd: Bool) -> Set<NodeID> {
        var out: Set<NodeID> = []; var stk = roots
        while let n = stk.popLast() {
            if out.insert(n).inserted {
                stk.append(contentsOf: fwd ? (consumers[n] ?? []) : g.nodes[n]!.inputs)
            }
        }
        return out
    }

    // Find read→write corridors and mark them as scalar
    for c in reads.keys {
        guard let rs = reads[c], let ws = writes[c] else { continue }
        scalar.formUnion(reach(rs, fwd: true).intersection(reach(ws, fwd: false)))
    }
    return scalar
}

// ───────────── 2. topo-sort, split into blocks ──────────────────
public func topo(_ g: Graph) -> [NodeID] {
    var indeg = [NodeID: Int](); g.nodes.keys.forEach { indeg[$0] = 0 }
    g.nodes.values.forEach { n in n.inputs.forEach { indeg[$0]! += 1 } }
    var q = indeg.filter { $0.value == 0 }.map { $0.key }
    var out: [NodeID] = []
    while let n = q.first {
        q.removeFirst(); out.append(n)
        g.nodes[n]!.inputs.forEach { indeg[$0]! -= 1; if indeg[$0] == 0 { q.append($0) } }
    }
    return out.reversed()
}

func getBlockIndexWithDependency(nodeID: NodeID, g: Graph, blocks: [Block], kind: Kind) -> Int? {
    guard let node = g.nodes[nodeID] else { return nil }

    return blocks.firstIndex { block in
        guard block.kind == kind else { return false }
        return node.inputs.contains { inputID in
            block.nodes.contains(inputID)
        }
    }
}

// Helper function to check if adding a node to a block would exceed the node limit
func wouldExceedNodeLimit(_ block: Block, maxNodesPerBlock: Int) -> Bool {
    return block.nodes.count >= maxNodesPerBlock
}

// partitions sorted nodes & set of scalar ndoes, into blocks (of kind: simd or scalar)
// cannot exceed maxNodesPerBlock (due to buffer limits on metal kernels)
public func determineBlocks(sorted: [NodeID], scalar: Set<NodeID>, g: Graph, maxNodesPerBlock: Int = Int.max) -> [Block] {
    var b: [Block] = []

    for n in sorted {
        let k: Kind = scalar.contains(n) ? .scalar : .simd

        var foundOutput = false
        if let node = g.nodes[n] {
            switch node.op {
            case .output:
                b[b.count-1].nodes.append(n)
                foundOutput = true
                break
            default:
                break
            }
        }

        if (foundOutput) {
            break
        }

        if k == .scalar {
            let scalarBlockIdx = getBlockIndexWithDependency(nodeID: n, g: g, blocks: b, kind: .scalar)
            let simdBlockIdx = getBlockIndexWithDependency(nodeID: n, g: g, blocks: b, kind: .simd)

            if let _ = simdBlockIdx {
               // then we need to place this ahead of the simd block
                if let scalarIdx = scalarBlockIdx, !wouldExceedNodeLimit(b[scalarIdx], maxNodesPerBlock: maxNodesPerBlock) {
                    var scalarBlock = b[scalarIdx]
                    scalarBlock.nodes.append(n)
                    // we need pull that index
                    b.remove(at: scalarIdx)
                    b.append(scalarBlock)
                } else {
                    // theres no scalar block with a dependency or it would exceed limit, so we just place a new block ahead
                    var scalarBlock = Block(kind: .scalar)
                    scalarBlock.nodes.append(n)
                    b.append(scalarBlock)
                }
            } else {
                // otherwise it has no dependencies in simd
                 // then we need to place this ahead of the simd block
                if let scalarIdx = scalarBlockIdx, !wouldExceedNodeLimit(b[scalarIdx], maxNodesPerBlock: maxNodesPerBlock) {
                    b[scalarIdx].nodes.append(n)
                } else {
                    // theres no scalar block with a dependency or it would exceed limit, so we just place a new block ahead
                    var scalarBlock = Block(kind: .scalar)
                    scalarBlock.nodes.append(n)
                    b.append(scalarBlock)
                }
            }
        } else {
            // This is a SIMD node
            if b.isEmpty || b.last?.kind == .scalar || wouldExceedNodeLimit(b.last!, maxNodesPerBlock: maxNodesPerBlock) {
                // if no blocks exist, last block is scalar, or last block would exceed node limit, create a new SIMD block
                b.append(Block(kind: .simd))
            }
            b[b.count - 1].nodes.append(n)
        }
    }
    return b
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
public func findNodesWithOutboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> Set<NodeID> {
    // node -> block idx map
    var nodeBlock = [NodeID: Int]()
    let idx = blks.firstIndex{b in return b == block}
    block.nodes.forEach { nodeBlock[$0] = idx }
    
    let outputNodeNeeds = findOutputNodeNeeds(block, g)

    var need: Set<NodeID> = []
    for b in blks {
        for nID in b.nodes {
            g.nodes[nID]!.inputs.forEach {
                if let nodeBlockIdx = nodeBlock[$0] {
                    if (nodeBlockIdx != nID && !outputNodeNeeds.contains($0)) { need.insert($0) }  // producer in diff block
                }
            }
        }
    }
    return need
}

public func findNodesAsInboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> Set<NodeID> {
    let outputNodeNeeds = findOutputNodeNeeds(block, g)

    // find which block each node is defined in
    var nodeBlock = [NodeID: Int]()
    for (idx, b) in blks.enumerated() { b.nodes.forEach { nodeBlock[$0] = idx } }

    var need: Set<NodeID> = []
    // go thru each node in this block and if its defined in some other block, add it here
    for nID in block.nodes {
        g.nodes[nID]!.inputs.forEach {
            if (nodeBlock[$0]! != nID && !outputNodeNeeds.contains($0)) { need.insert($0) }  // producer in diff block
        }
    }
    return need
}



// ─── decide which forward values from other blocks are needed for backward pass ─────
public func crossBlockForwardValues(_ blks: [Block], _ g: Graph, block: Block) -> Set<NodeID> {
    // node -> block idx map
    var nodeBlock = [NodeID: Int]()
    for (idx, b) in blks.enumerated() { b.nodes.forEach { nodeBlock[$0] = idx } }

    // Find the current block index
    guard let currentBlockIdx = blks.firstIndex(where: { $0.nodes == block.nodes }) else {
        return Set<NodeID>()
    }

    var needForwardValues: Set<NodeID> = []
    for nID in block.nodes {
        let node = g.nodes[nID]!
        for inputID in node.inputs {
            // Check if the input is from a different block
            if let inputBlockIdx = nodeBlock[inputID], inputBlockIdx != currentBlockIdx {
                // For backward pass, we need the forward value of this input
                needForwardValues.insert(inputID)
            }
        }
    }
    return needForwardValues
}

// Sort blocks by dependencies to determine execution order
public func sortBlocksByDependencies(_ blks: [Block], _ g: Graph) -> [Int] {
    var blockDependencies: [Int: Set<Int>] = [:]

    for (blockIdx, block) in blks.enumerated() {
        blockDependencies[blockIdx] = Set<Int>()

        for nodeID in block.nodes {
            let node = g.nodes[nodeID]!
            for inputID in node.inputs {
                // Find which block produces this input
                for (otherBlockIdx, otherBlock) in blks.enumerated() {
                    if otherBlockIdx != blockIdx && otherBlock.nodes.contains(inputID) {
                        blockDependencies[blockIdx]!.insert(otherBlockIdx)
                    }
                }
            }
        }
    }

    // Topological sort of blocks
    var visited = Set<Int>()
    var result: [Int] = []

    func visitBlock(_ blockIdx: Int) {
        if visited.contains(blockIdx) { return }
        visited.insert(blockIdx)

        for depIdx in blockDependencies[blockIdx] ?? [] {
            visitBlock(depIdx)
        }

        result.append(blockIdx)
    }

    for blockIdx in blks.indices {
        visitBlock(blockIdx)
    }

    return result
}


public func emitBlockUOps (ctx: IRContext, block: Block, blocks: [Block], g: Graph, debug: Bool=false) -> [UOp]  {
    var emittedNodes: Set<NodeID> = []

    var uops: [UOp] = []
    for nodeId in block.nodes {
        if let node = g.nodes[nodeId] {
            var indentLevel = 0

            for uop in node.op.emit(ctx: ctx, g: g, nodeId: nodeId) {
                emittedNodes.insert(nodeId)
                var typedUOp = uop
                typedUOp.kind = block.kind
                uops.append(typedUOp)
            }
        }
    }

    let outbound = findNodesWithOutboundDependencies(blocks, g, block: block);
    for nodeId in outbound {
        if (emittedNodes.contains(nodeId)) {
            if let lz = ctx.values[nodeId] {
                switch lz {
                case .variable(let a,_):
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
   
    let inbound = findNodesAsInboundDependencies(blocks, g, block: block);
    for nodeId in inbound {
        if let lz = ctx.values[nodeId] {
            switch lz {
            case .variable(let a,_):
                var loadGlobalUOp = UOp(op: .loadGlobal(a), value: .global(a))
                loadGlobalUOp.kind = block.kind
                uops.insert(loadGlobalUOp, at: 0)
            default:
                break
            }
        }
    }

    if (debug) {
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

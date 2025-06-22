let g = Graph()

let freq1 = g.n(.mul, g.n(.constant(440)), g.n(.constant(0.5)))
let freq2 = g.n(.mul, g.n(.constant(2)), g.n(.constant(3.5)))
let ph1 = g.n(.phasor(0),
  freq1,
  g.n(.constant(0)));

let ph2 = g.n(.phasor(1),
  freq2,
  g.n(.constant(0)));

let cond = g.n(.lt, ph2, g.n(.constant(0.3)));

let latched = g.n(.latch(2), ph1, cond)

let mult = g.n(.mul, latched, ph1)
    
let sorted = topo(g)
let scalar = scalarNodes(g)
let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, maxNodesPerBlock: 320);
let sortedBlockIds = sortBlocksByDependencies(blocks, g)
var sortedBlocks: [Block] = [];
sortedBlockIds.forEach{ sortedBlocks.append(blocks[$0]) }

var ctx = IRContext()

for blockIdx in sortedBlockIds {
    let block = blocks[blockIdx]
    print("")
    print("\(ANSI.bold)block ----- #\(blockIdx) --- \(block.kind)\(ANSI.reset)")

    _ = emitBlock(ctx: ctx, block: block, blocks: sortedBlocks, g: g, debug: true)
}

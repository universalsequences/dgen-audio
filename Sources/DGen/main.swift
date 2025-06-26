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

let out = g.n(.output(0), mult)
    
let sorted = topo(g)
let scalar = scalarNodes(g)
let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, maxNodesPerBlock: 320);
let sortedBlockIds = sortBlocksByDependencies(blocks, g)
var sortedBlocks: [Block] = [];
sortedBlockIds.forEach{ sortedBlocks.append(blocks[$0]) }

var ctx = IRContext()

public struct BlockUOps {
    public var ops: [UOp]
    public let kind: Kind
}

var uopBlocks: [BlockUOps] = []

for blockIdx in sortedBlockIds {
    let block = blocks[blockIdx]
    print("")
    print("\(ANSI.bold)block ----- #\(blockIdx) --- \(block.kind)\(ANSI.reset)")

    uopBlocks.append(BlockUOps(ops: emitBlockUOps(ctx: ctx, block: block, blocks: sortedBlocks, g: g, debug: true), kind: block.kind))
}

let kernels = lowerUOpBlocks(&uopBlocks, renderer: CRenderer(), ctx: ctx, frameCount: 128)

for kernel in kernels {
    print("Kernel \(kernel.name):")
    for buffer in kernel.buffers {
        print("- buffer: \(buffer)")
    }
    print(kernel.source)
    let source = kernel.source

    let compiled = CCompiledKernel(source: source)
    try compiled.compileAndLoad()

    var outputs = [Float](repeating: 0, count: 128)
    var inputs = [Float](repeating: 1, count: 128)

    compiled.run(outputs: &outputs, inputs: inputs, frameCount: 128)

    print(outputs)
}

import DGen

let g = Graph()

let freq1 = g.n(.mul, g.n(.constant(90)), g.n(.constant(2.5)))
//let freq2 = g.n(.mul, g.n(.constant(2)), g.n(.constant(3.5)))
let ph1 = g.n(.phasor(0),
  freq1,
  g.n(.constant(0)));

let m = g.n(.mul, freq1, g.n(.constant(2)))
let scaled = g.n(.add, m, g.n(.constant(-1)))

let out = g.n(.output(0), scaled)

/*
let ph2 = g.n(.phasor(1),
  freq2,
  g.n(.constant(0)));

let cond = g.n(.lt, ph2, g.n(.constant(0.1)));

let latched = g.n(.latch(2), ph1, cond)

let mult = g.n(.mul, latched, ph1)

let out = g.n(.output(0), mult)
 */
    
let sorted = topo(g)
let scalar = scalarNodes(g)
let blocks = determineBlocks(sorted: sorted, scalar: scalar, g: g, maxNodesPerBlock: 320);
let sortedBlockIds = sortBlocksByDependencies(blocks, g)
var sortedBlocks: [Block] = [];
sortedBlockIds.forEach{ sortedBlocks.append(blocks[$0]) }

var ctx = IRContext()

var uopBlocks: [BlockUOps] = []

for blockIdx in sortedBlockIds {
    let block = blocks[blockIdx]
    print("")
    print("\(ANSI.bold)block ----- #\(blockIdx) --- \(block.kind)\(ANSI.reset)")

    uopBlocks.append(BlockUOps(ops: emitBlockUOps(ctx: ctx, block: block, blocks: sortedBlocks, g: g, debug: true), kind: block.kind))
}

// Choose target device
let targetDevice: Device = .Metal  // Change to .C for C backend

let renderer: Renderer = targetDevice == .Metal ? MetalRenderer() : CRenderer()
let kernels = lowerUOpBlocks(&uopBlocks, renderer: renderer, ctx: ctx, frameCount: 128)

print("\n🎯 Using \(targetDevice) backend")
print("📊 Generated \(kernels.count) kernel(s)")

for kernel in kernels {
    print("\nKernel \(kernel.name):")
    for buffer in kernel.buffers {
        print("- buffer: \(buffer)")
    }
    print("- threadGroupSize: \(kernel.threadGroupSize)")
    if targetDevice == .Metal {
        print("Metal source:")
        print(kernel.source)
    } else {
        print(kernel.source)
    }
}

let runtime: CompiledKernelRuntime
if targetDevice == .Metal {
    runtime = try MetalCompiledKernel(kernels: kernels)
} else {
    // For C backend, use the first kernel (existing behavior)
    let source = kernels.first?.source ?? ""
    let compiled = CCompiledKernel(source: source)
    try compiled.compileAndLoad()
    runtime = compiled
}

try runtime.runAndPlay(durationSeconds: 2.0, sampleRate: 44100.0, channels: 1, volumeScale: 0.1)

// Test buffer readback for Metal
if let metalRuntime = runtime as? MetalCompiledKernel {
    print("\n📈 Buffer readback test:")
    for bufferName in Set(kernels.flatMap { $0.buffers }) {
        if let bufferData = metalRuntime.readBuffer(named: bufferName) {
            print("Buffer \(bufferName): [\(bufferData.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", "))...]")
        }
    }
}

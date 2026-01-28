
/*
public func compileDGenGraph(g: Graph) {
    do {
        let sorted          = topo(g)
        let scalar          = scalarNodes(g)
        let blocks          = determineBlocks(sorted: sorted,
                                              scalar: scalar,
                                              g: g,
                                              maxNodesPerBlock: 320)
        let resolvedBlocks  = resolveBlockDependencies(blocks, g, originalTopoOrder: sorted)
        let sortedBlockIds  = sortBlocksByDependencies(resolvedBlocks, g)
        let sortedBlocks    = sortedBlockIds.map { resolvedBlocks[$0] }
        let fusedBlocks     = fuseConsecutiveBlocks(sortedBlocks)
        let finalBlockIds   = Array(0..<fusedBlocks.count)

        print("topo sort =", sorted)
        print("sorted blocks =", fusedBlocks)

        let ctx         = IRContext()
        var uopBlocks   = [BlockUOps]()

        for blockIdx in finalBlockIds {
            let block = fusedBlocks[blockIdx]
            print("\n\(ANSI.bold)block ----- #\(blockIdx) --- \(block.kind)\(ANSI.reset)")

            let ops = emitBlockUOps(ctx: ctx,
                                    block: block,
                                    blocks: fusedBlocks,
                                    g: g,
                                    debug: true)
            uopBlocks.append(BlockUOps(ops: ops, kind: block.kind))
        }

        // Choose target device
        let targetDevice: Device = .Metal    // Change to .Metal for Metal backend
        let renderer: Renderer   = (targetDevice == .Metal) ? MetalRenderer()
                                                            : CRenderer()
        let kernels = lowerUOpBlocks(&uopBlocks,
                                     renderer: renderer,
                                     ctx: ctx,
                                     frameCount: 128,
                                     graph: g,
                                     totalMemorySlots: g.totalMemoryCells)

        print("\n🎯 Using \(targetDevice) backend")
        print("📊 Generated \(kernels.count) kernel(s)")

        for kernel in kernels {
            print("\nKernel \(kernel.name):")
            kernel.buffers.forEach { print("- buffer: \($0)") }
            print("- threadGroupSize: \(kernel.threadGroupSize)")
            print(kernel.source)        // prints either Metal or C source
        }

        // Build & run
        let runtime: CompiledKernelRuntime
        if targetDevice == .Metal {
            runtime = try MetalCompiledKernel(kernels: kernels)
        } else {
            let source   = kernels.first?.source ?? ""
            let memorySize = kernels.first?.memorySize ?? 1024
            let compiled = CCompiledKernel(source: source, cellAllocations: CellAllocations(), memorySize: memorySize)
            try compiled.compileAndLoad()
            runtime = compiled
        }

        try runtime.runAndPlay(durationSeconds: 8.0,
                               sampleRate: 44100,
                               channels: 1,
                               volumeScale: 0.1)

        // Optional Metal buffer read-back
        if let metalRuntime = runtime as? MetalCompiledKernel {
            print("\n📈 Buffer readback test:")
            for name in Set(kernels.flatMap(\.buffers)) {
                if let data = metalRuntime.readBuffer(named: name) {
                    let preview = data.prefix(5)
                                     .map { String(format: "%.3f", $0) }
                                     .joined(separator: ", ")
                    print("Buffer \(name): [\(preview)...]")
                }
            }
        }

    } catch {
        // This clause matches *every* possible Error.
        // Use `localizedDescription` if you just want the message.
        print("❌ compileDGenGraph failed with error: \(error)")
    }
}

 */

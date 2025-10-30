import XCTest

@testable import DGen

final class BasicCompilationTests: XCTestCase {

    func testBasicGraphCompilation() throws {
        // Create a simple graph
        let graph = Graph()

        // Add some basic operations: x + y where x = 2.0, y = 3.0
        let x = graph.n(.constant(2.0))
        let y = graph.n(.constant(3.0))
        let _ = graph.n(.output(0), graph.n(.add, x, y))

        // Compile the graph to UOps
        let compilationResult = try CompilationPipeline.compile(
            graph: graph,
            backend: .c,
            options: CompilationPipeline.Options(debug: true)
        )

        // Print the UOps
        print("=== UOp Blocks ===")
        for (blockIndex, block) in compilationResult.uopBlocks.enumerated() {
            print("Block \(blockIndex) (\(block.kind)):")
            for (opIndex, uop) in block.ops.enumerated() {
                print("  [\(opIndex)] \(uop.op) -> \(uop.value)")
            }
        }

        print("\n=== Generated Source ===")
        print(compilationResult.source)

        // Basic assertions
        XCTAssertFalse(compilationResult.uopBlocks.isEmpty)
        XCTAssertFalse(compilationResult.kernels.isEmpty)
        XCTAssertFalse(compilationResult.source.isEmpty)
    }

    func testBiquadMetalEqualsC() throws {
        // Build a biquad lowpass graph and compare Metal vs C
        let g = Graph()
        let input = g.n(.constant(1.0))
        let cutoff = g.n(.constant(1000.0))
        let resonance = g.n(.constant(0.7))
        let gain = g.n(.constant(1.0))
        let mode = g.n(.constant(0.0))  // lowpass

        let y = g.biquad(
            g.n(.phasor(g.alloc()), input, g.n(.constant(0.0))), cutoff, resonance, gain, mode)
        _ = g.n(.output(0), y)

        let frameCount = 128

        // Compile both backends
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount, debug: true)
        )
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        // Print kernels for debugging
        for kernel in mResult.kernels { print(kernel.source) }
        for kernel in cResult.kernels { print(kernel.source) }

        // Prepare runtimes
        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        let mRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context
        )

        var outC = [Float](repeating: 0, count: frameCount)
        var outM = [Float](repeating: 0, count: frameCount)
        let inBuf = [Float](repeating: 0, count: frameCount)

        if let mem = cRuntime.allocateNodeMemory() {
            outC.withUnsafeMutableBufferPointer { outPtr in
                inBuf.withUnsafeBufferPointer { inPtr in
                    cRuntime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: mem,
                        frameCount: frameCount
                    )
                }
            }
            cRuntime.deallocateNodeMemory(mem)
        } else {
            XCTFail("Failed to allocate C runtime node memory")
        }

        outM.withUnsafeMutableBufferPointer { outPtr in
            inBuf.withUnsafeBufferPointer { inPtr in
                mRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        let tol: Float = 1e-4
        for i in 0..<frameCount {
            XCTAssertEqual(outM[i], outC[i], accuracy: tol, "Mismatch at sample \(i)")
        }
    }

    func testPhasorMetalEqualsC() throws {
        // Build a simple phasor graph: phasor at 10 Hz, no reset, output to channel 0
        let g = Graph()
        let freq = g.n(.constant(10.0))
        let reset = g.n(.constant(0.0))
        let amp = g.n(.constant(0.5))
        let phase = g.n(.phasor(0), freq, reset)
        _ = g.n(.output(0), g.n(.mul, phase, amp))

        let frameCount = 128

        // Compile both backends
        let cResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount)
        )
        let mResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount)
        )

        // Prepare runtimes
        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        let mRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context
        )

        var outC = [Float](repeating: 0, count: frameCount)
        var outM = [Float](repeating: 0, count: frameCount)
        let inBuf = [Float](repeating: 0, count: frameCount)

        if let mem = cRuntime.allocateNodeMemory() {
            outC.withUnsafeMutableBufferPointer { outPtr in
                inBuf.withUnsafeBufferPointer { inPtr in
                    cRuntime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: mem,
                        frameCount: frameCount
                    )
                }
            }
            cRuntime.deallocateNodeMemory(mem)
        } else {
            XCTFail("Failed to allocate C runtime node memory")
        }

        outM.withUnsafeMutableBufferPointer { outPtr in
            inBuf.withUnsafeBufferPointer { inPtr in
                mRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount,
                    volumeScale: 1.0
                )
            }
        }

        // Compare Metal vs C outputs sample-by-sample
        let tol: Float = 1e-4
        for i in 0..<frameCount {
            XCTAssertEqual(outM[i], outC[i], accuracy: tol, "Mismatch at sample \(i)")
        }
    }
}

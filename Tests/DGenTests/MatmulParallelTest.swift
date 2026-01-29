import Foundation
import XCTest

@testable import DGen

final class MatmulParallelTest: XCTestCase {

    func testMatmulParallelKernels() throws {
        print("\n========================================")
        print("🧪 testMatmulParallelKernels")
        print("========================================")

        let g = Graph()

        // A: 2x3 matrix
        let a = g.tensor(shape: [2, 3], data: [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ])

        // B: 3x4 matrix
        let b = g.tensor(shape: [3, 4], data: [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ])

        let c = try g.matmul(a, b)
        let sum = g.n(.sum, c)
        _ = g.n(.output(0), sum)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: false)
        )

        XCTAssertGreaterThan(compileResult.kernels.count, 0)

        // Print UOp blocks (debug: true already prints UOps during compilation)
        print("\n=== UOP BLOCKS ===")
        for (blockIdx, block) in compileResult.uopBlocks.enumerated() {
            print("Block \(blockIdx): kind=\(block.kind), temporality=\(block.temporality)")
            var indentLevel = 0
            for uop in block.ops {
                switch uop.op {
                case .beginIf, .beginForLoop, .beginParallelRange, .beginLoop, .beginRange:
                    print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
                    indentLevel += 1
                case .endIf, .endLoop, .endParallelRange, .endRange:
                    indentLevel = max(0, indentLevel - 1)
                    print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
                default:
                    print("\(String(repeating: "  ", count: indentLevel))\(uop.prettyDescription())")
                }
            }
        }
        print("=== END UOP BLOCKS ===\n")

        // Write kernels to file for inspection
        let kernelPath = "/tmp/matmul_parallel_kernels.metal"
        var kernelSource = "// Generated kernels for testMatmulParallelKernels\n"
        kernelSource += "// Total kernels: \(compileResult.kernels.count)\n\n"

        for (i, kernel) in compileResult.kernels.enumerated() {
            kernelSource += "// ========================================\n"
            kernelSource += "// Kernel \(i): \(kernel.name)\n"
            kernelSource += "// ThreadGroupSize: \(String(describing: kernel.threadGroupSize))\n"
            kernelSource += "// ThreadCount: \(String(describing: kernel.threadCount))\n"
            kernelSource += "// Buffers: \(kernel.buffers)\n"
            kernelSource += "// ========================================\n"
            kernelSource += kernel.source
            kernelSource += "\n\n"
        }

        try kernelSource.write(toFile: kernelPath, atomically: true, encoding: .utf8)
        print("📝 Wrote kernels to: \(kernelPath)")

        // Also print to console
        print("\n=== GENERATED KERNELS ===")
        for (i, kernel) in compileResult.kernels.enumerated() {
            print("\n--- Kernel \(i): \(kernel.name) ---")
            print("ThreadGroupSize: \(String(describing: kernel.threadGroupSize))")
            print("ThreadCount: \(String(describing: kernel.threadCount))")
            print("Buffers: \(kernel.buffers)")
            print(kernel.source)
        }
        print("=== END KERNELS ===\n")

        // Execute with Metal runtime and validate result
        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount
        )

        if let memoryBuffer = runtime.getBuffer(name: "memory") {
            let memPtr = memoryBuffer.contents().assumingMemoryBound(to: Float.self)
            injectTensorData(result: compileResult, memory: memPtr)
        } else {
            XCTFail("Missing memory buffer in Metal runtime")
            return
        }

        var output = [Float](repeating: 0, count: frameCount)
        let input = [Float](repeating: 0, count: frameCount)

        output.withUnsafeMutableBufferPointer { outPtr in
            input.withUnsafeBufferPointer { inPtr in
                runtime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount
                )
            }
        }

        print("=== Matmul Parallel Result ===")
        print("Output: \(output[0])")

        // Expected sum of matmul result = 610.0
        XCTAssertEqual(output[0], 610.0, accuracy: 0.001, "Matmul sum should be 610")
    }
}

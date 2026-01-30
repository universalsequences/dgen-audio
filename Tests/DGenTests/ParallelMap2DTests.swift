import XCTest

@testable import DGen

final class ParallelMap2DTests: XCTestCase {
    func testParallelMap2DProducesExpectedSum() throws {
        print("\n🧪 Test: parallelMap2D produces expected per-frame sums")

        let g = Graph()
        let bins = 8
        let out = g.parallelMap2DTest(bins: bins)
        _ = g.n(.output(0), out)

        let frameCount = 32
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: false, backwards: false)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)
        let memory = runtime.allocateNodeMemory()!

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: memory,
                    frameCount: frameCount
                )
            }
        }

        runtime.deallocateNodeMemory(memory)

        let expectedBinSum = Float(bins * (bins - 1)) / 2.0
        let scale = Float(bins) * 100.0

        for i in 0..<frameCount {
            let expected = Float(i) * scale + expectedBinSum
            XCTAssertEqual(
                outputBuffer[i], expected, accuracy: 1e-3,
                "Frame \(i) expected \(expected) got \(outputBuffer[i])")
        }
    }
}

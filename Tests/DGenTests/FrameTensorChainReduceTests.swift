import XCTest

@testable import DGen

final class FrameTensorChainReduceTests: XCTestCase {
        /// Verify frame-tensor chain sum lowers to a map+reduce kernel and produces correct output.
        func testFrameTensorChainSumUsesMapReduce() throws {
                let frameCount = 4
                let controlFrames = 4
                let numHarmonics = 6
                let sampleRate: Float = Float(frameCount)

                let g = Graph(sampleRate: sampleRate)

                // Column-major layout for peekRow: offset = col * numRows + row
                var data = [Float](repeating: 0, count: controlFrames * numHarmonics)
                for r in 0..<controlFrames {
                        for c in 0..<numHarmonics {
                                let idx = c * controlFrames + r
                                data[idx] = Float(r * 10 + c)
                        }
                }
                let ampsTensor = g.tensor(shape: [controlFrames, numHarmonics], data: data)

                let zero = g.n(.constant(0.0))
                // Deterministic phasor: freq / sampleRate = 1 / frameCount
                let frameIdx = g.phasor(freq: g.n(.constant(1.0)), reset: zero)
                let playhead = g.n(
                        .floor, g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames)))))

                let row = try g.peekRow(tensor: ampsTensor, rowIndex: playhead)
                let sum = g.n(.sum, row)
                _ = g.n(.output(0), sum)

                let compileResult = try CompilationPipeline.compile(
                        graph: g, backend: .metal,
                        options: .init(frameCount: frameCount, backwards: true))

                writeKernelsToDisk(compileResult, "/tmp/chain_reduce_test.metal")
                // Ensure map kernel uses scaled thread count (frameCount * numHarmonics)
                XCTAssertTrue(
                        compileResult.kernels.contains { $0.threadCountScale == numHarmonics },
                        "Expected a kernel with threadCountScale == numHarmonics")

                let runtime = try MetalCompiledKernel(
                        kernels: compileResult.kernels,
                        cellAllocations: compileResult.cellAllocations,
                        context: compileResult.context)

                let ctx = TrainingContext(
                        parameters: [],
                        tensorParameters: [],
                        optimizer: SGD(lr: 0.0),
                        lossNode: sum)

                ctx.initializeMemory(
                        runtime: runtime,
                        cellAllocations: compileResult.cellAllocations,
                        context: compileResult.context,
                        frameCount: frameCount,
                        graph: g)

                _ = ctx.runStepGPU()
                let outputs = runtime.getOutputBuffer()

                XCTAssertEqual(outputs.count, frameCount)
                for frame in 0..<frameCount {
                        let expected = Float(60 * frame + 15)
                        XCTAssertEqual(
                                outputs[frame], expected, accuracy: 1e-3,
                                "Frame \(frame) expected \(expected) got \(outputs[frame])")
                }
        }
}

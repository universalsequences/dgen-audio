import XCTest

@testable import DGen

final class FilterTableTests: XCTestCase {

    func testFilterTableCoreUsesHopHeldTableFrame() throws {
        let N = 16
        let hop = 8
        let framesPerRun = 64
        let numRuns = 1
        let totalFrames = framesPerRun * numRuns
        let switchFrame = 32
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let input = g.n(.input(0))
        let buffered = g.bufferView(input, size: N, hopSize: hop)
        let flatInput = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flatInput, N: N)

        var tableData = [Float](repeating: 0.0, count: 2 * N)
        tableData[N] = 1.0
        let table = g.tensor(shape: [2, N], data: tableData)

        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let maxFrame = g.n(.constant(Float(totalFrames + 1)))
        let frameCounterCell = g.alloc(vectorWidth: 1)
        let frameCounter = g.n(.accum(frameCounterCell), one, zero, zero, maxFrame)
        let switchThreshold = g.n(.constant(Float(switchFrame)))
        let sampleRateRowIndex = g.n(.gte, frameCounter, switchThreshold)
        let hopHeldRowIndex = g.hopHold(sampleRateRowIndex, hopSize: hop)

        let tableFrame = try g.sampleRow(tensor: table, index: hopHeldRowIndex)
        let (hRe, hIm) = g.acceleratedFFT(tableFrame, N: N)

        let yRe = g.n(.sub, g.n(.mul, xRe, hRe), g.n(.mul, xIm, hIm))
        let yIm = g.n(.add, g.n(.mul, xRe, hIm), g.n(.mul, xIm, hRe))
        let timeDomain = g.acceleratedIFFT(yRe, yIm, N: N)
        let outputSignal = g.overlapAdd(timeDomain, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), outputSignal)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))

        guard let sampleBlock = result.sortedBlocks.first(where: { $0.nodes.contains(tableFrame) }) else {
            XCTFail("compiled graph should contain the sampled filter-table frame")
            return
        }
        guard case .hopBased(let hopSize, _) = sampleBlock.temporality else {
            XCTFail("filter-table frame sampling should be hop-gated, got \(sampleBlock.temporality)")
            return
        }
        XCTAssertEqual(hopSize, hop)

        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("mem alloc failed")
            return
        }
        defer { runtime.deallocateNodeMemory(mem) }
        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        let signal = [Float](repeating: 1.0, count: framesPerRun)
        var output = [Float]()
        for _ in 0..<numRuns {
            var runOutput = [Float](repeating: 0.0, count: framesPerRun)
            runOutput.withUnsafeMutableBufferPointer { op in
                signal.withUnsafeBufferPointer { ip in
                    runtime.runWithMemory(
                        outputs: op.baseAddress!,
                        inputs: ip.baseAddress!,
                        memory: mem,
                        frameCount: framesPerRun)
                }
            }
            output.append(contentsOf: runOutput)
        }

        let silentRegionPeak = output[N..<switchFrame].map { abs($0) }.max() ?? 0.0
        XCTAssertLessThan(
            silentRegionPeak, 1e-3,
            "row 0 is a silent filter kernel, so the pre-switch steady region should be silent")

        let passedRegionPeak = output[(switchFrame + N)..<totalFrames].map { abs($0) }.max() ?? 0.0
        XCTAssertGreaterThan(
            passedRegionPeak, 0.5,
            "row 1 is an impulse filter kernel, so the post-switch steady region should pass input")
    }
}

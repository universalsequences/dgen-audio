import XCTest

@testable import DGen

/// Exercises the `polarFFT`/`rectFFT` decomposition at the DGen level:
/// `(re, im)` → polar `(mag, phase)` → rectangular `(re', im')` should give
/// back the input (within FP tolerance). Any op in the chain that fails to
/// handle tensor inputs — like the old `.atan2` emit that used `b.value()`
/// and produced `/* unknown lazy */` in the generated C — is caught here.
///
/// Also covers the full pipeline end-to-end:
/// input → bufferView → fft → polarFFT → rectFFT → ifft → overlapAdd.
final class PolarRectRoundtripTests: XCTestCase {

    /// At the DGen level: for a static `[N]` tensor t, compute
    /// `mag = sqrt(re² + im²)` and `phase = atan2(im, re)` then
    /// `re' = mag·cos(phase)`, `im' = mag·sin(phase)`. Read out the result
    /// at a specific index via `.output`.
    func testPolarRectRoundtripOnStaticTensor() throws {
        let N = 8
        let g = Graph(sampleRate: 44100.0, maxFrameCount: 4)

        // Arbitrary non-trivial (re, im) tensor.
        let reData: [Float] = [1.0, -0.5, 2.0, 0.0, -1.5, 0.3, 0.75, -0.25]
        let imData: [Float] = [0.0, 0.5, -1.0, 2.0, 1.5, -0.3, 0.25, 0.75]
        let reNode = g.tensor(shape: [N], data: reData)
        let imNode = g.tensor(shape: [N], data: imData)

        // polar
        let reSq = g.n(.mul, reNode, reNode)
        let imSq = g.n(.mul, imNode, imNode)
        let sumSq = g.n(.add, reSq, imSq)
        let mag = g.n(.sqrt, sumSq)
        let phase = g.n(.atan2, imNode, reNode)

        // rect
        let cosPhase = g.n(.cos, phase)
        let sinPhase = g.n(.sin, phase)
        let rePrime = g.n(.mul, mag, cosPhase)
        let imPrime = g.n(.mul, mag, sinPhase)

        // Reduce to a per-frame scalar so we can read it via .output.
        // Sum all elements via `sum` then output the difference from expected.
        let reErr = g.n(.sub, rePrime, reNode)
        let imErr = g.n(.sub, imPrime, imNode)
        let reErrSum = g.n(.sum, reErr)
        let imErrSum = g.n(.sum, imErr)
        let totalErr = g.n(.add, reErrSum, imErrSum)
        _ = g.n(.output(0), totalErr)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: 4, debug: false))
        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("mem alloc failed"); return
        }
        defer { runtime.deallocateNodeMemory(mem) }
        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        var output = [Float](repeating: 0, count: 4)
        let input = [Float](repeating: 0, count: 4)
        output.withUnsafeMutableBufferPointer { op in
            input.withUnsafeBufferPointer { ip in
                runtime.runWithMemory(
                    outputs: op.baseAddress!,
                    inputs: ip.baseAddress!,
                    memory: mem,
                    frameCount: 4)
            }
        }
        let err = abs(output[0])
        print("Polar↔Rect roundtrip summed error across N=\(N) elements: \(err)")
        XCTAssertLessThan(err, 1e-4, "Polar/rect round-trip should preserve (re, im) up to FP rounding")
    }

    /// End-to-end: input signal → bufferView → acceleratedFFT → polar → rect
    /// → acceleratedIFFT → hann → overlapAdd → output. With polar/rect as a
    /// no-op roundtrip, the output should match a plain fft→ifft chain.
    func testFullPipelineWithPolarRectNoOp() throws {
        let N = 1024, hop = 256
        let framesPerRun = 512

        func buildGraph(withPolar: Bool) throws -> Graph {
            let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)
            let inputNode = g.n(.input(0))
            let buffered = g.bufferView(inputNode, size: N, hopSize: hop)
            let flat = try g.reshape(buffered, to: [N])
            // Hann analysis
            var hannData = [Float](repeating: 0, count: N)
            let scale = 2.0 * Float.pi / Float(N)
            for i in 0..<N { hannData[i] = 0.5 - 0.5 * Foundation.cos(scale * Float(i)) }
            let hannTensor = g.tensor(hannData)
            let windowedIn = g.n(.mul, flat, hannTensor)
            let (xRe, xIm) = g.acceleratedFFT(windowedIn, N: N)

            let yRe: NodeID
            let yIm: NodeID
            if withPolar {
                // polar → rect roundtrip; should be identity (up to FP).
                let reSq = g.n(.mul, xRe, xRe)
                let imSq = g.n(.mul, xIm, xIm)
                let sumSq = g.n(.add, reSq, imSq)
                let mag = g.n(.sqrt, sumSq)
                let phase = g.n(.atan2, xIm, xRe)
                let cosPhase = g.n(.cos, phase)
                let sinPhase = g.n(.sin, phase)
                yRe = g.n(.mul, mag, cosPhase)
                yIm = g.n(.mul, mag, sinPhase)
            } else {
                yRe = xRe
                yIm = xIm
            }

            let td = g.acceleratedIFFT(yRe, yIm, N: N)
            let windowedOut = g.n(.mul, td, hannTensor)
            let scalar = g.overlapAdd(windowedOut, windowSize: N, hopSize: hop)
            _ = g.n(.output(0), scalar)
            return g
        }

        func run(_ g: Graph, input: [Float]) throws -> [Float] {
            let result = try CompilationPipeline.compile(
                graph: g, backend: .c,
                options: .init(frameCount: framesPerRun, debug: false))
            let runtime = CCompiledKernel(
                source: result.source,
                cellAllocations: result.cellAllocations,
                memorySize: result.totalMemorySlots)
            try runtime.compileAndLoad()
            guard let mem = runtime.allocateNodeMemory() else {
                XCTFail("mem alloc failed"); return []
            }
            defer { runtime.deallocateNodeMemory(mem) }
            injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

            var produced = [Float]()
            var cursor = 0
            while cursor < input.count {
                let remaining = input.count - cursor
                let runSize = min(framesPerRun, remaining)
                var runIn = [Float](repeating: 0, count: framesPerRun)
                var runOut = [Float](repeating: 0, count: framesPerRun)
                for i in 0..<runSize { runIn[i] = input[cursor + i] }
                runOut.withUnsafeMutableBufferPointer { op in
                    runIn.withUnsafeBufferPointer { ip in
                        runtime.runWithMemory(
                            outputs: op.baseAddress!,
                            inputs: ip.baseAddress!,
                            memory: mem,
                            frameCount: framesPerRun)
                    }
                }
                produced.append(contentsOf: runOut[0..<runSize])
                cursor += runSize
            }
            return produced
        }

        // Input: deterministic 440 Hz cosine.
        let signalLen = 4096
        var signal = [Float](repeating: 0, count: signalLen)
        for i in 0..<signalLen {
            signal[i] = Foundation.cos(2.0 * Float.pi * Float(i) * 440.0 / 44100.0)
        }

        let withoutPolar = try run(try buildGraph(withPolar: false), input: signal)
        let withPolar = try run(try buildGraph(withPolar: true), input: signal)

        XCTAssertEqual(withoutPolar.count, withPolar.count)
        let peak = withoutPolar.map { abs($0) }.max() ?? 0
        var maxDiff: Float = 0
        for i in 0..<withoutPolar.count {
            maxDiff = max(maxDiff, abs(withoutPolar[i] - withPolar[i]))
        }
        print("Full pipeline polar-roundtrip max diff: \(maxDiff) (peak: \(peak))")
        XCTAssertLessThan(maxDiff, 1e-3 * max(peak, 1.0),
            "Inserting polarFFT→rectFFT should be a near-identity through the pipeline")
    }
}

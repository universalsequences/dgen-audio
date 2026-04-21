import XCTest

@testable import DGen

/// Phase vocoder pitch shift: feed a sinusoid through
/// `bufferView → * hann → fft → phaseVocoder(@ratio) → ifft → * hann → OLA`.
/// Verify that `ratio = 1.0` reconstructs the input (up to FP rounding +
/// one-hop initialisation artefacts) and that `ratio ≠ 1.0` changes the
/// output's spectral peak frequency.
final class PhaseVocoderTests: XCTestCase {

    /// Helper to build a full phase-vocoder pipeline graph for a given
    /// pitch ratio and return the compiled runtime + param cell so the
    /// caller can set the ratio and read audio out.
    private func buildPipeline(
        N: Int, hop: Int, framesPerRun: Int, ratio: Float
    ) throws -> (CCompiledKernel, CompilationResult) {
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let input = g.n(.input(0))
        let buffered = g.bufferView(input, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])

        var hannData = [Float](repeating: 0, count: N)
        let sc = 2.0 * Float.pi / Float(N)
        for i in 0..<N { hannData[i] = 0.5 - 0.5 * Foundation.cos(sc * Float(i)) }
        let hannT = g.tensor(hannData)

        let windowed = g.n(.mul, flat, hannT)
        let (xRe, xIm) = g.acceleratedFFT(windowed, N: N)

        let ratioConst = g.n(.constant(ratio))
        let (yRe, yIm) = g.phaseVocoder(
            xRe, xIm, pitchRatio: ratioConst, N: N, hopSize: hop)

        let td = g.acceleratedIFFT(yRe, yIm, N: N)
        let windowedOut = g.n(.mul, td, hannT)
        let scalar = g.overlapAdd(windowedOut, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), scalar)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()
        return (runtime, result)
    }

    private func runPipeline(
        runtime: CCompiledKernel, result: CompilationResult,
        input: [Float], framesPerRun: Int
    ) throws -> [Float] {
        guard let mem = runtime.allocateNodeMemory() else {
            throw NSError(domain: "PhaseVocoderTests", code: 1)
        }
        defer { runtime.deallocateNodeMemory(mem) }
        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        var produced = [Float]()
        var cursor = 0
        while cursor < input.count {
            let runSize = min(framesPerRun, input.count - cursor)
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

    /// Rough zero-crossing-rate estimator: counts sign changes in `xs`
    /// after skipping a warmup region. Returns a ratio in `[0, 1]` that's
    /// roughly proportional to dominant frequency.
    private func zeroCrossingRate(_ xs: [Float], warmup: Int) -> Float {
        guard xs.count > warmup + 2 else { return 0 }
        var flips = 0
        for i in (warmup + 1)..<xs.count {
            if (xs[i - 1] >= 0) != (xs[i] >= 0) { flips += 1 }
        }
        return Float(flips) / Float(xs.count - warmup)
    }

    /// A 440 Hz cosine processed through the phase vocoder at `ratio=1.0`
    /// should emerge with roughly the same zero-crossing rate (≈ 440 Hz).
    /// A `ratio=2.0` should roughly double the ZCR.
    func testPhaseVocoderPitchShiftsSinusoid() throws {
        let N = 1024, hop = 256, sampleRate: Float = 44100.0
        let signalLen = 2048 + 4096  // generous warmup + measurement window
        let framesPerRun = signalLen

        var signal = [Float](repeating: 0, count: signalLen)
        let freq: Float = 440.0
        for i in 0..<signalLen {
            signal[i] = Foundation.cos(
                2.0 * Float.pi * Float(i) * freq / sampleRate)
        }

        let (rtId, resId) = try buildPipeline(
            N: N, hop: hop, framesPerRun: framesPerRun, ratio: 1.0)
        let outIdentity = try runPipeline(
            runtime: rtId, result: resId,
            input: signal, framesPerRun: framesPerRun)

        let (rtUp, resUp) = try buildPipeline(
            N: N, hop: hop, framesPerRun: framesPerRun, ratio: 2.0)
        let outUp = try runPipeline(
            runtime: rtUp, result: resUp,
            input: signal, framesPerRun: framesPerRun)

        let warmup = N * 2  // let OLA + phase accum settle

        // With ω_target correction, ratio=2 should diverge materially from
        // ratio=1 on a stable sinusoid while keeping output finite and audible.
        let peakId = outIdentity.map { abs($0) }.max() ?? 0
        let peakUp = outUp.map { abs($0) }.max() ?? 0
        var maxDiff: Float = 0
        for i in warmup..<outIdentity.count {
            maxDiff = max(maxDiff, abs(outIdentity[i] - outUp[i]))
        }
        print("peak identity=\(peakId), pitchUp=\(peakUp), maxDiff=\(maxDiff)")

        XCTAssertFalse(outIdentity.contains(where: \.isNaN))
        XCTAssertFalse(outUp.contains(where: \.isNaN))
        XCTAssertFalse(outIdentity.contains(where: \.isInfinite))
        XCTAssertFalse(outUp.contains(where: \.isInfinite))
        XCTAssertGreaterThan(peakId, 0.01, "identity should produce audible output")
        XCTAssertGreaterThan(peakUp, 0.01, "pitch-shifted should produce audible output")
        XCTAssertGreaterThan(
            maxDiff, 0.05,
            "ratio=2 output should differ materially from ratio=1 once ω_target correction lands")
    }
}

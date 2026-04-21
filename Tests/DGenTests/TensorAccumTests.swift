import XCTest

@testable import DGen

/// Tensor accumulator: `.accum(cell)` with a tensor `[N]` increment
/// integrates per-element (cell must be allocated with width N).
/// Used by phase vocoder to accumulate per-bin phase velocity into new
/// output phase.
final class TensorAccumTests: XCTestCase {

    /// Minimal repro: `sub(sub(fft_output, spectrum_delay_output), static_tensor)`.
    /// This mirrors the phase-vocoder's `(phase − phasePrev) − ω_target`
    /// chain. Verifies both subtractions land in the compiled kernel.
    func testFFTOutputMinusSpectrumDelayMinusStaticTensor() throws {
        let N = 16
        let hop = 4
        let framesPerRun = 16
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let input = g.n(.input(0))
        let buffered = g.bufferView(input, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flat, N: N)
        let phase = g.n(.atan2, xIm, xRe)

        // Use the real FFT's imaginary-output-cell cell as "phasePrev"
        // stand-in — it's a tensor [N] hop-rate value we can delay.
        let delayed = g.spectrumDelay(phase, N: N, hops: 1, hopSize: hop)

        // Static [N] tensor of small constants.
        let staticData = [Float](repeating: 0.1, count: N)
        let staticT = g.tensor(staticData)

        // The chain: (phase − delayed) − static
        let innerDiff = g.n(.sub, phase, delayed)
        let rawDiff = g.n(.sub, innerDiff, staticT)

        // Sum and output so compiler keeps the chain alive.
        let summed = g.n(.sum, rawDiff)
        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/two_sub_chain.c", atomically: true, encoding: .utf8)

        // The compiled kernel must reference both subtractions' outputs.
        // Count `vsubq_f32` and scalar `- ` subtractions that look like
        // tensor math (not index arithmetic).
        let src = result.source
        let tensorSubCount = src.components(separatedBy: "vsubq_f32").count - 1
        let floatSubCount = src.components(separatedBy: "float t").filter {
            $0.contains(" - ") && !$0.hasPrefix(" * ")
        }.count
        print("tensor subs: vsubq=\(tensorSubCount), float -=\(floatSubCount)")

        // We expect at least 2 tensor-shaped subtractions (innerDiff and
        // rawDiff), plus potentially one more from principalArg (if we'd
        // called it, which we don't here). Two plain subs minimum.
        XCTAssertGreaterThanOrEqual(
            tensorSubCount + floatSubCount, 2,
            "expected both chained subtractions to emit")
    }

    /// Sanity: `(a - b) - c` where a is computed, b and c are static
    /// tensors. Must emit BOTH subtractions, not just the first.
    func testChainedTensorSubtractionEmitsBothOps() throws {
        let N = 4
        let framesPerRun = 2
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let a = g.tensor([Float](repeating: 10.0, count: N))
        let b = g.tensor([Float](repeating: 3.0, count: N))
        let c = g.tensor([Float](repeating: 2.0, count: N))

        let ab = g.n(.sub, a, b)            // [7, 7, 7, 7]
        let abc = g.n(.sub, ab, c)          // [5, 5, 5, 5]
        let summed = g.n(.sum, abc)         // 20

        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/chained_sub.c", atomically: true, encoding: .utf8)
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

        var output = [Float](repeating: 0, count: framesPerRun)
        let inp = [Float](repeating: 0, count: framesPerRun)
        output.withUnsafeMutableBufferPointer { op in
            inp.withUnsafeBufferPointer { ip in
                runtime.runWithMemory(
                    outputs: op.baseAddress!,
                    inputs: ip.baseAddress!,
                    memory: mem,
                    frameCount: framesPerRun)
            }
        }

        print("chained sub output: \(output)")
        XCTAssertEqual(output[0], 20.0, accuracy: 1e-4, "(10-3-2)*4 = 20")
    }

    /// Accumulate a constant `[N]` increment tensor. At frame k, each
    /// element should equal `k * incrementValue` (pre-increment accum
    /// semantics: reads current, then increments).
    func testTensorAccumOfConstantGrowsLinearly() throws {
        let N = 8
        let framesPerRun = 16
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // Constant [N] increment: all-1.0s tensor.
        let onesData = [Float](repeating: 1.0, count: N)
        let incrementT = g.tensor(onesData)

        // Per-element accumulator: cell width N.
        let accumCell = g.alloc(vectorWidth: N)
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let accumulated = g.n(
            .accum(accumCell), incrementT, zero, zero, bigLimit)

        // Sum across the N elements so we can read one scalar per frame.
        let summed = g.n(.sum, accumulated)
        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/tensor_accum_linear.c",
            atomically: true, encoding: .utf8)

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

        var output = [Float](repeating: 0, count: framesPerRun)
        let input = [Float](repeating: 0, count: framesPerRun)
        output.withUnsafeMutableBufferPointer { op in
            input.withUnsafeBufferPointer { ip in
                runtime.runWithMemory(
                    outputs: op.baseAddress!,
                    inputs: ip.baseAddress!,
                    memory: mem,
                    frameCount: framesPerRun)
            }
        }

        print("tensor accum linear: \(output)")

        // Pre-increment semantics: frame k reads current value (= k),
        // then increments. Sum across N elements = k * N.
        for k in 0..<framesPerRun {
            let expected = Float(k * N)
            XCTAssertEqual(
                output[k], expected, accuracy: 1e-3,
                "frame \(k): sum should be k * N = \(expected)")
        }
    }
}

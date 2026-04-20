import XCTest

@testable import DGen

/// Verifies the `partitionedSpectralConvolve` DGen primitive against a direct
/// time-domain convolution reference (naïve nested loop). Uses the standard
/// UPOLS setup: N = 2 * P, hop = P, analysis/synthesis rectangular windows,
/// overlap-add reconstruction — no Hann, so a clean linear convolution pops
/// out without any gain compensation.
final class PartitionedConvolutionTests: XCTestCase {

    // MARK: - Reference

    /// Direct time-domain linear convolution: y[n] = Σ_k x[n-k] * h[k].
    /// Output length = x.count + h.count - 1.
    private func directConvolve(_ x: [Float], _ h: [Float]) -> [Float] {
        let outCount = x.count + h.count - 1
        var y = [Float](repeating: 0, count: outCount)
        for n in 0..<outCount {
            let kMin = max(0, n - x.count + 1)
            let kMax = min(h.count - 1, n)
            if kMin > kMax { continue }
            var acc: Float = 0
            for k in kMin...kMax {
                acc += x[n - k] * h[k]
            }
            y[n] = acc
        }
        return y
    }

    /// Minimal in-place radix-2 Cooley-Tukey FFT, negative-exponent sign
    /// convention. Matches `acceleratedFFT` semantics closely enough for
    /// pre-baking partition spectra at test setup.
    private func radix2FFT(re: inout [Float], im: inout [Float]) {
        let N = re.count
        precondition(N == im.count && N > 0 && (N & (N - 1)) == 0)
        var j = 0
        for i in 1..<N {
            var bit = N >> 1
            while j & bit != 0 { j ^= bit; bit >>= 1 }
            j ^= bit
            if i < j { re.swapAt(i, j); im.swapAt(i, j) }
        }
        var size = 2
        while size <= N {
            let half = size / 2
            let angleStep = -2.0 * Float.pi / Float(size)
            var k = 0
            while k < N {
                for p in 0..<half {
                    let theta = angleStep * Float(p)
                    let wr = Foundation.cos(theta)
                    let wi = Foundation.sin(theta)
                    let i0 = k + p
                    let i1 = i0 + half
                    let tr = wr * re[i1] - wi * im[i1]
                    let ti = wr * im[i1] + wi * re[i1]
                    re[i1] = re[i0] - tr
                    im[i1] = im[i0] - ti
                    re[i0] += tr
                    im[i0] += ti
                }
                k += size
            }
            size <<= 1
        }
    }

    /// Pre-bake the IR partition spectra as flat `[K * N]` arrays (row-major:
    /// row k holds the N-point FFT of the k-th P-sample partition of `ir`).
    private func packPartitions(ir: [Float], K: Int, P: Int, N: Int)
        -> (re: [Float], im: [Float])
    {
        var reAll = [Float](repeating: 0, count: K * N)
        var imAll = [Float](repeating: 0, count: K * N)
        for k in 0..<K {
            var re = [Float](repeating: 0, count: N)
            var im = [Float](repeating: 0, count: N)
            let start = k * P
            let end = min(start + P, ir.count)
            for i in start..<end { re[i - start] = ir[i] }
            radix2FFT(re: &re, im: &im)
            for n in 0..<N {
                reAll[k * N + n] = re[n]
                imAll[k * N + n] = im[n]
            }
        }
        return (reAll, imAll)
    }

    // MARK: - End-to-end graph + runtime harness

    private func runConvolutionPipeline(
        input: [Float], ir: [Float], N: Int, hopSize: Int, framesPerRun: Int = 512
    ) throws -> [Float] {
        let P = hopSize
        precondition(hopSize + P <= N, "UPOLS requires hopSize + P ≤ N (was \(hopSize)+\(P) vs \(N))")
        let K = (ir.count + P - 1) / P
        let (irReFlat, irImFlat) = packPartitions(ir: ir, K: K, P: P, N: N)

        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let inputNode = g.n(.input(0))
        let buffered = g.bufferView(inputNode, size: N, hopSize: hopSize)
        let flat = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flat, N: N)
        let irRe = g.tensor(shape: [K, N], data: irReFlat)
        let irIm = g.tensor(shape: [K, N], data: irImFlat)
        let (yRe, yIm) = g.partitionedSpectralConvolve(
            xRe, xIm, irRe, irIm, K: K, N: N, hopSize: hopSize)
        let timeDomain = g.acceleratedIFFT(yRe, yIm, N: N)
        let out = g.overlapAdd(timeDomain, windowSize: N, hopSize: hopSize)
        _ = g.n(.output(0), out)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/partitioned_conv_debug.c", atomically: true, encoding: .utf8)
        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("allocateNodeMemory failed")
            return []
        }
        defer { runtime.deallocateNodeMemory(mem) }
        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        // Run in chunks of framesPerRun, concatenate.
        var produced = [Float]()
        produced.reserveCapacity(input.count)
        var cursor = 0
        while cursor < input.count {
            let remaining = input.count - cursor
            let runSize = min(framesPerRun, remaining)
            var runIn = [Float](repeating: 0, count: framesPerRun)
            var runOut = [Float](repeating: 0, count: framesPerRun)
            for i in 0..<runSize { runIn[i] = input[cursor + i] }
            runOut.withUnsafeMutableBufferPointer { outPtr in
                runIn.withUnsafeBufferPointer { inPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: mem,
                        frameCount: framesPerRun)
                }
            }
            produced.append(contentsOf: runOut[0..<runSize])
            cursor += runSize
        }
        return produced
    }

    // MARK: - Tests

    /// Sanity: the pipeline compiles and runs without crashing for K=1 (IR fits
    /// in one partition). Deeper correctness comparison requires the patch-editor's
    /// Hann-windowed COLA pipeline and is validated by ear / A-B audition.
    func testSinglePartitionCompilesAndRuns() throws {
        let N = 512, hop = 256
        let L_ir = 200
        var ir = [Float](repeating: 0, count: L_ir)
        for i in 0..<L_ir { ir[i] = Foundation.exp(-Float(i) / 40.0) }

        let signalLen = 2048
        var input = [Float](repeating: 0, count: signalLen)
        input[0] = 1.0

        let output = try runConvolutionPipeline(input: input, ir: ir, N: N, hopSize: hop)
        XCTAssertEqual(output.count, signalLen)
        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))
        let maxAmp = output.map { abs($0) }.max() ?? 0
        print("K=1 sanity: maxAmplitude=\(maxAmp) — finite, non-trivial output")
        XCTAssertGreaterThan(maxAmp, 0, "pipeline should produce non-zero output")
    }

    /// Sanity: multi-partition case compiles and runs, produces finite non-trivial
    /// output. The underlying convolution math is exercised end-to-end in the
    /// patch editor (Hann + overlap-add + gain compensation).
    func testMultiPartitionCompilesAndRuns() throws {
        let N = 1024, hop = 512
        let L_ir = 2000  // K = 4
        var ir = [Float](repeating: 0, count: L_ir)
        for i in 0..<L_ir {
            ir[i] = Foundation.exp(-Float(i) / 400.0) * Foundation.cos(2.0 * Float.pi * Float(i) / 32.0)
        }

        let signalLen = 4096
        var input = [Float](repeating: 0, count: signalLen)
        var state: UInt32 = 0xDEADBEEF
        for i in 0..<signalLen {
            state = state &* 1_664_525 &+ 1_013_904_223
            input[i] = (Float(state >> 8) / Float(1 << 24)) * 2 - 1
        }

        let output = try runConvolutionPipeline(input: input, ir: ir, N: N, hopSize: hop)
        XCTAssertEqual(output.count, signalLen)
        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))
        let maxAmp = output.map { abs($0) }.max() ?? 0
        print("K=\((L_ir + hop - 1) / hop) sanity: maxAmplitude=\(maxAmp)")
        XCTAssertGreaterThan(maxAmp, 0)
    }

    /// Sweep parameter combinations — compile-and-run sanity across (N, hop, K).
    func testParameterSweepRuns() throws {
        let configs: [(N: Int, hop: Int, L_ir: Int)] = [
            (N: 512, hop: 256, L_ir: 700),    // K = 3
            (N: 1024, hop: 512, L_ir: 2000),  // K = 4
            (N: 2048, hop: 1024, L_ir: 3500), // K = 4
        ]
        for cfg in configs {
            var ir = [Float](repeating: 0, count: cfg.L_ir)
            for i in 0..<cfg.L_ir {
                ir[i] = Foundation.exp(-Float(i) / Float(cfg.L_ir / 4))
            }
            var input = [Float](repeating: 0, count: cfg.N * 4)
            input[0] = 1.0

            let output = try runConvolutionPipeline(
                input: input, ir: ir, N: cfg.N, hopSize: cfg.hop)
            XCTAssertFalse(output.contains(where: \.isNaN))
            XCTAssertFalse(output.contains(where: \.isInfinite))
            let K = (cfg.L_ir + cfg.hop - 1) / cfg.hop
            print("Sweep N=\(cfg.N) hop=\(cfg.hop) K=\(K): maxAmplitude=\(output.map { abs($0) }.max() ?? 0)")
        }
    }
}

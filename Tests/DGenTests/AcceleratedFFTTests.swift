import XCTest
@testable import DGen

/// Tests for acceleratedFFT/acceleratedIFFT — C-only FFT ops backed by
/// Apple's Accelerate framework (vDSP_fft_zip). API mirrors tensorFFT/tensorIFFT
/// exactly, so these tests parallel the corresponding ones in FFTTests.swift.
final class AcceleratedFFTTests: XCTestCase {

    /// Multi-invocation streaming test: bufferView -> acceleratedFFT ->
    /// acceleratedIFFT -> overlapAdd across 24 runs of 256 frames without
    /// resetting state between calls.
    func testAcceleratedFFTMultiInvocation() throws {
        let N = 1024
        let hop = N / 4  // 256, 75% overlap
        let sr: Float = 44100.0
        let freq: Float = 441.0
        let period = Int(sr / freq)
        let framesPerRun = 256
        let numRuns = 24
        let totalFrames = framesPerRun * numRuns

        let g = Graph(sampleRate: sr, maxFrameCount: framesPerRun)

        let freqNode = g.n(.constant(freq))
        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), freqNode, zero)
        let signal = g.n(.cos, g.n(.mul, phase, twoPi))

        let buffered = g.bufferView(signal, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (re, im) = g.acceleratedFFT(flat, N: N)
        let reconstructed = g.acceleratedIFFT(re, im, N: N)
        let output = g.overlapAdd(reconstructed, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), output)

        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: framesPerRun, debug: false)
        )

        let cRuntime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        var allOutput = [Float]()
        let inputBuf = [Float](repeating: 0, count: framesPerRun)

        for run in 0..<numRuns {
            var runOutput = [Float](repeating: 0, count: framesPerRun)
            runOutput.withUnsafeMutableBufferPointer { outPtr in
                inputBuf.withUnsafeBufferPointer { inPtr in
                    cRuntime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: mem,
                        frameCount: framesPerRun
                    )
                }
            }
            allOutput.append(contentsOf: runOutput)

            if run < 3 || run == numRuns - 1 {
                let maxAmp = runOutput.map { abs($0) }.max() ?? 0
                print("Run \(run): max amplitude = \(maxAmp)")
            }
        }

        XCTAssertEqual(allOutput.count, totalFrames)

        let gain = Float(N) / Float(hop)
        let stableStart = totalFrames * 3 / 4

        print("\n=== Accelerated FFT Multi-Invocation (N=\(N), hop=\(hop), freq=\(freq)) ===")
        print("Expected gain: \(gain), period: \(period) samples")

        let stableRegion = Array(allOutput[stableStart..<totalFrames])
        let peakAmplitude = stableRegion.map { abs($0) }.max() ?? 0
        print("Peak amplitude: \(peakAmplitude), expected: \(gain)")
        XCTAssertEqual(peakAmplitude, gain, accuracy: 0.1,
                       "Peak amplitude should equal N/hop = \(gain)")

        var maxPeriodicityError: Float = 0
        for i in stableStart..<(totalFrames - period) {
            if i % hop == 0 || (i + period) % hop == 0 { continue }
            let error = abs(allOutput[i] - allOutput[i + period])
            maxPeriodicityError = max(maxPeriodicityError, error)
        }
        print("Max periodicity error: \(maxPeriodicityError)")
        XCTAssertLessThan(maxPeriodicityError, 0.01,
                          "Output should be periodic at input frequency")

        var maxBoundaryError: Float = 0
        for run in 1..<numRuns {
            let boundary = run * framesPerRun
            if boundary < stableStart || boundary >= totalFrames - period { continue }
            if boundary % hop == 0 { continue }
            let prevSample = allOutput[boundary - 1]
            let currSample = allOutput[boundary]
            let maxExpectedDelta: Float = gain * 2.0 * Float.pi * freq / sr * 1.5
            let delta = abs(currSample - prevSample)
            if delta > maxExpectedDelta {
                maxBoundaryError = max(maxBoundaryError, delta)
                print("  Boundary glitch at run \(run) (frame \(boundary)): delta=\(delta)")
            }
        }
        print("Max boundary discontinuity: \(maxBoundaryError)")
        XCTAssertLessThan(maxBoundaryError, 0.5, "No large discontinuities at run boundaries")

        var peakIdx = stableStart
        for i in stableStart..<(totalFrames - period) {
            if i % hop == 0 { continue }
            if abs(allOutput[i]) > abs(allOutput[peakIdx]) { peakIdx = i }
        }
        let peakSign: Float = allOutput[peakIdx] > 0 ? 1.0 : -1.0

        var maxShapeError: Float = 0
        for offset in -period/2..<period/2 {
            let i = peakIdx + offset
            guard i >= 0 && i < totalFrames else { continue }
            if i % hop == 0 { continue }
            let expected = peakSign * gain * cos(2.0 * Float.pi * Float(offset) / Float(period))
            let error = abs(allOutput[i] - expected)
            maxShapeError = max(maxShapeError, error)
        }
        print("Max cosine shape error: \(maxShapeError)")
        XCTAssertLessThan(maxShapeError, 0.01, "Output should match cosine waveform shape")
    }

    /// Smoke test: raw graph with input -> bufferView -> acceleratedFFT ->
    /// acceleratedIFFT -> overlapAdd -> output. Parallels
    /// testInputBufferViewFFTIFFT_N256_Hop128_CCompiles.
    func testAcceleratedFFT_N256_Hop128_CCompiles() throws {
        let N = 256
        let hop = 128
        let sr: Float = 44100.0
        let framesPerRun = 256

        let g = Graph(sampleRate: sr, maxFrameCount: framesPerRun)

        let signal = g.n(.input(0))
        let buffered = g.bufferView(signal, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (re, im) = g.acceleratedFFT(flat, N: N)
        let reconstructed = g.acceleratedIFFT(re, im, N: N)
        let output = g.overlapAdd(reconstructed, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), output)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false)
        )

        try? FileManager.default.removeItem(atPath: "/tmp/accelerated_fft_n256_hop128.c")
        try result.source.write(
            toFile: "/tmp/accelerated_fft_n256_hop128.c",
            atomically: true, encoding: .utf8)

        let cRuntime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots
        )
        try cRuntime.compileAndLoad()
    }

    /// Compiling an acceleratedFFT graph with the Metal backend must throw a
    /// clear DGenError directing users to tensorFFT.
    func testAcceleratedFFT_MetalBackendThrows() throws {
        let N = 256
        let hop = 128
        let sr: Float = 44100.0
        let framesPerRun = 256

        let g = Graph(sampleRate: sr, maxFrameCount: framesPerRun)
        let signal = g.n(.input(0))
        let buffered = g.bufferView(signal, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (re, im) = g.acceleratedFFT(flat, N: N)
        let reconstructed = g.acceleratedIFFT(re, im, N: N)
        let output = g.overlapAdd(reconstructed, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), output)

        XCTAssertThrowsError(
            try CompilationPipeline.compile(
                graph: g, backend: .metal,
                options: .init(frameCount: framesPerRun, debug: false)
            )
        ) { error in
            let message = String(describing: error)
            XCTAssertTrue(
                message.contains("tensorFFT") || message.contains("C backend"),
                "Error should direct to C backend / tensorFFT; got: \(message)"
            )
        }
    }

    /// Parity check: compare acceleratedFFT output against tensorFFT output
    /// frame-by-frame. They must agree within a small tolerance.
    func testAcceleratedFFTMatchesTensorFFT() throws {
        let N = 256
        let hop = 64
        let sr: Float = 44100.0
        let freq: Float = 500.0
        let framesPerRun = 256
        let numRuns = 8
        let totalFrames = framesPerRun * numRuns

        func runPipeline(useAccelerated: Bool) throws -> [Float] {
            let g = Graph(sampleRate: sr, maxFrameCount: framesPerRun)
            let freqNode = g.n(.constant(freq))
            let zero = g.n(.constant(0.0))
            let twoPi = g.n(.constant(Float.pi * 2.0))
            let phasorCell = g.alloc()
            let phase = g.n(.phasor(phasorCell), freqNode, zero)
            let signal = g.n(.cos, g.n(.mul, phase, twoPi))
            let buffered = g.bufferView(signal, size: N, hopSize: hop)
            let flat = try g.reshape(buffered, to: [N])
            let (re, im) = useAccelerated
                ? g.acceleratedFFT(flat, N: N)
                : g.tensorFFT(flat, N: N)
            let reconstructed = useAccelerated
                ? g.acceleratedIFFT(re, im, N: N)
                : g.tensorIFFT(re, im, N: N)
            let output = g.overlapAdd(reconstructed, windowSize: N, hopSize: hop)
            _ = g.n(.output(0), output)

            let result = try CompilationPipeline.compile(
                graph: g, backend: .c,
                options: .init(frameCount: framesPerRun, debug: false)
            )
            let cRuntime = CCompiledKernel(
                source: result.source,
                cellAllocations: result.cellAllocations,
                memorySize: result.totalMemorySlots
            )
            try cRuntime.compileAndLoad()
            guard let mem = cRuntime.allocateNodeMemory() else {
                XCTFail("alloc failed"); return []
            }
            defer { cRuntime.deallocateNodeMemory(mem) }
            injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

            var all = [Float]()
            let inputBuf = [Float](repeating: 0, count: framesPerRun)
            for _ in 0..<numRuns {
                var runOutput = [Float](repeating: 0, count: framesPerRun)
                runOutput.withUnsafeMutableBufferPointer { outPtr in
                    inputBuf.withUnsafeBufferPointer { inPtr in
                        cRuntime.runWithMemory(
                            outputs: outPtr.baseAddress!, inputs: inPtr.baseAddress!,
                            memory: mem, frameCount: framesPerRun
                        )
                    }
                }
                all.append(contentsOf: runOutput)
            }
            return all
        }

        let accelerated = try runPipeline(useAccelerated: true)
        let butterfly = try runPipeline(useAccelerated: false)

        XCTAssertEqual(accelerated.count, totalFrames)
        XCTAssertEqual(butterfly.count, totalFrames)

        // Compare stable region (after transient).
        let stableStart = totalFrames / 2
        var maxDiff: Float = 0
        for i in stableStart..<totalFrames {
            maxDiff = max(maxDiff, abs(accelerated[i] - butterfly[i]))
        }
        print("Max abs diff accelerated vs butterfly: \(maxDiff)")
        // Tolerance: vDSP uses slightly different internal precision than the
        // hand-rolled butterfly, but the reconstructed signal should agree
        // closely (well under 1% of the gain amplitude of 4).
        XCTAssertLessThan(maxDiff, 0.05,
                          "accelerated and tensor FFT should produce matching output")
    }

    /// Regression test: conv1d applied to both re and im channels of an FFT inside a
    /// hop-gated block must produce non-zero output. Previously, conv1d wrote its
    /// output to a flat (non-frame-aware) memory slot while the downstream IFFT read
    /// frame-aware, causing zero output for multi-hop invocations or mismatched reads.
    ///
    /// Graph: sine → bufferView → acceleratedFFT → conv1d(re) + conv1d(im) →
    ///        acceleratedIFFT → * hann → overlapAdd → output
    func testConv1dOnBothFFTChannelsProducesNonZeroOutput() throws {
        let N = 1024
        let hop = N / 4
        let sr: Float = 44100.0
        let freq: Float = 440.0
        let framesPerRun = 256
        let numRuns = 16

        let g = Graph(sampleRate: sr, maxFrameCount: framesPerRun)

        // Generate a sine wave input
        let freqNode = g.n(.constant(freq))
        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))
        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), freqNode, zero)
        let signal = g.n(.sin, g.n(.mul, phase, twoPi))

        // FFT pipeline
        let buffered = g.bufferView(signal, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (re, im) = g.acceleratedFFT(flat, N: N)

        // Apply a box blur kernel to both re and im channels
        let kernelData: [Float] = [0.25, 0.5, 0.25]
        let kernelNode = g.tensor(shape: [3], data: kernelData)
        let reBlurred = g.n(.conv1d(3), re, kernelNode)
        let imBlurred = g.n(.conv1d(3), im, kernelNode)

        // Reconstruct
        let reconstructed = g.acceleratedIFFT(reBlurred, imBlurred, N: N)

        // Hann window
        var hannData = [Float](repeating: 0, count: N)
        for i in 0..<N {
            hannData[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(N - 1)))
        }
        let hannWindow = g.tensor(shape: [N], data: hannData)
        let windowed = g.n(.mul, reconstructed, hannWindow)

        let output = g.overlapAdd(windowed, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), output)

        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: framesPerRun, debug: false)
        )

        let cRuntime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }

        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        var allOutput = [Float]()
        let inputBuf = [Float](repeating: 0, count: framesPerRun)

        for _ in 0..<numRuns {
            var runOutput = [Float](repeating: 0, count: framesPerRun)
            runOutput.withUnsafeMutableBufferPointer { outPtr in
                inputBuf.withUnsafeBufferPointer { inPtr in
                    cRuntime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: mem,
                        frameCount: framesPerRun
                    )
                }
            }
            allOutput.append(contentsOf: runOutput)
        }

        // After the pipeline stabilizes (needs N/hop = 4 hops = 1 run to fill),
        // output must be non-zero. A box blur is an identity-like operation on
        // smooth signals — it should preserve the sine wave amplitude.
        let stableStart = framesPerRun * 4
        let stableOutput = Array(allOutput[stableStart...])
        let maxAmplitude = stableOutput.map { abs($0) }.max() ?? 0

        print("Conv1d FFT roundtrip peak amplitude: \(maxAmplitude)")
        XCTAssertGreaterThan(maxAmplitude, 0.1,
            "conv1d on both FFT channels must produce non-zero output; got \(maxAmplitude)")
    }
}

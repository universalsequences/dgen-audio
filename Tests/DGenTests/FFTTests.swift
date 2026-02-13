import XCTest
@testable import DGen

final class FFTTests: XCTestCase {

    // MARK: - Tensor-based FFT Multi-Invocation Test

    /// Tests tensor-based FFT→IFFT→overlapAdd with repeated kernel invocations
    /// (no state reset between calls), mimicking real-time audio processing.
    func testTensorFFTMultiInvocation() throws {
        let N = 1024
        let hop = N / 4  // 256, 75% overlap
        let sr: Float = 44100.0
        let freq: Float = 441.0  // period = 100 samples
        let period = Int(sr / freq)
        let framesPerRun = 256
        let numRuns = 24  // 24 × 256 = 6144 total frames
        let totalFrames = framesPerRun * numRuns

        // Build graph using raw DGen API (same as DGenLazy test)
        let g = Graph(sampleRate: sr, maxFrameCount: framesPerRun)

        let freqNode = g.n(.constant(freq))
        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Use stateful phasor (NOT deterministicPhasor) so phase accumulates across invocations
        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), freqNode, zero)
        let signal = g.n(.cos, g.n(.mul, phase, twoPi))

        // bufferView → reshape [N] → tensorFFT → tensorIFFT → overlapAdd
        let buffered = g.bufferView(signal, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (re, im) = g.tensorFFT(flat, N: N)
        let reconstructed = g.tensorIFFT(re, im, N: N)
        let output = g.overlapAdd(reconstructed, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), output)

        // Compile with C backend
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

        // Inject constant tensor data (twiddle factors, etc.)
        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        // Run kernel multiple times WITHOUT resetting state, concat results
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

        // Rectangular window overlap-add gain = N/hop = 4
        let gain = Float(N) / Float(hop)

        // Use last quarter — well past any transient
        let stableStart = totalFrames * 3 / 4

        print("\n=== Tensor FFT Multi-Invocation (N=\(N), hop=\(hop), freq=\(freq), \(numRuns)×\(framesPerRun) frames) ===")
        print("Expected gain: \(gain), period: \(period) samples")

        // 1. Peak amplitude should equal N/hop
        let stableRegion = Array(allOutput[stableStart..<totalFrames])
        let peakAmplitude = stableRegion.map { abs($0) }.max() ?? 0
        print("Peak amplitude: \(peakAmplitude), expected: \(gain)")
        XCTAssertEqual(peakAmplitude, gain, accuracy: 0.1, "Peak amplitude should equal N/hop = \(gain)")

        // 2. Periodicity: result[i] ≈ result[i + period]
        var maxPeriodicityError: Float = 0
        for i in stableStart..<(totalFrames - period) {
            if i % hop == 0 || (i + period) % hop == 0 { continue }
            let error = abs(allOutput[i] - allOutput[i + period])
            maxPeriodicityError = max(maxPeriodicityError, error)
        }
        print("Max periodicity error: \(maxPeriodicityError)")
        XCTAssertLessThan(maxPeriodicityError, 0.01, "Output should be periodic at input frequency")

        // 3. Check for glitches at run boundaries
        // Samples at run boundaries should NOT have discontinuities
        var maxBoundaryError: Float = 0
        for run in 1..<numRuns {
            let boundary = run * framesPerRun
            if boundary < stableStart || boundary >= totalFrames - period { continue }
            if boundary % hop == 0 { continue }  // skip hop boundaries
            // Compare with expected periodic continuation
            let prevSample = allOutput[boundary - 1]
            let currSample = allOutput[boundary]
            // Adjacent samples of a 441 Hz cosine at 44100 Hz change by at most
            // gain * 2π * freq / sr ≈ 4 * 0.0628 ≈ 0.25
            let maxExpectedDelta: Float = gain * 2.0 * Float.pi * freq / sr * 1.5
            let delta = abs(currSample - prevSample)
            if delta > maxExpectedDelta {
                maxBoundaryError = max(maxBoundaryError, delta)
                print("  Boundary glitch at run \(run) (frame \(boundary)): delta=\(delta), prev=\(prevSample), curr=\(currSample)")
            }
        }
        print("Max boundary discontinuity: \(maxBoundaryError)")
        XCTAssertLessThan(maxBoundaryError, 0.5, "No large discontinuities at run boundaries")

        // 4. Waveform shape: verify cosine shape around a peak
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

    /// Same as above but with reshape [N, 1] instead of [N], matching the user's patch.
    func testTensorFFTMultiInvocationReshapeN1() throws {
        let N = 1024
        let hop = N / 4
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
        // Key difference: reshape to [N, 1] instead of [N]
        let flat = try g.reshape(buffered, to: [N, 1])
        let (re, im) = g.tensorFFT(flat, N: N)
        let reconstructed = g.tensorIFFT(re, im, N: N)
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
                print("Run \(run) [N,1]: max amplitude = \(maxAmp)")
            }
        }

        let gain = Float(N) / Float(hop)
        let stableStart = totalFrames * 3 / 4

        print("\n=== Tensor FFT [N,1] reshape (N=\(N), hop=\(hop), \(numRuns)×\(framesPerRun)) ===")

        let stableRegion = Array(allOutput[stableStart..<totalFrames])
        let peakAmplitude = stableRegion.map { abs($0) }.max() ?? 0
        print("Peak amplitude: \(peakAmplitude), expected: \(gain)")
        XCTAssertEqual(peakAmplitude, gain, accuracy: 0.1, "Peak amplitude should equal N/hop")

        var maxPeriodicityError: Float = 0
        for i in stableStart..<(totalFrames - period) {
            if i % hop == 0 || (i + period) % hop == 0 { continue }
            let error = abs(allOutput[i] - allOutput[i + period])
            maxPeriodicityError = max(maxPeriodicityError, error)
        }
        print("Max periodicity error: \(maxPeriodicityError)")
        XCTAssertLessThan(maxPeriodicityError, 0.01, "Output should be periodic")

        var maxShapeError: Float = 0
        var peakIdx = stableStart
        for i in stableStart..<(totalFrames - period) {
            if i % hop == 0 { continue }
            if abs(allOutput[i]) > abs(allOutput[peakIdx]) { peakIdx = i }
        }
        let peakSign: Float = allOutput[peakIdx] > 0 ? 1.0 : -1.0
        for offset in -period/2..<period/2 {
            let i = peakIdx + offset
            guard i >= 0 && i < totalFrames else { continue }
            if i % hop == 0 { continue }
            let expected = peakSign * gain * cos(2.0 * Float.pi * Float(offset) / Float(period))
            maxShapeError = max(maxShapeError, abs(allOutput[i] - expected))
        }
        print("Max cosine shape error: \(maxShapeError)")
        XCTAssertLessThan(maxShapeError, 0.01, "Output should match cosine shape")
    }

    /// Diagnostic: dump raw samples around hop boundaries to see the glitch pattern
    func testTensorFFTHopBoundaryDiagnostic() throws {
        let N = 1024
        let hop = N / 4
        let sr: Float = 44100.0
        let freq: Float = 100.0  // Match user's patch
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
        let (re, im) = g.tensorFFT(flat, N: N)
        let reconstructed = g.tensorIFFT(re, im, N: N)
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
            XCTFail("Failed to allocate memory"); return
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
                        outputs: outPtr.baseAddress!, inputs: inPtr.baseAddress!,
                        memory: mem, frameCount: framesPerRun
                    )
                }
            }
            allOutput.append(contentsOf: runOutput)
        }

        let gain = Float(N) / Float(hop)
        let stableStart = totalFrames * 3 / 4

        // Show samples around each hop boundary in stable region
        print("\n=== Hop Boundary Diagnostic (freq=\(freq) Hz) ===")
        for boundary in stride(from: stableStart, to: totalFrames - 5, by: hop) {
            if boundary < 5 { continue }
            print("  Hop boundary at frame \(boundary):")
            for offset in -3...3 {
                let i = boundary + offset
                guard i >= 0 && i < totalFrames else { continue }
                // Expected: gain * cos(2π * freq * (i+1) / sr)
                // (+1 because phasor is post-increment)
                let phasorPhase = Float(i + 1) * freq / sr
                let expected = gain * cos(2.0 * Float.pi * phasorPhase)
                let error = allOutput[i] - expected
                let marker = offset == 0 ? " <<<" : ""
                print("    [\(offset >= 0 ? "+" : "")\(offset)] frame \(i): got \(String(format: "%+.6f", allOutput[i])), expected \(String(format: "%+.6f", expected)), error \(String(format: "%+.8f", error))\(marker)")
            }
        }

        // Periodicity check (the real test — doesn't need pipeline delay)
        let period = Int(sr / freq)  // 441 for 100 Hz
        var maxPeriodicityAtHop: Float = 0
        var maxPeriodicityNotHop: Float = 0
        for i in stableStart..<(totalFrames - period) {
            let error = abs(allOutput[i] - allOutput[i + period])
            if i % hop == 0 || (i + period) % hop == 0 {
                maxPeriodicityAtHop = max(maxPeriodicityAtHop, error)
            } else {
                maxPeriodicityNotHop = max(maxPeriodicityNotHop, error)
            }
        }
        print("\nPeriodicity error AT hop boundaries: \(maxPeriodicityAtHop)")
        print("Periodicity error NOT at hop boundaries: \(maxPeriodicityNotHop)")

        // Adjacent sample smoothness (detect staircase/sample-rate-reduction)
        var maxDelta: Float = 0
        var avgDelta: Float = 0
        var deltaCount = 0
        for i in stableStart..<(totalFrames - 1) {
            if i % hop == 0 { continue }
            let delta = abs(allOutput[i + 1] - allOutput[i])
            maxDelta = max(maxDelta, delta)
            avgDelta += delta
            deltaCount += 1
        }
        avgDelta /= Float(deltaCount)
        print("Max adjacent delta (smoothness): \(maxDelta)")
        print("Avg adjacent delta: \(avgDelta)")
        let expectedMaxDelta = gain * 2.0 * Float.pi * freq / sr  // max slope of cosine
        print("Expected max delta (cosine slope): \(expectedMaxDelta)")
    }

    /// Export FFT→IFFT overlap-add output to WAV for listening/waveform inspection
    func testTensorFFTExportWav() throws {
        let N = 1024
        let hop = N / 4
        let sr: Float = 44100.0
        let freq: Float = 500.0
        let framesPerRun = 256
        let numRuns = 200  // ~1.16 seconds at 44100 Hz
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
        let (re, im) = g.tensorFFT(flat, N: N)
        let reconstructed = g.tensorIFFT(re, im, N: N)
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
            XCTFail("Failed to allocate memory"); return
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
                        outputs: outPtr.baseAddress!, inputs: inPtr.baseAddress!,
                        memory: mem, frameCount: framesPerRun
                    )
                }
            }
            allOutput.append(contentsOf: runOutput)
        }

        // Normalize: overlap-add gain = N/hop = 4
        let gain = Float(N) / Float(hop)
        let normalized = allOutput.map { $0 / gain }

        // Write WAV
        let wavPath = "/tmp/fft_ifft_overlap_add_test.wav"
        writeWav(samples: normalized, sampleRate: Int(sr), path: wavPath)
        print("Wrote \(totalFrames) samples (\(Float(totalFrames) / sr) sec) to \(wavPath)")
    }

    /// Same as WAV export but with varying frame counts per run (128, 256, 512)
    /// to match real audio graph behavior where buffer sizes aren't stable.
    func testTensorFFTExportWavVaryingFrameCounts() throws {
        let N = 1024
        let hop = N / 4
        let sr: Float = 44100.0
        let freq: Float = 500.0
        let maxFrameCount = 512  // Compile with max possible

        let g = Graph(sampleRate: sr, maxFrameCount: maxFrameCount)

        let freqNode = g.n(.constant(freq))
        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let phasorCell = g.alloc()
        let phase = g.n(.phasor(phasorCell), freqNode, zero)
        let signal = g.n(.cos, g.n(.mul, phase, twoPi))

        let buffered = g.bufferView(signal, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (re, im) = g.tensorFFT(flat, N: N)
        let reconstructed = g.tensorIFFT(re, im, N: N)
        let output = g.overlapAdd(reconstructed, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), output)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: maxFrameCount, debug: false)
        )

        let cRuntime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let mem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory"); return
        }
        defer { cRuntime.deallocateNodeMemory(mem) }
        injectTensorData(result: result, memory: mem.assumingMemoryBound(to: Float.self))

        // Vary frame counts like a real audio graph: 128, 256, 512 randomly
        let frameSizes = [128, 256, 512]
        var allOutput = [Float]()
        var totalFrames = 0
        let targetFrames = 44100  // ~1 second

        while totalFrames < targetFrames {
            let framesThisRun = frameSizes[totalFrames / 256 % frameSizes.count]
            let inputBuf = [Float](repeating: 0, count: framesThisRun)
            var runOutput = [Float](repeating: 0, count: framesThisRun)
            runOutput.withUnsafeMutableBufferPointer { outPtr in
                inputBuf.withUnsafeBufferPointer { inPtr in
                    cRuntime.runWithMemory(
                        outputs: outPtr.baseAddress!, inputs: inPtr.baseAddress!,
                        memory: mem, frameCount: framesThisRun
                    )
                }
            }
            allOutput.append(contentsOf: runOutput)
            totalFrames += framesThisRun
        }

        let gain = Float(N) / Float(hop)
        let normalized = allOutput.map { $0 / gain }

        let wavPath = "/tmp/fft_ifft_varying_framecounts.wav"
        writeWav(samples: normalized, sampleRate: Int(sr), path: wavPath)
        print("Wrote \(totalFrames) samples (\(Float(totalFrames) / sr) sec) to \(wavPath)")
        print("Used varying frame sizes: \(frameSizes)")
    }
}

// MARK: - WAV Writer

private func writeWav(samples: [Float], sampleRate: Int, path: String) {
    let numSamples = samples.count
    let bitsPerSample: Int = 16
    let numChannels: Int = 1
    let byteRate = sampleRate * numChannels * bitsPerSample / 8
    let blockAlign = numChannels * bitsPerSample / 8
    let dataSize = numSamples * blockAlign
    let fileSize = 36 + dataSize

    var data = Data()

    // RIFF header
    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)

    // fmt chunk
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
    data.append(contentsOf: withUnsafeBytes(of: UInt16(numChannels).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(bitsPerSample).littleEndian) { Array($0) })

    // data chunk
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })

    for sample in samples {
        let clamped = max(-1.0, min(1.0, sample))
        let int16Val = Int16(clamped * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: int16Val.littleEndian) { Array($0) })
    }

    try! data.write(to: URL(fileURLWithPath: path))
}

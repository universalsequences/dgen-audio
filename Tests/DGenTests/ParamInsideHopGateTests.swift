import XCTest

@testable import DGen

/// Reproduces a reported bug: `.param` used inside a hop-gated block (e.g. as
/// the interpolation weight in a `mix(a, b, t)` feeding rectFFT.phase) does
/// not behave correctly. The compiled kernel should respond to changes in the
/// param value, but apparently it doesn't.
///
/// The minimal shape of the problem:
///   - `a` is a hop-rate tensor (e.g. random phase from `noise @size N @hopSize M`)
///   - `b` is a static tensor (e.g. zeros)
///   - `t` is a scalar `.param`
///   - `mix(a, b, t)` feeds a hop-rate consumer (sin/cos → rectFFT → ifft → OLA)
///
/// We run the patch twice with two different param values. If the param is
/// wired correctly, the output should differ. If the bug reproduces, the two
/// runs will produce identical output (param is effectively dead).
final class ParamInsideHopGateTests: XCTestCase {

    /// Minimal repro: mix two hop-rate tensors with a scalar param, sum,
    /// output. No FFT, no spectral chain — isolates the param-inside-hop bug.
    func testParamMixOfHopRateTensorsRespondsToParamValue() throws {
        let N = 64
        let hop = 16
        let framesPerRun = 64
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // Two hop-rate tensors: one generates random values, the other is
        // a static constant tensor of ones.
        let randomHop = g.noise(size: N, hopSize: hop)   // `[N]` hop-rate
        let onesData = [Float](repeating: 1.0, count: N)
        let onesTensor = g.tensor(onesData)               // `[N]` static

        let paramCell = g.alloc()
        let param = g.n(.param(paramCell))

        // mix(a, b, t) = a*(1-t) + b*t
        // t = 0 → output = random (hop-rate)
        // t = 1 → output = ones (sum should be exactly N)
        let mixed = g.n(.mix, randomHop, onesTensor, param)
        let summed = g.n(.sum, mixed)
        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/param_hop_mix_minimal.c", atomically: true, encoding: .utf8)

        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        // Helper that allocates memory, injects tensor data, writes the
        // param value directly into its cell, runs the kernel, returns
        // the output buffer.
        func runWith(paramValue: Float) -> [Float] {
            guard let mem = runtime.allocateNodeMemory() else {
                XCTFail("mem alloc failed"); return []
            }
            defer { runtime.deallocateNodeMemory(mem) }
            let memPtr = mem.assumingMemoryBound(to: Float.self)
            injectTensorData(result: result, memory: memPtr)

            // Scalar param values are stored in their cell by the host —
            // look up the physical offset for paramCell and write.
            let physicalParamSlot =
                result.cellAllocations.cellMappings[paramCell] ?? paramCell
            memPtr[physicalParamSlot] = paramValue

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
            return output
        }

        let outAllA = runWith(paramValue: 0.0)   // full randomHop
        let outAllB = runWith(paramValue: 1.0)   // full ones → sum = N on every hop

        print("param=0 sums: \(outAllA.prefix(16))")
        print("param=1 sums: \(outAllB.prefix(16))")

        // With param=1, mixed = ones; sum = N on every frame. This is the
        // strongest sanity check — if the param is truly plumbed, every
        // output sample where the consumer fires should be exactly N.
        let hopBoundaries = stride(from: 0, to: framesPerRun, by: hop)
        for i in hopBoundaries {
            XCTAssertEqual(
                outAllB[i], Float(N), accuracy: 1e-3,
                "param=1 should set mixed = ones → sum = N at hop frame \(i), got \(outAllB[i])")
        }

        // With param=0, mixed = randomHop; the sum has the stddev profile
        // of `sum of N uniform [-1, 1]` ≈ sqrt(N/3) ≈ \(Foundation.sqrt(Float(N)/3.0)),
        // which should differ substantially from param=1's `N` result.
        var anyDifference = false
        for i in hopBoundaries {
            if abs(outAllA[i] - outAllB[i]) > 1e-2 {
                anyDifference = true
                break
            }
        }
        XCTAssertTrue(
            anyDifference,
            "outputs with param=0 and param=1 should differ — param is being ignored")
    }

    /// The user's actual scenario: mix two phase sources inside a
    /// full rectFFT → ifft → OLA chain. Verifies the whole hop-gated
    /// spectral pipeline sees param changes.
    func testParamMixOfPhasesFedIntoRectFFTChain() throws {
        let N = 256
        let hop = 64
        let framesPerRun = 256
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // Input → bufferView → FFT → (xRe, xIm)
        let inputNode = g.n(.input(0))
        let buffered = g.bufferView(inputNode, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flat, N: N)

        // Compute magnitude directly (avoid polarFFT's unused atan2).
        let magSq = g.n(.add, g.n(.mul, xRe, xRe), g.n(.mul, xIm, xIm))
        let mag = g.n(.sqrt, magSq)

        // Phase source A: hop-rate random phase (paulstretch style).
        let noiseTensor = g.noise(size: N, hopSize: hop)
        let twoPi = g.n(.constant(2.0 * .pi))
        let randPhase = g.n(.mul, noiseTensor, twoPi)

        // Phase source B: zero phase (identity — reconstructs original mag-only).
        let zeros = [Float](repeating: 0.0, count: N)
        let zeroPhase = g.tensor(zeros)

        // Param that interpolates between the two phase sources.
        let paramCell = g.alloc()
        let paramNode = g.n(.param(paramCell))
        let mixedPhase = g.n(.mix, randPhase, zeroPhase, paramNode)

        // rectFFT: apply the interpolated phase to the magnitude.
        let cosP = g.n(.cos, mixedPhase)
        let sinP = g.n(.sin, mixedPhase)
        let yRe = g.n(.mul, mag, cosP)
        let yIm = g.n(.mul, mag, sinP)

        let td = g.acceleratedIFFT(yRe, yIm, N: N)
        let scalar = g.overlapAdd(td, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), scalar)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/param_hop_mix_rectfft.c", atomically: true, encoding: .utf8)

        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        var signal = [Float](repeating: 0, count: framesPerRun)
        for i in 0..<framesPerRun {
            signal[i] = Foundation.cos(2.0 * Float.pi * Float(i) * 440.0 / 44100.0)
        }

        func runWith(paramValue: Float) -> [Float] {
            guard let mem = runtime.allocateNodeMemory() else {
                XCTFail("mem alloc failed"); return []
            }
            defer { runtime.deallocateNodeMemory(mem) }
            let memPtr = mem.assumingMemoryBound(to: Float.self)
            injectTensorData(result: result, memory: memPtr)

            let physicalParamSlot =
                result.cellAllocations.cellMappings[paramCell] ?? paramCell
            memPtr[physicalParamSlot] = paramValue

            var output = [Float](repeating: 0, count: framesPerRun)
            output.withUnsafeMutableBufferPointer { op in
                signal.withUnsafeBufferPointer { ip in
                    runtime.runWithMemory(
                        outputs: op.baseAddress!,
                        inputs: ip.baseAddress!,
                        memory: mem,
                        frameCount: framesPerRun)
                }
            }
            return output
        }

        let outZero = runWith(paramValue: 0.0)   // full random phase
        let outOne = runWith(paramValue: 1.0)    // zero phase (identity-ish)

        // Measure divergence — with zero phase the output should track the
        // input; with random phase it should smear into alien drone. They
        // should be clearly distinct.
        var maxDiff: Float = 0
        for i in 0..<framesPerRun {
            maxDiff = max(maxDiff, abs(outZero[i] - outOne[i]))
        }

        let peakZero = outZero.map { abs($0) }.max() ?? 0
        let peakOne = outOne.map { abs($0) }.max() ?? 0
        print("param=0 peak=\(peakZero), param=1 peak=\(peakOne), maxDiff=\(maxDiff)")

        XCTAssertFalse(outZero.contains(where: \.isNaN))
        XCTAssertFalse(outOne.contains(where: \.isNaN))
        XCTAssertGreaterThan(
            maxDiff, 1e-3,
            "param values 0 vs 1 should produce audibly distinct outputs — param is dead")
    }

    /// The user's real bug: full paulstretch chain where the random phase
    /// is scaled by a `.param` instead of the constant `2π`. Regardless of
    /// what value is written into the param's memory cell, the output is
    /// reportedly silent.
    ///
    /// Graph:
    ///   in → bufferView → fft → mag (sqrt(re² + im²))
    ///   noise @size N @hopSize M → * paramScale → randPhase
    ///   rectFFT(mag, randPhase) → ifft → overlapAdd → out
    ///
    /// We run it with the param set to 2π (should sound paulstretch-y,
    /// i.e. non-silent) and assert the output has non-trivial energy.
    func testPaulstretchWithParamScaleOnPhaseProducesAudio() throws {
        let N = 256
        let hop = 64
        let framesPerRun = 512
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let inputNode = g.n(.input(0))
        let buffered = g.bufferView(inputNode, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flat, N: N)

        let magSq = g.n(.add, g.n(.mul, xRe, xRe), g.n(.mul, xIm, xIm))
        let mag = g.n(.sqrt, magSq)

        let noiseTensor = g.noise(size: N, hopSize: hop)

        // The thing under test: a `.param` replacing the constant 2π.
        let scaleCell = g.alloc()
        let scaleParam = g.n(.param(scaleCell))
        let randPhase = g.n(.mul, noiseTensor, scaleParam)

        let cosP = g.n(.cos, randPhase)
        let sinP = g.n(.sin, randPhase)
        let yRe = g.n(.mul, mag, cosP)
        let yIm = g.n(.mul, mag, sinP)

        let td = g.acceleratedIFFT(yRe, yIm, N: N)
        let scalar = g.overlapAdd(td, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), scalar)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/paulstretch_param_scale.c",
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
        let memPtr = mem.assumingMemoryBound(to: Float.self)
        injectTensorData(result: result, memory: memPtr)

        // Write 2π into the scaleParam's memory cell directly. This is the
        // path the audiograph layer uses (direct writes into the shared
        // memory buffer from the audio callback via the param ring).
        let physicalSlot =
            result.cellAllocations.cellMappings[scaleCell] ?? scaleCell
        memPtr[physicalSlot] = 2.0 * .pi

        var signal = [Float](repeating: 0, count: framesPerRun)
        for i in 0..<framesPerRun {
            signal[i] = Foundation.cos(2.0 * Float.pi * Float(i) * 440.0 / 44100.0)
        }
        var output = [Float](repeating: 0, count: framesPerRun)
        output.withUnsafeMutableBufferPointer { op in
            signal.withUnsafeBufferPointer { ip in
                runtime.runWithMemory(
                    outputs: op.baseAddress!,
                    inputs: ip.baseAddress!,
                    memory: mem,
                    frameCount: framesPerRun)
            }
        }

        let peak = output.map { abs($0) }.max() ?? 0
        let energy = output.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = Foundation.sqrt(energy / Float(framesPerRun))
        print("paulstretch+param: peak=\(peak), rms=\(rms)")

        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))
        XCTAssertGreaterThan(
            peak, 1e-3,
            "output is silent — param scale of 2π on random phase should produce an audible signal")
    }

    /// Regression for the SIMD-broadcast bug. A scalar `.param` multiplied
    /// against a hop-rate `[N]` tensor should broadcast the param value to
    /// all N elements. The old CRenderer emitted `vld1q_f32(&memory[cell])`
    /// (load 4 consecutive memory slots) instead of `vdupq_n_f32(...)`
    /// (broadcast scalar), so 3 of every 4 tensor elements were multiplied
    /// by garbage (zeros from the unrelated adjacent memory).
    ///
    /// Test: set a param to a known value (say 7.0), fill a tensor with
    /// ones, multiply, sum. The sum should equal N * 7 = 7N, not some
    /// quarter of that.
    func testScalarParamBroadcastsAcrossSIMDTensorMul() throws {
        let N = 1024
        let framesPerRun = 8
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let onesData = [Float](repeating: 1.0, count: N)
        let onesTensor = g.tensor(onesData)

        let pCell = g.alloc()
        let pNode = g.n(.param(pCell))
        let scaled = g.n(.mul, onesTensor, pNode)
        let summed = g.n(.sum, scaled)
        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))

        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("mem alloc failed"); return
        }
        defer { runtime.deallocateNodeMemory(mem) }
        let memPtr = mem.assumingMemoryBound(to: Float.self)
        injectTensorData(result: result, memory: memPtr)

        let paramValue: Float = 7.0
        let physicalSlot =
            result.cellAllocations.cellMappings[pCell] ?? pCell
        memPtr[physicalSlot] = paramValue

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

        let expected = Float(N) * paramValue
        print("param broadcast test: output[0] = \(output[0]), expected \(expected)")
        XCTAssertEqual(
            output[0], expected, accuracy: 1e-3,
            "scalar param in SIMD tensor-mul must broadcast across all lanes")
    }

    /// Recreates the user's paulstretch patch exactly:
    ///   in1 → bufferView(1024, 256) → * hann → fft → polarFFT
    ///   in2 → hopHold(256) → (trigger) → latch(mag) → heldMag
    ///   noise @size 1024 @hopSize 256 → * paramTau → randPhase
    ///   rectFFT(heldMag, randPhase) → ifft → * hann → overlapAdd → out
    ///
    /// Since the test framework doesn't have UI-driven `in` ports, `in1`
    /// is substituted with a phasor-driven sine, `in2` with a slow
    /// phasor. The resulting compiled graph shape is the same one from
    /// the patch-editor screenshot. The test writes `vectorWidth` copies
    /// of `2π` into the param's physical slots (mirroring what the
    /// audiograph layer does) and asserts the output is non-silent.
    ///
    /// Prints `paramTau.cellVectorWidth` so we can see whether the param
    /// cell got upgraded to 4 by `remapVectorMemorySlots` (which is
    /// what the fix would look like if this turns out to be a missing
    /// upgrade).
    func testPaulstretchPatchScreenshotReproWithParamTau() throws {
        let N = 1024
        let hop = 256
        let framesPerRun = 512
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // substitute for `in 1`: a 440 Hz sinewave via phasor
        let freqA = g.n(.constant(440.0))
        let resetA = g.n(.constant(0.0))
        let phaseA = g.phasor(freq: freqA, reset: resetA)
        let twoPi = g.n(.constant(2.0 * .pi))
        let in1 = g.n(.sin, g.n(.mul, phaseA, twoPi))

        // substitute for `in 2`: a slow phasor used as freeze-trigger source
        let freqB = g.n(.constant(0.5))
        let resetB = g.n(.constant(0.0))
        let in2 = g.phasor(freq: freqB, reset: resetB)
        let hopTrigger = g.hopHold(in2, hopSize: hop)

        // bufferView → * hann → fft
        let buffered = g.bufferView(in1, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        var hannData = [Float](repeating: 0, count: N)
        let sc = 2.0 * Float.pi / Float(N)
        for i in 0..<N { hannData[i] = 0.5 - 0.5 * Foundation.cos(sc * Float(i)) }
        let hannTensor = g.tensor(hannData)
        let windowedIn = g.n(.mul, flat, hannTensor)
        let (xRe, xIm) = g.acceleratedFFT(windowedIn, N: N)

        // mag = sqrt(re²+im²) — skip atan2 which polarFFT would emit unused.
        let reSq = g.n(.mul, xRe, xRe)
        let imSq = g.n(.mul, xIm, xIm)
        let sumSq = g.n(.add, reSq, imSq)
        let mag = g.n(.sqrt, sumSq)

        // latch(mag, hopTrigger) → heldMag
        let heldMag = g.latch(mag, hopTrigger)

        // noise @size 1024 @hopSize 256 → * paramTau
        let noiseTensor = g.noise(size: N, hopSize: hop)
        let paramCell = g.alloc()
        let paramTau = g.n(.param(paramCell))
        let randPhase = g.n(.mul, noiseTensor, paramTau)

        // rectFFT(heldMag, randPhase)
        let cosP = g.n(.cos, randPhase)
        let sinP = g.n(.sin, randPhase)
        let yRe = g.n(.mul, heldMag, cosP)
        let yIm = g.n(.mul, heldMag, sinP)

        // ifft → * hann → overlapAdd
        let td = g.acceleratedIFFT(yRe, yIm, N: N)
        let windowedOut = g.n(.mul, td, hannTensor)
        let scalar = g.overlapAdd(windowedOut, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), scalar)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/paulstretch_patch_repro.c",
            atomically: true, encoding: .utf8)

        let paramWidth = result.cellAllocations.cellVectorWidths[paramCell]
        let paramPhysical = result.cellAllocations.cellMappings[paramCell]
        print("paramTau cell: logical=\(paramCell), physical=\(paramPhysical ?? -1), vectorWidth=\(paramWidth ?? -1)")

        let runtime = CCompiledKernel(
            source: result.source,
            cellAllocations: result.cellAllocations,
            memorySize: result.totalMemorySlots)
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("mem alloc failed"); return
        }
        defer { runtime.deallocateNodeMemory(mem) }
        let memPtr = mem.assumingMemoryBound(to: Float.self)
        injectTensorData(result: result, memory: memPtr)

        // Mirror what the audiograph layer does: write `vectorWidth`
        // copies of the value. If `remapVectorMemorySlots` upgraded the
        // param to width 4, this fills all 4 slots.
        if let physical = paramPhysical, let width = paramWidth {
            for i in 0..<width {
                memPtr[physical + i] = 2.0 * .pi
            }
        }

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

        let peak = output.map { abs($0) }.max() ?? 0
        let energy = output.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = Foundation.sqrt(energy / Float(framesPerRun))
        print("patch repro: peak=\(peak), rms=\(rms), paramWidth=\(paramWidth ?? -1)")

        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))
        XCTAssertGreaterThan(
            peak, 1e-3,
            "output silent — paramWidth=\(paramWidth ?? -1)")
    }
}

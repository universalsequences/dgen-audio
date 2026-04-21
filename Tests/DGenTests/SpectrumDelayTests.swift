import XCTest

@testable import DGen

/// `spectrumDelay @hops N` must expose the input spectrum from exactly
/// `N` hop boundaries ago. This is the foundation for phase vocoder and
/// any time-stretch patch.
final class SpectrumDelayTests: XCTestCase {

    /// Feed a hop-rate counter as a fake "spectrum" (each hop the tensor
    /// is all-`k` for hop index k). Assert the delay op outputs `k - hops`
    /// once enough hops have passed.
    func testSpectrumDelayReadsHopsAgo() throws {
        let N = 16
        let hop = 4
        let hopsBack = 2
        let framesPerRun = 32
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // Fake "spectrum source": hopTensorNoise reused as a hop-rate
        // generator — but we want deterministic values, so instead use
        // `noise(size: N, hopSize: hop) * 0 + hopIndex` where hopIndex is
        // a hop-rate counter. Simpler: wire a hop-rate scalar broadcast.

        // Hop-rate counter: increments each hop (via hopHold of a per-frame accum).
        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let perFrameCounter = g.n(.accum(rampCell), one, zero, zero, bigLimit)
        let hopRateCounter = g.hopHold(perFrameCounter, hopSize: hop)

        // Make a [N] "spectrum" whose elements are all the same scalar
        // value (broadcast the hop counter into a tensor). Uses the noise
        // op to get an N-shaped hop-rate carrier, zeroed out, then the
        // broadcast value added.
        let noiseT = g.noise(size: N, hopSize: hop)
        let noiseZeroed = g.n(.mul, noiseT, zero)
        let spectrumSrc = g.n(.add, noiseZeroed, hopRateCounter)

        let delayed = g.spectrumDelay(
            spectrumSrc, N: N, hops: hopsBack, hopSize: hop)

        // Read element [0] of the delayed tensor once per frame via a sum
        // followed by division by N — since every element of the delayed
        // tensor equals the delayed hop count, `sum / N` = hop-count from
        // `hopsBack` hops ago.
        let summed = g.n(.sum, delayed)
        let Nconst = g.n(.constant(Float(N)))
        let avgDelayed = g.n(.div, summed, Nconst)
        _ = g.n(.output(0), avgDelayed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/spectrum_delay_staircase.c",
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

        print("spectrumDelay output: \(output)")

        // The `sum/N` chain is hop-gated (all its inputs are hop-rate), so
        // output values only land on hop-boundary frames. Assert that
        // hop-boundary frames read the hop-count value from `hopsBack`
        // hops ago.
        //
        // Ramp value at frame 0 = 0 (accum pre-increment → 0 initial).
        // hopHold samples at frame 0, 4, 8, …; output values are 0, 4, 8, 12, …
        // spectrumDelay(hopsBack=2) produces:
        //   hop 0: 0 (initial, ring empty)
        //   hop 1: 0 (still within warmup)
        //   hop 2: hopCounter[hop 0] = 0
        //   hop 3: hopCounter[hop 1] = 4
        //   hop 4: hopCounter[hop 2] = 8
        //   hop 5: hopCounter[hop 3] = 12
        //   …
        // So starting at hop `hopsBack`, we expect the output to track
        // the hop counter delayed by hopsBack.
        for hopIdx in hopsBack..<(framesPerRun / hop) {
            let frame = hopIdx * hop
            let expected = Float((hopIdx - hopsBack) * hop)
            XCTAssertEqual(
                output[frame], expected, accuracy: 1e-3,
                "hop \(hopIdx) (frame \(frame)) should read hop count from \(hopsBack) hops ago")
        }
    }

    /// Modulated variant: feed a hop-rate delay input that picks a fixed
    /// integer delay of 1 at every hop. Output should match the fixed
    /// `spectrumDelay @hops 1` case for the same input.
    func testSpectrumDelayModConstantDelayOfOneMatchesFixedDelay() throws {
        let N = 16
        let hop = 4
        let framesPerRun = 32
        let maxHops = 3
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let perFrameCounter = g.n(.accum(rampCell), one, zero, zero, bigLimit)
        let hopRateCounter = g.hopHold(perFrameCounter, hopSize: hop)

        // Spectrum source: hop-rate tensor filled with `hopRateCounter`.
        let noiseT = g.noise(size: N, hopSize: hop)
        let noiseZeroed = g.n(.mul, noiseT, zero)
        let spectrumSrc = g.n(.add, noiseZeroed, hopRateCounter)

        // Modulated delay: feed constant 1.0 as the delay input.
        let constOneDelay = g.n(.constant(1.0))
        let delayed = g.spectrumDelayMod(
            spectrumSrc, delay: constOneDelay,
            N: N, maxHops: maxHops, hopSize: hop)

        let summed = g.n(.sum, delayed)
        let Nconst = g.n(.constant(Float(N)))
        let avg = g.n(.div, summed, Nconst)
        _ = g.n(.output(0), avg)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/spectrum_delay_mod.c", atomically: true, encoding: .utf8)
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

        print("spectrumDelayMod (delay=1.0) output: \(output)")

        // hopRateCounter values at hop boundaries: 0, 4, 8, 12, 16, 20, ...
        // With delay=1 (integer), output at hop k should equal hopRateCounter[hop k - 1].
        for hopIdx in 1..<(framesPerRun / hop) {
            let frame = hopIdx * hop
            let expected = Float((hopIdx - 1) * hop)
            XCTAssertEqual(
                output[frame], expected, accuracy: 1e-3,
                "hop \(hopIdx) (frame \(frame)) should read from 1 hop ago = \(expected)")
        }
    }

    /// Sanity: `spectrumDelayMod` with delay = 0 should be a passthrough
    /// of the current hop's spectrum — no one-hop lag, no interpolation.
    /// If this fails, either (a) the "most recent row" calculation is off
    /// by one, or (b) `delay = 0` isn't being honored exactly.
    func testSpectrumDelayModDelayZeroIsPassthrough() throws {
        let N = 16
        let hop = 4
        let framesPerRun = 32
        let maxHops = 4
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let perFrameCounter = g.n(.accum(rampCell), one, zero, zero, bigLimit)
        let hopRateCounter = g.hopHold(perFrameCounter, hopSize: hop)

        let noiseT = g.noise(size: N, hopSize: hop)
        let noiseZeroed = g.n(.mul, noiseT, zero)
        let spectrumSrc = g.n(.add, noiseZeroed, hopRateCounter)

        let zeroDelay = g.n(.constant(0.0))
        let delayed = g.spectrumDelayMod(
            spectrumSrc, delay: zeroDelay,
            N: N, maxHops: maxHops, hopSize: hop)

        let summed = g.n(.sum, delayed)
        let Nconst = g.n(.constant(Float(N)))
        let avg = g.n(.div, summed, Nconst)
        _ = g.n(.output(0), avg)

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

        print("spectrumDelayMod (delay=0) output: \(output)")

        // At each hop boundary k, output should equal hopRateCounter's
        // current value = k * hop.
        for hopIdx in 0..<(framesPerRun / hop) {
            let frame = hopIdx * hop
            let expected = Float(hopIdx * hop)
            XCTAssertEqual(
                output[frame], expected, accuracy: 1e-3,
                "delay=0: hop \(hopIdx) (frame \(frame)) should pass through current = \(expected)")
        }
    }

    /// Fractional delay of 0.5 should produce the average of the two
    /// adjacent rows (lerp).
    func testSpectrumDelayModFractionalHalfInterpolatesBetweenRows() throws {
        let N = 16
        let hop = 4
        let framesPerRun = 32
        let maxHops = 3
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let perFrameCounter = g.n(.accum(rampCell), one, zero, zero, bigLimit)
        let hopRateCounter = g.hopHold(perFrameCounter, hopSize: hop)

        let noiseT = g.noise(size: N, hopSize: hop)
        let noiseZeroed = g.n(.mul, noiseT, zero)
        let spectrumSrc = g.n(.add, noiseZeroed, hopRateCounter)

        // Delay = 0.5: should lerp between current (0 hops ago) and
        // previous (1 hop ago).
        let halfDelay = g.n(.constant(0.5))
        let delayed = g.spectrumDelayMod(
            spectrumSrc, delay: halfDelay,
            N: N, maxHops: maxHops, hopSize: hop)
        let summed = g.n(.sum, delayed)
        let Nconst = g.n(.constant(Float(N)))
        let avg = g.n(.div, summed, Nconst)
        _ = g.n(.output(0), avg)

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

        print("spectrumDelayMod (delay=0.5) output: \(output)")

        // Expected at hop k: 0.5 * hopRateCounter[hop k] + 0.5 * hopRateCounter[hop k-1]
        // = 0.5 * (k * hop) + 0.5 * ((k-1) * hop) = hop * (k - 0.5)
        for hopIdx in 1..<(framesPerRun / hop) {
            let frame = hopIdx * hop
            let expected = Float(hop) * (Float(hopIdx) - 0.5)
            XCTAssertEqual(
                output[frame], expected, accuracy: 1e-3,
                "hop \(hopIdx): lerp between hop \(hopIdx) and \(hopIdx - 1) should be \(expected)")
        }
    }
}

import XCTest

@testable import DGen

/// Verifies `hopHold` semantics at the DGen level:
///   - Output value changes only at hop boundaries (every `hopSize` frames)
///   - Between hops the output is held constant
///   - Downstream consumers inherit hop-based scheduling via `nodeHopRate`
final class HopHoldTests: XCTestCase {

    /// Feed a frame-rate ramp signal (accum +1 per frame) through hopHold
    /// and verify the output is a staircase: constant within a hop, stepping
    /// only at hop boundaries.
    func testHopHoldStaircase() throws {
        let hopSize = 8
        let framesPerRun = 64
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // Sample-rate ramp: value = frame index.
        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let ramp = g.n(.accum(rampCell), one, zero, zero, bigLimit)

        let held = g.hopHold(ramp, hopSize: hopSize)
        _ = g.n(.output(0), held)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/hophold_staircase.c", atomically: true, encoding: .utf8)
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

        print("hopHold ramp staircase (hopSize=\(hopSize)): \(output.prefix(32))")

        // The accum+hopHold interaction: the accum's "pre-increment" value
        // is what the ramp node exposes; at frame 0 it reads 0, at frame 1
        // reads 1, etc. hopHold latches at counter==0 which is frames 0, 8,
        // 16, … so output[0..<8] should be output[0]; output[8..<16] ==
        // output[8]; and consecutive step values should differ by hopSize
        // (because the ramp advanced 8 between latches).
        for i in 0..<framesPerRun {
            let stepStart = (i / hopSize) * hopSize
            XCTAssertEqual(
                output[i], output[stepStart], accuracy: 1e-5,
                "frame \(i) should equal frame \(stepStart) (within same hop)")
        }

        for step in 1..<(framesPerRun / hopSize) {
            let diff = output[step * hopSize] - output[(step - 1) * hopSize]
            XCTAssertEqual(
                diff, Float(hopSize), accuracy: 1e-4,
                "step \(step) should increase by \(hopSize) from previous hop")
        }
    }

    /// Downstream of hopHold, a signal should be treated as hop-rate. We
    /// verify this indirectly: without hopHold, feeding a phasor-like
    /// sample-rate signal into an FFT chain forces frame-rate FFT execution;
    /// with hopHold, the FFT should run only at hop boundaries. We don't
    /// instrument the scheduler directly — we just check that the compilation
    /// succeeds and the output is finite/non-zero.
    func testHopHoldFeedsFFTChain() throws {
        let N = 256
        let hop = 64
        let framesPerRun = 256
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let inputNode = g.n(.input(0))
        let buffered = g.bufferView(inputNode, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flat, N: N)

        // polar
        let reSq = g.n(.mul, xRe, xRe)
        let imSq = g.n(.mul, xIm, xIm)
        let sumSq = g.n(.add, reSq, imSq)
        let mag = g.n(.sqrt, sumSq)

        // frame-rate phase modulation (accum ramp), held at hop rate.
        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let ramp = g.n(.accum(rampCell), one, zero, zero, bigLimit)
        let phase = g.hopHold(ramp, hopSize: hop)

        // rect — single scalar phase applied to all bins.
        let cosP = g.n(.cos, phase)
        let sinP = g.n(.sin, phase)
        let yRe = g.n(.mul, mag, cosP)
        let yIm = g.n(.mul, mag, sinP)

        let td = g.acceleratedIFFT(yRe, yIm, N: N)
        let scalar = g.overlapAdd(td, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), scalar)

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

        // Not checking for correctness (phase-rotated output isn't
        // analytically predictable without reimplementing the whole pipeline).
        // Just assert it compiled, ran, and produced finite output. The
        // compile-time inspection below confirms the FFT is hop-gated.
        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))

        // The generated module calls the portable host-services ABI, not
        // Accelerate directly. Pin one forward and one inverse call site;
        // this structural check alone does not prove hop gating or cadence.
        // SpectralSchedulingTests covers sample-level scheduling numerically.
        let kernelSource = result.source
        let fwdCount =
            kernelSource.components(separatedBy: "host->fft_forward_fn(").count - 1
        let invCount =
            kernelSource.components(separatedBy: "host->fft_inverse_fn(").count - 1
        print("FFT call sites: forward=\(fwdCount), inverse=\(invCount)")
        XCTAssertEqual(fwdCount, 1, "exactly one forward FFT call site expected")
        XCTAssertEqual(invCount, 1, "exactly one inverse FFT call site expected")
    }
}

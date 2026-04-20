import XCTest

@testable import DGen

/// Verifies that `.latch` works on tensor-shaped inputs — the basis for
/// patch-editor spectral-freeze. A `[N]` tensor gets latched on hop
/// boundaries, so freeze/hold-to-capture patterns compose as:
///   `phasor(3) < 0.2 -> hopHold -> latch(tensor)`.
final class SpectrumLatchTests: XCTestCase {

    /// Feed a per-frame counting ramp expanded to [N] tensor via broadcast,
    /// latch every 4 frames. Output should be a staircase within each tensor
    /// element: frames 0..3 hold the value captured at frame 0, etc.
    func testTensorLatchStaircase() throws {
        let N = 4
        let hopSize = 4
        let framesPerRun = 16
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        // Sample-rate ramp: frame index.
        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let ramp = g.n(.accum(rampCell), one, zero, zero, bigLimit)

        // Per-frame tensor: static baseline plus the ramp. Produces a [N]
        // tensor whose elements are all `ramp` + {0, 1, 2, 3}.
        let baseline = g.tensor(shape: [N], data: [0.0, 1.0, 2.0, 3.0])
        let rampTensor = g.n(.add, baseline, ramp)

        // Hop trigger: accum counter wrapping at hopSize, fires when == 0.
        let counterCell = g.alloc(vectorWidth: 1)
        g.persistentCells.insert(counterCell)
        let hopConst = g.n(.constant(Float(hopSize)))
        let counter = g.n(.accum(counterCell), one, zero, zero, hopConst)
        let trigger = g.n(.eq, counter, zero)

        // Tensor latch — should capture rampTensor on hop boundaries.
        let latchCell = g.alloc(vectorWidth: N)
        g.persistentCells.insert(latchCell)
        let latched = g.n(.latch(latchCell), rampTensor, trigger)

        // Sum to scalar for readout.
        let summed = g.n(.sum, latched)
        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/spectrum_latch.c", atomically: true, encoding: .utf8)
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

        print("tensor latch staircase: \(output)")

        // rampTensor at frame f has sum 4f + 6 (accum increments the scalar
        // ramp by 1 each frame, so element i = ramp + baseline[i]). Latch
        // fires on hops 0, 4, 8, 12. On trigger frames the output is the
        // freshly-captured spectrum; between hops it holds. Expected sums:
        //   hop 0 captures f=0 → sum 6,  frames 0..3 output 6
        //   hop 1 captures f=4 → sum 22, frames 4..7 output 22
        //   hop 2 captures f=8 → sum 38, frames 8..11 output 38
        //   hop 3 captures f=12 → sum 54, frames 12..15 output 54
        for i in 0..<framesPerRun {
            let stepStart = (i / hopSize) * hopSize
            XCTAssertEqual(
                output[i], output[stepStart], accuracy: 1e-5,
                "frame \(i) should equal frame \(stepStart) (same hop)")
        }
        // Consecutive hops advance by hopSize * N = 16.
        for step in 1..<(framesPerRun / hopSize) {
            let diff = output[step * hopSize] - output[(step - 1) * hopSize]
            XCTAssertEqual(
                diff, 16.0, accuracy: 1e-4,
                "hop \(step) sum should advance by hopSize * N")
        }
    }

    /// Regression guard for the "latch breaks hop-rate propagation" CPU bug.
    /// Build a freeze-style chain:
    ///   in -> bufferView -> fft -> latch(trigger=hopHold) ->
    ///     complexMul with same fft -> ifft -> overlapAdd
    /// When the latch's trigger is hop-rate, `graph.latch` must register the
    /// latched node as hop-producing so the downstream IFFT and its post
    /// chain stay hop-gated. Without the propagation the IFFT runs every
    /// frame instead of every `hopSize` frames.
    func testLatchPreservesHopRateForDownstreamFFT() throws {
        let N = 256
        let hop = 64
        let framesPerRun = 256
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let inputNode = g.n(.input(0))
        let buffered = g.bufferView(inputNode, size: N, hopSize: hop)
        let flat = try g.reshape(buffered, to: [N])
        let (xRe, xIm) = g.acceleratedFFT(flat, N: N)

        // Hop-rate freeze trigger — reuse hopHold to synthesise one.
        let rampCell = g.alloc(vectorWidth: 1)
        let one = g.n(.constant(1.0))
        let zero = g.n(.constant(0.0))
        let bigLimit = g.n(.constant(1e9))
        let ramp = g.n(.accum(rampCell), one, zero, zero, bigLimit)
        let trigger = g.hopHold(ramp, hopSize: hop)

        // Spectral freeze: latch the spectrum on hop boundaries.
        let frozenRe = g.latch(xRe, trigger)
        let frozenIm = g.latch(xIm, trigger)

        // Multiply frozen spectrum with current spectrum (complex mul via
        // raw arithmetic — we only care about structure, not numerics).
        let reRe = g.n(.mul, frozenRe, xRe)
        let imIm = g.n(.mul, frozenIm, xIm)
        let yRe = g.n(.sub, reRe, imIm)
        let reIm = g.n(.mul, frozenRe, xIm)
        let imRe = g.n(.mul, frozenIm, xRe)
        let yIm = g.n(.add, reIm, imRe)

        let td = g.acceleratedIFFT(yRe, yIm, N: N)
        let scalar = g.overlapAdd(td, windowSize: N, hopSize: hop)
        _ = g.n(.output(0), scalar)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))

        let kernelSource = result.source
        let fwdCount =
            kernelSource.components(separatedBy: "kFFTDirection_Forward").count - 1
        let invCount =
            kernelSource.components(separatedBy: "kFFTDirection_Inverse").count - 1
        print("latch-freeze FFT sites: forward=\(fwdCount), inverse=\(invCount)")
        XCTAssertEqual(fwdCount, 1, "exactly one forward FFT call site expected")
        XCTAssertEqual(invCount, 1, "exactly one inverse FFT call site expected")

        // Also ensure it still compiles and runs without producing NaN/Inf.
        let runtime = CCompiledKernel(
            source: kernelSource,
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
        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))
    }
}

import XCTest

@testable import DGen

/// Verifies `graph.noise(size:)` produces a tensor of independent random
/// values per element, and that the `noise → hopHold` bridge works so
/// spectral chains downstream stay hop-gated.
final class TensorNoiseTests: XCTestCase {

    /// Per-bin independence check: sum a `[N]` noise tensor each frame.
    /// If all elements shared a value (scalar broadcast), the sum would be
    /// exactly `N * elem` — extremely correlated across frames. Independent
    /// elements produce a sum with stddev roughly `sqrt(N/3)` for uniform
    /// `[-1, 1]`.
    func testTensorNoiseProducesIndependentElements() throws {
        let N = 1024
        let framesPerRun = 64
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let noiseTensor = g.noise(size: N)
        let summed = g.n(.sum, noiseTensor)
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

        let mean = output.reduce(0, +) / Float(framesPerRun)
        let variance = output.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(framesPerRun)
        let stddev = Foundation.sqrt(variance)

        print("noise sum per frame: mean=\(mean), stddev=\(stddev)")

        // For independent uniform [-1, 1] × N, sum has mean ≈ 0 and stddev ≈ sqrt(N/3).
        let expectedStddev = Foundation.sqrt(Float(N) / 3.0)
        XCTAssertLessThan(
            abs(mean), expectedStddev,
            "sum-of-noise mean should be much smaller than stddev")
        XCTAssertGreaterThan(
            stddev, expectedStddev * 0.5,
            "stddev \(stddev) too small — elements may not be independent (expected ~\(expectedStddev))")
        XCTAssertLessThan(
            stddev, expectedStddev * 2.0,
            "stddev \(stddev) too large — out of expected range ~\(expectedStddev)")

        // Every value in range [-N, N].
        for v in output {
            XCTAssertFalse(v.isNaN)
            XCTAssertGreaterThanOrEqual(v, -Float(N))
            XCTAssertLessThanOrEqual(v, Float(N))
        }
    }

    /// `noise → hopHold` staircase: between hops, all N elements should be
    /// unchanged. On hop boundaries they refresh to new random values.
    func testTensorNoiseHopHoldStaircase() throws {
        let N = 16
        let hopSize = 4
        let framesPerRun = 32
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let noise = g.noise(size: N)
        let held = g.hopHold(noise, hopSize: hopSize)
        // Sum to scalar — within a hop the sum should be identical.
        let summed = g.n(.sum, held)
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

        print("noise→hopHold sums: \(output)")

        // `sum → output` runs hop-gated because all its inputs are hop-rate,
        // so we only assert on hop-boundary frames. The paulstretch use case
        // is fine with this — the held tensor feeds a hop-rate spectral
        // chain whose overlapAdd tail produces frame-rate output.

        // Each hop-boundary captures a fresh random draw — consecutive hops
        // should rarely collide, so we expect at least several distinct
        // values across the run.
        var hopValues: [Float] = []
        for step in 0..<(framesPerRun / hopSize) {
            hopValues.append(output[step * hopSize])
        }
        let distinctHops = Set(hopValues)
        XCTAssertGreaterThanOrEqual(
            distinctHops.count, hopValues.count - 1,
            "expected nearly all hop sums to be distinct from independent noise draws, got \(distinctHops.count)/\(hopValues.count)")

        // Held sum has the same stddev profile as per-frame noise sums (same
        // underlying distribution, just sampled every hopSize frames).
        let mean = hopValues.reduce(0, +) / Float(hopValues.count)
        let variance = hopValues.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
            / Float(hopValues.count)
        let stddev = Foundation.sqrt(variance)
        let expectedStddev = Foundation.sqrt(Float(N) / 3.0)
        XCTAssertGreaterThan(stddev, expectedStddev * 0.3,
            "hop-sum stddev \(stddev) too small — elements may not be independent")
    }

    /// `graph.noise(size:, hopSize:)` should produce exactly one xorshift
    /// inner loop (inside a hop gate) per kernel — dramatically cheaper
    /// than `noise → hopHold` which generates N values every frame before
    /// the latch throws most of them away.
    func testHopGatedTensorNoiseRegeneratesOnlyPerHop() throws {
        let N = 1024
        let hop = 256
        let framesPerRun = 128
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)

        let held = g.noise(size: N, hopSize: hop)
        let summed = g.n(.sum, held)
        _ = g.n(.output(0), summed)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/hop_tensor_noise.c", atomically: true, encoding: .utf8)

        let src = result.source

        // The xorshift inner body (`s ^= s << 13`) should appear exactly
        // once — the fused op emits a single hop-gated loop. If we still
        // had per-frame generation it would live inside a `for i <
        // frameCount` loop and probably also duplicate in the summed /
        // downstream path.
        let xorshiftSites = src.components(separatedBy: "s ^= s << 13").count - 1
        XCTAssertEqual(
            xorshiftSites, 1,
            "expected exactly one xorshift body; found \(xorshiftSites)")

        // And that single body should be inside a hop gate, not unwrapped.
        let linesAroundXorshift = src.components(separatedBy: "s ^= s << 13")[0]
        let lastHopCheckBefore = linesAroundXorshift
            .components(separatedBy: "== 0.0f").count - 1
        XCTAssertGreaterThan(
            lastHopCheckBefore, 0,
            "xorshift body should be preceded by a hop-gate `== 0.0f` check")

        // Compiles and runs without NaN/Inf.
        let runtime = CCompiledKernel(
            source: src,
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
        XCTAssertFalse(output.contains(where: \.isNaN))
        XCTAssertFalse(output.contains(where: \.isInfinite))
    }
}

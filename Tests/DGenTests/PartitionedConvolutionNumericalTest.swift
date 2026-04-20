import AVFoundation
import Foundation
import XCTest

@testable import DGen

/// Numerical truth test for the end-to-end partitionedConvolve pipeline that
/// the patch-editor builds: bufferView → Hann → acceleratedFFT →
/// partitionedSpectralConvolve → acceleratedIFFT → Hann → overlapAdd → gain.
/// Fed with the 808kicklong.wav asset and compared against a direct-conv
/// reference. Prints output samples so regressions (e.g. the "latches to one
/// value for 1+ seconds" bug) are visible.
final class PartitionedConvolutionNumericalTest: XCTestCase {

    private func loadWAV(_ path: String) -> [Float]? {
        let url = URL(fileURLWithPath: path)
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
            (try? file.read(into: buffer)) != nil,
            let chans = buffer.floatChannelData
        else { return nil }
        let n = Int(frameCount)
        let c = Int(format.channelCount)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var acc: Float = 0
            for ch in 0..<c { acc += chans[ch][i] }
            out[i] = acc / Float(c)
        }
        return out
    }

    private func directConvolve(_ x: [Float], _ h: [Float]) -> [Float] {
        let outCount = x.count + h.count - 1
        var y = [Float](repeating: 0, count: outCount)
        for n in 0..<outCount {
            let kMin = max(0, n - x.count + 1)
            let kMax = min(h.count - 1, n)
            if kMin > kMax { continue }
            var acc: Float = 0
            for k in kMin...kMax { acc += x[n - k] * h[k] }
            y[n] = acc
        }
        return y
    }

    /// Mirror of the patch-editor `GenPartitionedConvolveOperator` graph: the
    /// exact same wiring (Hann-windowed analysis/synthesis, hop-gated FFT,
    /// partitioned MAC, overlapAdd, COLA gain).
    private func buildPartitionedConvolveGraph(
        input: NodeID, ir: [Float], N: Int, hopSize: Int, graph: Graph
    ) {
        let P = hopSize
        let K = (ir.count + P - 1) / P

        // Offline: K partitions, each P samples zero-padded to N, FFT'd.
        var irReFlat = [Float](repeating: 0, count: K * N)
        var irImFlat = [Float](repeating: 0, count: K * N)
        for k in 0..<K {
            var re = [Float](repeating: 0, count: N)
            var im = [Float](repeating: 0, count: N)
            let start = k * P
            let end = min(start + P, ir.count)
            for i in start..<end { re[i - start] = ir[i] }
            radix2FFTInPlaceLocal(re: &re, im: &im)
            for n in 0..<N {
                irReFlat[k * N + n] = re[n]
                irImFlat[k * N + n] = im[n]
            }
        }
        let irRe = graph.tensor(shape: [K * N], data: irReFlat)
        let irIm = graph.tensor(shape: [K * N], data: irImFlat)

        // Real periodic Hann window
        var hannData = [Float](repeating: 0, count: N)
        let scale = 2.0 * Float.pi / Float(N)
        for i in 0..<N { hannData[i] = 0.5 - 0.5 * Foundation.cos(scale * Float(i)) }
        let hannTensor = graph.tensor(hannData)

        let buffered = graph.bufferView(input, size: N, hopSize: hopSize)
        let flat = try! graph.reshape(buffered, to: [N])
        let windowedIn = graph.n(.mul, flat, hannTensor)
        let (xRe, xIm) = graph.acceleratedFFT(windowedIn, N: N)
        let (yRe, yIm) = graph.partitionedSpectralConvolve(
            xRe, xIm, irRe, irIm, K: K, N: N, hopSize: hopSize)
        let timeDomain = graph.acceleratedIFFT(yRe, yIm, N: N)
        let windowedOut = graph.n(.mul, timeDomain, hannTensor)
        let scalar = graph.overlapAdd(windowedOut, windowSize: N, hopSize: hopSize)
        let gain: Float = Float(hopSize) * 8.0 / (3.0 * Float(N) * Float(N))
        let out = graph.n(.mul, scalar, graph.n(.constant(gain)))
        _ = graph.n(.output(0), out)
    }

    // Inline radix-2 FFT so this test doesn't depend on the patch-editor helper.
    private func radix2FFTInPlaceLocal(re: inout [Float], im: inout [Float]) {
        let N = re.count
        precondition(N > 0 && (N & (N - 1)) == 0)
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
            let step = -2.0 * Float.pi / Float(size)
            var k = 0
            while k < N {
                for p in 0..<half {
                    let theta = step * Float(p)
                    let wr = Foundation.cos(theta), wi = Foundation.sin(theta)
                    let i0 = k + p, i1 = i0 + half
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

    private func runPipeline(
        input: [Float], ir: [Float], N: Int, hopSize: Int, framesPerRun: Int = 512
    ) throws -> [Float] {
        let g = Graph(sampleRate: 44100.0, maxFrameCount: framesPerRun)
        let inputNode = g.n(.input(0))
        buildPartitionedConvolveGraph(
            input: inputNode, ir: ir, N: N, hopSize: hopSize, graph: g)

        let result = try CompilationPipeline.compile(
            graph: g, backend: .c,
            options: .init(frameCount: framesPerRun, debug: false))
        try? result.source.write(
            toFile: "/tmp/partitioned_conv_numerical.c", atomically: true, encoding: .utf8)

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

        // DEBUG: dump a handful of memory slots so we can see where data ACTUALLY lives
        let memFloat = mem.assumingMemoryBound(to: Float.self)
        // Dump tensor data regions
        print("🔍 First 8 at mem[0]: \((0..<8).map { memFloat[$0] })")
        print("🔍 First 8 at mem[1024]: \((1024..<1032).map { memFloat[$0] })")
        print("🔍 First 8 at mem[2048]: \((2048..<2056).map { memFloat[$0] })")
        print("🔍 First 8 at mem[3072]: \((3072..<3080).map { memFloat[$0] })")
        print("🔍 Around mid at mem[3584]: \((3584..<3592).map { memFloat[$0] })")

        var produced = [Float]()
        produced.reserveCapacity(input.count)
        var cursor = 0
        while cursor < input.count {
            let remaining = input.count - cursor
            let runSize = min(framesPerRun, remaining)
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

    /// Feed a unit impulse into the partitionedConvolve pipeline with the 808
    /// as the IR. The wet output should be a Hann-COLA-reconstructed
    /// approximation of the IR itself. Print 40 contiguous samples every
    /// 2000 samples so if the output latches on a constant we can see it.
    func testDiagnoseWith808KickIR() throws {
        let assetsDir = "/Users/alecresende/code/swift/dgen/Assets"
        let irPath = "\(assetsDir)/808kicklong.wav"
        guard let ir = loadWAV(irPath), !ir.isEmpty else {
            XCTFail("Failed to load IR at \(irPath)")
            return
        }
        // Normalize IR so convolution output is in a reasonable range.
        let peak = ir.map { abs($0) }.max() ?? 1
        let normIR = ir.map { $0 / max(peak, 1e-12) }
        print("IR loaded: \(ir.count) samples (peak=\(peak)), K=\((ir.count + 255) / 256)")

        // Use the real 808 IR to see the actual scale users hear.
        let irForTest = normIR

        // Input: continuous cosine that's well-supported across the Hann window.
        let signalLen = 8192
        var signal = [Float](repeating: 0, count: signalLen)
        for i in 0..<signalLen {
            signal[i] = Foundation.cos(2.0 * Float.pi * Float(i) * 440.0 / 44100.0)
        }

        let output = try runPipeline(
            input: signal, ir: irForTest, N: 1024, hopSize: 256, framesPerRun: 512)

        print("\n=== First 1600 output samples ===")
        for row in 0..<16 {
            let start = row * 100
            let end = min(start + 16, output.count)
            let vals = output[start..<end].map { String(format: "%+.4f", $0) }.joined(separator: " ")
            print("  [\(start)..<\(end)]: \(vals)")
        }
        print("\n=== Output samples (looking for latched-constant pathology) ===")
        let stride = 2000
        for start in Swift.stride(from: 0, to: output.count, by: stride) {
            let end = min(start + 40, output.count)
            let window = Array(output[start..<end])
            let maxAbs = window.map { abs($0) }.max() ?? 0
            let uniqueValues = Set(window.map { Int($0 * 1000) })
            print(
                "  frames [\(start)..<\(end)]: maxAbs=\(String(format: "%.4f", maxAbs)) "
                    + "unique~=\(uniqueValues.count) "
                    + "first8=\(window.prefix(8).map { String(format: "%.3f", $0) })"
            )
        }

        // Run direct convolution for reference and compare overall energy.
        let reference = directConvolve(signal, irForTest)
        let refEnergy = reference.map { $0 * $0 }.reduce(0, +)
        let outEnergy = output.map { $0 * $0 }.reduce(0, +)
        let compareLen = min(reference.count, output.count)
        let refPeak = (0..<compareLen).map { abs(reference[$0]) }.max() ?? 0
        let outPeak = output.map { abs($0) }.max() ?? 0
        print(
            "\n=== Energy summary ===\n"
                + "  reference: energy=\(refEnergy) peak=\(refPeak)\n"
                + "  output   : energy=\(outEnergy) peak=\(outPeak)\n"
        )

        // Detect "latched" behavior: any 40-frame window where the max-min
        // range is smaller than 0.1% of peak is suspicious.
        var latchedWindows = 0
        let windowLen = 40
        var i = 0
        while i + windowLen < output.count {
            let w = output[i..<(i + windowLen)]
            let mn = w.min() ?? 0
            let mx = w.max() ?? 0
            let outPeakF: Float = outPeak
            let small: Float = 0.001 * outPeakF
            let threshold: Float = small > 1e-8 ? small : 1e-8
            if mx - mn < threshold && abs(mx) > 0.05 * outPeakF {
                latchedWindows += 1
            }
            i += windowLen
        }
        print("  latched 40-frame windows with |v| > 5% peak: \(latchedWindows) out of \(output.count / windowLen)")

        XCTAssertGreaterThan(outPeak, 0, "output is silent")
        for v in output {
            XCTAssertFalse(v.isNaN)
            XCTAssertFalse(v.isInfinite)
        }
    }
}

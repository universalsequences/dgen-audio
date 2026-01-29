import Foundation
import XCTest

@testable import DGen

/// Tests for neural synthesis using MLPs to control differentiable synthesizers.
/// Uses spectral loss for perceptually meaningful training.
final class NeuralSynthTests: XCTestCase {

    // MARK: - Test 1: Mini-DDSP with Spectral Loss

    /// MLP outputs harmonic amplitudes, trained with spectral loss.
    /// This demonstrates learning to decompose timbre into harmonics.
    func testMiniDDSP_HarmonicSynth() throws {
        // Load piano samples
        let samplesURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Assets/piano_samples_short.json")

        guard FileManager.default.fileExists(atPath: samplesURL.path) else {
            print("Skipping - piano_samples_short.json not found")
            return
        }

        let jsonData = try Data(contentsOf: samplesURL)
        let json = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let targetSamples = (json["samples"] as! [Double]).map { Float($0) }
        let sampleRate = json["target_sample_rate"] as! Int
        let f0 = Float(json["detected_pitch_hz"] as! Double)

        // Use fewer frames for speed, but enough for spectral loss window
        let frameCount = 128  // Reduced for testing
        let windowSize = 64
        let numHarmonics = 6

        print(
            "Mini-DDSP: \(frameCount) frames, f0=\(f0) Hz, \(numHarmonics) harmonics, spectral loss"
        )

        let g = Graph(sampleRate: Float(sampleRate))

        // Truncate/resample target to frameCount
        let targetData = Array(targetSamples.prefix(frameCount))
        let targetTensor = g.tensor(shape: [frameCount, 1], data: targetData)

        // MLP: time → harmonic amplitudes
        let hiddenSize = 12

        let W1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: (0..<hiddenSize).map { _ in Float.random(in: -0.5...0.5) }, name: "W1")
        let b1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: [Float](repeating: 0.0, count: hiddenSize), name: "b1")
        let W2 = TensorParameter(
            graph: g, shape: [hiddenSize, numHarmonics],
            data: (0..<hiddenSize * numHarmonics).map { _ in Float.random(in: -0.3...0.3) },
            name: "W2")
        let b2 = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: [Float](repeating: 0.1, count: numHarmonics), name: "b2")

        let zero = g.n(.constant(0.0))
        let one = g.n(.constant(1.0))

        // BATCHED MLP: Pre-compute ALL time values as tensor [frameCount, 1]
        // This lets the MLP matmuls run in parallel across all frames!
        let timeData = (0..<frameCount).map { Float($0) / Float(frameCount - 1) }
        let allTimes = g.tensor(shape: [frameCount, 1], data: timeData)

        // MLP forward on BATCHED input: [frameCount, 1] → [frameCount, numHarmonics]
        // These matmuls are fully parallel!
        let h1 = try g.matmul(allTimes, W1.node())  // [frameCount, 1] × [1, 12] = [frameCount, 12]
        let h1_bias = g.n(.add, h1, b1.node())       // b1 [1,12] broadcasts to [frameCount, 12]
        let h1_act = g.n(.tanh, h1_bias)
        let h2 = try g.matmul(h1_act, W2.node())    // [frameCount, 12] × [12, 6] = [frameCount, 6]
        let h2_bias = g.n(.add, h2, b2.node())       // b2 [1,6] broadcasts to [frameCount, 6]

        // Sigmoid: 1 / (1 + exp(-x))
        let neg_h2 = g.n(.mul, h2_bias, g.n(.constant(-1.0)))
        let exp_neg = g.n(.exp, neg_h2)
        let amplitudes = g.n(.div, one, g.n(.add, one, exp_neg))  // [frameCount, 6]

        // Frame index for looking up amplitudes - 0 to frameCount-1
        let frameIdx = g.phasor(freq: g.n(.constant(Float(sampleRate) / Float(frameCount))), reset: zero)
        let frameIdxScaled = g.n(.mul, frameIdx, g.n(.constant(Float(frameCount - 1))))

        // Generate harmonics - phases are parallel, amplitude lookup uses peek
        let twoPi = g.n(.constant(Float.pi * 2.0))

        var harmonicNodes: [NodeID] = []
        for k in 1...numHarmonics {
            let freqK = g.n(.constant(f0 * Float(k)))
            // Uses deterministicPhasor since freq is constant - parallelizable!
            let phase = g.phasor(freq: freqK, reset: zero)
            let sine = g.n(.sin, g.n(.mul, twoPi, phase))

            // Look up amplitude for this frame from pre-computed [frameCount, 6] tensor
            let kIdx = g.n(.constant(Float(k - 1)))
            let ampK = try g.peek(tensor: amplitudes, index: frameIdxScaled, channel: kIdx)
            harmonicNodes.append(g.n(.mul, sine, ampK))
        }

        // Sum harmonics
        var synthOutput = harmonicNodes[0]
        for i in 1..<harmonicNodes.count {
            synthOutput = g.n(.add, synthOutput, harmonicNodes[i])
        }
        synthOutput = g.n(.mul, synthOutput, g.n(.constant(1.0 / Float(numHarmonics))))
        let targetSample = try g.peek(tensor: targetTensor, index: frameIdxScaled, channel: zero)

        // SPECTRAL LOSS - perceptually meaningful!
        let spectralLoss = g.spectralLoss(synthOutput, targetSample, windowSize: windowSize)

        // Also add small MSE for sample-level accuracy
        let diff = g.n(.sub, synthOutput, targetSample)
        let mseLoss = g.n(.mul, diff, diff)

        // Combined loss: spectral + small MSE term
        // Scale up by 1e6 to get meaningful gradients (they're ~1e-6 otherwise)
        let lossRaw = g.n(.add, spectralLoss, g.n(.mul, mseLoss, g.n(.constant(0.1))))
        let loss = g.n(.mul, lossRaw, g.n(.constant(1.0)))
        _ = g.n(.output(0), loss)

        print("gonna try compiling")
        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))
        print("finished compiling")

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)
        print("finished initialing metal compiled kernel")

        // Dump kernels for analysis
        var kernelDump = "// Mini-DDSP Kernels - \(compileResult.kernels.count) total\n\n"
        for (i, kernel) in compileResult.kernels.enumerated() {
            kernelDump += "// ===== KERNEL \(i): \(kernel.name) =====\n"
            kernelDump += "// ThreadGroupSize: \(kernel.threadGroupSize)\n"
            kernelDump += "// Buffers: \(kernel.buffers)\n\n"
            kernelDump += kernel.source
            kernelDump += "\n\n"
        }
        try kernelDump.write(toFile: "/tmp/miniddsp_kernels.metal", atomically: true, encoding: .utf8)
        print("Wrote \(compileResult.kernels.count) kernels to /tmp/miniddsp_kernels.metal")

        // With loss scaled up by 1e6, use normal learning rate
        let ctx = TrainingContext(
            tensorParameters: [W1, b1, W2, b2],
            optimizer: Adam(lr: 0.3),
            lossNode: loss)

        print("initial memory")
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Helper to print gradient/weight stats for a tensor parameter
        func printTensorStats(_ param: TensorParameter, label: String) {
            let weights = param.data
            let grads = param.grads

            let wMin = weights.min() ?? 0
            let wMax = weights.max() ?? 0
            let wMean = weights.reduce(0, +) / Float(weights.count)

            let gMin = grads.min() ?? 0
            let gMax = grads.max() ?? 0
            let gMean = grads.reduce(0, +) / Float(grads.count)
            let gAbsMean = grads.map { abs($0) }.reduce(0, +) / Float(grads.count)

            print("  \(label): weights[min=\(String(format: "%.4f", wMin)), max=\(String(format: "%.4f", wMax)), mean=\(String(format: "%.4f", wMean))]")
            print("           grads[min=\(String(format: "%.4e", gMin)), max=\(String(format: "%.4e", gMax)), mean=\(String(format: "%.4e", gMean)), |mean|=\(String(format: "%.4e", gAbsMean))]")
        }
        print("zero grad initial")
        // Initial loss
        ctx.zeroGrad()
        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: ctx.getMemory(),
                    frameCount: frameCount)
            }
        }
        let initialLoss = outputBuffer.reduce(0, +) / Float(frameCount)
        print("Initial loss: \(initialLoss)")

        // Check if tensor parameters have gradient IDs allocated
        print("Tensor gradient IDs (nil = no gradients being tracked!):")
        print("  W1.baseGradId: \(W1.baseGradId.map { String($0) } ?? "nil")")
        print("  b1.baseGradId: \(b1.baseGradId.map { String($0) } ?? "nil")")
        print("  W2.baseGradId: \(W2.baseGradId.map { String($0) } ?? "nil")")
        print("  b2.baseGradId: \(b2.baseGradId.map { String($0) } ?? "nil")")

        // Extract initial gradients to see what we're starting with
        ctx.step()  // This extracts gradients
        print("Initial gradients (before training):")
        printTensorStats(W1, label: "W1")
        printTensorStats(b1, label: "b1")
        printTensorStats(W2, label: "W2")
        printTensorStats(b2, label: "b2")

        // Train - fewer epochs since spectral loss is more informative
        let epochs = 100
        for epoch in 0..<epochs {
            // Forward+backward pass first
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            // Print BEFORE step() applies gradients (grads are populated from backward pass)
            if epoch % 2 == 0 || epoch == epochs - 1 {
                let loss = outputBuffer.reduce(0, +) / Float(frameCount)
                print("Epoch \(epoch): loss = \(loss)")

                // Extract gradients manually to see them before step() might modify things
                let tensorGrads = [W1, b1, W2, b2].map { param -> [Float] in
                    guard let baseGradId = param.baseGradId else { return [] }
                    let gradPtr = runtime.getGradientsBuffer()
                    var grads = [Float](repeating: 0.0, count: param.size)
                    for i in 0..<param.size {
                        let gradId = baseGradId + i
                        var total: Float = 0
                        for f in 0..<frameCount {
                            total += gradPtr[frameCount * gradId + f]
                        }
                        grads[i] = total / Float(frameCount)
                    }
                    return grads
                }

                print("  Gradients (from buffer):")
                for (i, name) in ["W1", "b1", "W2", "b2"].enumerated() {
                    let g = tensorGrads[i]
                    if g.isEmpty {
                        print("    \(name): NO GRADIENT ID")
                    } else {
                        let gMin = g.min() ?? 0
                        let gMax = g.max() ?? 0
                        let gAbsMean = g.map { abs($0) }.reduce(0, +) / Float(g.count)
                        print("    \(name): min=\(String(format: "%.4e", gMin)), max=\(String(format: "%.4e", gMax)), |mean|=\(String(format: "%.4e", gAbsMean))")
                    }
                }
            }

            // Now apply gradients
            ctx.step()
        }

        let finalLoss = outputBuffer.reduce(0, +) / Float(frameCount)
        let numParams = hiddenSize + hiddenSize + hiddenSize * numHarmonics + numHarmonics
        let improvement = (1.0 - finalLoss / initialLoss) * 100
        print(
            "Final: \(finalLoss), params: \(numParams), improvement: \(String(format: "%.1f", improvement))%"
        )

        XCTAssertLessThan(finalLoss, initialLoss * 0.9, "Should improve with spectral loss")
    }

    // MARK: - Test 1b: Simple Gradient Flow Diagnostic

    /// Simplest possible test: learn a single amplitude to match a sine wave.
    /// If this doesn't work, there's a fundamental gradient bug.
    func testSimpleGradientFlow() throws {
        let frameCount = 256
        let sampleRate: Float = 1000.0
        let targetFreq: Float = 50.0  // Simple frequency

        print("Simple gradient test: learn amplitude to match sine wave")

        let g = Graph()

        // Single learnable amplitude parameter
        let amp = TensorParameter(
            graph: g, shape: [1, 1],
            data: [0.1],  // Start at 0.1, target is 1.0
            name: "amp")

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Oscillator
        let freqNorm = g.n(.constant(targetFreq / sampleRate))
        let phaseCell = g.alloc()
        let phase = g.n(.phasor(phaseCell), freqNorm, zero)
        let osc = g.n(.sin, g.n(.mul, twoPi, phase))

        // Synth output: osc * learnable_amp
        let ampScalar = g.n(.sum, amp.node())  // Convert [1,1] tensor to scalar
        let synthOutput = g.n(.mul, osc, ampScalar)

        // Target: same oscillator with amplitude 1.0
        let targetOutput = osc  // amplitude = 1.0

        // MSE loss
        let diff = g.n(.sub, synthOutput, targetOutput)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [amp],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        print("Initial amp: \(amp.data[0])")
        print("Target amp: 1.0")
        print("amp.baseGradId: \(amp.baseGradId.map { String($0) } ?? "nil")")

        // Train
        let epochs = 50
        for epoch in 0..<epochs {
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let lossVal = outputBuffer.reduce(0, +) / Float(frameCount)

            // Get gradient before step
            var grad: Float = 0
            if let baseGradId = amp.baseGradId {
                let gradPtr = runtime.getGradientsBuffer()
                for f in 0..<frameCount {
                    grad += gradPtr[frameCount * baseGradId + f]
                }
                grad /= Float(frameCount)
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                print("Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amp=\(String(format: "%.4f", amp.data[0])), grad=\(String(format: "%.6f", grad))")
            }

            ctx.step()
        }

        print("Final amp: \(amp.data[0]) (target: 1.0)")

        // Should have learned amplitude close to 1.0
        XCTAssertGreaterThan(amp.data[0], 0.8, "Should learn amplitude toward 1.0")
        XCTAssertLessThan(amp.data[0], 1.2, "Should not overshoot too much")
    }

    // MARK: - Test 1c: Learn Single Amplitude via Peek

    /// Simplest test: learn ONE amplitude via peek to verify peek gradient flow.
    func testLearnSingleAmplitudeViaPeek() throws {
        let frameCount = 256
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0

        // Target amplitude
        let targetAmp: Float = 0.8

        print("Learn single amplitude via peek: target = \(targetAmp)")

        let g = Graph()

        // Learnable amplitude stored in a [1,1] tensor
        let ampTensor = TensorParameter(
            graph: g, shape: [1, 1],
            data: [0.3],  // Start far from target
            name: "amp")

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Oscillator
        let freqNorm = g.n(.constant(f0 / sampleRate))
        let phaseCell = g.alloc()
        let phase = g.n(.phasor(phaseCell), freqNorm, zero)
        let osc = g.n(.sin, g.n(.mul, twoPi, phase))

        // Read amplitude via peek
        let amp = try g.peek(tensor: ampTensor.node(), index: zero, channel: zero)

        // Synth output
        let synthOutput = g.n(.mul, osc, amp)

        // Target output
        let targetConst = g.n(.constant(targetAmp))
        let targetOutput = g.n(.mul, osc, targetConst)

        // MSE loss
        let diff = g.n(.sub, synthOutput, targetOutput)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [ampTensor],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        print("Initial amp: \(ampTensor.data[0])")
        print("ampTensor.baseGradId: \(ampTensor.baseGradId.map { String($0) } ?? "nil")")

        // Train
        let epochs = 50
        for epoch in 0..<epochs {
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let lossVal = outputBuffer.reduce(0, +) / Float(frameCount)

            if epoch % 10 == 0 || epoch == epochs - 1 {
                // Get raw gradient
                var grad: Float = 0
                if let baseGradId = ampTensor.baseGradId {
                    let gradPtr = runtime.getGradientsBuffer()
                    for f in 0..<frameCount {
                        grad += gradPtr[frameCount * baseGradId + f]
                    }
                    grad /= Float(frameCount)
                }
                print("Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amp=\(String(format: "%.4f", ampTensor.data[0])), grad=\(String(format: "%.6e", grad))")
            }

            ctx.step()
        }

        print("Final amp: \(ampTensor.data[0]) (target: \(targetAmp))")

        XCTAssertEqual(ampTensor.data[0], targetAmp, accuracy: 0.1,
            "Should learn amplitude close to target")
    }

    // MARK: - Test 1d: Learn TWO Amplitudes via Peek

    /// Test with just 2 amplitudes to understand the multi-peek issue.
    func testLearnTwoAmplitudes() throws {
        // The phasor uses graph.sampleRate internally, so we pass the frequency in Hz.
        // f0=441Hz at 44100Hz gives period=100 samples
        // frameCount=1000 gives 10 complete periods at 441Hz
        let frameCount = 1000
        let f0: Float = 441.0  // Base frequency in Hz

        // Target amplitudes: [1.0, 0.3]
        let targetAmps: [Float] = [1.0, 0.3]

        print("Learn 2 amplitudes: target = \(targetAmps)")

        let g = Graph()  // Uses default sampleRate=44100

        // Learnable amplitudes in [1, 2] tensor
        let amps = TensorParameter(
            graph: g, shape: [1, 2],
            data: [0.5, 0.5],  // Start at 0.5 for both
            name: "amps")

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Two oscillators at different frequencies (f0 and 2*f0)
        // Phasor expects frequency in Hz, divides by sampleRate internally
        let freq1 = g.n(.constant(f0))
        let phase1Cell = g.alloc()
        let phase1 = g.n(.phasor(phase1Cell), freq1, zero)
        let osc1 = g.n(.sin, g.n(.mul, twoPi, phase1))

        let freq2 = g.n(.constant(2.0 * f0))
        let phase2Cell = g.alloc()
        let phase2 = g.n(.phasor(phase2Cell), freq2, zero)
        let osc2 = g.n(.sin, g.n(.mul, twoPi, phase2))

        // Read amplitudes via peek
        let amp1 = try g.peek(tensor: amps.node(), index: zero, channel: zero)
        let amp2 = try g.peek(tensor: amps.node(), index: zero, channel: g.n(.constant(1.0)))

        // Synth output
        let synth1 = g.n(.mul, osc1, amp1)
        let synth2 = g.n(.mul, osc2, amp2)
        let synthOutput = g.n(.add, synth1, synth2)

        // Target output
        let target1 = g.n(.mul, osc1, g.n(.constant(targetAmps[0])))
        let target2 = g.n(.mul, osc2, g.n(.constant(targetAmps[1])))
        let targetOutput = g.n(.add, target1, target2)

        // MSE loss
        let diff = g.n(.sub, synthOutput, targetOutput)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [amps],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        print("Initial amps: \(amps.data)")
        print("amps.baseGradId: \(amps.baseGradId.map { String($0) } ?? "nil")")

        // Write all kernel sources to a file for analysis
        var allKernels = ""
        for (i, kernel) in compileResult.kernels.enumerated() {
            allKernels += "// ========== KERNEL \(i): \(kernel.name) ==========\n"
            allKernels += kernel.source
            allKernels += "\n\n"
        }
        try! allKernels.write(toFile: "/tmp/dgen_kernels.metal", atomically: true, encoding: .utf8)
        print("Wrote kernels to /tmp/dgen_kernels.metal")

        // Train
        let epochs = 100
        for epoch in 0..<epochs {
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let lossVal = outputBuffer.reduce(0, +) / Float(frameCount)

            if epoch % 20 == 0 || epoch == epochs - 1 {
                // Get raw gradients for each element
                var grads: [Float] = [0, 0]
                if let baseGradId = amps.baseGradId {
                    let gradPtr = runtime.getGradientsBuffer()
                    for elem in 0..<2 {
                        for f in 0..<frameCount {
                            grads[elem] += gradPtr[frameCount * (baseGradId + elem) + f]
                        }
                        grads[elem] /= Float(frameCount)
                    }
                }
                let ampsStr = amps.data.map { String(format: "%.4f", $0) }.joined(separator: ", ")
                let gradsStr = grads.map { String(format: "%.4e", $0) }.joined(separator: ", ")
                print("Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amps=[\(ampsStr)], grads=[\(gradsStr)]")
            }

            ctx.step()
        }

        print("Final amps: \(amps.data)")
        print("Target amps: \(targetAmps)")

        XCTAssertEqual(amps.data[0], targetAmps[0], accuracy: 0.15, "Amp 0 should be close to target")
        XCTAssertEqual(amps.data[1], targetAmps[1], accuracy: 0.15, "Amp 1 should be close to target")
    }

    // MARK: - Test 1e: Learn Harmonic Amplitudes (No MLP)

    /// Learn harmonic amplitudes directly (no MLP) to match a target.
    /// This isolates whether the issue is MLP gradients or the harmonic synth itself.
    func testLearnHarmonicAmplitudes() throws {
        let frameCount = 512
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0
        let numHarmonics = 4

        // Target amplitudes we want to learn
        let targetAmps: [Float] = [1.0, 0.5, 0.25, 0.1]

        print("Learn harmonic amplitudes: target = \(targetAmps)")

        let g = Graph()

        // Learnable amplitudes (start at uniform 0.5)
        let amps = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: [Float](repeating: 0.5, count: numHarmonics),
            name: "amps")

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Generate shared oscillators (same phase for synth and target)
        var oscillators: [NodeID] = []
        for k in 1...numHarmonics {
            let freqNorm = g.n(.constant(f0 * Float(k) / sampleRate))
            let phaseCell = g.alloc()
            let phase = g.n(.phasor(phaseCell), freqNorm, zero)
            let sine = g.n(.sin, g.n(.mul, twoPi, phase))
            oscillators.append(sine)
        }

        // Generate synth harmonics with learnable amplitudes (using shared oscillators)
        var synthHarmonics: [NodeID] = []
        for k in 1...numHarmonics {
            let kIdx = g.n(.constant(Float(k - 1)))
            let ampK = try g.peek(tensor: amps.node(), index: zero, channel: kIdx)
            synthHarmonics.append(g.n(.mul, oscillators[k - 1], ampK))
        }

        var synthOutput = synthHarmonics[0]
        for i in 1..<synthHarmonics.count {
            synthOutput = g.n(.add, synthOutput, synthHarmonics[i])
        }

        // Generate target harmonics with fixed amplitudes (using SAME oscillators)
        var targetHarmonics: [NodeID] = []
        for k in 1...numHarmonics {
            let ampK = g.n(.constant(targetAmps[k - 1]))
            targetHarmonics.append(g.n(.mul, oscillators[k - 1], ampK))
        }

        var targetOutput = targetHarmonics[0]
        for i in 1..<targetHarmonics.count {
            targetOutput = g.n(.add, targetOutput, targetHarmonics[i])
        }

        // MSE loss
        let diff = g.n(.sub, synthOutput, targetOutput)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [amps],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        print("Initial amps: \(amps.data)")
        print("amps.nodeId: \(amps.nodeId)")
        print("amps.baseGradId: \(amps.baseGradId.map { String($0) } ?? "nil")")

        // Debug: check what's in tensorGradients
        print("tensorGradients: \(compileResult.context.tensorGradients)")

        // Train
        let epochs = 100
        for epoch in 0..<epochs {
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount)
                }
            }

            let lossVal = outputBuffer.reduce(0, +) / Float(frameCount)

            if epoch % 20 == 0 || epoch == epochs - 1 {
                let ampsStr = amps.data.map { String(format: "%.3f", $0) }.joined(separator: ", ")
                print("Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amps=[\(ampsStr)]")

                // Debug: print raw gradient values for each element
                if let baseGradId = amps.baseGradId {
                    let gradPtr = runtime.getGradientsBuffer()
                    var elemGrads: [Float] = []
                    for elem in 0..<amps.size {
                        var total: Float = 0
                        for f in 0..<frameCount {
                            total += gradPtr[frameCount * (baseGradId + elem) + f]
                        }
                        elemGrads.append(total / Float(frameCount))
                    }
                    let gradsStr = elemGrads.map { String(format: "%.6e", $0) }.joined(separator: ", ")
                    print("  Element gradients: [\(gradsStr)]")
                }
            }

            ctx.step()
        }

        print("Final amps: \(amps.data)")
        print("Target amps: \(targetAmps)")

        // Check each amplitude is close to target
        for i in 0..<numHarmonics {
            XCTAssertEqual(amps.data[i], targetAmps[i], accuracy: 0.1,
                "Harmonic \(i+1) amplitude should be close to target")
        }
    }

    // MARK: - Test 2: Pitch-Conditioned Synthesis (Generalization Test)

    /// Train MLP on 2 pitches, test on unseen pitch between them.
    /// This demonstrates the VALUE of neural networks: generalization.
    func testConditionedSynth_Generalization() throws {
        let trainingPitches: [Float] = [200.0, 400.0]  // Two octaves apart
        let testPitch: Float = 283.0  // Between them (~D4)

        let frameCount = 128
        let sampleRate: Float = 2048.0

        print("Conditioned Synth: training on \(trainingPitches), testing on \(testPitch)")

        let g = Graph()
        let hiddenSize = 16

        // MLP: (time, pitch) → amplitude envelope
        let W1 = TensorParameter(
            graph: g, shape: [2, hiddenSize],
            data: (0..<2 * hiddenSize).map { _ in Float.random(in: -0.5...0.5) }, name: "W1")
        let b1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: [Float](repeating: 0.0, count: hiddenSize), name: "b1")
        let W2 = TensorParameter(
            graph: g, shape: [hiddenSize, 1],
            data: (0..<hiddenSize).map { _ in Float.random(in: -0.3...0.3) }, name: "W2")
        let b2 = TensorParameter(
            graph: g, shape: [1, 1],
            data: [0.0], name: "b2")

        let zero = g.n(.constant(0.0))
        let one = g.n(.constant(1.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Time
        let timeCell = g.alloc()
        let t = g.n(.phasor(timeCell), g.n(.constant(1.0 / Float(frameCount))), zero)

        // Pitch input from memory (we'll set this externally)
        let pitchInputCell = g.alloc()
        let pitchNorm = g.n(.memoryRead(pitchInputCell), zero)

        // Stack inputs
        let inputTensor = try g.stack([t, pitchNorm], shape: [1, 2])

        // MLP forward
        let h1 = try g.matmul(inputTensor, W1.node())
        let h1_bias = g.n(.add, h1, b1.node())
        let h1_act = g.n(.tanh, h1_bias)
        let h2 = try g.matmul(h1_act, W2.node())
        let envelope_raw = g.n(.add, h2, b2.node())

        // Sigmoid output
        let neg = g.n(.mul, envelope_raw, g.n(.constant(-1.0)))
        let expNeg = g.n(.exp, neg)
        let envelope = g.n(.div, one, g.n(.add, one, expNeg))
        let envelopeScalar = g.n(.sum, envelope)

        // Oscillator
        let pitchHz = g.n(.add, g.n(.constant(100.0)), g.n(.mul, pitchNorm, g.n(.constant(400.0))))
        let freqNorm = g.n(.div, pitchHz, g.n(.constant(sampleRate)))
        let oscCell = g.alloc()
        let phase = g.n(.phasor(oscCell), freqNorm, zero)
        let osc = g.n(.sin, g.n(.mul, twoPi, phase))

        let synthOutput = g.n(.mul, osc, envelopeScalar)

        // Target: NON-TRIVIAL pitch-dependent behavior
        // Higher pitch = faster decay AND different attack shape
        // envelope = (1 - exp(-attackRate * t)) * exp(-decayRate * t)
        // where attackRate and decayRate both depend on pitch
        let attackRate = g.n(.add, g.n(.constant(10.0)), g.n(.mul, pitchNorm, g.n(.constant(20.0))))
        let decayRate = g.n(.add, g.n(.constant(2.0)), g.n(.mul, pitchNorm, g.n(.constant(6.0))))

        let negAttack = g.n(.mul, g.n(.constant(-1.0)), g.n(.mul, attackRate, t))
        let attackPart = g.n(.sub, one, g.n(.exp, negAttack))

        let negDecay = g.n(.mul, g.n(.constant(-1.0)), g.n(.mul, decayRate, t))
        let decayPart = g.n(.exp, negDecay)

        let targetEnvelope = g.n(.mul, attackPart, decayPart)
        let targetOutput = g.n(.mul, osc, targetEnvelope)

        // Loss
        let diff = g.n(.sub, synthOutput, targetOutput)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [W1, b1, W2, b2],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let pitchCellPhysical =
            compileResult.cellAllocations.cellMappings[pitchInputCell] ?? pitchInputCell
        let memoryRaw = ctx.getMemory()
        let memory = memoryRaw.assumingMemoryBound(to: Float.self)

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Get initial loss on training pitches
        var initialAvgLoss: Float = 0
        for pitch in trainingPitches {
            let pitchNormalized = (pitch - 100.0) / 400.0
            memory[pitchCellPhysical] = pitchNormalized
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: memoryRaw,
                        frameCount: frameCount)
                }
            }
            initialAvgLoss += outputBuffer.reduce(0, +) / Float(frameCount)
        }
        initialAvgLoss /= Float(trainingPitches.count)
        print("Initial avg training loss: \(initialAvgLoss)")

        // Train
        let epochs = 100
        for epoch in 0..<epochs {
            for pitch in trainingPitches {
                let pitchNormalized = (pitch - 100.0) / 400.0
                memory[pitchCellPhysical] = pitchNormalized
                ctx.zeroGrad()
                inputBuffer.withUnsafeBufferPointer { inPtr in
                    outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                        runtime.runWithMemory(
                            outputs: outPtr.baseAddress!,
                            inputs: inPtr.baseAddress!,
                            memory: memoryRaw,
                            frameCount: frameCount)
                    }
                }
                ctx.step()
            }

            if epoch % 25 == 0 || epoch == epochs - 1 {
                var totalLoss: Float = 0
                for pitch in trainingPitches {
                    let pitchNormalized = (pitch - 100.0) / 400.0
                    memory[pitchCellPhysical] = pitchNormalized
                    ctx.zeroGrad()
                    inputBuffer.withUnsafeBufferPointer { inPtr in
                        outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                            runtime.runWithMemory(
                                outputs: outPtr.baseAddress!,
                                inputs: inPtr.baseAddress!,
                                memory: memoryRaw,
                                frameCount: frameCount)
                        }
                    }
                    totalLoss += outputBuffer.reduce(0, +) / Float(frameCount)
                }
                print("Epoch \(epoch): avg loss = \(totalLoss / Float(trainingPitches.count))")
            }
        }

        // Test generalization on UNSEEN pitch
        let testPitchNorm = (testPitch - 100.0) / 400.0
        memory[pitchCellPhysical] = testPitchNorm
        ctx.zeroGrad()
        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: memoryRaw,
                    frameCount: frameCount)
            }
        }
        let testLoss = outputBuffer.reduce(0, +) / Float(frameCount)

        // Final training loss
        var finalTrainLoss: Float = 0
        for pitch in trainingPitches {
            let pitchNormalized = (pitch - 100.0) / 400.0
            memory[pitchCellPhysical] = pitchNormalized
            ctx.zeroGrad()
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: memoryRaw,
                        frameCount: frameCount)
                }
            }
            finalTrainLoss += outputBuffer.reduce(0, +) / Float(frameCount)
        }
        finalTrainLoss /= Float(trainingPitches.count)

        print("Final training loss: \(finalTrainLoss)")
        print("Test pitch \(testPitch) Hz (UNSEEN): loss = \(testLoss)")

        // Verify training improved
        XCTAssertLessThan(
            finalTrainLoss, initialAvgLoss * 0.3, "Training should improve significantly")

        // Verify generalization - test loss should be reasonable (not way worse than training)
        XCTAssertLessThan(testLoss, finalTrainLoss * 5.0, "Should generalize to unseen pitch")

        print("SUCCESS: MLP generalizes to unseen pitch!")
    }
}

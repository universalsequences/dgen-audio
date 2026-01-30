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

        // PER-FRAME MLP: Compute time as normalized frame index [0, 1]
        // Frame index phasor goes from 0 to 1 over frameCount samples
        let frameIdx = g.phasor(
            freq: g.n(.constant(Float(sampleRate) / Float(frameCount))), reset: zero)
        let frameIdxScaled = g.n(.mul, frameIdx, g.n(.constant(Float(frameCount - 1))))

        // MLP forward pass runs per-frame with scalar time input (frameIdx)

        // Hidden layer: h1 = tanh(time * W1 + b1)
        // W1 is [1, hiddenSize], we compute time * W1[0,j] + b1[0,j] for each j
        var h1_activations: [NodeID] = []
        for j in 0..<hiddenSize {
            // W1[0,j] and b1[0,j]
            let w1_j = try g.peek(tensor: W1.node(), index: zero, channel: g.n(.constant(Float(j))))
            let b1_j = try g.peek(tensor: b1.node(), index: zero, channel: g.n(.constant(Float(j))))
            let h1_j_pre = g.n(.add, g.n(.mul, frameIdx, w1_j), b1_j)
            let h1_j = g.n(.tanh, h1_j_pre)
            h1_activations.append(h1_j)
        }

        // Output layer: h2 = sigmoid(h1 * W2 + b2)
        // W2 is [hiddenSize, numHarmonics], compute sum_j(h1[j] * W2[j,k]) + b2[0,k]
        var amplitudeNodes: [NodeID] = []
        for k in 0..<numHarmonics {
            var sum = g.n(.constant(0.0))
            for j in 0..<hiddenSize {
                let w2_jk = try g.peek(
                    tensor: W2.node(), index: g.n(.constant(Float(j))),
                    channel: g.n(.constant(Float(k))))
                sum = g.n(.add, sum, g.n(.mul, h1_activations[j], w2_jk))
            }
            let b2_k = try g.peek(tensor: b2.node(), index: zero, channel: g.n(.constant(Float(k))))
            let h2_k_pre = g.n(.add, sum, b2_k)
            // Sigmoid: 1 / (1 + exp(-x))
            let neg_h2_k = g.n(.mul, h2_k_pre, g.n(.constant(-1.0)))
            let exp_neg_k = g.n(.exp, neg_h2_k)
            let amp_k = g.n(.div, one, g.n(.add, one, exp_neg_k))
            amplitudeNodes.append(amp_k)
        }

        // Generate harmonics - phases are parallel, amplitudes computed above
        let twoPi = g.n(.constant(Float.pi * 2.0))

        var harmonicNodes: [NodeID] = []
        for k in 1...numHarmonics {
            let freqK = g.n(.constant(f0 * Float(k)))
            // Uses deterministicPhasor since freq is constant - parallelizable!
            let phase = g.phasor(freq: freqK, reset: zero)
            let sine = g.n(.sin, g.n(.mul, twoPi, phase))

            // Use the per-frame computed amplitude
            harmonicNodes.append(g.n(.mul, sine, amplitudeNodes[k - 1]))
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
        let loss = g.n(.mul, lossRaw, g.n(.constant(1000.0)))
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
        try kernelDump.write(
            toFile: "/tmp/miniddsp_kernels.metal", atomically: true, encoding: .utf8)
        print("Wrote \(compileResult.kernels.count) kernels to /tmp/miniddsp_kernels.metal")

        // With loss scaled up by 1e6, use normal learning rate
        let ctx = TrainingContext(
            tensorParameters: [W1, b1, W2, b2],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        print("initial memory")
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

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

            print(
                "  \(label): weights[min=\(String(format: "%.4f", wMin)), max=\(String(format: "%.4f", wMax)), mean=\(String(format: "%.4f", wMean))]"
            )
            print(
                "           grads[min=\(String(format: "%.4e", gMin)), max=\(String(format: "%.4e", gMax)), mean=\(String(format: "%.4e", gMean)), |mean|=\(String(format: "%.4e", gAbsMean))]"
            )
        }
        // Initial step (also seeds grads for stats)
        let initialLoss = ctx.runStepGPU()
        print("Initial loss: \(initialLoss)")

        // Check if tensor parameters have gradient IDs allocated
        print("Tensor gradient IDs (nil = no gradients being tracked!):")
        print("  W1.baseGradId: \(W1.baseGradId.map { String($0) } ?? "nil")")
        print("  b1.baseGradId: \(b1.baseGradId.map { String($0) } ?? "nil")")
        print("  W2.baseGradId: \(W2.baseGradId.map { String($0) } ?? "nil")")
        print("  b2.baseGradId: \(b2.baseGradId.map { String($0) } ?? "nil")")

        // Extract initial gradients to see what we're starting with
        print("Initial gradients (after first step):")
        printTensorStats(W1, label: "W1")
        printTensorStats(b1, label: "b1")
        printTensorStats(W2, label: "W2")
        printTensorStats(b2, label: "b2")

        // Train - fewer epochs since spectral loss is more informative
        let epochs = 40
        var finalLoss = initialLoss
        for epoch in 0..<epochs {
            let lossVal = ctx.runStepGPU()
            finalLoss = lossVal

            // Print after step (grads were reduced for this update)
            if epoch % 2 == 0 || epoch == epochs - 1 {
                print("Epoch \(epoch): loss = \(lossVal)")
                printTensorStats(W1, label: "W1")
                printTensorStats(b1, label: "b1")
                printTensorStats(W2, label: "W2")
                printTensorStats(b2, label: "b2")
            }
        }

        let numParams = hiddenSize + hiddenSize + hiddenSize * numHarmonics + numHarmonics
        let improvement = (1.0 - finalLoss / initialLoss) * 100
        print(
            "Final: \(finalLoss), params: \(numParams), improvement: \(String(format: "%.1f", improvement))%"
        )

        XCTAssertLessThan(finalLoss, initialLoss * 0.9, "Should improve with spectral loss")
    }

    // MARK: - Test 1a: Control-Rate MLP -> Tensor -> Audio Read

    /// Build a control-rate amplitude tensor with an MLP, then read it at audio rate.
    /// This mirrors the DDSP pattern: control network + synth, with a time-indexed lookup.
    func testMiniDDSP_ControlRateMLP() throws {
        let frameCount = 64
        let controlFrames = 16
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0
        let numHarmonics = 4
        let hiddenSize = 6

        let g = Graph(sampleRate: sampleRate)

        // Control-rate time tensor [controlFrames, 1]
        let timeData = (0..<controlFrames).map {
            Float($0) / Float(controlFrames - 1)
        }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        // Learnable MLP weights
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

        // Teacher weights (fixed target)
        let teacherW1Data = (0..<hiddenSize).map { i in 0.35 * sin(Float(i) * 0.7) }
        let teacherB1Data = (0..<hiddenSize).map { i in 0.1 * cos(Float(i) * 0.5) }
        let teacherW2Data = (0..<(hiddenSize * numHarmonics)).map { i in
            0.25 * sin(Float(i) * 0.3)
        }
        let teacherB2Data = (0..<numHarmonics).map { i in 0.2 + 0.05 * sin(Float(i)) }

        let teacherW1 = g.tensor(shape: [1, hiddenSize], data: teacherW1Data)
        let teacherB1 = g.tensor(shape: [1, hiddenSize], data: teacherB1Data)
        let teacherW2 = g.tensor(shape: [hiddenSize, numHarmonics], data: teacherW2Data)
        let teacherB2 = g.tensor(shape: [1, numHarmonics], data: teacherB2Data)

        // Helper: tensor MLP with sigmoid output
        func mlpAmps(timeTensor: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let one = g.n(.constant(1.0))
            let h1 = try g.matmul(timeTensor, W1)
            let h1b = g.n(.add, h1, b1)
            let h1a = g.n(.tanh, h1b)
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, h2, b2)
            let neg = g.n(.mul, h2b, g.n(.constant(-1.0)))
            let expNeg = g.n(.exp, neg)
            return g.n(.div, one, g.n(.add, one, expNeg))
        }

        // Control-rate amplitude tensors (row-major layout)
        let ampsPred = try mlpAmps(
            timeTensor: timeTensor, W1: W1.node(), b1: b1.node(), W2: W2.node(), b2: b2.node())
        let ampsTarget = try mlpAmps(
            timeTensor: timeTensor, W1: teacherW1, b1: teacherB1, W2: teacherW2, b2: teacherB2)

        // Reshape to [numHarmonics, controlFrames] so peek(index=harmonic, channel=time)
        // maps to row-major offsets (time * numHarmonics + harmonic).
        let ampsPredView = try g.reshape(ampsPred, to: [numHarmonics, controlFrames])
        let ampsTargetView = try g.reshape(ampsTarget, to: [numHarmonics, controlFrames])

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Audio-rate playhead through control frames
        let frameIdx = g.phasor(
            freq: g.n(.constant(sampleRate / Float(frameCount))), reset: zero)
        let playhead = g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames - 1))))

        // Harmonic synth using control-rate envelopes
        var synthOutput = g.n(.constant(0.0))
        var targetOutput = g.n(.constant(0.0))
        for k in 0..<numHarmonics {
            let kIdx = g.n(.constant(Float(k)))
            let ampK = try g.peek(tensor: ampsPredView, index: kIdx, channel: playhead)
            let targetAmpK = try g.peek(tensor: ampsTargetView, index: kIdx, channel: playhead)

            let freqK = g.n(.constant(f0 * Float(k + 1)))
            let phase = g.phasor(freq: freqK, reset: zero)
            let sine = g.n(.sin, g.n(.mul, twoPi, phase))

            synthOutput = g.n(.add, synthOutput, g.n(.mul, sine, ampK))
            targetOutput = g.n(.add, targetOutput, g.n(.mul, sine, targetAmpK))
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

        // Dump kernels for analysis
        var kernelDump = "// Mini-DDSP Kernels - \(compileResult.kernels.count) total\n\n"
        for (i, kernel) in compileResult.kernels.enumerated() {
            kernelDump += "// ===== KERNEL \(i): \(kernel.name) =====\n"
            kernelDump += "// ThreadGroupSize: \(kernel.threadGroupSize)\n"
            kernelDump += "// ThreadCount: \(kernel.threadCount)\n"
            kernelDump += "// Buffers: \(kernel.buffers)\n\n"
            kernelDump += kernel.source
            kernelDump += "\n\n"
        }
        try kernelDump.write(
            toFile: "/tmp/miniddsp_control_kernels.metal", atomically: true, encoding: .utf8)
        print("Wrote \(compileResult.kernels.count) kernels to /tmp/miniddsp_kernels.metal")

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

        // Initial loss
        let initialLoss = ctx.runStepGPU()

        // Train
        let epochs = 40
        var finalLoss = initialLoss
        for i in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            print("epoch=\(i) loss=\(finalLoss)")
        }
        print("Control-rate MLP test - Initial loss: \(initialLoss), Final loss: \(finalLoss)")

        XCTAssertLessThan(finalLoss, initialLoss * 0.8, "Control-rate MLP should reduce loss")
    }

    // MARK: - Test 1b: Simple Gradient Flow Diagnostic

    /// Simplest possible test: learn a single amplitude to match a sine wave.
    /// If this doesn't work, there's a fundamental gradient bug.
    func testSimpleGradientFlow() throws {
        let frameCount = 256
        let sampleRate: Float = 1000.0
        let targetFreq: Float = 50.0  // Simple frequency

        print("Simple gradient test: learn amplitude to match sine wave")

        let g = Graph(sampleRate: sampleRate)

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

        print("Initial amp: \(amp.data[0])")
        print("Target amp: 1.0")
        print("amp.baseGradId: \(amp.baseGradId.map { String($0) } ?? "nil")")

        // Train
        let epochs = 30
        for epoch in 0..<epochs {
            let lossVal = ctx.runStepGPU()
            let grad = amp.grads.first ?? 0

            if epoch % 10 == 0 || epoch == epochs - 1 {
                print(
                    "Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amp=\(String(format: "%.4f", amp.data[0])), grad=\(String(format: "%.6f", grad))"
                )
            }
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

        let g = Graph(sampleRate: sampleRate)

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

        print("Initial amp: \(ampTensor.data[0])")
        print("ampTensor.baseGradId: \(ampTensor.baseGradId.map { String($0) } ?? "nil")")

        // Train
        let epochs = 50
        for epoch in 0..<epochs {
            let lossVal = ctx.runStepGPU()

            if epoch % 10 == 0 || epoch == epochs - 1 {
                let grad = ampTensor.grads.first ?? 0
                print(
                    "Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amp=\(String(format: "%.4f", ampTensor.data[0])), grad=\(String(format: "%.6e", grad))"
                )
            }
        }

        print("Final amp: \(ampTensor.data[0]) (target: \(targetAmp))")

        XCTAssertEqual(
            ampTensor.data[0], targetAmp, accuracy: 0.1,
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
        let epochs = 40
        for epoch in 0..<epochs {
            let lossVal = ctx.runStepGPU()

            if epoch % 20 == 0 || epoch == epochs - 1 {
                let grads = amps.grads
                let ampsStr = amps.data.map { String(format: "%.4f", $0) }.joined(separator: ", ")
                let gradsStr = grads.map { String(format: "%.4e", $0) }.joined(separator: ", ")
                print(
                    "Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amps=[\(ampsStr)], grads=[\(gradsStr)]"
                )
            }
        }

        print("Final amps: \(amps.data)")
        print("Target amps: \(targetAmps)")

        XCTAssertEqual(
            amps.data[0], targetAmps[0], accuracy: 0.15, "Amp 0 should be close to target")
        XCTAssertEqual(
            amps.data[1], targetAmps[1], accuracy: 0.15, "Amp 1 should be close to target")
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

        let g = Graph(sampleRate: sampleRate)

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
            // phasor expects frequency in Hz (Graph uses sampleRate internally)
            let freqHz = g.n(.constant(f0 * Float(k)))
            let phaseCell = g.alloc()
            let phase = g.n(.phasor(phaseCell), freqHz, zero)
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

        print("Initial amps: \(amps.data)")
        print("amps.nodeId: \(amps.nodeId)")
        print("amps.baseGradId: \(amps.baseGradId.map { String($0) } ?? "nil")")

        // Debug: check what's in tensorGradients
        print("tensorGradients: \(compileResult.context.tensorGradients)")

        // Train
        let epochs = 40
        for epoch in 0..<epochs {
            let lossVal = ctx.runStepGPU()

            if epoch % 20 == 0 || epoch == epochs - 1 {
                let ampsStr = amps.data.map { String(format: "%.3f", $0) }.joined(separator: ", ")
                print("Epoch \(epoch): loss=\(String(format: "%.6f", lossVal)), amps=[\(ampsStr)]")

                // Debug: print raw gradient values for each element
                let gradsStr = amps.grads.map { String(format: "%.6e", $0) }.joined(
                    separator: ", ")
                print("  Element gradients: [\(gradsStr)]")
            }
        }

        print("Final amps: \(amps.data)")
        print("Target amps: \(targetAmps)")

        // Check each amplitude is close to target
        for i in 0..<numHarmonics {
            XCTAssertEqual(
                amps.data[i], targetAmps[i], accuracy: 0.1,
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

        // Get initial loss on training pitches
        var initialAvgLoss: Float = 0
        for pitch in trainingPitches {
            let pitchNormalized = (pitch - 100.0) / 400.0
            let lossVal = ctx.runStepGPU { mem in
                mem[pitchCellPhysical] = pitchNormalized
            }
            initialAvgLoss += lossVal
        }
        initialAvgLoss /= Float(trainingPitches.count)
        print("Initial avg training loss: \(initialAvgLoss)")

        // Train
        let epochs = 100
        for epoch in 0..<epochs {
            var totalLoss: Float = 0
            for pitch in trainingPitches {
                let pitchNormalized = (pitch - 100.0) / 400.0
                let lossVal = ctx.runStepGPU { mem in
                    mem[pitchCellPhysical] = pitchNormalized
                }
                totalLoss += lossVal
            }

            if epoch % 25 == 0 || epoch == epochs - 1 {
                print("Epoch \(epoch): avg loss = \(totalLoss / Float(trainingPitches.count))")
            }
        }

        // Test generalization on UNSEEN pitch
        let testPitchNorm = (testPitch - 100.0) / 400.0
        let testLoss = ctx.runStepGPU { mem in
            mem[pitchCellPhysical] = testPitchNorm
        }

        // Final training loss
        var finalTrainLoss: Float = 0
        for pitch in trainingPitches {
            let pitchNormalized = (pitch - 100.0) / 400.0
            let lossVal = ctx.runStepGPU { mem in
                mem[pitchCellPhysical] = pitchNormalized
            }
            finalTrainLoss += lossVal
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

    // MARK: - Test: Static MLP with Piano Target (12 Harmonics)

    /// Static MLP computes amplitude envelope as tensor ops, then read at audio rate.
    /// Uses real piano samples as target with 12 harmonics and spectral loss.
    func testStaticMLP_PianoTarget() throws {
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

        let frameCount = 128
        let controlFrames = 32  // Control rate resolution
        let windowSize = 64
        let numHarmonics = 6
        let hiddenSize = 16

        print(
            "Static MLP Piano: \(frameCount) frames, \(controlFrames) control frames, f0=\(f0) Hz, \(numHarmonics) harmonics"
        )

        let g = Graph(sampleRate: Float(sampleRate))

        // Target audio tensor
        let targetData = Array(targetSamples.prefix(frameCount))
        let targetTensor = g.tensor(shape: [frameCount, 1], data: targetData)

        // Control-rate time tensor [controlFrames, 1] - normalized 0 to 1
        let timeData = (0..<controlFrames).map {
            Float($0) / Float(controlFrames - 1)
        }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        // Learnable MLP weights
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
            data: (0..<numHarmonics).map { i in 0.5 / Float(i + 1) },  // Harmonic falloff init
            name: "b2")

        // Static MLP: timeTensor -> amplitude envelope tensor
        // This runs as tensor ops (matmul) that can be parallelized by shape
        func mlpAmplitudes(time: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let one = g.n(.constant(1.0))
            // Hidden layer: tanh(time @ W1 + b1)
            let h1 = try g.matmul(time, W1)
            let h1b = g.n(.add, h1, b1)
            let h1a = g.n(.tanh, h1b)
            // Output layer: sigmoid(h1 @ W2 + b2)
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, h2, b2)
            let neg = g.n(.mul, h2b, g.n(.constant(-1.0)))
            let expNeg = g.n(.exp, neg)
            return g.n(.div, one, g.n(.add, one, expNeg))
        }

        // Compute amplitude envelope as static tensor [controlFrames, numHarmonics]
        let ampsTensor = try mlpAmplitudes(
            time: timeTensor, W1: W1.node(), b1: b1.node(), W2: W2.node(), b2: b2.node())

        // Reshape to [numHarmonics, controlFrames] for peek(index=harmonic, channel=time)
        let ampsView = try g.reshape(ampsTensor, to: [numHarmonics, controlFrames])

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Audio-rate playhead: phasor from 0 to 1 over frameCount samples
        let frameIdx = g.phasor(
            freq: g.n(.constant(Float(sampleRate) / Float(frameCount))), reset: zero)
        // Scale to control frame index [0, controlFrames-1]
        let playhead = g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames - 1))))
        // Scale to audio frame index [0, frameCount-1] for target lookup
        let audioIdx = g.n(.mul, frameIdx, g.n(.constant(Float(frameCount - 1))))

        // VECTORIZED HARMONIC SYNTHESIS
        // 1. Create frequency tensor [numHarmonics] with harmonic frequencies
        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)

        // 2. Deterministic phasor on frequency tensor -> phases tensor [numHarmonics]
        // Using deterministicPhasor directly to enable parallel execution
        // phase[k] = fmod(freq[k] / sampleRate * frameIndex, 1.0)
        let phasesTensor = g.n(.deterministicPhasor, freqTensor)

        // 3. sin(2*pi*phases) -> sines tensor [numHarmonics]
        let sinesTensor = g.n(.sin, g.n(.mul, twoPi, phasesTensor))

        // 4. peekRow to get amplitude row at current playhead -> amplitudes tensor [numHarmonics]
        // ampsView is [numHarmonics, controlFrames], peekRow reads a row (all columns for given row index)
        // But we want to read at a specific time (column), so we need the transpose
        let ampsTransposed = try g.transpose(ampsView, axes: [1, 0])  // [controlFrames, numHarmonics]
        let ampsAtTime = try g.peekRow(tensor: ampsTransposed, rowIndex: playhead)

        // 5. Element-wise multiply sines * amplitudes -> weighted tensor [numHarmonics]
        let weightedSines = g.n(.mul, sinesTensor, ampsAtTime)

        // 6. Sum to get final output
        var synthOutput = g.n(.sum, weightedSines)

        // Normalize output
        synthOutput = g.n(.mul, synthOutput, g.n(.constant(1.0 / Float(numHarmonics))))

        // Target sample lookup
        let targetSample = try g.peek(tensor: targetTensor, index: audioIdx, channel: zero)

        // Spectral loss for perceptually meaningful training
        let spectralLoss = g.spectralLoss(synthOutput, targetSample, windowSize: windowSize)

        // Small MSE term for sample-level accuracy
        let diff = g.n(.sub, synthOutput, targetSample)
        let mseLoss = g.n(.mul, diff, diff)

        // Combined loss scaled for meaningful gradients
        let lossRaw = g.n(.add, spectralLoss, g.n(.mul, mseLoss, g.n(.constant(0.1))))
        let loss = g.n(.mul, lossRaw, g.n(.constant(1000.0)))
        _ = g.n(.output(0), loss)

        print("Compiling static MLP piano model...")
        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))
        print("Compiled \(compileResult.kernels.count) kernels")

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        // Dump kernels for analysis
        var kernelDump = "// Static MLP Piano Kernels - \(compileResult.kernels.count) total\n\n"
        for (i, kernel) in compileResult.kernels.enumerated() {
            kernelDump += "// ===== KERNEL \(i): \(kernel.name) =====\n"
            kernelDump += "// ThreadGroupSize: \(kernel.threadGroupSize)\n"
            kernelDump += "// Buffers: \(kernel.buffers)\n\n"
            kernelDump += kernel.source
            kernelDump += "\n\n"
        }
        try kernelDump.write(
            toFile: "/tmp/static_mlp_piano_kernels.metal", atomically: true, encoding: .utf8)
        print("Wrote kernels to /tmp/static_mlp_piano_kernels.metal")

        let ctx = TrainingContext(
            tensorParameters: [W1, b1, W2, b2],
            optimizer: Adam(lr: 0.1),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        // Training
        let initialLoss = ctx.runStepGPU()
        print("Initial loss: \(initialLoss)")

        let epochs = 100
        var finalLoss = initialLoss
        for epoch in 0..<epochs {
            let lossVal = ctx.runStepGPU()
            finalLoss = lossVal
            print("Epoch \(epoch): loss = \(lossVal)")
        }

        let improvement = (1.0 - Double(finalLoss) / Double(initialLoss)) * 100
        print(
            "Static MLP Piano: Initial=\(initialLoss), Final=\(finalLoss), Improvement=\(String(format: "%.1f", improvement))%"
        )

        XCTAssertLessThan(finalLoss, initialLoss * 0.9, "Should improve with spectral loss")
    }
}

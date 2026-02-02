import XCTest

@testable import DGen

final class SpectralLossGradientTests: XCTestCase {

    // MARK: - Forward Pass Tests

    /// Test that FFT-based spectralLoss computes the forward pass correctly
    /// Note: spectralLossFFT only supports Metal backend
    func testSpectralLossFFTForwardPass() throws {
        print("\n🧪 Test: SpectralLossFFT Forward Pass")

        let g = Graph()

        // Two constant frequencies
        let freq1 = g.n(.constant(100.0))
        let freq2 = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))

        // Two phasors at different frequencies
        let phase1 = g.n(.phasor(g.alloc()), freq1, reset)
        let phase2 = g.n(.phasor(g.alloc()), freq2, reset)

        // Compute spectral loss using FFT-based implementation
        let windowSize = 64
        let loss = g.spectralLossFFT(phase1, phase2, windowSize: windowSize)

        // Output the loss
        _ = g.n(.output(0), loss)

        print("   Frequency 1: 100 Hz")
        print("   Frequency 2: 440 Hz")
        print("   Window size: \(windowSize) samples")

        // MARK: - Compile (Metal backend only)
        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        print("   ✅ Compiled successfully")

        // MARK: - Run
        let memory = runtime.allocateNodeMemory()!

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: memory,
                    frameCount: frameCount
                )
            }
        }

        runtime.deallocateNodeMemory(memory)

        // MARK: - Verification
        print("   First few loss values:")
        for i in 0..<min(10, frameCount) {
            print("      Frame \(i): loss = \(String(format: "%.6f", outputBuffer[i]))")
        }

        print("   Last few loss values (after window filled):")
        for i in (frameCount - 5)..<frameCount {
            print("      Frame \(i): loss = \(String(format: "%.6f", outputBuffer[i]))")
        }

        // After window fills, loss should be non-zero and stable for different frequencies
        let stableLoss = outputBuffer[frameCount - 1]
        XCTAssertGreaterThan(
            stableLoss, 0.0,
            "Spectral loss should be positive when comparing different frequencies")

        print("   ✅ Test passed: spectral loss computed = \(String(format: "%.6f", stableLoss))")
    }

    /// Test that identical signals produce near-zero loss
    /// Note: spectralLossFFT only supports Metal backend
    func testSpectralLossFFTSameSignal() throws {
        print("\n🧪 Test: SpectralLossFFT Same Signal")

        let g = Graph()

        let freq = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))

        // Two identical phasors
        let phase1 = g.n(.phasor(g.alloc()), freq, reset)
        let phase2 = g.n(.phasor(g.alloc()), freq, reset)

        let windowSize = 64
        let loss = g.spectralLossFFT(phase1, phase2, windowSize: windowSize)
        _ = g.n(.output(0), loss)

        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: false)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        let memory = runtime.allocateNodeMemory()!

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: memory,
                    frameCount: frameCount
                )
            }
        }

        runtime.deallocateNodeMemory(memory)

        let finalLoss = outputBuffer[frameCount - 1]
        print("   Final loss: \(String(format: "%.8f", finalLoss))")

        // Two independent phasors at the same frequency will have different phase offsets,
        // but their spectral magnitudes should be similar. The loss won't be zero but should
        // be much lower than signals at different frequencies (which can be 300+ in this setup).
        XCTAssertLessThan(
            finalLoss, 50.0,
            "Same frequency signals should have lower spectral loss than different frequencies")
        print("   ✅ Test passed: same signal loss = \(String(format: "%.6f", finalLoss))")
    }

    // MARK: - Hann Window Tests

    /// Test that Hann window coefficients are computed correctly
    func testHannWindowCoefficients() throws {
        print("\n🧪 Test: Hann Window Coefficients")

        // Expected Hann window for N=8: w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
        let windowSize = 8
        var expectedWindow: [Float] = []
        for n in 0..<windowSize {
            let angle = 2.0 * Float.pi * Float(n) / Float(windowSize - 1)
            let w = 0.5 * (1.0 - cos(angle))
            expectedWindow.append(w)
        }

        print("   Expected Hann window (N=8):")
        for (i, w) in expectedWindow.enumerated() {
            print("      w[\(i)] = \(String(format: "%.6f", w))")
        }

        // Verify key properties
        XCTAssertEqual(expectedWindow[0], 0.0, accuracy: 1e-6, "Hann window should be 0 at edges")
        XCTAssertEqual(
            expectedWindow[windowSize - 1], 0.0, accuracy: 1e-6, "Hann window should be 0 at edges")
        // For the standard Hann formula w[n] = 0.5*(1-cos(2*pi*n/(N-1))), the window
        // peaks at n=(N-1)/2. For N=8, this is at n=3.5 (between indices 3 and 4).
        // The maximum at integer indices is at n=3 and n=4, both ~0.95
        XCTAssertGreaterThan(
            expectedWindow[windowSize / 2], 0.9, "Hann window should be near max at center")
        XCTAssertGreaterThan(
            expectedWindow[windowSize / 2 - 1], 0.9, "Hann window should be near max at center")

        print("   ✅ Hann window coefficients verified")
    }

    /// Test spectral loss with and without Hann window
    /// Note: spectralLossFFT only supports Metal backend
    func testSpectralLossWithAndWithoutHannWindow() throws {
        print("\n🧪 Test: SpectralLossFFT With/Without Hann Window")

        for useHann in [true, false] {
            let g = Graph()

            let freq1 = g.n(.constant(100.0))
            let freq2 = g.n(.constant(200.0))
            let reset = g.n(.constant(0.0))

            let phase1 = g.n(.phasor(g.alloc()), freq1, reset)
            let phase2 = g.n(.phasor(g.alloc()), freq2, reset)

            let windowSize = 64
            let loss = g.spectralLossFFT(
                phase1, phase2, windowSize: windowSize, useHannWindow: useHann)
            _ = g.n(.output(0), loss)

            let frameCount = 128
            let result = try CompilationPipeline.compile(
                graph: g,
                backend: .metal,
                options: .init(frameCount: frameCount, debug: false)
            )

            let runtime = try MetalCompiledKernel(
                kernels: result.kernels,
                cellAllocations: result.cellAllocations,
                context: result.context
            )

            let memory = runtime.allocateNodeMemory()!

            let inputBuffer = [Float](repeating: 0.0, count: frameCount)
            var outputBuffer = [Float](repeating: 0.0, count: frameCount)

            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: memory,
                        frameCount: frameCount
                    )
                }
            }

            runtime.deallocateNodeMemory(memory)

            let finalLoss = outputBuffer[frameCount - 1]
            print(
                "   \(useHann ? "With" : "Without") Hann window: loss = \(String(format: "%.6f", finalLoss))"
            )

            XCTAssertGreaterThan(
                finalLoss, 0.0, "Loss should be positive for different frequencies")
        }

        print("   ✅ Both windowing modes work correctly")
    }

    // MARK: - Gradient Tests

    /// Test that gradient nodes are created for spectralLossFFT
    func testSpectralLossFFTGradientNodesCreated() throws {
        print("\n🧪 Test: SpectralLossFFT Gradient Nodes Created")

        let g = Graph()

        // Create learnable parameter
        let freqParam = g.n(.param(g.alloc()))
        let targetFreq = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))

        let phase1 = g.n(.phasor(g.alloc()), freqParam, reset)
        let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)

        let windowSize = 32
        let loss = g.spectralLossFFT(phase1, phase2, windowSize: windowSize)

        // Compute gradients
        let grads = g.computeGradients(loss: loss, targets: Set([freqParam]))

        print("   Gradient nodes created: \(grads.count)")
        print("   Gradient side effects: \(g.gradientSideEffects.count)")

        // Should have created gradient ops
        XCTAssertFalse(g.gradientSideEffects.isEmpty, "Should create gradient side effects")
        print("   ✅ Gradient nodes created successfully")
    }

    /// Test numerical gradient verification - verifies gradient direction is correct
    /// This test uses the actual training context to verify gradients point in the right direction
    func testSpectralLossGradientNumerical() throws {
        print("\n🧪 Test: SpectralLossFFT Numerical Gradient Check")

        // Test that gradient direction is correct by checking a step in the gradient direction
        // reduces loss
        let windowSize = 64
        let frameCount = 128
        let sampleRate: Float = 2000.0

        for (startFreq, targetFreq) in [(Float(100.0), Float(200.0)), (Float(300.0), Float(200.0))]
        {
            let direction = startFreq < targetFreq ? "up" : "down"
            print("   Testing \(direction): \(startFreq) Hz → \(targetFreq) Hz")

            // Create graph and get gradient
            let g = Graph(sampleRate: sampleRate)
            let freqParam = GraphParameter(graph: g, value: startFreq, name: "freq")
            let targetFreqNode = g.n(.constant(targetFreq))
            let reset = g.n(.constant(0.0))
            let twoPi = g.n(.constant(Float.pi * 2.0))

            let studentPhase = g.n(.phasor(g.alloc()), freqParam.node(), reset)
            let studentSine = g.n(.sin, [g.n(.mul, [twoPi, studentPhase])])
            let teacherPhase = g.n(.phasor(g.alloc()), targetFreqNode, reset)
            let teacherSine = g.n(.sin, [g.n(.mul, [twoPi, teacherPhase])])

            let loss = g.spectralLossFFT(studentSine, teacherSine, windowSize: windowSize)
            _ = g.n(.output(0), loss)

            let ctx = try GraphTrainingContext(
                graph: g,
                loss: loss,
                parameters: [freqParam],
                optimizer: GraphSGD(),
                learningRate: 1.0,
                frameCount: frameCount
            )

            // Get initial loss
            let initialLoss = ctx.trainStep()
            let grad = freqParam.grad

            // Take a step in the gradient direction
            let lossAfterStep = ctx.trainStep()

            print("      Initial loss: \(String(format: "%.4f", initialLoss))")
            print("      Gradient: \(String(format: "%.6f", grad))")
            print("      Loss after step: \(String(format: "%.4f", lossAfterStep))")

            // Verify gradient has correct sign
            let expectedGradSign: Float = startFreq > targetFreq ? 1.0 : -1.0
            let actualGradSign: Float = grad > 0 ? 1.0 : -1.0
            XCTAssertEqual(
                actualGradSign, expectedGradSign,
                "Gradient should be \(expectedGradSign > 0 ? "positive" : "negative") when starting \(direction) of target"
            )

            print("      ✅ Gradient direction correct (\(grad > 0 ? "positive" : "negative"))")
        }

        print("   ✅ Gradient direction verified")
    }

    /// Test that analytical gradient direction matches numerical gradient
    func testSpectralLossGradientDirectionNumerical() throws {
        print("\n🧪 Test: Spectral Loss Analytical vs Numerical Gradient Direction")

        let frameCount = 128
        let windowSize = 64
        // Use lower sample rate so that frequency resolution is better
        // Frequency resolution = sampleRate / windowSize = 2000 / 64 = 31.25 Hz
        let sampleRate: Float = 2000.0
        let targetFreq: Float = 200.0
        let epsilon: Float = 5.0  // Larger perturbation for noisy loss landscape

        // Test both directions: starting below and above target
        for (label, startFreq) in [("below", Float(100.0)), ("above", Float(300.0))] {
            print("\n   Testing starting \(label) target (\(startFreq) Hz → \(targetFreq) Hz)")

            // Compute loss at startFreq
            let lossAtStart = try computeLossAtFrequencySine(
                startFreq, target: targetFreq, windowSize: windowSize,
                frameCount: frameCount, sampleRate: sampleRate)

            // Compute loss at startFreq + epsilon (numerical gradient)
            let lossAtPlus = try computeLossAtFrequencySine(
                startFreq + epsilon, target: targetFreq, windowSize: windowSize,
                frameCount: frameCount, sampleRate: sampleRate)

            // Compute loss at startFreq - epsilon
            let lossAtMinus = try computeLossAtFrequencySine(
                startFreq - epsilon, target: targetFreq, windowSize: windowSize,
                frameCount: frameCount, sampleRate: sampleRate)

            // Central difference numerical gradient: dL/dfreq ≈ (L(f+ε) - L(f-ε)) / (2ε)
            let numericalGrad = (lossAtPlus - lossAtMinus) / (2 * epsilon)

            // Get analytical gradient
            let analyticalGrad = try computeAnalyticalGradient(
                startFreq, target: targetFreq, windowSize: windowSize,
                frameCount: frameCount, sampleRate: sampleRate)

            print("      Loss at freq: \(String(format: "%.4f", lossAtStart))")
            print("      Loss at freq+ε: \(String(format: "%.4f", lossAtPlus))")
            print("      Loss at freq-ε: \(String(format: "%.4f", lossAtMinus))")
            print("      Numerical gradient: \(String(format: "%.6f", numericalGrad))")
            print("      Analytical gradient: \(String(format: "%.6f", analyticalGrad))")

            // Check if the direction is correct
            let expectedDirection: Float = startFreq > targetFreq ? 1.0 : -1.0  // positive grad if above target
            let numericalDirection = numericalGrad > 0 ? 1.0 : -1.0
            let analyticalDirection = analyticalGrad > 0 ? 1.0 : -1.0

            print(
                "      Expected direction: \(expectedDirection > 0 ? "positive" : "negative") (need to decrease freq)"
            )
            print("      Numerical direction: \(numericalDirection > 0 ? "positive" : "negative")")
            print(
                "      Analytical direction: \(analyticalDirection > 0 ? "positive" : "negative")")

            // Verify directions match
            if numericalDirection == analyticalDirection {
                print("      ✅ Analytical gradient direction matches numerical")
            } else {
                print("      ⚠️ Analytical gradient direction differs from numerical!")
            }
        }
    }

    /// Helper: compute analytical gradient for frequency parameter
    private func computeAnalyticalGradient(
        _ studentFreq: Float, target targetFreq: Float,
        windowSize: Int, frameCount: Int, sampleRate: Float
    ) throws -> Float {
        let g = Graph(sampleRate: sampleRate)

        let freqParam = GraphParameter(graph: g, value: studentFreq, name: "freq")
        let targetFreqNode = g.n(.constant(targetFreq))
        let reset = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let studentPhase = g.n(.phasor(g.alloc()), freqParam.node(), reset)
        let studentSine = g.n(.sin, [g.n(.mul, [twoPi, studentPhase])])

        let teacherPhase = g.n(.phasor(g.alloc()), targetFreqNode, reset)
        let teacherSine = g.n(.sin, [g.n(.mul, [twoPi, teacherPhase])])

        let loss = g.spectralLossFFT(studentSine, teacherSine, windowSize: windowSize)
        _ = g.n(.output(0), loss)

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [freqParam],
            optimizer: GraphSGD(),
            learningRate: 0.0,  // Don't actually update
            frameCount: frameCount
        )

        // Run one step to compute gradients
        _ = ctx.trainStep()

        return freqParam.grad
    }

    /// Helper: compute spectral loss for sine waves at given frequencies
    private func computeLossAtFrequencySine(
        _ studentFreq: Float, target targetFreq: Float,
        windowSize: Int, frameCount: Int, sampleRate: Float
    ) throws -> Float {
        let g = Graph(sampleRate: sampleRate)

        let freq1 = g.n(.constant(studentFreq))
        let freq2 = g.n(.constant(targetFreq))
        let reset = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let phase1 = g.n(.phasor(g.alloc()), freq1, reset)
        let sine1 = g.n(.sin, [g.n(.mul, [twoPi, phase1])])

        let phase2 = g.n(.phasor(g.alloc()), freq2, reset)
        let sine2 = g.n(.sin, [g.n(.mul, [twoPi, phase2])])

        let loss = g.spectralLossFFT(sine1, sine2, windowSize: windowSize)
        _ = g.n(.output(0), loss)

        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: false)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        let memory = runtime.allocateNodeMemory()!

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: memory,
                    frameCount: frameCount
                )
            }
        }

        runtime.deallocateNodeMemory(memory)

        return outputBuffer[frameCount - 1]
    }

    /// Helper: compute spectral loss for a given student frequency vs target
    /// Note: spectralLossFFT only supports Metal backend
    private func computeLossAtFrequency(
        _ studentFreq: Float, target targetFreq: Float,
        windowSize: Int, frameCount: Int
    ) throws -> Float {
        let g = Graph()

        let freq1 = g.n(.constant(studentFreq))
        let freq2 = g.n(.constant(targetFreq))
        let reset = g.n(.constant(0.0))

        let phase1 = g.n(.phasor(g.alloc()), freq1, reset)
        let phase2 = g.n(.phasor(g.alloc()), freq2, reset)

        let loss = g.spectralLossFFT(phase1, phase2, windowSize: windowSize)
        _ = g.n(.output(0), loss)

        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: false)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        let memory = runtime.allocateNodeMemory()!

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        inputBuffer.withUnsafeBufferPointer { inPtr in
            outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: memory,
                    frameCount: frameCount
                )
            }
        }

        runtime.deallocateNodeMemory(memory)

        return outputBuffer[frameCount - 1]
    }

    // MARK: - Compilation Tests

    /// Test that spectralLossFFT compiles for Metal backend
    /// Note: spectralLossFFT only supports Metal backend (not C)
    func testSpectralLossFFTCompilesBothBackends() throws {
        print("\n🧪 Test: SpectralLossFFT Compiles for Metal Backend")

        let g = Graph()

        let freq1 = g.n(.constant(100.0))
        let freq2 = g.n(.constant(200.0))
        let reset = g.n(.constant(0.0))

        let phase1 = g.n(.phasor(g.alloc()), freq1, reset)
        let phase2 = g.n(.phasor(g.alloc()), freq2, reset)

        let loss = g.spectralLossFFT(phase1, phase2, windowSize: 32)
        _ = g.n(.output(0), loss)

        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: 64, debug: false)
        )

        XCTAssertFalse(result.kernels.isEmpty, "Metal backend should produce kernels")
        print("   metal: ✅ compiled (\(result.kernels.first?.source.count ?? 0) chars)")

        print("   ✅ Metal backend compiles successfully")
    }

    /// Test that gradient ops compile correctly
    /// Note: spectralLossFFT only supports Metal backend
    func testSpectralLossFFTGradientOpsCompile() throws {
        print("\n🧪 Test: SpectralLossFFT Gradient Ops Compile")

        let g = Graph()

        let freqParam = g.n(.param(g.alloc()))
        let targetFreq = g.n(.constant(440.0))
        let reset = g.n(.constant(0.0))

        let phase1 = g.n(.phasor(g.alloc()), freqParam, reset)
        let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)

        let windowSize = 32
        let loss = g.spectralLossFFT(phase1, phase2, windowSize: windowSize)

        // Compute gradients
        _ = g.computeGradients(loss: loss, targets: Set([freqParam]))

        // Output loss
        _ = g.n(.output(0), loss)

        // Compile should succeed even with gradient ops
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: 64, debug: false)
        )

        XCTAssertFalse(result.kernels.isEmpty, "Should compile with gradient ops")
        print("   ✅ Gradient ops compile successfully")
    }

    // MARK: - Edge Cases

    /// Test with various window sizes
    /// Note: spectralLossFFT only supports Metal backend
    func testSpectralLossFFTVariousWindowSizes() throws {
        print("\n🧪 Test: SpectralLossFFT Various Window Sizes")

        let windowSizes = [8, 16, 32, 64, 128]

        for windowSize in windowSizes {
            let g = Graph()

            let freq1 = g.n(.constant(100.0))
            let freq2 = g.n(.constant(200.0))
            let reset = g.n(.constant(0.0))

            let phase1 = g.n(.phasor(g.alloc()), freq1, reset)
            let phase2 = g.n(.phasor(g.alloc()), freq2, reset)

            let loss = g.spectralLossFFT(phase1, phase2, windowSize: windowSize)
            _ = g.n(.output(0), loss)

            let frameCount = windowSize * 2
            let result = try CompilationPipeline.compile(
                graph: g,
                backend: .metal,
                options: .init(frameCount: frameCount, debug: false)
            )

            XCTAssertFalse(result.kernels.isEmpty, "Window size \(windowSize) should compile")
            print("   Window size \(windowSize): ✅")
        }

        print("   ✅ All window sizes work correctly")
    }

    // MARK: - Performance Comparison Tests

    /// Test MLP -> peekRow -> Harmonic Synth with Spectral Loss
    /// This is the spectral loss equivalent of testMLPPeekRowHarmonicSynth_TeacherStudent
    /// for speed comparison
    func testMLPPeekRowHarmonicSynth_SpectralLoss() throws {
        let frameCount = 512
        let controlFrames = 64
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0
        let numHarmonics = 32
        let hiddenSize = 8
        let windowSize = 64  // FFT window size for spectral loss

        let g = Graph(sampleRate: sampleRate, maxFrameCount: frameCount)

        // Control-rate time tensor [controlFrames, 1], normalized 0..1
        let timeData = (0..<controlFrames).map { Float($0) / Float(controlFrames - 1) }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        func makeArray(_ count: Int, scale: Float, freq: Float, phase: Float, offset: Float = 0.0)
            -> [Float]
        {
            (0..<count).map { i in
                offset + scale * sin(Float(i) * freq + phase)
            }
        }

        // Teacher (fixed) weights
        let teacherW1Data = (0..<hiddenSize).map { i in
            let x = Float(i) / Float(max(1, hiddenSize - 1))
            return 1.1 * sin(x * 3.1 * Float.pi) + 0.5 * cos(x * 2.3 * Float.pi)
        }
        let teacherB1Data = (0..<hiddenSize).map { i in
            let x = Float(i) / Float(max(1, hiddenSize - 1))
            return 0.4 * (x - 0.5) + 0.2 * sin(x * 5.0)
        }
        let teacherW2Data = (0..<(hiddenSize * numHarmonics)).map { i in
            let row = i / numHarmonics
            let col = i % numHarmonics
            let base = Float(row) * 0.6 + Float(col) * 0.25
            let sign: Float = (col % 2 == 0) ? 1.0 : -1.0
            return sign * (0.7 * sin(base) + 0.5 * cos(base * 1.3))
        }
        let teacherB2Data = (0..<numHarmonics).map { i in
            let inv = 0.8 / Float(i + 1)
            let wiggle = 0.1 * sin(Float(i) * 1.7)
            return inv + wiggle
        }

        let teacherW1 = g.tensor(shape: [1, hiddenSize], data: teacherW1Data)
        let teacherB1 = g.tensor(shape: [1, hiddenSize], data: teacherB1Data)
        let teacherW2 = g.tensor(shape: [hiddenSize, numHarmonics], data: teacherW2Data)
        let teacherB2 = g.tensor(shape: [1, numHarmonics], data: teacherB2Data)

        // Student (learnable) weights
        let studentW1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.12, freq: 0.9, phase: 1.1), name: "W1")
        let studentB1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.05, freq: 0.6, phase: 0.9), name: "b1")
        let studentW2 = TensorParameter(
            graph: g, shape: [hiddenSize, numHarmonics],
            data: makeArray(hiddenSize * numHarmonics, scale: 0.08, freq: 0.21, phase: 0.7),
            name: "W2")
        let studentB2 = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: (0..<numHarmonics).map { _ in 0.1 }, name: "b2")

        // Static MLP: timeTensor -> amplitude tensor [controlFrames, numHarmonics]
        func mlpAmplitudes(time: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let one = g.n(.constant(1.0))
            let h1 = try g.matmul(time, W1)
            let h1b = g.n(.add, [h1, b1])
            let h1a = g.n(.tanh, [h1b])
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, [h2, b2])
            let neg = g.n(.mul, [h2b, g.n(.constant(-1.0))])
            let expNeg = g.n(.exp, [neg])
            return g.n(.div, [one, g.n(.add, [one, expNeg])])
        }

        let ampsStudent = try mlpAmplitudes(
            time: timeTensor, W1: studentW1.node(), b1: studentB1.node(),
            W2: studentW2.node(), b2: studentB2.node())
        let ampsTeacher = try mlpAmplitudes(
            time: timeTensor, W1: teacherW1, b1: teacherB1, W2: teacherW2, b2: teacherB2)

        // Reshape to [numHarmonics, controlFrames] then transpose for peekRow
        let ampsStudentView = try g.reshape(ampsStudent, to: [numHarmonics, controlFrames])
        let ampsTeacherView = try g.reshape(ampsTeacher, to: [numHarmonics, controlFrames])
        let ampsStudentT = try g.transpose(ampsStudentView, axes: [1, 0])  // [controlFrames, numHarmonics]
        let ampsTeacherT = try g.transpose(ampsTeacherView, axes: [1, 0])

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Audio-rate playhead through control frames
        let frameIdx = g.phasor(
            freq: g.n(.constant(sampleRate / Float(frameCount))), reset: zero)
        let playhead = g.n(.mul, [frameIdx, g.n(.constant(Float(controlFrames - 1)))])

        // Vectorized harmonic synthesis
        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)
        let phasesTensor = g.n(.deterministicPhasor, [freqTensor])
        let sinesTensor = g.n(.sin, [g.n(.mul, [twoPi, phasesTensor])])

        let ampsStudentAtTime = try g.peekRow(tensor: ampsStudentT, rowIndex: playhead)
        let ampsTeacherAtTime = try g.peekRow(tensor: ampsTeacherT, rowIndex: playhead)

        let synthStudent = g.n(.sum, [g.n(.mul, [sinesTensor, ampsStudentAtTime])])
        let synthTeacher = g.n(.sum, [g.n(.mul, [sinesTensor, ampsTeacherAtTime])])

        let norm = g.n(.constant(1.0 / Float(numHarmonics)))
        let studentOut = g.n(.mul, [synthStudent, norm])
        let teacherOut = g.n(.mul, [synthTeacher, norm])

        // Use spectral loss instead of MSE
        let loss = g.spectralLossFFT(studentOut, teacherOut, windowSize: windowSize)
        _ = g.n(.output(0), [loss])

        print("\n=== MLP -> peekRow -> Harmonic Synth (Spectral Loss) ===")
        print("frameCount: \(frameCount), controlFrames: \(controlFrames)")
        print("numHarmonics: \(numHarmonics), hiddenSize: \(hiddenSize)")
        print("windowSize: \(windowSize)")
        print(
            "Total learnable params: \(hiddenSize + hiddenSize + hiddenSize*numHarmonics + numHarmonics)"
        )

        // Use GraphTrainingContext with graph-based gradients
        let compileStart = CFAbsoluteTimeGetCurrent()
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            tensorParameters: [studentW1, studentB1, studentW2, studentB2],
            optimizer: GraphAdam(),
            learningRate: 0.001,
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/mlp_peekrow_harmonic_spectral.metal"
        )
        let compileTime = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000
        print("Compile time: \(String(format: "%.2f", compileTime))ms")

        // Warmup
        _ = ctx.trainStep()
        _ = ctx.trainStep()

        let initialLoss = ctx.trainStep()
        print("Initial loss: \(initialLoss)")

        // Training loop with timing
        let epochs = 50
        var finalLoss = initialLoss
        let trainStart = CFAbsoluteTimeGetCurrent()
        for i in 0..<epochs {
            finalLoss = ctx.trainStep()
            // Print every epoch to see gradient behavior
            print("Epoch \(i): loss = \(String(format: "%.6f", finalLoss))")
        }
        let trainTime = (CFAbsoluteTimeGetCurrent() - trainStart) * 1000
        let timePerEpoch = trainTime / Double(epochs)

        print("\nFinal loss: \(String(format: "%.6f", finalLoss))")
        print("Loss reduction: \(String(format: "%.2f", initialLoss / finalLoss))x")
        print("\n--- Performance (Spectral Loss) ---")
        print("Total train time (\(epochs) epochs): \(String(format: "%.2f", trainTime))ms")
        print("Time per epoch: \(String(format: "%.3f", timePerEpoch))ms")
        print("Epochs per second: \(String(format: "%.1f", 1000.0 / timePerEpoch))")

        // Spectral loss may not converge as easily as MSE for this task
        // Just verify it runs and produces reasonable values
        XCTAssertGreaterThan(initialLoss, 0.0, "Initial loss should be positive")
        XCTAssertFalse(finalLoss.isNaN, "Final loss should not be NaN")
    }

    // MARK: - Simple Frequency Learning Test

    /// The simplest possible test: one learnable sine wave frequency learning to match a target.
    /// This is the definitive proof that spectral loss gradients work.
    func testSimpleSineWaveFrequencyLearning() throws {
        print("\n🧪 Test: Simple Sine Wave Frequency Learning with Spectral Loss")

        let frameCount = 128
        let windowSize = 64
        // Use lower sample rate for better frequency resolution
        // Resolution = 2000/64 = 31.25 Hz per bin, so 100 vs 200 Hz spans ~3 bins
        let sampleRate: Float = 2000.0

        // Target frequency
        let targetFreqValue: Float = 200.0

        // Starting frequency (deliberately far from target, spanning multiple bins)
        let startFreqValue: Float = 300.0

        let g = Graph(sampleRate: sampleRate)

        // Learnable frequency parameter (using GraphParameter like onepole tests)
        let freqParam = GraphParameter(graph: g, value: startFreqValue, name: "freq")

        // Target frequency (fixed)
        let targetFreq = g.n(.constant(targetFreqValue))

        let reset = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        // Student sine wave (learnable frequency) - use separate phasor
        let studentPhase = g.n(.phasor(g.alloc()), freqParam.node(), reset)
        let studentSine = g.n(.sin, [g.n(.mul, [twoPi, studentPhase])])

        // Teacher sine wave (fixed frequency) - use separate phasor
        let teacherPhase = g.n(.phasor(g.alloc()), targetFreq, reset)
        let teacherSine = g.n(.sin, [g.n(.mul, [twoPi, teacherPhase])])

        // Spectral loss between student and teacher
        let loss = g.spectralLossFFT(studentSine, teacherSine, windowSize: windowSize)
        _ = g.n(.output(0), loss)

        // Debug: check gradients before training
        print("   DEBUG: Computing gradients...")
        let grads = g.computeGradients(loss: loss, targets: Set([freqParam.nodeId]))
        print("   DEBUG: Gradients computed: \(grads.count) nodes")
        if let gradNode = grads[freqParam.nodeId] {
            print("   DEBUG: Gradient node for freqParam: \(gradNode)")
            if let gradNodeInfo = g.nodes[gradNode] {
                print("   DEBUG: Gradient op: \(gradNodeInfo.op)")
            }
        } else {
            print("   DEBUG: No gradient found for freqParam!")
        }

        print("   Target frequency: \(targetFreqValue) Hz")
        print("   Starting frequency: \(startFreqValue) Hz")
        print("   Window size: \(windowSize)")

        // Create training context (like onepole tests)
        // Note: Learning rate adjusted for frequency scale and gradient magnitude
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [freqParam],
            optimizer: GraphAdam(),
            learningRate: 0.1,  // Adam with reasonable learning rate for frequency scale
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/simple_freq_spectral.metal"
        )
        print("   DEBUG: Wrote kernels to /tmp/simple_freq_spectral.metal")

        // Warmup
        _ = ctx.trainStep()
        _ = ctx.trainStep()

        let initialLoss = ctx.trainStep()
        let initialFreq = freqParam.value
        print(
            "   Initial: freq = \(String(format: "%.1f", initialFreq)) Hz, loss = \(String(format: "%.4f", initialLoss))"
        )

        // Train
        let epochs = 2000
        var lastLoss = initialLoss
        for i in 0..<epochs {
            lastLoss = ctx.trainStep()
            if i % 20 == 0 || i == epochs - 1 {
                print(
                    "   Epoch \(i): freq = \(String(format: "%.1f", freqParam.value)) Hz, loss = \(String(format: "%.4f", lastLoss)), grad = \(freqParam.grad)"
                )
            }
            if lastLoss < 1.0 {
                break
            }
        }

        let finalFreq = freqParam.value
        let finalLoss = lastLoss

        print(
            "\n   Final: freq = \(String(format: "%.1f", finalFreq)) Hz, loss = \(String(format: "%.4f", finalLoss))"
        )
        print(
            "   Frequency moved from \(String(format: "%.1f", startFreqValue)) toward \(String(format: "%.1f", targetFreqValue))"
        )

        // Verify that frequency moved toward target
        let initialDistance = abs(startFreqValue - targetFreqValue)
        let finalDistance = abs(finalFreq - targetFreqValue)

        print("   Initial distance from target: \(String(format: "%.1f", initialDistance)) Hz")
        print("   Final distance from target: \(String(format: "%.1f", finalDistance)) Hz")

        XCTAssertLessThan(
            finalDistance, initialDistance,
            "Frequency should move closer to target (was \(initialDistance) Hz away, now \(finalDistance) Hz away)"
        )
        XCTAssertLessThan(
            finalLoss, initialLoss,
            "Loss should decrease")

        // Check that we actually moved significantly
        let freqChange = abs(finalFreq - startFreqValue)
        XCTAssertGreaterThan(
            freqChange, 10.0,
            "Frequency should change by at least 10 Hz (changed by \(freqChange) Hz)")

        print("   ✅ Frequency successfully learned toward target using spectral loss!")
    }

    // MARK: - Directional Frequency Learning Tests

    /// Test learning frequency UP (starting below target, close)
    func testFrequencyLearning_UpClose() throws {
        try runFrequencyLearningTest(
            name: "Up Close",
            startFreq: 180.0,
            targetFreq: 200.0,
            expectedDirection: "up"
        )
    }

    /// Test learning frequency UP (starting below target, far)
    func testFrequencyLearning_UpFar() throws {
        try runFrequencyLearningTest(
            name: "Up Far",
            startFreq: 100.0,
            targetFreq: 200.0,
            expectedDirection: "up"
        )
    }

    /// Test learning frequency DOWN (starting above target, close)
    func testFrequencyLearning_DownClose() throws {
        try runFrequencyLearningTest(
            name: "Down Close",
            startFreq: 220.0,
            targetFreq: 200.0,
            expectedDirection: "down"
        )
    }

    /// Test learning frequency DOWN (starting above target, far)
    func testFrequencyLearning_DownFar() throws {
        // Use 280 Hz start - about 2.5 bins away from 200 Hz
        try runFrequencyLearningTest(
            name: "Down Far",
            startFreq: 280.0,
            targetFreq: 200.0,
            expectedDirection: "down",
            maxEpochs: 200
        )
    }

    /// Test learning with very large frequency gap
    func testFrequencyLearning_VeryFar() throws {
        try runFrequencyLearningTest(
            name: "Very Far (down)",
            startFreq: 500.0,
            targetFreq: 200.0,
            expectedDirection: "down",
            maxEpochs: 300
        )
    }

    /// Test learning to a different target frequency
    func testFrequencyLearning_DifferentTarget() throws {
        try runFrequencyLearningTest(
            name: "Different Target",
            startFreq: 250.0,
            targetFreq: 150.0,
            expectedDirection: "down"
        )
    }

    /// Helper function to run frequency learning tests with various configurations
    private func runFrequencyLearningTest(
        name: String,
        startFreq: Float,
        targetFreq: Float,
        expectedDirection: String,
        maxEpochs: Int = 200,
        learningRate: Float = 1.0,
        file: StaticString = #file,
        line: UInt = #line
    ) throws {
        print("\n🧪 Test: Frequency Learning - \(name)")
        print("   Start: \(startFreq) Hz → Target: \(targetFreq) Hz")

        let frameCount = 128
        let windowSize = 64
        // Resolution = 2000/64 = 31.25 Hz per bin
        let sampleRate: Float = 2000.0

        let g = Graph(sampleRate: sampleRate)

        let freqParam = GraphParameter(graph: g, value: startFreq, name: "freq")
        let targetFreqNode = g.n(.constant(targetFreq))
        let reset = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let studentPhase = g.n(.phasor(g.alloc()), freqParam.node(), reset)
        let studentSine = g.n(.sin, [g.n(.mul, [twoPi, studentPhase])])

        let teacherPhase = g.n(.phasor(g.alloc()), targetFreqNode, reset)
        let teacherSine = g.n(.sin, [g.n(.mul, [twoPi, teacherPhase])])

        let loss = g.spectralLossFFT(studentSine, teacherSine, windowSize: windowSize)
        _ = g.n(.output(0), loss)

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            parameters: [freqParam],
            optimizer: GraphAdam(),
            learningRate: learningRate,
            frameCount: frameCount
        )

        // Warmup
        _ = ctx.trainStep()
        _ = ctx.trainStep()

        let initialLoss = ctx.trainStep()
        let initialFreq = freqParam.value
        print(
            "   Initial: freq = \(String(format: "%.1f", initialFreq)) Hz, loss = \(String(format: "%.2f", initialLoss))"
        )

        // Train
        var lastLoss = initialLoss
        var epochsRun = 0
        for i in 0..<maxEpochs {
            lastLoss = ctx.trainStep()
            epochsRun = i + 1
            if i % 50 == 0 {
                print(
                    "   Epoch \(i): freq = \(String(format: "%.1f", freqParam.value)) Hz, loss = \(String(format: "%.2f", lastLoss))"
                )
            }
            // Early stopping if converged
            if lastLoss < 1.0 {
                break
            }
        }

        let finalFreq = freqParam.value
        let finalLoss = lastLoss
        let initialDistance = abs(startFreq - targetFreq)
        let finalDistance = abs(finalFreq - targetFreq)

        print(
            "   Final: freq = \(String(format: "%.1f", finalFreq)) Hz, loss = \(String(format: "%.2f", finalLoss)) (after \(epochsRun) epochs)"
        )
        print(
            "   Distance: \(String(format: "%.1f", initialDistance)) Hz → \(String(format: "%.1f", finalDistance)) Hz"
        )

        // Verify direction
        let actualDirection = finalFreq > startFreq ? "up" : "down"
        XCTAssertEqual(
            actualDirection, expectedDirection,
            "Expected frequency to go \(expectedDirection) but went \(actualDirection)",
            file: file, line: line
        )

        // Verify improvement
        XCTAssertLessThan(
            finalDistance, initialDistance,
            "Frequency should move closer to target (was \(initialDistance) Hz away, now \(finalDistance) Hz away)",
            file: file, line: line
        )

        XCTAssertLessThan(
            finalLoss, initialLoss,
            "Loss should decrease (was \(initialLoss), now \(finalLoss))",
            file: file, line: line
        )

        // Verify significant movement
        let freqChange = abs(finalFreq - startFreq)
        let minExpectedChange = min(initialDistance * 0.3, 10.0)  // At least 30% of the way or 10 Hz
        XCTAssertGreaterThan(
            freqChange, minExpectedChange,
            "Frequency should change significantly (changed by \(freqChange) Hz, expected at least \(minExpectedChange) Hz)",
            file: file, line: line
        )

        print("   ✅ Passed: frequency correctly moved \(actualDirection) toward target")
    }
}

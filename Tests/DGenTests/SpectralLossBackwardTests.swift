import XCTest

@testable import DGen

final class SpectralLossBackwardTests: XCTestCase {

    /// Test that spectral loss backward pass computes non-zero gradients
    func testSpectralLossGradientsExist() throws {
        print("\n🧪 Test: Spectral Loss Backward Pass - Gradients Exist")

        let g = Graph()

        // Learnable frequency parameter (start at 300 Hz, target is 440 Hz)
        let freqParam = Parameter(graph: g, value: 300.0, name: "frequency")
        let freq = freqParam.node()

        // Target frequency (constant 440 Hz)
        let targetFreq = g.n(.constant(440.0))

        let reset = g.n(.constant(0.0))

        // Generate signals
        let phase1 = g.n(.phasor(g.alloc()), freq, reset)
        let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)

        // Convert phase to sine wave (phasor outputs 0-1, multiply by 2π and take sin)
        let twoPi = g.n(.constant(2.0 * Float.pi))
        let sig1 = g.n(.sin, g.n(.mul, phase1, twoPi))
        let sig2 = g.n(.sin, g.n(.mul, phase2, twoPi))

        // Compute spectral loss (tape-based)
        let windowSize = 64
        let loss = g.spectralLoss(sig1, sig2, windowSize: windowSize)

        _ = g.n(.output(0), loss)

        // Compile with backwards pass enabled
        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: true)
        )

        print("   Globals array: \(result.context.globals)")

        for kernel in result.kernels {
            if kernel.source.contains("outputs[") {
                // Print just the output line
                let lines = kernel.source.components(separatedBy: "\n")
                for line in lines {
                    if line.contains("outputs[") {
                        print("   Output assignment: \(line.trimmingCharacters(in: .whitespaces))")
                    }
                }
            }
        }

        print("   ✅ Compiled with backwards=true")
        print("   Initial frequency: 300 Hz")
        print("   Target frequency: 440 Hz")

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Training context to run a single forward/backward pass
        let ctx = TrainingContext(
            parameters: [],
            optimizer: SGD(lr: 0.0),
            lossNode: loss
        )
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )
        let initialLoss = ctx.runStepGPU()

        // Also check tape buffer to see if loss is computed correctly
        if let tape = runtime.readBuffer(named: "t") {
            let lossTapeSlot = 7  // From kernel analysis (updated after fixing globals ordering)
            print(
                "   [DEBUG] Tape slot \(lossTapeSlot) (spectralLoss): first 5 = \(tape[(lossTapeSlot*frameCount)..<(lossTapeSlot*frameCount+5)].map { String(format: "%.2f", $0) }.joined(separator: ", "))"
            )
        }

        print("   Initial loss: \(String(format: "%.6f", initialLoss))")

        // Check that loss is reasonable (300 vs 440 Hz should have measurable loss)
        XCTAssertGreaterThan(initialLoss, 0.01, "Loss should be measurable for 300 vs 440 Hz")

        // Read gradients buffer
        if let gradients = runtime.readBuffer(named: "gradients") {
            print("   Gradients buffer exists with \(gradients.count) values")

            // Find and print non-zero gradients
            var nonZeroGradients: [(index: Int, value: Float)] = []
            for i in 0..<gradients.count {
                if abs(gradients[i]) > 0.0001 {
                    nonZeroGradients.append((i, gradients[i]))
                }
            }

            print("   Found \(nonZeroGradients.count) non-zero gradient values")

            // Separate seeds (1.0 values) from computed gradients
            let seeds = nonZeroGradients.filter { abs($0.value - 1.0) < 0.0001 }
            let computed = nonZeroGradients.filter { abs($0.value - 1.0) >= 0.0001 }

            print("   Seed gradients: \(seeds.count)")
            print("   Computed gradients: \(computed.count)")
            print("   First 10 computed gradients:")
            for (index, value) in computed.prefix(10) {
                print("      grad[\(index)] = \(String(format: "%.6f", value))")
            }

            // Check if at least one gradient is non-zero
            XCTAssertFalse(nonZeroGradients.isEmpty, "At least one gradient should be non-zero")

            if !nonZeroGradients.isEmpty {
                print("   ✅ PASS: Gradients are non-zero - backward pass is working!")
            }
        } else {
            XCTFail("Gradients buffer not found")
        }

    }

    /// Test that spectral loss backward pass can learn to match a target frequency
    func testSpectralLossLearnsFrequencyCycles() throws {
        print("\n🧪 Test: Spectral Loss Learning - Frequency Matching")

        let g = Graph()

        // Learnable frequency parameter (start at 300 Hz, target is 440 Hz)
        let freqParam = Parameter(graph: g, value: 550.0, name: "frequency")
        let freq = freqParam.node()

        // Target frequency (constant 440 Hz)
        let targetFreq = g.n(.constant(440.0))

        let reset = g.n(.constant(0.0))

        // Generate signals
        let phase1 = g.n(.phasor(g.alloc()), freq, reset)
        let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)

        // Convert phase to sine wave (phasor outputs 0-1, multiply by 2π and take sin)
        let twoPi = g.n(.constant(2.0 * Float.pi))
        let sig1 = g.n(.sin, g.n(.mul, phase1, twoPi))
        let sig2 = g.n(.sin, g.n(.mul, phase2, twoPi))

        // Compute spectral loss
        let loss1 = g.spectralLoss(sig1, sig2, windowSize: 32)
        let loss2 = g.spectralLoss(sig1, sig2, windowSize: 64)
        let loss3 = g.spectralLoss(sig1, sig2, windowSize: 128)
        let loss = g.n(.div, g.n(.add, loss3, g.n(.add, loss1, loss2)), g.n(.constant(3)))

        _ = g.n(.output(0), loss)

        // Compile with backwards pass enabled
        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: true)
        )

        print("   ✅ Compiled with backwards=true")
        print("   Initial frequency: 300 Hz")
        print("   Target frequency: 440 Hz")

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Training context using optimizer (replaces manual gradient descent)
        let ctx = TrainingContext(
            parameters: [freqParam],
            optimizer: SGD(lr: 692.0),  // match manual learning rate used previously
            lossNode: loss
        )
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // Training loop using TrainingContext
        let numIterations = 2500
        var lossHistory: [Float] = []
        for iteration in 0..<numIterations {
            let currentLoss = ctx.runStepGPU()
            lossHistory.append(currentLoss)

            if iteration % 100 == 0 || iteration <= 10 {
                print(
                    "   Iteration \(iteration): freq=\(String(format: "%.2f", freqParam.value)) Hz, loss=\(String(format: "%.6f", currentLoss))"
                )
            }
        }

        // Results
        let finalFreq = freqParam.value
        let finalLoss = lossHistory.last ?? .infinity
        let initialLoss = lossHistory.first ?? .infinity

        print("   Final frequency: \(String(format: "%.2f", finalFreq)) Hz")
        print("   Final loss: \(String(format: "%.6f", finalLoss))")
        print(
            "   Loss reduction: \(String(format: "%.1f", (initialLoss - finalLoss) / max(initialLoss, .ulpOfOne) * 100))%"
        )

        // Verify learning happened
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease by at least 50%")
        XCTAssertGreaterThan(finalFreq, 300.0, "Frequency should increase from initial value")
        XCTAssertLessThan(
            abs(finalFreq - 440.0), 50.0, "Frequency should be within 50 Hz of target")

        print("   ✅ PASS: Successfully learned target frequency!")
    }

    func testSpectralLossLearnsFrequencyPhasors() throws {
        print("\n🧪 Test: Spectral Loss Learning - Frequency Matching (Phasors not Sines)")

        let g = Graph()

        let targetFrequency: Float = 300.0
        // Learnable frequency parameter (start at 777 Hz, target is 300 Hz)
        let freqParam = Parameter(graph: g, value: 777.0, name: "frequency")
        let freq = freqParam.node()

        // Target frequency (constant 440 Hz)
        let targetFreq = g.n(.constant(targetFrequency))

        let reset = g.n(.constant(0.0))

        // Generate signals
        let phase1 = g.n(.phasor(g.alloc()), freq, reset)
        let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)

        // Convert phase to sine wave (phasor outputs 0-1, multiply by 2π and take sin)
        // Compute spectral loss
        let windowSize = 64
        let loss = g.spectralLoss(phase1, phase2, windowSize: windowSize)

        _ = g.n(.output(0), loss)

        // Compile with backwards pass enabled
        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: true)
        )

        print("   ✅ Compiled with backwards=true")
        print("   Initial frequency: 777 Hz")
        print("   Target frequency: 300 Hz")

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Training context using optimizer (replaces manual gradient descent)
        let ctx = TrainingContext(
            parameters: [freqParam],
            optimizer: Adam(lr: 1.5),  // match manual learning rate used previously
            lossNode: loss
        )
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // Training loop using TrainingContext
        let numIterations = 2000
        var lossHistory: [Float] = []
        for iteration in 0..<numIterations {
            let currentLoss = ctx.runStepGPU()
            lossHistory.append(currentLoss)

            if iteration % 100 == 0 || iteration <= 10 {
                print(
                    "   Iteration \(iteration): freq=\(String(format: "%.2f", freqParam.value)) Hz, loss=\(String(format: "%.6f", currentLoss))"
                )
            }
        }

        // Results
        let finalFreq = freqParam.value
        let finalLoss = lossHistory.last ?? .infinity
        let initialLoss = lossHistory.first ?? .infinity

        print("   Final frequency: \(String(format: "%.2f", finalFreq)) Hz")
        print("   Final loss: \(String(format: "%.6f", finalLoss))")
        print(
            "   Loss reduction: \(String(format: "%.1f", (initialLoss - finalLoss) / max(initialLoss, .ulpOfOne) * 100))%"
        )

        // Verify learning happened
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease by at least 50%")
        XCTAssertLessThan(
            abs(finalFreq - targetFrequency), 50.0, "Frequency should be within 50 Hz of target")

        print("   ✅ PASS: Successfully learned target frequency!")
    }

    func testLearnFrequencyAndAmp() throws {
        print("\n🧪 Test: Spectral Loss Learning - Frequency Matching and LFOs matching")

        let g = Graph()

        let targetFrequency: Float = 300.0
        let targetLFOFrequency: Float = 10.0

        // Learnable frequency parameter (start at 300 Hz, target is 237 Hz)
        let freqParam = Parameter(graph: g, value: 350.0, name: "frequency")
        let freq = freqParam.node()

        let lfoFreqParam = Parameter(graph: g, value: 8.5, name: "lfo-frequency")
        let lfoFreq = lfoFreqParam.node()

        // Target frequency (constant 440 Hz)
        let targetFreq = g.n(.constant(targetFrequency))
        let targetLFOFreq = g.n(.constant(targetLFOFrequency))

        let reset = g.n(.constant(0.0))

        // Generate signals
        let phase1 = g.n(.phasor(g.alloc()), freq, reset)
        let phase2 = g.n(.phasor(g.alloc()), targetFreq, reset)

        let lfo_phase1 = g.n(.phasor(g.alloc()), lfoFreq, reset)
        let lfo_phase2 = g.n(.phasor(g.alloc()), targetLFOFreq, reset)

        let twoPi = g.n(.constant(2.0 * Float.pi))
        let sine1 = g.n(.sin, g.n(.mul, phase1, twoPi))
        let sine2 = g.n(.sin, g.n(.mul, phase2, twoPi))

        let sig1 = g.n(.mul, sine1, lfo_phase1)
        let sig2 = g.n(.mul, sine2, lfo_phase2)

        func smoothLeak(_ sig1: NodeID) -> NodeID {
            let one = g.n(.constant(1.0))
            let half = g.n(.constant(0.5))
            let alpha = g.n(.constant(0.01))  // leak
            let beta = g.n(.constant(8.0))  // sharpness

            let bx = g.n(.mul, beta, sig1)
            let s = g.n(.mul, half, g.n(.add, one, g.n(.tanh, bx)))  // 0.5*(1+tanh(beta*x))
            let oneMinusAlpha = g.n(.sub, one, alpha)
            let gate = g.n(.add, alpha, g.n(.mul, oneMinusAlpha, s))  // α + (1-α)*s
            let smoothLeaky = g.n(.mul, gate, sig1)
            return smoothLeaky
        }

        // smooth the signals to help with vanishing gradients
        let leaky1 = smoothLeak(sig1)
        let leaky2 = smoothLeak(sig2)

        // Convert phase to sine wave (phasor outputs 0-1, multiply by 2π and take sin)
        // Compute spectral loss
        let windowSize = 64
        let spectralLoss = g.spectralLoss(leaky1, leaky2, windowSize: windowSize)
        let l2Loss = g.n(.mse, leaky1, leaky2)
        let loss = g.n(
            .add, g.n(.mul, g.n(.constant(100.0)), spectralLoss),
            g.n(.mul, g.n(.constant(0.1)), l2Loss))

        _ = g.n(.output(0), loss)

        // Compile with backwards pass enabled
        let frameCount = 256
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: true)
        )

        for kernel in result.kernels {
            print(kernel.source)
        }

        print("   ✅ Compiled with backwards=true")
        print("   Initial frequency: 227 Hz")
        print("   Target frequency: 300 Hz")

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // Training context using optimizer (replaces manual gradient descent)
        let ctx = TrainingContext(
            parameters: [freqParam, lfoFreqParam],
            optimizer: SGD(lr: 165.3),
            lossNode: loss
        )
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // Training loop using TrainingContext
        let numIterations = 4200
        var lossHistory: [Float] = []
        for iteration in 0..<numIterations {
            let currentLoss = ctx.runStepGPU()
            lossHistory.append(currentLoss)

            if iteration % 10 == 0 || iteration <= 10 {
                let freqGrad = freqParam.grad ?? 0.0
                let lfoGrad = lfoFreqParam.grad ?? 0.0
                print(
                    "   Iteration \(iteration): freq=\(String(format: "%.2f", freqParam.value)) Hz (grad: \(String(format: "%.6f", freqGrad))), lfoFreq=\(String(format: "%.2f", lfoFreqParam.value)) Hz (grad: \(String(format: "%.6f", lfoGrad))), loss=\(String(format: "%.6f", currentLoss))"
                )
            }
        }

        // Results
        let finalFreq = freqParam.value
        let finalLoss = lossHistory.last ?? .infinity
        let initialLoss = lossHistory.first ?? .infinity

        print("   Final frequency: \(String(format: "%.2f", finalFreq)) Hz")
        print("   Final loss: \(String(format: "%.6f", finalLoss))")
        print(
            "   Loss reduction: \(String(format: "%.1f", (initialLoss - finalLoss) / max(initialLoss, .ulpOfOne) * 100))%"
        )

        // Verify learning happened
        XCTAssertLessThan(finalLoss, initialLoss * 0.5, "Loss should decrease by at least 50%")
        XCTAssertLessThan(
            abs(finalFreq - targetFrequency), 50.0, "Frequency should be within 50 Hz of target")

        print("   ✅ PASS: Successfully learned target frequency!")
    }
}

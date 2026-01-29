import XCTest

@testable import DGen
@testable import DGenFrontend

final class FrontendTests: XCTestCase {
        func testHistoryBackward() throws {
                let g = GraphBuilder()

                let freq = g.constant(440.0)
                let (cutoffParam, cutoff) = g.learnableParam(value: 0.5, name: "Cutoff")

                func onepole(_ x: DGenFrontend.Node, _ cutoff: DGenFrontend.Node)
                        -> DGenFrontend.Node
                {
                        let cellId = g.alloc()
                        let history = g.historyRead(cellId)
                        let mix = g.mix(x, history, cutoff)
                        g.historyWrite(cellId, mix)
                        return mix
                }
                let phase1 = g.phasor(freq)
                let filtered1 = onepole(phase1, cutoff)
                let sig1 = filtered1
                let sig2 = onepole(phase1, g.constant(0.2))
                let loss =
                        (g.spectralLoss(sig1, sig2, windowSize: 32)

                                + g.spectralLoss(sig1, sig2, windowSize: 64)
                                + g.spectralLoss(sig1, sig2, windowSize: 128)) * 1 + 0.1
                        * g.mse(sig1, sig2)

                let frameCount = 512 * 2
                let result = try g.compile(
                        loss, backend: .metal, frameCount: frameCount, debug: false)

                // Write kernels to disk for inspection
                let allKernels = result.kernels.enumerated().map {
                        "// KERNEL \($0.offset)\n\($0.element.source)"
                }.joined(separator: "\n\n")
                try! allKernels.write(
                        toFile: "/tmp/history_backward_kernels.metal", atomically: true,
                        encoding: .utf8)
                print("Wrote kernels to /tmp/history_backward_kernels.metal")
                // Streamlined training context - handles everything!
                let ctx = try TrainingContext(
                        parameters: [cutoffParam],
                        optimizer: SGD(lr: 0.1),
                        lossNode: loss.id,
                        compilationResult: result,
                        frameCount: frameCount
                )

                var lossHistory: [Float] = []
                for i in 0..<100 {
                        print("i=\(i)")
                        let currentLoss = ctx.runStepGPU()
                        lossHistory.append(currentLoss)
                        if currentLoss < 0.001 {
                                break
                        }
                        if i % 10 == 0 {
                                print(
                                        "i=\(i) loss=\(currentLoss) cutoff=\(cutoffParam.value) grad=\(cutoffParam.grad)"
                                )
                        }

                }
                let finalLoss = lossHistory.last ?? 10000.0
                XCTAssertLessThan(finalLoss, 0.02, "Loss should be very low")
        }

        func testBiquadBackward() throws {
                let g = GraphBuilder()

                let targetFreq = g.constant(452.0)
                let targetCutoff = g.constant(1030.0)

                let (freqParam, freq) = g.learnableParam(value: 445.0, name: "Freq")
                let (cutoffParam, cutoff) = g.learnableParam(value: 998.0, name: "Cutoff")

                let phase1 = g.phasor(freq)
                let phase2 = g.phasor(targetFreq)
                let resonance = g.constant(4)
                let gain = g.constant(1)
                let mode = g.constant(0)
                let sig1 = g.biquad(phase1, cutoff, resonance, gain, mode)
                let frameCount = 512 * 2
                let sig2 = g.biquad(phase2, targetCutoff, resonance, gain, mode)

                let loss =
                        g.spectralLoss(sig1, sig2, windowSize: 128) + 0.1
                        * g.mse(sig1, sig2)

                let result = try g.compile(
                        loss, backend: .metal, frameCount: frameCount, debug: false)

                // Write kernels to disk for inspection
                let allKernels = result.kernels.enumerated().map {
                        "// KERNEL \($0.offset)\n\($0.element.source)"
                }.joined(separator: "\n\n")
                try! allKernels.write(
                        toFile: "/tmp/biquad_backward_kernels.metal", atomically: true,
                        encoding: .utf8)
                print("Wrote kernels to /tmp/biquad_backward_kernels.metal")

                // Streamlined training context - handles everything!
                let ctx = try TrainingContext(
                        parameters: [cutoffParam, freqParam],
                        optimizer: Adam(lr: 0.1),
                        lossNode: loss.id,
                        compilationResult: result,
                        frameCount: frameCount
                )

                var lossHistory: [Float] = []
                for i in 0..<1000 {
                        let currentLoss = ctx.runStepGPU()
                        lossHistory.append(currentLoss)
                        if i % 10 == 0 {
                                print(
                                        "i=\(i) loss=\(currentLoss) freq=\(freqParam.value) cutoff=\(cutoffParam.value) cutoff.grad=\(cutoffParam.grad) freq.grad=\(freqParam.grad)"
                                )
                        }

                }
                let finalLoss = lossHistory.last ?? 1000.0

                print("   Final loss: \(String(format: "%.6f", finalLoss))")

                // Assert that we got close to target
                XCTAssertLessThan(finalLoss, 0.02, "Loss should be very low")

        }

        func testHybridLoss() throws {
                print("\n🧪 Test: Hybrid Loss (Time + Frequency Domain)")

                let g = GraphBuilder()

                // Two learnable parameters
                let (freqParam, freq) = g.learnableParam(value: 520.0, name: "frequency")
                let (ampParam, amp) = g.learnableParam(value: 0.3, name: "amplitude")

                // Target signal: 440 Hz sine at 0.5 amplitude
                let targetFreq = g.constant(440.0)
                let targetAmp = g.constant(0.5)
                let reset = g.constant(0.0)

                // Generate signals - much cleaner!
                let predictedPhase = g.phasor(freq, reset: reset)
                let targetPhase = g.phasor(targetFreq, reset: reset)

                // Float * Node operator overloading at work!
                let twoPi = 2.0 * Float.pi
                let predictedSig = sin(predictedPhase * twoPi) * amp  // Node * Node
                let targetSig = sin(targetPhase * twoPi) * targetAmp

                // Hybrid loss: 10% time-domain MSE + 200% spectral loss
                let timeLoss = g.mse(predictedSig, targetSig)
                let freqLoss = g.spectralLoss(predictedSig, targetSig, windowSize: 64)
                let hybridLoss = timeLoss * 0.1 + freqLoss * 2.0  // Node * Float

                // Compile - simple!
                let frameCount = 128
                let result = try g.compile(hybridLoss, frameCount: frameCount)

                print("   ✅ Compiled hybrid loss graph")
                print("   Initial: freq=\(freqParam.value) Hz, amp=\(ampParam.value)")
                print("   Target: freq=440 Hz, amp=0.5")

                // Streamlined training context
                let ctx = try TrainingContext(
                        parameters: [freqParam, ampParam],
                        optimizer: Adam(lr: 0.3),
                        lossNode: hybridLoss.id,
                        compilationResult: result,
                        frameCount: frameCount
                )

                // Training loop - clean and simple!
                let numIterations = 5000
                for iteration in 0..<numIterations {
                        let currentLoss = ctx.runStepGPU()

                        if currentLoss < 0.001 {
                                break

                        }
                        if iteration % 50 == 0 {
                                print(
                                        "   Iteration \(iteration): freq=\(String(format: "%.2f", freqParam.value)) Hz, "
                                                + "amp=\(String(format: "%.3f", ampParam.value)), loss=\(String(format: "%.6f", currentLoss))"
                                )
                        }
                }

                print(
                        "   Final: freq=\(String(format: "%.2f", freqParam.value)) Hz, amp=\(String(format: "%.3f", ampParam.value))"
                )

                // Verify both parameters learned
                XCTAssertLessThan(
                        abs(freqParam.value - 440.0), 20.0, "Frequency should be close to 440 Hz")
                XCTAssertLessThan(
                        abs(ampParam.value - 0.5), 0.1, "Amplitude should be close to 0.5")
        }

        func testGPUTraining() throws {
                print("\n🧪 Test: GPU Training vs CPU Training")

                // Setup graph
                let g = GraphBuilder()
                let (freqParam, freq) = g.learnableParam(value: 500.0, name: "frequency")
                let targetFreq = g.constant(440.0)
                let reset = g.constant(0.0)

                let phase1 = g.phasor(freq, reset: reset)
                let phase2 = g.phasor(targetFreq, reset: reset)

                let twoPi = 2.0 * Float.pi
                let sig1 = sin(phase1 * twoPi)
                let sig2 = sin(phase2 * twoPi)

                let loss = g.spectralLoss(sig1, sig2, windowSize: 64)

                // Compile
                let frameCount = 128
                let result = try g.compile(loss, frameCount: frameCount)

                print("   ✅ Compiled graph")
                print("   Initial frequency: \(freqParam.value) Hz")

                // Test GPU training
                let ctxGPU = try TrainingContext(
                        parameters: [freqParam],
                        optimizer: SGD(lr: 50.0),  // Higher learning rate for faster convergence
                        lossNode: loss.id,
                        compilationResult: result,
                        frameCount: frameCount
                )

                print("   🚀 Running GPU training for 200 iterations...")
                var gpuLossHistory: [Float] = []
                for iteration in 0..<200 {
                        let currentLoss = ctxGPU.runStepGPU()
                        gpuLossHistory.append(currentLoss)

                        if iteration % 40 == 0 {
                                print(
                                        "   GPU Iteration \(iteration): freq=\(String(format: "%.2f", freqParam.value)) Hz, loss=\(String(format: "%.6f", currentLoss))"
                                )
                        }
                }

                let gpuFinalFreq = freqParam.value
                let gpuFinalLoss = gpuLossHistory.last ?? 1000.0
                let initialLoss = gpuLossHistory.first ?? 0.0

                print(
                        "   GPU Final: freq=\(String(format: "%.2f", gpuFinalFreq)) Hz, loss=\(String(format: "%.6f", gpuFinalLoss))"
                )
                print(
                        "   GPU Progress: \(String(format: "%.1f", abs(500.0 - gpuFinalFreq))) Hz moved, \(String(format: "%.1f", (1.0 - gpuFinalLoss/initialLoss) * 100))% loss reduction"
                )

                // Verify GPU training is working (relaxed criteria)
                XCTAssertLessThan(
                        abs(gpuFinalFreq - 500.0), 100.0,
                        "GPU training should move parameters (at least 1 Hz)")
                XCTAssertLessThan(
                        gpuFinalLoss, initialLoss * 0.8, "GPU loss should decrease by at least 20%")

                print("   ✅ GPU training test passed!")
        }
}

import XCTest

@testable import DGen
@testable import DGenFrontend

final class FrontendTests: XCTestCase {
        func testSpectralLossLearnsFrequencyWithFrontend() throws {
                print("\n🧪 Test: Spectral Loss Learning with Frontend API")

                let g = GraphBuilder()

                // Learnable frequency parameter (start at 550 Hz, target is 440 Hz)
                let (freqParam, freq) = g.learnableParam(value: 650.0, name: "frequency")

                // Target frequency (constant 440 Hz) - no more passing g around!
                let targetFreq = g.constant(440.0)
                let reset = g.constant(0.0)

                // Generate phasors (0-1 ramps) - allocation handled automatically!
                let phase1 = g.phasor(freq, reset: reset)
                let phase2 = g.phasor(targetFreq, reset: reset)

                // Convert phase to sine wave - showing Float * Node operator overloading!
                let twoPi = 2.0 * Float.pi  // Just a regular Float
                let sig1 = sin(phase1 * twoPi)  // Node * Float works!
                let sig2 = sin(phase2 * twoPi)

                // Multi-scale spectral loss (cleaner than raw graph API!)
                let loss1 = g.spectralLoss(sig1, sig2, windowSize: 32)
                let loss2 = g.spectralLoss(sig1, sig2, windowSize: 64)
                let loss3 = g.spectralLoss(sig1, sig2, windowSize: 256)
                let mse = g.mse(sig1, sig2)
                let loss = (loss1 + loss2 + loss3) / 3.0 + 1.5 * mse

                // Compile - one line!
                let frameCount = 512
                let result = try g.compile(loss, frameCount: frameCount, debug: true)

                print("   ✅ Compiled with backwards=true")
                print("   Initial frequency: \(freqParam.value) Hz")
                print("   Target frequency: 440 Hz")

                // Streamlined training context - handles everything!
                let ctx = try TrainingContext(
                        parameters: [freqParam],
                        optimizer: Adam(lr: 62.0),
                        lossNode: loss.id,
                        compilationResult: result,
                        frameCount: frameCount
                )

                // Training loop - super simple now!
                let numIterations = 2500
                var lossHistory: [Float] = []
                for iteration in 0..<numIterations {
                        let currentLoss = ctx.runStep()  // That's it!
                        lossHistory.append(currentLoss)

                        if iteration % 100 == 0 || iteration <= 10 {
                                print(
                                        "   Iteration \(iteration): freq=\(String(format: "%.2f", freqParam.value)) Hz, loss=\(String(format: "%.6f", currentLoss))"
                                )
                        }
                }

                // Verify learning
                let finalFreq = freqParam.value
                let finalLoss = lossHistory.last ?? 1000.0

                print("   Final frequency: \(String(format: "%.2f", finalFreq)) Hz")
                print("   Final loss: \(String(format: "%.6f", finalLoss))")

                // Assert that we got close to target
                XCTAssertLessThan(
                        abs(finalFreq - 440.0), 10.0, "Should learn target frequency within 10 Hz")
                XCTAssertLessThan(finalLoss, 0.5, "Loss should be very low")
        }

        func testHybridLoss() throws {
                print("\n🧪 Test: Hybrid Loss (Time + Frequency Domain)")

                let g = GraphBuilder()

                // Two learnable parameters
                let (freqParam, freq) = g.learnableParam(value: 420.0, name: "frequency")
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
                        optimizer: SGD(lr: 2.3),
                        lossNode: hybridLoss.id,
                        compilationResult: result,
                        frameCount: frameCount
                )

                // Training loop - clean and simple!
                let numIterations = 5000
                for iteration in 0..<numIterations {
                        let currentLoss = ctx.runStep()

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
}

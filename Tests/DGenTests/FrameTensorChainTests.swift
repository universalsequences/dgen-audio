import XCTest

@testable import DGen
@testable import DGenFrontend

final class FrameTensorChainTests: XCTestCase {

    /// Test: frequencies tensor -> phasor -> cos -> sum -> loss
    /// This isolates the frame-dependent tensor chain pattern to check for determinism
    func testPhasorReduceSumDeterminism() throws {
        let g = Graph()
        g.sampleRate = 2000.0

        let numFreqs = 8
        let frameCount = 512

        // Create a tensor of frequencies [8] - static, not learnable
        let frequencies: [Float] = (0..<numFreqs).map { Float($0 + 1) * 100.0 }  // 100, 200, 300, ...
        let freqTensor = g.tensor(shape: [numFreqs], data: frequencies)

        // phasor(frequencies) -> [8] tensor of phases (frame-dependent via deterministicPhasor)
        let phasors = g.n(.deterministicPhasor, [freqTensor])

        // cos(phasors * 2pi) -> [8] tensor of sinusoids
        let twoPi = g.n(.constant(Float.pi * 2), [])
        let scaledPhasors = g.n(.mul, [phasors, twoPi])
        let sinusoids = g.n(.cos, [scaledPhasors])

        // sum(sinusoids) -> scalar per frame
        let summed = g.n(.sum, [sinusoids])

        // Square to make a loss
        let squared = g.n(.mul, [summed, summed])

        _ = g.n(.output(0), [squared])

        print("\n=== Phasor -> Sum Chain Test ===")
        print("frameCount: \(frameCount), numFreqs: \(numFreqs)")
        print("frequencies: \(frequencies)")

        // Compile using GraphTrainingContext (even though we're not training)
        let ctx = try GraphTrainingContext(
            graph: g,
            loss: squared,
            parameters: [],
            optimizer: GraphSGD(),
            learningRate: 0.01,
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/phasor_reduce_chain.metal"
        )

        // Run forward multiple times and check for determinism
        var losses: [Float] = []
        for run in 0..<5 {
            ctx.zeroGrad()
            let loss = ctx.forward()
            losses.append(loss)
            print("Forward run \(run): loss=\(loss)")
        }

        // Check if all runs produced the same loss
        let firstLoss = losses[0]
        let allSame = losses.allSatisfy { abs($0 - firstLoss) < 1e-6 }

        print("\nDeterminism check: \(allSame ? "PASS" : "FAIL")")
        if !allSame {
            print("Losses varied: min=\(losses.min()!), max=\(losses.max()!)")
        }

        XCTAssertTrue(allSame, "Forward pass should be deterministic - got varying losses: \(losses)")
    }

    /// Similar to above but with a learnable tensor to see gradient flow too
    func testPhasorReduceSumWithLearnable() throws {
        let g = Graph()
        g.sampleRate = 2000.0

        let numFreqs = 8
        let frameCount = 512

        // Create a LEARNABLE tensor of frequencies [8]
        let freqData: [Float] = (0..<numFreqs).map { Float($0 + 1) * 100.0 }
        let freqParam = TensorParameter(graph: g, shape: [numFreqs], data: freqData, name: "frequencies")

        // phasor(frequencies) -> [8] tensor of phases (frame-dependent)
        let phasors = g.n(.deterministicPhasor, [freqParam.node()])

        // cos(phasors * 2pi) -> [8] tensor of sinusoids
        let twoPi = g.n(.constant(Float.pi * 2), [])
        let scaledPhasors = g.n(.mul, [phasors, twoPi])
        let sinusoids = g.n(.cos, [scaledPhasors])

        // sum(sinusoids) -> scalar per frame
        let summed = g.n(.sum, [sinusoids])

        // Target a specific output
        let target = g.n(.constant(0.5), [])
        let loss = g.n(.mse, [summed, target])

        _ = g.n(.output(0), [loss])

        print("\n=== Phasor -> Sum Chain (Learnable) Test ===")
        print("frameCount: \(frameCount), numFreqs: \(numFreqs)")

        let ctx = try GraphTrainingContext(
            graph: g,
            loss: loss,
            tensorParameters: [freqParam],
            optimizer: GraphSGD(),
            learningRate: 0.1,
            frameCount: frameCount,
            kernelDebugOutput: "/tmp/phasor_reduce_learnable.metal"
        )

        // Check determinism first
        var losses: [Float] = []
        for run in 0..<5 {
            ctx.zeroGrad()
            let lossVal = ctx.forward()
            losses.append(lossVal)
            print("Forward run \(run): loss=\(lossVal)")
        }

        let firstLoss = losses[0]
        let allSame = losses.allSatisfy { abs($0 - firstLoss) < 1e-6 }
        print("\nDeterminism check: \(allSame ? "PASS" : "FAIL")")

        XCTAssertTrue(allSame, "Forward pass should be deterministic")

        // Now check gradients
        let initialLoss = ctx.trainStep()
        print("\nInitial loss: \(initialLoss)")
        print("Gradients: \(freqParam.grads)")

        let hasNonZeroGrads = freqParam.grads.contains { $0 != 0.0 }
        XCTAssertTrue(hasNonZeroGrads, "Should have non-zero gradients")
    }
}

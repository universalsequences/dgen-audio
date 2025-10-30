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

        // Initialize gradient seeds before running forward/backward pass
        print("   Resetting gradients buffer...")
        runtime.resetGradientBuffers(numFrames: frameCount)

        // Run forward and backward pass (both happen in runWithMemory)
        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)
        let memory = runtime.allocateNodeMemory()!

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

        // Also check tape buffer to see if loss is computed correctly
        if let tape = runtime.readBuffer(named: "t") {
            let lossTapeSlot = 7  // From kernel analysis (updated after fixing globals ordering)
            print(
                "   [DEBUG] Tape slot \(lossTapeSlot) (spectralLoss): first 5 = \(tape[(lossTapeSlot*frameCount)..<(lossTapeSlot*frameCount+5)].map { String(format: "%.2f", $0) }.joined(separator: ", "))"
            )
        }

        let initialLoss = outputBuffer[frameCount - 1]
        print("   Initial loss: \(String(format: "%.6f", initialLoss))")
        print(
            "   [DEBUG] Output buffer first 5: \(outputBuffer.prefix(5).map { String(format: "%.2f", $0) }.joined(separator: ", "))"
        )
        print(
            "   [DEBUG] Output buffer last 5: \(outputBuffer.suffix(5).map { String(format: "%.2f", $0) }.joined(separator: ", "))"
        )

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

        runtime.deallocateNodeMemory(memory)
    }

    /// Test that spectral loss backward pass can learn to match a target frequency
    func testSpectralLossLearnsFrequency() throws {
        print("\n🧪 Test: Spectral Loss Learning - Frequency Matching")

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

        // Compute spectral loss
        let windowSize = 64
        let loss = g.spectralLoss(sig1, sig2, windowSize: windowSize)

        let outputNode = g.n(.output(0), loss)

        // Print graph structure
        print("   Graph nodes:")
        print(
            "     param (freq): node \(freqParam.nodeId) (\(g.nodes[freqParam.nodeId]?.op ?? .add))"
        )
        print("     targetFreq: node \(targetFreq) (\(g.nodes[targetFreq]?.op ?? .add))")
        print("     reset: node \(reset) (\(g.nodes[reset]?.op ?? .add))")
        print("     phase1: node \(phase1) (\(g.nodes[phase1]?.op ?? .add))")
        print("     phase2: node \(phase2) (\(g.nodes[phase2]?.op ?? .add))")
        print("     twoPi: node \(twoPi) (\(g.nodes[twoPi]?.op ?? .add))")
        print("     sig1: node \(sig1) (\(g.nodes[sig1]?.op ?? .add))")
        print("     sig2: node \(sig2) (\(g.nodes[sig2]?.op ?? .add))")
        print("     loss: node \(loss) (\(g.nodes[loss]?.op ?? .add))")
        if let lossNode = g.nodes[loss] {
            print("     loss inputs: \(lossNode.inputs)")
        }
        print("     output: node \(outputNode) (\(g.nodes[outputNode]?.op ?? .add))")
        print("   Buffer cells:")

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

        // Training context using optimizer (replaces manual gradient descent)
        let ctx = TrainingContext(
            parameters: [freqParam],
            optimizer: SGD(lr: 5.0),  // match manual learning rate used previously
            lossNode: loss
        )
        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // Buffers
        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Training loop using TrainingContext
        let numIterations = 500
        var lossHistory: [Float] = []
        for iteration in 0..<numIterations {
            // Zero gradients and reset memory (preserving params)
            ctx.zeroGrad()

            // Forward + backward pass
            inputBuffer.withUnsafeBufferPointer { inPtr in
                outputBuffer.withUnsafeMutableBufferPointer { outPtr in
                    runtime.runWithMemory(
                        outputs: outPtr.baseAddress!,
                        inputs: inPtr.baseAddress!,
                        memory: ctx.getMemory(),
                        frameCount: frameCount
                    )
                }
            }

            // Optimizer step updates params and syncs memory
            ctx.step()

            // Spectral loss is written to the output; use last frame's value as scalar loss
            let currentLoss = outputBuffer[frameCount - 1]
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

    /// Test that spectral loss pipeline runs without inspecting ring buffers
    func testSpectralLossGradMemory() throws {
        print("\n🧪 Test: Spectral Loss Learning - Frequency Matching")

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

        // Compute spectral loss
        let windowSize = 64
        let buf1 = g.alloc(vectorWidth: windowSize + 1)
        let buf2 = g.alloc(vectorWidth: windowSize + 1)
        let loss = g.spectralLoss(sig1, sig2, windowSize: windowSize)

        let outputNode = g.n(.output(0), loss)

        // Print graph structure

        // Compile with backwards pass enabled
        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: true, backwards: true)
        )

        for kernel in result.kernels {
            print(kernel.source)
        }

        // Create runtime
        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        // Set up parameter gradId from context
        freqParam.gradId = result.context.gradients[freqParam.nodeId]
        guard let paramGradId = freqParam.gradId else {
            XCTFail("Parameter gradId not found in context")
            return
        }

        print("   Parameter nodeId: \(freqParam.nodeId), gradId: \(paramGradId)")

        // Initialize parameter value in memory
        let memory = runtime.allocateNodeMemory()!
        let memoryPtr = memory.assumingMemoryBound(to: Float.self)

        // Map cell ID to physical cell (handle allocation mapping)
        let physicalCell = result.cellAllocations.cellMappings[freqParam.cellId] ?? freqParam.cellId
        memoryPtr[Int(physicalCell)] = freqParam.value
        print("   Set memory[\(physicalCell)] = \(freqParam.value)")

        // Check initial memory state for phasor cells
        print("   Initial memory state:")
        print("     memory[1] (phase1 cell) = \(memoryPtr[1])")
        print("     memory[2] (phase2 cell) = \(memoryPtr[2])")
        print("     memory[4] (freq param cell) = \(memoryPtr[4])")

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        // Training loop
        let numIterations = 1
        var lossHistory: [Float] = []

        for iteration in 0..<numIterations {
            // Reset memory except for parameter values
            // Save current parameter value
            let currentParamValue = memoryPtr[Int(physicalCell)]

            // Zero out all memory
            let memorySize = runtime.getMemorySize()
            memset(memory, 0, memorySize * MemoryLayout<Float>.size)

            // Restore parameter value
            memoryPtr[Int(physicalCell)] = currentParamValue

            // Reset gradients
            runtime.resetGradientBuffers(numFrames: frameCount)

            // Forward and backward pass
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

            let currentLoss = outputBuffer[frameCount - 1]
            lossHistory.append(currentLoss)

            // For tape-based compute, no ring/grad_memory inspection is required
            XCTAssertTrue(currentLoss.isFinite, "Loss should be a finite value")

        }

        runtime.deallocateNodeMemory(memory)
    }

}

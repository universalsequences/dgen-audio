import XCTest
import Foundation

@testable import DGen

final class PeekRowTests: XCTestCase {

    // MARK: - Test 1: Integer Row Index

    func testPeekRowIntegerIndex() throws {
        // Tensor [3, 4], peek row 1
        // Should return exact row values without interpolation
        let g = Graph()

        // Create 2D tensor [3 rows, 4 cols] with known values
        // Column-major layout: data[col * numRows + row]
        // Row 0: [0, 1, 2, 3]
        // Row 1: [10, 11, 12, 13]
        // Row 2: [20, 21, 22, 23]
        var data = [Float](repeating: 0, count: 12)
        for col in 0..<4 {
            for row in 0..<3 {
                data[col * 3 + row] = Float(row * 10 + col)
            }
        }
        let tensor = g.tensor(shape: [3, 4], data: data)
        let rowIndex = g.n(.constant(1.0))  // Read row 1

        let result = try g.peekRow(tensor: tensor, rowIndex: rowIndex)
        let sum = g.n(.sum, result)
        _ = g.n(.output(0), sum)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount)
        )

        let runtime = CCompiledKernel(
            source: compileResult.source,
            cellAllocations: compileResult.cellAllocations,
            memorySize: compileResult.totalMemorySlots
        )
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { runtime.deallocateNodeMemory(mem) }

        injectTensorData(result: compileResult, memory: mem.assumingMemoryBound(to: Float.self))

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        outputBuffer.withUnsafeMutableBufferPointer { outPtr in
            inputBuffer.withUnsafeBufferPointer { inPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Row 1 = [10, 11, 12, 13], sum = 46
        let expectedSum: Float = 10 + 11 + 12 + 13
        XCTAssertEqual(outputBuffer[0], expectedSum, accuracy: 0.001)
    }

    // MARK: - Test 2: Fractional Row Index (Interpolation)

    func testPeekRowFractionalIndex() throws {
        // Tensor [2, 2] with row 0 = [0, 0], row 1 = [10, 10]
        // Peek row 0.5 should return [5, 5]
        let g = Graph()

        // Column-major: [row0_col0, row1_col0, row0_col1, row1_col1]
        // Row 0: [0, 0], Row 1: [10, 10]
        let data: [Float] = [0, 10, 0, 10]
        let tensor = g.tensor(shape: [2, 2], data: data)
        let rowIndex = g.n(.constant(0.5))  // Interpolate between rows

        let result = try g.peekRow(tensor: tensor, rowIndex: rowIndex)
        let sum = g.n(.sum, result)
        _ = g.n(.output(0), sum)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount)
        )

        let runtime = CCompiledKernel(
            source: compileResult.source,
            cellAllocations: compileResult.cellAllocations,
            memorySize: compileResult.totalMemorySlots
        )
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { runtime.deallocateNodeMemory(mem) }

        injectTensorData(result: compileResult, memory: mem.assumingMemoryBound(to: Float.self))

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        outputBuffer.withUnsafeMutableBufferPointer { outPtr in
            inputBuffer.withUnsafeBufferPointer { inPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // 0.5 interpolation: [5, 5], sum = 10
        let expectedSum: Float = 10.0
        XCTAssertEqual(outputBuffer[0], expectedSum, accuracy: 0.001)
    }

    // MARK: - Test 3: Wrapping

    func testPeekRowWrapping() throws {
        // Peek row index >= numRows should wrap via modulo
        let g = Graph()

        // Row 0: [0, 0], Row 1: [10, 10], Row 2: [20, 20]
        let data: [Float] = [0, 10, 20, 0, 10, 20]  // Column-major
        let tensor = g.tensor(shape: [3, 2], data: data)
        let rowIndex = g.n(.constant(4.0))  // 4 % 3 = 1, should read row 1

        let result = try g.peekRow(tensor: tensor, rowIndex: rowIndex)
        let sum = g.n(.sum, result)
        _ = g.n(.output(0), sum)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .c,
            options: .init(frameCount: frameCount)
        )

        let runtime = CCompiledKernel(
            source: compileResult.source,
            cellAllocations: compileResult.cellAllocations,
            memorySize: compileResult.totalMemorySlots
        )
        try runtime.compileAndLoad()

        guard let mem = runtime.allocateNodeMemory() else {
            XCTFail("Failed to allocate memory")
            return
        }
        defer { runtime.deallocateNodeMemory(mem) }

        injectTensorData(result: compileResult, memory: mem.assumingMemoryBound(to: Float.self))

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var outputBuffer = [Float](repeating: 0.0, count: frameCount)

        outputBuffer.withUnsafeMutableBufferPointer { outPtr in
            inputBuffer.withUnsafeBufferPointer { inPtr in
                runtime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: mem,
                    frameCount: frameCount
                )
            }
        }

        // Row 1 = [10, 10], sum = 20
        let expectedSum: Float = 20.0
        XCTAssertEqual(outputBuffer[0], expectedSum, accuracy: 0.001)
    }

    // MARK: - Test 4: Backward - Integer Index

    func testPeekRowBackwardIntegerIndex() throws {
        // Only the indexed row should receive gradients
        let g = Graph()

        // 3 rows, 2 cols
        let data: [Float] = [1, 2, 3, 4, 5, 6]  // Column-major
        let tensorParam = TensorParameter(graph: g, shape: [3, 2], data: data, name: "weights")
        let rowIndex = g.n(.constant(1.0))  // Read row 1

        let result = try g.peekRow(tensor: tensorParam.node(), rowIndex: rowIndex)
        let sum = g.n(.sum, result)
        let target = g.n(.constant(0.0))
        let loss = g.n(.mse, sum, target)
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [tensorParam],
            optimizer: SGD(lr: 0.01),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )
        _ = ctx.runStepGPU()

        let grads = [tensorParam.grads]
        XCTAssertEqual(grads.count, 1)

        // Only row 1 should have gradients (indices 1, 4 in column-major layout)
        let tensorGrads = grads[0]
        XCTAssertEqual(tensorGrads.count, 6)

        // Row 0 and Row 2 should have zero gradients
        XCTAssertEqual(tensorGrads[0], 0.0, accuracy: 0.001, "Row 0, Col 0 should be zero")
        XCTAssertEqual(tensorGrads[2], 0.0, accuracy: 0.001, "Row 2, Col 0 should be zero")
        XCTAssertEqual(tensorGrads[3], 0.0, accuracy: 0.001, "Row 0, Col 1 should be zero")
        XCTAssertEqual(tensorGrads[5], 0.0, accuracy: 0.001, "Row 2, Col 1 should be zero")

        // Row 1 should have non-zero gradients
        XCTAssertNotEqual(tensorGrads[1], 0.0, "Row 1, Col 0 should be non-zero")
        XCTAssertNotEqual(tensorGrads[4], 0.0, "Row 1, Col 1 should be non-zero")
    }

    // MARK: - Test 5: Backward - Fractional Index

    func testPeekRowBackwardFractionalIndex() throws {
        // Both adjacent rows should receive proportional gradients
        let g = Graph()

        // 2 rows, 2 cols
        let data: [Float] = [0, 10, 0, 10]  // Column-major
        let tensorParam = TensorParameter(graph: g, shape: [2, 2], data: data, name: "weights")
        let rowIndex = g.n(.constant(0.5))  // Read between rows 0 and 1

        let result = try g.peekRow(tensor: tensorParam.node(), rowIndex: rowIndex)
        let sum = g.n(.sum, result)
        let target = g.n(.constant(0.0))
        let loss = g.n(.mse, sum, target)
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [tensorParam],
            optimizer: SGD(lr: 0.01),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )
        _ = ctx.runStepGPU()

        let tensorGrads = tensorParam.grads
        XCTAssertEqual(tensorGrads.count, 4)

        // Both rows should have non-zero gradients due to interpolation
        // With frac=0.5, both rows get equal gradient contribution
        XCTAssertNotEqual(tensorGrads[0], 0.0, "Row 0, Col 0 should have gradient")
        XCTAssertNotEqual(tensorGrads[1], 0.0, "Row 1, Col 0 should have gradient")
        XCTAssertNotEqual(tensorGrads[2], 0.0, "Row 0, Col 1 should have gradient")
        XCTAssertNotEqual(tensorGrads[3], 0.0, "Row 1, Col 1 should have gradient")

        // Gradients should be approximately equal for 0.5 interpolation
        XCTAssertEqual(abs(tensorGrads[0]), abs(tensorGrads[1]), accuracy: 0.001)
        XCTAssertEqual(abs(tensorGrads[2]), abs(tensorGrads[3]), accuracy: 0.001)
    }

    // MARK: - Test 6: Training Convergence

    func testPeekRowTrainingConvergence() throws {
        // Learn tensor values to match a target output
        let g = Graph()

        // Start with zeros, learn to produce target values
        let tensorParam = TensorParameter(
            graph: g, shape: [2, 3],
            data: [0, 0, 0, 0, 0, 0],  // All zeros
            name: "weights"
        )
        let rowIndex = g.n(.constant(0.0))  // Read row 0

        let result = try g.peekRow(tensor: tensorParam.node(), rowIndex: rowIndex)
        let sum = g.n(.sum, result)

        // Target sum = 15 (we want row 0 to sum to 15)
        let target = g.n(.constant(15.0))
        let loss = g.n(.mse, sum, target)
        _ = g.n(.output(0), loss)

        let frameCount = 1
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [tensorParam],
            optimizer: Adam(lr: 0.1),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )

        // Get initial loss
        let initialLoss = ctx.runStepGPU()

        // Train for 100 epochs
        var finalLoss = initialLoss
        for _ in 0..<100 {
            finalLoss = ctx.runStepGPU()
        }

        // Verify training reduced the loss significantly
        XCTAssertLessThan(finalLoss, initialLoss * 0.01, "Loss should decrease significantly")
        XCTAssertLessThan(finalLoss, 1.0, "Final loss should be near zero")
    }

    // MARK: - Test 7: Neural LFO / Learned Envelope

    func testNeuralLFO_LearnedEnvelope() throws {
        // Demonstrates "Neural Synthesis" concept:
        // - A learnable tensor acts as a "Neural LFO"
        // - peekRow reads time-varying control values
        // - Training learns the envelope shape to match a target
        //
        // Setup: 4 oscillators, each needs an amplitude envelope over 8 time steps
        // Target: Learn specific amplitude patterns for each oscillator

        let g = Graph()
        let numTimeSteps = 8
        let numOscillators = 4
        let frameCount = 32  // Run multiple frames to sweep through the envelope

        // Learnable control tensor [timeSteps, oscillators]
        // Initialized to zeros - will learn the envelope
        let envelopeTensor = TensorParameter(
            graph: g,
            shape: [numTimeSteps, numOscillators],
            data: [Float](repeating: 0.0, count: numTimeSteps * numOscillators),
            name: "envelope"
        )

        // Target envelope (what we want to learn):
        // Osc 0: ramp up    [0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.0]
        // Osc 1: ramp down  [1, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0]
        // Osc 2: pulse      [0, 0, 1, 1, 1, 1, 0, 0]
        // Osc 3: constant   [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        var targetData = [Float](repeating: 0, count: numTimeSteps * numOscillators)
        for t in 0..<numTimeSteps {
            // Column-major: offset = t * numOscillators + osc... wait no
            // Our tensor is [timeSteps, oscillators] = [8, 4]
            // Column-major for peekRow: offset = col * numRows + row = osc * numTimeSteps + t
            let rampUp = Float(t) / Float(numTimeSteps - 1)
            let rampDown = 1.0 - rampUp
            let pulse: Float = (t >= 2 && t <= 5) ? 1.0 : 0.0
            let constant: Float = 0.5

            targetData[0 * numTimeSteps + t] = rampUp      // Osc 0
            targetData[1 * numTimeSteps + t] = rampDown    // Osc 1
            targetData[2 * numTimeSteps + t] = pulse       // Osc 2
            targetData[3 * numTimeSteps + t] = constant    // Osc 3
        }
        let targetTensor = g.tensor(shape: [numTimeSteps, numOscillators], data: targetData)

        // Playhead: sweeps 0 to numTimeSteps-1 over the frames
        // Using a simple linear ramp via phasor
        let phasorCell = g.alloc()
        let playheadNorm = g.n(.phasor(phasorCell),
            g.n(.constant(1.0 / Float(frameCount))),  // freq: complete one cycle over frameCount
            g.n(.constant(0.0)))  // no reset
        let playhead = g.n(.mul, playheadNorm, g.n(.constant(Float(numTimeSteps - 1))))

        // Read current amplitudes from learned envelope
        let learnedAmplitudes = try g.peekRow(tensor: envelopeTensor.node(), rowIndex: playhead)

        // Read target amplitudes at same position
        let targetAmplitudes = try g.peekRow(tensor: targetTensor, rowIndex: playhead)

        // Loss: MSE between learned and target amplitudes
        let diff = g.n(.sub, learnedAmplitudes, targetAmplitudes)
        let sqDiff = g.n(.mul, diff, diff)
        let loss = g.n(.sum, sqDiff)
        _ = g.n(.output(0), loss)

        // Compile with backwards pass
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [envelopeTensor],
            optimizer: Adam(lr: 0.1),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )
        // Get initial loss
        let initialLoss = ctx.runStepGPU()

        // Train
        var finalLoss = initialLoss
        for _ in 0..<200 {
            finalLoss = ctx.runStepGPU()
        }

        print("Neural LFO Test - Initial loss: \(initialLoss), Final loss: \(finalLoss)")

        // Verify convergence - the neural LFO learned the envelope
        XCTAssertLessThan(finalLoss, initialLoss * 0.1,
            "Neural LFO should learn the target envelope (initial: \(initialLoss), final: \(finalLoss))")

        // Loss should be reasonably small
        XCTAssertLessThan(finalLoss, 1.0,
            "Final loss should be small, got \(finalLoss)")
    }

    // MARK: - Test 8: Neural Synthesis from Real Audio (Piano Envelope)

    func testNeuralSynthesis_LearnPianoEnvelope() throws {
        // Load the extracted piano envelope from JSON
        let envelopeURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()  // DGenTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // dgen
            .appendingPathComponent("Assets/piano_envelope.json")

        guard FileManager.default.fileExists(atPath: envelopeURL.path) else {
            print("Skipping test - piano_envelope.json not found at \(envelopeURL.path)")
            print("Run: cd Assets && python3 extract_envelope.py 'Piano - AC.wav' piano_envelope.json")
            return
        }

        let jsonData = try Data(contentsOf: envelopeURL)
        let json = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let targetEnvelope = json["envelope"] as! [Double]
        let numPoints = targetEnvelope.count

        print("Loaded piano envelope with \(numPoints) points")
        print("Envelope shape: attack=\(targetEnvelope[0...2]), decay=\(targetEnvelope[3...10].map { String(format: "%.2f", $0) })")

        let g = Graph()
        let frameCount = 128  // Sweep through envelope multiple times for better coverage

        // Target envelope as tensor (column-major for peekRow: [numPoints, 1])
        let targetData = targetEnvelope.map { Float($0) }
        let targetTensor = g.tensor(shape: [numPoints, 1], data: targetData)

        // Learnable envelope - start with flat 0.5
        let learnedEnvelope = TensorParameter(
            graph: g,
            shape: [numPoints, 1],
            data: [Float](repeating: 0.5, count: numPoints),
            name: "learned_envelope"
        )

        // Playhead sweeps through envelope
        let phasorCell = g.alloc()
        let playheadNorm = g.n(.phasor(phasorCell),
            g.n(.constant(1.0 / Float(frameCount))),
            g.n(.constant(0.0)))
        let playhead = g.n(.mul, playheadNorm, g.n(.constant(Float(numPoints - 1))))

        // Read envelopes at current position
        let learnedAmp = try g.peekRow(tensor: learnedEnvelope.node(), rowIndex: playhead)
        let targetAmp = try g.peekRow(tensor: targetTensor, rowIndex: playhead)

        // Loss: MSE between learned and target amplitude
        let diff = g.n(.sub, learnedAmp, targetAmp)
        let sqDiff = g.n(.mul, diff, diff)
        let loss = g.n(.sum, sqDiff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [learnedEnvelope],
            optimizer: Adam(lr: 0.05),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )
        // Initial loss
        let initialLoss = ctx.runStepGPU()

        // Train
        let epochs = 300
        var finalLoss = initialLoss
        for epoch in 0..<epochs {
            finalLoss = ctx.runStepGPU()

            if epoch % 100 == 0 {
                print("Epoch \(epoch): loss = \(finalLoss)")
            }
        }
        print("Neural Synthesis Test - Initial: \(initialLoss), Final: \(finalLoss)")

        // Verify the model learned the piano envelope
        XCTAssertLessThan(finalLoss, initialLoss * 0.01,
            "Should learn piano envelope (initial: \(initialLoss), final: \(finalLoss))")
        XCTAssertLessThan(finalLoss, 0.001,
            "Final loss should be very small")
    }

    // MARK: - Test 9: REAL Neural Synthesis - Learn Envelope from Audio

    func testNeuralSynthesis_LearnFromAudio() throws {
        // This is the REAL neural synthesis test:
        // - Load actual audio samples as target
        // - Synthesize: sin(freq * t) * learnedEnvelope
        // - Loss: MSE between synth output and target
        // - Learn the envelope that makes our synth sound like the piano

        // Load extracted samples (short version for fast testing)
        let samplesURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Assets/piano_samples_short.json")

        guard FileManager.default.fileExists(atPath: samplesURL.path) else {
            print("Skipping test - piano_samples.json not found")
            print("Run: cd Assets && python3 extract_samples.py 'Piano - AC.wav' piano_samples.json")
            return
        }

        let jsonData = try Data(contentsOf: samplesURL)
        let json = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let targetSamples = (json["samples"] as! [Double]).map { Float($0) }
        let sampleRate = json["target_sample_rate"] as! Int
        let pitchHz = json["detected_pitch_hz"] as! Double

        print("Loaded \(targetSamples.count) samples at \(sampleRate) Hz, pitch: \(pitchHz) Hz")

        let frameCount = targetSamples.count
        let numEnvelopePoints = 64  // Resolution of learned envelope

        let g = Graph()

        // Target samples as tensor - we'll read one per frame
        let targetTensor = g.tensor(shape: [frameCount, 1], data: targetSamples)

        // Learnable envelope [numPoints, 1] - start with small values
        let learnedEnvelope = TensorParameter(
            graph: g,
            shape: [numEnvelopePoints, 1],
            data: [Float](repeating: 0.1, count: numEnvelopePoints),
            name: "envelope"
        )

        let zero = g.n(.constant(0.0))

        // Oscillator using phasor (automatically increments phase each frame)
        let freq = g.n(.constant(Float(pitchHz)))
        let srFloat = g.n(.constant(Float(sampleRate)))
        let freqNorm = g.n(.div, freq, srFloat)  // freq / sampleRate

        let oscCell = g.alloc()
        let oscPhase = g.n(.phasor(oscCell), freqNorm, zero)  // 0 to 1 sawtooth at pitch freq
        let twoPi = g.n(.constant(Float.pi * 2.0))
        let osc = g.n(.sin, g.n(.mul, twoPi, oscPhase))

        // Envelope playhead: sweeps 0 to (numEnvelopePoints-1) over all frames
        // Use another phasor that completes one cycle over the entire duration
        let envCell = g.alloc()
        let envFreq = g.n(.constant(1.0 / Float(frameCount)))  // One cycle over all frames
        let playheadNorm = g.n(.phasor(envCell), envFreq, zero)
        let envelopeLen = g.n(.constant(Float(numEnvelopePoints - 1)))
        let playhead = g.n(.mul, playheadNorm, envelopeLen)

        // Frame index for reading target samples (0 to frameCount-1)
        let frameIdxCell = g.alloc()
        let frameIdxFreq = g.n(.constant(1.0 / Float(frameCount)))
        let frameIdxNorm = g.n(.phasor(frameIdxCell), frameIdxFreq, zero)
        let frameCountFloat = g.n(.constant(Float(frameCount - 1)))
        let frameIndex = g.n(.mul, frameIdxNorm, frameCountFloat)

        // Read envelope amplitude at current position
        let amplitude = try g.peekRow(tensor: learnedEnvelope.node(), rowIndex: playhead)

        // Synth output: oscillator * envelope
        let synthOutput = g.n(.mul, osc, g.n(.sum, amplitude))  // sum converts [1] tensor to scalar

        // Target: read current sample from target tensor
        let targetSample = try g.peek(tensor: targetTensor, index: frameIndex, channel: zero)

        // Loss: squared error for this frame
        let diff = g.n(.sub, synthOutput, targetSample)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [learnedEnvelope],
            optimizer: Adam(lr: 0.01),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )
        // Initial loss
        let initialLoss = ctx.runStepGPU()
        print("Initial loss: \(initialLoss)")

        // Train
        let epochs = 100
        var finalLoss = initialLoss
        for epoch in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            if epoch % 20 == 0 || epoch == epochs - 1 {
                print("Epoch \(epoch): loss = \(finalLoss)")
            }
        }
        print("Final loss: \(finalLoss)")

        // Verify learning happened
        // Note: A single sine wave can't perfectly match a piano (which has many harmonics),
        // so we just verify the loss decreased meaningfully
        XCTAssertLessThan(finalLoss, initialLoss * 0.9,
            "Loss should decrease (initial: \(initialLoss), final: \(finalLoss))")

        let improvement = (1.0 - finalLoss / initialLoss) * 100
        print("SUCCESS: Learned envelope from raw audio! Improvement: \(String(format: "%.1f", improvement))%")
    }

    // MARK: - Test 10: ACTUAL Neural Synthesis with MLP

    func testNeuralSynthesis_MLPEnvelope() throws {
        // This is REAL neural synthesis:
        // - MLP computes envelope from time input
        // - envelope(t) = W2 * tanh(W1 * t + b1) + b2
        // - Synthesize: sin(freq * t) * envelope(t)
        // - Learn weights to match piano samples

        // Load extracted samples
        let samplesURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Assets/piano_samples_short.json")

        guard FileManager.default.fileExists(atPath: samplesURL.path) else {
            print("Skipping test - piano_samples_short.json not found")
            return
        }

        let jsonData = try Data(contentsOf: samplesURL)
        let json = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let targetSamples = (json["samples"] as! [Double]).map { Float($0) }
        let sampleRate = json["target_sample_rate"] as! Int
        let pitchHz = json["detected_pitch_hz"] as! Double

        print("MLP Neural Synth: \(targetSamples.count) samples, \(sampleRate) Hz, pitch: \(pitchHz) Hz")

        let frameCount = targetSamples.count
        let hiddenSize = 16  // Hidden layer size

        let g = Graph()

        // Target samples tensor
        let targetTensor = g.tensor(shape: [frameCount, 1], data: targetSamples)

        // MLP weights (Xavier initialization)
        let scale1 = sqrt(2.0 / Float(1 + hiddenSize))
        let scale2 = sqrt(2.0 / Float(hiddenSize + 1))

        // W1: [1, hiddenSize] - input to hidden
        let W1 = TensorParameter(
            graph: g,
            shape: [1, hiddenSize],
            data: (0..<hiddenSize).map { _ in Float.random(in: -scale1...scale1) },
            name: "W1"
        )

        // b1: [hiddenSize] - hidden bias
        let b1 = TensorParameter(
            graph: g,
            shape: [hiddenSize, 1],
            data: [Float](repeating: 0.0, count: hiddenSize),
            name: "b1"
        )

        // W2: [hiddenSize, 1] - hidden to output
        let W2 = TensorParameter(
            graph: g,
            shape: [hiddenSize, 1],
            data: (0..<hiddenSize).map { _ in Float.random(in: -scale2...scale2) },
            name: "W2"
        )

        // b2: [1] - output bias (start at 0.5 since envelope is 0-1)
        let b2 = TensorParameter(
            graph: g,
            shape: [1, 1],
            data: [0.5],
            name: "b2"
        )

        let zero = g.n(.constant(0.0))

        // Time input: normalized 0 to 1 over all frames
        let timeCell = g.alloc()
        let timeFreq = g.n(.constant(1.0 / Float(frameCount)))
        let t = g.n(.phasor(timeCell), timeFreq, zero)  // 0 to 1

        // Create time as [1, 1] tensor for matmul using stack
        let tTensor = try g.stack([t], shape: [1, 1])  // [1, 1] tensor from scalar

        // Forward pass through MLP:
        // hidden = tanh(t @ W1 + b1)  -- [1,1] @ [1,16] = [1,16], + [16,1] broadcast
        let tW1 = try g.matmul(tTensor, W1.node())  // [1, 16]

        // Reshape b1 for broadcasting: [16,1] -> [1,16]
        let b1Reshaped = try g.transpose(b1.node())  // [1, 16]
        let hidden_pre = g.n(.add, tW1, b1Reshaped)  // [1, 16]
        let hidden = g.n(.tanh, hidden_pre)  // [1, 16]

        // output = hidden @ W2 + b2  -- [1,16] @ [16,1] = [1,1]
        let hiddenW2 = try g.matmul(hidden, W2.node())  // [1, 1]
        let b2Reshaped = try g.transpose(b2.node())  // [1, 1]
        let envelope_raw = g.n(.add, hiddenW2, b2Reshaped)  // [1, 1]

        // Clamp envelope to [0, 1] using sigmoid-like: 0.5 + 0.5 * tanh(x)
        // Or just use raw value and let it learn
        let envelope = g.n(.sum, envelope_raw)  // Convert [1,1] tensor to scalar

        // Oscillator
        let freq = g.n(.constant(Float(pitchHz)))
        let srFloat = g.n(.constant(Float(sampleRate)))
        let freqNorm = g.n(.div, freq, srFloat)
        let oscCell = g.alloc()
        let oscPhase = g.n(.phasor(oscCell), freqNorm, zero)
        let twoPi = g.n(.constant(Float.pi * 2.0))
        let osc = g.n(.sin, g.n(.mul, twoPi, oscPhase))

        // Synth output
        let synthOutput = g.n(.mul, osc, envelope)

        // Target sample
        let frameIdxCell = g.alloc()
        let frameIdxFreq = g.n(.constant(1.0 / Float(frameCount)))
        let frameIdxNorm = g.n(.phasor(frameIdxCell), frameIdxFreq, zero)
        let frameCountFloat = g.n(.constant(Float(frameCount - 1)))
        let frameIndex = g.n(.mul, frameIdxNorm, frameCountFloat)
        let targetSample = try g.peek(tensor: targetTensor, index: frameIndex, channel: zero)

        // Loss
        let diff = g.n(.sub, synthOutput, targetSample)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        // Compile
        let compileResult = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context
        )

        let ctx = TrainingContext(
            tensorParameters: [W1, b1, W2, b2],
            optimizer: Adam(lr: 0.01),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g
        )
        // Initial loss
        let initialLoss = ctx.runStepGPU()
        print("Initial loss: \(initialLoss)")

        // Train
        let epochs = 200
        var finalLoss = initialLoss
        for epoch in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            if epoch % 50 == 0 || epoch == epochs - 1 {
                print("Epoch \(epoch): loss = \(finalLoss)")
            }
        }
        print("Final loss: \(finalLoss)")

        // Count parameters
        let numParams = hiddenSize + hiddenSize + hiddenSize + 1  // W1 + b1 + W2 + b2
        print("MLP parameters: \(numParams) (vs 64 for lookup table)")

        XCTAssertLessThan(finalLoss, initialLoss * 0.9,
            "MLP should learn (initial: \(initialLoss), final: \(finalLoss))")

        let improvement = (1.0 - finalLoss / initialLoss) * 100
        print("SUCCESS: Neural network learned envelope! Improvement: \(String(format: "%.1f", improvement))%")
    }

    // MARK: - Test 11: Dual Backend

    func testPeekRowDualBackend() throws {
        // Run same computation on both C and Metal backends
        let frameCount = 1

        // Create graph for both backends
        func createGraph() -> (Graph, NodeID) {
            let g = Graph()
            // Column-major data for [3, 4] tensor
            var data = [Float](repeating: 0, count: 12)
            for col in 0..<4 {
                for row in 0..<3 {
                    data[col * 3 + row] = Float(row * 10 + col)
                }
            }
            let tensor = g.tensor(shape: [3, 4], data: data)
            let rowIndex = g.n(.constant(1.5))  // Fractional index
            let result = try! g.peekRow(tensor: tensor, rowIndex: rowIndex)
            let sum = g.n(.sum, result)
            _ = g.n(.output(0), sum)
            return (g, sum)
        }

        // C backend
        let (cGraph, _) = createGraph()
        let cResult = try CompilationPipeline.compile(
            graph: cGraph,
            backend: .c,
            options: .init(frameCount: frameCount)
        )

        let cRuntime = CCompiledKernel(
            source: cResult.source,
            cellAllocations: cResult.cellAllocations,
            memorySize: cResult.totalMemorySlots
        )
        try cRuntime.compileAndLoad()

        guard let cMem = cRuntime.allocateNodeMemory() else {
            XCTFail("Failed to allocate C memory")
            return
        }
        defer { cRuntime.deallocateNodeMemory(cMem) }

        injectTensorData(result: cResult, memory: cMem.assumingMemoryBound(to: Float.self))

        let inputBuffer = [Float](repeating: 0.0, count: frameCount)
        var cOutput = [Float](repeating: 0.0, count: frameCount)

        cOutput.withUnsafeMutableBufferPointer { outPtr in
            inputBuffer.withUnsafeBufferPointer { inPtr in
                cRuntime.runWithMemory(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    memory: cMem,
                    frameCount: frameCount
                )
            }
        }

        // Metal backend
        let (mGraph, _) = createGraph()
        let mResult = try CompilationPipeline.compile(
            graph: mGraph,
            backend: .metal,
            options: .init(frameCount: frameCount)
        )

        let mRuntime = try MetalCompiledKernel(
            kernels: mResult.kernels,
            cellAllocations: mResult.cellAllocations,
            context: mResult.context
        )

        guard let mMemBuffer = mRuntime.getBuffer(name: "memory") else {
            XCTFail("Failed to get Metal memory buffer")
            return
        }
        let mMem = mMemBuffer.contents().assumingMemoryBound(to: Float.self)
        injectTensorData(result: mResult, memory: mMem)

        var mOutput = [Float](repeating: 0.0, count: frameCount)

        mOutput.withUnsafeMutableBufferPointer { outPtr in
            inputBuffer.withUnsafeBufferPointer { inPtr in
                mRuntime.run(
                    outputs: outPtr.baseAddress!,
                    inputs: inPtr.baseAddress!,
                    frameCount: frameCount
                )
            }
        }

        // Both backends should produce the same result
        // Row 1.5 = lerp(row1, row2, 0.5)
        // Row 1 = [10, 11, 12, 13], Row 2 = [20, 21, 22, 23]
        // Result = [15, 16, 17, 18], sum = 66
        let expectedSum: Float = 66.0

        XCTAssertEqual(cOutput[0], expectedSum, accuracy: 0.001, "C backend incorrect")
        XCTAssertEqual(mOutput[0], expectedSum, accuracy: 0.001, "Metal backend incorrect")
        XCTAssertEqual(cOutput[0], mOutput[0], accuracy: 0.001, "Backends should match")
    }
}

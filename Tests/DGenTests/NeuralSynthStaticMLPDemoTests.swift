import Foundation
import XCTest

@testable import DGen

/// Demo-style tests for a DDSP-like harmonic synth driven by a static MLP.
/// This uses a teacher-student setup with a synthetic target so training is stable.
final class NeuralSynthStaticMLPDemoTests: XCTestCase {

    /// Static (control-rate) MLP -> amplitude tensor -> audio-rate harmonic synth.
    /// Trains student weights to match a teacher MLP's synthesized output.
    func testStaticMLP_HarmonicSynth_TeacherStudent() throws {
        let frameCount = 64
        let controlFrames = 16
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0
        let numHarmonics = 6
        let hiddenSize = 8

        let g = Graph(sampleRate: sampleRate)

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

        // Teacher (fixed) weights: structured and intentionally different from student init
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
            let h1b = g.n(.add, h1, b1)
            let h1a = g.n(.tanh, h1b)
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, h2, b2)
            let neg = g.n(.mul, h2b, g.n(.constant(-1.0)))
            let expNeg = g.n(.exp, neg)
            return g.n(.div, one, g.n(.add, one, expNeg))
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
        let playhead = g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames - 1))))

        // Vectorized harmonic synthesis
        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)
        let phasesTensor = g.n(.deterministicPhasor, freqTensor)
        let sinesTensor = g.n(.sin, g.n(.mul, twoPi, phasesTensor))

        let ampsStudentAtTime = try g.peekRow(tensor: ampsStudentT, rowIndex: playhead)
        let ampsTeacherAtTime = try g.peekRow(tensor: ampsTeacherT, rowIndex: playhead)

        let synthStudent = g.n(.sum, g.n(.mul, sinesTensor, ampsStudentAtTime))
        let synthTeacher = g.n(.sum, g.n(.mul, sinesTensor, ampsTeacherAtTime))

        let norm = g.n(.constant(1.0 / Float(numHarmonics)))
        let studentOut = g.n(.mul, synthStudent, norm)
        let teacherOut = g.n(.mul, synthTeacher, norm)

        let diff = g.n(.sub, studentOut, teacherOut)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        writeKernelsToDisk(compileResult, "/tmp/harmonic_teacher_student.metal")

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [studentW1, studentB1, studentW2, studentB2],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let initialLoss = ctx.runStepGPU()

        let epochs = 40
        var finalLoss = initialLoss
        for i in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            print("epoch i=\(i) loss=\(finalLoss*10000)")
        }

        XCTAssertLessThan(
            finalLoss, initialLoss * 0.25, "Loss should drop for teacher-student setup")
    }

    /// Static MLP drives harmonic amplitudes + time-varying noise gain/alpha.
    /// Noise is filtered with a one-pole IIR (history feedback), so this is sequential.
    /// Teacher-student setup keeps the target deterministic and learnable.
    func testStaticMLP_HarmonicPlusNoise_TeacherStudent() throws {
        let frameCount = 64
        let controlFrames = 16
        let sampleRate: Float = 2000.0
        let f0: Float = 100.0
        let numHarmonics = 6
        let hiddenSize = 8

        let g = Graph(sampleRate: sampleRate)

        let timeData = (0..<controlFrames).map { Float($0) / Float(controlFrames - 1) }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        func makeArray(_ count: Int, scale: Float, freq: Float, phase: Float, offset: Float = 0.0)
            -> [Float]
        {
            (0..<count).map { i in
                offset + scale * sin(Float(i) * freq + phase)
            }
        }

        // Teacher weights (harmonics)
        let teacherW1H = g.tensor(
            shape: [1, hiddenSize],
            data: (0..<hiddenSize).map { i in
                let x = Float(i) / Float(max(1, hiddenSize - 1))
                return 1.0 * sin(x * 3.3 * Float.pi) + 0.6 * cos(x * 1.7 * Float.pi)
            })
        let teacherB1H = g.tensor(
            shape: [1, hiddenSize],
            data: (0..<hiddenSize).map { i in
                let x = Float(i) / Float(max(1, hiddenSize - 1))
                return 0.3 * (x - 0.5) + 0.2 * sin(x * 4.0)
            })
        let teacherW2H = g.tensor(
            shape: [hiddenSize, numHarmonics],
            data: (0..<(hiddenSize * numHarmonics)).map { i in
                let row = i / numHarmonics
                let col = i % numHarmonics
                let base = Float(row) * 0.7 + Float(col) * 0.35
                let sign: Float = (row % 2 == 0) ? 1.0 : -1.0
                return sign * (0.9 * sin(base) + 0.4 * cos(base * 1.2))
            })
        let teacherB2H = g.tensor(
            shape: [1, numHarmonics],
            data: (0..<numHarmonics).map { i in
                0.9 / Float(i + 1) + 0.1 * sin(Float(i) * 1.3)
            })

        // Teacher weights (noise gain + alpha)
        let teacherW1N = g.tensor(
            shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.7, freq: 0.8, phase: 0.2))
        let teacherB1N = g.tensor(
            shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.25, freq: 0.5, phase: 0.6))
        let teacherW2N = g.tensor(
            shape: [hiddenSize, 2],
            data: makeArray(hiddenSize * 2, scale: 0.6, freq: 0.33, phase: 0.9))
        let teacherB2N = g.tensor(
            shape: [1, 2],
            data: [0.6, 0.2])

        // Student weights (harmonics)
        let studentW1H = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.12, freq: 0.9, phase: 1.1), name: "W1H")
        let studentB1H = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.05, freq: 0.6, phase: 0.9), name: "b1H")
        let studentW2H = TensorParameter(
            graph: g, shape: [hiddenSize, numHarmonics],
            data: makeArray(hiddenSize * numHarmonics, scale: 0.08, freq: 0.21, phase: 0.7),
            name: "W2H")
        let studentB2H = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: (0..<numHarmonics).map { _ in 0.1 }, name: "b2H")

        // Student weights (noise gain + alpha)
        let studentW1N = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.15, freq: 1.1, phase: 0.4), name: "W1N")
        let studentB1N = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.06, freq: 0.7, phase: 1.2), name: "b1N")
        let studentW2N = TensorParameter(
            graph: g, shape: [hiddenSize, 2],
            data: makeArray(hiddenSize * 2, scale: 0.12, freq: 0.4, phase: 0.3),
            name: "W2N")
        let studentB2N = TensorParameter(
            graph: g, shape: [1, 2],
            data: [0.1, 0.1], name: "b2N")

        func mlpOutputs(time: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let one = g.n(.constant(1.0))
            let h1 = try g.matmul(time, W1)
            let h1b = g.n(.add, h1, b1)
            let h1a = g.n(.tanh, h1b)
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, h2, b2)
            let neg = g.n(.mul, h2b, g.n(.constant(-1.0)))
            let expNeg = g.n(.exp, neg)
            return g.n(.div, one, g.n(.add, one, expNeg))
        }

        let ampsStudent = try mlpOutputs(
            time: timeTensor, W1: studentW1H.node(), b1: studentB1H.node(),
            W2: studentW2H.node(), b2: studentB2H.node())
        let ampsTeacher = try mlpOutputs(
            time: timeTensor, W1: teacherW1H, b1: teacherB1H, W2: teacherW2H, b2: teacherB2H)

        let noiseStudent = try mlpOutputs(
            time: timeTensor, W1: studentW1N.node(), b1: studentB1N.node(),
            W2: studentW2N.node(), b2: studentB2N.node())
        let noiseTeacher = try mlpOutputs(
            time: timeTensor, W1: teacherW1N, b1: teacherB1N, W2: teacherW2N, b2: teacherB2N)

        // Reshape and transpose for peekRow access (see peekRow column-major layout)
        let ampsStudentView = try g.reshape(ampsStudent, to: [numHarmonics, controlFrames])
        let ampsTeacherView = try g.reshape(ampsTeacher, to: [numHarmonics, controlFrames])
        let ampsStudentT = try g.transpose(ampsStudentView, axes: [1, 0])
        let ampsTeacherT = try g.transpose(ampsTeacherView, axes: [1, 0])

        let noiseStudentView = try g.reshape(noiseStudent, to: [2, controlFrames])
        let noiseTeacherView = try g.reshape(noiseTeacher, to: [2, controlFrames])
        let noiseStudentT = try g.transpose(noiseStudentView, axes: [1, 0])  // [controlFrames, 2]
        let noiseTeacherT = try g.transpose(noiseTeacherView, axes: [1, 0])

        let zero = g.n(.constant(0.0))
        let one = g.n(.constant(1.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let frameIdx = g.phasor(
            freq: g.n(.constant(sampleRate / Float(frameCount))), reset: zero)
        let playhead = g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames - 1))))

        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)
        let phasesTensor = g.n(.deterministicPhasor, freqTensor)
        let sinesTensor = g.n(.sin, g.n(.mul, twoPi, phasesTensor))

        let ampsStudentAtTime = try g.peekRow(tensor: ampsStudentT, rowIndex: playhead)
        let ampsTeacherAtTime = try g.peekRow(tensor: ampsTeacherT, rowIndex: playhead)

        let synthStudent = g.n(.sum, g.n(.mul, sinesTensor, ampsStudentAtTime))
        let synthTeacher = g.n(.sum, g.n(.mul, sinesTensor, ampsTeacherAtTime))

        let norm = g.n(.constant(1.0 / Float(numHarmonics)))
        let harmonicStudent = g.n(.mul, synthStudent, norm)
        let harmonicTeacher = g.n(.mul, synthTeacher, norm)

        // Shared noise source (deterministic sequence)
        let noiseCell = g.alloc()
        let noise = g.n(.noise(noiseCell))

        let idx0 = g.n(.constant(0.0))
        let idx1 = g.n(.constant(1.0))
        let noiseGainStudentRaw = try g.peek(tensor: noiseStudentT, index: playhead, channel: idx0)
        let noiseAlphaStudentRaw = try g.peek(tensor: noiseStudentT, index: playhead, channel: idx1)
        let noiseGainTeacherRaw = try g.peek(tensor: noiseTeacherT, index: playhead, channel: idx0)
        let noiseAlphaTeacherRaw = try g.peek(tensor: noiseTeacherT, index: playhead, channel: idx1)

        // Scale gains, clamp alpha to [0.05, 0.95]
        let gainScale = g.n(.constant(0.4))
        let noiseGainStudent = g.n(.mul, noiseGainStudentRaw, gainScale)
        let noiseGainTeacher = g.n(.mul, noiseGainTeacherRaw, gainScale)

        let alphaMin = g.n(.constant(0.05))
        let alphaMax = g.n(.constant(0.95))
        let alphaStudent = g.n(.min, alphaMax, g.n(.max, alphaMin, noiseAlphaStudentRaw))
        let alphaTeacher = g.n(.min, alphaMax, g.n(.max, alphaMin, noiseAlphaTeacherRaw))

        func onePole(_ input: NodeID, _ alpha: NodeID, _ cellId: CellID) -> NodeID {
            let history = g.n(.historyRead(cellId))
            let mixed = g.n(.mix, input, history, alpha)
            _ = g.n(.historyWrite(cellId), mixed)
            return mixed
        }

        let noiseStudentIn = g.n(.mul, noise, noiseGainStudent)
        let noiseTeacherIn = g.n(.mul, noise, noiseGainTeacher)

        let noiseStudentFiltered = onePole(noiseStudentIn, alphaStudent, g.alloc())
        let noiseTeacherFiltered = onePole(noiseTeacherIn, alphaTeacher, g.alloc())

        let studentOut = g.n(.add, harmonicStudent, noiseStudentFiltered)
        let teacherOut = g.n(.add, harmonicTeacher, noiseTeacherFiltered)

        let diff = g.n(.sub, studentOut, teacherOut)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [
                studentW1H, studentB1H, studentW2H, studentB2H,
                studentW1N, studentB1N, studentW2N, studentB2N,
            ],
            optimizer: Adam(lr: 0.05),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let initialLoss = ctx.runStepGPU()

        let epochs = 60
        var finalLoss = initialLoss
        for i in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            print("epoch i=\(i) loss=\(finalLoss*10000)")
        }

        XCTAssertLessThan(
            finalLoss, initialLoss * 0.35,
            "Loss should drop for harmonic+noise teacher-student setup")
    }

    /// 32-harmonic band-limited saw target with a smooth envelope.
    /// Student MLP learns time-varying harmonic amplitudes to match the target.
    func testStaticMLP_BandlimitedSawWithEnvelope() throws {
        let frameCount = 64
        let controlFrames = 16
        let sampleRate: Float = 8000.0
        let f0: Float = 100.0
        let numHarmonics = 32
        let hiddenSize = 12

        let g = Graph(sampleRate: sampleRate)

        let timeData = (0..<controlFrames).map { Float($0) / Float(controlFrames - 1) }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        func makeArray(_ count: Int, scale: Float, freq: Float, phase: Float, offset: Float = 0.0)
            -> [Float]
        {
            (0..<count).map { i in
                offset + scale * sin(Float(i) * freq + phase)
            }
        }

        // Student MLP weights (learnable)
        let studentW1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.1, freq: 0.8, phase: 1.0), name: "W1")
        let studentB1 = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.05, freq: 0.5, phase: 0.7), name: "b1")
        let studentW2 = TensorParameter(
            graph: g, shape: [hiddenSize, numHarmonics],
            data: makeArray(hiddenSize * numHarmonics, scale: 0.08, freq: 0.2, phase: 0.4),
            name: "W2")
        let studentB2 = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: (0..<numHarmonics).map { _ in 0.0 }, name: "b2")

        // Signed output MLP: tanh allows negative amplitudes for saw harmonics
        func mlpAmplitudesSigned(time: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let h1 = try g.matmul(time, W1)
            let h1b = g.n(.add, h1, b1)
            let h1a = g.n(.tanh, h1b)
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, h2, b2)
            let amps = g.n(.tanh, h2b)
            return g.n(.mul, amps, g.n(.constant(0.9)))
        }

        let ampsStudent = try mlpAmplitudesSigned(
            time: timeTensor, W1: studentW1.node(), b1: studentB1.node(),
            W2: studentW2.node(), b2: studentB2.node())

        // Band-limited saw harmonic weights: 2/pi * (-1)^(k+1) / k
        let harmonicWeights = (0..<numHarmonics).map { k -> Float in
            let n = Float(k + 1)
            let sign: Float = (k % 2 == 0) ? 1.0 : -1.0
            return (2.0 / Float.pi) * (sign / n)
        }
        let weightTensor = g.tensor(shape: [1, numHarmonics], data: harmonicWeights)

        // Smooth envelope (attack-decay) at control rate
        let one = g.n(.constant(1.0))
        let attackRate = g.n(.constant(8.0))
        let decayRate = g.n(.constant(3.0))
        let negAttack = g.n(.mul, g.n(.constant(-1.0)), g.n(.mul, timeTensor, attackRate))
        let negDecay = g.n(.mul, g.n(.constant(-1.0)), g.n(.mul, timeTensor, decayRate))
        let attackPart = g.n(.sub, one, g.n(.exp, negAttack))
        let decayPart = g.n(.exp, negDecay)
        let envelope = g.n(.mul, attackPart, decayPart)  // [controlFrames, 1]

        // Target amplitudes: envelope * saw harmonic weights
        let ampsTarget = try g.matmul(envelope, weightTensor)  // [controlFrames, numHarmonics]

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let frameIdx = g.phasor(
            freq: g.n(.constant(sampleRate / Float(frameCount))), reset: zero)
        let playhead = g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames - 1))))

        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)
        let phasesTensor = g.n(.deterministicPhasor, freqTensor)
        let sinesTensor = g.n(.sin, g.n(.mul, twoPi, phasesTensor))

        let ampsStudentAtTime = try g.peekRow(tensor: ampsStudent, rowIndex: playhead)
        let ampsTargetAtTime = try g.peekRow(tensor: ampsTarget, rowIndex: playhead)

        let synthStudent = g.n(.sum, g.n(.mul, sinesTensor, ampsStudentAtTime))
        let synthTarget = g.n(.sum, g.n(.mul, sinesTensor, ampsTargetAtTime))

        let diff = g.n(.sub, synthStudent, synthTarget)
        let loss = g.n(.mul, diff, diff)
        _ = g.n(.output(0), loss)

        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [studentW1, studentB1, studentW2, studentB2],
            optimizer: Adam(lr: 0.03),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let initialLoss = ctx.runStepGPU()

        let epochs = 80
        var finalLoss = initialLoss
        for i in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            print("epoch i=\(i) loss=\(finalLoss*10000)")
        }

        XCTAssertLessThan(finalLoss, initialLoss * 0.4, "Should learn band-limited saw target")
    }

    /// Real piano sample target using 32 harmonics + filtered noise.
    /// This is a demo-style test: we only expect modest improvement.
    func testStaticMLP_PianoTarget_HarmonicPlusNoise() throws {
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

        let frameCount = 256
        let controlFrames = 32
        let windowSize = 64
        let numHarmonics = 32
        let hiddenSize = 12

        let g = Graph(sampleRate: Float(sampleRate))

        let targetData = Array(targetSamples.prefix(frameCount))
        let targetTensor = g.tensor(shape: [frameCount, 1], data: targetData)

        let timeData = (0..<controlFrames).map { Float($0) / Float(controlFrames - 1) }
        let timeTensor = g.tensor(shape: [controlFrames, 1], data: timeData)

        func makeArray(_ count: Int, scale: Float, freq: Float, phase: Float, offset: Float = 0.0)
            -> [Float]
        {
            (0..<count).map { i in
                offset + scale * sin(Float(i) * freq + phase)
            }
        }

        let W1H = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.1, freq: 0.8, phase: 0.9), name: "W1H")
        let b1H = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.05, freq: 0.5, phase: 0.2), name: "b1H")
        let W2H = TensorParameter(
            graph: g, shape: [hiddenSize, numHarmonics],
            data: makeArray(hiddenSize * numHarmonics, scale: 0.08, freq: 0.2, phase: 0.3),
            name: "W2H")
        let b2H = TensorParameter(
            graph: g, shape: [1, numHarmonics],
            data: (0..<numHarmonics).map { _ in 0.0 }, name: "b2H")

        let W1N = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.12, freq: 0.9, phase: 1.1), name: "W1N")
        let b1N = TensorParameter(
            graph: g, shape: [1, hiddenSize],
            data: makeArray(hiddenSize, scale: 0.06, freq: 0.6, phase: 0.4), name: "b1N")
        let W2N = TensorParameter(
            graph: g, shape: [hiddenSize, 2],
            data: makeArray(hiddenSize * 2, scale: 0.1, freq: 0.35, phase: 0.8), name: "W2N")
        let b2N = TensorParameter(
            graph: g, shape: [1, 2],
            data: [0.1, 0.1], name: "b2N")

        func mlpAmps(time: NodeID, W1: NodeID, b1: NodeID, W2: NodeID, b2: NodeID)
            throws -> NodeID
        {
            let one = g.n(.constant(1.0))
            let h1 = try g.matmul(time, W1)
            let h1b = g.n(.add, h1, b1)
            let h1a = g.n(.tanh, h1b)
            let h2 = try g.matmul(h1a, W2)
            let h2b = g.n(.add, h2, b2)
            let neg = g.n(.mul, h2b, g.n(.constant(-1.0)))
            let expNeg = g.n(.exp, neg)
            return g.n(.div, one, g.n(.add, one, expNeg))
        }

        let ampsTensor = try mlpAmps(
            time: timeTensor, W1: W1H.node(), b1: b1H.node(), W2: W2H.node(), b2: b2H.node())
        let noiseTensor = try mlpAmps(
            time: timeTensor, W1: W1N.node(), b1: b1N.node(), W2: W2N.node(), b2: b2N.node())

        let zero = g.n(.constant(0.0))
        let twoPi = g.n(.constant(Float.pi * 2.0))

        let frameIdx = g.phasor(
            freq: g.n(.constant(Float(sampleRate) / Float(frameCount))), reset: zero)
        let playhead = g.n(.mul, frameIdx, g.n(.constant(Float(controlFrames - 1))))
        let audioIdx = g.n(.mul, frameIdx, g.n(.constant(Float(frameCount - 1))))

        let freqData = (1...numHarmonics).map { f0 * Float($0) }
        let freqTensor = g.tensor(shape: [numHarmonics], data: freqData)
        let phasesTensor = g.n(.deterministicPhasor, freqTensor)
        let sinesTensor = g.n(.sin, g.n(.mul, twoPi, phasesTensor))

        let ampsAtTime = try g.peekRow(tensor: ampsTensor, rowIndex: playhead)
        let synthHarmonics = g.n(.sum, g.n(.mul, sinesTensor, ampsAtTime))
        let harmonicOut = g.n(.mul, synthHarmonics, g.n(.constant(1.0 / Float(numHarmonics))))

        // Noise branch (shared noise source)
        let noiseCell = g.alloc()
        let noise = g.n(.noise(noiseCell))
        let idx0 = g.n(.constant(0.0))
        let idx1 = g.n(.constant(1.0))
        let noiseGainRaw = try g.peek(tensor: noiseTensor, index: playhead, channel: idx0)
        let noiseAlphaRaw = try g.peek(tensor: noiseTensor, index: playhead, channel: idx1)

        let gainScale = g.n(.constant(0.35))
        let noiseGain = g.n(.mul, noiseGainRaw, gainScale)

        let alphaMin = g.n(.constant(0.05))
        let alphaMax = g.n(.constant(0.95))
        let alpha = g.n(.min, alphaMax, g.n(.max, alphaMin, noiseAlphaRaw))

        func onePole(_ input: NodeID, _ alpha: NodeID, _ cellId: CellID) -> NodeID {
            let history = g.n(.historyRead(cellId))
            let mixed = g.n(.mix, input, history, alpha)
            _ = g.n(.historyWrite(cellId), mixed)
            return mixed
        }

        let noiseFiltered = onePole(g.n(.mul, noise, noiseGain), alpha, g.alloc())
        let synthOutput = g.n(.add, harmonicOut, noiseFiltered)

        let targetSample = try g.peek(tensor: targetTensor, index: audioIdx, channel: zero)
        let spectralLoss = g.spectralLoss(synthOutput, targetSample, windowSize: windowSize)
        let diff = g.n(.sub, synthOutput, targetSample)
        let mse = g.n(.mul, diff, diff)
        let loss = g.n(.add, spectralLoss, g.n(.mul, mse, g.n(.constant(0.05))))
        _ = g.n(.output(0), loss)

        let compileResult = try CompilationPipeline.compile(
            graph: g, backend: .metal,
            options: .init(frameCount: frameCount, backwards: true))

        let runtime = try MetalCompiledKernel(
            kernels: compileResult.kernels,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context)

        let ctx = TrainingContext(
            tensorParameters: [W1H, b1H, W2H, b2H, W1N, b1N, W2N, b2N],
            optimizer: Adam(lr: 0.02),
            lossNode: loss)

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: compileResult.cellAllocations,
            context: compileResult.context,
            frameCount: frameCount,
            graph: g)

        let initialLoss = ctx.runStepGPU()

        let epochs = 20
        var finalLoss = initialLoss
        for i in 0..<epochs {
            finalLoss = ctx.runStepGPU()
            print("epoch i=\(i) loss=\(finalLoss*10000)")
        }

        XCTAssertLessThan(finalLoss, initialLoss * 0.9, "Should show some improvement")
    }
}

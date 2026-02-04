import XCTest

@testable import DGenLazy

/// Single-layer linear -> peekRow -> Harmonic Synth teacher-student test using DGenLazy frontend
/// Uses single-layer to avoid overparameterization issues that prevent convergence to zero loss
final class MLPHarmonicSynthTests: XCTestCase {

  override func setUp() {
    super.setUp()
    // IMPORTANT: Set config BEFORE reset() - LazyGraph captures sampleRate at creation
    DGenConfig.sampleRate = 2000.0
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  // MARK: - Linear Layer Helper

  /// Single-layer linear: y = x @ W + b
  /// Input: [batchSize, 1], Output: [batchSize, outputSize]
  /// This avoids overparameterization - unique solution exists!
  func linearForward(
    input: Tensor,
    W: Tensor,  // [1, outputSize]
    b: Tensor   // [1, outputSize]
  ) -> Tensor {
    return input.matmul(W) + b
  }

  // MARK: - Full Test

  func testMLPPeekRowHarmonicSynth() throws {
    DGenConfig.kernelOutputPath = "/tmp/mlp_peek_row_harmonic_sync.metal"
    DGenConfig.debug = true
    let frameCount = 64
    let controlFrames = 16
    let numHarmonics = 16

    // ============================================
    // TEACHER WEIGHTS (fixed, target to match)
    // Single layer: y = time @ W + b
    // ============================================
    let teacherWData = (0..<numHarmonics).map { i in
      let x = Float(i) / Float(max(1, numHarmonics - 1))
      return 1.1 * Foundation.sin(x * 3.1 * Float.pi) + 0.5 * Foundation.cos(x * 2.3 * Float.pi)
    }
    let teacherBData = (0..<numHarmonics).map { i in
      let inv = 0.8 / Float(i + 1)
      let wiggle = 0.1 * Foundation.sin(Float(i) * 1.7)
      return inv + wiggle
    }

    // Create teacher tensors
    let teacherW = Tensor([teacherWData])  // [1, numHarmonics]
    let teacherB = Tensor([teacherBData])  // [1, numHarmonics]

    // ============================================
    // STUDENT WEIGHTS (learnable, start different)
    // 50% perturbation from teacher
    // ============================================
    let perturbation: Float = 0.5
    let studentW = Tensor.param(
      [1, numHarmonics],
      data: teacherWData.map { $0 * (1.0 + perturbation) })
    let studentB = Tensor.param(
      [1, numHarmonics],
      data: teacherBData.map { $0 + perturbation * 0.1 })

    // ============================================
    // HELPER: Build forward pass and compute loss
    // ============================================
    // This function rebuilds the graph each iteration (tinygrad-style)
    func buildLoss() -> (SignalTensor, Signal) {
      // Time tensor [controlFrames, 1]: each row is a single time value 0..1
      let timeRows = (0..<controlFrames).map { [Float($0) / Float(controlFrames - 1)] }
      let timeTensor = Tensor(timeRows)  // [controlFrames, 1]

      // Single-layer forward pass (unique solution - no overparameterization!)
      let ampsStudent = linearForward(input: timeTensor, W: studentW, b: studentB)
      let ampsTeacher = linearForward(input: timeTensor, W: teacherW, b: teacherB)

      // Reshape for peekRow [controlFrames, numHarmonics]
      let ampsStudentT = ampsStudent.reshape([controlFrames, numHarmonics])
      let ampsTeacherT = ampsTeacher.reshape([controlFrames, numHarmonics])

      // Audio-rate playhead that sweeps through control frames
      let playheadFreq = DGenConfig.sampleRate / Float(frameCount)
      let playhead = Signal.phasor(playheadFreq) * Float(controlFrames - 1)

      // Dynamic amplitude lookup via peekRow (linear interpolation)
      let ampsStudentAtTime = ampsStudentT.peekRow(playhead)
      let ampsTeacherAtTime = ampsTeacherT.peekRow(playhead)

      // MSE loss directly on amplitudes (cleaner than spectral loss)
      // This tests: linear layer → peekRow → MSE
      let diff = ampsStudentAtTime - ampsTeacherAtTime
      let loss = (diff * diff).sum() * (1.0 / Float(numHarmonics))

      return (ampsStudentAtTime, loss)
    }

    // ============================================
    // TRAINING LOOP
    // ============================================
    print("\n=== Single-Layer Linear -> peekRow -> Harmonic Synth (DGenLazy) ===")
    print("frameCount: \(frameCount), controlFrames: \(controlFrames)")
    print("numHarmonics: \(numHarmonics)")
    let totalParams = numHarmonics * 2  // W + b
    print("Total learnable params: \(totalParams)")

    // Create optimizer - MSE loss produces larger gradients, need lower LR
    let optimizer = SGD(
      params: [studentW, studentB],
      lr: 0.001  // Lower LR for MSE loss with peekRow
    )

    // Warmup - builds graph fresh
    _ = try buildLoss().1.backward(frames: frameCount)
    optimizer.zeroGrad()
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = nil

    // Get initial loss - builds graph fresh again
    let initialLossValues = try buildLoss().1.backward(frames: frameCount)
    let initialLoss = initialLossValues.reduce(0, +) / Float(frameCount)
    optimizer.zeroGrad()
    print("Initial loss: \(initialLoss)")

    // Training
    let epochs = 30
    var finalLoss = initialLoss
    var minLoss = initialLoss
    var minEpoch = 0
    var lastWGradNorm: Float = 0

    let trainStart = CFAbsoluteTimeGetCurrent()
    for epoch in 0..<epochs {
      // Rebuild forward graph fresh each iteration (tinygrad-style)
      let (_, loss) = buildLoss()
      let lossValues = try loss.backward(frames: frameCount)
      let epochLoss = lossValues.reduce(0, +) / Float(frameCount)

      finalLoss = epochLoss
      if epochLoss < minLoss {
        minLoss = epochLoss
        minEpoch = epoch
      }

      // Capture gradient norms BEFORE zeroGrad clears them
      lastWGradNorm = studentW.grad?.getData()?.map { $0 * $0 }.reduce(0, +) ?? 0

      // Log progress
      if epoch % 20 == 0 || epoch == epochs - 1 {
        print(
          "Epoch \(epoch): loss = \(String(format: "%.6f", epochLoss)), "
            + "W grad norm² = \(String(format: "%.2e", lastWGradNorm))")
      }

      // Update weights
      optimizer.step()
      optimizer.zeroGrad()

      // Early stopping if converged to very low loss
      if epochLoss < initialLoss * 0.01 {
        print("Converged at epoch \(epoch)")
        break
      }
    }
    let trainTime = (CFAbsoluteTimeGetCurrent() - trainStart) * 1000

    print("\nFinal loss: \(String(format: "%.6f", finalLoss))")
    print("Min loss: \(String(format: "%.6f", minLoss)) at epoch \(minEpoch)")
    print("Loss reduction: \(String(format: "%.2fx", initialLoss / minLoss))")
    print("Training time: \(String(format: "%.2f", trainTime))ms for \(epochs) epochs")
    print("Time per epoch: \(String(format: "%.3f", trainTime / Double(epochs)))ms")

    // Compare weights to verify convergence
    let teacherWVals = teacherWData
    let studentWVals = studentW.getData() ?? []
    let maxWDiff = zip(teacherWVals, studentWVals).map { abs($0 - $1) }.max() ?? Float.infinity
    print("\nWeight convergence:")
    print("Max W diff from teacher: \(String(format: "%.4f", maxWDiff))")

    // Verify loss decreased significantly (single layer should achieve >10x reduction)
    XCTAssertLessThan(
      minLoss, initialLoss * 0.1,
      "Single-layer should achieve >10x loss reduction")

    // Verify gradients flowed through
    XCTAssertGreaterThan(lastWGradNorm, 0, "W should have non-zero gradients")

    // Verify weights converged close to teacher (single layer has unique solution)
    XCTAssertLessThan(maxWDiff, 0.5, "Weights should converge close to teacher")
  }

  /// Test linear forward pass verification
  func testLinearForwardPass() throws {
    let batchSize = 4
    let outputSize = 3

    // Input: [batchSize, 1] - each row is one input sample
    let input = Tensor([[0.0], [0.25], [0.5], [1.0]])

    // Weights
    let W = Tensor([[1.0, 0.5, -0.5]])  // [1, outputSize]
    let b = Tensor([[0.1, 0.2, 0.3]])   // [1, outputSize]

    // Forward: y = x @ W + b
    let output = linearForward(input: input, W: W, b: b)

    XCTAssertEqual(output.shape, [batchSize, outputSize])

    // Realize and check
    let result = try output.realize()
    XCTAssertEqual(result.count, batchSize * outputSize)

    print("Linear output shape: \(output.shape)")
    print("Linear output values: \(result)")

    // Verify expected values:
    // row 0: 0.0 * [1, 0.5, -0.5] + [0.1, 0.2, 0.3] = [0.1, 0.2, 0.3]
    // row 1: 0.25 * [1, 0.5, -0.5] + [0.1, 0.2, 0.3] = [0.35, 0.325, 0.175]
    XCTAssertEqual(result[0], 0.1, accuracy: 0.001)
    XCTAssertEqual(result[1], 0.2, accuracy: 0.001)
    XCTAssertEqual(result[2], 0.3, accuracy: 0.001)
  }

  /// Test peekRow with linear output
  func testLinearWithPeekRow() throws {
    let controlFrames = 4
    let outputSize = 3
    let frameCount = 16

    // Create a simple amplitude tensor [controlFrames, outputSize]
    let amplitudes = Tensor([
      [0.1, 0.2, 0.3],  // t=0
      [0.4, 0.5, 0.6],  // t=0.33
      [0.7, 0.8, 0.9],  // t=0.67
      [1.0, 1.1, 1.2],  // t=1
    ])

    // Playhead cycles through control frames
    let playhead = Signal.phasor(Float(1000) / Float(frameCount)) * Float(controlFrames - 1)

    // Get amplitude at playhead
    let ampsAtTime = amplitudes.peekRow(playhead)

    XCTAssertEqual(ampsAtTime.shape, [outputSize])

    // Sum to scalar for loss
    let summed = ampsAtTime.sum()
    let target: Float = 1.5
    let loss = (summed - target) * (summed - target)

    // Run backward
    let lossValues = try loss.backward(frames: frameCount)

    XCTAssertEqual(lossValues.count, frameCount)
    print("Loss values (first 4): \(Array(lossValues.prefix(4)))")
  }
}

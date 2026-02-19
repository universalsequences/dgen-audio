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
    b: Tensor  // [1, outputSize]
  ) -> Tensor {
    return input.matmul(W) + b
  }

  // MARK: - Full Test

  /// Test linear forward pass verification
  func testLinearForwardPass() throws {
    let batchSize = 4
    let outputSize = 3

    // Input: [batchSize, 1] - each row is one input sample
    let input = Tensor([[0.0], [0.25], [0.5], [1.0]])

    // Weights
    let W = Tensor([[1.0, 0.5, -0.5]])  // [1, outputSize]
    let b = Tensor([[0.1, 0.2, 0.3]])  // [1, outputSize]

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

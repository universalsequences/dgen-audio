import XCTest
import Foundation

@testable import DGenLazy

final class SampleTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 2000.0
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  // MARK: - Forward Tests

  func testSample2DForwardIntegerIndex() throws {
    // 2D tensor [4, 3], sample at integer index 2 → should return row 2
    let data: [[Float]] = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]
    let tensor = Tensor(data)
    let index = Signal.constant(2.0)
    let sampled = tensor.sample(index)  // should be [7, 8, 9]
    let output = sampled.sum()

    let frameCount = 1
    let result = try output.backward(frames: frameCount)
    // sum of row 2 = 7 + 8 + 9 = 24
    XCTAssertEqual(result[0], 24.0, accuracy: 0.01, "Expected sum of row 2 = 24")
  }

  func testSample2DForwardFractionalIndex() throws {
    // Fractional index 1.5 → lerp between row 1 and row 2
    let data: [[Float]] = [
      [0, 0, 0],
      [2, 4, 6],
      [10, 20, 30],
      [0, 0, 0],
    ]
    let tensor = Tensor(data)
    let index = Signal.constant(1.5)
    let sampled = tensor.sample(index)
    // Expected: 0.5 * [2,4,6] + 0.5 * [10,20,30] = [6, 12, 18]
    let output = sampled.sum()

    let frameCount = 1
    let result = try output.backward(frames: frameCount)
    // sum = 6 + 12 + 18 = 36
    XCTAssertEqual(result[0], 36.0, accuracy: 0.01, "Expected lerp sum = 36")
  }

  func testSample3DForward() throws {
    // 3D tensor [3, 2, 2], sample at integer index 1 → [2, 2]
    // Data: slice 0 = [1,2,3,4], slice 1 = [5,6,7,8], slice 2 = [9,10,11,12]
    let tensor = Tensor.param([3, 2, 2], data: [
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
    ])
    let index = Signal.constant(1.0)
    let sampled = tensor.sample(index)  // should be [5,6,7,8] as [2,2]
    let output = sampled.sum()

    let frameCount = 1
    let result = try output.backward(frames: frameCount)
    // sum = 5 + 6 + 7 + 8 = 26
    XCTAssertEqual(result[0], 26.0, accuracy: 0.01, "Expected sum of slice 1 = 26")
  }

  func testSample2DMatchesPeekRow() throws {
    // Verify sample() produces same results as peekRow() for 2D tensors
    let data: [[Float]] = [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
    ]
    let frameCount = 64

    // Run with sample()
    LazyGraphContext.reset()
    let tensor1 = Tensor(data)
    let playhead1 = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 2.0
    let sampled1 = tensor1.sample(playhead1)
    let output1 = sampled1.sum()
    let result1 = try output1.backward(frames: frameCount)

    // Run with peekRow()
    LazyGraphContext.reset()
    let tensor2 = Tensor(data)
    let playhead2 = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 2.0
    let sampled2 = tensor2.peekRow(playhead2)
    let output2 = sampled2.sum()
    let result2 = try output2.backward(frames: frameCount)

    // Compare frame-by-frame
    for i in 0..<frameCount {
      XCTAssertEqual(
        result1[i], result2[i], accuracy: 0.01,
        "sample() and peekRow() differ at frame \(i)")
    }
  }

  // MARK: - Backward Tests

  func testSample2DBackward() throws {
    // Learnable 2D tensor, verify gradients are non-zero
    let frameCount = 64
    let param = Tensor.param([4, 3], data: [
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12,
    ])
    let target = Signal.constant(5.0)
    let playhead = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 3.0
    let sampled = param.sample(playhead)
    let loss = mse(sampled.sum(), target)

    _ = try loss.backward(frames: frameCount)

    let grad = param.grad?.getData() ?? []
    XCTAssertEqual(grad.count, 12, "Expected 12 gradient values for [4,3] tensor")
    let gradNorm = grad.map { $0 * $0 }.reduce(0, +)
    XCTAssertGreaterThan(gradNorm, 0.0, "Expected non-zero gradients")
  }

  func testSample2DBackwardMatchesPeekRowBackward() throws {
    let frameCount = 64

    // Run with sample()
    LazyGraphContext.reset()
    let param1 = Tensor.param([4, 3], data: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ])
    let playhead1 = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 3.0
    let sampled1 = param1.sample(playhead1)
    let loss1 = mse(sampled1.sum(), Signal.constant(5.0))
    _ = try loss1.backward(frames: frameCount)
    let grad1 = param1.grad?.getData() ?? []

    // Run with peekRow()
    LazyGraphContext.reset()
    let param2 = Tensor.param([4, 3], data: [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ])
    let playhead2 = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 3.0
    let sampled2 = param2.peekRow(playhead2)
    let loss2 = mse(sampled2.sum(), Signal.constant(5.0))
    _ = try loss2.backward(frames: frameCount)
    let grad2 = param2.grad?.getData() ?? []

    XCTAssertEqual(grad1.count, grad2.count, "Gradient sizes should match")
    for i in 0..<grad1.count {
      XCTAssertEqual(
        grad1[i], grad2[i], accuracy: 0.01,
        "sample() and peekRow() gradients differ at index \(i)")
    }
  }

  func testSample3DBackward() throws {
    // Learnable 3D tensor [R, B, H], sample at signal, verify gradients flow
    let frameCount = 64
    let param = Tensor.param([3, 2, 2], data: [
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
    ])
    let target = Signal.constant(3.0)
    let playhead = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 2.0
    let sampled = param.sample(playhead)  // [2, 2]
    let loss = mse(sampled.sum(), target)

    _ = try loss.backward(frames: frameCount)

    let grad = param.grad?.getData() ?? []
    XCTAssertEqual(grad.count, 12, "Expected 12 gradient values for [3,2,2] tensor")
    let gradNorm = grad.map { $0 * $0 }.reduce(0, +)
    XCTAssertGreaterThan(gradNorm, 0.0, "Expected non-zero gradients for 3D sample")
  }

  // MARK: - Training Convergence

  func testSampleTrainingConverges() throws {
    let frameCount = 64

    let param = Tensor.param([4, 3], data: [
      0.5, 0.5, 0.5,
      0.5, 0.5, 0.5,
      0.5, 0.5, 0.5,
      0.5, 0.5, 0.5,
    ])
    let optimizer = SGD(params: [param], lr: 0.01)

    func buildLoss() -> Signal {
      let playhead = Signal.phasor(DGenConfig.sampleRate / Float(frameCount)) * 3.0
      let sampled = param.sample(playhead)
      return mse(sampled.sum(), Signal.constant(2.0))
    }

    var losses: [Float] = []
    for _ in 0..<5 {
      let result = try buildLoss().backward(frames: frameCount)
      let avgLoss = result.reduce(0, +) / Float(frameCount)
      losses.append(avgLoss)

      optimizer.step()
      optimizer.zeroGrad()
    }

    // Loss should decrease
    XCTAssertLessThan(
      losses.last!, losses.first!,
      "Expected loss to decrease during training: \(losses)")
  }
}

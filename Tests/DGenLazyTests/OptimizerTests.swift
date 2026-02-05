import DGen
import XCTest

@testable import DGenLazy

final class OptimizerTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Signal SGD

  func testSignalSGDBasic() throws {
    // Create a learnable parameter starting at 5.0
    let param = Signal.param(5.0)
    XCTAssertEqual(param.data, 5.0, "Initial param should be 5.0")

    // Simple loss: (param - 2)^2, minimum at param=2
    let target = Signal.constant(2.0)
    let diff = param - target
    let loss = diff * diff

    // Backward pass
    try loss.backward(frames: 1)

    // Expected gradient: d/dparam (param-2)^2 = 2*(param-2) = 2*(5-2) = 6
    XCTAssertNotNil(param.grad, "Gradient should be computed")
    XCTAssertEqual(param.grad?.data ?? -999, 6.0, accuracy: 0.01, "Gradient should be 6.0")

    // SGD step with lr=0.1
    let optimizer = SGD(params: [param], lr: 0.1)
    optimizer.step()

    // Expected: 5.0 - 0.1 * 6 = 4.4
    XCTAssertEqual(param.data ?? -999, 4.4, accuracy: 0.01, "After step param should be 4.4")
  }

  func testSignalSGDTrainingLoop() throws {
    // Full training loop: minimize (param - 2)^2
    let param = Signal.param(10.0)
    let optimizer = SGD(params: [param], lr: 0.1)

    var losses: [Float] = []

    for i in 0..<100 {
      // Build loss graph (tinygrad style - rebuild each iteration)
      let target = Signal.constant(2.0)
      let diff = param - target
      let loss = diff * diff

      // Backward compiles, runs, and populates .grad - returns loss values
      let lossValue = try loss.backward(frames: 1).first ?? 0
      losses.append(lossValue)
      print("epoch \(i) loss=\(lossValue)")

      // Update parameters
      optimizer.step()
      optimizer.zeroGrad()
    }

    // Verify loss decreases significantly (started at (10-2)^2 = 64)
    XCTAssertGreaterThan(losses[0], losses[9], "Loss should decrease over training")
    XCTAssertLessThan(losses[9], losses[0] * 0.1, "Final loss should be <10% of initial")

    // Verify param moved toward target=2.0 (started at 10.0)
    let finalParam = param.data ?? -999
    XCTAssertLessThan(
      abs(finalParam - 2.0), abs(10.0 - 2.0), "Param should be closer to 2.0 than start")
  }

  func testTensorSGDTrainingLoop() throws {
    // Full training loop: minimize sum((w - target)^2)
    let w = Tensor([5.0, 8.0, -3.0], requiresGrad: true)
    let target = Tensor([0.0, 0.0, 0.0])
    let optimizer = SGD(params: [w], lr: 0.1)

    var losses: [Float] = []

    for i in 0..<100 {
      // Build loss graph (tinygrad style)
      let diff = w - target
      let loss = (diff * diff).sum()
      let diffValue = try diff.realize()

      // Backward compiles, runs, populates .grad - returns loss values
      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      losses.append(lossValue)
      print(
        "epoch \(i) loss =\(lossValue) graph.node.count=\(LazyGraphContext.current.graph.nodes.count) diff=\(diffValue)"
      )

      // Update and clear
      optimizer.step()
      optimizer.zeroGrad()
    }

    // Verify loss decreases
    XCTAssertGreaterThan(losses[0], losses[9], "Loss should decrease: \(losses)")
    XCTAssertLessThan(losses[9], losses[0] * 0.5, "Loss should decrease significantly")
  }

  // MARK: - Tensor SGD

  func testTensorSGDBasic() throws {
    // Create a learnable tensor
    let w = Tensor([1.0, 2.0, 3.0], requiresGrad: true)

    // Loss: sum((w - target)^2)
    let target = Tensor([0.0, 0.0, 0.0])
    let diff = w - target
    let loss = (diff * diff).sum()

    // Backward (frameCount=1 to avoid accumulating across multiple frames)
    try loss.backward(frameCount: 1)

    // Gradients: d/dw sum(w^2) = 2*w = [2, 4, 6]
    XCTAssertNotNil(w.grad, "Gradient should be computed")
    let grads = w.grad?.getData()
    XCTAssertEqual(grads?[0] ?? -999, 2.0, accuracy: 0.01)
    XCTAssertEqual(grads?[1] ?? -999, 4.0, accuracy: 0.01)
    XCTAssertEqual(grads?[2] ?? -999, 6.0, accuracy: 0.01)

    // SGD step
    let optimizer = SGD(params: [w], lr: 0.1)
    optimizer.step()

    // Expected: [1-0.2, 2-0.4, 3-0.6] = [0.8, 1.6, 2.4]
    let newData = w.getData()
    XCTAssertEqual(newData?[0] ?? -999, 0.8, accuracy: 0.01)
    XCTAssertEqual(newData?[1] ?? -999, 1.6, accuracy: 0.01)
    XCTAssertEqual(newData?[2] ?? -999, 2.4, accuracy: 0.01)
  }

  // MARK: - Adam Optimizer

  func testAdamBasic() throws {
    // Create a learnable tensor
    let w = Tensor([1.0, 2.0, 3.0], requiresGrad: true)

    // Loss: sum((w - target)^2)
    let target = Tensor([0.0, 0.0, 0.0])
    let diff = w - target
    let loss = (diff * diff).sum()

    // Backward
    try loss.backward(frameCount: 1)

    // Verify gradients exist
    XCTAssertNotNil(w.grad, "Gradient should be computed")

    // Adam step
    let optimizer = Adam(params: [w], lr: 0.1)
    optimizer.step()

    // Adam should move params toward zero (the target)
    let newData = w.getData()!
    XCTAssertLessThan(newData[0], 1.0, "w[0] should decrease")
    XCTAssertLessThan(newData[1], 2.0, "w[1] should decrease")
    XCTAssertLessThan(newData[2], 3.0, "w[2] should decrease")
  }

  func testAdamTrainingLoop() throws {
    // Full training loop: minimize sum((w - target)^2)
    let w = Tensor([5.0, 8.0, -3.0], requiresGrad: true)
    let target = Tensor([0.0, 0.0, 0.0])
    let optimizer = Adam(params: [w], lr: 0.5)

    var losses: [Float] = []

    for i in 0..<50 {
      // Build loss graph
      let diff = w - target
      let loss = (diff * diff).sum()

      // Backward
      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      losses.append(lossValue)

      if i % 10 == 0 {
        print("Adam epoch \(i): loss=\(lossValue)")
      }

      // Update and clear
      optimizer.step()
      optimizer.zeroGrad()
    }

    // Verify loss decreases significantly
    XCTAssertGreaterThan(losses[0], losses[49], "Loss should decrease")
    XCTAssertLessThan(losses[49], losses[0] * 0.1, "Final loss should be <10% of initial")

    // Verify params moved toward target
    let finalData = w.getData()!
    for (i, val) in finalData.enumerated() {
      XCTAssertLessThan(abs(val), 1.0, "w[\(i)] should be close to 0")
    }
  }

  func testAdamVsSGDConvergence() throws {
    // Compare Adam vs SGD on same problem

    // Train with SGD
    LazyGraphContext.reset()
    let wSGD = Tensor([5.0, 8.0, -3.0], requiresGrad: true)
    let sgd = SGD(params: [wSGD], lr: 0.1)
    var sgdLosses: [Float] = []

    for _ in 0..<30 {
      let target = Tensor([0.0, 0.0, 0.0])
      let diff = wSGD - target
      let loss = (diff * diff).sum()
      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      sgdLosses.append(lossValue)
      sgd.step()
      sgd.zeroGrad()
    }

    // Train with Adam
    LazyGraphContext.reset()
    let wAdam = Tensor([5.0, 8.0, -3.0], requiresGrad: true)
    let adam = Adam(params: [wAdam], lr: 0.1)
    var adamLosses: [Float] = []

    for _ in 0..<30 {
      let target = Tensor([0.0, 0.0, 0.0])
      let diff = wAdam - target
      let loss = (diff * diff).sum()
      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      adamLosses.append(lossValue)
      adam.step()
      adam.zeroGrad()
    }

    print("SGD final loss: \(sgdLosses.last!)")
    print("Adam final loss: \(adamLosses.last!)")

    // Both should converge
    XCTAssertLessThan(sgdLosses.last!, sgdLosses.first! * 0.5, "SGD should converge")
    XCTAssertLessThan(adamLosses.last!, adamLosses.first! * 0.5, "Adam should converge")
  }
}

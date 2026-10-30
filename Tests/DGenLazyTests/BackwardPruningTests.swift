import DGen
import XCTest

@testable import DGenLazy

/// Backward pruning: a history no target reaches carries no gradient, so the reverse walk skips
/// it (no BPTT carry, no frame-sequential backward block), and the remaining gradients are
/// unchanged. Grad cells stay out of buffer reuse whether or not the graph has a recursion.
final class BackwardPruningTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
  }

  /// y[n] = b·x[n] − a·y[n−1]
  private func iir(_ x: Signal, b: Signal, a: Signal) -> Signal {
    let (prev, write) = Signal.history()
    return write(b * x - a * prev)
  }

  /// Carry cells created when computing gradients of `loss` w.r.t. `params`.
  private func carryCount(loss: Signal, params: [Signal]) -> Int {
    let graph = LazyGraphContext.current.graph
    _ = graph.computeGradients(loss: loss.nodeId, targets: Set(params.map(\.nodeId)))
    return graph.gradCarryCells.count
  }

  /// A fixed filter over a fixed input, scaled by a learnable gain: the filter's state does not
  /// depend on the gain, so no carry is created.
  func testFixedFilterOverFixedInputHasNoCarry() {
    let g = Signal.param(0.7)
    let y = g * iir(Signal.phasor(440.0), b: Signal.constant(0.3), a: Signal.constant(-0.6))
    XCTAssertEqual(carryCount(loss: mse(y, Signal.constant(0.1)), params: [g]), 0)
  }

  /// A target that reaches a filter's input only through a comparison (zero derivative) does not
  /// make the filter's state relevant.
  func testHistoryReachedOnlyThroughComparisonHasNoCarry() {
    let g = Signal.param(0.7)
    let p = Signal.param(0.5)
    let impulses = gswitch(Signal.phasor(440.0) < p, 1.0, 0.0)
    let y = g * iir(impulses, b: Signal.constant(0.3), a: Signal.constant(-0.6))
    XCTAssertEqual(carryCount(loss: mse(y, Signal.constant(0.1)), params: [g, p]), 0)
  }

  /// A learnable coefficient inside the recursion keeps the carry.
  func testLearnableCoefficientKeepsCarry() {
    let a = Signal.param(-0.6)
    let y = iir(Signal.phasor(440.0), b: Signal.constant(0.3), a: a)
    XCTAssertEqual(carryCount(loss: mse(y, Signal.constant(0.1)), params: [a]), 1)
  }

  /// The gain's gradient through a pruned fixed filter still matches finite differences.
  func testGainGradientThroughFixedFilterMatchesFiniteDifference() throws {
    let frames = 512
    func run(_ gain: Float) throws -> (Float, Float?) {
      LazyGraphContext.reset()
      let g = Signal.param(gain)
      let y = g * iir(Signal.phasor(440.0), b: Signal.constant(0.3), a: Signal.constant(-0.6))
      let loss = try mse(y, Signal.constant(0.1)).backward(frames: frames).reduce(0, +)
      return (loss, g.grad?.data)
    }
    let (_, ad) = try run(0.7)
    let h: Float = 1e-3
    let fd = (try run(0.7 + h).0 - run(0.7 - h).0) / (2 * h)
    let grad = try XCTUnwrap(ad)
    XCTAssertEqual(grad, fd, accuracy: max(1e-4, abs(fd) * 1e-2))
  }

  /// A param whose only route to the loss is a comparison is cut from the walk, but still gets a
  /// gradient of zero (not nil), as before pruning.
  func testParamReachedOnlyThroughComparisonGetsZeroGradient() throws {
    let p = Signal.param(0.5)
    let g = Signal.param(0.7)
    let y = g * gswitch(Signal.phasor(440.0) < p, 1.0, 0.0)
    _ = try mse(y, Signal.constant(0.1)).backward(frames: 256)
    XCTAssertEqual(try XCTUnwrap(p.grad?.data), 0)
    XCTAssertNotEqual(try XCTUnwrap(g.grad?.data), 0)
  }

  /// Grad cells are read by the host after the run, so they are persistent even in a graph
  /// without tensor histories, where buffer reuse would otherwise hand their memory to a later
  /// block.
  func testTensorGradCellsArePersistentWithoutRecursion() {
    let w1 = DGenLazy.Tensor([0.1, 0.2, 0.3], requiresGrad: true)
    let w2 = DGenLazy.Tensor([0.4, 0.5, 0.6], requiresGrad: true)
    let loss = (w1 * w2).sum()
    let graph = LazyGraphContext.current
    let (tensorCells, _) = graph.setupGradients(loss: loss.nodeId, frameCount: 1)
    XCTAssertEqual(tensorCells.count, 2)
    for cell in tensorCells.values {
      XCTAssertTrue(graph.graph.persistentCells.contains(cell))
    }
  }

  func testScalarGradCellsArePersistentWithoutRecursion() {
    let a = Signal.param(0.3)
    let b = Signal.param(0.6)
    let graph = LazyGraphContext.current
    let (_, signalCells) = graph.setupGradients(loss: (a * b).nodeId, frameCount: 1)
    XCTAssertEqual(signalCells.count, 2)
    for cell in signalCells.values {
      XCTAssertTrue(graph.graph.persistentCells.contains(cell))
    }
  }
}

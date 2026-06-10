import XCTest

@testable import DGen
@testable import DGenLazy

final class GatherTests: XCTestCase {
  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    LazyGraphContext.reset()
  }

  func testTensorGatherUsesIndexTensorShape() throws {
    let source = Tensor([10, 20, 30, 40, 50])
    let indices = Tensor([4, 4, 2, 0])

    let gathered = source.gather(indices)

    XCTAssertEqual(gathered.shape, [4])
    XCTAssertEqual(try gathered.realize(), [50, 50, 30, 10])
  }

  func testTensorGatherClampsOutOfRangeIndices() throws {
    let source = Tensor([10, 20, 30])
    let indices = Tensor([-2, 0, 2, 99])

    XCTAssertEqual(try source.gather(indices).realize(), [10, 10, 30, 30])
  }

  func testSignalTensorGatherReadsPerFrameSourceWithStaticIndices() throws {
    let source = Tensor([10, 20, 30, 40, 50]) * Signal.constant(2)
    let indices = Tensor([3, 1])

    let out = source.gather(indices).sum()

    XCTAssertEqual(try out.realize(frames: 4), [120, 120, 120, 120])
  }
}

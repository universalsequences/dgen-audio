import XCTest

@testable import DGen

final class HopIslandPassTests: XCTestCase {
  func testAdjacentSameHopBlocksBecomeIsland() {
    let blocks = [
      block(.hopBased(hopSize: 8, counterNode: 100)),
      block(.hopBased(hopSize: 8, counterNode: 200)),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .hopIsland(HopIsland(domain: HopDomain(hopSize: 8), blockIndices: [0, 1])),
      ]
    )
  }

  func testDifferentHopSizesSplitIslands() {
    let blocks = [
      block(.hopBased(hopSize: 8, counterNode: 100)),
      block(.hopBased(hopSize: 16, counterNode: 100)),
      block(.hopBased(hopSize: 16, counterNode: 200)),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .block(0),
        .hopIsland(HopIsland(domain: HopDomain(hopSize: 16), blockIndices: [1, 2])),
      ]
    )
  }

  func testIndependentFrameBlocksCanBeScheduledBeforeDeferredIsland() {
    let blocks = [
      block(.frameBased),
      block(.hopBased(hopSize: 8, counterNode: 100)),
      block(.frameBased),
      block(.hopBased(hopSize: 8, counterNode: 200)),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .block(0),
        .block(2),
        .hopIsland(HopIsland(domain: HopDomain(hopSize: 8), blockIndices: [1, 3])),
      ]
    )
  }

  func testDependentFrameBlockIsCarriedInsideIsland() {
    let produced = Lazy.variable(10, nil)
    let blocks = [
      block(
        .hopBased(hopSize: 8, counterNode: 100),
        ops: [UOp(op: .identity(.constant(0, 1.0)), value: produced)]
      ),
      block(
        .frameBased,
        ops: [UOp(op: .identity(produced), value: .variable(11, nil))]
      ),
      block(.hopBased(hopSize: 8, counterNode: 200)),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .hopIsland(HopIsland(domain: HopDomain(hopSize: 8), blockIndices: [0, 1, 2])),
      ]
    )
  }

  func testStaticBlocksSplitIslands() {
    let blocks = [
      block(.hopBased(hopSize: 8, counterNode: 100)),
      block(.static_),
      block(.hopBased(hopSize: 8, counterNode: 200)),
      block(.hopBased(hopSize: 8, counterNode: 300)),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .block(0),
        .block(1),
        .hopIsland(HopIsland(domain: HopDomain(hopSize: 8), blockIndices: [2, 3])),
      ]
    )
  }

  func testSingletonHopBlockRemainsNormalBlock() {
    let blocks = [
      block(.frameBased),
      block(.hopBased(hopSize: 8, counterNode: 100)),
      block(.frameBased),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .block(0),
        .block(1),
        .block(2),
      ]
    )
  }

  func testDisallowedDispatchModeSplitsIsland() {
    let blocks = [
      block(.hopBased(hopSize: 8, counterNode: 100)),
      block(.hopBased(hopSize: 8, counterNode: 200), dispatchMode: .perFrameScaled(4)),
      block(.hopBased(hopSize: 8, counterNode: 300)),
      block(.hopBased(hopSize: 8, counterNode: 400)),
    ]

    XCTAssertEqual(
      HopIslandPass.buildRegions(for: blocks),
      [
        .block(0),
        .block(1),
        .hopIsland(HopIsland(domain: HopDomain(hopSize: 8), blockIndices: [2, 3])),
      ]
    )
  }

  private func block(
    _ temporality: Temporality,
    dispatchMode: DispatchMode = .singleThreaded,
    ops: [UOp] = []
  ) -> BlockUOps {
    BlockUOps(
      ops: ops,
      frameOrder: .sequential,
      vectorWidth: 1,
      temporality: temporality,
      dispatchMode: dispatchMode
    )
  }
}

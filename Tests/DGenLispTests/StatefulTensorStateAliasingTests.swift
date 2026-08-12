import XCTest

@testable import DGenLisp
import DGen
import DGenLazy

/// Regression coverage for stateful tensor cells being aliased onto tensor temporaries
/// by the buffer-reuse memory planner.
///
/// The planner's liveness pass (`remapVectorMemorySlots`) can only inspect UOps, where it
/// recognizes `load`/`store`/`delay1`/`noise` as persistent state. Stateful ops that lower
/// through the TENSOR path — `.phasor`/`.accum` under a `tensorIndex` — emit plain
/// `memoryRead`/`memoryWrite`, indistinguishable from transient tensor traffic. Those state
/// cells were therefore reuse-eligible and got aliased onto a tensor temporary.
///
/// Concretely, for `(phasor (* 110 (pow 2 (/ t 12))))` the per-element phase accumulator was
/// assigned the same physical region as the `/ 12` intermediate. The intermediate is recomputed
/// in a per-`process()` preamble, so every block start stomped the phase back to `t/12` and the
/// oscillators restarted each block (audible buzz at sr/blockSize).
final class StatefulTensorStateAliasingTests: XCTestCase {
  private var savedBackend: Backend!
  private var savedSampleRate: Float!
  private var savedMaxFrameCount: Int!

  override func setUpWithError() throws {
    try super.setUpWithError()
    savedBackend = DGenConfig.backend
    savedSampleRate = DGenConfig.sampleRate
    savedMaxFrameCount = DGenConfig.maxFrameCount
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48_000
    DGenConfig.maxFrameCount = 64
    LazyGraphContext.reset()
  }

  override func tearDownWithError() throws {
    DGenConfig.backend = savedBackend
    DGenConfig.sampleRate = savedSampleRate
    DGenConfig.maxFrameCount = savedMaxFrameCount
    LazyGraphContext.reset()
    try super.tearDownWithError()
  }

  /// The reported repro: arithmetic on a constant tensor feeding a tensor phasor.
  func testTensorPhasorStateIsDisjointFromTensorTemporaries() throws {
    try assertStateCellsDisjointFromTensorBuffers(
      source: """
      (def t (tensor @shape [2 2] @data [0 4 9 14]))
      (out (* 0.25 (sum (cos (* twopi (phasor (* 110 (pow 2 (/ t 12)))))))) 1 @name left)
      """,
      voiceCount: 12
    )
  }

  /// The explicit `stateful-phasor` spelling must be protected identically.
  func testStatefulPhasorSpellingStateIsDisjointFromTensorTemporaries() throws {
    try assertStateCellsDisjointFromTensorBuffers(
      source: """
      (def t (tensor @shape [2 2] @data [0 4 9 14]))
      (out (* 0.25 (sum (cos (* twopi (stateful-phasor (* 110 (pow 2 (/ t 12)))))))) 1 @name left)
      """,
      voiceCount: 1
    )
  }

  /// A tensor phasor with no arithmetic in front of it (the shape that always worked)
  /// must keep the same guarantee.
  func testConstantFrequencyTensorPhasorStateIsDisjointFromTensorTemporaries() throws {
    try assertStateCellsDisjointFromTensorBuffers(
      source: """
      (out (* 0.2 (sum (cos (* twopi (phasor (tensor @shape [2 2] @data [90 90 90 90])))))) 1 @name left)
      """,
      voiceCount: 12
    )
  }

  // MARK: - Helper

  /// Compiles `source` and asserts every persistent state cell owns a physical memory
  /// range that no materialized tensor buffer overlaps.
  private func assertStateCellsDisjointFromTensorBuffers(
    source: String,
    voiceCount: Int,
    file: StaticString = #filePath,
    line: UInt = #line
  ) throws {
    LazyGraphContext.reset()
    let evaluator = LispEvaluator()
    try evaluator.evaluate(nodes: parseSource(source))

    let graph = LazyGraphContext.current
    for output in evaluator.outputs {
      graph.addOutput(output.signal, channel: output.channel)
    }
    let compilation = try graph.compileOnly(frameCount: 64, voiceCount: voiceCount)

    let g = compilation.graph
    let mappings = compilation.cellAllocations.cellMappings
    let widths = compilation.cellAllocations.cellVectorWidths

    /// Physical [start, end) for a logical cell, matching the planner's alignment padding.
    func physicalRange(_ cellId: CellID) -> Range<Int>? {
      guard let base = mappings[cellId] else { return nil }
      let size = g.cellAllocationSizes[cellId] ?? 1
      let allocSize = (widths[cellId] ?? 1) > 1 ? max(4, size) : size
      return base..<(base + allocSize)
    }

    // Every cell any stateful op persists across frames.
    var stateCells: [CellID: String] = [:]
    for nodeId in g.nodes.keys.sorted() {
      guard let node = g.nodes[nodeId] else { continue }
      for cellId in node.op.persistentStateCellIds {
        stateCells[cellId] = "\(node.op)"
      }
    }
    XCTAssertFalse(
      stateCells.isEmpty,
      "expected at least one persistent state cell (the phasor accumulator)",
      file: file, line: line)

    // Every materialized tensor buffer: op outputs, intermediates and injected constants.
    var tensorCells: [CellID: TensorID] = [:]
    for tensorId in g.tensors.keys.sorted() {
      guard let tensor = g.tensors[tensorId], !tensor.isLazy else { continue }
      tensorCells[tensor.cellId] = tensorId
    }

    for (stateCell, opDescription) in stateCells.sorted(by: { $0.key < $1.key }) {
      guard let stateRange = physicalRange(stateCell) else { continue }
      for (tensorCell, tensorId) in tensorCells.sorted(by: { $0.key < $1.key }) {
        if tensorCell == stateCell { continue }
        guard let tensorRange = physicalRange(tensorCell) else { continue }
        XCTAssertTrue(
          stateRange.upperBound <= tensorRange.lowerBound
            || tensorRange.upperBound <= stateRange.lowerBound,
          """
          state cell \(stateCell) (\(opDescription)) at physical \(stateRange) overlaps \
          tensor \(tensorId) cell \(tensorCell) at physical \(tensorRange). \
          The tensor buffer is rewritten every process() call and would clobber the state.
          """,
          file: file, line: line)
      }
    }
  }
}

import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// Regression tests for the C-backend codegen passes that keep per-sample
/// loops lean: dead-code elimination, static (frame-invariant) hoisting, and
/// scalar block coalescing. Each pass is checked for exactness against its
/// own disabled variant (`DGEN_NO_*` toggles) on the same program, plus one
/// structural assertion about the emitted C.
final class CodegenPerfPassesTests: XCTestCase {
  private var tempDir: URL!
  private let toggles = ["DGEN_NO_DCE", "DGEN_NO_STATIC_HOIST", "DGEN_NO_COALESCE"]
  private var savedConfig: (backend: Backend, sampleRate: Float, maxFrameCount: Int)!

  override func setUpWithError() throws {
    try super.setUpWithError()
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 64
    for key in toggles { unsetenv(key) }
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-perfpass-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    for key in toggles { unsetenv(key) }
    // Other suites read these globals without setting them; leave no trace.
    DGenConfig.backend = savedConfig.backend
    DGenConfig.sampleRate = savedConfig.sampleRate
    DGenConfig.maxFrameCount = savedConfig.maxFrameCount
    LazyGraphContext.reset()
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private struct Compiled {
    let result: CompilationResult
    let source: String
    let paramCells: [String: Int]
    let paramDefaults: [Int: Float]
    let nodeCount: Int
  }

  private func compile(_ source: String, blockSize: Int = 64) throws -> Compiled {
    DGenConfig.maxFrameCount = blockSize
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: lowerModulation(in: parseSource(source)))
    for output in evaluator.outputs {
      lazy.addOutput(output.signal, channel: output.channel)
    }
    let result = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .c, options: .init(frameCount: blockSize, debug: false))
    var cells: [String: Int] = [:]
    var defaults: [Int: Float] = [:]
    for param in evaluator.params {
      if let logical = param.cellId {
        let physical = result.cellAllocations.cellMappings[logical] ?? logical
        cells[param.name] = physical
        defaults[physical] = param.defaultValue
      }
    }
    return Compiled(
      result: result,
      source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
      paramCells: cells,
      paramDefaults: defaults,
      nodeCount: lazy.graph.nodes.count)
  }

  /// Renders `blocks` blocks of `blockSize` frames, calling `beforeBlock` with
  /// the memory pointer before each so tests can move parameters mid-stream.
  private func render(
    _ compiled: Compiled, blockSize: Int = 64, blocks: Int = 4,
    sampleRate: Float? = nil,
    beforeBlock: (Int, UnsafeMutablePointer<Float>) -> Void = { _, _ in }
  ) throws -> [Float] {
    let runtime = try CLazyRuntime(
      kernels: compiled.result.kernels,
      cellAllocations: compiled.result.cellAllocations,
      memorySize: compiled.result.totalMemorySlots,
      frameCount: blockSize,
      defaultHostSampleRate: sampleRate ?? DGenConfig.sampleRate)
    runtime.zeroAllBuffers()
    guard let mem = runtime.memoryPointer(), let out = runtime.outputsPointer() else {
      XCTFail("runtime has no memory/output pointers")
      return []
    }
    DGen.injectTensorData(result: compiled.result, memory: mem)
    for (cell, value) in compiled.paramDefaults { mem[cell] = value }
    var samples: [Float] = []
    for b in 0..<blocks {
      beforeBlock(b, mem)
      runtime.runNoCopy(frameCount: blockSize)
      for i in 0..<blockSize { samples.append(out[i]) }
    }
    return samples
  }

  private func frameLoopCount(_ source: String) -> Int {
    source.components(separatedBy: "for (int i = 0; i < frameCount").count - 1
  }

  // MARK: - Dead-code elimination

  func testUnusedModulatedParamsAreRemovedAndParamCellsSurvive() throws {
    // Multi-channel inputs cannot be driven by the in-process runtime, so this
    // test checks the graph and the emitted C only; exactness is covered below.
    let source = """
      (def mod1 (in 5 @name mod1 @modulator 1))
      (def mod2 (in 6 @name mod2 @modulator 2))
      (param used @default 0.5 @min 0 @max 1 @mod true @mod-mode additive)
      (param unused_a @default 0.25 @min 0 @max 1 @mod true @mod-mode additive)
      (param unused_b @default 0.75 @min 0 @max 1 @mod true @mod-mode additive)
      (out (* (mod used) (phasor 100)) 1)
      """
    let optimized = try compile(source)
    setenv("DGEN_NO_DCE", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_NO_DCE")

    // Pruning is scoped to one compile and the lazy graph is restored after it,
    // so the evidence is in the emitted C, not the graph's node count.
    XCTAssertEqual(optimized.nodeCount, reference.nodeCount, "graph must be restored after compile")
    XCTAssertLessThan(optimized.source.count, reference.source.count, "DCE removed nothing")
    // Parameter cells survive DCE untouched: the host addresses them by cell id.
    XCTAssertEqual(optimized.paramCells, reference.paramCells)
    // Input reads are now shared graph nodes. Check emitted work rather than
    // counting repeated textual channel loads from the old opaque mod operator.
    XCTAssertLessThan(optimized.result.uopBlocks.reduce(0) { $0 + $1.ops.count },
                      reference.result.uopBlocks.reduce(0) { $0 + $1.ops.count })
  }

  func testDeadPureSubgraphRemovalIsExact() throws {
    let source = """
      (param g @default 0.5 @min 0 @max 1)
      (def junk (exp (sin (* 3 (phasor 300)))))
      (def junk2 (+ junk (* g 2)))
      (out (* g (phasor 100)) 1)
      """
    let optimized = try compile(source)
    setenv("DGEN_NO_DCE", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_NO_DCE")
    XCTAssertFalse(optimized.source.contains("expf("), "dead exp survived DCE")
    XCTAssertTrue(reference.source.contains("expf("), "reference build should still carry the dead exp")
    XCTAssertEqual(try render(optimized), try render(reference))
    XCTAssertGreaterThan(try render(optimized).map { abs($0) }.max() ?? 0, 1e-6)
  }

  // MARK: - Static hoisting

  func testConstantTensorMaskIsHoistedAwayFromFrameCounter() throws {
    // A scalar history makes block formation conservatively mark the tensor
    // arithmetic sequential too. Without tensor hoisting the mask shares the
    // clock's frame loop, exactly as the causal Filter Table's cepstral mask.
    let source = """
      (def idx (iota 16))
      (def mask (+ (+ (eq idx 0) (eq idx 8))
                   (* 2 (* (gte idx 1) (lte idx 7)))))
      (make-history clock)
      (def counter (write-history clock (% (+ (read-history clock) 1) 8)))
      (out (sum (* mask (+ counter 1))) 1)
      """
    let optimized = try compile(source)
    setenv("DGEN_NO_STATIC_HOIST", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_NO_STATIC_HOIST")

    let comparisons = optimized.result.graph.nodes.values.filter { node in
      guard case .tensor = node.shape else { return false }
      switch node.op {
      case .eq, .gte, .lte: return true
      default: return false
      }
    }
    XCTAssertEqual(comparisons.count, 4)
    for node in comparisons {
      let block = try XCTUnwrap(optimized.result.blocks.first { $0.nodes.contains(node.id) })
      XCTAssertEqual(block.temporality, .static_, "constant mask inherited the counter's rate")
    }
    let got = try render(optimized)
    XCTAssertEqual(got, try render(reference))
    XCTAssertEqual(Set(got), Set((1...8).map { Float($0 * 16) }))
  }

  func testHoistedTensorBroadcastFollowsHostUpdates() throws {
    let source = """
      (param gain @default 1 @min 0 @max 4)
      (def rows (+ (* (tensor @shape [2 1] @data [1 2]) gain) (/ samplerate 1024)))
      (def weights (+ rows (iota 4)))
      (make-history clock)
      (def counter (write-history clock (% (+ (read-history clock) 1) 8)))
      (out (sum (* weights (+ counter 1))) 1)
      """
    let optimized = try compile(source)
    setenv("DGEN_NO_STATIC_HOIST", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_NO_STATIC_HOIST")
    let cell = try XCTUnwrap(optimized.paramCells["gain"])
    let refCell = try XCTUnwrap(reference.paramCells["gain"])
    func tableCell(_ compiled: Compiled) throws -> Int {
      let tensor = try XCTUnwrap(compiled.result.graph.tensors.values.first {
        $0.shape == [2, 1] && $0.data == [1, 2]
      })
      return try XCTUnwrap(compiled.result.cellAllocations.cellMappings[tensor.cellId])
    }
    let table = try tableCell(optimized)
    let refTable = try tableCell(reference)
    for rate: Float in [32_000, 48_000] {
      let got = try render(optimized, sampleRate: rate) { block, memory in
        memory[cell] = block < 2 ? 1 : 3
        if block == 3 { memory[table] = 2; memory[table + 1] = 4 }
      }
      let want = try render(reference, sampleRate: rate) { block, memory in
        memory[refCell] = block < 2 ? 1 : 3
        if block == 3 { memory[refTable] = 2; memory[refTable + 1] = 4 }
      }
      XCTAssertEqual(got, want)
      for index in got.indices {
        let gain: Float = index < 128 ? 1 : 3
        let tableScale: Float = index < 192 ? 1 : 2
        let weightSum = 12 * gain * tableScale + 8 * rate / 1024 + 12
        XCTAssertEqual(got[index], weightSum * Float((index + 1) % 8 + 1))
      }
    }
  }

  func testFrameVaryingTensorComparisonIsNotHoisted() throws {
    let source = """
      (make-history clock)
      (def counter (write-history clock (% (+ (read-history clock) 1) 16)))
      (def mask (eq (iota 16) counter))
      (out (sum (* mask (+ counter 1))) 1)
      """
    let optimized = try compile(source)
    let comparison = try XCTUnwrap(optimized.result.graph.nodes.values.first {
      if case .eq = $0.op { return true }
      return false
    })
    let block = try XCTUnwrap(optimized.result.blocks.first { $0.nodes.contains(comparison.id) })
    XCTAssertEqual(block.temporality, .frameBased)
    let got = try render(optimized)
    XCTAssertEqual(got, (0..<256).map { Float(($0 + 1) % 16 + 1) })
    let single = try render(try compile(source, blockSize: 1), blockSize: 1, blocks: 256)
    XCTAssertEqual(got, single)
  }

  func testFrameInvariantMathIsHoistedAndFollowsParamChanges() throws {
    let source = """
      (param gain_db @default 0 @min -60 @max 12)
      (out (* (exp (* 0.11512925465 gain_db)) (phasor 100)) 1)
      """
    let optimized = try compile(source)
    setenv("DGEN_NO_STATIC_HOIST", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_NO_STATIC_HOIST")

    // The exponential is emitted before the first frame loop when hoisted.
    let firstLoop = optimized.source.range(of: "for (int i = 0; i < frameCount")
    let firstExp = optimized.source.range(of: "expf(")
    XCTAssertNotNil(firstLoop)
    XCTAssertNotNil(firstExp)
    if let firstLoop, let firstExp {
      XCTAssertLessThan(firstExp.lowerBound, firstLoop.lowerBound, "expf still inside a frame loop")
    }

    // A static block runs once per process call, so a parameter moved between
    // calls must take effect on the next block exactly as in the unhoisted build.
    let cell = try XCTUnwrap(optimized.paramCells["gain_db"])
    let refCell = try XCTUnwrap(reference.paramCells["gain_db"])
    let move: (Int, UnsafeMutablePointer<Float>) -> Void = { block, mem in
      mem[cell] = block < 2 ? -6 : 6
    }
    let moveRef: (Int, UnsafeMutablePointer<Float>) -> Void = { block, mem in
      mem[refCell] = block < 2 ? -6 : 6
    }
    let got = try render(optimized, beforeBlock: move)
    let want = try render(reference, beforeBlock: moveRef)
    XCTAssertEqual(got, want)
    let early = got[0..<128].map { abs($0) }.max() ?? 0
    let late = got[128..<256].map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(late, early * 2, "hoisted gain did not follow the parameter change")
  }

  // MARK: - Scalar block coalescing

  func testCircularWindowWrapsMatchIntegerHistory() throws {
    // Exercise all ring positions with complete SIMD groups. Partial groups
    // already lose buffer samples at HEAD and are tracked in dgen-j6r.
    for blockSize in [4, 8, 64] {
      for window in [1, 3, 17, 64, 256] {
        let source = """
          (make-history clock)
          (def value (write-history clock (+ (read-history clock) 1)))
          (def window (reshape (buffer value \(window)) @shape [\(window)]))
          (out (sum (+ window (iota \(window)))) 1)
          """
        let compiled = try compile(source, blockSize: blockSize)
        // Enough calls to cross the persistent ring boundary several times,
        // including windows larger than the processing block and size one.
        let blocks = (4 * (blockSize + window)) / blockSize + 1
        let got = try render(compiled, blockSize: blockSize, blocks: blocks)
        for index in got.indices {
          let frame = index + 1
          let active = min(frame, window)
          let historySum = active * (2 * frame - active + 1) / 2
          XCTAssertEqual(got[index], Float(historySum + window * (window - 1) / 2),
            "block \(blockSize), window \(window), frame \(index)")
        }
      }
    }
  }

  func testChainedSmoothersCoalesceIntoOneLoopExactly() throws {
    let source = """
      (defmacro smooth (target)
        (make-history v)
        (def old (read-history v))
        (def value (+ old (* 0.01 (- target old))))
        (write-history v value)
        value)
      (param a @default 1 @min 0 @max 4)
      (param b @default 2 @min 0 @max 4)
      (param c @default 3 @min 0 @max 4)
      (out (+ (smooth a) (* 2 (smooth b)) (smooth (* c (phasor 50)))) 1)
      """
    let optimized = try compile(source)
    setenv("DGEN_NO_COALESCE", "1", 1)
    let reference = try compile(source)
    unsetenv("DGEN_NO_COALESCE")

    XCTAssertLessThan(
      frameLoopCount(optimized.source), frameLoopCount(reference.source),
      "coalescing did not reduce the loop count")
    // Fusing loops changes clang's fast-math contraction choices, so allow
    // ulp-level drift; the miscompiles this guards against are off by >1e-2.
    let got = try render(optimized)
    let want = try render(reference)
    XCTAssertEqual(got.count, want.count)
    XCTAssertLessThan(zip(got, want).map { abs($0 - $1) }.max() ?? 0, 1e-5)
    XCTAssertGreaterThan(got.map { abs($0) }.max() ?? 0, 1e-6)
  }

  func testCoalescedProgramIsBlockSizeInvariant() throws {
    let source = """
      (defmacro smooth (target)
        (make-history v)
        (def old (read-history v))
        (def value (+ old (* 0.05 (- target old))))
        (write-history v value)
        value)
      (out (smooth (* 3 (smooth (phasor 200)))) 1)
      """
    let truth = try render(try compile(source, blockSize: 1), blockSize: 1, blocks: 256)
    let wide = try render(try compile(source, blockSize: 64), blockSize: 64, blocks: 4)
    XCTAssertEqual(truth.count, wide.count)
    let maxDiff = zip(truth, wide).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThan(maxDiff, 1e-5)
  }
}

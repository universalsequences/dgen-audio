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

  override func setUpWithError() throws {
    try super.setUpWithError()
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
    beforeBlock: (Int, UnsafeMutablePointer<Float>) -> Void = { _, _ in }
  ) throws -> [Float] {
    let runtime = try CLazyRuntime(
      kernels: compiled.result.kernels,
      cellAllocations: compiled.result.cellAllocations,
      memorySize: compiled.result.totalMemorySlots,
      frameCount: blockSize,
      defaultHostSampleRate: DGenConfig.sampleRate)
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
    // The unused destinations' modulation sums (input channel times depth) are gone.
    let modReads = { (src: String) in src.components(separatedBy: "in[5][").count - 1 }
    XCTAssertLessThan(modReads(optimized.source), modReads(reference.source))
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

import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// `(mod p)` lowers to explicit arithmetic plus a `gswitch` on the C backend
/// (`ModulationGateLoweringPass`), which reshapes scheduling around it: the
/// resolved parameter becomes a per-frame value instead of one opaque node.
/// With no modulator routed the value is bit-identical to the plain parameter,
/// so the rendered audio must be too — any difference is a scheduling bug in
/// the graph downstream, not a change of intent.
///
/// Both programs here regressed when the lowering landed:
///   * a hop-held control feeding a tensor chain read back through a
///     frame-rate `latch` — the latch was swallowed by the chain's hop guard
///     and stopped re-emitting its held value between hops (silence),
///   * `(sum (* a b))` where one operand became a per-frame intra-block temp —
///     `SumOfMulFusionPass` moved the operand read into the reduce's block
///     while the producer still skipped its store (reads an untouched
///     allocation).
final class ModulatedParamSchedulingTests: XCTestCase {
  private var tempDir: URL!
  private var savedConfig: (backend: Backend, sampleRate: Float, maxFrameCount: Int)!

  override func setUpWithError() throws {
    try super.setUpWithError()
    savedConfig = (DGenConfig.backend, DGenConfig.sampleRate, DGenConfig.maxFrameCount)
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 44100
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-modsched-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
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
    let paramDefaults: [Int: Float]
  }

  private func compile(_ source: String, blockSize: Int) throws -> Compiled {
    DGenConfig.maxFrameCount = blockSize
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: lowerModulation(in: parseSource(source)))
    for output in evaluator.outputs { lazy.addOutput(output.signal, channel: output.channel) }
    let result = try CompilationPipeline.compile(
      graph: lazy.graph, backend: .c,
      options: .init(frameCount: blockSize, debug: false))
    var defaults: [Int: Float] = [:]
    for param in evaluator.params {
      if let logical = param.cellId {
        defaults[result.cellAllocations.cellMappings[logical] ?? logical] = param.defaultValue
      }
    }
    return Compiled(
      result: result,
      source: result.kernels.map { $0.source }.joined(separator: "\n\n"),
      paramDefaults: defaults)
  }

  private func render(
    _ compiled: Compiled, blockSize: Int, blocks: Int, input: Float
  ) throws -> [Float] {
    let kernel = CCompiledKernel(
      source: compiled.source, cellAllocations: compiled.result.cellAllocations,
      memorySize: compiled.result.totalMemorySlots, defaultHostSampleRate: 44100)
    try kernel.compileAndLoad()
    defer { kernel.cleanup() }
    let slots = max(1024, compiled.result.totalMemorySlots)
    let mem = UnsafeMutablePointer<Float>.allocate(capacity: slots)
    mem.initialize(repeating: 0, count: slots)
    // Six channels covers audio in plus the four modulator inlets `(mod …)`
    // materializes; unrouted lanes stay silent.
    let inputs = (0..<6).map { channel -> UnsafeMutablePointer<Float> in
      let buffer = UnsafeMutablePointer<Float>.allocate(capacity: blockSize + 4)
      buffer.initialize(repeating: channel == 0 ? input : 0, count: blockSize + 4)
      return buffer
    }
    let out = UnsafeMutablePointer<Float>.allocate(capacity: blockSize + 4)
    out.initialize(repeating: 0, count: blockSize + 4)
    defer { mem.deallocate(); out.deallocate(); inputs.forEach { $0.deallocate() } }
    let process = try XCTUnwrap(kernel.getProcessFunction())
    let inputPointers: [UnsafePointer<Float>?] = inputs.map { UnsafePointer($0) }
    let outputPointers: [UnsafeMutablePointer<Float>?] = [out]
    var context = DGenProcessContextV1(sampleRate: 44100)
    DGen.injectTensorData(result: compiled.result, memory: mem)
    for (cell, value) in compiled.paramDefaults { mem[cell] = value }
    var samples: [Float] = []
    for _ in 0..<blocks {
      inputPointers.withUnsafeBufferPointer { ip in
        outputPointers.withUnsafeBufferPointer { op in
          withUnsafePointer(to: &context) { contextPointer in
            process(ip.baseAddress, op.baseAddress, UInt32(blockSize), mem,
              UnsafeRawPointer(contextPointer), nil)
          }
        }
      }
      for i in 0..<blockSize { samples.append(out[i]) }
    }
    return samples
  }

  /// Renders `program` twice — once reading the control through `(mod ctl)`
  /// with no modulator routed, once reading the plain parameter — and requires
  /// the two streams to agree.
  private func assertInactiveModulationMatchesPlainParam(
    _ program: String, label: String, blockSize: Int = 512, blocks: Int = 4,
    input: Float = 1, file: StaticString = #filePath, line: UInt = #line
  ) throws {
    let modulated = program.replacingOccurrences(of: "@@CTL@@", with: "(mod ctl)")
      .replacingOccurrences(of: "@@DECL@@", with:
        "(def mod1 (in 3 @name mod1 @modulator 1))\n"
        + "(param ctl @default 0.5 @min 0 @max 1 @mod true @mod-mode additive)")
    let plain = program.replacingOccurrences(of: "@@CTL@@", with: "ctl")
      .replacingOccurrences(of: "@@DECL@@", with: "(param ctl @default 0.5 @min 0 @max 1)")

    let expected = try render(
      compile(plain, blockSize: blockSize), blockSize: blockSize, blocks: blocks, input: input)
    let peak = expected.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(
      peak, 1e-6, "[\(label)] reference render is silent — the repro is vacuous",
      file: file, line: line)
    XCTAssertTrue(
      expected.allSatisfy { $0.isFinite }, "[\(label)] reference render is non-finite",
      file: file, line: line)

    let got = try render(
      compile(modulated, blockSize: blockSize), blockSize: blockSize, blocks: blocks, input: input)
    XCTAssertEqual(got.count, expected.count, file: file, line: line)
    // The lowered form is the same arithmetic in the same order once the
    // gswitch selects the base value, so only -ffast-math reassociation
    // separates the two streams.
    let tolerance = max(1e-5, 1e-5 * peak)
    var maxDiff: Float = 0
    var maxDiffFrame = -1
    for i in 0..<min(got.count, expected.count) {
      let d = abs(got[i] - expected[i])
      if d.isNaN || d > maxDiff { maxDiff = d.isNaN ? .infinity : d; maxDiffFrame = i }
    }
    XCTAssertLessThanOrEqual(
      maxDiff, tolerance,
      "[\(label)] inactive `(mod ctl)` diverges from the plain param: max|diff|=\(maxDiff) "
        + "at frame \(maxDiffFrame) (expected=\(expected[max(maxDiffFrame, 0)]), "
        + "got=\(got[max(maxDiffFrame, 0)]))",
      file: file, line: line)
  }

  /// A hop-held control drives a hop-rate tensor chain; a frame-rate `latch`
  /// holds the gathered, window-faded result and a per-sample reduce reads it.
  /// The latch must run on every frame, not only on hop frames.
  func testHopHeldControlFeedsFrameRateLatch() throws {
    try assertInactiveModulationMatchesPlainParam(
      """
      @@DECL@@
      (def HOP 512)
      (def ctl-h (hop-hold (clip @@CTL@@ 0 1) HOP))
      (def ir (* (+ (iota 2048) 1) ctl-h))
      (make-history hop-frame-ctr)
      (def hop-phase-next
        (write-history hop-frame-ctr (% (+ (read-history hop-frame-ctr) 1) HOP)))
      (def rev-idx (- 255 (iota 256)))
      (def fade (+ (iota 256) 1))
      (def kernel (latch (* (gather ir rev-idx) fade) (eq hop-phase-next 1)))
      (out (sum kernel) 1 @name left)
      """,
      label: "hop-held control -> latch")
  }

  /// `(sum (* a b))` where `b` is a per-frame tensor whose only graph consumer
  /// is the product itself. The fused reduce reads `b` from memory in its own
  /// block, so `b`'s producer must store it.
  func testFusedSumOfMulReadsAStoredOperand() throws {
    try assertInactiveModulationMatchesPlainParam(
      """
      (def in-l (in 1 @name left))
      @@DECL@@
      (def weights (* (+ (iota 8) 1) @@CTL@@))
      (def signal (* in-l (+ (* 0.01 (iota 8)) 1)))
      (out (sum (* signal weights)) 1 @name left)
      """,
      label: "fused sum-of-mul operand")
  }

  /// A ready flag has an independent constant writer. When its read feeds a
  /// feedback delay's time input, both join the feedback region and must still
  /// observe read-before-write semantics on the first frame.
  func testSeededSmootherInDelayFeedbackReadsPreviousReadyFlag() throws {
    let program = """
      (param target @default 0.5 @min 0 @max 1)
      (make-history previous)
      (make-history ready)
      (def value @@VALUE@@)
      (write-history previous value)
      (write-history ready 1)
      (make-history wave)
      (def delayed (delay (read-history wave) (+ 8 (* value 4))))
      (write-history wave (+ (* delayed 0.5) 0.01))
      (out (+ value (* delayed 0.001)) 1 @name audio)
      """
    let seeded = program.replacingOccurrences(of: "@@VALUE@@", with:
      "(gswitch (read-history ready) (mix target (read-history previous) 0.99) target)")
    let constant = program.replacingOccurrences(of: "@@VALUE@@", with: "target")
    for blockSize in [1, 8, 128] {
      let expected = try render(compile(constant, blockSize: blockSize),
        blockSize: blockSize, blocks: 4, input: 0)
      let actual = try render(compile(seeded, blockSize: blockSize),
        blockSize: blockSize, blocks: 4, input: 0)
      XCTAssertEqual(actual[0], 0.5, accuracy: 1e-6, "block \(blockSize): initialize immediately")
      let maxDifference = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
      XCTAssertLessThanOrEqual(maxDifference, 1e-6, "block \(blockSize): previous-frame history")
    }
  }
}

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

  /// Control ticks follow persistent sample time, including calls that begin
  /// between ticks. A tensor's scatter mask must use the producer's clock,
  /// rather than restarting at local frame zero on every process call.
  func testHopTensorLatchPreservesStreamingPhase() throws {
    let program = """
      (def input (in 1 @name audio))
      (def ramp (accum 0.002 0 0 1000))
      (def control (hop-hold ramp 16))
      (def modes (+ (iota 4) 1))
      (def coefficients (cos (* modes control)))
      (def tick (eq (accum 1 0 0 16) 0))
      (def held (latch coefficients tick))
      (out (+ (sum held) (* input 0.01)) 1)
      """
    for blockSize in [1, 4, 12, 28, 64, 128] {
      let output = try render(compile(program, blockSize: blockSize),
        blockSize: blockSize, blocks: 12, input: 1)
      var ramp: Float = 0
      var expected: Float = 0
      for i in output.indices {
        if i % 16 == 0 {
          expected = (1...4).reduce(Float(0.01)) { $0 + cos(Float($1) * ramp) }
        }
        XCTAssertEqual(output[i], expected, accuracy: 1e-5,
          "block \(blockSize), sample \(i): coefficients must hold between global ticks")
        ramp += 0.002
      }
    }
    // The same compiled kernel also receives shorter calls at runtime. Check
    // raw hop-domain reads separately: they must scatter zeros, not hold.
    let scattered = program.replacingOccurrences(of: "(sum held)", with: "(sum (* coefficients input))")
    let compiledScatter = try compile(scattered, blockSize: 128)
    let compiledHold = try compile(program, blockSize: 128)
    for blockSize in [1, 7, 12, 28, 63] {
      for (compiled, holds) in [(compiledHold, true), (compiledScatter, false)] {
        let output = try render(compiled, blockSize: blockSize, blocks: 8, input: 1)
        var ramp: Float = 0
        var expected: Float = 0
        for i in output.indices {
          if i % 16 == 0 {
            expected = (1...4).reduce(Float(0)) { $0 + cos(Float($1) * ramp) }
          }
          let wanted: Float = 0.01 + ((holds || i % 16 == 0) ? expected : 0)
          XCTAssertEqual(output[i], wanted, accuracy: 1e-5,
            "short call \(blockSize), sample \(i), hold=\(holds)")
          ramp += 0.002
        }
      }
    }
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
  /// A tensor-selected waveguide loss must not move the adjacent exciter's
  /// history reads into a separate full-block pass. Synthetic coefficients
  /// preserve the original mixed control/filter/delay scheduling pattern.
  func testTensorSelectedLossPreservesExciterFeedback() throws {
    let program = """
      (def mod1 (in 1 @name mod1 @modulator 1))
      (def gate (phasor 0))
      (def pitch (+ 261.625565 (phasor 0)))
      (def velocity (+ 1 (phasor 0)))
      (def trigger (eq (accum 1 0 0 100000) 0))
      (param character @group voicing @default 0.5 @min 0 @max 1 @mod true @mod-mode additive)
      (param size @group body @default 1 @min 0.5 @max 2 @mod true @mod-mode additive)
      (param decay @group body @default 1 @min 0.2 @max 3 @mod true @mod-mode additive)
      (param damping @group body @default 1 @min 0.25 @max 3 @mod true @mod-mode additive)
      (param hardness @group stick @default 0.5 @min 0 @max 1 @mod true @mod-mode additive)
      (param touch @group contact @default 0 @min 0 @max 1 @mod true @mod-mode additive)
      (param tracking @group tuning @default 0 @min 0 @max 1)
      (param openness @group contact @default 0 @min 0 @max 1 @mod true @mod-mode additive)
      (def closed_material (tensor @shape [24] @data [
        1 2 3 4 5 6 0.0006 0
        2 3 4 5 6 7 0.0006 0.3
        3 4 5 6 7 8 0.0006 1]))
      (def open_material (tensor @shape [24] @data [
        1 2 3 4 5 6 0.008 0
        2 3 4 5 6 7 0.0006 0.3
        3 4 5 6 7 8 0.0025 1]))
      (defmacro cymbal-smooth (target ms)
        (make-history previous)
        (make-history ready)
        (def pole (exp (/ -1 (* 0.001 ms samplerate))))
        (def value (gswitch (read-history ready) (mix target (read-history previous) pole) target))
        (write-history previous value)
        (write-history ready 1)
        value)
      (defmacro cymbal-pole (input pole)
        (make-history previous)
        (def value (mix input (read-history previous) pole))
        (write-history previous value)
        value)
      (make-history last_gate)
      (def held (gt gate 0.5))
      (def onset (max (gt trigger 0.5) (* held (lte (read-history last_gate) 0.5))))
      (write-history last_gate held)
      (def tick (max onset (eq (accum 1 0 0 16) 0)))
      (def strength (latch (clip velocity 0 1) onset))
      (def scale (latch (cymbal-smooth (/ (clip (mod size) 0.5 2)
        (pow (/ (clip pitch 65.406391 1046.50226) 261.625565) (clip tracking 0 1))) 12) tick))
      (def decay_v (latch (cymbal-smooth (clip (mod decay) 0.2 3) 8) tick))
      (def damping_v (latch (cymbal-smooth (clip (mod damping) 0.25 3) 8) tick))
      (def touch_v (cymbal-smooth (clip (mod touch) 0 1) 2))
      (def character_v (latch (cymbal-smooth (clip (mod character) 0 1) 12) tick))
      (def closed_row (* character_v 2))
      (def closed_lo (floor closed_row))
      (def closed_hi (min 2 (+ closed_lo 1)))
      (def closed_mix (- closed_row closed_lo))
      (def closed_material_v (mix (gather closed_material (+ (iota 8) (* closed_lo 8))) (gather closed_material (+ (iota 8) (* closed_hi 8))) closed_mix))
      (def open_row (* character_v 2))
      (def open_lo (floor open_row))
      (def open_hi (min 2 (+ open_lo 1)))
      (def open_mix (- open_row open_lo))
      (def open_material_v (mix (gather open_material (+ (iota 8) (* open_lo 8))) (gather open_material (+ (iota 8) (* open_hi 8))) open_mix))
      (def openness_v (latch (cymbal-smooth (clip (mod openness) 0 1) 2) tick))
      (def material (mix closed_material_v open_material_v openness_v))
      (def base_rate0 (sample material 0))
      (def base_contact_s (sample material 0.75))
      (def contact_loss (* 180 touch_v touch_v))
      (def hardness_v (latch (clip (mod hardness) 0 1) tick))
      (def contact_s (latch (* base_contact_s (pow 2 (* 2 (- 0.5 hardness_v)))) tick))
      (def age (accum (/ 1 samplerate) onset 0 100000))
      (def pulse_phase (clip (/ age contact_s) 0 1))
      (def friction (* (noise) (sin (* pi pulse_phase)) (lt age contact_s)))
      (def force_pole (exp (/ -1 (* samplerate 0.000025 (pow 2 (* 3 (- 0.5 hardness_v)))))))
      (def force (cymbal-pole (cymbal-pole
        (* strength friction 0.23) force_pole) force_pole))
      (def region_rate0 (+ (/ (* base_rate0 (pow damping_v 0)) decay_v) contact_loss))
      (make-history outgoing0)
      (def path_samples0 (max 8 (* 109 scale (/ samplerate 48000))))
      (def integer_delay0 (- (floor path_samples0) 2))
      (def fractional0 (+ 1 (- path_samples0 (floor path_samples0))))
      (def allpass_a0 (/ (- 1 fractional0) (+ 1 fractional0)))
      (def incoming0 (delay (read-history outgoing0) integer_delay0))
      (make-history ap_x0)
      (make-history ap_y0)
      (def arrival0 (- (+ (* allpass_a0 incoming0) (read-history ap_x0)) (* allpass_a0 (read-history ap_y0))))
      (write-history ap_x0 incoming0)
      (write-history ap_y0 arrival0)
      (def path_seconds0 (/ path_samples0 samplerate))
      (def damped0 (* (exp (* (- region_rate0) path_seconds0)) arrival0))
      (write-history outgoing0 (+ force (* damped0 0.9)))
      (out (+ force damped0) 1 @name audio)
      """
    let expected = try render(
      compile(program, blockSize: 1), blockSize: 1, blocks: 512, input: 0)
    let actual = try render(
      compile(program, blockSize: 128), blockSize: 128, blocks: 4, input: 0)
    XCTAssertGreaterThan(expected.map(abs).max() ?? 0, 0.01)
    let error = zip(expected, actual).map { abs($0 - $1) }.max() ?? .infinity
    XCTAssertLessThan(error, 1e-5, "filter state must advance every sample")
  }

}

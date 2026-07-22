import Foundation
import XCTest

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

/// A compiled patch must produce the SAME sample stream regardless of the block
/// size (`--max-frames`) it was compiled at: block size is an execution-batching
/// detail, not a semantic one. These tests compile identical lisp programs at
/// blockSize=1 (ground truth: per-sample execution, trivially correct ordering)
/// and blockSize=8/128, drive them with deterministic state for 512 samples the
/// way the sequencer host drives a dylib (repeated process() calls, persistent
/// memory), and assert sample-exact agreement.
///
/// Three graph shapes are known to miscompile at multi-frame block sizes
/// (found 2026-07-01 via CLI A/B while building the membrane-snare instrument;
/// drums/membrane-kick is affected in production through its biquad exciter):
///
///   1. a scalar `biquad` feeding a tensor-history feedback loop
///   2. reading the same tensor history more than once
///   3. `(write-history h (sum <tensor>))` whose value feeds back into the
///      tensor dynamics on later frames
///
/// A plain FDTD membrane (tensor-history pair + conv2d + scalar exciter) is
/// exact at every block size — `testControl_PlainMembrane` proves the harness —
/// so each failing test isolates exactly one construct.
final class MultiFrameBlockEquivalenceTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-blockeq-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    try super.tearDownWithError()
  }

  private let totalFrames = 512

  /// Compile `source` at the given block size and render `totalFrames` samples
  /// by calling the kernel repeatedly with persistent memory — the same way the
  /// host drives a compiled dylib.
  private func renderBlocked(source: String, blockSize: Int) throws -> [Float] {
    precondition(totalFrames % blockSize == 0)
    DGenConfig.maxFrameCount = blockSize
    LazyGraphContext.reset()
    let lazy = LazyGraphContext.current

    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    guard case .signal(let outSig)? = evaluator.definitions["outsig"] else {
      XCTFail("expected a signal named 'outsig' in the test program")
      return []
    }
    lazy.addOutput(outSig, channel: 0)

    let result = try CompilationPipeline.compile(
      graph: lazy.graph,
      backend: .c,
      options: .init(frameCount: blockSize, debug: false)
    )
    if let dumpDir = ProcessInfo.processInfo.environment["DGEN_BLOCKEQ_DUMP"] {
      let combined = result.kernels.map { $0.source }.joined(separator: "\n\n")
      try? combined.write(
        toFile: "\(dumpDir)/kernel-mf\(blockSize).c", atomically: true, encoding: .utf8)
    }
    let runtime = try CLazyRuntime(
      kernels: result.kernels,
      cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots,
      frameCount: blockSize,
      defaultHostSampleRate: DGenConfig.sampleRate
    )
    runtime.zeroAllBuffers()
    if let mem = runtime.memoryPointer() {
      DGen.injectTensorData(result: result, memory: mem)
    }

    var out = [Float]()
    out.reserveCapacity(totalFrames)
    guard let outPtr = runtime.outputsPointer() else {
      XCTFail("no outputs pointer")
      return []
    }
    for _ in 0..<(totalFrames / blockSize) {
      runtime.runNoCopy(frameCount: blockSize)
      for i in 0..<blockSize { out.append(outPtr[i]) }
    }
    return out
  }

  /// Render at blockSize=1 (ground truth) and at each multi-frame size, and
  /// assert the streams agree sample-exactly.
  private func assertBlockSizeInvariant(
    source: String, label: String,
    blockSizes: [Int] = [8, 128],
    file: StaticString = #filePath, line: UInt = #line
  ) throws {
    let truth = try renderBlocked(source: source, blockSize: 1)
    XCTAssertEqual(truth.count, totalFrames, file: file, line: line)
    XCTAssertTrue(
      truth.allSatisfy { $0.isFinite }, "[\(label)] blockSize=1 output is non-finite",
      file: file, line: line)
    // The program must actually do something, or the comparison is vacuous.
    let maxAbsTruth = truth.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(
      maxAbsTruth, 1e-6,
      "[\(label)] ground-truth render is silent — repro is not exercising the graph",
      file: file, line: line)

    // Tolerance scales with signal amplitude: the kernels compile under
    // -O3 -ffast-math, where clang's vectorization/FMA choices can depend on
    // the block-size-dependent memory base constants, so resonant feedback
    // patches legitimately accumulate ~1e-4 relative noise over 512 frames.
    // The scheduling bugs this suite guards against produce diffs 5-6 orders
    // of magnitude larger (0.1 .. 50).
    let tolerance = max(1e-5, 5e-5 * maxAbsTruth)

    for blockSize in blockSizes {
      let got = try renderBlocked(source: source, blockSize: blockSize)
      XCTAssertEqual(got.count, totalFrames, file: file, line: line)
      var maxDiff: Float = 0
      var maxDiffFrame = -1
      var firstBadFrame = -1
      for i in 0..<totalFrames {
        let d = abs(got[i] - truth[i])
        if d.isNaN || d > maxDiff {
          maxDiff = d.isNaN ? Float.infinity : d
          maxDiffFrame = i
        }
        if firstBadFrame < 0 && (d.isNaN || d > tolerance) { firstBadFrame = i }
      }
      if ProcessInfo.processInfo.environment["DGEN_BLOCKEQ_DUMP"] != nil {
        if let firstDiff = (0..<totalFrames).first(where: { got[$0] != truth[$0] }) {
          print(
            "BLOCKEQ [\(label)] bs=\(blockSize) first bitwise diff at frame \(firstDiff): "
              + "truth=\(truth[firstDiff]) got=\(got[firstDiff])")
        } else {
          print("BLOCKEQ [\(label)] bs=\(blockSize) bit-exact")
        }
      }
      XCTAssertLessThanOrEqual(
        maxDiff, tolerance,
        "[\(label)] blockSize=\(blockSize) diverges from blockSize=1: "
          + "max|diff|=\(maxDiff) at frame \(maxDiffFrame), first divergence at frame "
          + "\(firstBadFrame) (truth=\(truth[max(firstBadFrame, 0)]), got=\(got[max(firstBadFrame, 0)]))",
        file: file, line: line)
    }
  }

  // MARK: - Biquad impulse response must be block-size invariant

  /// The sharpest form of the delay1 bug: a bare biquad driven by a one-frame
  /// impulse. The C backend's SIMD delay1 lowering (vextq_f32 + 4-lane carry
  /// cell) corrupted x[n-1]/x[n-2] whenever a process() call covered fewer
  /// than 4 frames: at blockSize=1 the IR grew ~linearly (integrator-like),
  /// at blockSize=2 it decayed with the wrong sign pattern. Assert the IR
  /// matches the recursion y[n] = b·x - a1·y[n-1] - a2·y[n-2] computed from
  /// the blockSize=4 render (full SIMD groups, verified correct analytically)
  /// at every block size including 1 and 2.
  func testBiquadImpulseResponse_BlockSizeInvariant() throws {
    let source = """
      (make-history started)
      (def imp (- 1 (min (read-history started) 1)))
      (write-history started 1)
      (def outsig (biquad imp 60 1 4 1))
      """
    let reference = try renderBlocked(source: source, blockSize: 4)
    // Sanity: y[1] = b1 + a1·y[0] ≈ -0.0342 for this HP at fc=60/Q=1/gain=4.
    // The broken lowerings gave +7.93 (bs=1) and -4.02 (bs=2) at y[2].
    XCTAssertEqual(reference[0], 3.9829, accuracy: 1e-3)
    XCTAssertEqual(reference[1], -0.0342, accuracy: 1e-3)
    for bs in [1, 2, 8, 128] {
      let y = try renderBlocked(source: source, blockSize: bs)
      for i in 0..<totalFrames {
        XCTAssertEqual(
          y[i], reference[i], accuracy: 1e-6,
          "biquad IR diverges at blockSize=\(bs), frame \(i)")
      }
    }
  }

  // MARK: - Control: this shape is exact, proving the harness itself

  /// Plain 4x4 FDTD membrane: tensor-history pair, conv2d Laplacian, scalar
  /// cos/phasor exciter. No known-bad construct. Must (and does) pass — any
  /// failure here means the harness, not the compiler constructs under test.
  func testControl_PlainMembrane() throws {
    let source = """
      (make-tensor-history h1 @shape [4 4])
      (make-tensor-history h2 @shape [4 4])
      (def lap (tensor @shape [3 3] @data [0 1 0  1 -4 1  0 1 0]))
      (def inj (tensor @shape [4 4] @data [0 0 0 0  0 1 0 0  0 0 0 0  0 0 0 0]))
      (def exc (* (cos (* (phasor 100 0) 6.283185)) 0.01))
      (def s (read-tensor-history h1))
      (def sp (read-tensor-history h2))
      (def nxt (+ (- (* 1.99 s) (* 0.99 sp))
                  (* (conv2d s lap @padding same) 0.02)
                  (* inj exc)))
      (def nc (max (min nxt 3) -3))
      (write-tensor-history h2 s)
      (write-tensor-history h1 nc)
      (def outsig (sum nc))
      """
    try assertBlockSizeInvariant(source: source, label: "control membrane")
  }

  // MARK: - Construct 1: biquad feeding tensor feedback

  /// Identical to the control except the scalar exciter passes through a
  /// `biquad`. This is membrane-kick's production topology.
  ///
  /// The membrane is kept well-damped and driven softly on purpose: the
  /// kernels are compiled with -O3 -ffast-math, where clang's vectorization /
  /// FMA-contraction choices can depend on the (block-size-dependent) memory
  /// base constants, producing legitimate ~1-ulp differences. A marginally
  /// stable, railed membrane amplifies those chaotically and would fail the
  /// comparison even with correct scheduling; the pre-fix delay1 corruption
  /// produced diffs of ~40, so a stable membrane still catches it.
  func testBiquadIntoTensorFeedback() throws {
    let source = """
      (make-tensor-history h1 @shape [4 4])
      (make-tensor-history h2 @shape [4 4])
      (def lap (tensor @shape [3 3] @data [0 1 0  1 -4 1  0 1 0]))
      (def inj (tensor @shape [4 4] @data [0 0 0 0  0 1 0 0  0 0 0 0  0 0 0 0]))
      (def exc (biquad (cos (* (phasor 100 0) 6.283185)) 60 1 4 1))
      (def s (read-tensor-history h1))
      (def sp (read-tensor-history h2))
      (def nxt (+ (- (* 1.97 s) (* 0.98 sp))
                  (* (conv2d s lap @padding same) 0.02)
                  (* inj (* exc 0.02))))
      (def nc (max (min nxt 3) -3))
      (write-tensor-history h2 s)
      (write-tensor-history h1 nc)
      (def outsig (sum nc))
      """
    try assertBlockSizeInvariant(source: source, label: "biquad->tensor feedback")
  }

  // MARK: - Construct 2: duplicate reads of one tensor history

  /// Two coupled membranes where history r1 is read twice (once for the wire
  /// contact limit, once for the head update). Sharing a single read is exact;
  /// the duplicate read diverges.
  func testDuplicateTensorHistoryReads() throws {
    let source = """
      (make-tensor-history r1 @shape [4 4])
      (make-tensor-history r2 @shape [4 4])
      (make-tensor-history w1 @shape [4 4])
      (make-tensor-history w2 @shape [4 4])
      (def lap (tensor @shape [3 3] @data [0 1 0  1 -4 1  0 1 0]))
      (def wlap (tensor @shape [3 3] @data [0 0 0  1 -2 1  0 0 0]))
      (def inj (tensor @shape [4 4] @data [0 0 0 0  0 1 0 0  0 0 0 0  0 0 0 0]))
      (def exc (* (cos (* (phasor 100 0) 6.283185)) 0.01))
      (def ws (read-tensor-history w1))
      (def wp (read-tensor-history w2))
      (def rw (read-tensor-history r1))
      (def wfree (+ (- (* 1.999 ws) (* 0.999 wp))
                    (* (conv2d ws wlap @padding same) 0.05)))
      (def wlim (+ rw 0.003))
      (def wover (max (- wfree wlim) 0))
      (def wn (- wfree (* wover 1.24)))
      (def wnc (max (min wn 3) -3))
      (def cf (min (* wover 1.24) 0.005))
      (def rs (read-tensor-history r1))
      (def rp (read-tensor-history r2))
      (def rn (+ (- (* 1.99 rs) (* 0.99 rp))
                 (* (conv2d rs lap @padding same) 0.03)
                 (* cf 0.02)
                 (* inj exc)))
      (def rnc (max (min rn 3) -3))
      (def outsig (+ (sum rnc) (sum wnc) (sum cf)))
      (write-tensor-history r2 rs)
      (write-tensor-history r1 rnc)
      (write-tensor-history w2 ws)
      (write-tensor-history w1 wnc)
      """
    try assertBlockSizeInvariant(
      source: source, label: "duplicate tensor-history reads",
      blockSizes: [2, 4, 8, 32, 128])
  }

  // MARK: - Construct 3: tensor reduce written to a scalar history that feeds
  // back into the tensor dynamics

  /// A lumped-mass striker senses the membrane through
  /// `(write-history hp (sum (* nc inj)))` and pushes back on it next frame.
  /// Forward-only reduces (reduce -> scalar filter -> out) are exact; the
  /// feedback path through the scalar history diverges.
  func testTensorReduceIntoHistoryFeedback() throws {
    let source = """
      (make-tensor-history h1 @shape [4 4])
      (make-tensor-history h2 @shape [4 4])
      (make-history sx)
      (make-history sv)
      (make-history hp)
      (make-history started)
      (def lap (tensor @shape [3 3] @data [0 1 0  1 -4 1  0 1 0]))
      (def inj (tensor @shape [4 4] @data [0 0 0 0  0 1 0 0  0 0 0 0  0 0 0 0]))
      (def trigger (- 1 (min (read-history started) 1)))
      (write-history started 1)
      (def stick-x (+ (read-history sx) (* trigger 0.0001)))
      (def stick-v (- (read-history sv) (* trigger 0.02)))
      (def pen (max (- (read-history hp) stick-x) 0))
      (def f (min (* 0.004 pen (sqrt pen)) 0.01))
      (def vn (+ stick-v f))
      (write-history sx (+ stick-x vn))
      (write-history sv vn)
      (def s (read-tensor-history h1))
      (def sp (read-tensor-history h2))
      (def nxt (+ (- (* 1.99 s) (* 0.99 sp))
                  (* (conv2d s lap @padding same) 0.02)
                  (* inj (* f -3))
                  (* inj (* trigger 0.01))))
      (def nc (max (min nxt 3) -3))
      (write-tensor-history h2 s)
      (write-tensor-history h1 nc)
      (write-history hp (sum (* nc inj)))
      (def outsig (sum nc))
      """
    try assertBlockSizeInvariant(source: source, label: "tensor reduce -> history feedback")
  }
}

import DGen
import DGenLazy
import XCTest

@testable import DGenLisp

/// Behavioral tests for the dgenlisp Filter Table patch
/// (`toolchain/fixtures/filter-table.lisp`): a wavetable frame reinterpreted as
/// a filter magnitude response and applied by FFT convolution + overlap-add,
/// assembled entirely from existing primitives on the accelerated
/// (inference-only, vDSP) path that compiled C plugins use.
///
/// Everything is measured at `sampleRate == fftSize == 64`, so DFT bin `k` is
/// exactly `k` Hz and a test tone at bin `k` is `cos(2 pi k n / 64)`. That makes
/// "energy in band k" an exact, aliasing-free quantity rather than a estimate
/// smeared across neighbouring bins.
///
/// The 4x-overlap Hann-squared analysis/synthesis pair has a reconstruction gain
/// of 1.5, so an all-pass response reproduces a unit tone at amplitude 1.5.
/// Ratios below are always taken against that reference rather than against 1.
final class FilterTableLispTests: XCTestCase {
  private var tempDir: URL!

  private let N = 64
  private let hop = 16
  private let nBins = 33
  private let irLength = 24
  private let frames = 512

  /// Gain of the Hann-squared overlap-add pair at hop = N/4.
  private let olaGain = 1.5

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 64
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-filtertable-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    DGenConfig.sampleRate = 44100
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
    try super.tearDownWithError()
  }

  // MARK: - Harness

  /// Compile whatever `build` wires into the current lazy graph and run it with
  /// a real input buffer.
  ///
  /// `Signal.realize` feeds silence to input channels, so a test that filters an
  /// actual signal has to take the longer road: compile, hand the C kernel the
  /// input array, read the outputs back.
  @discardableResult
  private func render(
    input: [Float],
    params: [String: Float] = [:],
    build: () throws -> (Signal, [ParamInfo])
  ) throws -> (output: [Float], result: CompilationResult) {
    let (out, paramInfos) = try build()
    let graph = LazyGraphContext.current
    graph.addOutput(out, channel: 0)
    let result = try graph.compileOnly(frameCount: frames)

    let runtime = CCompiledKernel(
      source: result.source,
      cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots)
    try runtime.compileAndLoad()
    defer { runtime.cleanup() }

    guard let mem = runtime.allocateNodeMemory() else {
      throw XCTSkip("memory allocation failed")
    }
    defer { runtime.deallocateNodeMemory(mem) }

    let memPtr = mem.assumingMemoryBound(to: Float.self)
    injectTensorData(result: result, memory: memPtr)

    let mappings = result.cellAllocations.cellMappings
    for info in paramInfos {
      guard let cellId = info.cellId else { continue }
      memPtr[mappings[cellId] ?? cellId] = params[info.name] ?? info.defaultValue
    }

    var output = [Float](repeating: 0, count: frames)
    output.withUnsafeMutableBufferPointer { op in
      input.withUnsafeBufferPointer { ip in
        runtime.runWithMemory(
          outputs: op.baseAddress!, inputs: ip.baseAddress!, memory: mem, frameCount: frames)
      }
    }
    return (output, result)
  }

  // MARK: - Fixture plumbing

  private static let patchMarker = "; %%% PATCH %%%"

  private func fixtureSource() throws -> String {
    let url = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()  // DGenLispTests
      .deletingLastPathComponent()  // Tests
      .deletingLastPathComponent()  // repo root
      .appendingPathComponent("toolchain/fixtures/filter-table.lisp")
    return try String(contentsOf: url, encoding: .utf8)
  }

  /// Everything in the fixture above the `%%% PATCH %%%` marker: the macros and
  /// the wavetable, with no `param`/`in`/`out` of its own. Tests splice their
  /// own instantiation onto this so the code under test is literally the
  /// fixture's, not a copy of it.
  private func fixtureLibrary() throws -> String {
    let source = try fixtureSource()
    guard let range = source.range(of: Self.patchMarker) else {
      throw LispError.invalidArgument("fixture is missing the \(Self.patchMarker) marker")
    }
    return String(source[source.startIndex..<range.lowerBound])
  }

  /// Evaluate lisp source and return its first output plus its parameter table.
  @discardableResult
  private func lispBuild(_ source: String) throws -> (Signal, [ParamInfo]) {
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    lastEvaluator = evaluator
    guard let first = evaluator.outputs.first else {
      throw LispError.invalidArgument("source produced no output")
    }
    return (first.signal, evaluator.params)
  }

  private var lastEvaluator: LispEvaluator?

  /// Render the fixture's `filter-table` with **constant** controls. Constant
  /// controls keep the whole response chain in tensor arithmetic, which the
  /// compiler folds into a static block — the "frozen patch" case.
  private func renderFrozen(
    frame: Double, cutoff: Double = 1, resonance: Double = 1,
    table: String? = nil, input: [Float]? = nil
  ) throws -> [Float] {
    LazyGraphContext.reset()
    var source = try fixtureLibrary()
    if let table {
      source += "\n(def table \(table))\n"
    }
    source += """

      (def dry (in 1 @name audio-in))
      (out (filter-table table dry \(frame) \(cutoff) \(resonance)) 1 @name audio-out)
      """
    return try render(input: input ?? twoToneInput()) { try self.lispBuild(source) }.output
  }

  // MARK: - Measurement helpers

  /// Amplitude of DFT bin `k` (k cycles per `N` samples) measured over `range`.
  private func bandAmplitude(_ x: [Float], bin k: Int, range: Range<Int>) -> Double {
    var re = 0.0
    var im = 0.0
    for n in range {
      let phase = 2.0 * Double.pi * Double(k) * Double(n) / Double(N)
      re += Double(x[n]) * Foundation.cos(phase)
      im -= Double(x[n]) * Foundation.sin(phase)
    }
    return 2.0 * (re * re + im * im).squareRoot() / Double(range.count)
  }

  /// Two unit cosines: one low (bin 4) and one high (bin 24).
  private func twoToneInput() -> [Float] {
    (0..<frames).map { n in
      let t = Double(n) / Double(N)
      return Float(Foundation.cos(2 * .pi * 4 * t) + Foundation.cos(2 * .pi * 24 * t))
    }
  }

  /// Skip the first two windows so the overlap-add pipeline is primed.
  private var steady: Range<Int> { (2 * N)..<frames }

  private func rms(_ x: [Float], _ range: Range<Int>) -> Double {
    let sum = range.reduce(0.0) { $0 + Double(x[$1]) * Double(x[$1]) }
    return (sum / Double(range.count)).squareRoot()
  }

  // MARK: - 1. The patch builds and runs

  func testFixturePatchCompilesAndRuns() throws {
    let (output, _) = try render(input: twoToneInput()) { try self.lispBuild(try self.fixtureSource()) }
    XCTAssertTrue(output.allSatisfy { $0.isFinite }, "patch produced non-finite samples")
    XCTAssertEqual(lastEvaluator?.params.map(\.name), ["frame", "cutoff", "resonance", "mix"])
  }

  // MARK: - 2. Filter shapes actually filter

  /// Frame 0 is a lowpass and frame 1 a highpass, so the two must trade which
  /// tone survives. Measured against the all-pass frame (3) rather than against
  /// the raw input, which folds out the 1.5x overlap-add gain.
  func testLowpassAndHighpassFramesSwapWhichToneSurvives() throws {
    let flat = try renderFrozen(frame: 3)
    let low = try renderFrozen(frame: 0)
    let high = try renderFrozen(frame: 1)

    let flatLo = bandAmplitude(flat, bin: 4, range: steady)
    let flatHi = bandAmplitude(flat, bin: 24, range: steady)
    // The flat frame is the identity filter: it should reproduce both tones at
    // exactly the overlap-add gain.
    XCTAssertEqual(flatLo, olaGain, accuracy: 0.01)
    XCTAssertEqual(flatHi, olaGain, accuracy: 0.01)

    let lowLo = bandAmplitude(low, bin: 4, range: steady) / flatLo
    let lowHi = bandAmplitude(low, bin: 24, range: steady) / flatHi
    XCTAssertGreaterThan(lowLo, 0.95, "lowpass frame should pass bin 4 essentially untouched")
    XCTAssertLessThan(lowHi, 0.02, "lowpass frame should reject bin 24")

    let highLo = bandAmplitude(high, bin: 4, range: steady) / flatLo
    let highHi = bandAmplitude(high, bin: 24, range: steady) / flatHi
    XCTAssertLessThan(highLo, 0.02, "highpass frame should reject bin 4")
    XCTAssertGreaterThan(highHi, 0.95, "highpass frame should pass bin 24")

    // And the two are genuinely opposite, not merely both quiet somewhere.
    XCTAssertGreaterThan(lowLo / max(lowHi, 1e-9), 20.0)
    XCTAssertGreaterThan(highHi / max(highLo, 1e-9), 20.0)
  }

  // MARK: - 3. Scanning the frame morphs the response

  /// Frame 2 is a bandpass centered on bin 16, which rejects both test tones,
  /// so scanning 0 -> 1 -> 2 must produce three measurably different responses,
  /// and a position halfway between two frames must land between them.
  func testFrameScanMorphsAndInterpolates() throws {
    let flat = try renderFrozen(frame: 3)
    let flatLo = bandAmplitude(flat, bin: 4, range: steady)
    let flatHi = bandAmplitude(flat, bin: 24, range: steady)

    func response(_ frame: Double) throws -> (lo: Double, hi: Double) {
      let y = try renderFrozen(frame: frame)
      return (
        bandAmplitude(y, bin: 4, range: steady) / flatLo,
        bandAmplitude(y, bin: 24, range: steady) / flatHi
      )
    }

    let f0 = try response(0)  // lowpass
    let f1 = try response(1)  // highpass
    let f2 = try response(2)  // bandpass at bin 16 — rejects both tones
    XCTAssertLessThan(f2.lo, 0.15, "bandpass frame should reject bin 4")
    XCTAssertLessThan(f2.hi, 0.15, "bandpass frame should reject bin 24")

    // Three distinct responses.
    XCTAssertGreaterThan(abs(f0.lo - f1.lo), 0.5)
    XCTAssertGreaterThan(abs(f1.hi - f2.hi), 0.5)

    // Halfway between the lowpass and the highpass, both tones are half-passed
    // (magnitudes interpolate linearly, so bin 4 goes 1 -> 0.5 and bin 24
    // 0 -> 0.5); the mid position must sit strictly between its neighbours.
    let mid = try response(0.5)
    XCTAssertGreaterThan(mid.lo, f1.lo)
    XCTAssertLessThan(mid.lo, f0.lo)
    XCTAssertGreaterThan(mid.hi, f0.hi)
    XCTAssertLessThan(mid.hi, f1.hi)
    XCTAssertEqual(mid.lo, 0.5, accuracy: 0.08)
    XCTAssertEqual(mid.hi, 0.5, accuracy: 0.08)

    // Interpolation is monotone across the frame axis for a tone in the
    // lowpass's passband.
    let quarter = try response(0.25)
    let threeQuarter = try response(0.75)
    XCTAssertGreaterThan(quarter.lo, mid.lo)
    XCTAssertGreaterThan(mid.lo, threeQuarter.lo)
  }

  // MARK: - 4. Cutoff slides the response in frequency

  /// The lowpass frame rolls off around bin 10. Bin k reads curve position
  /// k/cutoff, so cutoff 2.5 drags the edge up past bin 24 and the high tone
  /// comes back; cutoff 0.4 drags it down below bin 4 and the low tone leaves.
  func testCutoffSlidesTheResponseInFrequency() throws {
    let flat = try renderFrozen(frame: 3)
    let flatLo = bandAmplitude(flat, bin: 4, range: steady)
    let flatHi = bandAmplitude(flat, bin: 24, range: steady)

    let closed = try renderFrozen(frame: 0, cutoff: 0.4)
    let nominal = try renderFrozen(frame: 0, cutoff: 1.0)
    let open = try renderFrozen(frame: 0, cutoff: 3.0)

    let closedLo = bandAmplitude(closed, bin: 4, range: steady) / flatLo
    let nominalLo = bandAmplitude(nominal, bin: 4, range: steady) / flatLo
    let openHi = bandAmplitude(open, bin: 24, range: steady) / flatHi
    let nominalHi = bandAmplitude(nominal, bin: 24, range: steady) / flatHi

    // Opening the cutoff lets bin 24 through; at nominal it was rejected.
    XCTAssertLessThan(nominalHi, 0.02)
    XCTAssertGreaterThan(openHi, 0.9)

    // Closing it starts to eat bin 4, which was fully passed at nominal.
    XCTAssertGreaterThan(nominalLo, 0.95)
    XCTAssertLessThan(closedLo, 0.6)

    // Monotone in the control: bin 24's gain only ever rises with cutoff.
    var previous = 0.0
    for cutoff in [1.0, 1.5, 2.0, 2.5, 3.0] {
      let y = try renderFrozen(frame: 0, cutoff: cutoff)
      let gain = bandAmplitude(y, bin: 24, range: steady) / flatHi
      XCTAssertGreaterThan(
        gain, previous - 0.02, "bin 24 gain should not fall as cutoff opens (cutoff=\(cutoff))")
      previous = gain
    }
  }

  // MARK: - 5. Resonance exaggerates the curve

  /// Resonance raises the curve to a power about its own mean, so bins below the
  /// mean sink and bins above it rise. The bandpass frame has a clear peak and
  /// clear skirts, so raising resonance must sharpen it: the skirt at bin 24
  /// loses more than the peak region does.
  func testResonanceSharpensTheCurve() throws {
    // A gentle bandpass whose skirt still passes bin 24, so there is something
    // to sharpen away.
    let gentle = (0..<nBins).map { k -> String in
      let d = Double(k - 16) / 16.0
      return String(format: "%.4f", 1.0 / (1.0 + 6.0 * d * d))
    }.joined(separator: " ")
    let table = "(tensor @shape [1 33] @data [\(gentle)])"

    // A tone at the peak and a tone on the skirt.
    let probe: [Float] = (0..<frames).map { n in
      let t = Double(n) / Double(N)
      return Float(Foundation.cos(2 * .pi * 16 * t) + Foundation.cos(2 * .pi * 24 * t))
    }

    // Normalize against the all-pass frame driven by the *same* probe, so both
    // references are the overlap-add gain rather than a bin with no content.
    let flat = try renderFrozen(frame: 3, input: probe)
    let flatPeak = bandAmplitude(flat, bin: 16, range: steady)
    let flatSkirt = bandAmplitude(flat, bin: 24, range: steady)
    XCTAssertEqual(flatPeak, olaGain, accuracy: 0.01)
    XCTAssertEqual(flatSkirt, olaGain, accuracy: 0.01)

    func contrast(_ res: Double) throws -> Double {
      let y = try renderFrozen(frame: 0, resonance: res, table: table, input: probe)
      let peak = bandAmplitude(y, bin: 16, range: steady) / flatPeak
      let skirt = bandAmplitude(y, bin: 24, range: steady) / flatSkirt
      return peak / max(skirt, 1e-9)
    }

    // Measured peak:skirt contrast — 2.20 at res 1, 8.56 at res 3, 26.6 at
    // res 5. The ideal is (1/0.4)^res = 2.5 / 15.6 / 97.7; the bounded IR
    // smooths the curve, so the realized contrast trails the ideal but keeps
    // the same direction and ordering.
    let flatRes = try contrast(1.0)
    let sharp = try contrast(3.0)
    let sharper = try contrast(5.0)
    XCTAssertGreaterThan(sharp, flatRes * 1.5, "resonance 3 should sharpen the curve")
    XCTAssertGreaterThan(sharper, sharp, "resonance 5 should sharpen it further")
  }

  // MARK: - 6. The IR window is load-bearing

  /// Without the IR window, multiplying a frame's spectrum by a magnitude curve
  /// is *circular* convolution: a brickwall curve's true impulse response is a
  /// sinc far longer than the 64-sample frame, so its tail wraps around and the
  /// effective response fills the whole frame. The wrapped-Hann window bounds
  /// the IR to `irLength` taps and the wraparound disappears.
  ///
  /// The measurement is the effect's own impulse response — feed a lone impulse
  /// and see how far the output spreads. Note the *steady-tone* rejection is not
  /// the observable here: a tone sitting exactly on a bin is killed perfectly
  /// either way (more perfectly without the window, since an unwindowed
  /// brickwall is a literal zero at that bin). Time-domain smearing is what the
  /// window fixes, so time-domain support is what the test measures.
  func testIRWindowBoundsTheImpulseResponse() throws {
    // A hard brickwall at bin 8 — the worst case for IR length.
    let brick = (0..<nBins).map { $0 <= 8 ? "1" : "0" }.joined(separator: " ")
    let table = "(tensor @shape [1 33] @data [\(brick)])"

    var impulse = [Float](repeating: 0, count: frames)
    impulse[frames / 2] = 1

    let windowed = try renderFrozen(frame: 0, table: table, input: impulse)
    let unwindowed = try renderUnwindowed(table: table, input: impulse)

    /// Width of the region where |y| exceeds 1% of peak, and the fraction of
    /// total energy lying more than 20 samples from the peak.
    func spread(_ y: [Float]) -> (width: Int, farEnergyFraction: Double) {
      let peak = y.map { abs($0) }.max() ?? 0
      let peakIndex = y.firstIndex { abs($0) == peak } ?? 0
      let first = y.firstIndex { abs($0) > 0.01 * peak } ?? 0
      let last = y.lastIndex { abs($0) > 0.01 * peak } ?? 0
      var total = 0.0
      var far = 0.0
      for (n, v) in y.enumerated() {
        let e = Double(v) * Double(v)
        total += e
        if abs(n - peakIndex) > 20 { far += e }
      }
      return (last - first, far / Swift.max(total, 1e-30))
    }

    let bounded = spread(windowed)
    let circular = spread(unwindowed)

    // The bounded IR is ~24 taps; the unwindowed one fills the 64-sample frame.
    XCTAssertLessThanOrEqual(
      bounded.width, irLength,
      "the windowed response should stay inside its \(irLength)-tap budget")
    XCTAssertGreaterThanOrEqual(
      circular.width, 2 * irLength,
      "the unwindowed response should smear across the whole frame")

    // Measured: 6.5e-11 windowed vs 1.6e-3 unwindowed — seven orders of
    // magnitude. The thresholds leave two orders of headroom on each side.
    XCTAssertLessThan(bounded.farEnergyFraction, 1e-8)
    XCTAssertGreaterThan(circular.farEnergyFraction, 1e-4)
  }

  /// The fixture's `filter-table` with the IR window replaced by all-ones, i.e.
  /// no bounding at all. Everything else — mirror, IFFT, FFT, overlap-add — is
  /// identical, so the two renders differ in exactly one term.
  private func renderUnwindowed(table: String, input: [Float]) throws -> [Float] {
    LazyGraphContext.reset()
    var source = try fixtureLibrary()
    let windowExpression = "(* (* 0.5 (+ 1 (cos (* PI (/ dist half))))) (lte dist half))"
    guard source.contains(windowExpression) else {
      throw LispError.invalidArgument("fixture's ir-window body changed; update this test")
    }
    source = source.replacingOccurrences(of: windowExpression, with: "(+ (* 0 dist) 1)")
    source += """

      (def table \(table))
      (def dry (in 1 @name audio-in))
      (out (filter-table table dry 0 1 1) 1 @name audio-out)
      """
    return try render(input: input) { try self.lispBuild(source) }.output
  }

  // MARK: - 7. Agreement with the Swift reference

  /// `spectralFilterPerHop` in `Sources/DGenLazy/SpectralFilter.swift` is the
  /// same operator written directly in Swift, and it is the strongest oracle
  /// available: same mirror, same zero-phase IR, same wrapped-Hann IR window,
  /// same overlap-add.
  ///
  /// The two disagree only in arithmetic detail. The lisp patch runs vDSP's
  /// split-radix FFT; the Swift reference runs the graph's own composed tensor
  /// FFT. Both are float32, both do O(N log N) butterflies, and the input is
  /// order 1, so reassociation alone accounts for a few float32 ulps.
  ///
  /// Measured worst-case disagreement is 2.4e-7 absolute against a reference of
  /// RMS 1.0 — that is 2 ulps of float32, i.e. the two implementations agree as
  /// closely as two float32 FFTs can. The 1e-5 relative bound below leaves ~40x
  /// headroom for FFT-library drift while staying four orders of magnitude below
  /// any real semantic disagreement (a wrong mirror index or a missing IR window
  /// moves samples by O(0.1)).
  func testMatchesSwiftSpectralFilterReference() throws {
    let magnitudes: [Float] = (0..<nBins).map { k in
      let x = Double(k) / 10.0
      return Float(1.0 / (1.0 + x * x * x * x))
    }
    let input = twoToneInput()

    let table = "(tensor @shape [1 33] @data [\(magnitudes.map { String($0) }.joined(separator: " "))])"
    let lisp = try renderFrozen(frame: 0, table: table, input: input)

    LazyGraphContext.reset()
    let swiftSide = try render(input: input) { () -> (Signal, [ParamInfo]) in
      // Tensors must be created after reset (see CLAUDE.md).
      let mags = Tensor([magnitudes])
      let out = spectralFilterPerHop(
        Signal.input(0), magnitudes: mags, framePosition: Signal.constant(0),
        fftSize: self.N, hop: self.hop, irLength: self.irLength)
      return (out, [])
    }.output

    let reference = rms(swiftSide, steady)
    XCTAssertGreaterThan(reference, 0.1, "reference render is silent; the comparison is vacuous")

    var worst = 0.0
    for n in steady {
      worst = Swift.max(worst, abs(Double(lisp[n]) - Double(swiftSide[n])))
    }
    XCTAssertLessThan(
      worst / reference, 1e-5,
      "lisp patch and Swift spectralFilterPerHop disagree by \(worst) (reference rms \(reference))")
  }

  // MARK: - 8. Structure: the table read stays hop-gated

  /// The surviving idea from the old `Tests/DGenTests/FilterTableTests.swift`.
  /// With live controls the response is rebuilt per hop, not per sample, and the
  /// compiler must place the table read in a hop-gated block for that to be
  /// true. This is a performance property, so it is asserted on the compiled
  /// block structure rather than on the audio.
  func testTableReadCompilesIntoAHopGatedBlock() throws {
    LazyGraphContext.reset()
    let source = try fixtureSource()
    let (_, result) = try render(input: twoToneInput()) { try self.lispBuild(source) }

    // `gather` is how the patch reads the table (peek-vec lowers to two of
    // them), so a hop-gated gather is the property under test.
    let gatherBlocks = result.sortedBlocks.filter { block in
      block.nodes.contains { nodeId in
        if case .gather = result.graph.nodes[nodeId]?.op { return true }
        return false
      }
    }
    XCTAssertFalse(gatherBlocks.isEmpty, "expected the compiled patch to contain gathers")

    let hopGated = gatherBlocks.filter {
      if case .hopBased(let size, _) = $0.temporality { return size == hop }
      return false
    }
    XCTAssertFalse(
      hopGated.isEmpty,
      """
      the filter-table read should run at hop rate; \
      block temporalities were \(gatherBlocks.map { "\($0.temporality)" })
      """)
  }

  // MARK: - 9. Known-broken: live controls

  /// The same patch with *live* (signal) controls currently renders wrong audio.
  /// This is not a flaw in the construction above — it is an upstream
  /// regression in hop-sliced frame-aware tensor storage, bisected to commit
  /// 1db7025 ("svf surrugate attempt"), which taught
  /// `TensorMemoryMaterializationPass` to allocate one slot per hop for
  /// hop-based tensors and to address them as `frameIdx / hop` in
  /// `IRBuilder.frameAwareOffset`, but left the other frame-aware offset
  /// computations (the elementwise/shape-transition writers in `Emit+Tensor` and
  /// friends) still writing at `frameIdx * tensorSize`. Writers and readers
  /// therefore disagree by a factor of `hop`, and the writes run past the
  /// shortened allocation.
  ///
  /// Reverting the `hop` computation in `TensorMemoryMaterializationPass` to a
  /// constant 1 makes this test — and the old `DGenTests.FilterTableTests`, which
  /// is red at HEAD for the same reason — pass exactly.
  ///
  /// Left in place, skipped, so it turns green the moment the storage bug is
  /// fixed rather than being silently forgotten.
  func testLiveControlsMatchFrozenControls() throws {
    throw XCTSkip(
      "blocked on hop-sliced frame-aware tensor storage (regressed in 1db7025); see doc comment")

    // swift-format-ignore
    // (unreachable until the skip is removed)
    /*
    let frozen = try renderFrozen(frame: 1)

    LazyGraphContext.reset()
    var source = try fixtureLibrary()
    source += """

      (param frame @default 0 @min 0 @max 3)
      (def dry (in 1 @name audio-in))
      (out (filter-table table dry (hop-hold frame 16) 1 1) 1 @name audio-out)
      """
    let live = try render(input: twoToneInput(), params: ["frame": 1]) {
      try self.lispBuild(source)
    }.output

    for n in steady {
      XCTAssertEqual(Double(live[n]), Double(frozen[n]), accuracy: 1e-4)
    }
    */
  }
}

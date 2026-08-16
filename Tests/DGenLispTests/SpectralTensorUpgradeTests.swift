import DGenLazy
import XCTest

@testable import DGenLisp

final class SpectralTensorUpgradeTests: XCTestCase {
  private var tempDir: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 8
    DGenConfig.maxFrameCount = 32
    LazyGraphContext.reset()
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgenlisp-spectral-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir {
      try? FileManager.default.removeItem(at: tempDir)
    }
    try super.tearDownWithError()
  }

  private func evaluator(_ source: String) throws -> LispEvaluator {
    let evaluator = LispEvaluator(sourceDirectory: tempDir)
    try evaluator.evaluate(nodes: parseSource(source))
    return evaluator
  }

  private func tensor(_ evaluator: LispEvaluator, _ name: String) throws -> Tensor {
    guard case .tensor(let t)? = evaluator.definitions[name] else {
      throw XCTSkip("expected tensor \(name)")
    }
    return t
  }

  private func signalTensor(_ evaluator: LispEvaluator, _ name: String) throws -> SignalTensor {
    guard case .signalTensor(let t)? = evaluator.definitions[name] else {
      throw XCTSkip("expected signalTensor \(name)")
    }
    return t
  }

  private func signal(_ evaluator: LispEvaluator, _ name: String) throws -> Signal {
    guard case .signal(let s)? = evaluator.definitions[name] else {
      throw XCTSkip("expected signal \(name)")
    }
    return s
  }

  private func compileOutputs(_ evaluator: LispEvaluator, frames: Int = 16) throws {
    let graph = LazyGraphContext.current
    for output in evaluator.outputs {
      graph.addOutput(output.signal, channel: output.channel)
    }
    _ = try graph.compileOnly(frameCount: frames, voiceCount: 1)
  }

  private func makeWav(_ name: String, samples: [Float], sampleRate: Float = 8) throws -> URL {
    let url = tempDir.appendingPathComponent(name)
    try AudioFile.save(url: url, samples: samples, sampleRate: sampleRate)
    return url
  }

  private func makeStereoWav(_ name: String, frames: [(Float, Float)], sampleRate: Int = 8) throws {
    let url = tempDir.appendingPathComponent(name)
    var data = Data()
    func append(_ s: String) { data.append(s.data(using: .ascii)!) }
    func appendUInt16(_ value: UInt16) {
      var le = value.littleEndian
      data.append(Data(bytes: &le, count: 2))
    }
    func appendUInt32(_ value: UInt32) {
      var le = value.littleEndian
      data.append(Data(bytes: &le, count: 4))
    }
    func appendFloat(_ value: Float) {
      var le = value
      data.append(Data(bytes: &le, count: 4))
    }
    let channels = 2
    let bytesPerSample = 4
    let dataBytes = frames.count * channels * bytesPerSample
    append("RIFF")
    appendUInt32(UInt32(36 + dataBytes))
    append("WAVE")
    append("fmt ")
    appendUInt32(16)
    appendUInt16(3)
    appendUInt16(UInt16(channels))
    appendUInt32(UInt32(sampleRate))
    appendUInt32(UInt32(sampleRate * channels * bytesPerSample))
    appendUInt16(UInt16(channels * bytesPerSample))
    appendUInt16(32)
    append("data")
    appendUInt32(UInt32(dataBytes))
    for frame in frames {
      appendFloat(frame.0)
      appendFloat(frame.1)
    }
    try data.write(to: url)
  }

  // MARK: - Destructuring / Multi-output

  func testDestructuringDefBindsFFTOutputs() throws {
    let e = try evaluator(
      """
      (def x (tensor @shape [4] @data [1 0 0 0]))
      (def (re im) (fft x @N 4))
      """)
    XCTAssertEqual(try tensor(e, "re").shape, [4])
    XCTAssertEqual(try tensor(e, "im").shape, [4])
  }

  func testDestructuringDefRejectsArityMismatch() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def x (tensor @shape [4] @data [1 0 0 0]))
        (def (re im extra) (fft x @N 4))
        """)
    ) { error in
      XCTAssertTrue(String(describing: error).contains("expected 3 values"))
    }
  }

  func testTupleOperatorSupportsDestructuringDef() throws {
    let e = try evaluator(
      """
      (def (x1 x2 x3) (tuple (* 1 2) (* 2 3) (* 3 4)))
      """)
    guard case .float(let x1)? = e.definitions["x1"],
      case .float(let x2)? = e.definitions["x2"],
      case .float(let x3)? = e.definitions["x3"]
    else {
      return XCTFail("expected destructured float tuple bindings")
    }
    XCTAssertEqual(x1, 2)
    XCTAssertEqual(x2, 6)
    XCTAssertEqual(x3, 12)
  }

  func testMacroCanReturnTupleForDestructuringDef() throws {
    let e = try evaluator(
      """
      (defmacro multi (a b c)
        (tuple (* a 2) (* a b) (* b c)))
      (def (x1 x2 x3) (multi 1 2 3))
      """)
    guard case .float(let x1)? = e.definitions["x1"],
      case .float(let x2)? = e.definitions["x2"],
      case .float(let x3)? = e.definitions["x3"]
    else {
      return XCTFail("expected macro tuple outputs")
    }
    XCTAssertEqual(x1, 2)
    XCTAssertEqual(x2, 2)
    XCTAssertEqual(x3, 6)
  }

  func testTupleOperatorRejectsEmptyTuple() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def x (tuple))
        """)
    ) { error in
      XCTAssertTrue(String(describing: error).contains("tuple requires at least 1 argument"))
    }
  }

  func testMacroScopedDestructuringDoesNotCollide() throws {
    let e = try evaluator(
      """
      (defmacro both (x)
        (def (re im) (fft x @N 4))
        (+ re im))
      (def a (tensor @shape [4] @data [1 0 0 0]))
      (def b (tensor @shape [4] @data [0 1 0 0]))
      (def out1 (both a))
      (def out2 (both b))
      """)
    XCTAssertEqual(try tensor(e, "out1").shape, [4])
    XCTAssertEqual(try tensor(e, "out2").shape, [4])
  }

  // MARK: - FFT Backends

  func testFFTStillPublishesLegacyBindings() throws {
    let e = try evaluator(
      """
      (def x (tensor @shape [4] @data [1 0 0 0]))
      (def (re im) (fft x @N 4))
      """)
    XCTAssertNotNil(e.definitions["__fft_re"])
    XCTAssertNotNil(e.definitions["__fft_im"])
    XCTAssertEqual(try tensor(e, "re").shape, [4])
  }

  func testIFFTAcceptsAttributeSize() throws {
    let e = try evaluator(
      """
      (def re (tensor @shape [4] @data [1 1 1 1]))
      (def im (tensor @shape [4] @data [0 0 0 0]))
      (def x (ifft re im @N 4))
      """)
    XCTAssertEqual(try tensor(e, "x").shape, [4])
  }

  func testAcceleratedFFTBackendBuildsTuple() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (re im) (fft frame @N 4 @backend accelerated))
      """)
    XCTAssertEqual(try signalTensor(e, "re").shape, [4])
    XCTAssertEqual(try signalTensor(e, "im").shape, [4])
  }

  // MARK: - Spectral Coordinate Helpers

  func testPolarFFTOutputsMagnitudeAndPhase() throws {
    let e = try evaluator(
      """
      (def re (tensor @shape [2] @data [3 0]))
      (def im (tensor @shape [2] @data [4 1]))
      (def (mag phase) (polar-fft re im))
      """)
    XCTAssertEqual(try tensor(e, "mag").shape, [2])
    XCTAssertEqual(try tensor(e, "phase").shape, [2])
  }

  func testRectFFTOutputsRealAndImag() throws {
    let e = try evaluator(
      """
      (def mag (tensor @shape [2] @data [1 1]))
      (def phase (tensor @shape [2] @data [0 1.5707964]))
      (def (re im) (rect-fft mag phase))
      """)
    XCTAssertEqual(try tensor(e, "re").shape, [2])
    XCTAssertEqual(try tensor(e, "im").shape, [2])
  }

  func testAtanAndAtan2AndLog10AreExposed() throws {
    let e = try evaluator(
      """
      (def a (atan 1))
      (def b (atan2 1 0))
      (def c (log10 100))
      """)
    guard case .float(let a)? = e.definitions["a"],
      case .float(let b)? = e.definitions["b"],
      case .float(let c)? = e.definitions["c"]
    else {
      return XCTFail("expected floats")
    }
    XCTAssertEqual(a, Float.pi / 4, accuracy: 0.0001)
    XCTAssertEqual(b, Float.pi / 2, accuracy: 0.0001)
    XCTAssertEqual(c, 2, accuracy: 0.0001)
  }

  // MARK: - Complex Helpers

  func testComplexMulStaticTensors() throws {
    let e = try evaluator(
      """
      (def ar (tensor @shape [2] @data [1 0]))
      (def ai (tensor @shape [2] @data [0 1]))
      (def br (tensor @shape [2] @data [1 1]))
      (def bi (tensor @shape [2] @data [1 0]))
      (def (re im) (complex-mul ar ai br bi))
      """)
    XCTAssertEqual(try tensor(e, "re").shape, [2])
    XCTAssertEqual(try tensor(e, "im").shape, [2])
  }

  func testComplexMulSignalTensorWithStaticTensor() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (ar ai) (fft frame @N 4))
      (def br (tensor @shape [4] @data [1 1 1 1]))
      (def bi (tensor @shape [4] @data [0 0 0 0]))
      (def (re im) (complex-mul ar ai br bi))
      """)
    XCTAssertEqual(try signalTensor(e, "re").shape, [4])
    XCTAssertEqual(try signalTensor(e, "im").shape, [4])
  }

  func testComplexConjNegatesImaginaryOutputShape() throws {
    let e = try evaluator(
      """
      (def re (tensor @shape [2] @data [1 2]))
      (def im (tensor @shape [2] @data [3 4]))
      (def (cre cim) (complex-conj re im))
      """)
    XCTAssertEqual(try tensor(e, "cre").shape, [2])
    XCTAssertEqual(try tensor(e, "cim").shape, [2])
  }

  // MARK: - Windows

  func testHannReturnsTensorOfRequestedSize() throws {
    let e = try evaluator("(def w (hann 8))")
    XCTAssertEqual(try tensor(e, "w").shape, [8])
  }

  func testWindowAliasCreatesHann() throws {
    let e = try evaluator("(def w (window @type hann @N 4))")
    XCTAssertEqual(try tensor(e, "w").shape, [4])
  }

  func testHannHasExpectedEndpoints() throws {
    let values = try tensor(try evaluator("(def w (hann 4))"), "w").realize()
    XCTAssertEqual(values[0], 0, accuracy: 0.0001)
    XCTAssertEqual(values[2], 1, accuracy: 0.0001)
  }

  // MARK: - Hop Helpers

  func testTensorNoiseWithSizeReturnsSignalTensor() throws {
    let e = try evaluator("(def n (noise @size 8))")
    XCTAssertEqual(try signalTensor(e, "n").shape, [8])
  }

  func testTensorNoiseWithHopReturnsSignalTensor() throws {
    let e = try evaluator("(def n (noise @size 8 @hop 4))")
    XCTAssertEqual(try signalTensor(e, "n").shape, [8])
  }

  func testHopHoldWorksForSignalTensor() throws {
    let e = try evaluator(
      """
      (def n (noise @size 4))
      (def h (hop-hold n 2))
      """)
    XCTAssertEqual(try signalTensor(e, "h").shape, [4])
  }

  // MARK: - Spectrum Delay

  func testSpectrumDelayBuildsSignalTensor() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (re im) (fft frame @N 4))
      (def delayed (spectrum-delay re @N 4 @hops 1 @hop 2))
      """)
    XCTAssertEqual(try signalTensor(e, "delayed").shape, [4])
  }

  func testSpectrumDelayModBuildsSignalTensor() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (re im) (fft frame @N 4))
      (def delayed (spectrum-delay-mod re 1 @N 4 @max-hops 3 @hop 2))
      """)
    XCTAssertEqual(try signalTensor(e, "delayed").shape, [4])
  }

  func testSpectrumDelayRejectsStaticTensor() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def x (tensor @shape [4] @data [1 2 3 4]))
        (def bad (spectrum-delay x @N 4 @hops 1 @hop 2))
        """))
  }

  // MARK: - Phase Vocoder

  func testPhaseVocoderReturnsTwoSignalTensors() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (re im) (fft frame @N 4))
      (def (pre pim) (phase-vocoder re im 1.5 @N 4 @hop 2))
      """)
    XCTAssertEqual(try signalTensor(e, "pre").shape, [4])
    XCTAssertEqual(try signalTensor(e, "pim").shape, [4])
  }

  func testPhaseVocoderAcceptsSignalRatio() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def ratio (+ 1 (phasor 1)))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (re im) (fft frame @N 4))
      (def (pre pim) (phase-vocoder re im ratio @N 4 @hop 2))
      """)
    XCTAssertEqual(try signalTensor(e, "pre").shape, [4])
  }

  func testPhaseVocoderRejectsStaticTensors() throws {
    XCTAssertThrowsError(
      try evaluator(
        """
        (def re (tensor @shape [4] @data [1 1 1 1]))
        (def im (tensor @shape [4] @data [0 0 0 0]))
        (def (pre pim) (phase-vocoder re im 1 @N 4 @hop 2))
        """))
  }

  // MARK: - Audio / IR Tensors

  func testAudioTensorLoadsWavManifestData() throws {
    _ = try makeWav("tiny.wav", samples: [0.0, 0.5, -0.5, 1.0], sampleRate: 12_000)
    let e = try evaluator(
      """
      (def a (audio-tensor @file "tiny.wav"))
      (out (peek a 0) 1 @name audio)
      """)
    XCTAssertEqual(try tensor(e, "a").shape, [4])
    XCTAssertEqual(e.tensors.last?.kind, "audio")

    let graph = LazyGraphContext.current
    for output in e.outputs {
      graph.addOutput(output.signal, channel: output.channel)
    }
    let compilation = try graph.compileOnly(frameCount: 16, voiceCount: 1)
    let manifest = generateManifest(
      compilerResult: CompilerResult(
        dylibPath: "",
        cSourcePath: "",
        compilationResult: compilation,
        cSource: ""
      ),
      evaluator: e,
      options: CompilerOptions(
        outputDir: ".",
        name: "patch",
        sampleRate: 48_000,
        maxFrames: 16,
        voiceCount: 1,
        debug: false
      )
    )

    XCTAssertEqual(manifest.tensors.first?.sourceSampleRate, 12_000)
  }

  func testIRAliasUsesIRKind() throws {
    _ = try makeWav("ir.wav", samples: [1.0, 0.0, 0.0, 0.0])
    let e = try evaluator(
      """
      (def h (ir @file "ir.wav"))
      """)
    XCTAssertEqual(try tensor(e, "h").shape, [4])
    XCTAssertEqual(e.tensors.last?.kind, "ir")
  }

  func testAudioTensorTrimAndNormalize() throws {
    _ = try makeWav("trim.wav", samples: [0.0, 0.25, -0.5, 1.0], sampleRate: 4)
    let e = try evaluator(
      """
      (def a (audio-tensor @file "trim.wav" @start 0.25 @end 0.75 @normalize peak))
      """)
    XCTAssertEqual(try tensor(e, "a").shape, [2])
    XCTAssertEqual(e.tensors.last?.data?.map { Float(round($0 * 1000) / 1000) }, [0.5, -1.0])
  }

  func testAudioTensorCanExtractSpecificChannel() throws {
    try makeStereoWav("stereo.wav", frames: [(1, 10), (2, 20), (3, 30)])
    let e = try evaluator(
      """
      (def right (audio-tensor @file "stereo.wav" @channel 1))
      """)
    XCTAssertEqual(try tensor(e, "right").shape, [3])
    XCTAssertEqual(e.tensors.last?.data, [10, 20, 30])
  }

  // MARK: - Partitioned Convolution

  func testPartitionIRReturnsTwoTensors() throws {
    let e = try evaluator(
      """
      (def h (tensor @shape [4] @data [1 0 0 0]))
      (def (hre him) (partition-ir h @N 4 @hop 2))
      """)
    XCTAssertEqual(try tensor(e, "hre").shape, [8])
    XCTAssertEqual(try tensor(e, "him").shape, [8])
  }

  func testPartitionedSpectralMACBuildsOutputs() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def (xre xim) (fft frame @N 4 @backend accelerated))
      (def h (tensor @shape [4] @data [1 0 0 0]))
      (def (hre him) (partition-ir h @N 4 @hop 2))
      (def (yre yim) (partitioned-spectral-mac xre xim hre him @N 4))
      """)
    XCTAssertEqual(try signalTensor(e, "yre").shape, [4])
    XCTAssertEqual(try signalTensor(e, "yim").shape, [4])
  }

  func testPartitionedConvolveCompiles() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def h (tensor @shape [4] @data [1 0 0 0]))
      (out (partitioned-convolve input h @N 4 @hop 2) 1)
      """)
    try compileOutputs(e, frames: 8)
  }

  // MARK: - Tensor Ops

  func testConv1dStaticTensorShape() throws {
    let e = try evaluator(
      """
      (def x (tensor @shape [3] @data [1 2 3]))
      (def k (tensor @shape [3] @data [0 1 0]))
      (def y (conv1d x k))
      """)
    XCTAssertEqual(try tensor(e, "y").shape, [3])
  }

  func testConv1dSignalTensorShape() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def frame (reshape (buffer input 4 2) @shape [4]))
      (def k (tensor @shape [3] @data [0 1 0]))
      (def y (conv1d frame k))
      """)
    XCTAssertEqual(try signalTensor(e, "y").shape, [4])
  }

  func testWindowsExtractsExpectedShape() throws {
    let e = try evaluator(
      """
      (def x (tensor @shape [3 3] @data [1 2 3 4 5 6 7 8 9]))
      (def w (windows x @shape [2 2]))
      """)
    XCTAssertEqual(try tensor(e, "w").shape, [2, 2, 2, 2])
  }

  func testSignalTensorScalarClampAndPaddingSyntax() throws {
    let e = try evaluator(
      """
      (make-tensor-history h @shape [2 2] @data [0.12 0.12 0.12 0.12])
      (def state (read-tensor-history h))
      (def clamped (max (min state 0.24) 0.01))
      (def padded (pad clamped @padding [1:1 1:1]))
      """)
    XCTAssertEqual(try signalTensor(e, "clamped").shape, [2, 2])
    XCTAssertEqual(try signalTensor(e, "padded").shape, [4, 4])
  }

  func testSignalTensorPairwiseMinMaxSyntax() throws {
    let e = try evaluator(
      """
      (make-tensor-history a @shape [2 2] @data [0.1 0.8 0.3 0.6])
      (make-tensor-history b @shape [2 2] @data [0.2 0.7 0.4 0.5])
      (def aa (read-tensor-history a))
      (def bb (read-tensor-history b))
      (def hi (max aa bb))
      (def lo (min aa bb))
      (out (sum (- hi lo)) 1)
      """)
    XCTAssertEqual(try signalTensor(e, "hi").shape, [2, 2])
    XCTAssertEqual(try signalTensor(e, "lo").shape, [2, 2])
    try compileOutputs(e, frames: 4)
  }

  // MARK: - Tensor History

  func testTensorHistoryReadReturnsSignalTensor() throws {
    let e = try evaluator(
      """
      (make-tensor-history h @shape [2 2])
      (def state (read-tensor-history h))
      """)
    XCTAssertEqual(try signalTensor(e, "state").shape, [2, 2])
  }

  func testTensorHistoryWriteReturnsWrittenValue() throws {
    let e = try evaluator(
      """
      (make-tensor-history h @shape [2])
      (def x (tensor @shape [2] @data [1 2]))
      (def y (write-tensor-history h x))
      """)
    XCTAssertEqual(try tensor(e, "y").shape, [2])
  }

  func testTensorHistoryCompilesInFeedbackStyleGraph() throws {
    let e = try evaluator(
      """
      (make-tensor-history h @shape [2 2])
      (def state (read-tensor-history h))
      (def k (tensor @shape [2 2] @data [1 0 0 1]))
      (def next (conv2d state k))
      (def written (write-tensor-history h next))
      (out (sum written) 1)
      """)
    try compileOutputs(e, frames: 4)
  }

  func testAcceleratedSpectralFeedbackPatchCompilesToC() throws {
    let e = try evaluator(
      """
      (def input (in 1))
      (def win (sqrt (hann 16)))
      (def frame (* (reshape (buffer input 16 4) @shape [16]) win))
      (def (re im) (fft frame @N 16 @backend accelerated))
      (def mag (sqrt (+ (* re re) (* im im))))

      (def fold-idx
        (tensor @shape [16] @data [0 1 2 3 4 5 6 7 8 7 6 5 4 3 2 1]))
      (def fold-norm
        (tensor @shape [16] @data [0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 0.875 0.75 0.625 0.5 0.375 0.25 0.125]))
      (def bin-sign
        (tensor @shape [16] @data [0 1 1 1 1 1 1 1 0 -1 -1 -1 -1 -1 -1 -1]))
      (def phase-adv
        (tensor @shape [16] @data [0 0.4 0.8 1.2 1.6 2 2.4 2.8 3.2 -2.8 -2.4 -2 -1.6 -1.2 -0.8 -0.4]))

      (def freeze-h (hop-hold (in 2) 4))
      (def drift-s (hop-hold (+ 1 (* (in 3) 0.01)) 4))
      (def bloom-h (hop-hold (+ 0.25 (* 0 (in 4))) 4))

      (make-history bloom-mag @shape [16] @hop 4)
      (def prev (read-history bloom-mag))
      (def drift-idx (* fold-idx drift-s))
      (def idx-lo (floor drift-idx))
      (def idx-frac (- drift-idx idx-lo))
      (def drifted
        (+ (* (gather prev idx-lo) (- 1 idx-frac))
           (* (gather prev (+ idx-lo 1)) idx-frac)))
      (def blur-k (tensor @shape [3] @data [0.25 0.5 0.25]))
      (def diffused
        (+ (* (- 1 bloom-h) drifted) (* bloom-h (conv1d drifted blur-k))))
      (def gain (exp (* -0.1 (+ 1 fold-norm))))
      (def next-mag (max (* mag (- 1 freeze-h)) (* diffused gain)))
      (write-history bloom-mag next-mag)
      (def cloud-mag (hop-hold next-mag 4))

      (make-history bloom-phase @shape [16] @hop 4)
      (def ph-prev (read-history bloom-phase))
      (def jit (* (gather (noise @size 16 @hop 4) fold-idx) bin-sign 0.1))
      (def ph-next (wrap (+ ph-prev phase-adv jit) 0 twopi))
      (write-history bloom-phase ph-next)
      (def ph (hop-hold ph-next 4))

      (def wet-re (* cloud-mag (cos ph)))
      (def wet-im (* cloud-mag (sin ph)))
      (def wet
        (overlap-add (* (ifft wet-re wet-im @N 16 @backend accelerated) win) 4))
      (out wet 1)
      """)

    let result = try compilePatch(
      graph: LazyGraphContext.current,
      outputs: e.outputs,
      options: CompilerOptions(
        outputDir: tempDir.path,
        name: "accelerated-spectral-feedback",
        sampleRate: 8,
        maxFrames: 32,
        voiceCount: 1,
        skipInlineAudit: true,
        debug: false
      ))

    XCTAssertFalse(
      result.cSource.split(separator: "\n").contains {
        $0.contains("int t") && $0.contains("vdivq_f32")
      })
    XCTAssertTrue(FileManager.default.fileExists(atPath: result.dylibPath))
  }

  func testBendingMetalStylePlateGraphCompiles() throws {
    let e = try evaluator(
      """
      (make-tensor-history state @shape [4 4])
      (make-tensor-history prev @shape [4 4])
      (make-tensor-history tension @shape [4 4]
        @data [0.12 0.12 0.12 0.12
               0.12 0.12 0.12 0.12
               0.12 0.12 0.12 0.12
               0.12 0.12 0.12 0.12])

      (def excitation-pattern
        (tensor @shape [4 4]
          @data [0 0 0 0
                 0 0.8 0.5 0
                 0 0.5 0.3 0
                 0 0 0 0]))
      (def gated-excite (* excitation-pattern (click)))

      (def bend-mod-1 (sin (* (phasor 0.3) twopi)))
      (def bend-mod-2 (sin (* (phasor 0.17) twopi)))
      (def horiz-grad
        (tensor @shape [4 4]
          @data [-1 -1 -1 -1
                 -0.333333 -0.333333 -0.333333 -0.333333
                 0.333333 0.333333 0.333333 0.333333
                 1 1 1 1]))
      (def diag-grad
        (tensor @shape [4 4]
          @data [-1 -0.666667 -0.333333 0
                 -0.666667 -0.333333 0 0.333333
                 -0.333333 0 0.333333 0.666667
                 0 0.333333 0.666667 1]))
      (def bend-field
        (+ (* (* horiz-grad bend-mod-1) 0.035)
           (* (* diag-grad bend-mod-2) 0.021)))

      (def state-t-raw (read-tensor-history state))
      (def state-t-1 (read-tensor-history prev))
      (def tension-t (read-tensor-history tension))
      (def state-t (+ state-t-raw gated-excite))

      (def laplacian-kernel
        (tensor @shape [3 3] @data [0 1 0 1 -4 1 0 1 0]))
      (def laplacian (conv2d (pad state-t @padding [1:1 1:1]) laplacian-kernel))
      (def state-next
        (+ (- (* state-t 1.99999) (* state-t-1 0.99999))
           (* laplacian tension-t)))

      (def velocity (- state-t state-t-1))
      (def local-energy (* velocity velocity))
      (def relaxed (+ (* tension-t 0.9998) 0.000024))
      (def tension-unclamped
        (+ (+ relaxed (* local-energy 0.0003)) bend-field))
      (def tension-next (max (min tension-unclamped 0.24) 0.01))

      (write-tensor-history prev state-t)
      (write-tensor-history state state-next)
      (write-tensor-history tension tension-next)

      (def pickup-mask
        (tensor @shape [4 4]
          @data [0 0 0 0
                 0 1 0.8 0
                 0 0.7 0 0
                 0 0 0 0]))
      (out (sum (* state-next pickup-mask)) 1 @name audio)
      """)
    try compileOutputs(e, frames: 8)
  }
}

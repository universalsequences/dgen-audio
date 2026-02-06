import XCTest

@testable import DGenLazy

/// Audio ML experiments: learning audio synthesis parameters from spectral loss
final class AudioMLTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 44100.0  // Low rate: 64-sample window → 31.25 Hz resolution
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  // MARK: - Fourier Coefficients for Target Waveforms

  /// Returns the Fourier series amplitudes for a square wave (N harmonics)
  /// Square wave = sum of (1/n) * sin(n*w*t) for odd n only
  func squareWaveAmplitudes(_ numHarmonics: Int) -> [Float] {
    return (1...numHarmonics).map { n in
      n % 2 == 1 ? 1.0 / Float(n) : 0.0
    }
  }

  /// Returns the Fourier series amplitudes for a sawtooth wave (N harmonics)
  /// Sawtooth = sum of (1/n) * sin(n*w*t) for all n
  func sawtoothAmplitudes(_ numHarmonics: Int) -> [Float] {
    return (1...numHarmonics).map { n in
      1.0 / Float(n)
    }
  }

  /// Returns the Fourier series amplitudes for a triangle wave (N harmonics)
  /// Triangle = sum of ((-1)^k / n^2) * sin(n*w*t) for odd n, where k = (n-1)/2
  func triangleAmplitudes(_ numHarmonics: Int) -> [Float] {
    return (1...numHarmonics).map { n in
      guard n % 2 == 1 else { return Float(0.0) }
      let k = (n - 1) / 2
      let sign: Float = k % 2 == 0 ? 1.0 : -1.0
      return sign / Float(n * n)
    }
  }

  // MARK: - Harmonic Signal Builder

  /// Build an audio signal from harmonic amplitudes using additive synthesis.
  ///
  /// Creates N phasors at harmonic frequencies (fundamental, 2*fundamental, ...),
  /// converts to sine waves, scales by amplitudes, and mixes to a single Signal.
  ///
  /// - Parameters:
  ///   - amplitudes: Tensor of shape [numHarmonics] with amplitude per harmonic
  ///   - fundamental: Base frequency in Hz
  ///   - numHarmonics: Number of harmonics
  /// - Returns: Mixed audio Signal (sum of weighted sine waves)
  func buildHarmonicSignal(amplitudes: Tensor, fundamental: Float, numHarmonics: Int) -> Signal {
    let frequencies: [Float] = Array((0..<numHarmonics)).map { (Float($0) + 1.0) * fundamental }
    let sines = (Signal.phasor(Tensor(frequencies)) * Float.pi * 2.0).cos()
    return (sines * amplitudes).sum()
  }

  // MARK: - 808 Kick Learning

  func test808KickLearning() throws {
    let sr: Float = 32500.0
    DGenConfig.sampleRate = sr
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = "/tmp/808_kernel.metal"
    LazyGraphContext.reset()

    let numFrames = 4096  // ~126ms: covers transient + early body

    // Load target 808
    let inputURL = projectRoot.appendingPathComponent("Assets/808kicklong.wav")
    let (allSamples, _) = try AudioFile.load(url: inputURL)
    let targetSamples = Array(allSamples.prefix(numFrames))

    // Save target chunk for listening comparison
    try AudioFile.save(
      url: URL(fileURLWithPath: "/tmp/808_target.wav"),
      samples: targetSamples, sampleRate: sr)

    // Target as 1D tensor for peek playback
    let targetTensor = Tensor(targetSamples)

    // === Learnable parameters (initialized near Python analysis values) ===
    let bodyAmp = Signal.param(0.5)  // overall body amplitude
    let bodyDecay = Signal.param(0.9995)  // slow body envelope decay
    let startFreq = Signal.param(65.0)  // initial pitch (before sweep)
    let endFreq = Signal.param(48.0)  // settled pitch ~50Hz
    let freqDecay = Signal.param(0.999)  // pitch sweep rate
    let clickAmp = Signal.param(0.3)  // transient amplitude
    let clickDecay = Signal.param(0.98)  // fast click decay
    let clickFreq = Signal.param(180.0)  // click resonant frequency

    let params: [Signal] = [
      bodyAmp, bodyDecay, startFreq, endFreq,
      freqDecay, clickAmp, clickDecay, clickFreq,
    ]
    let optimizer = Adam(params: params, lr: 0.0005)

    // --- Build functions (called each epoch to rebuild graph fresh) ---

    // Differentiable envelope: starts at 1, decays by `rate` each frame.
    // History starts at 0 so: frame0 = (1 - 0) = 1, frame1 = (1 - decay) * decay + decay...
    // Simpler: prev * rate + (1 - rate) on frame 0 gives 1, then converges.
    // Actually cleanest: write 1.0 first frame, then decay.
    // Since history=0 on frame 0: out = max(prev, seed) * rate works but max isn't smooth.
    // Use: out = prev * rate, write(out + (1-out) * (1-prev))
    // Frame 0: prev=0, out=0, write = 0 + 1*1 = 1  ✓
    // Frame 1: prev=1, out=rate, write = rate + (1-rate)*0 = rate  ✓
    // Frame 2: prev=rate, out=rate^2, write = rate^2  ✓
    func envelope(_ rate: Signal) -> Signal {
      let (prev, write) = Signal.history()
      let out = prev * rate
      // Seed: (1-out)*(1-prev) is 1 only when both are 0 (frame 0), else ~0
      write(out + (Signal.constant(1.0) - out) * (Signal.constant(1.0) - prev))
      return out
    }

    func buildSynth() -> Signal {
      let bodyEnv = envelope(bodyDecay)
      let freqEnv = envelope(freqDecay)

      // Swept frequency: endFreq + (startFreq - endFreq) * freqEnv
      let freq = endFreq + (startFreq - endFreq) * freqEnv
      let body = sin(Signal.phasor(freq) * Float.pi * 2.0) * bodyEnv * bodyAmp

      // Click transient: high-freq sine burst with fast decay
      let clickEnv = envelope(clickDecay)
      let click = sin(Signal.phasor(clickFreq) * Float.pi * 2.0) * clickEnv * clickAmp

      return body + click
    }

    func buildTarget() -> Signal {
      return targetTensor.toSignal(maxFrames: numFrames)
    }

    let windowSize = 2048
    // --- Warmup (first compile) ---
    do {
      let s = buildSynth()
      let t = buildTarget()
      let loss = spectralLossFFT(s, t, windowSize: windowSize)
      let initialLoss = try loss.backward(frames: numFrames)
      print("INITIAL LOSS = \(initialLoss.reduce(0, +) / Float(numFrames))")
      optimizer.zeroGrad()
      DGenConfig.kernelOutputPath = nil
    }

    // --- Training loop ---
    print(
      "\n=== 808 Kick Learning (\(numFrames) frames, \(String(format: "%.0f", Float(numFrames) / sr * 1000))ms) ==="
    )
    var firstLoss: Float = 0
    var lastLoss: Float = 0
    let epochs = 400

    for epoch in 0..<epochs {
      let synth = buildSynth()
      let target = buildTarget()
      let loss = spectralLossFFT(synth, target, windowSize: windowSize)

      let lossValues = try loss.backward(frames: numFrames)
      let epochLoss = lossValues.reduce(0, +) / Float(numFrames)

      if epoch == 0 { firstLoss = epochLoss }
      lastLoss = epochLoss

      if epoch % 1 == 0 || epoch == epochs - 1 {
        let pv = params.compactMap { $0.data }
        print(
          "Epoch \(epoch): loss=\(String(format: "%.4f", epochLoss))")
        print(
          "  bodyAmp=\(String(format:"%.4f",pv[0])) bodyDecay=\(String(format:"%.6f",pv[1])) "
            + "startFreq=\(String(format:"%.2f",pv[2])) endFreq=\(String(format:"%.2f",pv[3]))")
        print(
          "  freqDecay=\(String(format:"%.6f",pv[4])) clickAmp=\(String(format:"%.4f",pv[5])) "
            + "clickDecay=\(String(format:"%.6f",pv[6])) clickFreq=\(String(format:"%.2f",pv[7]))")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("\nFirst loss: \(firstLoss), Final loss: \(lastLoss)")
    print(
      "Reduction: \(String(format: "%.2fx", firstLoss / max(lastLoss, 1e-10)))")

    XCTAssertLessThan(lastLoss, firstLoss, "Loss should decrease during training")

    // --- Export learned synth for listening ---
    let finalSynth = buildSynth()
    let synthSamples = try finalSynth.realize(frames: numFrames)
    try AudioFile.save(
      url: URL(fileURLWithPath: "/tmp/808_learned.wav"),
      samples: synthSamples, sampleRate: sr)

    print("Exported: /tmp/808_target.wav and /tmp/808_learned.wav")
  }

  // MARK: - Timbre Matching Tests

  func testSquareWaveTimbreMatch() throws {
    let numHarmonics = 32
    let fundamental: Float = 100.0  // 100 Hz fundamental
    let frameCount = 512
    let windowSize = 64

    // Teacher: known square wave amplitudes
    let targetAmps = squareWaveAmplitudes(numHarmonics)
    let teacherAmplitudes = Tensor(targetAmps)

    // Student: start with uniform amplitudes (all 0.5)
    let studentAmplitudes = Tensor.param(
      [numHarmonics],
      data: [Float](repeating: 0.5, count: numHarmonics)
    )

    let optimizer = Adam(params: [studentAmplitudes], lr: 0.8)

    // Warmup
    let teacherSignal = buildHarmonicSignal(
      amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let studentSignal = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let warmupLoss = spectralLossFFT(studentSignal, teacherSignal, windowSize: windowSize)
    _ = try warmupLoss.backward(frames: frameCount)
    optimizer.zeroGrad()

    // Get initial loss
    let (initTeacher, initStudent) = (
      buildHarmonicSignal(
        amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics),
      buildHarmonicSignal(
        amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    )
    let initLossValues = try spectralLossFFT(initStudent, initTeacher, windowSize: windowSize)
      .backward(frames: frameCount)
    let initialLoss = initLossValues.reduce(0, +) / Float(frameCount)
    optimizer.zeroGrad()
    print("\n=== Square Wave Timbre Matching ===")
    print("Initial loss: \(initialLoss)")

    // Training loop
    let epochs = 500
    var finalLoss = initialLoss
    for epoch in 0..<epochs {
      let teacher = buildHarmonicSignal(
        amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let student = buildHarmonicSignal(
        amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)

      let lossValues = try loss.backward(frames: frameCount)
      let epochLoss = lossValues.reduce(0, +) / Float(frameCount)
      finalLoss = epochLoss

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let learnedAmps = studentAmplitudes.getData() ?? []
        print(
          "Epoch \(epoch): loss = \(String(format: "%.6f", epochLoss)), "
            + "amps = \(learnedAmps.prefix(4).map { String(format: "%.3f", $0) })")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    // Verify loss decreased significantly
    print(
      "Final loss: \(finalLoss), reduction: \(String(format: "%.2fx", initialLoss / finalLoss))")
    XCTAssertLessThan(finalLoss, initialLoss * 0.3, "Should achieve >3x loss reduction")

    // Verify learned amplitudes approach target
    let learnedAmps = studentAmplitudes.getData() ?? []
    print("\nTarget amplitudes:  \(targetAmps.map { String(format: "%.3f", $0) })")
    print("Learned amplitudes: \(learnedAmps.map { String(format: "%.3f", $0) })")

    // Odd harmonics should be larger than even harmonics
    for i in stride(from: 0, to: numHarmonics - 1, by: 2) {
      XCTAssertGreaterThan(
        abs(learnedAmps[i]), abs(learnedAmps[i + 1]) + 0.01,
        "Odd harmonic \(i + 1) should be larger than even harmonic \(i + 2)")
    }
  }

  func testSawtoothTimbreMatch() throws {
    let numHarmonics = 8
    let fundamental: Float = 100.0
    let frameCount = 128
    let windowSize = 64

    let targetAmps = sawtoothAmplitudes(numHarmonics)
    let teacherAmplitudes = Tensor(targetAmps)

    let studentAmplitudes = Tensor.param(
      [numHarmonics],
      data: [Float](repeating: 0.3, count: numHarmonics)
    )

    let optimizer = Adam(params: [studentAmplitudes], lr: 0.01)

    // Warmup
    let t0 = buildHarmonicSignal(
      amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let s0 = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    _ = try spectralLossFFT(s0, t0, windowSize: windowSize).backward(frames: frameCount)
    optimizer.zeroGrad()

    // Initial loss
    let t1 = buildHarmonicSignal(
      amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let s1 = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let initLossValues = try spectralLossFFT(s1, t1, windowSize: windowSize)
      .backward(frames: frameCount)
    let initialLoss = initLossValues.reduce(0, +) / Float(frameCount)
    optimizer.zeroGrad()
    print("\n=== Sawtooth Timbre Matching ===")
    print("Initial loss: \(initialLoss)")

    let epochs = 50
    var finalLoss = initialLoss
    for epoch in 0..<epochs {
      let teacher = buildHarmonicSignal(
        amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let student = buildHarmonicSignal(
        amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)

      let lossValues = try loss.backward(frames: frameCount)
      finalLoss = lossValues.reduce(0, +) / Float(frameCount)

      if epoch % 10 == 0 || epoch == epochs - 1 {
        print("Epoch \(epoch): loss = \(String(format: "%.6f", finalLoss))")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print(
      "Final loss: \(finalLoss), reduction: \(String(format: "%.2fx", initialLoss / finalLoss))")
    XCTAssertLessThan(finalLoss, initialLoss * 0.3, "Should achieve >3x loss reduction")

    // All harmonics should be non-zero (sawtooth uses all harmonics)
    let learnedAmps = studentAmplitudes.getData() ?? []
    print("Target:  \(targetAmps.map { String(format: "%.3f", $0) })")
    print("Learned: \(learnedAmps.map { String(format: "%.3f", $0) })")
    for i in 0..<numHarmonics {
      XCTAssertGreaterThan(
        abs(learnedAmps[i]), 0.01,
        "Harmonic \(i + 1) should be non-zero for sawtooth")
    }
  }

  func testTriangleTimbreMatch() throws {
    let numHarmonics = 8
    let fundamental: Float = 100.0
    let frameCount = 128
    let windowSize = 64

    let targetAmps = triangleAmplitudes(numHarmonics)
    let teacherAmplitudes = Tensor(targetAmps)

    // Start uniform — triangle has small high-freq amplitudes (1/n²), so start small
    let studentAmplitudes = Tensor.param(
      [numHarmonics],
      data: [Float](repeating: 0.2, count: numHarmonics)
    )

    let optimizer = Adam(params: [studentAmplitudes], lr: 0.01)

    // Warmup
    let t0 = buildHarmonicSignal(
      amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let s0 = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    _ = try spectralLossFFT(s0, t0, windowSize: windowSize).backward(frames: frameCount)
    optimizer.zeroGrad()

    // Initial loss
    let t1 = buildHarmonicSignal(
      amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let s1 = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let initLossValues = try spectralLossFFT(s1, t1, windowSize: windowSize)
      .backward(frames: frameCount)
    let initialLoss = initLossValues.reduce(0, +) / Float(frameCount)
    optimizer.zeroGrad()
    print("\n=== Triangle Wave Timbre Matching ===")
    print("Initial loss: \(initialLoss)")
    print("Target amps: \(targetAmps.map { String(format: "%.4f", $0) })")

    let epochs = 50
    var finalLoss = initialLoss
    for epoch in 0..<epochs {
      let teacher = buildHarmonicSignal(
        amplitudes: teacherAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let student = buildHarmonicSignal(
        amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)

      let lossValues = try loss.backward(frames: frameCount)
      finalLoss = lossValues.reduce(0, +) / Float(frameCount)

      if epoch % 10 == 0 || epoch == epochs - 1 {
        print("Epoch \(epoch): loss = \(String(format: "%.6f", finalLoss))")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print(
      "Final loss: \(finalLoss), reduction: \(String(format: "%.2fx", initialLoss / finalLoss))")
    XCTAssertLessThan(finalLoss, initialLoss * 0.3, "Should achieve >3x loss reduction")

    // Verify learned amplitudes
    let learnedAmps = studentAmplitudes.getData() ?? []
    print("Target:  \(targetAmps.map { String(format: "%.4f", $0) })")
    print("Learned: \(learnedAmps.map { String(format: "%.4f", $0) })")
  }

  // MARK: - AudioFile Load / Export

  /// Resolve project root from the test file's location
  private var projectRoot: URL {
    // #file → .../Tests/DGenLazyTests/AudioMLTests.swift  →  go up 3 levels
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  func testLoadAndExport808() throws {
    let inputURL = projectRoot.appendingPathComponent("Assets/808kicklong.wav")

    // Load the WAV
    let (samples, sampleRate) = try AudioFile.load(url: inputURL)

    print("\n=== 808 Kick Load Test ===")
    print("Sample rate: \(sampleRate) Hz")
    print("Sample count: \(samples.count)")
    print("Duration: \(String(format: "%.3f", Float(samples.count) / sampleRate)) sec")
    print("Peak amplitude: \(samples.map { abs($0) }.max() ?? 0)")
    print("First 8 samples: \(samples.prefix(8).map { String(format: "%.4f", $0) })")

    // Basic sanity checks
    XCTAssertEqual(sampleRate, 32500.0, accuracy: 1.0, "Expected ~32500 Hz sample rate")
    XCTAssertGreaterThan(samples.count, 30000, "Should have ~37k samples")
    XCTAssertGreaterThan(samples.map { abs($0) }.max() ?? 0, 0.01, "Should have non-silent audio")

    // Round-trip: save to /tmp and reload
    let outputURL = URL(fileURLWithPath: "/tmp/808kicklong_roundtrip.wav")
    try AudioFile.save(url: outputURL, samples: samples, sampleRate: sampleRate)

    let (reloaded, reloadedRate) = try AudioFile.load(url: outputURL)
    XCTAssertEqual(reloadedRate, sampleRate, accuracy: 1.0, "Sample rate should survive round-trip")
    XCTAssertEqual(reloaded.count, samples.count, "Sample count should match")

    // Compare values (int16 → float32 → save as float32 → reload should be lossless)
    var maxDiff: Float = 0
    for i in 0..<min(samples.count, reloaded.count) {
      maxDiff = max(maxDiff, abs(samples[i] - reloaded[i]))
    }
    print("Round-trip max sample diff: \(maxDiff)")
    XCTAssertLessThan(maxDiff, 1e-5, "Round-trip should be nearly lossless")

    // Also test loadTensor
    LazyGraphContext.reset()
    let tensor = try AudioFile.loadTensor(url: inputURL)
    let tensorData = tensor.getData() ?? []
    XCTAssertEqual(tensorData.count, samples.count, "Tensor should have same sample count")

    print("808 kick loaded successfully into Tensor with shape \(tensor.shape)")
  }

  // MARK: - Raw Waveform Timbre Matching
  // Teacher is a raw waveform (not built from harmonics).
  // Student must learn the best N-harmonic approximation.

  /// Helper: run a training loop matching student harmonics against a raw teacher signal
  func trainAgainstRawWaveform(
    name: String,
    fundamental: Float,
    numHarmonics: Int,
    frameCount: Int,
    windowSize: Int,
    epochs: Int,
    buildTeacher: (Float) -> Signal
  ) throws -> (initialLoss: Float, finalLoss: Float, learnedAmps: [Float]) {
    let studentAmplitudes = Tensor.param(
      [numHarmonics],
      data: [Float](repeating: 0.3, count: numHarmonics)
    )
    let optimizer = Adam(params: [studentAmplitudes], lr: 0.01)

    // Warmup
    let warmupTeacher = buildTeacher(fundamental)
    let warmupStudent = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    _ = try spectralLossFFT(warmupStudent, warmupTeacher, windowSize: windowSize)
      .backward(frames: frameCount)
    optimizer.zeroGrad()

    // Initial loss
    let initTeacher = buildTeacher(fundamental)
    let initStudent = buildHarmonicSignal(
      amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
    let initLossValues = try spectralLossFFT(initStudent, initTeacher, windowSize: windowSize)
      .backward(frames: frameCount)
    let initialLoss = initLossValues.reduce(0, +) / Float(frameCount)
    optimizer.zeroGrad()
    print("\n=== Raw \(name) Timbre Matching ===")
    print("Initial loss: \(initialLoss)")

    // Training loop
    var finalLoss = initialLoss
    for epoch in 0..<epochs {
      let teacher = buildTeacher(fundamental)
      let student = buildHarmonicSignal(
        amplitudes: studentAmplitudes, fundamental: fundamental, numHarmonics: numHarmonics)
      let loss = spectralLossFFT(student, teacher, windowSize: windowSize)

      let lossValues = try loss.backward(frames: frameCount)
      finalLoss = lossValues.reduce(0, +) / Float(frameCount)

      if epoch % 10 == 0 || epoch == epochs - 1 {
        print("Epoch \(epoch): loss = \(String(format: "%.6f", finalLoss))")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    let learnedAmps = studentAmplitudes.getData() ?? []
    print(
      "Final loss: \(finalLoss), reduction: \(String(format: "%.2fx", initialLoss / finalLoss))")
    print("Learned amplitudes: \(learnedAmps.map { String(format: "%.4f", $0) })")
    return (initialLoss, finalLoss, learnedAmps)
  }

  func testRawSquareWaveTimbreMatch() throws {
    let result = try trainAgainstRawWaveform(
      name: "Square Wave",
      fundamental: 100.0,
      numHarmonics: 8,
      frameCount: 128,
      windowSize: 64,
      epochs: 50,
      buildTeacher: { fundamental in
        let phase = Signal.phasor(fundamental)
        return gswitch(phase < Signal.constant(0.5), 1.0, -1.0)
      }
    )

    XCTAssertLessThan(
      result.finalLoss, result.initialLoss * 0.5, "Should achieve >2x loss reduction")

    // Odd harmonics should dominate (square wave property)
    for i in stride(from: 0, to: 7, by: 2) {
      XCTAssertGreaterThan(
        abs(result.learnedAmps[i]), abs(result.learnedAmps[i + 1]),
        "Odd harmonic \(i + 1) should be larger than even harmonic \(i + 2)")
    }
  }

  func testRawSawtoothTimbreMatch() throws {
    let result = try trainAgainstRawWaveform(
      name: "Sawtooth",
      fundamental: 100.0,
      numHarmonics: 10,
      frameCount: 128,
      windowSize: 128,
      epochs: 50,
      buildTeacher: { fundamental in
        // Sawtooth: linear ramp from -1 to +1
        let phase = Signal.phasor(fundamental)
        return phase * 2.0 - 1.0
      }
    )

    XCTAssertLessThan(
      result.finalLoss, result.initialLoss * 0.5, "Should achieve >2x loss reduction")

    // All harmonics should be non-zero, decreasing in magnitude
    for i in 0..<8 {
      XCTAssertGreaterThan(
        abs(result.learnedAmps[i]), 0.01,
        "Harmonic \(i + 1) should be non-zero for sawtooth")
    }
  }

  func testRawTriangleTimbreMatch() throws {
    let result = try trainAgainstRawWaveform(
      name: "Triangle",
      fundamental: 100.0,
      numHarmonics: 8,
      frameCount: 128,
      windowSize: 64,
      epochs: 50,
      buildTeacher: { fundamental in
        // Triangle: |phase * 4 - 2| - 1  →  peaks at ±1, linear slopes
        let phase = Signal.phasor(fundamental)
        return abs(phase * 4.0 - 2.0) - 1.0
      }
    )

    XCTAssertLessThan(
      result.finalLoss, result.initialLoss * 0.5, "Should achieve >2x loss reduction")

    // Odd harmonics should dominate (triangle wave property)
    // But amplitudes fall off as 1/n², so higher harmonics are tiny
    XCTAssertGreaterThan(
      abs(result.learnedAmps[0]), abs(result.learnedAmps[1]),
      "Fundamental should be larger than 2nd harmonic")
  }
}

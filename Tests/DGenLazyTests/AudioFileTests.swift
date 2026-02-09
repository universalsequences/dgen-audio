import XCTest

@testable import DGenLazy

/// Simple audio synthesis tests — generate and export WAV files for listening
final class AudioFileTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  /// Resolve project root from the test file's location
  private var projectRoot: URL {
    URL(fileURLWithPath: #file)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  // MARK: - Basic 808 Kick Synthesis

  func testGenerate808Kick() throws {
    let sr: Float = 44100.0
    DGenConfig.sampleRate = sr
    let duration: Float = 1.0
    let numFrames = Int(sr * duration)
    DGenConfig.maxFrameCount = numFrames
    LazyGraphContext.reset()  // recreate graph with new maxFrameCount

    // --- Parametric 808 kick (rates tuned for 44100 Hz) ---

    // Pitch sweep: starts high, drops to fundamental in ~20ms
    let startFreq: Float = 150.0
    let endFreq: Float = 45.0
    let freqDecay: Float = 0.998  // half-life ~11ms at 44.1kHz

    // Body envelope: long sustain (~500ms half-life)
    let bodyAmp: Float = 0.8
    let bodyDecay: Float = 0.99997  // half-life ~500ms at 44.1kHz

    // Click transient: very short burst (~2ms)
    let clickAmp: Float = 0.6
    let clickDecay: Float = 0.99  // half-life ~1.6ms at 44.1kHz
    let clickFreq: Float = 300.0

    // Exponential decay envelope using accum to detect first frame
    let t = Signal.accum(Signal.constant(1.0), reset: 0.0, min: 0.0, max: Float(numFrames + 1))
    func envelope(_ rate: Signal) -> Signal {
      let (prev, write) = Signal.history()
      let seed = max(Signal.constant(1.0) - t, 0.0)  // 1 on frame 0, 0 after
      return write(prev * rate + seed)
    }

    // Build the synth
    let bodyEnv = envelope(Signal.constant(bodyDecay))
    let freqEnv = envelope(Signal.constant(freqDecay))

    // Swept frequency
    let freq =
      Signal.constant(endFreq)
      + (Signal.constant(startFreq) - Signal.constant(endFreq)) * freqEnv
    let body =
      sin(Signal.phasor(freq) * Float.pi * 2.0)
      * bodyEnv * Signal.constant(bodyAmp)

    // Click
    let clickEnv = envelope(Signal.constant(clickDecay))
    let click =
      sin(Signal.phasor(Signal.constant(clickFreq)) * Float.pi * 2.0)
      * clickEnv * Signal.constant(clickAmp)

    let kick = body + click

    // --- Diagnostic: realize envelopes independently ---
    print("\n=== Envelope Diagnostics ===")

    let bodyEnvSamples = try bodyEnv.realize(frames: numFrames)
    let freqEnvSamples = try freqEnv.realize(frames: numFrames)
    let clickEnvSamples = try clickEnv.realize(frames: numFrames)

    let checkPoints = [0, 1, 10, 100, 500, 1000, 4410, 22050, 44099]
    for i in checkPoints {
      print(
        "frame \(String(format: "%5d", i)): bodyEnv=\(String(format: "%.6f", bodyEnvSamples[i]))  freqEnv=\(String(format: "%.6f", freqEnvSamples[i]))  clickEnv=\(String(format: "%.6f", clickEnvSamples[i]))"
      )
    }

    // Realize and export
    let samples = try kick.realize(frames: numFrames)

    let peakAmp = samples.map { abs($0) }.max() ?? 0
    print("\n=== 808 Kick Generator ===")
    print("Sample rate: \(sr) Hz")
    print("Duration: \(duration)s (\(numFrames) frames)")
    print("Peak amplitude: \(String(format: "%.4f", peakAmp))")
    print("Params: startFreq=\(startFreq) endFreq=\(endFreq) freqDecay=\(freqDecay)")
    print("        bodyAmp=\(bodyAmp) bodyDecay=\(bodyDecay)")
    print("        clickAmp=\(clickAmp) clickDecay=\(clickDecay) clickFreq=\(clickFreq)")

    let outputPath = "/tmp/808_baseline.wav"
    try AudioFile.save(
      url: URL(fileURLWithPath: outputPath),
      samples: samples, sampleRate: sr)
    print("Exported: \(outputPath)")

    XCTAssertGreaterThan(peakAmp, 0.1, "Should produce audible output")
  }

  // MARK: - Simple 808 Kick Learning

  func testGenerate808KickSimple() throws {
    let sr: Float = 44100.0
    DGenConfig.sampleRate = sr
    DGenConfig.debug = false
    DGenConfig.kernelOutputPath = "/tmp/808_simple_kernel.metal"
    let numFrames = 4096 * 4  // ~186ms: covers transient + early body
    let prevMaxFrameCount = DGenConfig.maxFrameCount
    DGenConfig.maxFrameCount = numFrames
    defer { DGenConfig.maxFrameCount = prevMaxFrameCount }
    LazyGraphContext.reset()

    // Load target 808
    let inputURL = projectRoot.appendingPathComponent("Assets/808kicklong.wav")
    let (allSamples, _) = try AudioFile.load(url: inputURL)
    let targetSamples = Array(allSamples.prefix(numFrames))
    let targetTensor = Tensor(targetSamples)

    try AudioFile.save(
      url: URL(fileURLWithPath: "/tmp/808_simple_target.wav"),
      samples: targetSamples, sampleRate: sr)

    // === Learnable parameters (same architecture as testGenerate808Kick) ===
    // NOTE: Envelopes use analytical exp(log(decay)*t) instead of history() feedback,
    // because history() BPTT carry cells propagate gradients forward-through-time
    // (wrong direction), causing decay param gradients to be incorrect.
    let startFreq = Signal.param(80.0, min: 20.0)
    let endFreq = Signal.param(50.0, min: 20.0)
    // Reparameterize decay rates as log-decay (unconstrained, negative = decay)
    // envelope = exp(logDecay * t), where logDecay < 0 gives exponential decay
    let freqLogDecay = Signal.param(log(0.998), max: -1e-6)  // ~log(0.998)
    let bodyLogDecay = Signal.param(log(0.99997), max: -1e-6)  // ~log(0.99997)
    let bodyAmp = Signal.param(0.5, min: 0.0)
    let clickLogDecay = Signal.param(log(0.99), max: -1e-6)  // ~log(0.99)
    let clickAmp = Signal.param(0.3, min: 0.0)
    let clickFreq = Signal.param(200.0, min: 20.0)

    let params: [Signal] = [
      startFreq, endFreq, freqLogDecay,
      bodyAmp, bodyLogDecay,
      clickAmp, clickLogDecay, clickFreq,
    ]
    let paramNames = [
      "startFreq", "endFreq", "freqLogDecay",
      "bodyAmp", "bodyLogDecay",
      "clickAmp", "clickLogDecay", "clickFreq",
    ]
    // Per-group learning rates: Adam normalizes steps to ~lr, so decay params
    // (natural scale ~1e-5 to ~1e-2) need much smaller lr than freq/amp params.
    let freqOptimizer = Adam(params: [startFreq, endFreq, clickFreq], lr: 1.0)
    let ampOptimizer = Adam(params: [bodyAmp, clickAmp], lr: 0.01)
    let decayOptimizer = Adam(params: [freqLogDecay, bodyLogDecay, clickLogDecay], lr: 0.0001)

    // --- Build synth (called each epoch to rebuild graph fresh) ---
    func buildSynth() -> Signal {
      // Frame counter: 0, 1, 2, ..., numFrames-1
      let t = Signal.accum(Signal.constant(1.0), reset: 0.0, min: 0.0, max: Float(numFrames + 1))

      // Analytical envelopes: exp(logDecay * t) = decay^t
      // Avoids history() feedback which has broken BPTT gradients.
      let freqEnv = exp(freqLogDecay * t)
      let freq = endFreq + (startFreq - endFreq) * freqEnv

      let bodyEnv = exp(bodyLogDecay * t)
      let body = sin(Signal.phasor(freq) * Float.pi * 2.0) * bodyEnv * bodyAmp

      let clickEnv = exp(clickLogDecay * t)
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
      let loss = spectralLossFFT(s, t, windowSize: windowSize, normalize: true)
      let initialLoss = try loss.backward(frames: numFrames)
      print("INITIAL LOSS = \(initialLoss.reduce(0, +))")
      freqOptimizer.zeroGrad()
      ampOptimizer.zeroGrad()
      decayOptimizer.zeroGrad()
      DGenConfig.kernelOutputPath = nil
    }

    // --- Training loop ---
    print(
      "\n=== 808 Kick Simple Learning (\(numFrames) frames, \(String(format: "%.0f", Float(numFrames) / sr * 1000))ms) ==="
    )
    var firstLoss: Float = 0
    var lastLoss: Float = 0
    let epochs = 50

    for epoch in 0..<epochs {
      let synth = buildSynth()
      let target = buildTarget()
      let loss = spectralLossFFT(synth, target, windowSize: windowSize, normalize: true)

      let lossValues = try loss.backward(frames: numFrames)
      let epochLoss = lossValues.reduce(0, +)

      if epoch == 0 { firstLoss = epochLoss }
      lastLoss = epochLoss

      if epoch % 10 == 0 || epoch == epochs - 1 {
        let pv = params.compactMap { $0.data }
        let gv = params.map { $0.grad?.data ?? Float.nan }

        print("Epoch \(epoch): loss=\(String(format: "%.4f", epochLoss))")
        for (i, name) in paramNames.enumerated() {
          print(
            "  \(name)=\(String(format:"%12.6f", pv[i]))  grad=\(String(format:"%12.6e", gv[i]))")
        }
      }

      if epochLoss.isNaN || epochLoss.isInfinite {
        print("*** NaN/Inf detected at epoch \(epoch) — stopping ***")
        break
      }

      freqOptimizer.step()
      ampOptimizer.step()
      decayOptimizer.step()
      freqOptimizer.zeroGrad()
      ampOptimizer.zeroGrad()
      decayOptimizer.zeroGrad()
    }

    print("\nFirst loss: \(firstLoss), Final loss: \(lastLoss)")
    print("Reduction: \(String(format: "%.2fx", firstLoss / max(lastLoss, 1e-10)))")

    XCTAssertLessThan(lastLoss, firstLoss, "Loss should decrease during training")

    // --- Export learned synth for listening ---
    let finalSynth = buildSynth()
    let synthSamples = try finalSynth.realize(frames: numFrames)
    try AudioFile.save(
      url: URL(fileURLWithPath: "/tmp/808_simple_learned.wav"),
      samples: synthSamples, sampleRate: sr)

    print("Exported: /tmp/808_simple_target.wav and /tmp/808_simple_learned.wav")
  }
}

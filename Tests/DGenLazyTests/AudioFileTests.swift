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
      let out = prev * rate + seed
      write(out)
      return out
    }

    // Build the synth
    let bodyEnv = envelope(Signal.constant(bodyDecay))
    let freqEnv = envelope(Signal.constant(freqDecay))

    // Swept frequency
    let freq = Signal.constant(endFreq)
      + (Signal.constant(startFreq) - Signal.constant(endFreq)) * freqEnv
    let body = sin(Signal.phasor(freq) * Float.pi * 2.0)
      * bodyEnv * Signal.constant(bodyAmp)

    // Click
    let clickEnv = envelope(Signal.constant(clickDecay))
    let click = sin(Signal.phasor(Signal.constant(clickFreq)) * Float.pi * 2.0)
      * clickEnv * Signal.constant(clickAmp)

    let kick = body + click

    // --- Diagnostic: realize envelopes independently ---
    print("\n=== Envelope Diagnostics ===")

    let bodyEnvSamples = try bodyEnv.realize(frames: numFrames)
    let freqEnvSamples = try freqEnv.realize(frames: numFrames)
    let clickEnvSamples = try clickEnv.realize(frames: numFrames)

    let checkPoints = [0, 1, 10, 100, 500, 1000, 4410, 22050, 44099]
    for i in checkPoints {
      print("frame \(String(format: "%5d", i)): bodyEnv=\(String(format: "%.6f", bodyEnvSamples[i]))  freqEnv=\(String(format: "%.6f", freqEnvSamples[i]))  clickEnv=\(String(format: "%.6f", clickEnvSamples[i]))")
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
}

import DGen
import XCTest

@testable import DGenLazy

/// Tests for biquad, compressor, and delay DGenLazy wrappers.
/// Each test runs the C backend as oracle, then compares Metal output against it.
final class AudioEffectsTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 44100.0
    DGenConfig.backend = .metal
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44100.0
    super.tearDown()
  }

  // MARK: - Helpers

  /// Run a graph builder on both C and Metal backends, return (cResult, metalResult).
  /// Uses deterministic inputs only (no noise) so both backends see identical data.
  private func realizeBothBackends(
    sampleRate: Float = 44100.0,
    frames: Int = 128,
    build: () -> Signal
  ) throws -> (c: [Float], metal: [Float]) {
    // C backend (oracle)
    DGenConfig.sampleRate = sampleRate
    DGenConfig.backend = .c
    LazyGraphContext.reset()
    let cSignal = build()
    let cResult = try cSignal.realize(frames: frames)

    // Metal backend
    DGenConfig.sampleRate = sampleRate
    DGenConfig.backend = .metal
    LazyGraphContext.reset()
    let metalSignal = build()
    let metalResult = try metalSignal.realize(frames: frames)

    return (cResult, metalResult)
  }

  /// Assert two Float arrays match element-wise within tolerance.
  private func assertClose(
    _ a: [Float], _ b: [Float],
    accuracy: Float = 1e-4,
    _ message: String = "",
    file: StaticString = #file, line: UInt = #line
  ) {
    XCTAssertEqual(a.count, b.count, "Count mismatch \(message)", file: file, line: line)
    var maxDiff: Float = 0
    var maxDiffIdx = 0
    for i in 0..<Swift.min(a.count, b.count) {
      let diff = Swift.abs(a[i] - b[i])
      if diff > maxDiff { maxDiff = diff; maxDiffIdx = i }
    }
    if maxDiff > accuracy {
      XCTFail(
        "Max diff \(maxDiff) at frame \(maxDiffIdx) (C=\(a[maxDiffIdx]) Metal=\(b[maxDiffIdx])) exceeds \(accuracy) \(message)",
        file: file, line: line)
    }
  }

  /// Build a deterministic input signal: sum of two sine waves
  private func makeTestSignal() -> Signal {
    let osc1 = Signal.phasor(440.0)
    let osc2 = Signal.phasor(1000.0)
    return sin(osc1 * 2.0 * Float.pi) * 0.5 + sin(osc2 * 2.0 * Float.pi) * 0.3
  }

  /// Build a simple sine input
  private func makeSine(_ freq: Float) -> Signal {
    return sin(Signal.phasor(freq) * 2.0 * Float.pi)
  }

  // MARK: - Biquad Tests

  func testBiquadLowpass() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 500.0, resonance: 0.707, gain: 1.0, mode: 0)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad lowpass")
  }

  func testBiquadHighpass() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 2000.0, resonance: 0.707, gain: 1.0, mode: 1)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad highpass")
  }

  func testBiquadBandpass() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 1000.0, resonance: 2.0, gain: 1.0, mode: 2)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad bandpass")
  }

  func testBiquadNotch() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 1000.0, resonance: 2.0, gain: 1.0, mode: 5)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad notch")
  }

  func testBiquadAllpass() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 1000.0, resonance: 1.0, gain: 1.0, mode: 4)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad allpass")
  }

  func testBiquadHighShelf() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 3000.0, resonance: 1.0, gain: 2.0, mode: 6)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad high shelf")
  }

  func testBiquadLowShelf() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.biquad(cutoff: 500.0, resonance: 1.0, gain: 2.0, mode: 7)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad low shelf")
  }

  func testBiquadWithSignalParams() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      let cutoff = Signal.constant(1000.0)
      let resonance = Signal.constant(1.0)
      let gain = Signal.constant(1.0)
      let mode = Signal.constant(0.0)
      return input.biquad(cutoff: cutoff, resonance: resonance, gain: gain, mode: mode)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad signal params")
  }

  func testBiquadAttenuatesSineAboveCutoff() throws {
    // Lowpass at 500 Hz on a 1000 Hz sine: output should be attenuated
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let sine = self.makeSine(1000.0)
      return sine.biquad(cutoff: 500.0, resonance: 0.707, gain: 1.0, mode: 0)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad attenuates above cutoff")

    // After settling, the filtered signal amplitude should be less than the input
    let lastC = c.suffix(64)
    let cPeak = lastC.map { Swift.abs($0) }.max()!
    XCTAssertLessThan(cPeak, 0.5, "1000 Hz sine through 500 Hz LP should be attenuated")
  }

  // MARK: - Compressor Tests

  func testCompressorBasic() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      return input.compressor(
        ratio: 4.0, threshold: -20.0, knee: 6.0,
        attack: 0.005, release: 0.05)
    }
    assertClose(c, metal, accuracy: 1e-3, "compressor basic")
  }

  func testCompressorWithSignalParams() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      return input.compressor(
        ratio: Signal.constant(4.0),
        threshold: Signal.constant(-20.0),
        knee: Signal.constant(6.0),
        attack: Signal.constant(0.005),
        release: Signal.constant(0.05))
    }
    assertClose(c, metal, accuracy: 1e-3, "compressor signal params")
  }

  func testCompressorHighRatio() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      return input.compressor(
        ratio: 20.0, threshold: -10.0, knee: 0.1,
        attack: 0.001, release: 0.01)
    }
    assertClose(c, metal, accuracy: 1e-3, "compressor high ratio")
  }

  func testCompressorSoftKnee() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      return input.compressor(
        ratio: 4.0, threshold: -20.0, knee: 20.0,
        attack: 0.01, release: 0.1)
    }
    assertClose(c, metal, accuracy: 1e-3, "compressor soft knee")
  }

  func testCompressorSidechain() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let main = self.makeSine(440.0)
      let sidechain = self.makeSine(100.0)
      return main.compressor(
        ratio: Signal.constant(4.0),
        threshold: Signal.constant(-10.0),
        knee: Signal.constant(6.0),
        attack: Signal.constant(0.005),
        release: Signal.constant(0.05),
        sidechain: sidechain)
    }
    assertClose(c, metal, accuracy: 1e-3, "compressor sidechain")
  }

  func testCompressorReducesLevel() throws {
    // A loud sine through a compressor should have lower peak than input
    let (c, _) = try realizeBothBackends(frames: 512) {
      let input = self.makeSine(440.0)
      return input.compressor(
        ratio: 10.0, threshold: -6.0, knee: 0.1,
        attack: 0.001, release: 0.01)
    }
    let lastC = c.suffix(128)
    let cPeak = lastC.map { Swift.abs($0) }.max()!
    XCTAssertLessThan(cPeak, 1.0, "Compressor should reduce peak level")
  }

  // MARK: - Delay Tests

  func testDelayBasic() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      return input.delay(Signal.constant(10.0))
    }
    assertClose(c, metal, accuracy: 1e-3, "delay basic")
  }

  func testDelayFloat() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      return input.delay(10.0)
    }
    assertClose(c, metal, accuracy: 1e-3, "delay float")
  }

  func testDelayLargeTime() throws {
    let (c, metal) = try realizeBothBackends(frames: 512) {
      let input = self.makeSine(440.0)
      return input.delay(200.0)
    }
    assertClose(c, metal, accuracy: 1e-3, "delay large")
  }

  func testDelayFractional() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      return input.delay(5.5)
    }
    assertClose(c, metal, accuracy: 1e-3, "delay fractional")
  }

  func testDelayModulated() throws {
    // Time-varying delay (chorus-like)
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      let lfo = Signal.phasor(2.0)
      let delayTime = sin(lfo * 2.0 * Float.pi) * 5.0 + 10.0  // 5..15 samples
      return input.delay(delayTime)
    }
    assertClose(c, metal, accuracy: 1e-3, "delay modulated")
  }

  func testDelayZero() throws {
    let (c, metal) = try realizeBothBackends(frames: 128) {
      let input = self.makeSine(440.0)
      return input.delay(0.0)
    }
    assertClose(c, metal, accuracy: 1e-3, "delay zero")
  }

  func testDelayOutputNotSilent() throws {
    // Verify the C backend produces non-zero output for delayed signal
    DGenConfig.sampleRate = 44100.0
    DGenConfig.backend = .c
    LazyGraphContext.reset()

    let input = makeSine(440.0)
    let delayed = input.delay(10.0)
    let result = try delayed.realize(frames: 128)

    // After the delay period, output should be non-zero
    let tail = result.suffix(64)
    let peak = tail.map { Swift.abs($0) }.max()!
    XCTAssertGreaterThan(peak, 0.1, "Delayed sine should produce non-zero output")
  }

  func testDelayFeedback() throws {
    // Feedback delay (echo): output = input + 0.5 * delay(output, 20)
    // This creates a feedback loop that forces scalar execution.
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      let (prev, write) = Signal.history()
      let delayed = prev.delay(20.0)
      let output = input + delayed * 0.5
      let _ = write(output)
      return output
    }
    assertClose(c, metal, accuracy: 1e-3, "delay feedback")
  }

  // MARK: - Combined Effects

  func testBiquadThenCompressor() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeTestSignal()
      let filtered = input.biquad(cutoff: 1000.0, resonance: 1.0, gain: 1.0, mode: 0)
      return filtered.compressor(
        ratio: 4.0, threshold: -20.0, knee: 6.0,
        attack: 0.005, release: 0.05)
    }
    assertClose(c, metal, accuracy: 1e-3, "biquad then compressor")
  }

  func testDelayThenBiquad() throws {
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      let delayed = input.delay(20.0)
      return delayed.biquad(cutoff: 2000.0, resonance: 0.707, gain: 1.0, mode: 0)
    }
    assertClose(c, metal, accuracy: 1e-3, "delay then biquad")
  }

  func testFullChain() throws {
    // Full signal chain: sine → delay → biquad → compressor
    let (c, metal) = try realizeBothBackends(frames: 256) {
      let input = self.makeSine(440.0)
      let delayed = input.delay(10.0)
      let filtered = delayed.biquad(cutoff: 2000.0, resonance: 1.0, gain: 1.0, mode: 0)
      return filtered.compressor(
        ratio: 4.0, threshold: -10.0, knee: 6.0,
        attack: 0.005, release: 0.05)
    }
    assertClose(c, metal, accuracy: 1e-3, "full chain")
  }
}

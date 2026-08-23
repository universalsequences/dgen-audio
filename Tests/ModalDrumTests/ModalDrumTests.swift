import DGen
import DGenLazy
import XCTest

@testable import ModalDrum

final class ModalDrumTests: XCTestCase {
  func testFrequencyGridIsLogSpacedAndFixed() {
    let grid = modalFrequencyGrid(count: 32)
    XCTAssertEqual(grid.first!, 120, accuracy: 1e-4)
    XCTAssertEqual(grid.last!, 14_000, accuracy: 0.05)
    let ratios = zip(grid.dropFirst(), grid).map(/)
    for ratio in ratios.dropFirst() {
      XCTAssertEqual(ratio, ratios[0], accuracy: 1e-5)
    }
  }

  func testParameterTransformsRoundTripAndRespectBounds() {
    LazyGraphContext.reset()
    let patch = knownModalPatch(modes: 4)
    let recovered = ModalDrumParameters(patch: patch, trainable: true).naturalPatch()
    for (actual, expected) in zip(recovered.gains, patch.gains) {
      XCTAssertEqual(actual, expected, accuracy: 1e-5)
    }
    for (actual, expected) in zip(recovered.decaySeconds, patch.decaySeconds) {
      XCTAssertEqual(actual, expected, accuracy: 1e-5)
      XCTAssertTrue(ModalRanges.modeTau.contains(actual))
    }
    XCTAssertEqual(recovered.noiseDecaySeconds, patch.noiseDecaySeconds, accuracy: 1e-5)
    XCTAssertTrue(ModalRanges.noiseTau.contains(recovered.noiseDecaySeconds))
    XCTAssertEqual(recovered.noiseTailFIRTaps, patch.noiseTailFIRTaps)
    XCTAssertEqual(recovered.noiseTailGain, patch.noiseTailGain, accuracy: 1e-5)
    XCTAssertEqual(
      recovered.noiseTailDecaySeconds, patch.noiseTailDecaySeconds, accuracy: 1e-5)
    XCTAssertTrue(ModalRanges.noiseTailTau.contains(recovered.noiseTailDecaySeconds))
    XCTAssertLessThan(ModalRanges.noiseTau.upperBound, ModalRanges.noiseTailTau.upperBound)
  }

  func testRealSnareCalibrationDerivesConcreteGateFromNegativeControls() throws {
    var config = ModalDrumConfig()
    config.spectralWindows = [64, 128]
    let count = 512
    let target = (0..<count).map { i in
      Float(sin(2 * Double.pi * 440 * Double(i) / Double(config.sampleRate)))
    }
    let wrong = (0..<count).map { i in
      Float(sin(2 * Double.pi * 880 * Double(i) / Double(config.sampleRate)))
    }
    let calibration = try RealSnareFitter.calibrate(
      target: target, wrongSnare: wrong, config: config)
    XCTAssertEqual(calibration.selfScore, 0, accuracy: 1e-6)
    XCTAssertGreaterThan(calibration.whiteNoiseScore, calibration.selfScore)
    XCTAssertGreaterThan(calibration.wrongSnareScore, calibration.selfScore)
    XCTAssertGreaterThan(calibration.numericGate, calibration.selfScore)
    XCTAssertLessThan(
      calibration.numericGate, min(calibration.whiteNoiseScore, calibration.wrongSnareScore))
  }

  func testSpectralEnvelopeWarmStartFindsDominantGridFrequency() {
    let sampleRate: Float = 44_100
    let frequencies: [Float] = [220, 440, 880]
    let samples = (0..<8_192).map { i in
      Float(sin(2 * Double.pi * 440 * Double(i) / Double(sampleRate)))
    }
    let gains = RealSnareFitter.spectralEnvelopeWarmStart(
      samples: samples, frequencies: frequencies, sampleRate: sampleRate)
    XCTAssertEqual(gains.count, frequencies.count)
    XCTAssertEqual(gains.max(), gains[1])
    XCTAssertGreaterThan(gains[1], gains[0] * 10)
    XCTAssertGreaterThan(gains[1], gains[2] * 10)
  }

  /// The envelope only sets the shape; the bank must also start at the
  /// target's level, and must do so independently of K or the sweep is
  /// comparing initialization loudness instead of model capacity. Checked by
  /// rendering the actual synth, not by replaying the closed form the warm
  /// start uses.
  func testWarmStartLevelMatchesRenderedTargetIndependentlyOfModeCount() throws {
    var config = ModalDrumConfig()
    let duration: Float = 0.25
    let tau: Float = 0.15
    config.frames = Int(duration * config.sampleRate)
    let frames = config.frames
    // Noise-ish broadband target so every grid frequency gets some envelope.
    var state: UInt64 = 12_345
    let samples = (0..<frames).map { _ -> Float in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return (Float((state >> 40) & 0xFFFFFF) / Float(0xFFFFFF)) * 0.6 - 0.3
    }
    func rms(_ x: [Float]) -> Float { sqrt(x.reduce(0) { $0 + $1 * $1 } / Float(max(1, x.count))) }
    let targetRMS = rms(samples)

    for modes in [32, 128] {
      let frequencies = modalFrequencyGrid(count: modes)
      let gains = RealSnareFitter.spectralEnvelopeWarmStart(
        samples: samples, frequencies: frequencies, sampleRate: config.sampleRate,
        decaySeconds: tau, durationSeconds: duration)
      XCTAssertLessThanOrEqual(gains.max() ?? 0, 0.9)

      var patch = flatInitialPatch(modes: modes)
      patch.gains = gains
      patch.decaySeconds = [Float](repeating: tau, count: modes)
      config.applyRuntime()
      LazyGraphContext.reset()
      let params = ModalDrumParameters(patch: patch, trainable: false)
      let rendered = try ModalDrumSynth.render(
        params: params, frequencies: Tensor(frequencies), includeModal: true, includeNoise: false
      ).realize(frames: frames)

      XCTAssertEqual(
        rms(rendered), targetRMS, accuracy: targetRMS * 0.3,
        "warm-start level drifts from the target at K=\(modes)")
    }
  }

  func testSelfRenderIsDeterministic() throws {
    var config = ModalDrumConfig()
    config.frames = 256
    config.modes = 4
    let patch = knownModalPatch(modes: config.modes)
    let first = try ModalDrumTrainer.render(patch: patch, config: config)
    let second = try ModalDrumTrainer.render(patch: patch, config: config)
    XCTAssertEqual(first, second)
    XCTAssertTrue(first.allSatisfy(\.isFinite))
    XCTAssertGreaterThan(first.map(abs).max() ?? 0, 0.001)
  }

  func testModalOnlyClosedFormRendersWithoutNoiseState() throws {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = 128
    LazyGraphContext.reset()
    let patch = knownModalPatch(modes: 4)
    let frequencies = Tensor(modalFrequencyGrid(count: 4))
    let params = ModalDrumParameters(patch: patch, trainable: false)
    let samples = try ModalDrumSynth.render(
      params: params, frequencies: frequencies, includeModal: true, includeNoise: false
    ).realize(frames: 128)
    XCTAssertEqual(samples.count, 128)
    XCTAssertEqual(samples[0], 0, accuracy: 1e-6)
    XCTAssertTrue(samples.allSatisfy(\.isFinite))
  }

  /// The envelope time base must span the whole render without wrapping; a 1 Hz
  /// phasor silently re-attacked the drum once per second.
  func testEnvelopeDoesNotRestartOnLongRenders() throws {
    var config = ModalDrumConfig()
    config.frames = 88_200  // 2 seconds at 44.1 kHz
    config.modes = 8
    let samples = try ModalDrumTrainer.render(
      patch: knownModalPatch(modes: config.modes), config: config)
    XCTAssertEqual(samples.count, config.frames)
    func rms(_ start: Int) -> Float {
      let slice = samples[start..<Swift.min(start + 500, samples.count)]
      return (slice.reduce(0) { $0 + $1 * $1 } / Float(slice.count)).squareRoot()
    }
    let before = rms(43_000)
    for start in stride(from: 43_500, to: 88_000, by: 500) {
      XCTAssertLessThanOrEqual(rms(start), before * 1.05, "re-attack near frame \(start)")
    }
  }
}

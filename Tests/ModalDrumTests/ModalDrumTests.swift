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

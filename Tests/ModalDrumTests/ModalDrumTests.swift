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
}

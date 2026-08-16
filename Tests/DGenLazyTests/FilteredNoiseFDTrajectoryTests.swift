import XCTest

@testable import DDSPE2E
@testable import DGenLazy

/// R2 piece 1 (docs/DDSP_REVIVAL_SPEC.md): the time-varying frequency-sampled
/// noise branch. Covers both halves of the claim — that the operator filters
/// as advertised, and that audio error can train its per-frame magnitudes.
final class FilteredNoiseFDTrajectoryTests: XCTestCase {
  private let fftSize = 64
  private let hop = 16
  private let frameCount = 1024
  private var nBins: Int { fftSize / 2 + 1 }

  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = frameCount
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
    super.tearDown()
  }

  private func makeNoise(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(state >> 40) / Float(1 << 24) * 2.0 - 1.0
    }
  }

  private func playhead(featureFrames: Int) -> Signal {
    let maxIndex = Float(max(0, featureFrames - 1))
    let step = maxIndex / Float(max(1, frameCount - 1))
    let raw = Signal.accum(
      Signal.constant(step), reset: 0.0, min: 0.0, max: maxIndex)
    return raw.clip(0.0, Double(max(0.0, maxIndex - 1e-4)))
  }

  /// Ratio of first-difference energy to signal energy: a dependency-free proxy
  /// for how much high-frequency content survived the filter.
  private func highFrequencyRatio(_ x: [Float]) -> Float {
    let stable = Array(x.dropFirst(fftSize * 2))
    guard stable.count > 1 else { return 0 }
    var diffSum: Float = 0
    var magSum: Float = 0
    for i in 1..<stable.count {
      diffSum += Swift.abs(stable[i] - stable[i - 1])
      magSum += Swift.abs(stable[i])
    }
    return magSum > 0 ? diffSum / magSum : 0
  }

  /// Per-frame magnitudes that are flat below `cutoffBin` and zero above.
  private func brickwall(cutoffBins: [Int]) -> Tensor {
    let rows = cutoffBins.map { cutoff in
      (0..<nBins).map { $0 <= cutoff ? Float(1) : Float(0) }
    }
    return Tensor(rows)
  }

  func testLowpassPassesLessHighFrequencyThanHighpass() throws {
    let noiseData = makeNoise(count: frameCount, seed: 12345)

    func render(_ magnitudes: [[Float]]) throws -> [Float] {
      LazyGraphContext.reset()
      let noise = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let out = FilteredNoiseFD.render(
        magnitudes: Tensor(magnitudes),
        noise: noise,
        framePosition: playhead(featureFrames: magnitudes.count),
        fftSize: fftSize,
        hop: hop,
        irLength: fftSize / 2
      )
      return try out.realize(frames: frameCount)
    }

    let lowBins = (0..<nBins).map { $0 <= nBins / 4 ? Float(1) : Float(0) }
    let highBins = (0..<nBins).map { $0 >= nBins * 3 / 4 ? Float(1) : Float(0) }

    let lowpass = try render([lowBins, lowBins])
    let highpass = try render([highBins, highBins])

    let lowRatio = highFrequencyRatio(lowpass)
    let highRatio = highFrequencyRatio(highpass)

    XCTAssertGreaterThan(
      highRatio, lowRatio * 2.0,
      "highpass output should carry far more first-difference energy "
        + "(low=\(lowRatio), high=\(highRatio))")
    XCTAssertTrue(lowpass.allSatisfy { $0.isFinite })
    XCTAssertGreaterThan(lowpass.map { Swift.abs($0) }.max() ?? 0, 1e-4, "lowpass output is silent")
  }

  /// Train per-frame magnitudes to match audio rendered from a known
  /// trajectory. The gate is audio-level: IR windowing smooths the requested
  /// response, so distinct magnitude vectors can produce identical output and
  /// per-bin recovery is not guaranteed (nor required by the model).
  /// Smooth per-frame rolloff: representable by a windowed IR, unlike a brickwall.
  private func smoothLowpass(cutoffFractions: [Float]) -> Tensor {
    let rows = cutoffFractions.map { frac -> [Float] in
      (0..<nBins).map { k in
        let x = Float(k) / Float(nBins - 1) / Swift.max(1e-3, frac)
        return 1.0 / (1.0 + 4.0 * x * x)
      }
    }
    return Tensor(rows)
  }

  func testLearnsMagnitudeTrajectoryFromAudio() throws {
    let featureFrames = 4
    let noiseData = makeNoise(count: frameCount, seed: 6789)

    // Target sweeps from mostly-open to mostly-closed across the clip.
    let targetMagnitudes = smoothLowpass(cutoffFractions: [0.9, 0.6, 0.35, 0.15])
    let learned = Tensor.param(
      [featureFrames, nBins],
      data: [Float](repeating: 0.5, count: featureFrames * nBins))

    func buildLoss() -> Signal {
      let noise = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let position = playhead(featureFrames: featureFrames)
      func branch(_ magnitudes: Tensor) -> Signal {
        FilteredNoiseFD.render(
          magnitudes: magnitudes,
          noise: noise,
          framePosition: position,
          fftSize: fftSize,
          hop: hop,
          irLength: fftSize / 2
        )
      }
      return mse(branch(learned), branch(targetMagnitudes))
    }

    let optimizer = Adam(params: [learned], lr: 0.005)
    let initialLosses = try buildLoss().backward(frames: frameCount)
    let initialLoss = initialLosses.reduce(0, +) / Float(initialLosses.count)
    optimizer.zeroGrad()

    var finalLoss = initialLoss
    // Decay the step size: the objective is well-conditioned early but a fixed
    // step overshoots near the optimum and the loss drifts back up.
    for epoch in 0..<150 {
      let values = try buildLoss().backward(frames: frameCount)
      finalLoss = values.reduce(0, +) / Float(values.count)
      if epoch % 25 == 0 { print("epoch \(epoch) loss=\(finalLoss)") }
      optimizer.step()
      optimizer.zeroGrad()
      optimizer.lr *= 0.995
    }

    XCTAssertTrue(finalLoss.isFinite, "loss went non-finite")
    XCTAssertLessThan(
      finalLoss, initialLoss * 0.3,
      "audio error should fall substantially; got \(initialLoss) -> \(finalLoss)")
  }
}

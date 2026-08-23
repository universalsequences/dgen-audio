import XCTest

@testable import DGenLazy

/// R2 gate (docs/DDSP_REVIVAL_SPEC.md): the DDSP paper's noise branch predicts
/// per-frame filter magnitudes and applies them by frequency-domain
/// multiplication, rather than the time-domain FIR shortcut DDSPE2E shipped.
///
/// This pins the prerequisite: gradients must flow from audio error back to
/// per-bin filter magnitudes through tensorFFT -> per-bin multiply -> tensorIFFT.
/// Recovery is exact-in-principle here (Y[k] = H[k]X[k] with a fixed noise
/// realization determines H), so a failure is a gradient bug, not a hard
/// optimization problem.
final class FilteredNoiseFDTests: XCTestCase {

  /// Deterministic pseudo-noise so prediction and target see the same excitation.
  private func makeNoise(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let unit = Float(state >> 40) / Float(1 << 24)  // [0,1)
      return unit * 2.0 - 1.0
    }
  }

  func testRecoverFilterMagnitudesThroughFFT() throws {
    let N = 32
    LazyGraphContext.reset()

    let noiseData = makeNoise(count: N, seed: 0xDD_5B_2026)
    let noise = Tensor(noiseData)

    // Real, symmetric (zero-phase) target response: smooth low-pass.
    var targetFilterData = [Float](repeating: 0, count: N)
    for k in 0...(N / 2) {
      let norm = Float(k) / Float(N / 2)
      let mag = 1.0 / (1.0 + 8.0 * norm * norm)
      targetFilterData[k] = mag
      if k > 0 && k < N / 2 { targetFilterData[N - k] = mag }
    }
    let targetFilter = Tensor(targetFilterData)

    // Learnable response starts flat — wrong everywhere.
    let learnedFilter = Tensor.param([N], data: [Float](repeating: 0.5, count: N))

    func buildLoss() -> Tensor {
      let (re, im) = tensorFFT(noise, N: N)
      let predicted = tensorIFFT(re * learnedFilter, im * learnedFilter, N: N)
      let target = tensorIFFT(re * targetFilter, im * targetFilter, N: N)
      let diff = predicted - target
      return (diff * diff).sum()
    }

    let optimizer = Adam(params: [learnedFilter], lr: 0.05)
    let initialLoss = try buildLoss().backward(frameCount: 1)[0]
    optimizer.zeroGrad()

    var finalLoss = initialLoss
    for _ in 0..<300 {
      let loss = buildLoss()
      finalLoss = try loss.backward(frameCount: 1)[0]
      optimizer.step()
      optimizer.zeroGrad()
    }

    let recovered = learnedFilter.getData() ?? []
    XCTAssertEqual(recovered.count, N)
    XCTAssertLessThan(
      finalLoss, initialLoss * 0.01,
      "filtered-noise loss should collapse; got \(initialLoss) -> \(finalLoss)")

    // Only bins the excitation actually reaches are identifiable.
    for k in 0...(N / 2) {
      XCTAssertEqual(
        recovered[k], targetFilterData[k], accuracy: 0.05,
        "bin \(k) magnitude not recovered")
    }
  }
}

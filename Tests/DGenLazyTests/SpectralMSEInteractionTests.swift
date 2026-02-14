import DGen
import XCTest

@testable import DGenLazy

final class SpectralMSEInteractionTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 16_000
    DGenConfig.maxFrameCount = 16_384
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  /// Mixed objective probe:
  /// total = manualMSE + 0.1 * spectralLossFFT(hop=8)
  ///
  /// This should produce finite, non-zero average loss on every step.
  func testMixedManualMSEAndHoppedSpectralLossStaysFiniteAndNonZero() throws {
    let frameCount = 2048
    let twoPi = Float.pi * 2.0

    func buildLoss() -> Signal {
      let pred = sin(Signal.phasor(220.0) * twoPi) * 0.45
      let target = sin(Signal.phasor(233.0) * twoPi) * 0.40

      // Manual per-frame MSE term (same signal shape as spectral term output).
      let diff = pred - target
      let mseTerm = diff * diff

      let windows = [64, 128, 256]
      var spec = Signal.constant(0.0)
      for w in windows {
        let hop = max(1, w / 4)  // Matches DDSP hop-divisor behavior.
        spec = spec + spectralLossFFT(pred, target, windowSize: w, hop: hop, normalize: true)
      }
      let specTerm = spec * (0.1 / Float(windows.count))
      return mseTerm + specTerm
    }

    var zeroLikeCount = 0
    var nonFiniteCount = 0
    var minLoss = Float.greatestFiniteMagnitude
    var maxLoss: Float = 0

    for _ in 0..<20 {
      let values = try buildLoss().backward(frames: frameCount)
      let avgLoss = values.reduce(0, +) / Float(max(1, values.count))

      if !avgLoss.isFinite {
        nonFiniteCount += 1
      } else {
        minLoss = min(minLoss, avgLoss)
        maxLoss = max(maxLoss, avgLoss)
        if avgLoss < 1e-8 {
          zeroLikeCount += 1
        }
      }
    }

    XCTAssertEqual(nonFiniteCount, 0, "Encountered non-finite loss values")
    XCTAssertEqual(
      zeroLikeCount, 0,
      "Observed near-zero average loss on \(zeroLikeCount) steps (min=\(minLoss), max=\(maxLoss))")
  }

  /// Closer to DDSPE2E: target arrives as a tensor-backed signal.
  func testMixedLossWithTensorBackedTargetStaysFiniteAndNonZero() throws {
    let frameCount = 16_384
    let twoPi = Float.pi * 2.0

    var targetFrames = [Float]()
    targetFrames.reserveCapacity(frameCount)
    let targetFreq: Float = 233.0
    let sampleRate: Float = 16_000
    for n in 0..<frameCount {
      let t = Float(n) / sampleRate
      targetFrames.append(sinf(twoPi * targetFreq * t) * 0.40)
    }

    func buildLoss() -> Signal {
      let pred = sin(Signal.phasor(220.0) * twoPi) * 0.45
      let target = Tensor(targetFrames).toSignal(maxFrames: frameCount)

      let diff = pred - target
      let mseTerm = diff * diff
      let windows = [64, 128, 256]
      var spec = Signal.constant(0.0)
      for w in windows {
        let hop = max(1, w / 4)  // Matches DDSP hop-divisor behavior.
        spec = spec + spectralLossFFT(pred, target, windowSize: w, hop: hop, normalize: true)
      }
      let specTerm = spec * (0.1 / Float(windows.count))
      return mseTerm + specTerm
    }

    var zeroLikeCount = 0
    var nonFiniteCount = 0
    var minLoss = Float.greatestFiniteMagnitude
    var maxLoss: Float = 0

    for _ in 0..<12 {
      let values = try buildLoss().backward(frames: frameCount)
      let avgLoss = values.reduce(0, +) / Float(max(1, values.count))

      if !avgLoss.isFinite {
        nonFiniteCount += 1
      } else {
        minLoss = min(minLoss, avgLoss)
        maxLoss = max(maxLoss, avgLoss)
        if avgLoss < 1e-8 {
          zeroLikeCount += 1
        }
      }
    }

    XCTAssertEqual(nonFiniteCount, 0, "Encountered non-finite loss values")
    XCTAssertEqual(
      zeroLikeCount, 0,
      "Observed near-zero average loss on \(zeroLikeCount) steps (min=\(minLoss), max=\(maxLoss))")
  }
}

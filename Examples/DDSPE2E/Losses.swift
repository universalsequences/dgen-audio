import DGenLazy
import Foundation

enum DDSPTrainingLosses {
  static func fullLoss(
    prediction: Signal,
    target: Signal,
    spectralWindowSizes: [Int],
    frameCount: Int,
    mseWeight: Float,
    spectralWeight: Float
  ) -> Signal {
    let usableWindows = spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    var total = Signal.constant(0.0)

    if mseWeight > 0 {
      total = total + mse(prediction, target) * mseWeight
    }

    if spectralWeight > 0, !usableWindows.isEmpty {
      var spec = Signal.constant(0.0)
      for w in usableWindows {
        spec = spec + spectralLossFFT(prediction, target, windowSize: w, normalize: true)
      }
      spec = spec * (1.0 / Float(usableWindows.count))
      total = total + spec * spectralWeight
    }

    // Avoid accidental zero-loss graph if both weights are zero.
    return total + mse(prediction, target) * 0.0
  }
}

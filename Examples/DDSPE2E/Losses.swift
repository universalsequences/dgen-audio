import DGenLazy
import Foundation

enum DDSPTrainingLosses {
  static func fullLoss(
    prediction: Signal,
    target: Signal,
    spectralWindowSizes: [Int],
    spectralHopDivisor: Int,
    frameCount: Int,
    mseWeight: Float,
    spectralWeight: Float
  ) -> Signal {
    let usableWindows = spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    var total = Signal.constant(0.0)
    var hasTerm = false

    if mseWeight > 0 {
      total = total + mse(prediction, target) * mseWeight
      hasTerm = true
    }

    if spectralWeight > 0, !usableWindows.isEmpty {
      var spec = Signal.constant(0.0)
      for w in usableWindows {
        let hop = max(1, w / max(1, spectralHopDivisor))
        spec = spec + spectralLossFFT(prediction, target, windowSize: w, hop: hop, normalize: true)
      }
      spec = spec * (1.0 / Float(usableWindows.count))
      total = total + spec * spectralWeight
      hasTerm = true
    }

    // Preserve a valid scalar loss signal without forcing extra loss terms into the graph.
    return hasTerm ? total : Signal.constant(0.0)
  }

  /// Batched loss for [B]-shaped SignalTensor prediction and target.
  static func fullBatchedLoss(
    prediction: SignalTensor,
    target: SignalTensor,
    batchSize: Int,
    spectralWindowSizes: [Int],
    spectralHopDivisor: Int,
    frameCount: Int,
    mseWeight: Float,
    spectralWeight: Float
  ) -> Signal {
    let usableWindows = spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    var total = Signal.constant(0.0)
    var hasTerm = false

    if mseWeight > 0 {
      let diff = prediction - target
      let batchMSE = (diff * diff).sum() * (1.0 / Float(batchSize))
      total = total + batchMSE * mseWeight
      hasTerm = true
    }

    if spectralWeight > 0, !usableWindows.isEmpty {
      var spec = Signal.constant(0.0)
      for w in usableWindows {
        let hop = max(1, w / max(1, spectralHopDivisor))
        spec = spec + spectralLossFFT(prediction, target, windowSize: w, hop: hop, normalize: true)
      }
      spec = spec * (1.0 / Float(usableWindows.count))
      total = total + spec * spectralWeight
      hasTerm = true
    }

    return hasTerm ? total : Signal.constant(0.0)
  }
}

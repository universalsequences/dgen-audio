import DGen
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
    spectralWeight: Float,
    spectralLogmagWeight: Float,
    spectralLossMode: SpectralLossModeOption,
    loudnessWeight: Float = 0.0,
    loudnessLossMode: LoudnessLossModeOption = .linearL2,
    noiseDominanceWeight: Float = 0.0,
    noiseDominanceTargetRatio: Float = 1.0,
    harmonicGain: DGenLazy.Tensor? = nil,
    noiseGain: DGenLazy.Tensor? = nil,
    targetLoudnessNorm: DGenLazy.Tensor? = nil,
    uvMask: DGenLazy.Tensor? = nil
  ) -> Signal {
    let usableWindows = spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    let lossMode: SpectralLossMode = spectralLossMode == .l1 ? .l1 : .l2
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
        spec =
          spec
          + spectralLossFFT(
            prediction, target, windowSize: w, lossMode: lossMode, hop: hop, normalize: true)
      }
      spec = spec * (1.0 / Float(usableWindows.count))
      total = total + spec * spectralWeight
      hasTerm = true
    }

    if spectralLogmagWeight > 0, !usableWindows.isEmpty {
      var specLog = Signal.constant(0.0)
      for w in usableWindows {
        let hop = max(1, w / max(1, spectralHopDivisor))
        specLog =
          specLog
          + spectralLossFFT(
            prediction, target, windowSize: w, useLogMagnitude: true, lossMode: lossMode,
            hop: hop, normalize: true)
      }
      specLog = specLog * (1.0 / Float(usableWindows.count))
      total = total + specLog * spectralLogmagWeight
      hasTerm = true
    }

    if loudnessWeight > 0,
      let harmonicGain,
      let targetLoudnessNorm
    {
      let target = targetLoudnessNorm
      let predGain: DGenLazy.Tensor
      if let noiseGain, let uvMask {
        let voicedMask = uvMask
        let unvoicedMask = 1.0 - uvMask
        predGain = harmonicGain * voicedMask + noiseGain * unvoicedMask
      } else {
        predGain = harmonicGain
      }

      let envLossTensor: DGenLazy.Tensor
      switch loudnessLossMode {
      case .linearL2:
        let err = predGain - target
        envLossTensor = (err * err).mean()
      case .dbL1:
        // Compare loudness envelopes in normalized dB space for more robust scaling.
        let eps: Float = 1e-4
        let dbScale: Float = 20.0 / Float(Foundation.log(10.0))
        let predDbNorm = (((predGain + eps).log() * dbScale) + 80.0) * (1.0 / 80.0)
        let predNorm = predDbNorm.clip(0.0, 1.0)
        let targetNorm = target.clip(0.0, 1.0)
        envLossTensor = abs(predNorm - targetNorm).mean()
      }
      // Ensure a rank-1 tensor before peek; mean() can become a scalar lazy node.
      let envLoss = (DGenLazy.Tensor([0.0]) + envLossTensor).peek(Signal.constant(0.0))
      total = total + envLoss * loudnessWeight
      hasTerm = true
    }

    if noiseDominanceWeight > 0,
      let harmonicGain,
      let noiseGain
    {
      // Penalize excess noise gain beyond a harmonic-scaled target:
      // excess = max(noiseGain - targetRatio * harmonicGain, 0)
      // penalty = mean(excess^2)
      let targetRatio = max(0.0, noiseDominanceTargetRatio)
      let excess = max(noiseGain - harmonicGain * targetRatio, 0.0)
      let penaltyTensor = (excess * excess).mean()
      let penalty = (DGenLazy.Tensor([0.0]) + penaltyTensor).peek(Signal.constant(0.0))
      total = total + penalty * noiseDominanceWeight
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
    spectralWeight: Float,
    spectralLogmagWeight: Float,
    spectralLossMode: SpectralLossModeOption,
    loudnessWeight: Float = 0.0,
    loudnessLossMode: LoudnessLossModeOption = .linearL2,
    noiseDominanceWeight: Float = 0.0,
    noiseDominanceTargetRatio: Float = 1.0,
    harmonicGain: DGenLazy.Tensor? = nil,
    noiseGain: DGenLazy.Tensor? = nil,
    targetLoudnessNorm: DGenLazy.Tensor? = nil,
    uvMask: DGenLazy.Tensor? = nil
  ) -> Signal {
    let usableWindows = spectralWindowSizes.filter { $0 > 1 && $0 <= frameCount }
    let lossMode: SpectralLossMode = spectralLossMode == .l1 ? .l1 : .l2
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
        spec =
          spec
          + spectralLossFFT(
            prediction, target, windowSize: w, lossMode: lossMode, hop: hop, normalize: true)
      }
      spec = spec * (1.0 / Float(usableWindows.count))
      total = total + spec * spectralWeight
      hasTerm = true
    }

    if spectralLogmagWeight > 0, !usableWindows.isEmpty {
      var specLog = Signal.constant(0.0)
      for w in usableWindows {
        let hop = max(1, w / max(1, spectralHopDivisor))
        specLog =
          specLog
          + spectralLossFFT(
            prediction, target, windowSize: w, useLogMagnitude: true, lossMode: lossMode,
            hop: hop, normalize: true)
      }
      specLog = specLog * (1.0 / Float(usableWindows.count))
      total = total + specLog * spectralLogmagWeight
      hasTerm = true
    }

    if loudnessWeight > 0,
      let harmonicGain,
      let targetLoudnessNorm
    {
      let target = targetLoudnessNorm
      let predGain: DGenLazy.Tensor
      if let noiseGain, let uvMask {
        let voicedMask = uvMask
        let unvoicedMask = 1.0 - uvMask
        predGain = harmonicGain * voicedMask + noiseGain * unvoicedMask
      } else {
        predGain = harmonicGain
      }

      let envLossTensor: DGenLazy.Tensor
      switch loudnessLossMode {
      case .linearL2:
        let err = predGain - target
        envLossTensor = (err * err).mean()
      case .dbL1:
        // Compare loudness envelopes in normalized dB space for more robust scaling.
        let eps: Float = 1e-4
        let dbScale: Float = 20.0 / Float(Foundation.log(10.0))
        let predDbNorm = (((predGain + eps).log() * dbScale) + 80.0) * (1.0 / 80.0)
        let predNorm = predDbNorm.clip(0.0, 1.0)
        let targetNorm = target.clip(0.0, 1.0)
        envLossTensor = abs(predNorm - targetNorm).mean()
      }
      // Ensure a rank-1 tensor before peek; mean() can become a scalar lazy node.
      let envLoss = (DGenLazy.Tensor([0.0]) + envLossTensor).peek(Signal.constant(0.0))
      total = total + envLoss * loudnessWeight
      hasTerm = true
    }

    if noiseDominanceWeight > 0,
      let harmonicGain,
      let noiseGain
    {
      let targetRatio = max(0.0, noiseDominanceTargetRatio)
      let excess = max(noiseGain - harmonicGain * targetRatio, 0.0)
      let penaltyTensor = (excess * excess).mean()
      let penalty = (DGenLazy.Tensor([0.0]) + penaltyTensor).peek(Signal.constant(0.0))
      total = total + penalty * noiseDominanceWeight
      hasTerm = true
    }

    return hasTerm ? total : Signal.constant(0.0)
  }
}

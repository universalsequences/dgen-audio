import DGen
import DGenLazy

enum ModalDrumLosses {
  private static func weight(_ value: Float) -> Signal {
    Tensor([value]).peek(Signal.constant(0))
  }

  static func trainingLoss(
    prediction: Signal,
    target: Signal,
    config: ModalDrumConfig
  ) -> Signal {
    let windows = config.spectralWindows.filter { $0 > 1 && $0 <= config.frames }
    var total = Signal.constant(0)
    if !windows.isEmpty {
      var linear = Signal.constant(0)
      var logarithmic = Signal.constant(0)
      for window in windows {
        let hop = max(1, window / max(1, config.spectralHopDivisor))
        linear =
          linear
          + spectralLossFFT(
            prediction, target, windowSize: window, lossMode: .l2, hop: hop, normalize: true)
        logarithmic =
          logarithmic
          + spectralLossFFT(
            prediction, target, windowSize: window, useLogMagnitude: true,
            lossMode: .l2, hop: hop, normalize: true)
      }
      let scale = 1 / Float(windows.count)
      total = total + linear * weight(config.linearSpectralWeight * scale)
      total = total + logarithmic * weight(config.logSpectralWeight * scale)
    }

    if config.loudnessWeight > 0 {
      // Overlapping 256-sample frame-RMS envelope, compared with L1.
      let window = min(256, max(2, config.frames))
      let predBuffer = prediction.buffer(size: window).reshape([window])
      let targetBuffer = target.buffer(size: window).reshape([window])
      let predRMS = sqrt((predBuffer * predBuffer).sum() * (1 / Float(window)) + 1e-8)
      let targetRMS = sqrt((targetBuffer * targetBuffer).sum() * (1 / Float(window)) + 1e-8)
      total = total + abs(predRMS - targetRMS) * weight(config.loudnessWeight)
    }
    return total
  }

  /// Smooth and cheap objective used only by finite-difference validation.
  static func fdLoss(prediction: Signal, target: Signal) -> Signal {
    mse(prediction, target)
  }
}

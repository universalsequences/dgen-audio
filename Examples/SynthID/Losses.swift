import DGen
import DGenLazy
import Foundation

enum SynthIDLosses {
  static func multiResolutionSpectralLoss(
    synth: Signal,
    target: Signal,
    config: SynthIDConfig
  ) -> Signal {
    var total = Signal.constant(0.0)
    for (index, window) in config.spectralWindows.enumerated() {
      let weight =
        index < config.windowWeights.count
        ? config.windowWeights[index]
        : 1.0
      let hop = max(1, window / 4)
      let logMag = spectralLossFFT(
        synth,
        target,
        windowSize: window,
        useHannWindow: config.useHannWindow,
        useLogMagnitude: true,
        lossMode: .l1,
        hop: hop,
        normalize: true)
      total = total + logMag * weight

      if config.includeLinearMagnitude {
        let linear = spectralLossFFT(
          synth,
          target,
          windowSize: window,
          useHannWindow: config.useHannWindow,
          useLogMagnitude: false,
          lossMode: .l1,
          hop: hop,
          normalize: true)
        total = total + linear * (weight * config.linearMagnitudeWeight)
      }
    }
    return total
  }
}

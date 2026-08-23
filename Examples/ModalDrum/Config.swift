import DGen
import DGenLazy
import Foundation

struct ModalDrumConfig: Codable {
  var sampleRate: Float = 44_100
  var frames: Int = 8_192
  var modes: Int = 32
  var steps: Int = 300
  var learningRate: Float = 0.1
  var seed: UInt64 = 70
  var spectralWindows = [64, 128, 256, 512, 1_024, 2_048]
  var spectralHopDivisor = 4
  var linearSpectralWeight: Float = 1
  var logSpectralWeight: Float = 1
  var loudnessWeight: Float = 0.01
  var highModeL1Weight: Float = 1e-3
  var logMagnitudeEpsilon: Float = 1e-3
  var renderEvery = 50
  var checkpointEvery = 50

  func applyRuntime(kernelOutputPath: String? = nil) {
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = max(1, frames)
    DGenConfig.kernelOutputPath = kernelOutputPath
    DGenConfig.debug = false
    DGenSpectralConfig.logMagnitudeEpsilon = logMagnitudeEpsilon
  }
}

/// Runtime defaults for `ModalDrumSynth.render`, read from the active DGen
/// configuration so every render derives its non-wrapping envelope time base
/// from the frame count it is actually compiled for.
enum ModalRuntime {
  static var frames: Int { DGenConfig.maxFrameCount }
  static var sampleRate: Float { DGenConfig.sampleRate }
}

struct ModalPatch: Codable, Equatable {
  var gains: [Float]
  var decaySeconds: [Float]
  var firTaps: [Float]
  var noiseGain: Float
  var noiseDecaySeconds: Float
  var noiseTailFIRTaps: [Float]
  var noiseTailGain: Float
  var noiseTailDecaySeconds: Float
}

enum ModalRanges {
  static let modeTau: ClosedRange<Float> = 0.005...2.0
  // Keep the two wire layers ordered by construction: the first captures the
  // crack while the second is free to retain a spectrally different tail.
  static let noiseTau: ClosedRange<Float> = 0.005...0.050
  static let noiseTailTau: ClosedRange<Float> = 0.020...0.250

  static func raw(for value: Float, in range: ClosedRange<Float>) -> Float {
    let lo = Foundation.log(range.lowerBound)
    let hi = Foundation.log(range.upperBound)
    let p = min(max((Foundation.log(value) - lo) / (hi - lo), 1e-5), 1 - 1e-5)
    return Foundation.log(p / (1 - p))
  }

  static func logMapped(_ raw: DGenLazy.Tensor, range: ClosedRange<Float>) -> DGenLazy.Tensor {
    let lo = Foundation.log(range.lowerBound)
    let span = Foundation.log(range.upperBound) - lo
    return (raw.sigmoid() * span + lo).exp()
  }
}

func modalFrequencyGrid(count: Int) -> [Float] {
  guard count > 1 else { return [120] }
  let lo = Foundation.log(Float(120))
  let hi = Foundation.log(Float(14_000))
  return (0..<count).map { i in
    Foundation.exp(lo + (hi - lo) * Float(i) / Float(count - 1))
  }
}

func knownModalPatch(modes: Int) -> ModalPatch {
  let frequencies = modalFrequencyGrid(count: modes)
  let membraneRatios: [Float] = [1, 1.59, 2.14, 2.30, 2.65, 2.92, 3.16, 3.50]
  let anchors = membraneRatios.map { $0 * 180 }
  let gains = frequencies.enumerated().map { i, f -> Float in
    let proximity = anchors.map { abs(Foundation.log(f / $0)) }.min() ?? 1
    let shell = 0.002 + 0.82 * Foundation.exp(-proximity * proximity / 0.0075)
    return min(0.92, shell * Foundation.exp(-Float(i) / Float(max(1, modes)) * 0.8))
  }
  let taus = frequencies.enumerated().map { i, f in
    min(0.45, max(0.025, 0.28 * Foundation.pow(120 / f, 0.34) * (1 + 0.08 * Float(i % 3))))
  }
  return ModalPatch(
    gains: gains,
    decaySeconds: taus,
    firTaps: [
      0.02, 0.04, 0.07, 0.11, 0.16, 0.20, 0.16, 0.11, 0.07, 0.04, 0.02, -0.01, -0.02, -0.01, 0,
    ],
    noiseGain: 0.18,
    noiseDecaySeconds: 0.018,
    noiseTailFIRTaps: [
      0, -0.01, -0.02, -0.01, 0.02, 0.08, 0.14, 0.20, 0.14, 0.08, 0.02, -0.01, -0.02,
      -0.01, 0,
    ],
    noiseTailGain: 0.10,
    noiseTailDecaySeconds: 0.090)
}

func flatInitialPatch(modes: Int) -> ModalPatch {
  ModalPatch(
    gains: [Float](repeating: 0.20, count: modes),
    decaySeconds: [Float](repeating: 0.15, count: modes),
    firTaps: [0, 0, 0.02, 0.05, 0.10, 0.16, 0.22, 0.16, 0.10, 0.05, 0.02, 0, 0, 0, 0],
    noiseGain: 0.08,
    noiseDecaySeconds: 0.018,
    noiseTailFIRTaps: [Float](repeating: 0, count: ModalDrumSynth.firSize),
    noiseTailGain: 0.10,
    noiseTailDecaySeconds: 0.080)
}

import DGenLazy
import Foundation

final class ModalDrumParameters {
  let rawGains: Tensor
  let rawModeTaus: Tensor
  let firTaps: Tensor
  let rawNoiseGain: Tensor
  let rawNoiseTau: Tensor

  var all: [any LazyValue] { [rawGains, rawModeTaus, firTaps, rawNoiseGain, rawNoiseTau] }

  init(patch: ModalPatch, trainable: Bool) {
    precondition(patch.gains.count == patch.decaySeconds.count)
    func logit(_ x: Float) -> Float {
      let p = min(max(x, 1e-5), 1 - 1e-5)
      return Foundation.log(p / (1 - p))
    }
    let gains = patch.gains.map(logit)
    let taus = patch.decaySeconds.map { ModalRanges.raw(for: $0, in: ModalRanges.modeTau) }
    let ng = [logit(patch.noiseGain)]
    let nt = [ModalRanges.raw(for: patch.noiseDecaySeconds, in: ModalRanges.noiseTau)]
    if trainable {
      rawGains = Tensor.param([gains.count], data: gains)
      rawModeTaus = Tensor.param([taus.count], data: taus)
      firTaps = Tensor.param([patch.firTaps.count], data: patch.firTaps)
      rawNoiseGain = Tensor.param([1], data: ng)
      rawNoiseTau = Tensor.param([1], data: nt)
    } else {
      rawGains = Tensor(gains)
      rawModeTaus = Tensor(taus)
      firTaps = Tensor(patch.firTaps)
      rawNoiseGain = Tensor(ng)
      rawNoiseTau = Tensor(nt)
    }
  }

  var gains: Tensor { rawGains.sigmoid() }
  var modeTaus: Tensor { ModalRanges.logMapped(rawModeTaus, range: ModalRanges.modeTau) }
  var noiseGain: Tensor { rawNoiseGain.sigmoid() }
  var noiseTau: Tensor { ModalRanges.logMapped(rawNoiseTau, range: ModalRanges.noiseTau) }

  func naturalPatch() -> ModalPatch {
    func sigmoid(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }
    func mapped(_ x: Float, _ range: ClosedRange<Float>) -> Float {
      let p = sigmoid(x)
      return Foundation.exp(
        Foundation.log(range.lowerBound)
          + p * (Foundation.log(range.upperBound) - Foundation.log(range.lowerBound)))
    }
    return ModalPatch(
      gains: (rawGains.getData() ?? []).map(sigmoid),
      decaySeconds: (rawModeTaus.getData() ?? []).map { mapped($0, ModalRanges.modeTau) },
      firTaps: firTaps.getData() ?? [],
      noiseGain: sigmoid((rawNoiseGain.getData() ?? [0])[0]),
      noiseDecaySeconds: mapped((rawNoiseTau.getData() ?? [0])[0], ModalRanges.noiseTau))
  }

  func rawGroups() -> [(name: String, tensor: Tensor)] {
    [
      ("g", rawGains), ("log_tau", rawModeTaus), ("fir", firTaps),
      ("noise_gain", rawNoiseGain), ("noise_log_tau", rawNoiseTau),
    ]
  }
}

enum ModalDrumSynth {
  static let firSize = 15

  /// Closed-form training synth. The modal oscillator uses deterministicPhasor,
  /// whose phase is computed directly from the frame index; consequently the
  /// modal branch has no history cells and compiles frame-parallel.
  ///
  /// The envelope time base is a phasor running at `sampleRate / (2 * frames)`,
  /// so its ramp spans [0, 0.5) over the whole render and never wraps; the
  /// `timeScale` factor converts that ramp back to seconds. Using a 1 Hz phasor
  /// instead would silently re-attack the whole drum once per second.
  static func render(
    params: ModalDrumParameters,
    frequencies: Tensor,
    frames: Int = ModalRuntime.frames,
    sampleRate: Float = ModalRuntime.sampleRate,
    includeModal: Bool = true,
    includeNoise: Bool = true
  ) -> Signal {
    var output = Signal.constant(0)
    let renderFrames = Float(max(1, frames))
    let rampFrequency = sampleRate / (2 * renderFrames)
    let timeScale = 2 * renderFrames / sampleRate

    if includeModal {
      let phase = Signal.phasor(frequencies)
      let modeTime = Signal.phasor(
        Tensor([Float](repeating: rampFrequency, count: frequencies.shape.reduce(1, *))))
      let sines = sin(phase * (2 * Float.pi))
      let envelope = (modeTime * (timeScale / params.modeTaus) * -1.0).exp()
      output = output + (sines * envelope * params.gains).sum()
    }

    if includeNoise {
      precondition(params.firTaps.shape == [firSize])
      let noiseBuffer = Signal.noise().buffer(size: firSize).reshape([firSize])
      let filtered = (noiseBuffer * params.firTaps).sum()
      let noiseTime = Signal.phasor(Tensor([rampFrequency]))
      let noiseEnvelope = (noiseTime * (timeScale / params.noiseTau) * -1.0).exp().sum()
      let gain = params.noiseGain.peek(Signal.constant(0))
      output = output + filtered * noiseEnvelope * gain
    }
    return output
  }
}

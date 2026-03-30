import DGenLazy
import Foundation

struct HarmonicControls {
  var harmonicAmps: Tensor
  var harmonicGain: Tensor
  var noiseGain: Tensor
  var noiseFilter: Tensor
}

struct NamedTensorSnapshot: Codable {
  var name: String
  var shape: [Int]
  var data: [Float]
}

final class HarmonicDecoderModel {
  let inputSize = 3
  let hiddenSize: Int
  let numLayers: Int
  let numHarmonics: Int
  let noiseFilterSize: Int

  let trunkWeights: [Tensor]
  let trunkBiases: [Tensor]
  let W_harm: Tensor
  let b_harm: Tensor
  let W_gain: Tensor
  let b_gain: Tensor
  let W_noise: Tensor
  let b_noise: Tensor
  let W_noiseFilter: Tensor
  let b_noiseFilter: Tensor

  init(config: HarmonicE2EConfig) {
    self.hiddenSize = max(1, config.modelHiddenSize)
    self.numLayers = max(1, config.modelNumLayers)
    self.numHarmonics = max(1, config.numHarmonics)
    self.noiseFilterSize = max(7, config.noiseFilterSize)

    var rng = SeededGenerator(seed: config.seed)
    var weights: [Tensor] = []
    var biases: [Tensor] = []

    for i in 0..<numLayers {
      let fanIn = i == 0 ? inputSize : hiddenSize
      let scale: Float = sqrt(2.0 / Float(fanIn)) * 0.5
      weights.append(
        Tensor.param(
          [fanIn, hiddenSize],
          data: Self.randomArray(count: fanIn * hiddenSize, scale: scale, rng: &rng)
        )
      )
      biases.append(Tensor.param([1, hiddenSize], data: [Float](repeating: 0, count: hiddenSize)))
    }

    self.trunkWeights = weights
    self.trunkBiases = biases
    self.W_harm = Tensor.param(
      [hiddenSize, numHarmonics],
      data: Self.randomArray(count: hiddenSize * numHarmonics, scale: 0.06, rng: &rng)
    )
    self.b_harm = Tensor.param([1, numHarmonics], data: [Float](repeating: 0, count: numHarmonics))
    self.W_gain = Tensor.param(
      [hiddenSize, 1],
      data: Self.randomArray(count: hiddenSize, scale: 0.05, rng: &rng)
    )
    self.b_gain = Tensor.param([1, 1], data: [0.0])
    self.W_noise = Tensor.param(
      [hiddenSize, 1],
      data: Self.randomArray(count: hiddenSize, scale: 0.05, rng: &rng)
    )
    self.b_noise = Tensor.param([1, 1], data: [-3.0])
    self.W_noiseFilter = Tensor.param(
      [hiddenSize, noiseFilterSize],
      data: Self.randomArray(count: hiddenSize * noiseFilterSize, scale: 0.05, rng: &rng)
    )
    self.b_noiseFilter = Tensor.param(
      [1, noiseFilterSize],
      data: [Float](repeating: 0, count: noiseFilterSize)
    )
  }

  var parameters: [any LazyValue] {
    var params: [any LazyValue] = []
    for i in 0..<numLayers {
      params.append(trunkWeights[i])
      params.append(trunkBiases[i])
    }
    params.append(contentsOf: [
      W_harm, b_harm, W_gain, b_gain, W_noise, b_noise, W_noiseFilter, b_noiseFilter,
    ])
    return params
  }

  func forward(features: Tensor) -> HarmonicControls {
    var hidden = features
    for i in 0..<numLayers {
      hidden = tanh(hidden.matmul(trunkWeights[i]) + trunkBiases[i])
    }

    let harmLogits = hidden.matmul(W_harm) + b_harm
    let positive = softplus(harmLogits)
    let rows = hidden.shape[0]
    let denom = positive.sum(axis: 1).reshape([rows, 1]).expand([rows, numHarmonics]) + 1e-6
    let harmonicAmps = positive / denom
    let harmonicGain = softplus(hidden.matmul(W_gain) + b_gain)
    let noiseGain = softplus(hidden.matmul(W_noise) + b_noise)
    let noiseFilterLogits = hidden.matmul(W_noiseFilter) + b_noiseFilter
    let noiseFilterSigned = tanh(noiseFilterLogits)
    let noiseFilterNorm =
      noiseFilterSigned.abs().sum(axis: 1).reshape([rows, 1]).expand([rows, noiseFilterSize]) + 1e-4
    let noiseFilter = noiseFilterSigned / noiseFilterNorm
    return HarmonicControls(
      harmonicAmps: harmonicAmps,
      harmonicGain: harmonicGain,
      noiseGain: noiseGain,
      noiseFilter: noiseFilter
    )
  }

  func snapshots() -> [NamedTensorSnapshot] {
    var snaps: [NamedTensorSnapshot] = []
    for i in 0..<numLayers {
      snaps.append(snapshot("W\(i + 1)", trunkWeights[i]))
      snaps.append(snapshot("b\(i + 1)", trunkBiases[i]))
    }
    snaps.append(snapshot("W_harm", W_harm))
    snaps.append(snapshot("b_harm", b_harm))
    snaps.append(snapshot("W_gain", W_gain))
    snaps.append(snapshot("b_gain", b_gain))
    snaps.append(snapshot("W_noise", W_noise))
    snaps.append(snapshot("b_noise", b_noise))
    snaps.append(snapshot("W_noiseFilter", W_noiseFilter))
    snaps.append(snapshot("b_noiseFilter", b_noiseFilter))
    return snaps
  }

  private func snapshot(_ name: String, _ tensor: Tensor) -> NamedTensorSnapshot {
    NamedTensorSnapshot(name: name, shape: tensor.shape, data: tensor.getData() ?? [])
  }

  private static func randomArray(
    count: Int,
    scale: Float,
    rng: inout SeededGenerator
  ) -> [Float] {
    var data = [Float](repeating: 0, count: count)
    for i in 0..<count {
      let r = Float(rng.next() & 0xFFFF) / Float(UInt16.max)
      data[i] = (r * 2.0 - 1.0) * scale
    }
    return data
  }

  private func softplus(_ x: Tensor) -> Tensor {
    let positive = max(x, 0.0)
    let correction = (1.0 + (-abs(x)).exp()).log()
    return positive + correction
  }
}

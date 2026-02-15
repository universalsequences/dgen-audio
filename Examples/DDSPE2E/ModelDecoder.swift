import DGenLazy
import Foundation

struct DecoderControls {
  var harmonicAmps: Tensor      // [frames, numHarmonics]
  var harmonicGain: Tensor      // [frames, 1]
  var noiseGain: Tensor         // [frames, 1]
}

struct NamedTensorSnapshot: Codable {
  var name: String
  var shape: [Int]
  var data: [Float]
}

final class DDSPDecoderModel {
  let inputSize: Int = 3
  let hiddenSize: Int
  let numLayers: Int
  let numHarmonics: Int

  // Trunk layers: [(W, b), ...]
  let trunkWeights: [Tensor]
  let trunkBiases: [Tensor]

  // Heads
  let W_harm: Tensor
  let b_harm: Tensor

  let W_hgain: Tensor
  let b_hgain: Tensor

  let W_noise: Tensor
  let b_noise: Tensor

  init(config: DDSPE2EConfig) {
    self.hiddenSize = config.modelHiddenSize
    self.numLayers = max(1, config.modelNumLayers)
    self.numHarmonics = config.numHarmonics

    var rng = SeededGenerator(seed: config.seed)

    var weights: [Tensor] = []
    var biases: [Tensor] = []
    for i in 0..<numLayers {
      let fanIn = i == 0 ? inputSize : hiddenSize
      let scale: Float = sqrt(2.0 / Float(fanIn)) * 0.5
      let W = Tensor.param([fanIn, hiddenSize], data: Self.randomArray(
        count: fanIn * hiddenSize, scale: scale, rng: &rng))
      let b = Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize))
      weights.append(W)
      biases.append(b)
    }
    self.trunkWeights = weights
    self.trunkBiases = biases

    self.W_harm = Tensor.param([hiddenSize, numHarmonics], data: Self.randomArray(
      count: hiddenSize * numHarmonics, scale: 0.06, rng: &rng))
    self.b_harm = Tensor.param([1, numHarmonics], data: [Float](repeating: 0.0, count: numHarmonics))

    self.W_hgain = Tensor.param([hiddenSize, 1], data: Self.randomArray(count: hiddenSize, scale: 0.05, rng: &rng))
    self.b_hgain = Tensor.param([1, 1], data: [0.0])

    self.W_noise = Tensor.param([hiddenSize, 1], data: Self.randomArray(count: hiddenSize, scale: 0.05, rng: &rng))
    self.b_noise = Tensor.param([1, 1], data: [0.0])

    // Keep non-negative controls after updates
    self.b_hgain.minBound = -8.0
    self.b_noise.minBound = -8.0
  }

  var parameters: [any LazyValue] {
    var params: [any LazyValue] = []
    for i in 0..<numLayers {
      params.append(trunkWeights[i])
      params.append(trunkBiases[i])
    }
    params.append(contentsOf: [W_harm, b_harm, W_hgain, b_hgain, W_noise, b_noise])
    return params
  }

  func forward(features: Tensor) -> DecoderControls {
    // trunk: [F,3] -> [F,H] -> ... -> [F,H]
    var hidden = features
    for i in 0..<numLayers {
      hidden = tanh(hidden.matmul(trunkWeights[i]) + trunkBiases[i])
    }

    // harmonic amplitudes [F,K]
    let harmLogits = hidden.matmul(W_harm) + b_harm
    let harmonicAmps = sigmoid(harmLogits)

    // global harmonic gain [F,1]
    let hGainLogits = hidden.matmul(W_hgain) + b_hgain
    let harmonicGain = sigmoid(hGainLogits)

    // broadband noise gain [F,1]
    let nGainLogits = hidden.matmul(W_noise) + b_noise
    let noiseGain = sigmoid(nGainLogits)

    return DecoderControls(
      harmonicAmps: harmonicAmps,
      harmonicGain: harmonicGain,
      noiseGain: noiseGain
    )
  }

  func snapshots() -> [NamedTensorSnapshot] {
    var snaps: [NamedTensorSnapshot] = []
    for i in 0..<numLayers {
      snaps.append(snapshot("W\(i+1)", trunkWeights[i]))
      snaps.append(snapshot("b\(i+1)", trunkBiases[i]))
    }
    snaps.append(contentsOf: [
      snapshot("W_harm", W_harm),
      snapshot("b_harm", b_harm),
      snapshot("W_hgain", W_hgain),
      snapshot("b_hgain", b_hgain),
      snapshot("W_noise", W_noise),
      snapshot("b_noise", b_noise),
    ])
    return snaps
  }

  func loadSnapshots(_ snapshots: [NamedTensorSnapshot]) {
    let byName = Dictionary(uniqueKeysWithValues: snapshots.map { ($0.name, $0) })
    for i in 0..<numLayers {
      loadTensor(trunkWeights[i], from: byName["W\(i+1)"])
      loadTensor(trunkBiases[i], from: byName["b\(i+1)"])
    }
    loadTensor(W_harm, from: byName["W_harm"])
    loadTensor(b_harm, from: byName["b_harm"])
    loadTensor(W_hgain, from: byName["W_hgain"])
    loadTensor(b_hgain, from: byName["b_hgain"])
    loadTensor(W_noise, from: byName["W_noise"])
    loadTensor(b_noise, from: byName["b_noise"])
  }

  private func snapshot(_ name: String, _ tensor: Tensor) -> NamedTensorSnapshot {
    NamedTensorSnapshot(name: name, shape: tensor.shape, data: tensor.getData() ?? [])
  }

  private func loadTensor(_ tensor: Tensor, from snapshot: NamedTensorSnapshot?) {
    guard let snapshot else { return }
    guard snapshot.shape == tensor.shape else {
      return
    }
    guard snapshot.data.count == tensor.shape.reduce(1, *) else {
      return
    }
    tensor.updateDataLazily(snapshot.data)
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
}

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
  let numHarmonics: Int

  // Shared trunk
  let W1: Tensor
  let b1: Tensor

  // Heads
  let W_harm: Tensor
  let b_harm: Tensor

  let W_hgain: Tensor
  let b_hgain: Tensor

  let W_noise: Tensor
  let b_noise: Tensor

  init(config: DDSPE2EConfig) {
    self.hiddenSize = config.modelHiddenSize
    self.numHarmonics = config.numHarmonics

    var rng = SeededGenerator(seed: config.seed)

    self.W1 = Tensor.param([inputSize, hiddenSize], data: Self.randomArray(
      count: inputSize * hiddenSize,
      scale: 0.08,
      rng: &rng
    ))
    self.b1 = Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize))

    self.W_harm = Tensor.param([hiddenSize, numHarmonics], data: Self.randomArray(
      count: hiddenSize * numHarmonics,
      scale: 0.06,
      rng: &rng
    ))
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
    [W1, b1, W_harm, b_harm, W_hgain, b_hgain, W_noise, b_noise]
  }

  func forward(features: Tensor) -> DecoderControls {
    // trunk: [F,3] -> [F,H]
    let hidden = tanh(features.matmul(W1) + b1)

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
    return [
      snapshot("W1", W1),
      snapshot("b1", b1),
      snapshot("W_harm", W_harm),
      snapshot("b_harm", b_harm),
      snapshot("W_hgain", W_hgain),
      snapshot("b_hgain", b_hgain),
      snapshot("W_noise", W_noise),
      snapshot("b_noise", b_noise),
    ]
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
}

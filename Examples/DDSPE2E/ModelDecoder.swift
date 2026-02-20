import DGenLazy
import Foundation

struct DecoderControls {
  var harmonicAmps: Tensor      // [frames, numHarmonics]
  var harmonicGain: Tensor      // [frames, 1]
  var noiseGain: Tensor         // [frames, 1]
  var noiseFilter: Tensor?      // [frames, noiseFilterSize] — nil when noise filter disabled
  var harmonicHeadMode: HarmonicHeadMode
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
  let harmonicHeadMode: HarmonicHeadMode
  var softmaxTemperature: Float
  let softmaxAmpFloor: Float
  let softmaxGainMinDB: Float
  let softmaxGainMaxDB: Float
  let enableNoiseFilter: Bool
  let noiseFilterSize: Int

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

  // Learned FIR filter head (nil when enableFIRNoise is false)
  let W_filter: Tensor?
  let b_filter: Tensor?

  private static let ln10Over20: Float = 0.11512925464970229  // ln(10)/20

  init(config: DDSPE2EConfig) {
    self.hiddenSize = config.modelHiddenSize
    self.numLayers = max(1, config.modelNumLayers)
    self.numHarmonics = config.numHarmonics
    self.harmonicHeadMode = config.harmonicHeadMode
    self.softmaxTemperature = config.softmaxTemperature
    self.softmaxAmpFloor = config.softmaxAmpFloor
    self.softmaxGainMinDB = config.softmaxGainMinDB
    self.softmaxGainMaxDB = config.softmaxGainMaxDB
    self.enableNoiseFilter = config.enableNoiseFilter
    self.noiseFilterSize = max(2, config.noiseFilterSize)

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

    // Learned FIR filter: [hiddenSize → noiseFilterSize], sigmoid → positive taps in (0,1)
    if config.enableNoiseFilter {
      let K = max(2, config.noiseFilterSize)
      self.W_filter = Tensor.param(
        [hiddenSize, K],
        data: Self.randomArray(count: hiddenSize * K, scale: 0.05, rng: &rng))
      self.b_filter = Tensor.param([1, K], data: [Float](repeating: 0.0, count: K))
    } else {
      self.W_filter = nil
      self.b_filter = nil
    }
  }

  var parameters: [any LazyValue] {
    var params: [any LazyValue] = []
    for i in 0..<numLayers {
      params.append(trunkWeights[i])
      params.append(trunkBiases[i])
    }
    params.append(contentsOf: [W_harm, b_harm, W_hgain, b_hgain, W_noise, b_noise])
    if let wf = W_filter, let bf = b_filter {
      params.append(contentsOf: [wf, bf])
    }
    return params
  }

  func forward(features: Tensor) -> DecoderControls {
    // trunk: [F,3] -> [F,H] -> ... -> [F,H]
    var hidden = features
    for i in 0..<numLayers {
      hidden = tanh(hidden.matmul(trunkWeights[i]) + trunkBiases[i])
    }

    // Harmonic head:
    // - legacy: sigmoid amplitudes + sigmoid gain
    // - normalized: softplus amplitudes normalized per frame + softplus gain
    // - softmax-db: softmax amplitudes + bounded dB gain mapped back to linear
    // - exp-sigmoid: DDSP-like positive controls; synth path handles Nyquist-aware renorm
    let harmLogits = hidden.matmul(W_harm) + b_harm
    let hGainLogits = hidden.matmul(W_hgain) + b_hgain
    let harmonicAmps: Tensor
    let harmonicGain: Tensor
    switch harmonicHeadMode {
    case .legacy:
      harmonicAmps = sigmoid(harmLogits)
      harmonicGain = sigmoid(hGainLogits)
    case .normalized:
      let harmonicPositive = Self.softplus(harmLogits)
      let rows = hidden.shape[0]
      let harmonicDenom =
        harmonicPositive.sum(axis: 1).reshape([rows, 1]).expand([rows, numHarmonics]) + 1e-6
      harmonicAmps = harmonicPositive / harmonicDenom
      harmonicGain = Self.softplus(hGainLogits)
    case .softmaxDB:
      let temperature = max(1e-4, softmaxTemperature)
      let softmaxDist = (harmLogits / temperature).softmax(axis: 1)
      if softmaxAmpFloor > 0 {
        let floorMix = min(max(softmaxAmpFloor, 0), 1)
        let uniform = floorMix / Float(max(1, numHarmonics))
        harmonicAmps = softmaxDist * (1.0 - floorMix) + uniform
      } else {
        harmonicAmps = softmaxDist
      }
      let gainUnit = sigmoid(hGainLogits)
      let gainDb =
        softmaxGainMinDB + gainUnit * (softmaxGainMaxDB - softmaxGainMinDB)
      harmonicGain = (gainDb * Self.ln10Over20).exp()
    case .expSigmoid:
      harmonicAmps = Self.expSigmoid(harmLogits)
      harmonicGain = Self.expSigmoid(hGainLogits)
    }

    // broadband noise gain [F,1]
    let nGainLogits = hidden.matmul(W_noise) + b_noise
    let noiseGain = harmonicHeadMode == .expSigmoid ? Self.expSigmoid(nGainLogits) : sigmoid(nGainLogits)

    // learned FIR filter taps [F,K] — sigmoid keeps taps positive in (0,1)
    var noiseFilter: Tensor? = nil
    if enableNoiseFilter, let wf = W_filter, let bf = b_filter {
      let filterLogits = hidden.matmul(wf) + bf
      noiseFilter = sigmoid(filterLogits)
    }

    return DecoderControls(
      harmonicAmps: harmonicAmps,
      harmonicGain: harmonicGain,
      noiseGain: noiseGain,
      noiseFilter: noiseFilter,
      harmonicHeadMode: harmonicHeadMode
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
    if let wf = W_filter, let bf = b_filter {
      snaps.append(contentsOf: [snapshot("W_filter", wf), snapshot("b_filter", bf)])
    }
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
    if let wf = W_filter, let bf = b_filter {
      loadTensor(wf, from: byName["W_filter"])
      loadTensor(bf, from: byName["b_filter"])
    }
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

  /// Numerically stable softplus: log(1 + exp(x)).
  /// Implemented as max(x, 0) + log(1 + exp(-abs(x))) to avoid overflow.
  private static func softplus(_ x: Tensor) -> Tensor {
    let positive = max(x, 0.0)
    let correction = (1.0 + (-abs(x)).exp()).log()
    return positive + correction
  }

  /// DDSP exp_sigmoid: maxValue * sigmoid(x)^log(exponent) + threshold
  private static func expSigmoid(
    _ x: Tensor,
    exponent: Float = 10.0,
    maxValue: Float = 2.0,
    threshold: Float = 1e-7
  ) -> Tensor {
    let shaped = sigmoid(x).pow(Float(Foundation.log(Double(exponent))))
    return maxValue * shaped + threshold
  }
}

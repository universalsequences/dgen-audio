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

private struct TransformerLayerParams {
  let W_q: Tensor
  let W_k: Tensor
  let W_v: Tensor
  let W_o: Tensor
  let W_ff1: Tensor
  let b_ff1: Tensor
  let W_ff2: Tensor
  let b_ff2: Tensor
  let ln1Gamma: Tensor
  let ln1Beta: Tensor
  let ln2Gamma: Tensor
  let ln2Beta: Tensor
}

final class DDSPDecoderModel {
  let inputSize: Int = 5
  let decoderBackbone: DecoderBackbone
  let hiddenSize: Int
  let numLayers: Int
  let transformerCausal: Bool
  let transformerUsePositionalEncoding: Bool
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
  private let transformerInputW: Tensor?
  private let transformerInputB: Tensor?
  private let transformerLayers: [TransformerLayerParams]

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
    self.decoderBackbone = config.decoderBackbone
    switch config.decoderBackbone {
    case .mlp:
      self.hiddenSize = config.modelHiddenSize
      self.numLayers = max(1, config.modelNumLayers)
    case .transformer:
      self.hiddenSize = max(1, config.transformerDModel)
      self.numLayers = max(1, config.transformerLayers)
    }
    self.transformerCausal = config.transformerCausal
    self.transformerUsePositionalEncoding = config.transformerUsePositionalEncoding
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
    var trInputW: Tensor? = nil
    var trInputB: Tensor? = nil
    var trLayers: [TransformerLayerParams] = []
    switch decoderBackbone {
    case .mlp:
      for i in 0..<numLayers {
        let fanIn = i == 0 ? inputSize : hiddenSize
        let scale: Float = sqrt(2.0 / Float(fanIn)) * 0.5
        let W = Tensor.param([fanIn, hiddenSize], data: Self.randomArray(
          count: fanIn * hiddenSize, scale: scale, rng: &rng))
        let b = Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize))
        weights.append(W)
        biases.append(b)
      }
    case .transformer:
      let inScale: Float = sqrt(2.0 / Float(inputSize)) * 0.5
      trInputW = Tensor.param(
        [inputSize, hiddenSize],
        data: Self.randomArray(count: inputSize * hiddenSize, scale: inScale, rng: &rng))
      trInputB = Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize))

      let ffMultiplier = max(1, config.transformerFFMultiplier)
      let ffSize = hiddenSize * ffMultiplier
      let attnScale: Float = sqrt(2.0 / Float(hiddenSize)) * 0.4
      let ff1Scale: Float = sqrt(2.0 / Float(hiddenSize)) * 0.4
      let ff2Scale: Float = sqrt(2.0 / Float(ffSize)) * 0.4
      for _ in 0..<numLayers {
        let layer = TransformerLayerParams(
          W_q: Tensor.param(
            [hiddenSize, hiddenSize],
            data: Self.randomArray(count: hiddenSize * hiddenSize, scale: attnScale, rng: &rng)),
          W_k: Tensor.param(
            [hiddenSize, hiddenSize],
            data: Self.randomArray(count: hiddenSize * hiddenSize, scale: attnScale, rng: &rng)),
          W_v: Tensor.param(
            [hiddenSize, hiddenSize],
            data: Self.randomArray(count: hiddenSize * hiddenSize, scale: attnScale, rng: &rng)),
          W_o: Tensor.param(
            [hiddenSize, hiddenSize],
            data: Self.randomArray(count: hiddenSize * hiddenSize, scale: attnScale, rng: &rng)),
          W_ff1: Tensor.param(
            [hiddenSize, ffSize],
            data: Self.randomArray(count: hiddenSize * ffSize, scale: ff1Scale, rng: &rng)),
          b_ff1: Tensor.param([1, ffSize], data: [Float](repeating: 0.0, count: ffSize)),
          W_ff2: Tensor.param(
            [ffSize, hiddenSize],
            data: Self.randomArray(count: ffSize * hiddenSize, scale: ff2Scale, rng: &rng)),
          b_ff2: Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize)),
          ln1Gamma: Tensor.param([1, hiddenSize], data: [Float](repeating: 1.0, count: hiddenSize)),
          ln1Beta: Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize)),
          ln2Gamma: Tensor.param([1, hiddenSize], data: [Float](repeating: 1.0, count: hiddenSize)),
          ln2Beta: Tensor.param([1, hiddenSize], data: [Float](repeating: 0.0, count: hiddenSize))
        )
        trLayers.append(layer)
      }
    }
    self.trunkWeights = weights
    self.trunkBiases = biases
    self.transformerInputW = trInputW
    self.transformerInputB = trInputB
    self.transformerLayers = trLayers

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
    switch decoderBackbone {
    case .mlp:
      for i in 0..<numLayers {
        params.append(trunkWeights[i])
        params.append(trunkBiases[i])
      }
    case .transformer:
      if let wIn = transformerInputW, let bIn = transformerInputB {
        params.append(contentsOf: [wIn, bIn])
      }
      for layer in transformerLayers {
        params.append(contentsOf: [
          layer.W_q, layer.W_k, layer.W_v, layer.W_o,
          layer.W_ff1, layer.b_ff1, layer.W_ff2, layer.b_ff2,
          layer.ln1Gamma, layer.ln1Beta, layer.ln2Gamma, layer.ln2Beta,
        ])
      }
    }
    params.append(contentsOf: [W_harm, b_harm, W_hgain, b_hgain, W_noise, b_noise])
    if let wf = W_filter, let bf = b_filter {
      params.append(contentsOf: [wf, bf])
    }
    return params
  }

  func forward(features: Tensor, batchSize: Int = 1, featureFrames: Int? = nil) -> DecoderControls {
    let hidden: Tensor
    switch decoderBackbone {
    case .mlp:
      hidden = forwardMLP(features: features)
    case .transformer:
      hidden = forwardTransformer(features: features, batchSize: batchSize, featureFrames: featureFrames)
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
    switch decoderBackbone {
    case .mlp:
      for i in 0..<numLayers {
        snaps.append(snapshot("W\(i+1)", trunkWeights[i]))
        snaps.append(snapshot("b\(i+1)", trunkBiases[i]))
      }
    case .transformer:
      if let wIn = transformerInputW, let bIn = transformerInputB {
        snaps.append(snapshot("tr_in_W", wIn))
        snaps.append(snapshot("tr_in_b", bIn))
      }
      for i in 0..<transformerLayers.count {
        let layer = transformerLayers[i]
        let prefix = "tr_l\(i+1)"
        snaps.append(contentsOf: [
          snapshot("\(prefix)_Wq", layer.W_q),
          snapshot("\(prefix)_Wk", layer.W_k),
          snapshot("\(prefix)_Wv", layer.W_v),
          snapshot("\(prefix)_Wo", layer.W_o),
          snapshot("\(prefix)_Wff1", layer.W_ff1),
          snapshot("\(prefix)_bff1", layer.b_ff1),
          snapshot("\(prefix)_Wff2", layer.W_ff2),
          snapshot("\(prefix)_bff2", layer.b_ff2),
          snapshot("\(prefix)_ln1_g", layer.ln1Gamma),
          snapshot("\(prefix)_ln1_b", layer.ln1Beta),
          snapshot("\(prefix)_ln2_g", layer.ln2Gamma),
          snapshot("\(prefix)_ln2_b", layer.ln2Beta),
        ])
      }
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
    switch decoderBackbone {
    case .mlp:
      for i in 0..<numLayers {
        loadTensor(trunkWeights[i], from: byName["W\(i+1)"])
        loadTensor(trunkBiases[i], from: byName["b\(i+1)"])
      }
    case .transformer:
      if let wIn = transformerInputW, let bIn = transformerInputB {
        loadTensor(wIn, from: byName["tr_in_W"])
        loadTensor(bIn, from: byName["tr_in_b"])
      }
      for i in 0..<transformerLayers.count {
        let layer = transformerLayers[i]
        let prefix = "tr_l\(i+1)"
        loadTensor(layer.W_q, from: byName["\(prefix)_Wq"])
        loadTensor(layer.W_k, from: byName["\(prefix)_Wk"])
        loadTensor(layer.W_v, from: byName["\(prefix)_Wv"])
        loadTensor(layer.W_o, from: byName["\(prefix)_Wo"])
        loadTensor(layer.W_ff1, from: byName["\(prefix)_Wff1"])
        loadTensor(layer.b_ff1, from: byName["\(prefix)_bff1"])
        loadTensor(layer.W_ff2, from: byName["\(prefix)_Wff2"])
        loadTensor(layer.b_ff2, from: byName["\(prefix)_bff2"])
        loadTensor(layer.ln1Gamma, from: byName["\(prefix)_ln1_g"])
        loadTensor(layer.ln1Beta, from: byName["\(prefix)_ln1_b"])
        loadTensor(layer.ln2Gamma, from: byName["\(prefix)_ln2_g"])
        loadTensor(layer.ln2Beta, from: byName["\(prefix)_ln2_b"])
      }
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

  private func forwardMLP(features: Tensor) -> Tensor {
    var hidden = features
    for i in 0..<numLayers {
      hidden = tanh(hidden.matmul(trunkWeights[i]) + trunkBiases[i])
    }
    return hidden
  }

  private func forwardTransformer(features: Tensor, batchSize: Int, featureFrames: Int?) -> Tensor {
    guard let wIn = transformerInputW, let bIn = transformerInputB else {
      return features
    }
    let rows = features.shape[0]
    var hidden = features.matmul(wIn) + bIn
    if transformerUsePositionalEncoding {
      hidden = hidden + Self.positionalEncoding(
        length: rows,
        dim: hiddenSize,
        batchSize: batchSize,
        featureFrames: featureFrames
      )
    }
    for layer in transformerLayers {
      hidden = transformerBlock(
        hidden,
        layer: layer,
        batchSize: batchSize,
        featureFrames: featureFrames
      )
    }
    return hidden
  }

  private func transformerBlock(
    _ x: Tensor,
    layer: TransformerLayerParams,
    batchSize: Int,
    featureFrames: Int?
  ) -> Tensor {
    let attnInput = Self.layerNorm(x, gamma: layer.ln1Gamma, beta: layer.ln1Beta)
    let attnOut = selfAttention(
      attnInput,
      layer: layer,
      batchSize: batchSize,
      featureFrames: featureFrames
    )
    let attnResidual = x + attnOut

    let ffInput = Self.layerNorm(attnResidual, gamma: layer.ln2Gamma, beta: layer.ln2Beta)
    let ffHidden = relu(ffInput.matmul(layer.W_ff1) + layer.b_ff1)
    let ffOut = ffHidden.matmul(layer.W_ff2) + layer.b_ff2
    return attnResidual + ffOut
  }

  private func selfAttention(
    _ x: Tensor,
    layer: TransformerLayerParams,
    batchSize: Int,
    featureFrames: Int?
  ) -> Tensor {
    let frames = x.shape[0]
    let scale = 1.0 / Foundation.sqrt(Float(max(1, hiddenSize)))
    let Q = x.matmul(layer.W_q)
    let K = x.matmul(layer.W_k)
    let V = x.matmul(layer.W_v)
    var scores = Q.matmul(K.transpose()) * scale
    if let mask = Self.attentionMask(
      length: frames,
      batchSize: batchSize,
      featureFrames: featureFrames,
      causal: transformerCausal
    ) {
      scores = scores + mask
    }
    let weights = scores.softmax(axis: -1)
    let context = weights.matmul(V)
    return context.matmul(layer.W_o)
  }

  private static func layerNorm(_ x: Tensor, gamma: Tensor, beta: Tensor, eps: Float = 1e-5) -> Tensor {
    guard x.shape.count == 2 else { return x }
    let rows = x.shape[0]
    let cols = x.shape[1]
    let mean = x.mean(axis: 1).reshape([rows, 1]).expand([rows, cols])
    let centered = x - mean
    let variance = (centered * centered).mean(axis: 1).reshape([rows, 1]).expand([rows, cols])
    let normalized = centered / (variance + eps).sqrt()
    let g = gamma.expand([rows, cols])
    let b = beta.expand([rows, cols])
    return normalized * g + b
  }

  private static func positionalEncoding(length: Int, dim: Int, batchSize: Int, featureFrames: Int?) -> Tensor {
    guard length > 0, dim > 0 else {
      return Tensor.zeros([max(0, length), max(0, dim)])
    }
    let useChunkedPositions =
      batchSize > 1
      && (featureFrames ?? 0) > 0
      && (featureFrames ?? 0) * batchSize == length
    let perChunkFrames = featureFrames ?? length

    var data = [Float](repeating: 0.0, count: length * dim)
    for pos in 0..<length {
      let positionIndex = useChunkedPositions ? (pos % perChunkFrames) : pos
      for i in stride(from: 0, to: dim, by: 2) {
        let exponent = Float(i) / Float(dim)
        let denom = Foundation.pow(10_000.0, Double(exponent))
        let angle = Float(positionIndex) / Float(denom)
        data[pos * dim + i] = Foundation.sin(angle)
        if i + 1 < dim {
          data[pos * dim + i + 1] = Foundation.cos(angle)
        }
      }
    }
    return Tensor(data).reshape([length, dim])
  }

  private static func causalMask(length: Int) -> Tensor {
    guard length > 0 else { return Tensor.zeros([0, 0]) }
    var data = [Float](repeating: 0.0, count: length * length)
    for row in 0..<length {
      for col in (row + 1)..<length {
        data[row * length + col] = -1e9
      }
    }
    return Tensor(data).reshape([length, length])
  }

  private static func attentionMask(
    length: Int,
    batchSize: Int,
    featureFrames: Int?,
    causal: Bool
  ) -> Tensor? {
    guard length > 0 else { return nil }
    if batchSize <= 1 {
      return causal ? causalMask(length: length) : nil
    }
    guard let featureFrames, featureFrames > 0, featureFrames * batchSize == length else {
      return causal ? causalMask(length: length) : nil
    }

    var data = [Float](repeating: -1e9, count: length * length)
    for batch in 0..<batchSize {
      let start = batch * featureFrames
      let end = start + featureFrames
      for row in start..<end {
        let colEnd = causal ? (row + 1) : end
        for col in start..<colEnd {
          data[row * length + col] = 0.0
        }
      }
    }
    return Tensor(data).reshape([length, length])
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

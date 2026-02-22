import XCTest

@testable import DDSPE2E
@testable import DGenLazy

final class DDSPE2ETransformerDecoderTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func makeTransformerConfig(
    harmonics: Int = 8,
    dModel: Int = 32,
    layers: Int = 1,
    ffMultiplier: Int = 2,
    causal: Bool = true
  ) -> DDSPE2EConfig {
    var config = DDSPE2EConfig.default
    config.seed = 7
    config.decoderBackbone = .transformer
    config.transformerDModel = dModel
    config.transformerLayers = layers
    config.transformerFFMultiplier = ffMultiplier
    config.transformerCausal = causal
    config.transformerUsePositionalEncoding = true
    config.numHarmonics = harmonics
    config.harmonicHeadMode = .legacy
    config.enableNoiseFilter = false
    return config
  }

  private func makeFeatures(frames: Int, deltaLastFrame: Float = 0) -> Tensor {
    var data = [[Float]]()
    data.reserveCapacity(frames)
    for i in 0..<frames {
      let x = Float(i) / Float(max(1, frames - 1))
      var row: [Float] = [
        x * 0.5 - 0.25,
        0.3 + 0.2 * x,
        i % 2 == 0 ? 1.0 : 0.0,
        i == 0 ? 0 : 0.01,
        i == 0 ? 0 : -0.01,
      ]
      if i == frames - 1 {
        row[0] += deltaLastFrame
        row[1] -= deltaLastFrame * 0.5
      }
      data.append(row)
    }
    return Tensor(data)
  }

  private func isFinite(_ values: [Float]) -> Bool {
    values.allSatisfy { !$0.isNaN && !$0.isInfinite }
  }

  private func hasNonZeroGrad(_ tensor: Tensor) throws -> Bool {
    guard let grad = tensor.grad else { return false }
    let values = try grad.realize()
    return values.contains { abs($0) > 1e-8 }
  }

  func testTransformerForwardShapeAndFiniteOutputs() throws {
    let config = makeTransformerConfig(harmonics: 12, dModel: 32, layers: 2)
    let model = DDSPDecoderModel(config: config)
    let frames = 11
    let controls = model.forward(features: makeFeatures(frames: frames))

    XCTAssertEqual(controls.harmonicAmps.shape, [frames, config.numHarmonics])
    XCTAssertEqual(controls.harmonicGain.shape, [frames, 1])
    XCTAssertEqual(controls.noiseGain.shape, [frames, 1])
    XCTAssertNil(controls.noiseFilter)

    let ampValues = try controls.harmonicAmps.realize()
    let gainValues = try controls.harmonicGain.realize()
    let noiseValues = try controls.noiseGain.realize()
    XCTAssertTrue(isFinite(ampValues), "harmonic amps must be finite")
    XCTAssertTrue(isFinite(gainValues), "harmonic gain must be finite")
    XCTAssertTrue(isFinite(noiseValues), "noise gain must be finite")
  }

  func testTransformerBackboneReceivesGradients() throws {
    let config = makeTransformerConfig(harmonics: 8, dModel: 24, layers: 2)
    let model = DDSPDecoderModel(config: config)
    let frames = 9
    let features = makeFeatures(frames: frames)
    let controls = model.forward(features: features)

    let gainTarget = Tensor([[Float]](repeating: [0.5], count: frames))
    let ampTarget = Tensor([[Float]](repeating: [Float](repeating: 0.1, count: config.numHarmonics), count: frames))
    let gainDiff = controls.harmonicGain - gainTarget
    let ampDiff = controls.harmonicAmps - ampTarget
    let loss = (gainDiff * gainDiff).mean() + (ampDiff * ampDiff).mean()
    _ = try loss.backward(frameCount: 1)

    let params = model.parameters.compactMap { $0 as? Tensor }
    XCTAssertGreaterThan(params.count, 20, "transformer model should expose transformer + head params")
    XCTAssertTrue(try hasNonZeroGrad(params[0]), "input projection should get gradients")
    XCTAssertTrue(try hasNonZeroGrad(params[2]), "layer-1 query projection should get gradients")
    XCTAssertTrue(try hasNonZeroGrad(params[14]), "layer-2 query projection should get gradients")
  }

  func testCausalTransformerDoesNotUseFutureFrames() throws {
    let config = makeTransformerConfig(harmonics: 6, dModel: 16, layers: 1, causal: true)
    let model = DDSPDecoderModel(config: config)
    let frames = 8

    let controlsA = model.forward(features: makeFeatures(frames: frames, deltaLastFrame: 0))
    let controlsB = model.forward(features: makeFeatures(frames: frames, deltaLastFrame: 5.0))

    let gainA = try controlsA.harmonicGain.realize()
    let gainB = try controlsB.harmonicGain.realize()
    let ampsA = try controlsA.harmonicAmps.realize()
    let ampsB = try controlsB.harmonicAmps.realize()
    let K = config.numHarmonics

    for frame in 0..<(frames - 1) {
      XCTAssertEqual(gainA[frame], gainB[frame], accuracy: 1e-5, "future frame changed past gain")
      let base = frame * K
      for h in 0..<K {
        XCTAssertEqual(ampsA[base + h], ampsB[base + h], accuracy: 1e-5, "future frame changed past harmonics")
      }
    }
  }

  func testTransformerTrainingLoopStaysFinite() throws {
    let config = makeTransformerConfig(harmonics: 8, dModel: 16, layers: 1)
    let model = DDSPDecoderModel(config: config)
    let optimizer = Adam(params: model.parameters, lr: 1e-3)
    let frames = 10

    for _ in 0..<8 {
      let controls = model.forward(features: makeFeatures(frames: frames))
      let gainTarget = Tensor([[Float]](repeating: [0.4], count: frames))
      let noiseTarget = Tensor([[Float]](repeating: [0.2], count: frames))
      let ampTarget = Tensor([[Float]](repeating: [Float](repeating: 1.0 / Float(config.numHarmonics), count: config.numHarmonics), count: frames))

      let gainDiff = controls.harmonicGain - gainTarget
      let noiseDiff = controls.noiseGain - noiseTarget
      let ampDiff = controls.harmonicAmps - ampTarget
      let loss =
        (gainDiff * gainDiff).mean()
        + (noiseDiff * noiseDiff).mean()
        + (ampDiff * ampDiff).mean()

      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      XCTAssertFalse(lossValue.isNaN, "loss must remain finite")
      XCTAssertFalse(lossValue.isInfinite, "loss must remain finite")
      optimizer.step()
      optimizer.zeroGrad()
    }

    let final = model.forward(features: makeFeatures(frames: frames))
    XCTAssertTrue(isFinite(try final.harmonicAmps.realize()))
    XCTAssertTrue(isFinite(try final.harmonicGain.realize()))
    XCTAssertTrue(isFinite(try final.noiseGain.realize()))
  }

  func testBatchedTransformerMaskPreventsCrossChunkLeakage() throws {
    let config = makeTransformerConfig(harmonics: 6, dModel: 16, layers: 1, causal: true)
    let model = DDSPDecoderModel(config: config)
    let batchSize = 2
    let featureFrames = 5

    var chunk0 = [[Float]]()
    var chunk1A = [[Float]]()
    var chunk1B = [[Float]]()
    for i in 0..<featureFrames {
      let x = Float(i) / Float(max(1, featureFrames - 1))
      chunk0.append([x, 0.2 + x * 0.1, 1, i == 0 ? 0 : 0.01, i == 0 ? 0 : -0.01])
      chunk1A.append([x * 0.5, 0.7 - x * 0.1, 0, i == 0 ? 0 : -0.02, i == 0 ? 0 : 0.02])
      chunk1B.append([x * 3.0 + 5.0, -2.0 - x, 1, 0.5, -0.5])
    }

    let featuresA = Tensor(chunk0 + chunk1A)
    let featuresB = Tensor(chunk0 + chunk1B)

    let outA = model.forward(features: featuresA, batchSize: batchSize, featureFrames: featureFrames)
    let outB = model.forward(features: featuresB, batchSize: batchSize, featureFrames: featureFrames)
    let gainA = try outA.harmonicGain.realize()
    let gainB = try outB.harmonicGain.realize()

    for frame in 0..<featureFrames {
      XCTAssertEqual(
        gainA[frame],
        gainB[frame],
        accuracy: 1e-5,
        "chunk-0 output changed when only chunk-1 inputs changed"
      )
    }
  }
}

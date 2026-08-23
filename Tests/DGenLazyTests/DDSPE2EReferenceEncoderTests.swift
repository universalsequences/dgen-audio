import Foundation
import XCTest

@testable import DDSPE2E
@testable import DGenLazy

final class DDSPE2EReferenceEncoderTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func makeTemporalConfig(seed: UInt64) -> DDSPE2EConfig {
    var config = DDSPE2EConfig.default
    config.seed = seed
    config.decoderBackbone = .transformer
    config.transformerDModel = 16
    config.transformerLayers = 1
    config.numHarmonics = 8
    config.numInstruments = 2
    config.referenceConditioning = true
    config.referenceEncoderMode = .temporal
    config.referenceTimeFrames = 8
    config.referenceMelBins = 12
    config.referenceEncoderHidden = 16
    config.referenceLatentSize = 8
    config.enableNoiseFilter = true
    config.noiseFilterSize = 5
    return config
  }

  private func referenceTensor(seedValue: Float, config: DDSPE2EConfig) -> Tensor {
    var rows = [[Float]]()
    for t in 0..<config.referenceTimeFrames {
      var row = [Float](repeating: 0, count: config.referenceMelBins)
      for m in 0..<config.referenceMelBins {
        let i = t * config.referenceMelBins + m
        row[m] = tanh(sin(seedValue + Float(i) * 0.37) * 0.8)
      }
      rows.append(row)
    }
    return Tensor(rows)
  }

  // MARK: Old cache compatibility

  func testOldChunkJSONDecodesWithoutTemporalFields() throws {
    let legacyJSON = """
      {
        "id": "chunk_00000000",
        "sourceFile": "Flute/note.wav",
        "instrumentIndex": 1,
        "sourceSampleRate": 44100,
        "sampleRate": 16000,
        "startSample": 0,
        "audio": [0.0, 0.1, -0.1, 0.05],
        "f0Hz": [440.0],
        "loudnessDB": [-20.0],
        "uvMask": [1.0]
      }
      """
    let chunk = try JSONDecoder().decode(CachedChunk.self, from: Data(legacyJSON.utf8))
    XCTAssertNil(chunk.timbreFeatures)
    XCTAssertNil(chunk.timbreFrames)
    XCTAssertNil(chunk.timbreFrameCount)
    XCTAssertNil(chunk.timbreMelBins)

    // The loader-side fallback must synthesize frames from audio.
    var config = DDSPE2EConfig.default
    config.referenceTimeFrames = 4
    config.referenceMelBins = 8
    config.frameSize = 4
    config.frameHop = 2
    let frames = referenceTimbreFrames(chunk: chunk, config: config)
    XCTAssertEqual(frames.count, 4 * 8)
    XCTAssertTrue(frames.allSatisfy { $0.isFinite })
  }

  func testConfigDecodesWithoutNewReferenceKeys() throws {
    let legacyConfigJSON = """
      { "referenceConditioning": true, "referenceFeatureSize": 64 }
      """
    let config = try JSONDecoder().decode(DDSPE2EConfig.self, from: Data(legacyConfigJSON.utf8))
    XCTAssertEqual(config.referenceEncoderMode, .averaged)
    XCTAssertEqual(config.referenceFeatureSize, 64)
    XCTAssertEqual(config.referenceTimeFrames, DDSPE2EConfig.default.referenceTimeFrames)
    XCTAssertEqual(config.referenceMelBins, DDSPE2EConfig.default.referenceMelBins)
  }

  // MARK: Feature extraction

  func testTimbreLogMelFramesShapeAndDistinctness() {
    let sampleRate: Float = 16_000
    let n = 4_096
    var sine = [Float](repeating: 0, count: n)
    var rng = SeededGenerator(seed: 7)
    var noise = [Float](repeating: 0, count: n)
    for i in 0..<n {
      sine[i] = 0.5 * sin(2 * Float.pi * 200 * Float(i) / sampleRate)
      noise[i] = (Float(rng.next() & 0xFFFF) / Float(UInt16.max) - 0.5) * 0.8
    }
    let timeFrames = 8
    let melBins = 16
    let sineFrames = FeatureExtractor.timbreLogMelFrames(
      samples: sine, sampleRate: sampleRate, frameSize: 1024, frameHop: 256,
      timeFrames: timeFrames, melBins: melBins)
    let noiseFrames = FeatureExtractor.timbreLogMelFrames(
      samples: noise, sampleRate: sampleRate, frameSize: 1024, frameHop: 256,
      timeFrames: timeFrames, melBins: melBins)

    XCTAssertEqual(sineFrames.count, timeFrames * melBins)
    XCTAssertEqual(noiseFrames.count, timeFrames * melBins)
    XCTAssertTrue(sineFrames.allSatisfy { $0.isFinite && abs($0) <= 1.0 })
    XCTAssertTrue(noiseFrames.allSatisfy { $0.isFinite && abs($0) <= 1.0 })

    var diff: Float = 0
    for i in 0..<sineFrames.count {
      diff += abs(sineFrames[i] - noiseFrames[i])
    }
    XCTAssertGreaterThan(
      diff / Float(sineFrames.count), 0.05,
      "sine and noise references should have clearly different log-mel frames")
  }

  // MARK: Temporal encoder

  func testTemporalEncoderShapesAndDistinctLatents() throws {
    let config = makeTemporalConfig(seed: 11)
    let model = DDSPDecoderModel(config: config)

    let refA = referenceTensor(seedValue: 0.0, config: config)
    let refB = referenceTensor(seedValue: 2.5, config: config)

    guard let zA = model.encodeReferenceLatent(refA),
      let zB = model.encodeReferenceLatent(refB)
    else {
      XCTFail("temporal model must produce a reference latent")
      return
    }
    XCTAssertEqual(zA.shape, [1, config.referenceLatentSize])

    let zAData = try zA.realize()
    let zBData = try zB.realize()
    XCTAssertEqual(zAData.count, config.referenceLatentSize)
    XCTAssertTrue(zAData.allSatisfy { $0.isFinite })
    XCTAssertTrue(zBData.allSatisfy { $0.isFinite })

    var diff: Float = 0
    for i in 0..<zAData.count {
      diff += abs(zAData[i] - zBData[i])
    }
    XCTAssertGreaterThan(
      diff, 1e-4, "distinct references must map to distinct latents at initialization")
  }

  func testAveragedModelHasNoTemporalLatentPath() {
    var config = makeTemporalConfig(seed: 11)
    config.referenceEncoderMode = .averaged
    let model = DDSPDecoderModel(config: config)
    let ref = referenceTensor(seedValue: 0.0, config: config)
    XCTAssertNil(model.encodeReferenceLatent(ref))
  }

  func testForwardControlsRespondToReferenceSwap() throws {
    let config = makeTemporalConfig(seed: 3)
    let model = DDSPDecoderModel(config: config)

    let features = Tensor(
      [
        [0.1, 0.3, 1.0, 0.0, 0.0],
        [0.2, 0.4, 1.0, 0.1, 0.1],
        [0.3, 0.5, 1.0, 0.1, 0.1],
        [0.4, 0.6, 1.0, 0.1, 0.1],
      ])
    // Tensors must be created after any graph clear (see CLAUDE.md): build
    // each reference immediately before the forward pass that uses it.
    let refA = referenceTensor(seedValue: 0.0, config: config)
    let ampsA = try model.forward(features: features, reference: refA).harmonicAmps.realize()
    LazyGraphContext.current.clearComputationGraph()
    let featuresB = Tensor(
      [
        [0.1, 0.3, 1.0, 0.0, 0.0],
        [0.2, 0.4, 1.0, 0.1, 0.1],
        [0.3, 0.5, 1.0, 0.1, 0.1],
        [0.4, 0.6, 1.0, 0.1, 0.1],
      ])
    let refB = referenceTensor(seedValue: 2.5, config: config)
    let ampsB = try model.forward(features: featuresB, reference: refB).harmonicAmps.realize()

    XCTAssertEqual(ampsA.count, ampsB.count)
    XCTAssertTrue(ampsA.allSatisfy { $0.isFinite })
    var diff: Float = 0
    for i in 0..<ampsA.count {
      diff += abs(ampsA[i] - ampsB[i])
    }
    XCTAssertGreaterThan(
      diff, 1e-5,
      "swapping the reference must change harmonic amplitudes even at initialization")
  }

  func testConfigDecodesWithoutDynamicRangeKeysUsesDefaults() throws {
    let legacyConfigJSON = """
      { "referenceConditioning": true, "referenceEncoderMode": "temporal" }
      """
    let config = try JSONDecoder().decode(DDSPE2EConfig.self, from: Data(legacyConfigJSON.utf8))
    XCTAssertEqual(config.referenceZResidualScale, 1.0)
    XCTAssertEqual(config.referenceFiLMGammaScale, 0.5)
  }

  /// The R8 dynamic-range lever: a larger z-residual scale must widen how much
  /// a reference swap moves the harmonic amplitudes, and a wider FiLM gamma
  /// bound must change the forward output. Both models share a seed so the
  /// only difference is the scale knobs.
  func testZResidualScaleWidensReferenceSwing() throws {
    func swapDiff(zScale: Float, filmGamma: Float, seed: UInt64) throws -> Float {
      var config = makeTemporalConfig(seed: seed)
      config.referenceZResidualScale = zScale
      config.referenceFiLMGammaScale = filmGamma
      let model = DDSPDecoderModel(config: config)
      let featureRows: [[Float]] = [
        [0.1, 0.3, 1.0, 0.0, 0.0],
        [0.2, 0.4, 1.0, 0.1, 0.1],
        [0.3, 0.5, 1.0, 0.1, 0.1],
      ]
      // Tensors after each graph clear (see CLAUDE.md).
      let refA = referenceTensor(seedValue: 0.0, config: config)
      let ampsA = try model.forward(features: Tensor(featureRows), reference: refA)
        .harmonicAmps.realize()
      LazyGraphContext.current.clearComputationGraph()
      let refB = referenceTensor(seedValue: 2.5, config: config)
      let ampsB = try model.forward(features: Tensor(featureRows), reference: refB)
        .harmonicAmps.realize()
      LazyGraphContext.current.clearComputationGraph()
      var diff: Float = 0
      for i in 0..<ampsA.count { diff += abs(ampsA[i] - ampsB[i]) }
      XCTAssertTrue(ampsA.allSatisfy { $0.isFinite })
      XCTAssertTrue(ampsB.allSatisfy { $0.isFinite })
      return diff
    }

    let narrow = try swapDiff(zScale: 1.0, filmGamma: 0.5, seed: 21)
    let wide = try swapDiff(zScale: 8.0, filmGamma: 0.5, seed: 21)
    XCTAssertGreaterThan(
      wide, narrow * 2.0,
      "z residual scale 8 must widen the reference-swap swing well beyond scale 1")

    let wideGamma = try swapDiff(zScale: 1.0, filmGamma: 1.0, seed: 21)
    XCTAssertGreaterThan(
      abs(wideGamma - narrow), 1e-6,
      "widening the FiLM gamma bound must change the reference-driven output")
  }

  // MARK: Gradient correctness

  /// FD-checks the classification-loss gradient through the temporal encoder
  /// (matmuls + tanh + attention softmax over a reshaped view + pooling).
  /// A corrupted adjoint anywhere in this chain silently caps how separable z
  /// can become while still "training".
  func testPretrainClassificationGradientMatchesFiniteDifferences() throws {
    var config = makeTemporalConfig(seed: 5)
    config.referenceTimeFrames = 4
    config.referenceMelBins = 6
    config.referenceEncoderHidden = 8
    config.referenceLatentSize = 4
    let model = DDSPDecoderModel(config: config)

    func lossForward() throws -> Float {
      let ref = referenceTensor(seedValue: 0.7, config: config)
      let target = Tensor([[Float(1), 0]])
      guard let loss = model.referenceClassificationLossTensor(reference: ref, target: target)
      else {
        XCTFail("no classification loss")
        return 0
      }
      let value = try loss.realize()
      LazyGraphContext.current.clearComputationGraph()
      return value[0]
    }

    // Analytic gradients.
    let refForGrad = referenceTensor(seedValue: 0.7, config: config)
    let targetForGrad = Tensor([[Float(1), 0]])
    guard let lossTensor = model.referenceClassificationLossTensor(
      reference: refForGrad, target: targetForGrad)
    else {
      XCTFail("no classification loss")
      return
    }
    _ = try lossTensor.peek(Signal.constant(0.0)).backward(frames: 1)

    for tensor in model.temporalEncoderParameters {
      guard let analytic = tensor.grad?.getData(), var base = tensor.getData() else {
        XCTFail("missing grad or data")
        continue
      }
      var fd = [Float](repeating: 0, count: base.count)
      let eps: Float = 2e-2
      let checkedIndices = Array(stride(from: 0, to: base.count, by: max(1, base.count / 12)))
      for index in checkedIndices {
        let original = base[index]
        base[index] = original + eps
        tensor.updateDataLazily(base)
        let plus = try lossForward()
        base[index] = original - eps
        tensor.updateDataLazily(base)
        let minus = try lossForward()
        base[index] = original
        tensor.updateDataLazily(base)
        fd[index] = (plus - minus) / (2 * eps)
      }
      // Float32 loss values quantize the central difference to ~1 ULP of the
      // loss over 2*eps; only compare where the FD signal clears that floor.
      // (The attention bias also has a true gradient of exactly zero — softmax
      // is invariant to a uniform logit shift — so it never clears the floor.)
      let fdResolution: Float = 3 * (Foundation.pow(2, -23) * 0.7) / (2 * eps)
      let usable = checkedIndices.filter { abs(fd[$0]) > fdResolution }
      guard usable.count >= 3 else { continue }
      var dot: Float = 0
      var normA: Float = 0
      var normB: Float = 0
      for index in usable {
        dot += analytic[index] * fd[index]
        normA += analytic[index] * analytic[index]
        normB += fd[index] * fd[index]
      }
      let cosine = dot / Swift.max(1e-12, sqrt(normA) * sqrt(normB))
      XCTAssertGreaterThan(
        cosine, 0.97,
        "encoder gradient mismatch (cosine \(cosine)) for a tensor of shape \(tensor.shape)")
    }
  }

  // MARK: Checkpoint round-trip

  func testTemporalCheckpointRoundTrip() throws {
    let source = DDSPDecoderModel(config: makeTemporalConfig(seed: 101))
    let target = DDSPDecoderModel(config: makeTemporalConfig(seed: 999))

    let snapshots = source.snapshots()
    for expected in [
      "ref_tenc_W1", "ref_tenc_b1", "ref_tenc_W2", "ref_tenc_b2",
      "ref_tenc_Wattn", "ref_tenc_battn", "ref_tenc_Wz", "ref_tenc_bz",
      "ref_zharm_W", "ref_zhgain_W", "ref_znoise_W", "ref_zfilter_W",
      "ref_classifier_W", "ref_classifier_b",
    ] {
      XCTAssertTrue(
        snapshots.contains { $0.name == expected },
        "temporal snapshot missing \(expected)")
    }

    target.loadSnapshots(snapshots)

    let ref = referenceTensor(seedValue: 1.2, config: makeTemporalConfig(seed: 101))
    guard let zSource = source.encodeReferenceLatent(ref) else {
      XCTFail("source model must encode a latent")
      return
    }
    let zSourceData = try zSource.realize()
    LazyGraphContext.current.clearComputationGraph()
    let ref2 = referenceTensor(seedValue: 1.2, config: makeTemporalConfig(seed: 101))
    guard let zTarget = target.encodeReferenceLatent(ref2) else {
      XCTFail("target model must encode a latent")
      return
    }
    let zTargetData = try zTarget.realize()

    XCTAssertEqual(zSourceData.count, zTargetData.count)
    for i in 0..<zSourceData.count {
      XCTAssertEqual(
        zSourceData[i], zTargetData[i], accuracy: 1e-5,
        "latent mismatch after checkpoint round-trip at index \(i)")
    }
  }
}

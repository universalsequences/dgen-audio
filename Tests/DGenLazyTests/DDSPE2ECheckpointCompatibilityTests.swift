import Foundation
import XCTest

@testable import DDSPE2E
@testable import DGenLazy

final class DDSPE2ECheckpointCompatibilityTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func makeMLPConfig(
    seed: UInt64,
    hidden: Int = 16,
    layers: Int = 1,
    harmonics: Int = 8
  ) -> DDSPE2EConfig {
    var config = DDSPE2EConfig.default
    config.seed = seed
    config.decoderBackbone = .mlp
    config.modelHiddenSize = hidden
    config.modelNumLayers = layers
    config.numHarmonics = harmonics
    config.enableNoiseFilter = false
    return config
  }

  private func makeTransformerConfig(
    seed: UInt64,
    dModel: Int = 16,
    layers: Int = 1,
    harmonics: Int = 8
  ) -> DDSPE2EConfig {
    var config = DDSPE2EConfig.default
    config.seed = seed
    config.decoderBackbone = .transformer
    config.transformerDModel = dModel
    config.transformerLayers = layers
    config.transformerFFMultiplier = 2
    config.transformerCausal = true
    config.transformerUsePositionalEncoding = true
    config.numHarmonics = harmonics
    config.enableNoiseFilter = false
    return config
  }

  private func requireData(_ tensor: Tensor, _ name: String, file: StaticString = #filePath, line: UInt = #line)
    -> [Float]
  {
    guard let data = tensor.getData() else {
      XCTFail("Missing tensor data for \(name)", file: file, line: line)
      return []
    }
    return data
  }

  private func assertArraysEqual(
    _ lhs: [Float],
    _ rhs: [Float],
    accuracy: Float = 1e-6,
    message: String = "",
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
    for i in 0..<min(lhs.count, rhs.count) {
      XCTAssertEqual(lhs[i], rhs[i], accuracy: accuracy, "\(message) at index \(i)", file: file, line: line)
    }
  }

  private func hasFiniteValues(_ values: [Float]) -> Bool {
    values.allSatisfy { !$0.isNaN && !$0.isInfinite }
  }

  func testMLPSnapshotsRoundTripStillLoads() {
    let source = DDSPDecoderModel(config: makeMLPConfig(seed: 101))
    let target = DDSPDecoderModel(config: makeMLPConfig(seed: 999))

    let sourceWHarm = requireData(source.W_harm, "source.W_harm")
    let targetWHarmBefore = requireData(target.W_harm, "target.W_harm(before)")
    XCTAssertNotEqual(sourceWHarm, targetWHarmBefore, "sanity check: models should differ before load")

    target.loadSnapshots(source.snapshots())

    let targetWHarmAfter = requireData(target.W_harm, "target.W_harm(after)")
    let sourceBHarm = requireData(source.b_harm, "source.b_harm")
    let targetBHarm = requireData(target.b_harm, "target.b_harm")
    let sourceW1 = requireData(source.trunkWeights[0], "source.W1")
    let targetW1 = requireData(target.trunkWeights[0], "target.W1")

    assertArraysEqual(targetWHarmAfter, sourceWHarm, message: "W_harm mismatch after load")
    assertArraysEqual(targetBHarm, sourceBHarm, message: "b_harm mismatch after load")
    assertArraysEqual(targetW1, sourceW1, message: "W1 mismatch after load")
  }

  func testTransformerLoadsLegacyMLPSnapshotsMissingTransformerParams() throws {
    let mlp = DDSPDecoderModel(config: makeMLPConfig(seed: 123, hidden: 16, layers: 1, harmonics: 8))
    let transformer = DDSPDecoderModel(
      config: makeTransformerConfig(seed: 555, dModel: 16, layers: 1, harmonics: 8))

    // Legacy-style snapshots do not contain transformer-specific keys.
    let legacySnapshots = mlp.snapshots()
    XCTAssertTrue(legacySnapshots.contains(where: { $0.name == "W1" }), "expected legacy MLP key W1")
    XCTAssertFalse(legacySnapshots.contains(where: { $0.name.hasPrefix("tr_l") }))

    transformer.loadSnapshots(legacySnapshots)

    // Shared heads should still transfer.
    assertArraysEqual(
      requireData(transformer.W_harm, "transformer.W_harm"),
      requireData(mlp.W_harm, "mlp.W_harm"),
      message: "shared head W_harm failed to transfer from legacy snapshot"
    )
    assertArraysEqual(
      requireData(transformer.b_hgain, "transformer.b_hgain"),
      requireData(mlp.b_hgain, "mlp.b_hgain"),
      message: "shared head b_hgain failed to transfer from legacy snapshot"
    )

    // Forward remains valid after partial snapshot load.
    let features = Tensor(
      [
        [0.1, 0.3, 1.0, 0.0, 0.0],
        [0.2, 0.4, 0.0, 0.1, -0.1],
        [0.3, 0.5, 1.0, 0.1, 0.1],
        [0.4, 0.6, 0.0, 0.1, -0.1],
      ])
    let controls = transformer.forward(features: features)
    XCTAssertTrue(hasFiniteValues(try controls.harmonicAmps.realize()))
    XCTAssertTrue(hasFiniteValues(try controls.harmonicGain.realize()))
    XCTAssertTrue(hasFiniteValues(try controls.noiseGain.realize()))
  }

  func testCheckpointStoreReadWriteAllowsLoadingLegacyIntoTransformer() throws {
    let mlp = DDSPDecoderModel(config: makeMLPConfig(seed: 42, hidden: 16, layers: 1, harmonics: 8))
    let tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("ddsp_checkpoint_compat_\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

    try CheckpointStore.writeModelState(checkpointsDir: tempDir, step: 7, params: mlp.snapshots())
    let checkpointURL = tempDir.appendingPathComponent("model_step_00000007.json")
    let checkpoint = try CheckpointStore.readModelState(from: checkpointURL)
    XCTAssertEqual(checkpoint.step, 7)
    XCTAssertFalse(checkpoint.params.isEmpty)

    let transformer = DDSPDecoderModel(
      config: makeTransformerConfig(seed: 77, dModel: 16, layers: 1, harmonics: 8))
    transformer.loadSnapshots(checkpoint.params)

    let features = Tensor(
      [
        [0.0, 0.2, 1.0, 0.0, 0.0],
        [0.2, 0.3, 1.0, 0.2, -0.1],
        [0.1, 0.5, 0.0, -0.1, 0.2],
      ])
    let controls = transformer.forward(features: features)
    XCTAssertTrue(hasFiniteValues(try controls.harmonicAmps.realize()))
    XCTAssertTrue(hasFiniteValues(try controls.harmonicGain.realize()))
    XCTAssertTrue(hasFiniteValues(try controls.noiseGain.realize()))
  }
}


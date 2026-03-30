import XCTest

@testable import DGen
@testable import DGenLazy
@testable import HarmonicE2E

final class HarmonicE2EKernelTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 24_000.0
    DGenConfig.maxFrameCount = 16_384
    DGenConfig.backend = .metal
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.kernelOutputPath = nil
    DGenConfig.sampleRate = 44_100.0
    DGenConfig.maxFrameCount = 4096
    DGenConfig.backend = .metal
    super.tearDown()
  }

  func testActualHarmonicE2EGraphAvoidsSerializingControlFrames() throws {
    var config = HarmonicE2EConfig.default
    config.sampleRate = 24_000.0
    config.chunkSize = 16_384
    config.modelHiddenSize = 64
    config.modelNumLayers = 2
    config.numHarmonics = 32

    let frameCount = config.chunkSize
    let featureFrames = 64
    let kernelPath = "/tmp/harmonic_e2e_actual_graph_backward.metal"
    DGenConfig.kernelOutputPath = kernelPath

    let featuresTensor = Tensor(
      [[Float]](repeating: [Float](repeating: 0, count: 3), count: featureFrames)
    )
    let targetTensor = Tensor([Float](repeating: 0, count: frameCount))
    let synthTensors = HarmonicSynth.PreallocatedTensors(
      featureFrames: featureFrames,
      numHarmonics: config.numHarmonics
    )

    var conditioningData: [Float] = []
    conditioningData.reserveCapacity(featureFrames * 3)
    for i in 0..<featureFrames {
      let x = Float(i) / Float(max(1, featureFrames - 1))
      conditioningData.append(x)  // f0 norm
      conditioningData.append(0.35 + 0.4 * x)  // loudness norm
      conditioningData.append(1.0)  // uv
    }
    featuresTensor.updateDataLazily(conditioningData)

    let f0Frames = (0..<featureFrames).map { i in
      220.0 + (Float(i) / Float(max(1, featureFrames - 1))) * 110.0
    }
    let uvFrames = [Float](repeating: 1.0, count: featureFrames)
    synthTensors.updateChunkData(f0Frames: f0Frames, uvFrames: uvFrames)
    targetTensor.updateDataLazily([Float](repeating: 0.0, count: frameCount))

    let model = HarmonicDecoderModel(config: config)
    let controls = model.forward(features: featuresTensor)
    let prediction = HarmonicSynth.renderSignal(
      controls: controls,
      tensors: synthTensors,
      featureFrames: featureFrames,
      frameCount: frameCount,
      numHarmonics: config.numHarmonics
    )
    let target = targetTensor.toSignal(maxFrames: frameCount)
    let loss = mse(prediction, target) * 0.1

    _ = try loss.backward(frames: frameCount)

    let kernelSource = try String(contentsOfFile: kernelPath, encoding: .utf8)
    XCTAssertTrue(kernelSource.contains("kernel void"))

    let hasPathologicalSerialization =
      kernelSource.contains("DispatchMode: perFrameScaled(64)")
      && kernelSource.contains("if (id >= 0 && id < (uint)(1))")
      && kernelSource.contains("for (uint i = 0; i < t")
      && kernelSource.contains("frameCount + _frameIndex")

    XCTExpectFailure(
      "Known issue: actual HarmonicE2E backward graph serializes frameCount * 64 work in one thread."
    )
    XCTAssertFalse(
      hasPathologicalSerialization,
      "Actual HarmonicE2E graph should avoid serializing frameCount * 64 work in one thread. See \(kernelPath)."
    )
  }
}

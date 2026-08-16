import DGen
import XCTest

@testable import DGenLazy

final class MSEKernelProbeTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = 16_000
    DGenConfig.maxFrameCount = 4096
    DGenConfig.kernelOutputPath = nil
    LazyGraphContext.reset()
  }

  func testMSEScalarGradientIsNormalizedAcrossFrameCounts() throws {
    func gradForFrames(_ frames: Int) throws -> Float {
      LazyGraphContext.reset()
      let p = Signal.param(0.5)
      let loss = mse(p, Signal.constant(0.0))
      _ = try loss.backward(frames: frames)
      return p.grad?.data ?? .nan
    }

    let g64 = try gradForFrames(64)
    let g128 = try gradForFrames(128)
    let ratio = g128 / g64

    // MSE is a mean, so changing the number of frames must not scale a scalar parameter's gradient.
    XCTAssertEqual(g64, 1.0, accuracy: 0.01)
    XCTAssertEqual(g128, 1.0, accuracy: 0.01)
    XCTAssertEqual(ratio, 1.0, accuracy: 0.01)
  }

  func testMSEPhasorKernelDumpProbe() throws {
    let kernelPath = "/tmp/mse_phasor_probe.metal"
    try? FileManager.default.removeItem(atPath: kernelPath)

    DGenConfig.kernelOutputPath = kernelPath
    defer { DGenConfig.kernelOutputPath = nil }

    let amp = Signal.param(0.5)
    let pred = sin(Signal.phasor(233.0) * Float.pi * 2.0) * amp
    let loss = mse(pred, Signal.constant(0.2))

    let lossValues = try loss.backward(frames: 64)
    let meanLoss = lossValues.reduce(0, +) / Float(lossValues.count)
    let grad = amp.grad?.data ?? .nan

    XCTAssertTrue(meanLoss.isFinite)
    XCTAssertTrue(grad.isFinite)
    XCTAssertTrue(FileManager.default.fileExists(atPath: kernelPath))

    let source = try String(contentsOfFile: kernelPath, encoding: .utf8)
    XCTAssertTrue(source.contains("kernel void"))
  }
}

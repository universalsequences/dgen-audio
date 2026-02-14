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

  func testMSEScalarGradientScalesWithFrameCount() throws {
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

    // Current behavior: MSE is per-frame (a-b)^2 and gradients are accumulated across frames.
    XCTAssertEqual(g64, 64.0, accuracy: 0.5)
    XCTAssertEqual(g128, 128.0, accuracy: 1.0)
    XCTAssertEqual(ratio, 2.0, accuracy: 0.1)
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

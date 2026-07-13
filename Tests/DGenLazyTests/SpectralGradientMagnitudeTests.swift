import DGen
import XCTest

@testable import DGenLazy

/// Finite-difference vs autograd MAGNITUDE agreement for scalar params through
/// spectral loss. Direction-only checks (see SpectralHopGradientTests) let a
/// uniform (N*numBins)^1.5 attenuation of spectral gradients go unnoticed for a
/// long time; this suite pins the actual scale.
final class SpectralGradientMagnitudeTests: XCTestCase {
  private var frameCount = 256
  private var windowSize = 64
  private let sampleRate: Float = 2000.0

  private var savedSampleRate: Float = 0
  private var savedMaxFrameCount: Int = 0
  private var savedBackend = DGenConfig.backend
  private var savedLogMagnitudeEpsilon: Float = 0

  override func setUp() {
    super.setUp()
    savedSampleRate = DGenConfig.sampleRate
    savedMaxFrameCount = DGenConfig.maxFrameCount
    savedBackend = DGenConfig.backend
    savedLogMagnitudeEpsilon = DGenSpectralConfig.logMagnitudeEpsilon
  }

  override func tearDown() {
    DGenConfig.sampleRate = savedSampleRate
    DGenConfig.maxFrameCount = savedMaxFrameCount
    DGenConfig.backend = savedBackend
    DGenSpectralConfig.logMagnitudeEpsilon = savedLogMagnitudeEpsilon
    super.tearDown()
  }

  private func configure() {
    LazyGraphContext.reset()
    DGenConfig.backend = .metal
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = frameCount
    DGenConfig.debug = false
  }

  private enum LossKind {
    case spectralLinear
    case spectralLogL1Hop
    case spectralSmoothLogL2
    case mse
  }

  private func buildLoss(amp: Signal, kind: LossKind) -> Signal {
    let twoPi = Float.pi * 2.0
    let student = sin(Signal.phasor(Signal.constant(200.0)) * twoPi) * amp
    let teacher = sin(Signal.phasor(Signal.constant(200.0)) * twoPi) * 0.5
    switch kind {
    case .spectralLinear:
      return spectralLossFFT(student, teacher, windowSize: windowSize)
    case .spectralLogL1Hop:
      return spectralLossFFT(
        student, teacher, windowSize: windowSize,
        useLogMagnitude: true, lossMode: .l1, hop: windowSize / 4, normalize: true)
    case .spectralSmoothLogL2:
      return spectralLossFFT(
        student, teacher, windowSize: windowSize,
        useLogMagnitude: true, useSmoothLogMagnitude: true,
        lossMode: .l2, hop: windowSize / 4, normalize: true)
    case .mse:
      let diff = student - teacher
      return diff * diff
    }
  }

  private func lossSum(ampValue: Float, kind: LossKind) throws -> Float {
    configure()
    let loss = buildLoss(amp: Signal.param(ampValue), kind: kind)
    let values = try loss.backward(frames: frameCount)
    return values.reduce(0, +)
  }

  private func autogradAmp(ampValue: Float, kind: LossKind) throws -> Float {
    configure()
    let amp = Signal.param(ampValue)
    let loss = buildLoss(amp: amp, kind: kind)
    _ = try loss.backward(frames: frameCount)
    return amp.grad?.data ?? .nan
  }

  /// FD and autograd must agree in MAGNITUDE (not just direction).
  private func assertMagnitudeAgreement(
    kind: LossKind, tolerance: Float, _ label: String
  ) throws {
    let eps: Float = 1e-2
    let base: Float = 0.75
    let plus = try lossSum(ampValue: base + eps, kind: kind)
    let minus = try lossSum(ampValue: base - eps, kind: kind)
    let fd = (plus - minus) / (2 * eps)
    let auto = try autogradAmp(ampValue: base, kind: kind)
    let ratio = auto / fd
    print("[gradmag] \(label): fd=\(fd) autograd=\(auto) ratio=\(ratio)")
    XCTAssertTrue(auto.isFinite && fd.isFinite, "\(label): non-finite gradient")
    XCTAssertEqual(
      ratio, 1.0, accuracy: tolerance,
      "\(label): autograd/FD ratio should be ~1, got \(ratio) (fd=\(fd), autograd=\(auto))")
  }

  func testMSEGradMagnitude() throws {
    try assertMagnitudeAgreement(kind: .mse, tolerance: 0.01, "mse")
  }

  func testSpectralLinearGradMagnitude() throws {
    try assertMagnitudeAgreement(kind: .spectralLinear, tolerance: 0.02, "spectral-linear-l2-hop1")
  }

  func testSpectralLogL1HopGradMagnitude() throws {
    // L1 |.| kinks make FD slightly noisier; allow a looser tolerance.
    try assertMagnitudeAgreement(
      kind: .spectralLogL1Hop, tolerance: 0.05, "spectral-log-l1-hop16-norm")
  }

  func testSpectralSmoothLogL2GradMagnitude() throws {
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-3
    try assertMagnitudeAgreement(
      kind: .spectralSmoothLogL2, tolerance: 0.02, "spectral-smooth-log-l2-hop16-norm")
  }

  func testSpectralGradMagnitudeAcrossWindowSizes() throws {
    for (w, f) in [(64, 256), (128, 256), (256, 256), (64, 1024)] {
      windowSize = w
      frameCount = f
      try assertMagnitudeAgreement(kind: .spectralLinear, tolerance: 0.02, "linear w=\(w) f=\(f)")
    }
  }
}

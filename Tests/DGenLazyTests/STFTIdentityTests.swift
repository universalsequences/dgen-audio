import XCTest

@testable import DGen
@testable import DGenLazy

/// Diagnostics for the analysis/resynthesis sandwich used by spectral effects.
///
/// Before judging any soothe-style processing, the identity path must be nearly
/// transparent:
///   signal -> buffer -> window -> FFT -> IFFT -> window -> overlapAdd
///
/// The comparison allows fixed delay and gain, then asserts the remaining error
/// is small. That catches the audible "robotic" failure mode without depending
/// on exact OLA gain or latency conventions.
final class STFTIdentityTests: XCTestCase {
  private let n = 512
  private let hop = 128
  private let frames = 8192
  private let sampleRate: Float = 44_100

  override func setUp() {
    super.setUp()
    DGenConfig.backend = .c
    DGenConfig.sampleRate = sampleRate
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
    super.tearDown()
  }

  private enum FFTKind {
    case tensor
    case accelerated
  }

  private func testSignal() -> Signal {
    let twoPi = Float(2.0 * Float.pi)
    return
      sin(Signal.phasor(440.0) * twoPi) * 0.45
      + cos(Signal.phasor(973.0) * twoPi) * 0.25
      + sin(Signal.phasor(3210.0) * twoPi) * 0.15
      + cos(Signal.phasor(7000.0) * twoPi) * 0.08
  }

  private func renderBaseline() throws -> [Float] {
    LazyGraphContext.reset()
    return try testSignal().realize(frames: frames)
  }

  private func renderSTFTIdentity(_ kind: FFTKind) throws -> [Float] {
    LazyGraphContext.reset()
    let sig = testSignal()
    let win = hann(n)
    let frame = sig.buffer(size: n, hop: hop).reshape([n]) * win
    let spectrum: (re: SignalTensor, im: SignalTensor)
    switch kind {
    case .tensor:
      spectrum = signalTensorFFT(frame, N: n)
    case .accelerated:
      spectrum = acceleratedFFT(frame, N: n)
    }
    let recon: SignalTensor
    switch kind {
    case .tensor:
      recon = signalTensorIFFT(spectrum.re, spectrum.im, N: n)
    case .accelerated:
      recon = acceleratedIFFT(spectrum.re, spectrum.im, N: n)
    }
    return try (recon * win).overlapAdd(hop: hop).realize(frames: frames)
  }

  private struct Fit {
    let lag: Int
    let gain: Float
    let normalizedRMSError: Float
    let correlation: Float
  }

  private func bestLinearFit(
    reference x: [Float], output y: [Float], stableStart: Int, maxLag: Int
  ) -> Fit {
    var best = Fit(lag: 0, gain: 0, normalizedRMSError: .infinity, correlation: 0)

    for lag in -maxLag...maxLag {
      var dotXY: Float = 0
      var dotXX: Float = 0
      var dotYY: Float = 0
      var count = 0

      for i in stableStart..<min(x.count, y.count) {
        let xi = i - lag
        guard xi >= stableStart, xi < x.count else { continue }
        let xv = x[xi]
        let yv = y[i]
        dotXY += xv * yv
        dotXX += xv * xv
        dotYY += yv * yv
        count += 1
      }
      guard count > 0, dotXX > 0, dotYY > 0 else { continue }

      let gain = dotXY / dotXX
      var errSS: Float = 0
      for i in stableStart..<min(x.count, y.count) {
        let xi = i - lag
        guard xi >= stableStart, xi < x.count else { continue }
        let err = y[i] - gain * x[xi]
        errSS += err * err
      }

      let normalized = (errSS / dotYY).squareRoot()
      let corr = dotXY / (dotXX * dotYY).squareRoot()
      if normalized < best.normalizedRMSError {
        best = Fit(lag: lag, gain: gain, normalizedRMSError: normalized, correlation: corr)
      }
    }

    return best
  }

  private func assertSTFTIdentityIsTransparent(_ kind: FFTKind, file: StaticString = #filePath, line: UInt = #line)
    throws
  {
    let dry = try renderBaseline()
    let wet = try renderSTFTIdentity(kind)
    let fit = bestLinearFit(reference: dry, output: wet, stableStart: n * 4, maxLag: n * 2)
    print(
      "=== STFT identity \(kind): lag=\(fit.lag), gain=\(fit.gain), nrmse=\(fit.normalizedRMSError), corr=\(fit.correlation)"
    )

    XCTAssertLessThan(
      fit.normalizedRMSError, 0.02,
      "STFT identity path should match dry input after fixed gain/delay", file: file, line: line)
    XCTAssertGreaterThan(
      fit.correlation, 0.999,
      "STFT identity path should remain highly correlated with dry input", file: file, line: line)
  }

  func testTensorFFTWindowedSTFTIdentityMatchesInput() throws {
    try assertSTFTIdentityIsTransparent(.tensor)
  }

  func testAcceleratedFFTWindowedSTFTIdentityMatchesInput() throws {
    try assertSTFTIdentityIsTransparent(.accelerated)
  }

  func testDualAcceleratedFFTWindowedSTFTIdentityMatchesStereoInputs() throws {
    LazyGraphContext.reset()

    let twoPi = Float(2.0 * Float.pi)
    let left =
      sin(Signal.phasor(440.0) * twoPi) * 0.45
      + cos(Signal.phasor(973.0) * twoPi) * 0.2
    let right =
      sin(Signal.phasor(660.0) * twoPi) * 0.35
      + cos(Signal.phasor(1800.0) * twoPi) * 0.25

    func stftIdentity(_ signal: Signal) -> Signal {
      let win = hann(n)
      let frame = signal.buffer(size: n, hop: hop).reshape([n]) * win
      let (re, im) = acceleratedFFT(frame, N: n)
      let recon = acceleratedIFFT(re, im, N: n)
      // Hann^2 at 75% overlap sums to 1.5, so 2/3 should make identity gain ~1.
      return (recon * win).overlapAdd(hop: hop) * Signal.constant(2.0 / 3.0)
    }

    let wetL = stftIdentity(left)
    let wetR = stftIdentity(right)
    let refL = left.delay(Float(n - 1))
    let refR = right.delay(Float(n - 1))
    let err = (wetL - refL) * (wetL - refL) + (wetR - refR) * (wetR - refR)
    let energy = refL * refL + refR * refR

    let errFrames = try err.realize(frames: frames)
    let energyFrames = try energy.realize(frames: frames)
    let stableStart = n * 4
    let errRMS = rms(errFrames[stableStart..<frames])
    let refRMS = rms(energyFrames[stableStart..<frames])
    let normalized = errRMS / max(refRMS, 1e-9)
    print("=== dual stereo accelerated STFT identity: normalized error \(normalized)")

    XCTAssertLessThan(normalized, 0.01)
  }

  private func rms(_ values: ArraySlice<Float>) -> Float {
    guard !values.isEmpty else { return 0 }
    let meanSquare = values.reduce(Float(0)) { $0 + $1 } / Float(values.count)
    return meanSquare.squareRoot()
  }
}

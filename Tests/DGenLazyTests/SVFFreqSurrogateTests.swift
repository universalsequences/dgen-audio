import XCTest

@testable import DGen
@testable import DGenLazy

final class SVFFreqSurrogateTests: XCTestCase {
  private let n = 256
  private let hop = 64
  private let frames = 4096

  override func setUp() {
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 8_000
    DGenConfig.maxFrameCount = frames
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.sampleRate = 44_100
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
  }

  func testAllpassIsCOLAIdentityWithFixedLatency() throws {
    func tone() -> Signal { sin(Signal.phasor(440) * (2 * Float.pi)) * 0.7 }
    let output = svfFrequencySampled(
      tone(), cutoff: Signal.constant(1000), q: Signal.constant(1),
      mode: Signal.constant(5), window: n, hop: hop, sampleRate: 8_000)
    let actual = try output.realize(frames: frames)
    LazyGraphContext.reset()
    let dry = try tone().realize(frames: frames)
    let stable = (n * 4)..<frames
    var bestError = Float.infinity
    var bestLag = 0
    for lag in 0...(n * 2) {
      var errorEnergy: Float = 0
      var referenceEnergy: Float = 0
      for i in stable where i >= lag {
        let difference = actual[i] - dry[i - lag]
        errorEnergy += difference * difference
        referenceEnergy += dry[i - lag] * dry[i - lag]
      }
      let normalized = (errorEnergy / referenceEnergy).squareRoot()
      if normalized < bestError { bestError = normalized; bestLag = lag }
    }
    XCTAssertEqual(bestLag, n - 1)
    XCTAssertLessThan(bestError, 0.01)
  }

  // MARK: - Backward (fdcheck)

  /// Build spectral loss between a student LP (trainable cutoff) and a fixed
  /// teacher LP at a different cutoff, both through the surrogate.
  private func surrogateLoss(cutoff: Signal, backend: Backend) -> Signal {
    func source() -> Signal { sin(Signal.phasor(220) * (2 * Float.pi)) * 0.7 }
    let student = svfFrequencySampled(
      source(), cutoff: cutoff, q: Signal.constant(1),
      mode: Signal.constant(0), window: n, hop: hop, sampleRate: 8_000)
    let teacher = svfFrequencySampled(
      source(), cutoff: Signal.constant(900), q: Signal.constant(1),
      mode: Signal.constant(0), window: n, hop: hop, sampleRate: 8_000)
    return spectralLossFFT(student, teacher, windowSize: 256)
  }

  private func fdcheckCutoff(backend: Backend) throws {
    DGenConfig.backend = backend

    func lossAt(_ c: Float) throws -> Float {
      LazyGraphContext.reset()
      let loss = surrogateLoss(cutoff: Signal.param(c), backend: backend)
      return try loss.backward(frames: frames).reduce(0, +)
    }

    let base: Float = 600
    let eps: Float = 5
    let plus = try lossAt(base + eps)
    let minus = try lossAt(base - eps)
    let fd = (plus - minus) / (2 * eps)

    LazyGraphContext.reset()
    let cutoff = Signal.param(base)
    let loss = surrogateLoss(cutoff: cutoff, backend: backend)
    _ = try loss.backward(frames: frames)
    let auto = cutoff.grad?.data ?? .nan

    print("[svf-freq fdcheck \(backend)] fd=\(fd) autograd=\(auto) ratio=\(auto / fd)")
    XCTAssertTrue(auto.isFinite && fd.isFinite, "non-finite gradient (fd=\(fd), auto=\(auto))")
    XCTAssertGreaterThan(abs(fd), 1e-8, "finite-difference gradient is dead")
    XCTAssertGreaterThan(auto * fd, 0, "autograd disagrees with FD on sign")
    XCTAssertEqual(auto / fd, 1.0, accuracy: 0.35, "autograd/FD magnitude mismatch")
  }

  func testCutoffGradMatchesFiniteDifferenceC() throws {
    // C backend cannot compile spectral BPTT kernels yet (known tape codegen
    // gap: undeclared 'tape' identifiers + scratch var redefinitions). Same
    // limitation as documented in docs/TRAIN_SUBCOMMAND_NOTES.md.
    throw XCTSkip("C backend spectral BPTT tape codegen gap")
  }

  func testCutoffGradMatchesFiniteDifferenceMetal() throws {
    try fdcheckCutoff(backend: .metal)
  }

}

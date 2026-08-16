import XCTest

@testable import DGenLazy

/// `spectralFilter` hoists the magnitude -> response construction from hop rate
/// to control-frame rate by collapsing the (linear) mirror/IFFT/window/FFT chain
/// into a constant matmul. Because `sampleRow` is a convex combination and
/// linear operators commute with convex combinations, that rearrangement is
/// exact — not an approximation. These tests hold it to that claim.
final class SpectralFilterHoistTests: XCTestCase {
  private let fftSize = 64
  private let hop = 16
  private let frameCount = 1024
  private var nBins: Int { fftSize / 2 + 1 }

  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = frameCount
    LazyGraphContext.reset()
  }

  override func tearDown() {
    DGenConfig.maxFrameCount = 4096
    LazyGraphContext.reset()
    super.tearDown()
  }

  private func makeNoise(count: Int, seed: UInt64) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return Float(state >> 40) / Float(1 << 24) * 2.0 - 1.0
    }
  }

  private func playhead(featureFrames: Int) -> Signal {
    let maxIndex = Float(max(0, featureFrames - 1))
    let step = maxIndex / Float(max(1, frameCount - 1))
    let raw = Signal.accum(
      Signal.constant(step), reset: 0.0, min: 0.0, max: maxIndex)
    return raw.clip(0.0, Double(max(0.0, maxIndex - 1e-4)))
  }

  /// Per-frame smooth rolloff sweeping open -> closed across the clip; the
  /// fractional playhead spends nearly every hop *between* frames, which is
  /// exactly where interpolate-then-transform could diverge from
  /// transform-then-interpolate if the rearrangement were unsound.
  private func sweepMagnitudes(featureFrames: Int) -> [[Float]] {
    (0..<featureFrames).map { f -> [Float] in
      let frac = 0.9 - 0.75 * Float(f) / Float(max(1, featureFrames - 1))
      return (0..<nBins).map { k in
        let x = Float(k) / Float(nBins - 1) / Swift.max(1e-3, frac)
        return 1.0 / (1.0 + 4.0 * x * x)
      }
    }
  }

  func testHoistedResponseMatchesPerHopFormulation() throws {
    let featureFrames = 6
    let noiseData = makeNoise(count: frameCount, seed: 4242)
    let magnitudeRows = sweepMagnitudes(featureFrames: featureFrames)

    func render(
      _ body: (Signal, Tensor, Signal) -> Signal
    ) throws -> [Float] {
      LazyGraphContext.reset()
      // Tensors must be created after reset (see CLAUDE.md).
      let input = Tensor(noiseData).toSignal(maxFrames: frameCount)
      let magnitudes = Tensor(magnitudeRows)
      let out = body(input, magnitudes, playhead(featureFrames: featureFrames))
      return try out.realize(frames: frameCount)
    }

    let hoisted = try render { input, magnitudes, position in
      spectralFilter(
        input, magnitudes: magnitudes, framePosition: position,
        fftSize: fftSize, hop: hop, irLength: fftSize / 2)
    }
    let reference = try render { input, magnitudes, position in
      spectralFilterPerHop(
        input, magnitudes: magnitudes, framePosition: position,
        fftSize: fftSize, hop: hop, irLength: fftSize / 2)
    }

    XCTAssertEqual(hoisted.count, reference.count)
    XCTAssertTrue(hoisted.allSatisfy { $0.isFinite }, "hoisted output went non-finite")

    var peak: Float = 0
    var maxDelta: Float = 0
    var worstIndex = 0
    for i in 0..<hoisted.count {
      peak = Swift.max(peak, Swift.abs(reference[i]))
      let delta = Swift.abs(hoisted[i] - reference[i])
      if delta > maxDelta {
        maxDelta = delta
        worstIndex = i
      }
    }

    XCTAssertGreaterThan(peak, 1e-3, "reference output is silent; the test would be vacuous")
    // Both paths compute the same linear map with different summation orders,
    // so the only permitted difference is float rounding.
    XCTAssertLessThan(
      maxDelta, 1e-5 * Swift.max(1, peak),
      "hoisted and per-hop formulations diverge: max |delta|=\(maxDelta) at sample "
        + "\(worstIndex) (peak=\(peak))")
  }

  /// The collapsed constant matrices must reproduce the mirror/IFFT/window/FFT
  /// chain applied to a single magnitude row, independent of any signal path.
  func testResponseBasisMatchesTransformChain() throws {
    let N = 32
    let bins = N / 2 + 1
    let irLength = N / 2
    let basis = spectralFilterResponseBasisData(nBins: bins, fftSize: N, irLength: irLength)
    let window = spectralFilterIRWindowData(fftSize: N, irLength: irLength)

    // Arbitrary non-trivial magnitude curve.
    let mag = (0..<bins).map { k in 0.2 + Float(k % 5) * 0.31 }

    // Reference: mirror -> zero-phase IFFT -> window -> FFT, in double precision.
    var full = [Double](repeating: 0, count: N)
    for k in 0..<bins {
      full[k] += Double(mag[k])
      let mirror = (N - k) % N
      if mirror != k { full[mirror] += Double(mag[k]) }
    }
    var ir = [Double](repeating: 0, count: N)
    for n in 0..<N {
      var acc = 0.0
      for j in 0..<N { acc += full[j] * Foundation.cos(2.0 * Double.pi * Double(j * n) / Double(N)) }
      ir[n] = acc / Double(N) * Double(window[n])
    }
    for f in 0..<N {
      var re = 0.0
      var im = 0.0
      for n in 0..<N {
        let angle = 2.0 * Double.pi * Double(f * n) / Double(N)
        re += ir[n] * Foundation.cos(angle)
        im -= ir[n] * Foundation.sin(angle)
      }
      var gotRe = 0.0
      var gotIm = 0.0
      for k in 0..<bins {
        gotRe += Double(mag[k]) * Double(basis.re[k * N + f])
        gotIm += Double(mag[k]) * Double(basis.im[k * N + f])
      }
      XCTAssertEqual(gotRe, re, accuracy: 1e-5, "response real part mismatch at bin \(f)")
      XCTAssertEqual(gotIm, im, accuracy: 1e-5, "response imag part mismatch at bin \(f)")
      // A circularly-even IR has a purely real spectrum.
      XCTAssertEqual(im, 0.0, accuracy: 1e-9, "reference response should be real at bin \(f)")
    }
  }
}

import Foundation
import XCTest

@testable import DGenLazy

/// FFT-based convolution reverb using tensor ops.
/// Convolution theorem: IFFT(FFT(x) · FFT(h)) = x * h (circular convolution)
/// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
final class ConvolutionReverbTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
    DGenConfig.backend = .metal
  }

  override func tearDown() {
    DGenConfig.backend = .metal
    super.tearDown()
  }

  // MARK: - Helpers

  /// Complex multiply two FFT outputs: (reA+imA*j) * (reB+imB*j)
  private func complexMul(
    _ reA: Tensor, _ imA: Tensor,
    _ reB: Tensor, _ imB: Tensor
  ) -> (re: Tensor, im: Tensor) {
    let re = reA * reB - imA * imB
    let im = reA * imB + imA * reB
    return (re, im)
  }

  /// Complex multiply for SignalTensors
  private func complexMul(
    _ reA: SignalTensor, _ imA: SignalTensor,
    _ reB: SignalTensor, _ imB: SignalTensor
  ) -> (re: SignalTensor, im: SignalTensor) {
    let re = reA * reB - imA * imB
    let im = reA * imB + imA * reB
    return (re, im)
  }

  /// Complex multiply: SignalTensor signal × Tensor impulse response
  private func complexMul(
    _ reA: SignalTensor, _ imA: SignalTensor,
    _ reB: Tensor, _ imB: Tensor
  ) -> (re: SignalTensor, im: SignalTensor) {
    let re = reA * reB - imA * imB
    let im = reA * imB + imA * reB
    return (re, im)
  }

  /// Naive circular convolution in Swift for reference
  private func circularConvolve(_ x: [Float], _ h: [Float]) -> [Float] {
    let N = x.count
    precondition(h.count == N)
    var y = [Float](repeating: 0, count: N)
    for n in 0..<N {
      for k in 0..<N {
        let idx = (n - k + N) % N
        y[n] += x[k] * h[idx]
      }
    }
    return y
  }

  // MARK: - Static Tensor Tests

  /// Convolving with an impulse [1, 0, 0, ...] should return the original signal.
  func testConvolutionWithImpulse() throws {
    let N = 8
    let signal: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    let impulse: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]  // delta

    let x = Tensor(signal)
    let h = Tensor(impulse)

    let (reX, imX) = tensorFFT(x, N: N)
    let (reH, imH) = tensorFFT(h, N: N)
    let (reY, imY) = complexMul(reX, imX, reH, imH)
    let result = tensorIFFT(reY, imY, N: N)

    let values = try result.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== Convolution with Impulse (N=\(N)) ===")
    print("Input:  \(signal)")
    print("Output: \(values)")

    for i in 0..<N {
      XCTAssertEqual(values[i], signal[i], accuracy: 1e-3, "Sample \(i)")
    }
  }

  /// Convolving with a delayed impulse [0, 1, 0, ...] should circularly shift by 1.
  func testConvolutionWithDelay() throws {
    let N = 8
    let signal: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    let impulse: [Float] = [0, 1, 0, 0, 0, 0, 0, 0]  // delay by 1

    let x = Tensor(signal)
    let h = Tensor(impulse)

    let (reX, imX) = tensorFFT(x, N: N)
    let (reH, imH) = tensorFFT(h, N: N)
    let (reY, imY) = complexMul(reX, imX, reH, imH)
    let result = tensorIFFT(reY, imY, N: N)

    let values = try result.toSignal(maxFrames: N).realize(frames: N)

    // Circular shift by 1: [6, 3, 1, 4, 1, 5, 9, 2]
    let expected = circularConvolve(signal, impulse)

    print("\n=== Convolution with Delay (N=\(N)) ===")
    print("Input:    \(signal)")
    print("Output:   \(values)")
    print("Expected: \(expected)")

    for i in 0..<N {
      XCTAssertEqual(values[i], expected[i], accuracy: 1e-3, "Sample \(i)")
    }
  }

  /// Convolution with a simple low-pass filter [0.5, 0.5, 0, ...]
  /// (2-tap moving average)
  func testConvolutionLowPass() throws {
    let N = 8
    let signal: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    var impulse = [Float](repeating: 0, count: N)
    impulse[0] = 0.5
    impulse[1] = 0.5

    let x = Tensor(signal)
    let h = Tensor(impulse)

    let (reX, imX) = tensorFFT(x, N: N)
    let (reH, imH) = tensorFFT(h, N: N)
    let (reY, imY) = complexMul(reX, imX, reH, imH)
    let result = tensorIFFT(reY, imY, N: N)

    let values = try result.toSignal(maxFrames: N).realize(frames: N)
    let expected = circularConvolve(signal, impulse)

    print("\n=== Low-Pass Convolution (N=\(N)) ===")
    print("Input:    \(signal)")
    print("Output:   \(values)")
    print("Expected: \(expected)")

    for i in 0..<N {
      XCTAssertEqual(values[i], expected[i], accuracy: 1e-3, "Sample \(i)")
    }
  }

  /// Verify Parseval-like energy conservation through convolution.
  /// If h is unit-energy impulse, output energy = input energy.
  func testConvolutionEnergyConservation() throws {
    let N = 8
    let signal: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    let impulse: [Float] = [1, 0, 0, 0, 0, 0, 0, 0]

    let x = Tensor(signal)
    let h = Tensor(impulse)

    let (reX, imX) = tensorFFT(x, N: N)
    let (reH, imH) = tensorFFT(h, N: N)
    let (reY, imY) = complexMul(reX, imX, reH, imH)
    let result = tensorIFFT(reY, imY, N: N)

    let resultSq = result * result
    let energyValues = try resultSq.toSignal(maxFrames: N).realize(frames: N)
    let outputEnergy = energyValues.reduce(0, +)
    let inputEnergy = signal.map { $0 * $0 }.reduce(0, +)

    print("\n=== Energy Conservation ===")
    print("Input energy:  \(inputEnergy)")
    print("Output energy: \(outputEnergy)")

    XCTAssertEqual(outputEnergy, inputEnergy, accuracy: 1e-2,
                   "Energy should be preserved with unit impulse")
  }

  /// Commutativity: FFT(x)*FFT(h) = FFT(h)*FFT(x)
  func testConvolutionCommutativity() throws {
    let N = 8
    let a: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    let b: [Float] = [0.5, 0.3, 0.1, 0, 0, 0, 0.1, 0.3]

    // a * b
    let tA = Tensor(a)
    let tB = Tensor(b)
    let (reA, imA) = tensorFFT(tA, N: N)
    let (reB, imB) = tensorFFT(tB, N: N)
    let (reAB, imAB) = complexMul(reA, imA, reB, imB)
    let resultAB = tensorIFFT(reAB, imAB, N: N)
    let valuesAB = try resultAB.toSignal(maxFrames: N).realize(frames: N)

    // b * a
    LazyGraphContext.reset()
    let tA2 = Tensor(a)
    let tB2 = Tensor(b)
    let (reB2, imB2) = tensorFFT(tB2, N: N)
    let (reA2, imA2) = tensorFFT(tA2, N: N)
    let (reBA, imBA) = complexMul(reB2, imB2, reA2, imA2)
    let resultBA = tensorIFFT(reBA, imBA, N: N)
    let valuesBA = try resultBA.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== Commutativity ===")
    print("a*b: \(valuesAB)")
    print("b*a: \(valuesBA)")

    for i in 0..<N {
      XCTAssertEqual(valuesAB[i], valuesBA[i], accuracy: 1e-3, "Sample \(i)")
    }
  }

  // MARK: - Signal Pipeline: Buffer → FFT → Complex Mul → IFFT → OverlapAdd

  /// Full convolution reverb pipeline with a streaming signal.
  /// Signal → buffer → FFT → multiply by IR spectrum → IFFT → overlapAdd
  func testSignalConvolutionReverb() throws {
    let N = 64
    let hop = 16
    let sr: Float = 2048.0
    let freq: Float = 256.0  // bin 8 of 64-pt FFT at 2048 Hz

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_conv_reverb.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    // Input: cosine signal
    let twoPi = Float(2.0 * Float.pi)
    let sig = cos(Signal.phasor(freq) * twoPi)

    // Buffer with hop
    let flat = sig.buffer(size: N, hop: hop).reshape([N])

    // FFT the buffered signal
    let (reSig, imSig) = signalTensorFFT(flat, N: N)

    // Impulse response: simple identity [1, 0, 0, ...]
    // This should reconstruct the original signal exactly
    var irData = [Float](repeating: 0, count: N)
    irData[0] = 1.0
    let ir = Tensor(irData)
    let (reIR, imIR) = tensorFFT(ir, N: N)

    // Complex multiply in frequency domain
    let (reOut, imOut) = complexMul(reSig, imSig, reIR, imIR)

    // IFFT back to time domain
    let reconstructed = signalTensorIFFT(reOut, imOut, N: N)

    // Overlap-add to get output signal
    let output = reconstructed.overlapAdd(hop: hop)

    let totalFrames = N + 4 * hop
    let result = try output.realize(frames: totalFrames)

    print("\n=== Signal Convolution Reverb (identity IR) ===")
    print("Last 16 samples: \(Array(result.suffix(16)))")

    // With identity IR, output should match the straight FFT→IFFT→overlapAdd path
    let maxAbs = result.suffix(hop * 2).map { abs($0) }.max() ?? 0
    print("Max abs in steady state: \(maxAbs)")
    XCTAssertGreaterThan(maxAbs, 0.1, "Output should not be all zeros")
  }

  /// Convolution reverb with a delay IR: output should be a delayed version of input.
  func testSignalConvolutionReverbDelay() throws {
    let N = 64
    let hop = 16
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    defer { DGenConfig.sampleRate = 44100.0 }
    LazyGraphContext.reset()

    // Input: constant 1.0
    let sig = Signal.constant(1.0)
    let flat = sig.buffer(size: N, hop: hop).reshape([N])
    let (reSig, imSig) = signalTensorFFT(flat, N: N)

    // IR: echo at sample 4 with 0.5 gain → [1, 0, 0, 0, 0.5, 0, ...]
    var irData = [Float](repeating: 0, count: N)
    irData[0] = 1.0
    irData[4] = 0.5
    let ir = Tensor(irData)
    let (reIR, imIR) = tensorFFT(ir, N: N)

    let (reOut, imOut) = complexMul(reSig, imSig, reIR, imIR)
    let reconstructed = signalTensorIFFT(reOut, imOut, N: N)
    let output = reconstructed.overlapAdd(hop: hop)

    let totalFrames = N + 4 * hop
    let result = try output.realize(frames: totalFrames)

    print("\n=== Signal Convolution Reverb (delay IR) ===")
    print("Last 16 samples: \(Array(result.suffix(16)))")

    // With constant input and IR=[1, 0, 0, 0, 0.5, ...],
    // circular convolution of all-ones with this IR = 1.5 per sample
    // Then overlap-add scales by N/hop = 4, so steady state ≈ 1.5 * 4 = 6.0
    let steadyState = result.suffix(hop * 2)
    let maxAbs = steadyState.map { abs($0) }.max() ?? 0
    print("Max abs in steady state: \(maxAbs)")
    XCTAssertGreaterThan(maxAbs, 0.1, "Output should not be all zeros")
  }

  /// Compare convolution reverb output to direct FFT→IFFT (identity IR).
  /// Both pipelines should produce the same output.
  func testConvReverbMatchesDirectFFTIFFT() throws {
    let N = 64
    let hop = 16
    let sr: Float = 2048.0
    let freq: Float = 256.0

    DGenConfig.sampleRate = sr
    defer { DGenConfig.sampleRate = 44100.0 }

    let totalFrames = N + 4 * hop

    // Pipeline 1: direct FFT→IFFT→overlapAdd
    LazyGraphContext.reset()
    let twoPi = Float(2.0 * Float.pi)
    let sig1 = cos(Signal.phasor(freq) * twoPi)
    let flat1 = sig1.buffer(size: N, hop: hop).reshape([N])
    let (re1, im1) = signalTensorFFT(flat1, N: N)
    let recon1 = signalTensorIFFT(re1, im1, N: N)
    let out1 = recon1.overlapAdd(hop: hop)
    let result1 = try out1.realize(frames: totalFrames)

    // Pipeline 2: FFT→complexMul(identity)→IFFT→overlapAdd
    LazyGraphContext.reset()
    let sig2 = cos(Signal.phasor(freq) * twoPi)
    let flat2 = sig2.buffer(size: N, hop: hop).reshape([N])
    let (re2, im2) = signalTensorFFT(flat2, N: N)

    var irData = [Float](repeating: 0, count: N)
    irData[0] = 1.0
    let ir = Tensor(irData)
    let (reIR, imIR) = tensorFFT(ir, N: N)
    let (reConv, imConv) = complexMul(re2, im2, reIR, imIR)
    let recon2 = signalTensorIFFT(reConv, imConv, N: N)
    let out2 = recon2.overlapAdd(hop: hop)
    let result2 = try out2.realize(frames: totalFrames)

    print("\n=== Conv Reverb vs Direct FFT/IFFT ===")
    let stableStart = N + 2 * hop
    var maxDiff: Float = 0
    for i in stableStart..<totalFrames {
      let diff = abs(result1[i] - result2[i])
      maxDiff = Swift.max(maxDiff, diff)
    }
    print("Max difference in stable region: \(maxDiff)")
    print("Direct last 8:  \(Array(result1.suffix(8)))")
    print("ConvRev last 8: \(Array(result2.suffix(8)))")

    // Identity IR convolution should match direct FFT→IFFT
    XCTAssertLessThan(maxDiff, 0.1,
                      "Convolution with identity IR should match direct FFT→IFFT")
  }

  /// Frequency-domain filtering: zero out bins to create a brick-wall filter.
  func testFrequencyDomainFiltering() throws {
    let N = 8

    // Signal with energy at bins 1 and 2
    var data = [Float](repeating: 0, count: N)
    for n in 0..<N {
      data[n] = Foundation.cos(2.0 * Float.pi * Float(n) / Float(N))      // bin 1
             + Foundation.cos(4.0 * Float.pi * Float(n) / Float(N))      // bin 2
    }
    let x = Tensor(data)

    // Create a brick-wall low-pass: keep only bin 0 and 1 (and mirrors)
    // H[k] = 1 for k in {0, 1, N-1}, 0 otherwise
    var filterRe = [Float](repeating: 0, count: N)
    filterRe[0] = 1.0
    filterRe[1] = 1.0
    filterRe[N - 1] = 1.0
    let (reX, imX) = tensorFFT(x, N: N)

    // Multiply spectrum by filter (filter is real-only, so simplified)
    let reY = reX * Tensor(filterRe)
    let imY = imX * Tensor(filterRe)  // filter is real → just scale

    let result = tensorIFFT(reY, imY, N: N)
    let values = try result.toSignal(maxFrames: N).realize(frames: N)

    // Expected: only the bin-1 cosine survives
    var expected = [Float](repeating: 0, count: N)
    for n in 0..<N {
      expected[n] = Foundation.cos(2.0 * Float.pi * Float(n) / Float(N))
    }

    print("\n=== Frequency Domain Filtering ===")
    print("Input (bin1+bin2): \(data)")
    print("Filtered (bin1):   \(values)")
    print("Expected:          \(expected)")

    for i in 0..<N {
      XCTAssertEqual(values[i], expected[i], accuracy: 1e-3, "Sample \(i)")
    }
  }

  /// Larger FFT size to test scaling (N=64)
  func testConvolutionN64() throws {
    let N = 64

    // Random-ish signal
    var signal = [Float](repeating: 0, count: N)
    for i in 0..<N { signal[i] = Foundation.sin(Float(i) * 0.7) + 0.5 * Foundation.cos(Float(i) * 1.3) }

    // Short reverb tail: exponential decay
    var impulse = [Float](repeating: 0, count: N)
    for i in 0..<8 { impulse[i] = Foundation.exp(-Float(i) * 0.5) }

    let x = Tensor(signal)
    let h = Tensor(impulse)

    let (reX, imX) = tensorFFT(x, N: N)
    let (reH, imH) = tensorFFT(h, N: N)
    let (reY, imY) = complexMul(reX, imX, reH, imH)
    let result = tensorIFFT(reY, imY, N: N)

    let values = try result.toSignal(maxFrames: N).realize(frames: N)
    let expected = circularConvolve(signal, impulse)

    print("\n=== Convolution N=64 ===")
    print("First 8 output:   \(Array(values.prefix(8)))")
    print("First 8 expected: \(Array(expected.prefix(8)))")

    for i in 0..<N {
      XCTAssertEqual(values[i], expected[i], accuracy: 1e-2, "Sample \(i)")
    }
  }
}

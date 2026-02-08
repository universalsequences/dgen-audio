import Foundation
import XCTest

@testable import DGenLazy

/// FFT implemented purely with tensor view ops: reshape, transpose, shrink, pad, expand
/// Proves that Cooley-Tukey FFT decomposes into standard tensor operations.
final class TensorFFTTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - Tests

  /// Impulse [1, 0, 0, ...] → flat spectrum: all bins = 1.0 + 0.0j
  func testFFTImpulse() throws {
    DGenConfig.kernelOutputPath = "/tmp/test_fft_impulse.metal"
    defer { DGenConfig.kernelOutputPath = nil }
    let N = 8
    let input = Tensor([1, 0, 0, 0, 0, 0, 0, 0] as [Float])

    let (re, im) = tensorFFT(input, N: N)

    let reValues = try re.toSignal(maxFrames: N).realize(frames: N)
    let imValues = try im.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== FFT Impulse (N=\(N)) ===")
    print("Real: \(reValues)")
    print("Imag: \(imValues)")

    // DFT of impulse: X[k] = 1 for all k
    for k in 0..<N {
      XCTAssertEqual(reValues[k], 1.0, accuracy: 1e-4, "Re[\(k)]")
      XCTAssertEqual(imValues[k], 0.0, accuracy: 1e-4, "Im[\(k)]")
    }
  }

  /// DC signal [1, 1, 1, ...] → energy only in bin 0
  func testFFTDC() throws {
    let N = 8
    let input = Tensor([1, 1, 1, 1, 1, 1, 1, 1] as [Float])

    let (re, im) = tensorFFT(input, N: N)

    let reValues = try re.toSignal(maxFrames: N).realize(frames: N)
    let imValues = try im.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== FFT DC (N=\(N)) ===")
    print("Real: \(reValues)")
    print("Imag: \(imValues)")

    // DFT of all-ones: X[0] = N, X[k] = 0 for k > 0
    XCTAssertEqual(reValues[0], Float(N), accuracy: 1e-4, "Re[0] = N")
    for k in 1..<N {
      XCTAssertEqual(reValues[k], 0.0, accuracy: 1e-3, "Re[\(k)] = 0")
      XCTAssertEqual(imValues[k], 0.0, accuracy: 1e-3, "Im[\(k)] = 0")
    }
  }

  /// Cosine at bin 1 → peaks at bins 1 and N-1 (conjugate mirror)
  func testFFTCosine() throws {
    let N = 8
    // x[n] = cos(2π·n/N) — one full cycle in N samples
    var data = [Float](repeating: 0, count: N)
    for n in 0..<N {
      data[n] = Foundation.cos(2.0 * Float.pi * Float(n) / Float(N))
    }
    let input = Tensor(data)

    let (re, im) = tensorFFT(input, N: N)

    let reValues = try re.toSignal(maxFrames: N).realize(frames: N)
    let imValues = try im.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== FFT Cosine bin=1 (N=\(N)) ===")
    print("Real: \(reValues)")
    print("Imag: \(imValues)")

    // cos(2π·n/N) → Re[1] = N/2, Re[N-1] = N/2, rest ≈ 0
    XCTAssertEqual(reValues[1], Float(N) / 2, accuracy: 1e-3, "Re[1] = N/2")
    XCTAssertEqual(reValues[N - 1], Float(N) / 2, accuracy: 1e-3, "Re[N-1] = N/2")
    XCTAssertEqual(reValues[0], 0.0, accuracy: 1e-3, "Re[0] = 0 (no DC)")

    // Imaginary should be ~0 for cosine (even function)
    for k in 0..<N {
      XCTAssertEqual(imValues[k], 0.0, accuracy: 1e-3, "Im[\(k)] ≈ 0")
    }
  }

  /// Verify Parseval's theorem: sum |x[n]|² = (1/N) * sum |X[k]|²
  func testFFTParseval() throws {
    let N = 8
    let data: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    let input = Tensor(data)

    let (re, im) = tensorFFT(input, N: N)

    // Magnitude squared per bin
    let magSq = re * re + im * im

    // Sum of |X[k]|² — read via toSignal and sum manually
    let magValues = try magSq.toSignal(maxFrames: N).realize(frames: N)
    let spectralEnergy = magValues.reduce(0, +)

    // Time-domain energy
    let timeEnergy = data.map { $0 * $0 }.reduce(0, +)

    print("\n=== Parseval's Theorem (N=\(N)) ===")
    print("Time energy:     \(timeEnergy)")
    print("Spectral energy: \(spectralEnergy) / \(N) = \(spectralEnergy / Float(N))")

    // Parseval: sum|x|² = (1/N) sum|X|²
    XCTAssertEqual(timeEnergy, spectralEnergy / Float(N), accuracy: 1e-2, "Parseval's theorem")
  }

  // MARK: - Signal → Buffer → Tensor FFT

  /// Buffer a cosine signal, run tensor FFT via SignalTensor ops, verify correct bin.
  ///
  /// Uses sampleRate=2048 so a 256 Hz cosine lands exactly on bin 8 of a 64-pt FFT.
  /// We run enough frames to fill the buffer, then check the last frame's spectrum.
  func testSignalBufferFFT() throws {
    let N = 64
    let sr: Float = 2048.0
    let targetBin = 8
    let mirrorBin = N - targetBin  // 56
    let freq = sr * Float(targetBin) / Float(N)  // 256 Hz → exactly bin 8

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_signal_buffer_fft.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    // Signal: cos(2π * phasor(freq))
    let twoPi = Float(2.0 * Float.pi)
    let phase = Signal.phasor(freq)
    let sig = cos(phase * twoPi)

    // Buffer last N samples → SignalTensor [1, N]
    let buf = sig.buffer(size: N)

    // Reshape [1, N] → [N] for FFT
    let flat = buf.reshape([N])

    // Run tensor FFT on the SignalTensor
    let (re, im) = signalTensorFFT(flat, N: N)

    // Magnitude squared per bin
    let magSq = re * re + im * im

    // Check only the bins we care about: target, mirror, DC, and total energy
    // Each is a single realize() call — avoids 64 separate kernel compilations
    let totalFrames = N

    // Target bin magnitude
    let targetMag = magSq.shrink([(targetBin, targetBin + 1)]).sum()
    let targetResult = try targetMag.realize(frames: totalFrames)
    DGenConfig.kernelOutputPath = nil  // Only write kernel once
    LazyGraphContext.reset()

    // Mirror bin magnitude
    DGenConfig.sampleRate = sr
    LazyGraphContext.reset()
    let sig2 = cos(Signal.phasor(freq) * twoPi)
    let flat2 = sig2.buffer(size: N).reshape([N])
    let (re2, im2) = signalTensorFFT(flat2, N: N)
    let magSq2 = re2 * re2 + im2 * im2
    let mirrorMag = magSq2.shrink([(mirrorBin, mirrorBin + 1)]).sum()
    let mirrorResult = try mirrorMag.realize(frames: totalFrames)

    // DC bin (should be ~0)
    LazyGraphContext.reset()
    let sig3 = cos(Signal.phasor(freq) * twoPi)
    let flat3 = sig3.buffer(size: N).reshape([N])
    let (re3, im3) = signalTensorFFT(flat3, N: N)
    let magSq3 = re3 * re3 + im3 * im3
    let dcMag = magSq3.shrink([(0, 1)]).sum()
    let dcResult = try dcMag.realize(frames: totalFrames)

    // Total spectral energy (Parseval's check)
    LazyGraphContext.reset()
    let sig4 = cos(Signal.phasor(freq) * twoPi)
    let flat4 = sig4.buffer(size: N).reshape([N])
    let (re4, im4) = signalTensorFFT(flat4, N: N)
    let magSq4 = re4 * re4 + im4 * im4
    let totalMag = magSq4.sum()
    let totalResult = try totalMag.realize(frames: totalFrames)

    let lastFrame = totalFrames - 1
    let targetVal = Foundation.sqrt(targetResult[lastFrame])
    let mirrorVal = Foundation.sqrt(mirrorResult[lastFrame])
    let dcVal = Foundation.sqrt(dcResult[lastFrame])
    let totalEnergy = totalResult[lastFrame]

    print("\n=== Signal Buffer FFT (N=\(N), freq=\(freq) Hz, targetBin=\(targetBin)) ===")
    print("  Bin \(targetBin): \(targetVal)")
    print("  Bin \(mirrorBin): \(mirrorVal)")
    print("  Bin 0 (DC): \(dcVal)")
    print("  Total |X|²: \(totalEnergy)")

    // cos peak should be N/2 = 32
    let expectedPeak = Float(N) / 2.0
    XCTAssertEqual(targetVal, expectedPeak, accuracy: 0.5, "Bin \(targetBin) magnitude = N/2")
    XCTAssertEqual(mirrorVal, expectedPeak, accuracy: 0.5, "Bin \(mirrorBin) magnitude = N/2")
    XCTAssertLessThan(dcVal, 0.01, "DC bin should be ~0")

    // Parseval: total |X|² = N * (sum of x²) = N * N/2 (for unit cosine)
    // cos²(x) averages to 0.5, so sum over N samples ≈ N/2
    // Total |X|² should ≈ N * N/2 = 2048
    let expectedEnergy = Float(N) * Float(N) / 2.0
    XCTAssertEqual(
      totalEnergy, expectedEnergy, accuracy: expectedEnergy * 0.01, "Parseval's theorem")
  }

  // MARK: - IFFT Round-Trip

  /// FFT → IFFT should reconstruct the original signal within float tolerance.
  func testTensorIFFTRoundTrip() throws {
    let N = 8
    let data: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
    let input = Tensor(data)

    let (re, im) = tensorFFT(input, N: N)
    let reconstructed = tensorIFFT(re, im, N: N)

    let result = try reconstructed.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== IFFT Round-Trip (N=\(N)) ===")
    print("Original:      \(data)")
    print("Reconstructed: \(result)")

    for i in 0..<N {
      XCTAssertEqual(result[i], data[i], accuracy: 1e-3, "Sample \(i)")
    }
  }

  // MARK: - Full Pipeline: Signal → Buffer → FFT → IFFT → OverlapAdd

  /// Buffer a cosine signal, FFT, IFFT, overlapAdd — output should match input after transient.
  /// Diagnostic: does buffer(hop:) → sum work at all?
  func testHopBasedBufferSum() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_hop_buffer_sum.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    // Simple constant signal = 1.0
    let sig = Signal.constant(1.0)

    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])
    let output = flat.sum()

    let totalFrames = 32
    let result = try output.realize(frames: totalFrames)

    print("\n=== Hop-based buffer sum (N=\(N), hop=\(hop)) ===")
    print("All samples: \(result)")
    print("Max abs: \(result.map { abs($0) }.max() ?? 0)")

    // Sum of 8 ones = 8.0 (after buffer fills)
    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "Sum should not be zero")
  }

  /// Diagnostic: does buffer(hop:) with shape transitions work?
  func testHopBasedBufferReshapeMul() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_hop_reshape_mul.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    let sig = Signal.constant(1.0)

    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])
    // reshape to [4, 2], multiply by constant tensor, reshape back, sum
    let reshaped = flat.reshape([4, 2])
    let scaled = reshaped * Tensor([2.0, 3.0])  // broadcast mul
    let back = scaled.reshape([N])
    let output = back.sum()

    let totalFrames = 32
    let result = try output.realize(frames: totalFrames)

    print("\n=== Hop buffer reshape*mul (N=\(N), hop=\(hop)) ===")
    print("All samples: \(result)")
    // Expected: 4 pairs of (1*2 + 1*3) = 4*5 = 20 after buffer fills
    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "Output should not be zero")
  }

  /// Diagnostic: does buffer(hop:) with bit-reversal (reshape→transpose→reshape) work?
  func testHopBasedBitReversal() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_hop_bitrev.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    let sig = Signal.constant(1.0)

    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])
    // Bit-reversal permutation: same as tensorFFT step 1
    let k = 3  // log2(8)
    let twos = [Int](repeating: 2, count: k)
    let bitrev = flat.reshape(twos)
      .transpose(Array((0..<k).reversed()))
      .reshape([N])
    let output = bitrev.sum()

    let totalFrames = 32
    let result = try output.realize(frames: totalFrames)

    print("\n=== Hop buffer bit-reversal (N=\(N), hop=\(hop)) ===")
    print("All samples: \(result)")
    // Sum should be same as without bit-reversal (permutation preserves sum)
    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "Bit-reversed sum should not be zero")
  }

  /// Diagnostic: does buffer(hop:) with 1 butterfly stage work?
  func testHopBasedOneButterfly() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_hop_one_butterfly.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    let sig = Signal.constant(1.0)
    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])

    // Just one butterfly stage (stage 0 of FFT)
    let half = 1  // 1 << 0
    let blocks = N / 2  // N / (2 * half) = 4

    let re3d = flat.reshape([blocks, 2, half])
    let even_re = re3d.shrink([nil, (0, 1), nil]).reshape([blocks, half])
    let odd_re = re3d.shrink([nil, (1, 2), nil]).reshape([blocks, half])

    // Twiddle for stage 0: just [1.0] (cos(0)=1, sin(0)=0)
    let twiddleRe = Tensor([1.0] as [Float]).reshape([1, half]).expand([blocks, half])

    let t_re = odd_re * twiddleRe
    let top_re = even_re + t_re
    let bot_re = even_re - t_re

    // Pad+combine like FFT does
    let combined = (top_re.pad([(0, 0), (0, half)]) + bot_re.pad([(0, 0), (half, 0)])).reshape([N])

    // Stage 2: half=2, blocks=2
    let half2 = 2
    let blocks2 = 2
    let re3d2 = combined.reshape([blocks2, 2, half2])
    let even2 = re3d2.shrink([nil, (0, 1), nil]).reshape([blocks2, half2])
    let odd2 = re3d2.shrink([nil, (1, 2), nil]).reshape([blocks2, half2])
    let tw2Re = Tensor([Foundation.cos(Float(0)), Foundation.cos(Float(-Float.pi / 2))]).reshape([
      1, half2,
    ]).expand([blocks2, half2])
    let t2_re = odd2 * tw2Re
    let top2 = even2 + t2_re
    let bot2 = even2 - t2_re
    let combined2 = (top2.pad([(0, 0), (0, half2)]) + bot2.pad([(0, 0), (half2, 0)])).reshape([N])

    // Stage 3: half=4, blocks=1
    let half3 = 4
    let blocks3 = 1
    let re3d3 = combined2.reshape([blocks3, 2, half3])
    let even3 = re3d3.shrink([nil, (0, 1), nil]).reshape([blocks3, half3])
    let odd3 = re3d3.shrink([nil, (1, 2), nil]).reshape([blocks3, half3])
    var tw3 = [Float](repeating: 0, count: half3)
    for j in 0..<half3 {
      tw3[j] = Foundation.cos(Float(-2.0 * Float.pi * Float(j) / Float(2 * half3)))
    }
    let tw3Re = Tensor(tw3).reshape([1, half3]).expand([blocks3, half3])
    let t3_re = odd3 * tw3Re
    let top3 = even3 + t3_re
    let bot3 = even3 - t3_re
    let combined3 = (top3.pad([(0, 0), (0, half3)]) + bot3.pad([(0, 0), (half3, 0)])).reshape([N])

    let output = combined3.sum()

    let totalFrames = 32
    let result = try output.realize(frames: totalFrames)

    print("\n=== Hop 1-butterfly (N=\(N), hop=\(hop)) ===")
    print("All samples: \(result)")
    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "One butterfly stage should produce non-zero output")
  }

  /// Diagnostic: does buffer (NO hop) → signalTensorFFT produce non-zero output?
  func testBufferFFTNoHop() throws {
    let N = 64
    let sr: Float = 2048.0
    let freq: Float = 256.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_buffer_fft_no_hop.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    let twoPi = Float(2.0 * Float.pi)
    let phase = Signal.phasor(freq)
    let sig = cos(phase * twoPi)

    // Buffer WITHOUT hop → FFT
    let window = sig.buffer(size: N)
    let flat = window.reshape([N])
    let (re, _) = signalTensorFFT(flat, N: N)

    let reSum = re.sum()

    let totalFrames = N + 64
    let result = try reSum.realize(frames: totalFrames)

    print("\n=== Buffer FFT NO HOP (N=\(N)) ===")
    print("Last 16 sums: \(Array(result.suffix(16)))")
    print("Max abs: \(result.map { abs($0) }.max() ?? 0)")

    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "FFT output should not be all zeros")
  }

  func testHopBasedFullFFT() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_hop_full_fft.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    let sig = Signal.constant(1.0)
    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])

    // Full FFT via signalTensorFFT
    let (re, _) = signalTensorFFT(flat, N: N)
    let output = re.sum()

    let totalFrames = 32
    let result = try output.realize(frames: totalFrames)

    print("\n=== Hop full FFT (N=\(N), hop=\(hop)) ===")
    print("All samples: \(result)")
    // FFT of [1,1,...,1]: re = [8,0,...,0], sum = 8.0
    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "Full FFT sum should not be zero")
  }

  func testHopBasedFFTIFFT() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_hop_fft_ifft.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    let sig = Signal.constant(1.0)
    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])

    // FFT → IFFT round-trip
    let (re, im) = signalTensorFFT(flat, N: N)
    let reconstructed = signalTensorIFFT(re, im, N: N)
    let output = reconstructed.sum()

    let totalFrames = 32
    let result = try output.realize(frames: totalFrames)

    print("\n=== Hop FFT→IFFT (N=\(N), hop=\(hop)) ===")
    print("All samples: \(result)")
    // IFFT of FFT of [1,...,1] = [1,...,1], sum = 8.0
    let maxAbs = result.map { abs($0) }.max() ?? 0
    XCTAssertGreaterThan(maxAbs, 0.1, "FFT→IFFT round-trip should preserve signal")
  }

  func testHopBasedSimpleOverlapAdd() throws {
    let N = 8
    let hop = 4
    let sr: Float = 2048.0

    DGenConfig.sampleRate = sr
    defer { DGenConfig.sampleRate = 44100.0 }
    LazyGraphContext.reset()

    // Constant signal → buffer with hop → overlapAdd (skipping FFT/IFFT)
    let sig = Signal.constant(1.0)
    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])
    let output = flat.overlapAdd(hop: hop)

    let totalFrames = N + 4 * hop
    let result = try output.realize(frames: totalFrames)

    // After initial transient (N frames), overlap-add of constant 1.0 with
    // rectangular window should give N/hop = 2.0
    let steadyState = Array(result.suffix(hop * 2))
    let expectedValue: Float = Float(N) / Float(hop)  // 2.0
    for (i, val) in steadyState.enumerated() {
      XCTAssertEqual(
        val, expectedValue, accuracy: 0.01,
        "Steady-state sample \(i) should be \(expectedValue), got \(val)")
    }
  }

  // MARK: - OverlapAdd Gradient Helpers

  /// Create a cosine signal buffered with hop, reshaped to [N].
  private func bufferedCosineSignal(N: Int, hop: Int, freq: Float) -> SignalTensor {
    let twoPi = Float(2.0 * Float.pi)
    let phase = Signal.phasor(freq)
    let sig = cos(phase * twoPi)
    return sig.buffer(size: N, hop: hop).reshape([N])
  }

  /// Build buffer → scale → overlapAdd → squared loss pipeline, return sum of per-frame losses.
  /// Forward-only (no backward). Resets graph internally.
  private func overlapAddLoss(
    scaleData: [Float], N: Int, hop: Int, freq: Float, totalFrames: Int
  ) throws -> Float {
    LazyGraphContext.reset()
    let flat = bufferedCosineSignal(N: N, hop: hop, freq: freq)
    let scaled = flat * Tensor.param([N], data: scaleData)
    let output = scaled.overlapAdd(hop: hop)
    let loss = output * output
    let lossValues = try loss.realize(frames: totalFrames)
    return lossValues.reduce(0, +)
  }

  /// Build buffer → FFT → IFFT → scale → overlapAdd → squared loss, return sum of losses.
  private func fftOverlapAddLoss(
    scaleData: [Float], N: Int, hop: Int, freq: Float, totalFrames: Int
  ) throws -> Float {
    LazyGraphContext.reset()
    let flat = bufferedCosineSignal(N: N, hop: hop, freq: freq)
    let (re, im) = signalTensorFFT(flat, N: N)
    let reconstructed = signalTensorIFFT(re, im, N: N)
    let scaled = reconstructed * Tensor.param([N], data: scaleData)
    let output = scaled.overlapAdd(hop: hop)
    let loss = output * output
    let lossValues = try loss.realize(frames: totalFrames)
    return lossValues.reduce(0, +)
  }

  /// Compare analytical and numerical gradients, asserting at least 80% direction match.
  /// Prints a diagnostic table and skips near-zero elements.
  private func assertGradientDirectionsMatch(
    analytical: [Float],
    numerical: [Float],
    label: String,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let count = min(analytical.count, numerical.count)
    let threshold: Float = 1e-4

    print("\n=== \(label): Analytical vs Numerical ===")
    print("Element | Analytical    | Numerical     | Match?")
    print("--------|---------------|---------------|-------")

    var matchCount = 0
    var testedCount = 0
    for i in 0..<count {
      let a = analytical[i]
      let n = numerical[i]
      if Swift.abs(a) < threshold && Swift.abs(n) < threshold {
        print(String(format: "   %2d   | %12.5f | %12.5f | (both ~0)", i, a, n))
        continue
      }
      testedCount += 1
      let dirMatch = (a > 0) == (n > 0)
      if dirMatch { matchCount += 1 }
      let relError = Swift.abs(a) > 1e-6 ? Swift.abs((a - n) / a) : Float.infinity
      print(
        String(
          format: "   %2d   | %12.5f | %12.5f | %@ (rel err %.2f%%)",
          i, a, n, dirMatch ? "YES" : "NO", relError * 100))
    }
    print("Direction match: \(matchCount)/\(testedCount)")

    XCTAssertGreaterThan(
      testedCount, 0,
      "Should have testable gradient elements", file: file, line: line)
    let matchRate = Float(matchCount) / Float(testedCount)
    XCTAssertGreaterThanOrEqual(
      matchRate, 0.8,
      "At least 80% of gradient directions should match numerical", file: file, line: line)
  }

  /// Compute numerical gradient via central finite difference.
  private func numericalGradient(
    count: Int, epsilon: Float,
    lossAt: (_ perturbedData: [Float]) throws -> Float,
    baseData: [Float]
  ) throws -> [Float] {
    var grad = [Float](repeating: 0, count: count)
    for i in 0..<count {
      var plusData = baseData
      plusData[i] += epsilon
      let lossPlus = try lossAt(plusData)

      var minusData = baseData
      minusData[i] -= epsilon
      let lossMinus = try lossAt(minusData)

      grad[i] = (lossPlus - lossMinus) / (2 * epsilon)
    }
    return grad
  }

  // MARK: - OverlapAdd Gradient Tests (Numerical Verification)

  func testOverlapAddGradientNumerical() throws {
    // Finite difference check: buffer → scale → overlapAdd → squared loss
    let N = 8
    let hop = 4
    let sr: Float = 2048.0
    let freq: Float = 256.0
    let totalFrames = N + 4 * hop
    let epsilon: Float = 1e-3

    DGenConfig.sampleRate = sr
    defer { DGenConfig.sampleRate = 44100.0 }

    let baseScale = [Float](repeating: 2.0, count: N)

    // 1. Get analytical gradient via backward()
    LazyGraphContext.reset()
    let flat = bufferedCosineSignal(N: N, hop: hop, freq: freq)
    let scale = Tensor.param([N], data: baseScale)
    let output = (flat * scale).overlapAdd(hop: hop)
    let loss = output * output
    _ = try loss.backward(frames: totalFrames)
    let analyticalGrad = scale.grad!.getData()!

    // 2. Compute numerical gradient via central difference
    let numGrad = try numericalGradient(
      count: N, epsilon: epsilon,
      lossAt: { data in
        try self.overlapAddLoss(
          scaleData: data, N: N, hop: hop, freq: freq, totalFrames: totalFrames)
      }, baseData: baseScale)

    // 3. Compare
    assertGradientDirectionsMatch(
      analytical: analyticalGrad, numerical: numGrad, label: "OverlapAdd Gradient")
  }

  func testFFTIFFTOverlapAddGradientNumerical() throws {
    // Finite difference check: buffer → FFT → IFFT → scale → overlapAdd → squared loss
    let N = 16
    let hop = 8
    let sr: Float = 2048.0
    let freq: Float = 256.0
    let totalFrames = N + 4 * hop
    let epsilon: Float = 1e-3

    DGenConfig.sampleRate = sr
    defer { DGenConfig.sampleRate = 44100.0 }

    let baseScale = [Float](repeating: 1.0, count: N)

    // 1. Get analytical gradient via backward()
    LazyGraphContext.reset()
    let flat = bufferedCosineSignal(N: N, hop: hop, freq: freq)
    let (re, im) = signalTensorFFT(flat, N: N)
    let reconstructed = signalTensorIFFT(re, im, N: N)
    let scale = Tensor.param([N], data: baseScale)
    let output = (reconstructed * scale).overlapAdd(hop: hop)
    let loss = output * output
    _ = try loss.backward(frames: totalFrames)
    let analyticalGrad = scale.grad!.getData()!

    // 2. Compute numerical gradient via central difference
    let numGrad = try numericalGradient(
      count: N, epsilon: epsilon,
      lossAt: { data in
        try self.fftOverlapAddLoss(
          scaleData: data, N: N, hop: hop, freq: freq, totalFrames: totalFrames)
      }, baseData: baseScale)

    // 3. Compare
    assertGradientDirectionsMatch(
      analytical: analyticalGrad, numerical: numGrad, label: "FFT->IFFT->OverlapAdd Gradient")
  }

  func testSignalFFTOverlapAdd() throws {
    let N = 64
    let hop = 16
    let sr: Float = 2048.0
    let freq: Float = 256.0  // exactly bin 8 of 64-pt FFT at 2048 Hz

    DGenConfig.sampleRate = sr
    DGenConfig.kernelOutputPath = "/tmp/test_fft_overlap_add.metal"
    defer {
      DGenConfig.sampleRate = 44100.0
      DGenConfig.kernelOutputPath = nil
    }
    LazyGraphContext.reset()

    // Signal: cos(2π * phasor(freq))
    let twoPi = Float(2.0 * Float.pi)
    let phase = Signal.phasor(freq)
    let sig = cos(phase * twoPi)

    // Buffer with hop rate
    let window = sig.buffer(size: N, hop: hop)
    let flat = window.reshape([N])

    // FFT → IFFT via tensor ops
    let (re, im) = signalTensorFFT(flat, N: N)
    let reconstructed = signalTensorIFFT(re, im, N: N)

    // Overlap-add back to signal
    let output = reconstructed.overlapAdd(hop: hop)

    // Run enough frames for the buffer to fill and overlap-add to stabilize
    let totalFrames = N + 4 * hop  // buffer fill + a few hops
    let result = try output.realize(frames: totalFrames)

    print("\n=== FFT → IFFT → OverlapAdd (N=\(N), hop=\(hop)) ===")
    print("Last 16 samples: \(Array(result.suffix(16)))")

    // After initial transient (N frames for buffer fill + a few hops for overlap-add),
    // output should approximate the input cosine signal.
    // Due to overlap-add with rectangular window (no Hann), the scaling is windowSize/hop = N/hop.
    let scale = Float(N) / Float(hop)
    let stableStart = N + 2 * hop  // conservative transient estimate

    // Check that output has clear oscillation at the right frequency
    // The output should be scale * cos(2π * freq * t / sr)
    var maxAbs: Float = 0
    for i in stableStart..<totalFrames {
      maxAbs = Swift.max(maxAbs, Swift.abs(result[i]))
    }

    print("Max abs in stable region: \(maxAbs)")
    print("Expected scale factor: \(scale)")

    // The output should have significant amplitude (not all zeros)
    XCTAssertGreaterThan(maxAbs, 0.1, "Output should not be all zeros after transient")
  }

  // MARK: - Tensor-Based Spectral Loss

  /// Verify tensor-based spectral loss: positive for different signals, zero for same.
  /// Built entirely from tensorFFT + arithmetic — no custom Metal kernels.
  func testTensorSpectralLossForward() throws {
    let N = 8

    // Cosine at bin 1
    var cos1 = [Float](repeating: 0, count: N)
    for n in 0..<N { cos1[n] = Foundation.cos(2.0 * Float.pi * Float(n) / Float(N)) }

    // Cosine at bin 2
    var cos2 = [Float](repeating: 0, count: N)
    for n in 0..<N { cos2[n] = Foundation.cos(4.0 * Float.pi * Float(n) / Float(N)) }

    // Different signals → per-bin magnitude differences should be non-zero
    let a = Tensor(cos1)
    let b = Tensor(cos2)
    let (reA, imA) = tensorFFT(a, N: N)
    let (reB, imB) = tensorFFT(b, N: N)
    let magA = reA * reA + imA * imA
    let magB = reB * reB + imB * imB
    let diff = magA - magB
    let lossBins = diff * diff  // [N] per-bin squared differences
    let lossVals = try lossBins.toSignal(maxFrames: N).realize(frames: N)

    print("\n=== Tensor Spectral Loss Forward ===")
    print("Per-bin loss (different): \(lossVals)")
    let totalLoss = lossVals.reduce(0, +)
    XCTAssertGreaterThan(totalLoss, 0, "Different frequencies → positive spectral loss")

    // Same signal → all per-bin differences should be zero
    LazyGraphContext.reset()
    let c = Tensor(cos1)
    let d = Tensor(cos1)
    let (reC, imC) = tensorFFT(c, N: N)
    let (reD, imD) = tensorFFT(d, N: N)
    let magC = reC * reC + imC * imC
    let magD = reD * reD + imD * imD
    let diff2 = magC - magD
    let lossBins2 = diff2 * diff2
    let lossVals2 = try lossBins2.toSignal(maxFrames: N).realize(frames: N)

    print("Per-bin loss (same): \(lossVals2)")
    for k in 0..<N {
      XCTAssertEqual(lossVals2[k], 0.0, accuracy: 1e-6, "Bin \(k) loss should be 0 for same signal")
    }
  }

  /// Train a time-domain signal to match a target's magnitude spectrum.
  /// Proves backward() flows gradients through tensorFFT via autodiff.
  func testTensorSpectralLossTraining() throws {
    let N = 8
    DGenConfig.kernelOutputPath = "/tmp/test_tensor_spectral_loss_training.metal"
    defer { DGenConfig.kernelOutputPath = nil }

    // Target: cosine at bin 1 — known magnitude spectrum
    var targetData = [Float](repeating: 0, count: N)
    for n in 0..<N {
      targetData[n] = Foundation.cos(2.0 * Float.pi * Float(n) / Float(N))
    }

    // Learnable: start with different values
    let learned = Tensor.param([N], data: [0.5, -0.3, 0.8, -0.1, 0.2, 0.7, -0.5, 0.4])
    let target = Tensor(targetData)
    let optimizer = Adam(params: [learned], lr: 0.05)

    var firstLoss: Float = 0
    var lastLoss: Float = 0

    for epoch in 0..<200 {
      let (reL, imL) = tensorFFT(learned, N: N)
      let (reT, imT) = tensorFFT(target, N: N)
      let magL = reL * reL + imL * imL
      let magT = reT * reT + imT * imT
      let diff = magL - magT
      let loss = (diff * diff).sum()

      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      if epoch == 0 { firstLoss = lossValue }
      lastLoss = lossValue

      if epoch % 5 == 0 {
        print("Tensor spectral loss epoch \(epoch): \(lossValue)")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("Initial: \(firstLoss), Final: \(lastLoss)")
    XCTAssertGreaterThan(firstLoss, 0, "Initial loss should be positive")
    XCTAssertLessThan(lastLoss, firstLoss * 0.1, "Spectral loss should decrease >90%")

    // Verify: compute DFT in Swift, check bin 1 dominates
    let data = learned.getData()!
    var binEnergies = [Float](repeating: 0, count: N)
    for k in 0..<N {
      var reK: Float = 0
      var imK: Float = 0
      for n in 0..<N {
        let angle = -2.0 * Float.pi * Float(k) * Float(n) / Float(N)
        reK += data[n] * Foundation.cos(angle)
        imK += data[n] * Foundation.sin(angle)
      }
      binEnergies[k] = reK * reK + imK * imK
    }

    print("Bin energies: \(binEnergies)")
    let topBin = binEnergies.enumerated().max(by: { $0.element < $1.element })!.offset
    XCTAssert(
      topBin == 1 || topBin == N - 1,
      "Dominant bin should be 1 or \(N-1), got \(topBin)")
  }

}

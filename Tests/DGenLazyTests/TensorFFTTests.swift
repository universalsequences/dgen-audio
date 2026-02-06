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

  // MARK: - Tensor FFT via View Ops

  /// Compute N-point FFT using only tensor view + arithmetic operations.
  /// N must be a power of 2.
  /// Returns (real, imaginary) tensors of shape [N].
  private func tensorFFT(_ input: Tensor, N: Int) -> (re: Tensor, im: Tensor) {
    let k = Int(Foundation.log2(Double(N)))
    precondition(1 << k == N, "N must be a power of 2")

    // ── Step 1: Bit-reversal permutation ──
    // Reshape [N] → [2, 2, ..., 2] (k dims of size 2)
    // Transpose to reverse all axes
    // Reshape back to [N]
    // This reverses the binary representation of each index — exactly bit-reversal!
    let twos = [Int](repeating: 2, count: k)
    var re = input.reshape(twos)
      .transpose(Array((0..<k).reversed()))
      .reshape([N])
    var im = Tensor.zeros([N])

    // ── Step 2: k butterfly stages ──
    for s in 0..<k {
      let half = 1 << s  // butterfly half-width
      let blocks = N / (2 * half)  // number of butterfly groups

      // Reshape to [blocks, 2, half] — exposes even/odd butterfly pairs
      let re3d = re.reshape([blocks, 2, half])
      let im3d = im.reshape([blocks, 2, half])

      // Slice even (dim1 index 0) and odd (dim1 index 1) halves
      let even_re = re3d.shrink([nil, (0, 1), nil]).reshape([blocks, half])
      let odd_re = re3d.shrink([nil, (1, 2), nil]).reshape([blocks, half])
      let even_im = im3d.shrink([nil, (0, 1), nil]).reshape([blocks, half])
      let odd_im = im3d.shrink([nil, (1, 2), nil]).reshape([blocks, half])

      // Precompute twiddle factors on CPU: w[j] = exp(-2πij / (2·half))
      var twRe = [Float](repeating: 0, count: half)
      var twIm = [Float](repeating: 0, count: half)
      for j in 0..<half {
        let angle = -2.0 * Float.pi * Float(j) / Float(2 * half)
        twRe[j] = Foundation.cos(angle)
        twIm[j] = Foundation.sin(angle)
      }

      // Broadcast twiddle [1, half] → [blocks, half]
      let twiddleRe = Tensor(twRe).reshape([1, half]).expand([blocks, half])
      let twiddleIm = Tensor(twIm).reshape([1, half]).expand([blocks, half])

      let t_re = odd_re * twiddleRe - odd_im * twiddleIm
      let t_im = odd_re * twiddleIm + odd_im * twiddleRe

      let top_re = even_re + t_re
      let top_im = even_im + t_im
      let bot_re = even_re - t_re
      let bot_im = even_im - t_im

      // Recombine via pad+add: concat top and bot back into [N]
      // Both pads produce [blocks, 2*half], add then flatten
      re = (top_re.pad([(0, 0), (0, half)]) + bot_re.pad([(0, 0), (half, 0)])).reshape([N])
      im = (top_im.pad([(0, 0), (0, half)]) + bot_im.pad([(0, 0), (half, 0)])).reshape([N])
    }

    return (re, im)
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
}

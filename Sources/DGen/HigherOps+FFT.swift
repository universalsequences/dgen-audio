// HigherOps+FFT.swift
//
// Pure tensor-based FFT/IFFT using Cooley-Tukey via view transforms + arithmetic.
// Works for both static tensors and frame-aware signal tensors (same NodeID API).

import Foundation

extension Graph {

  /// N-point FFT using Cooley-Tukey butterfly with view transforms + arithmetic.
  ///
  /// Input must have tensor shape [N] where N is a power of 2.
  /// Returns (re, im) NodeIDs, both shape [N].
  ///
  /// Works for both static tensors and frame-aware signal tensors.
  public func tensorFFT(_ input: NodeID, N: Int) -> (re: NodeID, im: NodeID) {
    let k = Int(Foundation.log2(Double(N)))
    precondition(1 << k == N, "N must be a power of 2")

    // Bit-reversal permutation: reshape [2,2,...,2] → reverse transpose → flatten
    let twos = [Int](repeating: 2, count: k)
    var re = try! reshape(
      transpose(reshape(input, to: twos), axes: Array((0..<k).reversed())),
      to: [N])

    var im = tensor(shape: [N], data: [Float](repeating: 0.0, count: N))

    // k butterfly stages
    for s in 0..<k {
      (re, im) = butterflyStage(re: re, im: im, stage: s, N: N, inverse: false)
    }

    return (re, im)
  }

  /// N-point IFFT using Cooley-Tukey butterfly with positive twiddle angles.
  ///
  /// Takes (re, im) NodeIDs of shape [N], returns real part of shape [N] normalized by 1/N.
  /// Imaginary part is discarded (correct for real-valued signals).
  ///
  /// Works for both static tensors and frame-aware signal tensors.
  public func tensorIFFT(_ re: NodeID, _ im: NodeID, N: Int) -> NodeID {
    let k = Int(Foundation.log2(Double(N)))
    precondition(1 << k == N, "N must be a power of 2")

    // Bit-reversal permutation on both components
    let twos = [Int](repeating: 2, count: k)
    var reBR = try! reshape(
      transpose(reshape(re, to: twos), axes: Array((0..<k).reversed())),
      to: [N])
    var imBR = try! reshape(
      transpose(reshape(im, to: twos), axes: Array((0..<k).reversed())),
      to: [N])

    // k butterfly stages with positive twiddle
    for s in 0..<k {
      (reBR, imBR) = butterflyStage(re: reBR, im: imBR, stage: s, N: N, inverse: true)
    }

    // Normalize by 1/N
    return n(.mul, [reBR, n(.constant(1.0 / Float(N)))])
  }

  // MARK: - Internal

  /// Single butterfly stage shared by FFT and IFFT.
  /// `inverse: false` uses negative twiddle angles (FFT), `true` uses positive (IFFT).
  private func butterflyStage(
    re: NodeID, im: NodeID, stage s: Int, N: Int, inverse: Bool
  ) -> (re: NodeID, im: NodeID) {
    let half = 1 << s
    let blocks = N / (2 * half)

    let re3d = try! reshape(re, to: [blocks, 2, half])
    let im3d = try! reshape(im, to: [blocks, 2, half])

    // Split even/odd via shrink
    let even_re = try! reshape(shrink(re3d, ranges: [nil, (0, 1), nil]), to: [blocks, half])
    let odd_re = try! reshape(shrink(re3d, ranges: [nil, (1, 2), nil]), to: [blocks, half])
    let even_im = try! reshape(shrink(im3d, ranges: [nil, (0, 1), nil]), to: [blocks, half])
    let odd_im = try! reshape(shrink(im3d, ranges: [nil, (1, 2), nil]), to: [blocks, half])

    // Twiddle factors: w[j] = exp(±2πij / (2·half))
    let sign: Float = inverse ? 1.0 : -1.0
    var twReData = [Float](repeating: 0, count: half)
    var twImData = [Float](repeating: 0, count: half)
    for j in 0..<half {
      let angle = sign * 2.0 * Float.pi * Float(j) / Float(2 * half)
      twReData[j] = Foundation.cos(angle)
      twImData[j] = Foundation.sin(angle)
    }

    let twiddleRe = try! expandView(
      tensor(shape: [1, half], data: twReData), to: [blocks, half])
    let twiddleIm = try! expandView(
      tensor(shape: [1, half], data: twImData), to: [blocks, half])

    // Complex multiply: t = odd * twiddle
    let t_re = n(.sub, [n(.mul, [odd_re, twiddleRe]), n(.mul, [odd_im, twiddleIm])])
    let t_im = n(.add, [n(.mul, [odd_re, twiddleIm]), n(.mul, [odd_im, twiddleRe])])

    // Butterfly: top = even + t, bot = even - t
    let top_re = n(.add, [even_re, t_re])
    let top_im = n(.add, [even_im, t_im])
    let bot_re = n(.sub, [even_re, t_re])
    let bot_im = n(.sub, [even_im, t_im])

    // Concatenate via pad+add → reshape to [N]
    let outRe = try! reshape(
      n(.add, [
        pad(top_re, padding: [(0, 0), (0, half)]),
        pad(bot_re, padding: [(0, 0), (half, 0)]),
      ]),
      to: [N])
    let outIm = try! reshape(
      n(.add, [
        pad(top_im, padding: [(0, 0), (0, half)]),
        pad(bot_im, padding: [(0, 0), (half, 0)]),
      ]),
      to: [N])

    return (outRe, outIm)
  }
}

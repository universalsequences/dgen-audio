// LearnedReverb.swift
//
// Trainable impulse-response convolution — the DDSP paper's learned reverb
// (Engel et al., ICLR 2020, "Reverb" module), built strictly from the
// differentiable route: buffer -> tensorFFT -> per-bin complex multiply ->
// tensorIFFT -> overlapAdd. The Accelerate FFT path and
// `partitionedSpectralConvolve` are forward-only and must never appear here.
//
// MARK: Why this is *exact* linear convolution
//
// `buffer(size: N, hop: hop)` exposes the last N input samples each hop tick,
// oldest at index 0 (frame[i] = x[t-N+1+i]). Masking the frame to its oldest
// `hop` samples leaves consecutive ticks with adjacent, non-overlapping blocks
// that tile the input exactly: each input sample appears in exactly one kept
// block, always at the *start* of its frame. Circular convolution of a block
// of length `hop` starting at index 0 with an IR of length L stays inside the
// frame whenever hop + L - 1 <= N, so the FFT-domain product incurs no
// time-aliasing, and overlap-add of the un-windowed frames is the textbook
// overlap-add fast convolution — exact, not an STFT approximation. No analysis
// or synthesis window is wanted: windows are for COLA reconstruction of
// *overlapping* frames, and these blocks do not overlap.
//
// `overlapAdd` emits frame index n of a tick at n frames after the tick, so
// the output is the true convolution delayed by a fixed N-1 samples
// (`spectralConvolveLatency`). Callers training against a target should shift
// the target by the same amount.
//
// MARK: Why the IR spectrum is a matmul, not a tensorFFT
//
// The IR's DFT is linear in the taps, so `[L, N]` constant cos/-sin matrices
// map the tap vector to the spectrum exactly (same numbers as pad + tensorFFT,
// same trick as `spectralFilterResponse`). This is not just the usual hoist:
// backpropagating a *frame-varying* adjoint through a static tensorFFT chain
// is currently broken in the engine (adjoint magnitudes come out wrong even on
// the C backend — pinned by `StaticTensorFrameGradProbeTests`), while the
// matmul route differentiates correctly. Only the streaming signal goes
// through tensorFFT/tensorIFFT, whose gradients are proven.

import Foundation

/// Fixed output latency of `spectralConvolve`/`learnedReverb` in samples.
public func spectralConvolveLatency(fftSize N: Int) -> Int { N - 1 }

/// Memoized `[L, N]` constant DFT matrices taking a length-L tap vector to the
/// real and imaginary parts of its N-point spectrum.
private struct IRSpectrumBasisKey: Hashable {
  let irLength: Int
  let fftSize: Int
}

private var irSpectrumBasisCache: [IRSpectrumBasisKey: (re: [[Float]], im: [[Float]])] = [:]
private let irSpectrumBasisLock = NSLock()

private func irSpectrumBasisData(irLength L: Int, fftSize N: Int) -> (re: [[Float]], im: [[Float]]) {
  let key = IRSpectrumBasisKey(irLength: L, fftSize: N)
  irSpectrumBasisLock.lock()
  if let cached = irSpectrumBasisCache[key] {
    irSpectrumBasisLock.unlock()
    return cached
  }
  irSpectrumBasisLock.unlock()

  var re = [[Float]]()
  var im = [[Float]]()
  re.reserveCapacity(L)
  im.reserveCapacity(L)
  for l in 0..<L {
    var reRow = [Float](repeating: 0, count: N)
    var imRow = [Float](repeating: 0, count: N)
    for f in 0..<N {
      let angle = 2.0 * Double.pi * Double((l * f) % N) / Double(N)
      reRow[f] = Float(Foundation.cos(angle))
      imRow[f] = Float(-Foundation.sin(angle))
    }
    re.append(reRow)
    im.append(imRow)
  }

  irSpectrumBasisLock.lock()
  irSpectrumBasisCache[key] = (re, im)
  irSpectrumBasisLock.unlock()
  return (re, im)
}

/// Exact linear convolution of a streaming signal with a (typically trainable)
/// impulse response, differentiable w.r.t. `ir`.
///
/// - Parameters:
///   - input: signal to convolve.
///   - ir: `[L]` impulse-response taps, `L <= fftSize - hop + 1`.
///   - fftSize: FFT block size (power of two).
///   - hop: fresh samples consumed per FFT block (the block partition size).
/// - Returns: `(input * ir)` delayed by `spectralConvolveLatency(fftSize:)`.
public func spectralConvolve(
  _ input: Signal,
  ir: Tensor,
  fftSize N: Int,
  hop: Int
) -> Signal {
  precondition(N > 1 && N & (N - 1) == 0, "spectralConvolve fftSize must be a power of two")
  precondition(hop >= 1 && hop <= N, "spectralConvolve hop must be in 1...fftSize")
  precondition(ir.shape.count == 1, "spectralConvolve ir must be a 1-D tensor")
  let L = ir.shape[0]
  precondition(
    hop + L - 1 <= N,
    "spectralConvolve requires hop + irLength - 1 <= fftSize (got hop=\(hop), irLength=\(L), "
      + "fftSize=\(N)); a longer block or IR wraps around as time-aliased artifacts")

  // Tensors are created here, during graph construction, so they always belong
  // to the current graph (see CLAUDE.md on stale nodeIds).
  var maskData = [Float](repeating: 0, count: N)
  for i in 0..<hop { maskData[i] = 1 }
  let blockMask = Tensor(maskData)

  let frame = input.buffer(size: N, hop: hop).reshape([N])
  let block = frame * blockMask
  let (reX, imX) = signalTensorFFT(block, N: N)

  let basis = irSpectrumBasisData(irLength: L, fftSize: N)
  let ir2d = ir.reshape([1, L])
  let reH = ir2d.matmul(Tensor(basis.re)).reshape([N])
  let imH = ir2d.matmul(Tensor(basis.im)).reshape([N])

  // (reX + i·imX)(reH + i·imH)
  let reY = reX * reH - imX * imH
  let imY = reX * imH + imX * reH

  let convolved = signalTensorIFFT(reY, imY, N: N)
  return convolved.overlapAdd(hop: hop)
}

/// The DDSP paper's learned reverb: identity dry path plus a learned wet tail
/// whose first `maskedTaps` taps are forced to zero, so the model cannot use
/// the IR to re-synthesize the dry signal.
///
/// The dry path is routed through the same convolution (as a fixed unit tap at
/// index 0), so dry and wet share the operator's `spectralConvolveLatency`
/// delay and stay time-aligned with each other.
///
/// - Parameters:
///   - input: dry signal.
///   - ir: `[L]` trainable wet-tail taps, `L <= fftSize - hop + 1`.
///   - fftSize: FFT block size (power of two).
///   - hop: fresh samples consumed per FFT block.
///   - maskedTaps: leading taps of `ir` zeroed before adding the dry delta
///     (default 1; pass 0 to let tap 0 learn on top of the dry delta).
/// - Returns: `input + (input * maskedIR)`, both delayed by
///   `spectralConvolveLatency(fftSize:)`.
public func learnedReverb(
  _ input: Signal,
  ir: Tensor,
  fftSize N: Int,
  hop: Int,
  maskedTaps: Int = 1
) -> Signal {
  precondition(ir.shape.count == 1, "learnedReverb ir must be a 1-D tensor")
  let L = ir.shape[0]
  precondition(maskedTaps >= 0 && maskedTaps <= L, "learnedReverb maskedTaps must be in 0...irLength")

  var maskData = [Float](repeating: 1, count: L)
  for i in 0..<maskedTaps { maskData[i] = 0 }
  var deltaData = [Float](repeating: 0, count: L)
  deltaData[0] = 1
  let effectiveIR = ir * Tensor(maskData) + Tensor(deltaData)

  return spectralConvolve(input, ir: effectiveIR, fftSize: N, hop: hop)
}

// SpectralFilter.swift
//
// Frequency-sampled, time-varying FIR filtering of an arbitrary signal.
//
// This is the operator behind the DDSP paper's filtered-noise branch (Engel et
// al., ICLR 2020, `frequency_impulse_response`) generalized to any input
// signal, so it can equally back a Kilohearts-Filter-Table-style effect.
//
// Per control frame the caller supplies a half-spectrum magnitude curve. That
// curve is mirrored into a full real symmetric spectrum, converted to a
// zero-phase impulse response, windowed to bound its length, and transformed
// back to a complex spectrum. The resulting response multiplies the spectrum of
// each windowed input frame before overlap-add.
//
// The IR windowing is not decoration: without it, multiplying a frame's
// spectrum by an arbitrary response is circular convolution, and any IR energy
// beyond the frame wraps around as time-aliased artifacts.
//
// Everything here is differentiable w.r.t. `magnitudes` — the whole
// magnitude -> response chain is a constant matmul, and
// tensorFFT/tensorIFFT/signalTensorFFT/signalTensorIFFT are built from view +
// arithmetic primitives. (The Accelerate FFT path is forward-only and must not
// be used on a training graph.)
//
// MARK: Why the response is a matmul
//
// Every step from the magnitude curve to the filter's complex spectrum —
// mirror, IFFT, multiply by a *fixed* window, FFT — is linear in the
// magnitudes. Composing them gives two constant matrices `[nBins, fftSize]`
// mapping a magnitude row directly to the response's real and imaginary parts.
//
// That matters for two reasons:
//
//  1. It hoists the work to control-frame rate. The naive formulation
//     interpolates the magnitudes at hop rate and then transforms, running an
//     IFFT and an FFT per hop even though the curve only changes per frame (on
//     a typical clip, 512 hops vs 61 frames).
//  2. It is exact, not an approximation. `sampleRow` is a convex combination of
//     two rows, and linear operators commute with convex combinations, so
//     interpolate-then-transform and transform-then-interpolate agree up to
//     float rounding.

import Foundation

/// `[nBins, fftSize]` constant mapping a half-spectrum to the full real
/// symmetric spectrum: bin k contributes to positions k and N-k.
///
/// Exposed for callers that want to build the response chain themselves.
public func spectralFilterMirrorMatrix(nBins: Int, fftSize N: Int) -> Tensor {
  var rows = [[Float]](repeating: [Float](repeating: 0, count: N), count: nBins)
  for k in 0..<nBins {
    rows[k][k] = 1
    let mirror = (N - k) % N
    if mirror != k { rows[k][mirror] = 1 }
  }
  return Tensor(rows)
}

/// Zero-phase Hann taper centered on sample 0 with circular wraparound, which
/// is where a real symmetric spectrum puts the impulse response's peak.
/// Bounds the IR to roughly `irLength` samples.
///
/// Exposed for callers that want to build the response chain themselves.
public func spectralFilterIRWindow(fftSize N: Int, irLength: Int) -> Tensor {
  return Tensor(spectralFilterIRWindowData(fftSize: N, irLength: irLength))
}

/// Raw sample data for `spectralFilterIRWindow` (no graph node created).
public func spectralFilterIRWindowData(fftSize N: Int, irLength: Int) -> [Float] {
  let half = Swift.max(1, Swift.min(irLength, N) / 2)
  var w = [Float](repeating: 0, count: N)
  for n in 0..<N {
    let distance = Swift.min(n, N - n)
    if distance <= half {
      let x = Float(distance) / Float(half)
      w[n] = 0.5 * (1.0 + Foundation.cos(Float.pi * x))
    }
  }
  return w
}

// MARK: - Response basis

/// Flattened `[nBins, fftSize]` row-major matrices taking a half-spectrum
/// magnitude row to the windowed response's real and imaginary parts.
///
/// Building these is O(nBins · fftSize²) on the CPU, so results are memoized per
/// `(nBins, fftSize, irLength)`.
private struct ResponseBasisKey: Hashable {
  let nBins: Int
  let fftSize: Int
  let irLength: Int
}

private var responseBasisCache: [ResponseBasisKey: (re: [Float], im: [Float])] = [:]
private let responseBasisLock = NSLock()

/// Raw response-basis matrices, row-major `[nBins, fftSize]`.
public func spectralFilterResponseBasisData(
  nBins: Int, fftSize N: Int, irLength: Int
) -> (re: [Float], im: [Float]) {
  let key = ResponseBasisKey(nBins: nBins, fftSize: N, irLength: irLength)
  responseBasisLock.lock()
  if let cached = responseBasisCache[key] {
    responseBasisLock.unlock()
    return cached
  }
  responseBasisLock.unlock()

  let window = spectralFilterIRWindowData(fftSize: N, irLength: irLength).map { Double($0) }

  // cos/sin tables indexed by (f·n mod N)
  var cosTable = [Double](repeating: 0, count: N)
  var sinTable = [Double](repeating: 0, count: N)
  for i in 0..<N {
    let angle = 2.0 * Double.pi * Double(i) / Double(N)
    cosTable[i] = Foundation.cos(angle)
    sinTable[i] = Foundation.sin(angle)
  }

  var re = [Float](repeating: 0, count: nBins * N)
  var im = [Float](repeating: 0, count: nBins * N)
  var bounded = [Double](repeating: 0, count: N)

  for k in 0..<nBins {
    // Mirror + IFFT of a unit magnitude at bin k, then the IR window.
    // full[j] = δ(j-k) + δ(j-(N-k)); ifft real part = (1/N)·Σ full[j]·cos(2πjn/N).
    let mirror = (N - k) % N
    let scale = (mirror == k ? 1.0 : 2.0) / Double(N)
    for n in 0..<N {
      bounded[n] = window[n] * scale * cosTable[(k * n) % N]
    }
    // Forward DFT of the bounded IR: re = Σ x·cos, im = -Σ x·sin.
    for f in 0..<N {
      var accRe = 0.0
      var accIm = 0.0
      for n in 0..<N {
        let idx = (f * n) % N
        accRe += bounded[n] * cosTable[idx]
        accIm -= bounded[n] * sinTable[idx]
      }
      re[k * N + f] = Float(accRe)
      im[k * N + f] = Float(accIm)
    }
  }

  responseBasisLock.lock()
  responseBasisCache[key] = (re, im)
  responseBasisLock.unlock()
  return (re, im)
}

/// Frame-rate filter response: `[frames, fftSize]` spectrum of the windowed,
/// zero-phase filter built from `magnitudes`.
///
/// The response is purely **real**, and that is structural rather than
/// incidental: the IR window is symmetric under `n -> N-n` and so is the
/// zero-phase IR, so their product is circularly even and its DFT has no
/// imaginary part. `spectralFilterResponseBasisData` returns the imaginary
/// basis too, and `SpectralFilterHoistTests` pins it at zero.
///
/// Carrying only the real part is what makes the hoist a win: `sampleRow` runs
/// at audio-frame rate, not hop rate, so a second `[fftSize]` row read would
/// cost more than the per-hop transforms this saves.
///
/// - Parameters:
///   - magnitudes: `[frames, nBins]` non-negative half-spectrum magnitudes.
///   - fftSize: transform size (power of two).
///   - irLength: impulse-response budget in samples (must be < fftSize).
public func spectralFilterResponse(
  magnitudes: Tensor, fftSize N: Int, irLength: Int
) -> Tensor {
  precondition(magnitudes.shape.count == 2, "spectralFilterResponse expects [frames, nBins]")
  let nBins = magnitudes.shape[1]
  let basis = spectralFilterResponseBasisData(nBins: nBins, fftSize: N, irLength: irLength)
  // Tensors are created here, during graph construction, so they always belong
  // to the current graph (see CLAUDE.md on stale nodeIds).
  let reBasis = Tensor((0..<nBins).map { Array(basis.re[$0 * N..<($0 + 1) * N]) })
  return magnitudes.matmul(reBasis)
}

// MARK: - The op

/// Filter `input` with a time-varying, frequency-sampled response.
///
/// The response's construction is hoisted to control-frame rate (see the file
/// header); only the input's own STFT runs per hop.
///
/// - Parameters:
///   - input: signal to filter (noise, an oscillator, a bus, …).
///   - magnitudes: `[frames, nBins]` non-negative half-spectrum magnitudes,
///     `nBins == fftSize/2 + 1`.
///   - framePosition: fractional frame index driving control interpolation.
///   - fftSize: STFT size (power of two).
///   - hop: STFT hop in samples.
///   - irLength: impulse-response budget in samples (must be < fftSize).
public func spectralFilter(
  _ input: Signal,
  magnitudes: Tensor,
  framePosition: Signal,
  fftSize N: Int,
  hop: Int,
  irLength: Int
) -> Signal {
  let response = spectralFilterResponse(
    magnitudes: magnitudes, fftSize: N, irLength: irLength)
  let filter = response.sampleRow(framePosition)

  let window = hann(N)
  let frame = input.buffer(size: N, hop: hop).reshape([N]) * window
  let (inputRe, inputIm) = signalTensorFFT(frame, N: N)

  // The response is real, so the complex multiply degenerates to a scale.
  let filtered = signalTensorIFFT(inputRe * filter, inputIm * filter, N: N)
  return (filtered * window).overlapAdd(hop: hop)
}

/// Reference formulation of `spectralFilter`: interpolates the magnitude curve
/// at hop rate and builds the response with a per-hop IFFT/FFT pair.
///
/// Kept as the numerical ground truth for `spectralFilter` (see
/// `SpectralFilterHoistTests`). Prefer `spectralFilter` in real graphs — this
/// runs two extra transforms per hop for identical output.
public func spectralFilterPerHop(
  _ input: Signal,
  magnitudes: Tensor,
  framePosition: Signal,
  fftSize N: Int,
  hop: Int,
  irLength: Int
) -> Signal {
  let nBins = magnitudes.shape[magnitudes.shape.count - 1]
  let fullSpectrum = magnitudes.matmul(spectralFilterMirrorMatrix(nBins: nBins, fftSize: N))
  let magnitudeAtTime = fullSpectrum.sampleRow(framePosition)

  // Zero-phase: imaginary part is zero, carried as a SignalTensor of the right
  // shape so the IFFT sees matching operands.
  let zeroPhase = magnitudeAtTime * 0.0
  let impulseResponse = signalTensorIFFT(magnitudeAtTime, zeroPhase, N: N)
  let bounded = impulseResponse * spectralFilterIRWindow(fftSize: N, irLength: irLength)
  let (filterRe, filterIm) = signalTensorFFT(bounded, N: N)

  let window = hann(N)
  let frame = input.buffer(size: N, hop: hop).reshape([N]) * window
  let (inputRe, inputIm) = signalTensorFFT(frame, N: N)

  let (outRe, outIm) = complexMul(inputRe, inputIm, filterRe, filterIm)
  let filtered = signalTensorIFFT(outRe, outIm, N: N)
  return (filtered * window).overlapAdd(hop: hop)
}

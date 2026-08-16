import DGenLazy
import Foundation

/// Frequency-sampled, time-varying filtered noise — the DDSP paper's noise
/// branch (Engel et al., ICLR 2020, `frequency_impulse_response`).
///
/// Per frame the decoder predicts half-spectrum filter magnitudes. Those are
/// mirrored into a full real symmetric spectrum, converted to a zero-phase
/// impulse response, windowed to bound its length, transformed back, and
/// applied to a windowed noise frame before overlap-add.
///
/// The IR windowing is not decoration: without it, multiplying a frame's
/// spectrum by an arbitrary response is circular convolution, and any IR energy
/// beyond the frame wraps around as time-aliased artifacts.
///
/// Everything here is differentiable — the mirror is a constant matmul, and
/// tensorFFT/tensorIFFT are built from view + arithmetic primitives. (The
/// Accelerate FFT path is forward-only and must not be used for training.)
enum FilteredNoiseFD {

  /// `[nBins, N]` constant mapping a half-spectrum to the full real symmetric
  /// spectrum: bin k contributes to positions k and N-k.
  static func mirrorMatrix(nBins: Int, fftSize N: Int) -> Tensor {
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
  static func irWindow(fftSize N: Int, irLength: Int) -> Tensor {
    let half = Swift.max(1, Swift.min(irLength, N) / 2)
    var w = [Float](repeating: 0, count: N)
    for n in 0..<N {
      let distance = Swift.min(n, N - n)
      if distance <= half {
        let x = Float(distance) / Float(half)
        w[n] = 0.5 * (1.0 + Foundation.cos(Float.pi * x))
      }
    }
    return Tensor(w)
  }

  /// - Parameters:
  ///   - magnitudes: `[frames, nBins]` non-negative filter magnitudes.
  ///   - noise: excitation signal.
  ///   - framePosition: fractional frame index driving control interpolation.
  ///   - irLength: impulse-response budget in samples (must be < fftSize).
  static func render(
    magnitudes: Tensor,
    noise: Signal,
    framePosition: Signal,
    fftSize N: Int,
    hop: Int,
    irLength: Int
  ) -> Signal {
    let nBins = magnitudes.shape[magnitudes.shape.count - 1]
    let fullSpectrum = magnitudes.matmul(mirrorMatrix(nBins: nBins, fftSize: N))
    let magnitudeAtTime = fullSpectrum.sampleRow(framePosition)

    // Zero-phase: imaginary part is zero, carried as a SignalTensor of the
    // right shape so the IFFT sees matching operands.
    let zeroPhase = magnitudeAtTime * 0.0
    let impulseResponse = signalTensorIFFT(magnitudeAtTime, zeroPhase, N: N)
    let bounded = impulseResponse * irWindow(fftSize: N, irLength: irLength)
    let (filterRe, filterIm) = signalTensorFFT(bounded, N: N)

    let window = hann(N)
    let frame = noise.buffer(size: N, hop: hop).reshape([N]) * window
    let (noiseRe, noiseIm) = signalTensorFFT(frame, N: N)

    let (outRe, outIm) = complexMul(noiseRe, noiseIm, filterRe, filterIm)
    let filtered = signalTensorIFFT(outRe, outIm, N: N)
    return (filtered * window).overlapAdd(hop: hop)
  }
}

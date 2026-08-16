import DGenLazy
import Foundation

/// Frequency-sampled, time-varying filtered noise — the DDSP paper's noise
/// branch (Engel et al., ICLR 2020, `frequency_impulse_response`).
///
/// This is now a thin wrapper over `spectralFilter` in DGenLazy, which
/// generalizes the same operator to an arbitrary input signal and hoists the
/// magnitude -> response construction from hop rate to control-frame rate. See
/// `Sources/DGenLazy/SpectralFilter.swift` for the details.
enum FilteredNoiseFD {

  /// `[nBins, N]` constant mapping a half-spectrum to the full real symmetric
  /// spectrum: bin k contributes to positions k and N-k.
  static func mirrorMatrix(nBins: Int, fftSize N: Int) -> Tensor {
    spectralFilterMirrorMatrix(nBins: nBins, fftSize: N)
  }

  /// Zero-phase Hann taper centered on sample 0 with circular wraparound.
  /// Bounds the IR to roughly `irLength` samples.
  static func irWindow(fftSize N: Int, irLength: Int) -> Tensor {
    spectralFilterIRWindow(fftSize: N, irLength: irLength)
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
    spectralFilter(
      noise,
      magnitudes: magnitudes,
      framePosition: framePosition,
      fftSize: N,
      hop: hop,
      irLength: irLength
    )
  }
}

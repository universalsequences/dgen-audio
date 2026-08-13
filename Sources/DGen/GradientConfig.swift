import Foundation

/// Runtime flags for gradient lowering strategies.
public enum DGenGradientConfig {
  /// `true`: use fast scatter-based `peekRow` backward reduction
  /// (parallel over frame*col, atomic add into row bins).
  /// `false`: use legacy row-bin scan reduction (slower, more stable convergence in some tests).
  ///
  /// Can be enabled for experiments via `DGEN_FAST_PEEKROW_GRAD=1`.
  public static var useFastPeekRowGradReduce: Bool =
    (ProcessInfo.processInfo.environment["DGEN_FAST_PEEKROW_GRAD"] == "1")

  /// `true`: stop-gradient at every phasor frequency input. The forward
  /// graph is unchanged (pitch modulation still sounds), but no gradient
  /// flows through oscillator frequency — the swept-f0 adjoint fails
  /// fdcheck (~0.47 rel error) and a trainable phasor frequency corrupts
  /// OTHER params' gradients ~10x (SVFBPTTScratchTests TrainableDetune).
  /// Params whose only influence is through frequency simply receive zero
  /// gradient; params with mixed paths train on their non-pitch component.
  /// Set by `dgenlisp train`; defaults off for all other uses.
  public static var detachPhasorFrequency: Bool = false
}

/// Runtime knobs for spectral loss semantics.
public enum DGenSpectralConfig {
  /// Epsilon added inside log() for log-magnitude spectral losses: log(|X| + eps).
  /// With the historical default 1e-8, spectrally empty bins (magnitude ~ float noise)
  /// produce O(1) per-bin log differences between two signals that are audibly and
  /// numerically identical to ~1e-7, creating a large irreducible loss floor.
  /// Raising this to ~1e-4 (≈ -80 dBFS for peak-normalized signals) makes silent
  /// bins contribute ~0. Read at kernel codegen time; set before building the graph.
  public static var logMagnitudeEpsilon: Float = 1e-8
}

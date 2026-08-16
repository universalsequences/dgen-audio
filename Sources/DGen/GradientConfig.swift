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
  /// flows through oscillator frequency. Off by default: the historical
  /// cross-param corruption (a trainable phasor frequency corrupting a
  /// coupled-history param's gradient ~30x) was a block-formation bug —
  /// the phasor's suffix-scan tape severed the scalar BPTT recurrence —
  /// fixed by consolidateScalarBPTTBackwardBlocks. Kept as an opt-out
  /// for experiments that want pitch pinned.
  public static var detachPhasorFrequency: Bool = false

  /// Stop-gradient through stateful accumulator controls. Useful when accum
  /// is only a shared sample clock inside a tensor-lane training graph; its
  /// suffix-scan adjoint otherwise creates a scalar temporal-gradient block
  /// that cannot compose with the lane-parallel tensor-history backward yet.
  public static var detachAccumInputs: Bool = false
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

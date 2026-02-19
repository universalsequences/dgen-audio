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
}

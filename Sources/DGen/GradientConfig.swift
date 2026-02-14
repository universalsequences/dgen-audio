import Foundation

/// Runtime flags for gradient lowering strategies.
public enum DGenGradientConfig {
  /// `true`: deterministic two-phase `peek` backward (write + reduce).
  /// `false`: legacy atomic scatter `peek` backward (typically faster, non-deterministic reduction order).
  public static var useDeterministicPeekGradients: Bool = false

  /// Debug-only compatibility switch:
  /// `true`: mimic legacy behavior where `peek` backward does not return a tensor-input
  /// gradient node (fast but breaks upstream learning through tensor producers).
  /// `false`: return a sequenced tensor gradient proxy for proper upstream backprop.
  public static var dropPeekTensorInputGradient: Bool = false

  /// `true`: use fast scatter-based `peekRow` backward reduction
  /// (parallel over frame*col, atomic add into row bins).
  /// `false`: use legacy row-bin scan reduction (slower, more stable convergence in some tests).
  ///
  /// Can be enabled for experiments via `DGEN_FAST_PEEKROW_GRAD=1`.
  public static var useFastPeekRowGradReduce: Bool =
    (ProcessInfo.processInfo.environment["DGEN_FAST_PEEKROW_GRAD"] == "1")
}

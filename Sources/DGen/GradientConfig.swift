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
}

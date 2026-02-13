import Foundation

/// Runtime flags for gradient lowering strategies.
public enum DGenGradientConfig {
  /// `true`: deterministic two-phase `peek` backward (write + reduce).
  /// `false`: legacy atomic scatter `peek` backward (typically faster, non-deterministic reduction order).
  public static var useDeterministicPeekGradients: Bool = false
}

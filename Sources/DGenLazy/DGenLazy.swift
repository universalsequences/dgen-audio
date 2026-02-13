// DGenLazy - Tinygrad-inspired lazy execution frontend for DGen
//
// Provides an ergonomic API for tensor operations, audio signal processing,
// and differentiable DSP with lazy evaluation.

import DGen

// MARK: - Global Configuration

/// Global configuration for DGenLazy
public enum DGenConfig {
    /// Execution backend (.metal or .cpu)
    public static var backend: Backend = .metal

    /// Audio sample rate in Hz
    public static var sampleRate: Float = 44100.0 {
        didSet { LazyGraphContext._current?.graph.sampleRate = sampleRate }
    }

    /// Default frame count for realize() calls
    public static var defaultFrameCount: Int = 1024

    /// Maximum frame count for GPU buffer allocations (default 4096)
    /// Set this before creating graphs if you need more than 4096 frames per realize() call
    public static var maxFrameCount: Int = 4096 {
        didSet { LazyGraphContext._current?.graph.maxFrameCount = maxFrameCount }
    }

    /// Optional path to write generated Metal kernels for debugging
    /// When set, kernels will be written to this file after compilation
    public static var kernelOutputPath: String? = nil

    /// Enable debug output during compilation (prints block structure, etc.)
    public static var debug: Bool = false

    /// Enable buffer liveness analysis and reuse to reduce memory allocations
    public static var enableBufferReuse: Bool = true

    /// Toggle `peek` backward strategy.
    /// `true`: deterministic write+reduce.
    /// `false`: atomic scatter (usually faster).
    public static var useDeterministicPeekGradients: Bool = false {
        didSet { DGenGradientConfig.useDeterministicPeekGradients = useDeterministicPeekGradients }
    }
}

// MARK: - Type Aliases for Convenience

/// Shape is an array of dimensions
public typealias Shape = [Int]

// MARK: - LazyValue Protocol

/// Base protocol for all lazy values (Tensor, Signal, SignalTensor)
public protocol LazyValue {
    /// Whether this value requires gradient computation
    var requiresGrad: Bool { get }

    /// The underlying graph node ID (internal)
    var nodeId: NodeID { get }

    /// The graph this value belongs to (internal)
    var graph: LazyGraph { get }
}

// MARK: - Backend enum (re-export from DGen if not public)

// Note: Backend is already defined in DGen, we just use it directly

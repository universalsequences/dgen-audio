// SignalTensor - Frame-varying tensor
//
// A tensor whose values vary per audio frame. Created implicitly when
// mixing Signal and Tensor operations.

import DGen

// MARK: - SignalTensor

/// A frame-varying tensor
///
/// SignalTensors are created implicitly when you mix Tensor and Signal operations.
/// Each audio frame has a full tensor of values.
///
/// ```swift
/// let freqs = Tensor([440, 880, 1320])
/// let phases = Signal.phasor(freqs)  // SignalTensor with shape [3]
///
/// let t = Tensor([0, 1, 1, 0])
/// let s = Signal.phasor(440)
/// let st = t * s  // SignalTensor with shape [4]
/// ```
public class SignalTensor: LazyValue {
    // MARK: - Properties

    /// The underlying node ID in the lazy graph
    public let nodeId: NodeID

    /// The graph this signal tensor belongs to
    public let graph: LazyGraph

    /// Shape of the tensor (per frame)
    public let shape: Shape

    /// Whether this signal tensor requires gradient computation
    public let requiresGrad: Bool

    /// Gradient (populated after backward())
    public var grad: SignalTensor?

    /// Internal tensor ID (if backed by a DGen tensor)
    internal let tensorId: TensorID?

    // MARK: - Initializers

    /// Internal initializer for creating signal tensors from operations
    internal init(nodeId: NodeID, graph: LazyGraph, shape: Shape, requiresGrad: Bool = false, tensorId: TensorID? = nil) {
        self.nodeId = nodeId
        self.graph = graph
        self.shape = shape
        self.requiresGrad = requiresGrad
        self.tensorId = tensorId
        self.grad = nil
    }

    // MARK: - Factory Methods

    /// Create a phasor with multiple frequencies
    /// - Parameters:
    ///   - freqs: Tensor of frequencies in Hz
    ///   - reset: Optional reset trigger
    /// - Returns: SignalTensor where each element is a phasor at the corresponding frequency
    public static func phasor(_ freqs: Tensor, reset: Signal? = nil) -> SignalTensor {
        let graph = freqs.graph

        // For tensor input, we use deterministicPhasor which handles tensor frequencies
        // TODO: This may need to be adjusted based on how DGen handles tensor phasors
        let nodeId = graph.node(.deterministicPhasor, [freqs.nodeId])

        return SignalTensor(
            nodeId: nodeId,
            graph: graph,
            shape: freqs.shape,
            requiresGrad: freqs.requiresGrad,
            tensorId: freqs.tensorId
        )
    }
}

// MARK: - SignalTensor Properties

extension SignalTensor {
    /// Total number of elements per frame
    public var size: Int {
        shape.reduce(1, *)
    }

    /// Number of dimensions
    public var ndim: Int {
        shape.count
    }
}

// MARK: - Reduction to Signal

extension SignalTensor {
    /// Sum all elements to produce a scalar Signal
    public func sum() -> Signal {
        let nodeId = graph.node(.sum, [self.nodeId])
        return Signal(nodeId: nodeId, graph: graph, requiresGrad: requiresGrad)
    }

    /// Mean of all elements to produce a scalar Signal
    public func mean() -> Signal {
        let sumNode = graph.node(.sum, [self.nodeId])
        let countNode = graph.node(.constant(Float(size)))
        let nodeId = graph.node(.div, [sumNode, countNode])
        return Signal(nodeId: nodeId, graph: graph, requiresGrad: requiresGrad)
    }
}


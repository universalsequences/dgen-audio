// Tensor - Lazy multi-dimensional array
//
// Static tensor that doesn't vary per frame. Supports automatic differentiation
// when requiresGrad is true.

import Foundation
import DGen

// MARK: - Tensor

/// A lazy multi-dimensional array
///
/// Tensors are created lazily and only computed when `realize()` is called.
/// Use `requiresGrad: true` to enable gradient computation for training.
///
/// ```swift
/// let t = Tensor([1, 2, 3, 4])
/// let w = Tensor.randn([64, 32], requiresGrad: true)
/// let result = (t * 2 + 1).realize()
/// ```
public class Tensor: LazyValue {
    // MARK: - Properties

    /// The underlying node ID in the lazy graph
    public let nodeId: NodeID

    /// The graph this tensor belongs to
    public let graph: LazyGraph

    /// Shape of the tensor
    public let shape: Shape

    /// Whether this tensor requires gradient computation
    public let requiresGrad: Bool

    /// Gradient tensor (populated after backward())
    public var grad: Tensor?

    /// Internal tensor ID (if backed by a DGen tensor)
    internal let tensorId: TensorID?

    // MARK: - Initializers

    /// Create a tensor from a 1D array
    /// - Parameters:
    ///   - data: The tensor data
    ///   - requiresGrad: Whether to compute gradients for this tensor
    public init(_ data: [Float], requiresGrad: Bool = false) {
        let graph = LazyGraphContext.current
        let (nodeId, tensorId) = graph.createTensor(shape: [data.count], data: data)

        self.nodeId = nodeId
        self.graph = graph
        self.shape = [data.count]
        self.requiresGrad = requiresGrad
        self.tensorId = tensorId
        self.grad = nil

        // Auto-register for gradient tracking
        if requiresGrad {
            graph.registerParameter(self)
        }
    }

    /// Create a tensor from a 2D array
    /// - Parameters:
    ///   - data: The tensor data (array of arrays)
    ///   - requiresGrad: Whether to compute gradients for this tensor
    public init(_ data: [[Float]], requiresGrad: Bool = false) {
        let graph = LazyGraphContext.current
        let rows = data.count
        let cols = data.first?.count ?? 0
        let flat = data.flatMap { $0 }

        let (nodeId, tensorId) = graph.createTensor(shape: [rows, cols], data: flat)

        self.nodeId = nodeId
        self.graph = graph
        self.shape = [rows, cols]
        self.requiresGrad = requiresGrad
        self.tensorId = tensorId
        self.grad = nil

        // Auto-register for gradient tracking
        if requiresGrad {
            graph.registerParameter(self)
        }
    }

    /// Internal initializer for creating tensors from operations
    internal init(nodeId: NodeID, graph: LazyGraph, shape: Shape, requiresGrad: Bool = false, tensorId: TensorID? = nil) {
        self.nodeId = nodeId
        self.graph = graph
        self.shape = shape
        self.requiresGrad = requiresGrad
        self.tensorId = tensorId
        self.grad = nil
    }

    // MARK: - Factory Methods

    /// Create a tensor filled with zeros
    public static func zeros(_ shape: Shape, requiresGrad: Bool = false) -> Tensor {
        let data = [Float](repeating: 0.0, count: shape.reduce(1, *))
        return Tensor(shape: shape, data: data, requiresGrad: requiresGrad)
    }

    /// Create a tensor filled with ones
    public static func ones(_ shape: Shape, requiresGrad: Bool = false) -> Tensor {
        let data = [Float](repeating: 1.0, count: shape.reduce(1, *))
        return Tensor(shape: shape, data: data, requiresGrad: requiresGrad)
    }

    /// Create a tensor filled with a specific value
    public static func full(_ shape: Shape, value: Float, requiresGrad: Bool = false) -> Tensor {
        let data = [Float](repeating: value, count: shape.reduce(1, *))
        return Tensor(shape: shape, data: data, requiresGrad: requiresGrad)
    }

    /// Create a tensor with random normal values
    /// - Parameters:
    ///   - shape: The tensor shape
    ///   - mean: Mean of the distribution (default: 0)
    ///   - std: Standard deviation (default: 1)
    ///   - requiresGrad: Whether to compute gradients
    public static func randn(_ shape: Shape, mean: Float = 0.0, std: Float = 1.0, requiresGrad: Bool = false) -> Tensor {
        let count = shape.reduce(1, *)
        var data = [Float](repeating: 0, count: count)

        // Box-Muller transform for normal distribution
        // Use Foundation math functions (not our Tensor overloads)
        for i in stride(from: 0, to: count - 1, by: 2) {
            let u1 = Float.random(in: Float.ulpOfOne...1)
            let u2 = Float.random(in: 0...1)
            let r = Foundation.sqrt(-2 * Foundation.log(u1))
            let theta = 2 * Float.pi * u2
            data[i] = mean + std * r * Foundation.cos(theta)
            data[i + 1] = mean + std * r * Foundation.sin(theta)
        }
        if count % 2 == 1 {
            let u1 = Float.random(in: Float.ulpOfOne...1)
            let u2 = Float.random(in: 0...1)
            let r = Foundation.sqrt(-2 * Foundation.log(u1))
            let theta = 2 * Float.pi * u2
            data[count - 1] = mean + std * r * Foundation.cos(theta)
        }

        return Tensor(shape: shape, data: data, requiresGrad: requiresGrad)
    }

    /// Create a learnable parameter tensor (shorthand for requiresGrad: true)
    public static func param(_ shape: Shape, data: [Float]? = nil) -> Tensor {
        if let data = data {
            return Tensor(shape: shape, data: data, requiresGrad: true)
        } else {
            return randn(shape, requiresGrad: true)
        }
    }

    // MARK: - Private Helpers

    /// Internal initializer with explicit shape and data
    private init(shape: Shape, data: [Float], requiresGrad: Bool) {
        let graph = LazyGraphContext.current
        let (nodeId, tensorId) = graph.createTensor(shape: shape, data: data)

        self.nodeId = nodeId
        self.graph = graph
        self.shape = shape
        self.requiresGrad = requiresGrad
        self.tensorId = tensorId
        self.grad = nil

        // Auto-register for gradient tracking
        if requiresGrad {
            graph.registerParameter(self)
        }
    }
}

// MARK: - Tensor Properties

extension Tensor {
    /// Total number of elements
    public var size: Int {
        shape.reduce(1, *)
    }

    /// Number of dimensions
    public var ndim: Int {
        shape.count
    }

    /// Whether this is a scalar (0-dimensional or single element)
    public var isScalar: Bool {
        size == 1
    }
}


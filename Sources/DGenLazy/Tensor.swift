// Tensor - Lazy multi-dimensional array
//
// Static tensor that doesn't vary per frame. Supports automatic differentiation
// when requiresGrad is true.

import DGen
import Foundation

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

  /// The underlying node ID in the lazy graph (mutable for lazy recreation after graph clear)
  private var _nodeId: NodeID

  /// The graph this tensor belongs to
  public let graph: LazyGraph

  /// Shape of the tensor
  public let shape: Shape

  /// Whether this tensor requires gradient computation
  public let requiresGrad: Bool

  /// Gradient tensor (populated after backward())
  public var grad: Tensor?

  /// Internal tensor ID (if backed by a DGen tensor)
  internal var tensorId: TensorID?

  /// Private data storage - survives graph clears for parameters
  private var _data: [Float]?

  private var graphId: Int = -1

  /// Computed property for nodeId
  public var nodeId: NodeID {
    return _nodeId
  }

  // MARK: - Initializers

  /// Create a tensor from a 1D array
  /// - Parameters:
  ///   - data: The tensor data
  ///   - requiresGrad: Whether to compute gradients for this tensor
  public init(_ data: [Float], requiresGrad: Bool = false) {
    self.shape = [data.count]
    self._data = data  // Always store locally - survives graph clears
    let graph = LazyGraphContext.current
    self.graphId = graph.id
    let (nodeId, tensorId) = graph.createTensor(shape: shape, data: data)
    self._nodeId = nodeId
    self.graph = graph
    self.tensorId = tensorId
    self.requiresGrad = requiresGrad
    self.grad = nil

    // Auto-register for gradient tracking
    if requiresGrad {
      graph.registerParameter(self)
    }
    graph.registerTensor(self)
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
    self.shape = [rows, cols]
    self._data = flat  // Always store locally - survives graph clears

    self.graphId = graph.id
    let (nodeId, tensorId) = graph.createTensor(shape: shape, data: flat)
    self._nodeId = nodeId
    self.tensorId = tensorId

    self.graph = graph
    self.requiresGrad = requiresGrad
    self.grad = nil

    // Auto-register for gradient tracking
    if requiresGrad {
      graph.registerParameter(self)
    }
    graph.registerTensor(self)
  }

  // To be called whenever an operator's inputs contains tensors
  // Checks whether we're in a new graph and adds the tensor
  public func refresh() {
    let graph = LazyGraphContext.current
    if let data = getData() {
      if graph.id != graphId {
        graphId = graph.id
        let (nodeId, tensorId) = graph.createTensor(shape: shape, data: data)
        self._nodeId = nodeId
        self.tensorId = tensorId
      }
    }
  }

  /// Read the tensor's current data (initial or last updated values)
  /// Returns nil if tensor has no stored data
  /// For parameters, prefers local storage (survives graph clears)
  public func getData() -> [Float]? {
    // Prefer local storage for parameters (survives graph clears)
    if let data = _data { return data }
    // Fallback to graph storage for non-parameters
    guard let tensorId = self.tensorId else { return nil }
    return graph.graph.tensors[tensorId]?.data
  }

  /// Optional element-wise bounds for parameter clamping (applied after optimizer step)
  public var minBound: Float?
  public var maxBound: Float?

  /// Update the tensor's data for the next forward pass
  /// Used by optimizers to apply parameter updates
  public func updateDataLazily(_ newData: [Float]) {
    var clamped = newData
    if let lo = minBound {
      clamped = clamped.map { Swift.max($0, lo) }
    }
    if let hi = maxBound {
      clamped = clamped.map { Swift.min($0, hi) }
    }
    // Always update local storage
    _data = clamped
    // Also update graph storage if tensor exists there
    if let tensorId = self.tensorId {
      graph.graph.tensors[tensorId]?.data = clamped
    }
    graph.markDirty()  // Invalidate caches so next realize uses new data
  }

  /// Internal initializer for creating tensors from operations
  internal init(
    nodeId: NodeID, graph: LazyGraph, shape: Shape, requiresGrad: Bool = false,
    tensorId: TensorID? = nil, data: [Float]? = nil
  ) {
    self._nodeId = nodeId
    self.graph = graph
    self.shape = shape
    self.requiresGrad = requiresGrad
    self.tensorId = tensorId
    self.grad = nil
    self._data = data  // Store if provided (for parameters)
  }

  /// TensorOps conformance: view-only initializer
  public required convenience init(_view nodeId: NodeID, graph: LazyGraph, shape: Shape, requiresGrad: Bool) {
    self.init(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: requiresGrad)
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
  public static func randn(
    _ shape: Shape, mean: Float = 0.0, std: Float = 1.0, requiresGrad: Bool = false
  ) -> Tensor {
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

    self._nodeId = nodeId
    self.graph = graph
    self.shape = shape
    self.requiresGrad = requiresGrad
    self.tensorId = tensorId
    self.grad = nil
    self._data = data  // Always store locally - survives graph clears
    self.graphId = graph.id  // Track which graph generation we belong to

    // Auto-register for gradient tracking
    if requiresGrad {
      graph.registerParameter(self)
    }
    graph.registerTensor(self)  // Track for refresh after clear
  }

  /// Ensure tensor data is synced to graph (called before compilation)
  internal func syncDataToGraph() {
    guard let data = _data, let tensorId = tensorId else { return }
    // Update data if tensor exists in graph
    if graph.graph.tensors[tensorId] != nil {
      graph.graph.tensors[tensorId]?.data = data
    }
    // If tensor doesn't exist, it will be recreated by nodeId accessor
    // via tensorRef which references the tensorId
  }
}

// MARK: - View Operations (data-propagating overrides)

// Overrides TensorOps.reshape to propagate _data through the view.
// Without this, reshaped tensors lose their source data and can't
// refresh after graph clears, causing stale nodeIds that silently
// alias wrong nodes in rebuilt graphs.
//
// Only reshape is safe to propagate because it doesn't change data layout.
// transpose/expand/shrink change data order or size, so they must NOT
// propagate _data (the default TensorOps implementations are correct).

extension Tensor {

  /// Reshape with data propagation for refresh support.
  public func reshape(_ newShape: Shape) -> Tensor {
    let nodeId = try! graph.graph.reshape(self.nodeId, to: newShape)
    return Tensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad, data: _data)
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

// MARK: - Tensor History (State Buffers)

/// A tensor history buffer for state that persists across frames
/// Used for feedback loops like membrane simulation, IIR filters, etc.
public class TensorHistory {
  /// The underlying DGen history buffer
  internal let buffer: DGen.Graph.TensorHistoryBuffer

  /// The lazy graph this belongs to
  public let graph: LazyGraph

  /// Shape of the tensor
  public let shape: Shape

  /// Cell ID for the history buffer
  public var cellId: CellID { buffer.cellId }

  /// Create a tensor history buffer
  /// - Parameters:
  ///   - shape: Shape of the tensor to store
  ///   - data: Optional initial data
  public init(shape: Shape, data: [Float]? = nil) {
    let graph = LazyGraphContext.current
    self.graph = graph
    self.shape = shape
    self.buffer = graph.graph.tensorHistoryBuffer(shape: shape, data: data)
  }

  /// Read the current state from the history buffer
  /// Returns a SignalTensor (varies per frame)
  public func read() -> SignalTensor {
    let nodeId = graph.graph.tensorHistoryRead(buffer)
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: false)
  }

  /// Write new state to the history buffer
  /// - Parameter value: The new tensor value to store
  public func write(_ value: SignalTensor) {
    let _ = graph.graph.tensorHistoryWrite(buffer, value.nodeId)
  }

  /// Write new state to the history buffer (from Tensor)
  /// - Parameter value: The new tensor value to store
  public func write(_ value: Tensor) {
    let _ = graph.graph.tensorHistoryWrite(buffer, value.nodeId)
  }
}

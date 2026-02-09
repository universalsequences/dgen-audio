// LazyGraph - Implicit graph management for lazy execution
//
// This is the internal graph that accumulates operations. Users don't interact
// with it directly; it's managed implicitly by Tensor/Signal operations.

import DGen

/// Weak reference wrapper for tracking tensors/signals without preventing deallocation
internal class WeakRef<T: AnyObject> {
  weak var value: T?
  init(_ value: T) { self.value = value }
}

// MARK: - LazyGraph

/// Internal graph that accumulates lazy operations
/// Users don't create this directly; it's managed implicitly
public class LazyGraph {
  public var id: Int = 0

  /// The underlying DGen graph
  internal let graph: Graph

  /// Parameters registered for gradient computation
  internal var parameters: [any LazyValue] = []

  /// Cached compilation result (invalidated when graph changes)
  internal var compilationCache: CompilationResult?

  /// Cached runtime (invalidated when graph changes)
  internal var runtimeCache: LazyRuntime?

  /// Whether the graph has been modified since last compilation
  internal var isDirty: Bool = true

  /// Initial values for stateful cells (e.g., click cells start at 1.0)
  internal var cellInitialValues: [CellID: Float] = [:]

  /// Cache compiled kernels by graph structure hash (persists across graph clears)
  internal var compilationCacheByStructure: [String: (CompilationResult, LazyRuntime)] = [:]

  internal var tensors: [WeakRef<Tensor>] = []
  internal var signals: [WeakRef<Signal>] = []

  public init(sampleRate: Float = DGenConfig.sampleRate,
              maxFrameCount: Int = DGenConfig.maxFrameCount) {
    self.graph = Graph(sampleRate: sampleRate, maxFrameCount: maxFrameCount)
  }

  /// Mark the graph as modified (invalidates caches)
  internal func markDirty() {
    isDirty = true
    compilationCache = nil
    runtimeCache = nil
  }

  /// Create a node in the underlying graph
  internal func node(_ op: LazyOp, _ inputs: [NodeID] = []) -> NodeID {
    markDirty()
    return graph.n(op, inputs)
  }

  /// Allocate a memory cell
  internal func alloc(vectorWidth: Int = 1) -> CellID {
    return graph.alloc(vectorWidth: vectorWidth)
  }

  /// Create a tensor in the underlying graph
  /// Returns (nodeId, tensorId) - the node is the tensorRef, tensorId is for data access
  internal func createTensor(shape: Shape, data: [Float]?) -> (nodeId: NodeID, tensorId: TensorID?)
  {
    markDirty()
    let nodeId: NodeID
    if let data = data {
      nodeId = graph.tensor(shape: shape, data: data)
    } else {
      nodeId = graph.tensor(shape: shape)
    }
    // Look up the tensorId from the nodeId
    let tensorId = graph.nodeToTensor[nodeId]
    return (nodeId, tensorId)
  }

  internal func getTensor(nodeId: NodeID) -> DGen.Tensor? {
    guard let tensorId = graph.nodeToTensor[nodeId] else { return nil }
    return graph.tensors[tensorId]
  }

  public func registerTensor(_ tensor: Tensor) {
    tensors.append(WeakRef(tensor))
  }

  public func registerSignal(_ signal: Signal) {
    signals.append(WeakRef(signal))
  }

  // MARK: - Graph Structure Hashing

  /// Compute hash of current graph structure (ops and connections, not values)
  /// Used to cache compiled kernels across graph rebuilds
  /// Note: This hashes the logical structure, not specific node IDs
  internal func graphStructureHash() -> String {
    var hasher = Hasher()

    // Build a topological representation independent of node IDs
    // We hash: (operation type, number of inputs, relative input positions)
    var nodeIndex: [NodeID: Int] = [:]
    var index = 0
    for (id, _) in graph.nodes.sorted(by: { $0.key < $1.key }) {
      nodeIndex[id] = index
      index += 1
    }

    // Now hash each node's op and its inputs (as relative indices)
    for (_, node) in graph.nodes.sorted(by: { $0.key < $1.key }) {
      hasher.combine(String(describing: node.op))
      // Hash input count and their relative positions
      hasher.combine(node.inputs.count)
      for inputId in node.inputs {
        // Use relative position in sorted order, or -1 if not found
        hasher.combine(nodeIndex[inputId] ?? -1)
      }
    }

    // Include tensor shapes (keyed by their sorted position, not ID)
    var tensorIndex = 0
    for (_, tensor) in graph.tensors.sorted(by: { $0.key < $1.key }) {
      hasher.combine(tensorIndex)
      hasher.combine(tensor.shape)
      tensorIndex += 1
    }

    // Include parameter count and shapes
    hasher.combine(parameterRegistry.tensors.count)
    for tensor in parameterRegistry.tensors {
      hasher.combine(tensor.shape)
    }
    hasher.combine(parameterRegistry.signals.count)

    return String(hasher.finalize())
  }

  // MARK: - Graph Clearing

  /// Clear computation nodes while preserving parameter state and compilation cache
  /// Called after backward() to prevent node accumulation
  public func clearComputationGraph() {
    // 1. Clear all graph nodes and reset allocation counters
    // Safe because cellAllocationSizes/lazyCells/frameAwareCells are also cleared below,
    // so old cell IDs have no remaining references. Resetting prevents unbounded growth
    // of cell ID space across training epochs.
    graph.nodes.removeAll()
    graph.resetCounters()

    // 2. Clear all tensors - refresh() will recreate them
    graph.tensors.removeAll()
    graph.nodeToTensor.removeAll()
    graph.cellToTensor.removeAll()

    // 3. Clear lazy cell state (will be recreated for new tensors)
    graph.lazyCells.removeAll()
    graph.frameAwareCells.removeAll()
    graph.cellAllocationSizes.removeAll()

    // 4. Clear gradient side effects
    graph.gradientSideEffects.removeAll()
    graph.tensorGradCells.removeAll()
    graph.gradCarryCells.removeAll()
    graph.lastForwardNodeId = nil

    // 5. Clear gradient cells from registry (they'll be recreated on next backward)
    parameterRegistry.tensorGradCells.removeAll()
    parameterRegistry.signalGradCells.removeAll()

    // 6. Clear stateful cell initial values (recreated on next build)
    cellInitialValues.removeAll()

    // 7. Invalidate instance caches
    compilationCache = nil
    runtimeCache = nil
    isDirty = true

    // 7. Increment identifier to signify we have a new graph
    id += 1

    // 8. Prune dead refs and refresh surviving tensors/signals
    tensors = tensors.filter { $0.value != nil }
    for ref in tensors {
      ref.value?.refresh()
    }
    signals = signals.filter { $0.value != nil }
    for ref in signals {
      ref.value?.refresh()
    }
  }
}

// MARK: - Default Graph

/// Thread-local default graph for implicit graph management
/// Each thread gets its own graph to avoid concurrency issues
public class LazyGraphContext {
  /// The current default graph (thread-local via static)
  /// Internal so DGenConfig.didSet can propagate changes
  internal static var _current: LazyGraph?

  /// Get or create the current default graph
  public static var current: LazyGraph {
    if _current == nil {
      _current = LazyGraph()
    }
    return _current!
  }

  /// Set a new default graph
  public static func setCurrent(_ graph: LazyGraph) {
    _current = graph
  }

  /// Reset to a fresh graph
  public static func reset() {
    // Clear the parameter registry when resetting graph
    _current?.parameterRegistry.clear()
    _current = LazyGraph()
  }
}

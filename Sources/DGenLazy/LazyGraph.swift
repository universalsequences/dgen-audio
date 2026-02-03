// LazyGraph - Implicit graph management for lazy execution
//
// This is the internal graph that accumulates operations. Users don't interact
// with it directly; it's managed implicitly by Tensor/Signal operations.

import DGen

// MARK: - LazyGraph

/// Internal graph that accumulates lazy operations
/// Users don't create this directly; it's managed implicitly
public class LazyGraph {
    /// The underlying DGen graph
    internal let graph: Graph

    /// Parameters registered for gradient computation
    internal var parameters: [any LazyValue] = []

    /// Cached compilation result (invalidated when graph changes)
    internal var compilationCache: CompilationResult?

    /// Cached runtime (invalidated when graph changes)
    internal var runtimeCache: MetalCompiledKernel?

    /// Whether the graph has been modified since last compilation
    internal var isDirty: Bool = true

    public init(sampleRate: Float = DGenConfig.sampleRate) {
        self.graph = Graph(sampleRate: sampleRate)
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
    internal func createTensor(shape: Shape, data: [Float]?) -> (nodeId: NodeID, tensorId: TensorID?) {
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
}

// MARK: - Default Graph

/// Thread-local default graph for implicit graph management
/// Each thread gets its own graph to avoid concurrency issues
public class LazyGraphContext {
    /// The current default graph (thread-local via static)
    private static var _current: LazyGraph?

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

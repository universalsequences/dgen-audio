import Foundation

public struct Node {
    public let id: NodeID
    public var op: LazyOp
    public let inputs: [NodeID]
    public var temporalDependencies: [NodeID] = []
    public var shape: ValueShape? = nil

    /// Returns all dependencies (both regular inputs and temporal dependencies)
    public var allDependencies: [NodeID] {
        return inputs + temporalDependencies
    }
}

open class Graph {
    public var next = 0
    public var nodes: [NodeID: Node] = [:]
    private var nextCellId = 0
    private var nextLazyCellId = -1  // Lazy cell IDs use negative numbers to avoid collision
    public var lazyCells: Set<CellID> = []  // Track which cells are lazy (not yet allocated)
    public var nextTensorId = 0
    public var tensors: [TensorID: Tensor] = [:]
    public var nodeToTensor: [NodeID: TensorID] = [:]
    public var cellToTensor: [CellID: TensorID] = [:]  // Maps cell IDs to their associated tensor

    /// Track allocation sizes for memory cells (especially large buffers like spectral scratch)
    public var cellAllocationSizes: [CellID: Int] = [:]

    /// Tracks hop-based update rate for nodes (hopSize, counterNodeId)
    /// Used for FFT/IFFT nodes and operations that inherit hop-based temporality
    public var nodeHopRate: [NodeID: (Int, NodeID)] = [:]

    /// Tracks buffer position dependencies for slidingWindow circular buffer mode.
    /// Maps bufferView result nodes to their writePos accum nodes.
    /// Propagated via temporalDependencies so defineGlobal/loadGlobal wiring works.
    public var nodePositionDep: [NodeID: NodeID] = [:]

    /// Sample rate for audio processing (default 44100 Hz)
    public var sampleRate: Float = 44100.0

    /// Maximum frame count for scratch buffer allocations (default 4096)
    /// Set this to match your actual frame count to reduce memory usage and computation
    public var maxFrameCount: Int = 4096

    /// Mapping from history cell IDs to gradient carry cell IDs.
    /// Used for temporal gradient flow through historyRead/historyWrite.
    public var gradCarryCells: [CellID: CellID] = [:]
    public var tensorGradCells: [NodeID: CellID] = [:]

    /// Gradient carry cells that mirror a tensor-registered history cell (width W).
    /// memoryRead/memoryWrite of these cells use per-element tensor addressing;
    /// all vector-width BPTT branches gate on this set so scalar graphs are untouched.
    public var tensorGradCarryCells: Set<CellID> = []

    /// Track frame-aware tensor allocations: cellId -> (tensorSize, frameCount)
    /// Used for tensors with outbound dependencies that need tensorSize * frameCount cells.
    public var frameAwareCells: [CellID: (tensorSize: Int, frameCount: Int)] = [:]

    /// Hop size for frame-aware cells produced by hop-based nodes. These cells
    /// are only written/read on hop boundaries, so their storage holds one slot
    /// per hop (frameAwareCells.frameCount is the slot count) and addressing
    /// divides the frame index by this hop. Cells absent here use hop = 1.
    public var frameAwareCellHops: [CellID: Int] = [:]

    /// Hop-sliced cells whose frame-rate reads must **zero-fill** between hop
    /// ticks instead of holding the tick's value.
    ///
    /// A forward hop-sliced tensor is a held signal: `x(frame) = slot[frame/hop]`
    /// for every frame, so holding is what the reader wants. An *adjoint* stored
    /// the same way is not. It carries `dLoss/dy(tick)`, and the frames between
    /// ticks are exactly the ones whose value the forward discarded — their
    /// adjoint is identically zero. Holding instead replays the tick's adjoint
    /// `hop` times, so anything downstream that integrates an adjoint over audio
    /// frames (`sampleGradWrite`, `peekGradWrite`, `selectRowGradWrite`) sums
    /// `hop` spurious copies, each weighted by the wrong control interpolation.
    ///
    /// Populated for gradient-subgraph cells only (see `lastForwardNodeId`).
    public var frameAwareCellScatter: Set<CellID> = []

    /// Cells that persist data across frame iterations (circular buffers, ring buffers, etc.)
    /// These must not be shared with other cells during buffer reuse optimization.
    public var persistentCells: Set<CellID> = []

    /// Host-addressable parameter cells.
    /// These must keep unique physical slots even when their compile-time lifetimes do not overlap.
    public var parameterCells: Set<CellID> = []

    /// Nodes that should have their tensor results materialized in memory (for realize())
    public var materializeNodes: Set<NodeID> = []

    /// Side-effect nodes created during backward pass (e.g., gradient carry writes)
    /// These need to be chained with gradient outputs to ensure they execute.
    public var gradientSideEffects: [NodeID] = []

    /// Last node ID before gradient nodes were added.
    /// Used to separate forward and gradient node ordering during compilation.
    public var lastForwardNodeId: NodeID?

    /// Conv2d nodes the Conv2DPass has annotated for SIMD-unrolled emission.
    public var simdOptimizedConv2Ds: Set<NodeID> = []

    /// Mask tensor cell associated with each SIMD-optimized conv2d — 12 floats laid
    /// out as three contiguous 4-lane masks: left-edge, full, right-edge.
    public var conv2dMaskCells: [NodeID: CellID] = [:]

    public init() {}

    public init(sampleRate: Float) {
        self.sampleRate = sampleRate
    }

    public init(maxFrameCount: Int) {
        self.maxFrameCount = maxFrameCount
    }

    public init(sampleRate: Float, maxFrameCount: Int) {
        self.sampleRate = sampleRate
        self.maxFrameCount = maxFrameCount
    }

    /// Reset node and cell counters for graph reuse
    /// Call this when clearing the graph to start fresh with IDs
    public func resetCounters() {
        next = 0
        nextCellId = 0
        nextLazyCellId = -1
        nextTensorId = 0
    }

    /// Returns the total number of allocated memory cells
    public var totalMemoryCells: Int { nextCellId }

    @discardableResult public func n(_ op: LazyOp, _ ins: NodeID...) -> NodeID {
        return n(op, ins)
    }

    @discardableResult public func n(_ op: LazyOp, _ ins: [NodeID], shape: ValueShape? = nil)
        -> NodeID
    {
        let id = next
        next += 1
        nodes[id] = Node(id: id, op: op, inputs: ins)

        if case .param(let cellId) = op {
            parameterCells.insert(cellId)
        }

        // If shape is explicitly provided, use it. Otherwise, infer from inputs.
        if let explicitShape = shape {
            nodes[id]?.shape = explicitShape
        } else {
            // Gather input shapes
            let inputShapes = ins.compactMap { nodes[$0]?.shape }
            // Try to infer shape - fall back to .scalar if inference fails
            let inferredShape =
                (try? inferShape(op: op, inputs: inputShapes, graph: self)) ?? .scalar
            nodes[id]?.shape = inferredShape
        }

        // Handle seq operator: find root dependencies of B and make them depend on A
        if case .seq = op, ins.count >= 2 {
            let a = ins[0]  // First input (e.g., writeOp)
            let b = ins[1]  // Second input (e.g., interpolated)

            // For seq(a, b), find all nodes in B's dependency tree that should wait for A
            // We traverse B's dependencies and find memory operations that should depend on A
            var visited = Set<NodeID>()
            var queue = [b]

            while !queue.isEmpty {
                let currentId = queue.removeFirst()
                if visited.contains(currentId) { continue }
                visited.insert(currentId)

                guard let node = nodes[currentId] else { continue }

                // Check if this node is a memory operation that should depend on A
                switch node.op {
                case .memoryRead(_), .historyRead(_):
                    // Memory reads should depend on the write
                    if var currentNode = nodes[currentId] {
                        currentNode.temporalDependencies.append(a)
                        nodes[currentId] = currentNode
                    }
                default:
                    // For other nodes, continue traversing
                    queue.append(contentsOf: node.inputs)
                }
            }
        }

        return id
    }

    /// Allocate a new cell ID for memory-based operations like phasor, latch, etc.
    /// For vector operations, this will allocate consecutive slots
    public func alloc(vectorWidth: Int = 1) -> CellID {
        let cellId = nextCellId
        nextCellId += vectorWidth
        // Track allocation size for later memory layout calculations
        cellAllocationSizes[cellId] = vectorWidth
        return cellId
    }

    /// Allocate a single cell (backward compatibility)
    public func alloc() -> CellID {
        return alloc(vectorWidth: 1)
    }

    /// Reserve a lazy cell ID (placeholder, no memory allocated yet).
    /// Used for tensor outputs that will be allocated later once we know temporality.
    /// Lazy cell IDs use negative numbers to avoid collision with real allocations.
    public func reserveLazyCellId() -> CellID {
        let cellId = nextLazyCellId
        nextLazyCellId -= 1
        lazyCells.insert(cellId)
        return cellId
    }

    /// Allocate real memory for a lazy cell, returning the new real cell ID.
    /// The tensor should be updated to use the returned cell ID.
    public func allocateLazyCell(_ lazyCellId: CellID, vectorWidth: Int) -> CellID {
        lazyCells.remove(lazyCellId)
        return alloc(vectorWidth: vectorWidth)
    }

    /// Allocate frame-aware tensor storage: tensorSize * frameCount cells.
    /// Used for tensors with outbound dependencies in frame-based blocks.
    /// Memory layout: memory[cellId + frameIdx * tensorSize + elemIdx]
    public func allocFrameAware(tensorSize: Int, frameCount: Int) -> CellID {
        let totalSize = tensorSize * frameCount
        let cellId = alloc(vectorWidth: totalSize)
        frameAwareCells[cellId] = (tensorSize: tensorSize, frameCount: frameCount)
        return cellId
    }

    public func seq(a: NodeID, b: NodeID) -> NodeID {
        return n(.seq, a, b)
    }
}

extension Lazy {
    public var varId: VarID? {
        switch self {
        case .variable(let id, _):
            return id
        default:
            return nil
        }
    }
}

extension Op {
    public var operands: [Lazy] {
        switch self {
        case .add(let a, let b):
            return [a, b]
        case .mul(let a, let b):
            return [a, b]
        case .sub(let a, let b):
            return [a, b]
        case .div(let a, let b):
            return [a, b]
        case .abs(let a):
            return [a]
        case .sign(let a):
            return [a]
        case .gt(let a, let b):
            return [a, b]
        case .lt(let a, let b):
            return [a, b]
        case .store(_, let b):
            return [b]
        case .load(_):
            return []
        case .beginIf(let a):
            return [a]
        case .mutate(let a, let b):
            return [a, b]
        default:
            return []
        }
    }
}

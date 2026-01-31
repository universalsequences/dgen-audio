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
    private var next = 0
    public var nodes: [NodeID: Node] = [:]
    private var nextCellId = 0
    public var nextTensorId = 0
    public var tensors: [TensorID: Tensor] = [:]
    public var nodeToTensor: [NodeID: TensorID] = [:]
    public var cellToTensor: [CellID: TensorID] = [:]  // Maps cell IDs to their associated tensor

    /// Track allocation sizes for memory cells (especially large buffers like spectral scratch)
    public var cellAllocationSizes: [CellID: Int] = [:]

    /// Tracks hop-based update rate for nodes (hopSize, counterCell)
    /// Used for FFT/IFFT nodes and operations that inherit hop-based temporality
    public var nodeHopRate: [NodeID: (Int, CellID)] = [:]

    /// Sample rate for audio processing (default 44100 Hz)
    public var sampleRate: Float = 44100.0

    /// Mapping from history cell IDs to gradient carry cell IDs.
    /// Used for temporal gradient flow through historyRead/historyWrite.
    public var gradCarryCells: [CellID: CellID] = [:]
    public var tensorGradCells: [NodeID: CellID] = [:]

    /// Side-effect nodes created during backward pass (e.g., gradient carry writes)
    /// These need to be chained with gradient outputs to ensure they execute.
    public var gradientSideEffects: [NodeID] = []

    /// Last node ID before gradient nodes were added.
    /// Used to separate forward and gradient node ordering during compilation.
    public var lastForwardNodeId: NodeID?

    public init() {}

    public init(sampleRate: Float) {
        self.sampleRate = sampleRate
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

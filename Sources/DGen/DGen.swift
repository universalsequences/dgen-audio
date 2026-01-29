import Foundation

public class IRContext {
    public var g: Graph
    private var varIdx = 0
    private var gradIdx = 0
    private var constantIdx = 0

    // Maps node ids to what the current "index" is for tensor-based computation
    // E.g we are on the 2nd element in tensor in a loop
    public var tensorIndices: [NodeID: Lazy] = [:]

    // Maximum gradient ID allocated (for buffer sizing)
    public var maxGradId: Int { return gradIdx }
    // Reuse constant IDs for identical values to reduce duplicate vdupq constants
    private var constantIdByValue: [Float: ConstantID] = [:]

    // Tensor register optimization:
    // - outboundTensorCells: tensor cells that must be written to memory (needed by later blocks)
    // - tensorCellToVar: maps cell IDs to computed Lazy values (register variables) within current block
    // This allows intermediate tensor values to stay in registers instead of going through memory.
    public var outboundTensorCells: Set<CellID> = []
    public var tensorCellToVar: [CellID: Lazy] = [:]

    // Frame-based nodes from temporality analysis (set during compilation)
    public var frameBasedNodes: Set<NodeID> = []

    /// Clear tensor register tracking (call at start of each tensor block)
    public func clearTensorRegisters() {
        tensorCellToVar = [:]
    }

    public init(g: Graph) {
        self.g = g
    }

    // Use Array instead of Set to maintain stable ordering for tape slot assignment
    public var globals: [VarID] = []

    // map of nodeId -> Lazy value (variable or constant)
    public var values: [NodeID: Lazy] = [:]
    public var gradients: [NodeID: GradID] = [:]
    public var constants: [ConstantID: Float] = [:]
    public var variables: [VarID: NodeID] = [:]
    public var tapeIndex: [NodeID: Int] = [:]
    public var seedGradients: [GradID] = []

    // Tensor gradient support: maps tensor nodes to base GradID for contiguous allocation
    public var tensorGradients: [NodeID: GradID] = [:]
    public var tensorGradientSizes: [NodeID: Int] = [:]

    // Track which tensor gradients are frame-based (need frameCount multiplier)
    // GradIDs not in this set are static (only need tensor size)
    public var frameBasedGradients: Set<GradID> = []

    // Track scalar gradients that are frame-based
    public var frameBasedScalarGradients: Set<GradID> = []

    public func getGlobalId(_ varId: VarID) -> Int {
        if let index = globals.firstIndex(of: varId) {
            return index
        }
        return 0
    }

    public func useConstant(src: NodeID?, value: Float) -> Lazy {
        if let existing = constantIdByValue[value] {
            let constant = Lazy.constant(existing, value)
            if let srcId = src { self.values[srcId] = constant }
            return constant
        }

        let constantId = self.constantIdx + 1
        self.constantIdx = constantId
        self.constants[constantId] = value
        constantIdByValue[value] = constantId

        let constant = Lazy.constant(constantId, value)
        if let srcId = src { self.values[srcId] = constant }
        return constant
    }

    public func useGradient(src: NodeID, seed: Bool = false) -> GradID {
        if let gradId = self.gradients[src] {
            return gradId
        }
        let gradId = self.gradIdx + 1
        self.gradIdx = gradId
        self.gradients[src] = gradId
        // Auto-detect if this node is frame-based from temporality analysis
        if frameBasedNodes.contains(src) {
            self.frameBasedScalarGradients.insert(gradId)
        }
        if seed {
            self.seedGradients.append(gradId)
        }
        return gradId
    }

    /// Allocate contiguous block of GradIDs for tensor gradients.
    /// Each tensor element gets its own GradID: baseGradId, baseGradId+1, ..., baseGradId+(size-1)
    /// - Parameters:
    ///   - src: The tensor node ID
    ///   - size: Number of elements in the tensor
    ///   - seed: If true, add all GradIDs to seedGradients
    /// - Returns: The base GradID for this tensor
    public func useTensorGradient(src: NodeID, size: Int, seed: Bool = false) -> GradID {
        if let existing = tensorGradients[src] {
            return existing
        }
        let baseGradId = gradIdx + 1
        // Auto-detect if this node is frame-based from temporality analysis
        let isFrameBased = frameBasedNodes.contains(src)
        gradIdx += size  // Reserve `size` contiguous IDs
        tensorGradients[src] = baseGradId
        tensorGradientSizes[src] = size
        if isFrameBased {
            frameBasedGradients.insert(baseGradId)
        }
        if seed {
            for i in 0..<size {
                seedGradients.append(baseGradId + i)
            }
        }
        return baseGradId
    }

    /// Compute total gradient buffer size.
    /// Current layout: gradients[(gradId) * frameCount + threadIndex]
    /// So total buffer size = (maxGradId + 1) * frameCount
    public func computeGradientBufferSize(frameCount: Int) -> Int {
        let currentSize = (maxGradId + 1) * frameCount
        return 2 * currentSize  // 2x for safety margin
    }

    public func useVariable(src: NodeID?, trackInValues: Bool = true) -> Lazy {
        let varId = self.varIdx + 1
        self.varIdx = varId
        let variable = Lazy.variable(varId, src)
        if let srcNodeId = src, trackInValues {
            self.values[srcNodeId] = variable
            self.variables[varId] = srcNodeId
        }
        return variable
    }
}

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
        case let .variable(id, _):
            return id
        default:
            return nil
        }
    }
}

extension Op {
    public var operands: [Lazy] {
        switch self {
        case let .add(a, b):
            return [a, b]
        case let .mul(a, b):
            return [a, b]
        case let .sub(a, b):
            return [a, b]
        case let .div(a, b):
            return [a, b]
        case let .abs(a):
            return [a]
        case let .sign(a):
            return [a]
        case let .gt(a, b):
            return [a, b]
        case let .lt(a, b):
            return [a, b]
        case let .store(_, b):
            return [b]
        case .load(_):
            return []
        case let .beginIf(a):
            return [a]
        case let .mutate(a, b):
            return [a, b]
        default:
            return []
        }
    }
}

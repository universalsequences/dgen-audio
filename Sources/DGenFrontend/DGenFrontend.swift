import DGen

/// A wrapper around NodeID that provides operator overloading and a clean API
public struct Node {
    public let id: NodeID
    public let graph: Graph

    public init(id: NodeID, graph: Graph) {
        self.id = id
        self.graph = graph
    }
}

// MARK: - GraphBuilder

/// Extended Graph with convenience methods for building audio graphs
public class GraphBuilder: Graph {

    // MARK: - Constants and Inputs

    /// Create a constant node
    public func constant(_ value: Float) -> Node {
        return Node(id: n(.constant(value)), graph: self)
    }

    /// Create an input node
    public func input(_ index: Int = 0) -> Node {
        return Node(id: n(.input(index)), graph: self)
    }

    /// Create an output node
    public func output(_ node: Node, index: Int = 0) {
        n(.output(index), node.id)
    }

    // MARK: - Learnable Parameters

    /// Create a learnable parameter (for training)
    public func learnableParam(value: Float, name: String? = nil) -> (Parameter, Node) {
        let param = Parameter(graph: self, value: value, name: name)
        return (param, Node(id: param.node(), graph: self))
    }

    // MARK: - Oscillators

    /// Phasor (ramp oscillator) with optional reset trigger
    public func phasor(_ freq: Node, reset: Node? = nil) -> Node {
        let cellId = alloc()
        if let reset = reset {
            return Node(id: n(.phasor(cellId), freq.id, reset.id), graph: self)
        } else {
            return Node(id: n(.phasor(cellId), freq.id, self.constant(0.0).id), graph: self)
        }
    }

    // MARK: - Loss Functions

    /// Mean squared error
    public func mse(_ a: Node, _ b: Node) -> Node {
        return Node(id: n(.mse, a.id, b.id), graph: self)
    }

    /// Spectral loss (DFT-based)
    public func spectralLoss(_ a: Node, _ b: Node, windowSize: Int = 64) -> Node {
        return Node(id: spectralLoss(a.id, b.id, windowSize: windowSize), graph: self)
    }

    // MARK: - Memory Operations

    /// History read - reads the previous value from a memory cell
    public func historyRead(_ cellId: CellID) -> Node {
        return Node(id: n(.historyRead(cellId)), graph: self)
    }

    /// History write - writes a value to a memory cell for next frame
    @discardableResult
    public func historyWrite(_ cellId: CellID, _ signal: Node) -> Node {
        return Node(id: n(.historyWrite(cellId), signal.id), graph: self)
    }

    /// Latch - sample and hold
    public func latch(_ signal: Node, trigger: Node) -> Node {
        let cellId = alloc()
        return Node(id: n(.latch(cellId), signal.id, trigger.id), graph: self)
    }

    /// Accumulator
    public func accum(_ signal: Node) -> Node {
        let cellId = alloc()
        return Node(id: n(.accum(cellId), signal.id), graph: self)
    }

    // MARK: - Filters and Effects

    /// Biquad filter
    public func biquad(_ input: Node, _ cutoff: Node, _ resonance: Node, _ gain: Node, _ mode: Node) -> Node {
        return Node(id: biquad(input.id, cutoff.id, resonance.id, gain.id, mode.id), graph: self)
    }

    /// Compressor
    public func compressor(
        _ input: Node, _ ratio: Node, _ threshold: Node, _ knee: Node, _ attack: Node,
        _ release: Node, _ isSideChain: Node, _ sidechainIn: Node
    ) -> Node {
        return Node(
            id: compressor(input.id, ratio.id, threshold.id, knee.id, attack.id, release.id, isSideChain.id, sidechainIn.id),
            graph: self
        )
    }

    /// Delay with linear interpolation
    public func delay(_ input: Node, _ delayTimeInSamples: Node) -> Node {
        return Node(id: delay(input.id, delayTimeInSamples.id), graph: self)
    }

    // MARK: - Utility Operations

    /// Mix (linear interpolation) - mix(a, b, 0.0) = a, mix(a, b, 1.0) = b
    public func mix(_ a: Node, _ b: Node, _ amount: Node) -> Node {
        return Node(id: n(.mix, a.id, b.id, amount.id), graph: self)
    }

    /// Delta - difference between current and previous value
    public func delta(_ input: Node) -> Node {
        return Node(id: delta(input.id), graph: self)
    }

    /// Change - sign of the difference between current and previous value
    public func change(_ input: Node) -> Node {
        return Node(id: change(input.id), graph: self)
    }

    /// RampToTrig - detects discontinuities in a ramp signal
    public func rampToTrig(_ ramp: Node) -> Node {
        return Node(id: rampToTrig(ramp.id), graph: self)
    }

    /// Scale - maps a value from one range to another with optional exponential curve
    public func scale(
        _ value: Node, _ min1: Node, _ max1: Node, _ min2: Node, _ max2: Node,
        _ exponent: Node? = nil
    ) -> Node {
        if let exp = exponent {
            return Node(id: scale(value.id, min1.id, max1.id, min2.id, max2.id, exp.id), graph: self)
        } else {
            return Node(id: scale(value.id, min1.id, max1.id, min2.id, max2.id), graph: self)
        }
    }

    /// Triangle - converts a ramp signal to a triangle wave
    public func triangle(_ ramp: Node, _ duty: Node? = nil) -> Node {
        if let d = duty {
            return Node(id: triangle(ramp.id, d.id), graph: self)
        } else {
            return Node(id: triangle(ramp.id), graph: self)
        }
    }

    /// Wrap - wraps value to range [min, max]
    public func wrap(_ input: Node, _ min: Node, _ max: Node) -> Node {
        return Node(id: wrap(input.id, min.id, max.id), graph: self)
    }

    /// Clip - clamps value to range [min, max]
    public func clip(_ input: Node, _ min: Node, _ max: Node) -> Node {
        return Node(id: clip(input.id, min.id, max.id), graph: self)
    }

    /// Selector - selects from multiple inputs based on condition
    public func selector(_ cond: Node, _ args: Node...) -> Node {
        let argIds = args.map { $0.id }
        return Node(id: n(.selector, [cond.id] + argIds), graph: self)
    }

    // MARK: - Compilation

    /// Compile the graph with a loss node as output
    /// - Parameters:
    ///   - loss: The loss node to use as output
    ///   - backend: Compilation backend (default: .metal)
    ///   - frameCount: Number of frames per batch (default: 128)
    ///   - debug: Enable debug output (default: false)
    ///   - backwards: Enable backwards pass for training (default: true)
    /// - Returns: Compilation result containing kernels and metadata
    public func compile(
        _ loss: Node,
        backend: Backend = .metal,
        frameCount: Int = 128,
        debug: Bool = false,
        backwards: Bool = true
    ) throws -> CompilationResult {
        // Set the loss as output
        output(loss)

        // Compile with specified options
        return try CompilationPipeline.compile(
            graph: self,
            backend: backend,
            options: CompilationPipeline.Options(
                frameCount: frameCount,
                debug: debug,
                backwards: backwards
            )
        )
    }
}

// MARK: - Arithmetic Operators

extension Node {
    /// Addition: a + b
    public static func + (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.add, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Subtraction: a - b
    public static func - (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.sub, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Multiplication: a * b
    public static func * (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.mul, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Division: a / b
    public static func / (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.div, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Negation: -a (using 0 - node)
    public static prefix func - (node: Node) -> Node {
        let zero = node.graph.n(.constant(0.0))
        return Node(id: node.graph.n(.sub, zero, node.id), graph: node.graph)
    }
}

// MARK: - Comparison Operators

extension Node {
    /// Greater than: a > b
    public static func > (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.gt, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Greater than or equal: a >= b
    public static func >= (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.gte, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Less than: a < b
    public static func < (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.lt, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Less than or equal: a <= b
    public static func <= (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.lte, lhs.id, rhs.id), graph: lhs.graph)
    }

    /// Equal: a == b (note: returns a Node, not a Bool)
    public static func == (lhs: Node, rhs: Node) -> Node {
        assert(lhs.graph === rhs.graph, "Nodes must belong to the same graph")
        return Node(id: lhs.graph.n(.eq, lhs.id, rhs.id), graph: lhs.graph)
    }
}

// MARK: - Scalar Operations

extension Node {
    /// Add scalar: node + 5.0
    public static func + (lhs: Node, rhs: Float) -> Node {
        let constId = lhs.graph.n(.constant(rhs))
        return Node(id: lhs.graph.n(.add, lhs.id, constId), graph: lhs.graph)
    }

    /// Add scalar (reversed): 5.0 + node
    public static func + (lhs: Float, rhs: Node) -> Node {
        return rhs + lhs
    }

    /// Subtract scalar: node - 5.0
    public static func - (lhs: Node, rhs: Float) -> Node {
        let constId = lhs.graph.n(.constant(rhs))
        return Node(id: lhs.graph.n(.sub, lhs.id, constId), graph: lhs.graph)
    }

    /// Subtract from scalar: 5.0 - node
    public static func - (lhs: Float, rhs: Node) -> Node {
        let constId = rhs.graph.n(.constant(lhs))
        return Node(id: rhs.graph.n(.sub, constId, rhs.id), graph: rhs.graph)
    }

    /// Multiply scalar: node * 0.5
    public static func * (lhs: Node, rhs: Float) -> Node {
        let constId = lhs.graph.n(.constant(rhs))
        return Node(id: lhs.graph.n(.mul, lhs.id, constId), graph: lhs.graph)
    }

    /// Multiply scalar (reversed): 0.5 * node
    public static func * (lhs: Float, rhs: Node) -> Node {
        return rhs * lhs
    }

    /// Divide by scalar: node / 2.0
    public static func / (lhs: Node, rhs: Float) -> Node {
        let constId = lhs.graph.n(.constant(rhs))
        return Node(id: lhs.graph.n(.div, lhs.id, constId), graph: lhs.graph)
    }

    /// Divide scalar by node: 2.0 / node
    public static func / (lhs: Float, rhs: Node) -> Node {
        let constId = rhs.graph.n(.constant(lhs))
        return Node(id: rhs.graph.n(.div, constId, rhs.id), graph: rhs.graph)
    }
}

// MARK: - Input/Output/Parameters

/// Create an input node
public func input(_ graph: Graph, index: Int = 0) -> Node {
    return Node(id: graph.n(.input(index)), graph: graph)
}

/// Create a parameter node (cellId should be allocated with graph.alloc())
public func param(_ graph: Graph, cellId: CellID) -> Node {
    return Node(id: graph.n(.param(cellId)), graph: graph)
}

/// Create a parameter with auto-allocation and configuration
public func param(_ graph: Graph, name: String, min: Float, max: Float, default defaultValue: Float)
    -> Node
{
    let cellId = graph.alloc()
    // Note: This simplified version doesn't store min/max/name metadata
    // You may want to extend Graph to track this if needed
    return Node(id: graph.n(.param(cellId)), graph: graph)
}

/// Create a learnable parameter (for training)
public func learnableParam(_ graph: Graph, value: Float, name: String? = nil) -> (Parameter, Node) {
    let param = Parameter(graph: graph, value: value, name: name)
    return (param, Node(id: param.node(), graph: graph))
}

/// Create a constant node
public func constant(_ graph: Graph, _ value: Float) -> Node {
    return Node(id: graph.n(.constant(value)), graph: graph)
}

/// Mark a node as output
public func output(_ graph: Graph, _ node: Node, index: Int = 0) {
    graph.n(.output(index), node.id)
}

// MARK: - Oscillators

/// Phasor (ramp oscillator) with optional reset trigger
public func phasor(_ freq: Node, reset: Node? = nil) -> Node {
    let cellId = freq.graph.alloc()
    if let reset = reset {
        return Node(id: freq.graph.n(.phasor(cellId), freq.id, reset.id), graph: freq.graph)
    } else {
        return Node(id: freq.graph.n(.phasor(cellId), freq.id), graph: freq.graph)
    }
}

// MARK: - Memory Operations

/// Latch - sample and hold
public func latch(_ graph: Graph, signal: Node, trigger: Node, cellId: CellID) -> Node {
    return Node(id: graph.n(.latch(cellId), signal.id, trigger.id), graph: graph)
}

/// Latch with auto-allocation
public func latch(_ signal: Node, trigger: Node) -> Node {
    let cellId = signal.graph.alloc()
    return Node(id: signal.graph.n(.latch(cellId), signal.id, trigger.id), graph: signal.graph)
}

/// Accumulator
public func accum(_ graph: Graph, signal: Node, cellId: CellID) -> Node {
    return Node(id: graph.n(.accum(cellId), signal.id), graph: graph)
}

/// Accumulator with auto-allocation
public func accum(_ signal: Node) -> Node {
    let cellId = signal.graph.alloc()
    return Node(id: signal.graph.n(.accum(cellId), signal.id), graph: signal.graph)
}

// MARK: - Math Functions

/// Absolute value
public func abs(_ node: Node) -> Node {
    return Node(id: node.graph.n(.abs, node.id), graph: node.graph)
}

/// Sign (-1, 0, or 1)
public func sign(_ node: Node) -> Node {
    return Node(id: node.graph.n(.sign, node.id), graph: node.graph)
}

/// Exponential
public func exp(_ node: Node) -> Node {
    return Node(id: node.graph.n(.exp, node.id), graph: node.graph)
}

/// Natural logarithm
public func log(_ node: Node) -> Node {
    return Node(id: node.graph.n(.log, node.id), graph: node.graph)
}

/// Base-10 logarithm
public func log10(_ node: Node) -> Node {
    return Node(id: node.graph.n(.log10, node.id), graph: node.graph)
}

/// Power (base^exponent)
public func pow(_ base: Node, _ exponent: Node) -> Node {
    return Node(id: base.graph.n(.pow, base.id, exponent.id), graph: base.graph)
}

/// Square root
public func sqrt(_ node: Node) -> Node {
    return Node(id: node.graph.n(.sqrt, node.id), graph: node.graph)
}

/// Sine
public func sin(_ node: Node) -> Node {
    return Node(id: node.graph.n(.sin, node.id), graph: node.graph)
}

/// Cosine
public func cos(_ node: Node) -> Node {
    return Node(id: node.graph.n(.cos, node.id), graph: node.graph)
}

/// Tangent
public func tan(_ node: Node) -> Node {
    return Node(id: node.graph.n(.tan, node.id), graph: node.graph)
}

/// Hyperbolic tangent (soft clipping)
public func tanh(_ node: Node) -> Node {
    return Node(id: node.graph.n(.tanh, node.id), graph: node.graph)
}

/// Arctangent2 (angle from x,y coordinates)
public func atan2(_ y: Node, _ x: Node) -> Node {
    return Node(id: y.graph.n(.atan2, y.id, x.id), graph: y.graph)
}

// MARK: - Loss Functions

/// Mean squared error
public func mse(_ a: Node, _ b: Node) -> Node {
    assert(a.graph === b.graph, "Nodes must belong to the same graph")
    return Node(id: a.graph.n(.mse, a.id, b.id), graph: a.graph)
}

/// Spectral loss (DFT-based)
public func spectralLoss(_ a: Node, _ b: Node, windowSize: Int = 64) -> Node {
    assert(a.graph === b.graph, "Nodes must belong to the same graph")
    return Node(id: a.graph.spectralLoss(a.id, b.id, windowSize: windowSize), graph: a.graph)
}

// MARK: - Control Flow

/// Sequence operator (for temporal ordering) - returns last value
public func seq(_ nodes: Node...) -> Node {
    guard let first = nodes.first else {
        fatalError("seq requires at least one node")
    }
    let graph = first.graph
    let ids = nodes.map { node in
        assert(node.graph === graph, "All nodes must belong to the same graph")
        return node.id
    }
    return Node(id: graph.n(.seq, ids), graph: graph)
}

// MARK: - Utility

/// Min of two nodes
public func min(_ a: Node, _ b: Node) -> Node {
    assert(a.graph === b.graph, "Nodes must belong to the same graph")
    return Node(id: a.graph.n(.min, a.id, b.id), graph: a.graph)
}

/// Max of two nodes
public func max(_ a: Node, _ b: Node) -> Node {
    assert(a.graph === b.graph, "Nodes must belong to the same graph")
    return Node(id: a.graph.n(.max, a.id, b.id), graph: a.graph)
}

/// Mix between two signals (linear interpolation)
/// mix(a, b, 0.0) = a, mix(a, b, 1.0) = b
public func mix(_ a: Node, _ b: Node, _ amount: Node) -> Node {
    assert(a.graph === b.graph && b.graph === amount.graph, "Nodes must belong to the same graph")
    return Node(id: a.graph.n(.mix, a.id, b.id, amount.id), graph: a.graph)
}

/// Floor (round down)
public func floor(_ node: Node) -> Node {
    return Node(id: node.graph.n(.floor, node.id), graph: node.graph)
}

/// Ceiling (round up)
public func ceil(_ node: Node) -> Node {
    return Node(id: node.graph.n(.ceil, node.id), graph: node.graph)
}

/// Round to nearest integer
public func round(_ node: Node) -> Node {
    return Node(id: node.graph.n(.round, node.id), graph: node.graph)
}

/// Modulo
public func mod(_ a: Node, _ b: Node) -> Node {
    assert(a.graph === b.graph, "Nodes must belong to the same graph")
    return Node(id: a.graph.n(.mod, a.id, b.id), graph: a.graph)
}

/// Clamp using min/max
public func clamp(_ value: Node, min minValue: Float, max maxValue: Float) -> Node {
    let minNode = constant(value.graph, minValue)
    let maxNode = constant(value.graph, maxValue)
    return min(max(value, minNode), maxNode)
}

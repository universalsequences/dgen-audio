// Operators - Arithmetic operator overloads for lazy types
//
// Provides +, -, *, / operators for Tensor, Signal, and SignalTensor
// with automatic type promotion.

import DGen

// MARK: - Tensor Operators

// Tensor + Tensor
public func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Tensor - Tensor
public func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.sub, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Tensor * Tensor
public func * (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Tensor / Tensor
public func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.div, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Tensor + Float
public func + (lhs: Tensor, rhs: Float) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad)
}

// Float + Tensor
public func + (lhs: Float, rhs: Tensor) -> Tensor {
    return rhs + lhs
}

// Tensor - Float
public func - (lhs: Tensor, rhs: Float) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.sub, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad)
}

// Float - Tensor
public func - (lhs: Float, rhs: Tensor) -> Tensor {
    let lhsNode = rhs.graph.node(.constant(lhs))
    let nodeId = rhs.graph.node(.sub, [lhsNode, rhs.nodeId])
    return Tensor(nodeId: nodeId, graph: rhs.graph, shape: rhs.shape, requiresGrad: rhs.requiresGrad)
}

// Tensor * Float
public func * (lhs: Tensor, rhs: Float) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad)
}

// Float * Tensor
public func * (lhs: Float, rhs: Tensor) -> Tensor {
    return rhs * lhs
}

// Tensor / Float
public func / (lhs: Tensor, rhs: Float) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.div, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad)
}

// Float / Tensor
public func / (lhs: Float, rhs: Tensor) -> Tensor {
    let lhsNode = rhs.graph.node(.constant(lhs))
    let nodeId = rhs.graph.node(.div, [lhsNode, rhs.nodeId])
    return Tensor(nodeId: nodeId, graph: rhs.graph, shape: rhs.shape, requiresGrad: rhs.requiresGrad)
}

// Negation
public prefix func - (x: Tensor) -> Tensor {
    let negOne = x.graph.node(.constant(-1.0))
    let nodeId = x.graph.node(.mul, [x.nodeId, negOne])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

// MARK: - Signal Operators

// Signal + Signal
public func + (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal - Signal
public func - (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.sub, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal * Signal
public func * (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal / Signal
public func / (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.div, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal + Float
public func + (lhs: Signal, rhs: Float) -> Signal {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad)
}

// Float + Signal
public func + (lhs: Float, rhs: Signal) -> Signal {
    return rhs + lhs
}

// Signal - Float
public func - (lhs: Signal, rhs: Float) -> Signal {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.sub, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad)
}

// Float - Signal
public func - (lhs: Float, rhs: Signal) -> Signal {
    let lhsNode = rhs.graph.node(.constant(lhs))
    let nodeId = rhs.graph.node(.sub, [lhsNode, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: rhs.graph, requiresGrad: rhs.requiresGrad)
}

// Signal * Float
public func * (lhs: Signal, rhs: Float) -> Signal {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad)
}

// Float * Signal
public func * (lhs: Float, rhs: Signal) -> Signal {
    return rhs * lhs
}

// Signal / Float
public func / (lhs: Signal, rhs: Float) -> Signal {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.div, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: lhs.requiresGrad)
}

// Float / Signal
public func / (lhs: Float, rhs: Signal) -> Signal {
    let lhsNode = rhs.graph.node(.constant(lhs))
    let nodeId = rhs.graph.node(.div, [lhsNode, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: rhs.graph, requiresGrad: rhs.requiresGrad)
}

// Negation
public prefix func - (x: Signal) -> Signal {
    let negOne = x.graph.node(.constant(-1.0))
    let nodeId = x.graph.node(.mul, [x.nodeId, negOne])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

// MARK: - Type Promotion: Tensor + Signal -> SignalTensor

// Tensor + Signal -> SignalTensor
public func + (lhs: Tensor, rhs: Signal) -> SignalTensor {
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal + Tensor -> SignalTensor
public func + (lhs: Signal, rhs: Tensor) -> SignalTensor {
    return rhs + lhs
}

// Tensor - Signal -> SignalTensor
public func - (lhs: Tensor, rhs: Signal) -> SignalTensor {
    let nodeId = lhs.graph.node(.sub, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal - Tensor -> SignalTensor
public func - (lhs: Signal, rhs: Tensor) -> SignalTensor {
    let nodeId = lhs.graph.node(.sub, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: rhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Tensor * Signal -> SignalTensor
public func * (lhs: Tensor, rhs: Signal) -> SignalTensor {
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal * Tensor -> SignalTensor
public func * (lhs: Signal, rhs: Tensor) -> SignalTensor {
    return rhs * lhs
}

// Tensor / Signal -> SignalTensor
public func / (lhs: Tensor, rhs: Signal) -> SignalTensor {
    let nodeId = lhs.graph.node(.div, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal / Tensor -> SignalTensor
public func / (lhs: Signal, rhs: Tensor) -> SignalTensor {
    let nodeId = lhs.graph.node(.div, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: rhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// MARK: - SignalTensor Operators

// SignalTensor + SignalTensor
public func + (lhs: SignalTensor, rhs: SignalTensor) -> SignalTensor {
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// SignalTensor * SignalTensor
public func * (lhs: SignalTensor, rhs: SignalTensor) -> SignalTensor {
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// SignalTensor + Signal (broadcast scalar)
public func + (lhs: SignalTensor, rhs: Signal) -> SignalTensor {
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal + SignalTensor
public func + (lhs: Signal, rhs: SignalTensor) -> SignalTensor {
    return rhs + lhs
}

// SignalTensor * Signal (broadcast scalar)
public func * (lhs: SignalTensor, rhs: Signal) -> SignalTensor {
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhs.nodeId])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad || rhs.requiresGrad)
}

// Signal * SignalTensor
public func * (lhs: Signal, rhs: SignalTensor) -> SignalTensor {
    return rhs * lhs
}

// SignalTensor + Float
public func + (lhs: SignalTensor, rhs: Float) -> SignalTensor {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.add, [lhs.nodeId, rhsNode])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad)
}

// SignalTensor * Float
public func * (lhs: SignalTensor, rhs: Float) -> SignalTensor {
    let rhsNode = lhs.graph.node(.constant(rhs))
    let nodeId = lhs.graph.node(.mul, [lhs.nodeId, rhsNode])
    return SignalTensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: lhs.requiresGrad)
}

// Negation
public prefix func - (x: SignalTensor) -> SignalTensor {
    let negOne = x.graph.node(.constant(-1.0))
    let nodeId = x.graph.node(.mul, [x.nodeId, negOne])
    return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

// MARK: - Comparison Operators (Signal)
// Returns 1.0 if true, 0.0 if false

/// Signal > Signal
public func > (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.gt, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Signal >= Signal
public func >= (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.gte, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Signal < Signal
public func < (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.lt, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Signal <= Signal
public func <= (lhs: Signal, rhs: Signal) -> Signal {
    let nodeId = lhs.graph.node(.lte, [lhs.nodeId, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Signal > Double (for literals like 5.0)
public func > (lhs: Signal, rhs: Double) -> Signal {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.gt, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Double > Signal
public func > (lhs: Double, rhs: Signal) -> Signal {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.gt, [lhsNode, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: rhs.graph, requiresGrad: false)
}

/// Signal >= Double
public func >= (lhs: Signal, rhs: Double) -> Signal {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.gte, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Double >= Signal
public func >= (lhs: Double, rhs: Signal) -> Signal {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.gte, [lhsNode, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: rhs.graph, requiresGrad: false)
}

/// Signal < Double
public func < (lhs: Signal, rhs: Double) -> Signal {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.lt, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Double < Signal
public func < (lhs: Double, rhs: Signal) -> Signal {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.lt, [lhsNode, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: rhs.graph, requiresGrad: false)
}

/// Signal <= Double
public func <= (lhs: Signal, rhs: Double) -> Signal {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.lte, [lhs.nodeId, rhsNode])
    return Signal(nodeId: nodeId, graph: lhs.graph, requiresGrad: false)
}

/// Double <= Signal
public func <= (lhs: Double, rhs: Signal) -> Signal {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.lte, [lhsNode, rhs.nodeId])
    return Signal(nodeId: nodeId, graph: rhs.graph, requiresGrad: false)
}

// MARK: - Comparison Operators (Tensor)

/// Tensor > Tensor
public func > (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.gt, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: false)
}

/// Tensor >= Tensor
public func >= (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.gte, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: false)
}

/// Tensor < Tensor
public func < (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.lt, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: false)
}

/// Tensor <= Tensor
public func <= (lhs: Tensor, rhs: Tensor) -> Tensor {
    let nodeId = lhs.graph.node(.lte, [lhs.nodeId, rhs.nodeId])
    let outShape = broadcastShape(lhs.shape, rhs.shape)
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: outShape, requiresGrad: false)
}

/// Tensor > Double
public func > (lhs: Tensor, rhs: Double) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.gt, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: false)
}

/// Double > Tensor
public func > (lhs: Double, rhs: Tensor) -> Tensor {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.gt, [lhsNode, rhs.nodeId])
    return Tensor(nodeId: nodeId, graph: rhs.graph, shape: rhs.shape, requiresGrad: false)
}

/// Tensor >= Double
public func >= (lhs: Tensor, rhs: Double) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.gte, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: false)
}

/// Double >= Tensor
public func >= (lhs: Double, rhs: Tensor) -> Tensor {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.gte, [lhsNode, rhs.nodeId])
    return Tensor(nodeId: nodeId, graph: rhs.graph, shape: rhs.shape, requiresGrad: false)
}

/// Tensor < Double
public func < (lhs: Tensor, rhs: Double) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.lt, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: false)
}

/// Double < Tensor
public func < (lhs: Double, rhs: Tensor) -> Tensor {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.lt, [lhsNode, rhs.nodeId])
    return Tensor(nodeId: nodeId, graph: rhs.graph, shape: rhs.shape, requiresGrad: false)
}

/// Tensor <= Double
public func <= (lhs: Tensor, rhs: Double) -> Tensor {
    let rhsNode = lhs.graph.node(.constant(Float(rhs)))
    let nodeId = lhs.graph.node(.lte, [lhs.nodeId, rhsNode])
    return Tensor(nodeId: nodeId, graph: lhs.graph, shape: lhs.shape, requiresGrad: false)
}

/// Double <= Tensor
public func <= (lhs: Double, rhs: Tensor) -> Tensor {
    let lhsNode = rhs.graph.node(.constant(Float(lhs)))
    let nodeId = rhs.graph.node(.lte, [lhsNode, rhs.nodeId])
    return Tensor(nodeId: nodeId, graph: rhs.graph, shape: rhs.shape, requiresGrad: false)
}

// MARK: - Equality (as methods to avoid conflict with Equatable)

extension Signal {
    /// Element-wise equality (returns 1.0 if equal, 0.0 otherwise)
    public func eq(_ other: Signal) -> Signal {
        let nodeId = graph.node(.eq, [self.nodeId, other.nodeId])
        return Signal(nodeId: nodeId, graph: graph, requiresGrad: false)
    }

    /// Element-wise equality with Float
    public func eq(_ other: Float) -> Signal {
        let otherNode = graph.node(.constant(other))
        let nodeId = graph.node(.eq, [self.nodeId, otherNode])
        return Signal(nodeId: nodeId, graph: graph, requiresGrad: false)
    }
}

extension Tensor {
    /// Element-wise equality (returns 1.0 if equal, 0.0 otherwise)
    public func eq(_ other: Tensor) -> Tensor {
        let nodeId = graph.node(.eq, [self.nodeId, other.nodeId])
        let outShape = broadcastShape(shape, other.shape)
        return Tensor(nodeId: nodeId, graph: graph, shape: outShape, requiresGrad: false)
    }

    /// Element-wise equality with Float
    public func eq(_ other: Float) -> Tensor {
        let otherNode = graph.node(.constant(other))
        let nodeId = graph.node(.eq, [self.nodeId, otherNode])
        return Tensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: false)
    }
}

// MARK: - Helper Functions

/// Compute the broadcast shape of two shapes
func broadcastShape(_ a: Shape, _ b: Shape) -> Shape {
    let maxDims = max(a.count, b.count)
    var result = Shape(repeating: 1, count: maxDims)

    for i in 0..<maxDims {
        let aDim = i < a.count ? a[a.count - 1 - i] : 1
        let bDim = i < b.count ? b[b.count - 1 - i] : 1

        if aDim == bDim {
            result[maxDims - 1 - i] = aDim
        } else if aDim == 1 {
            result[maxDims - 1 - i] = bDim
        } else if bDim == 1 {
            result[maxDims - 1 - i] = aDim
        } else {
            // Shape mismatch - will error at runtime
            result[maxDims - 1 - i] = max(aDim, bDim)
        }
    }

    return result
}

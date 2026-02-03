// Functions - Math functions for lazy types
//
// Global functions like sin, cos, relu, etc. that work with
// Tensor, Signal, and SignalTensor.

import DGen

// MARK: - Unary Math Functions (Tensor)

public func abs(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.abs, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func sign(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.sign, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func exp(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.exp, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func log(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.log, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func sqrt(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.sqrt, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func sin(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.sin, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func cos(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.cos, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func tan(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.tan, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func tanh(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.tanh, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func floor(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.floor, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func ceil(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.ceil, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func round(_ x: Tensor) -> Tensor {
    let nodeId = x.graph.node(.round, [x.nodeId])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

// MARK: - Unary Math Functions (Signal)

public func abs(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.abs, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func sign(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.sign, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func exp(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.exp, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func log(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.log, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func sqrt(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.sqrt, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func sin(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.sin, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func cos(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.cos, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func tan(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.tan, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func tanh(_ x: Signal) -> Signal {
    let nodeId = x.graph.node(.tanh, [x.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

// MARK: - Unary Math Functions (SignalTensor)

public func sin(_ x: SignalTensor) -> SignalTensor {
    let nodeId = x.graph.node(.sin, [x.nodeId])
    return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func cos(_ x: SignalTensor) -> SignalTensor {
    let nodeId = x.graph.node(.cos, [x.nodeId])
    return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func exp(_ x: SignalTensor) -> SignalTensor {
    let nodeId = x.graph.node(.exp, [x.nodeId])
    return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func tanh(_ x: SignalTensor) -> SignalTensor {
    let nodeId = x.graph.node(.tanh, [x.nodeId])
    return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

// MARK: - Binary Math Functions

public func pow(_ x: Tensor, _ y: Tensor) -> Tensor {
    let nodeId = x.graph.node(.pow, [x.nodeId, y.nodeId])
    let outShape = broadcastShape(x.shape, y.shape)
    return Tensor(nodeId: nodeId, graph: x.graph, shape: outShape, requiresGrad: x.requiresGrad || y.requiresGrad)
}

public func pow(_ x: Tensor, _ y: Float) -> Tensor {
    let yNode = x.graph.node(.constant(y))
    let nodeId = x.graph.node(.pow, [x.nodeId, yNode])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func pow(_ x: Signal, _ y: Signal) -> Signal {
    let nodeId = x.graph.node(.pow, [x.nodeId, y.nodeId])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad || y.requiresGrad)
}

public func pow(_ x: Signal, _ y: Float) -> Signal {
    let yNode = x.graph.node(.constant(y))
    let nodeId = x.graph.node(.pow, [x.nodeId, yNode])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

// MARK: - Activation Functions

public func relu(_ x: Tensor) -> Tensor {
    // relu(x) = max(x, 0)
    let zero = x.graph.node(.constant(0.0))
    let nodeId = x.graph.node(.max, [x.nodeId, zero])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func relu(_ x: Signal) -> Signal {
    let zero = x.graph.node(.constant(0.0))
    let nodeId = x.graph.node(.max, [x.nodeId, zero])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func relu(_ x: SignalTensor) -> SignalTensor {
    let zero = x.graph.node(.constant(0.0))
    let nodeId = x.graph.node(.max, [x.nodeId, zero])
    return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func sigmoid(_ x: Tensor) -> Tensor {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let one = x.graph.node(.constant(1.0))
    let negOne = x.graph.node(.constant(-1.0))
    let negX = x.graph.node(.mul, [x.nodeId, negOne])
    let expNegX = x.graph.node(.exp, [negX])
    let denom = x.graph.node(.add, [one, expNegX])
    let nodeId = x.graph.node(.div, [one, denom])
    return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func sigmoid(_ x: Signal) -> Signal {
    let one = x.graph.node(.constant(1.0))
    let negOne = x.graph.node(.constant(-1.0))
    let negX = x.graph.node(.mul, [x.nodeId, negOne])
    let expNegX = x.graph.node(.exp, [negX])
    let denom = x.graph.node(.add, [one, expNegX])
    let nodeId = x.graph.node(.div, [one, denom])
    return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

// MARK: - Min/Max Functions

public func min(_ a: Signal, _ b: Signal) -> Signal {
    let nodeId = a.graph.node(.min, [a.nodeId, b.nodeId])
    return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: a.requiresGrad || b.requiresGrad)
}

public func min(_ a: Signal, _ b: Double) -> Signal {
    let bNode = a.graph.node(.constant(Float(b)))
    let nodeId = a.graph.node(.min, [a.nodeId, bNode])
    return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: a.requiresGrad)
}

public func min(_ a: Double, _ b: Signal) -> Signal {
    return min(b, a)
}

public func max(_ a: Signal, _ b: Signal) -> Signal {
    let nodeId = a.graph.node(.max, [a.nodeId, b.nodeId])
    return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: a.requiresGrad || b.requiresGrad)
}

public func max(_ a: Signal, _ b: Double) -> Signal {
    let bNode = a.graph.node(.constant(Float(b)))
    let nodeId = a.graph.node(.max, [a.nodeId, bNode])
    return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: a.requiresGrad)
}

public func max(_ a: Double, _ b: Signal) -> Signal {
    return max(b, a)
}

public func min(_ a: Tensor, _ b: Tensor) -> Tensor {
    let nodeId = a.graph.node(.min, [a.nodeId, b.nodeId])
    let outShape = broadcastShape(a.shape, b.shape)
    return Tensor(nodeId: nodeId, graph: a.graph, shape: outShape, requiresGrad: a.requiresGrad || b.requiresGrad)
}

public func min(_ a: Tensor, _ b: Double) -> Tensor {
    let bNode = a.graph.node(.constant(Float(b)))
    let nodeId = a.graph.node(.min, [a.nodeId, bNode])
    return Tensor(nodeId: nodeId, graph: a.graph, shape: a.shape, requiresGrad: a.requiresGrad)
}

public func max(_ a: Tensor, _ b: Tensor) -> Tensor {
    let nodeId = a.graph.node(.max, [a.nodeId, b.nodeId])
    let outShape = broadcastShape(a.shape, b.shape)
    return Tensor(nodeId: nodeId, graph: a.graph, shape: outShape, requiresGrad: a.requiresGrad || b.requiresGrad)
}

public func max(_ a: Tensor, _ b: Double) -> Tensor {
    let bNode = a.graph.node(.constant(Float(b)))
    let nodeId = a.graph.node(.max, [a.nodeId, bNode])
    return Tensor(nodeId: nodeId, graph: a.graph, shape: a.shape, requiresGrad: a.requiresGrad)
}

// MARK: - Modulo

/// Modulo operation (remainder after division)
public func mod(_ a: Signal, _ b: Signal) -> Signal {
    let nodeId = a.graph.node(.mod, [a.nodeId, b.nodeId])
    return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: a.requiresGrad)
}

public func mod(_ a: Signal, _ b: Double) -> Signal {
    let bNode = a.graph.node(.constant(Float(b)))
    let nodeId = a.graph.node(.mod, [a.nodeId, bNode])
    return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: a.requiresGrad)
}

/// Infix % operator for modulo
public func % (lhs: Signal, rhs: Signal) -> Signal {
    return mod(lhs, rhs)
}

public func % (lhs: Signal, rhs: Double) -> Signal {
    return mod(lhs, rhs)
}

// MARK: - Conditional Selection

/// Conditional selection: returns `a` if `cond > 0`, else `b`
/// - Parameters:
///   - cond: Condition signal (positive selects a, zero/negative selects b)
///   - a: Value when condition is true
///   - b: Value when condition is false
/// - Returns: Selected value
public func gswitch(_ cond: Signal, _ a: Signal, _ b: Signal) -> Signal {
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, a.nodeId, b.nodeId])
    let needsGrad = cond.requiresGrad || a.requiresGrad || b.requiresGrad
    return Signal(nodeId: nodeId, graph: cond.graph, requiresGrad: needsGrad)
}

/// Conditional with Double values
public func gswitch(_ cond: Signal, _ a: Double, _ b: Double) -> Signal {
    let aNode = cond.graph.node(.constant(Float(a)))
    let bNode = cond.graph.node(.constant(Float(b)))
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, aNode, bNode])
    return Signal(nodeId: nodeId, graph: cond.graph, requiresGrad: cond.requiresGrad)
}

/// Conditional with Signal and Double
public func gswitch(_ cond: Signal, _ a: Signal, _ b: Double) -> Signal {
    let bNode = cond.graph.node(.constant(Float(b)))
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, a.nodeId, bNode])
    return Signal(nodeId: nodeId, graph: cond.graph, requiresGrad: cond.requiresGrad || a.requiresGrad)
}

public func gswitch(_ cond: Signal, _ a: Double, _ b: Signal) -> Signal {
    let aNode = cond.graph.node(.constant(Float(a)))
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, aNode, b.nodeId])
    return Signal(nodeId: nodeId, graph: cond.graph, requiresGrad: cond.requiresGrad || b.requiresGrad)
}

/// Tensor conditional selection
public func gswitch(_ cond: Tensor, _ a: Tensor, _ b: Tensor) -> Tensor {
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, a.nodeId, b.nodeId])
    let outShape = broadcastShape(broadcastShape(cond.shape, a.shape), b.shape)
    let needsGrad = cond.requiresGrad || a.requiresGrad || b.requiresGrad
    return Tensor(nodeId: nodeId, graph: cond.graph, shape: outShape, requiresGrad: needsGrad)
}

public func gswitch(_ cond: Tensor, _ a: Tensor, _ b: Double) -> Tensor {
    let bNode = cond.graph.node(.constant(Float(b)))
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, a.nodeId, bNode])
    let outShape = broadcastShape(cond.shape, a.shape)
    return Tensor(nodeId: nodeId, graph: cond.graph, shape: outShape, requiresGrad: cond.requiresGrad || a.requiresGrad)
}

public func gswitch(_ cond: Tensor, _ a: Double, _ b: Tensor) -> Tensor {
    let aNode = cond.graph.node(.constant(Float(a)))
    let nodeId = cond.graph.node(.gswitch, [cond.nodeId, aNode, b.nodeId])
    let outShape = broadcastShape(cond.shape, b.shape)
    return Tensor(nodeId: nodeId, graph: cond.graph, shape: outShape, requiresGrad: cond.requiresGrad || b.requiresGrad)
}

// MARK: - Loss Functions

/// Mean squared error loss
public func mse(_ pred: Tensor, _ target: Tensor) -> Tensor {
    let nodeId = pred.graph.node(.mse, [pred.nodeId, target.nodeId])
    return Tensor(nodeId: nodeId, graph: pred.graph, shape: [1], requiresGrad: pred.requiresGrad)
}

/// Mean squared error loss for signals
public func mse(_ pred: Signal, _ target: Signal) -> Signal {
    let nodeId = pred.graph.node(.mse, [pred.nodeId, target.nodeId])
    return Signal(nodeId: nodeId, graph: pred.graph, requiresGrad: pred.requiresGrad)
}

// MARK: - Tensor Reduction Operations

extension Tensor {
    /// Sum all elements
    public func sum() -> Tensor {
        let nodeId = graph.node(.sum, [self.nodeId])
        return Tensor(nodeId: nodeId, graph: graph, shape: [1], requiresGrad: requiresGrad)
    }

    /// Mean of all elements
    public func mean() -> Tensor {
        let sumNode = graph.node(.sum, [self.nodeId])
        let countNode = graph.node(.constant(Float(size)))
        let nodeId = graph.node(.div, [sumNode, countNode])
        return Tensor(nodeId: nodeId, graph: graph, shape: [1], requiresGrad: requiresGrad)
    }
}

// MARK: - Method Aliases

extension Tensor {
    public func abs() -> Tensor { DGenLazy.abs(self) }
    public func exp() -> Tensor { DGenLazy.exp(self) }
    public func log() -> Tensor { DGenLazy.log(self) }
    public func sqrt() -> Tensor { DGenLazy.sqrt(self) }
    public func sin() -> Tensor { DGenLazy.sin(self) }
    public func cos() -> Tensor { DGenLazy.cos(self) }
    public func tanh() -> Tensor { DGenLazy.tanh(self) }
    public func relu() -> Tensor { DGenLazy.relu(self) }
    public func sigmoid() -> Tensor { DGenLazy.sigmoid(self) }
    public func pow(_ y: Float) -> Tensor { DGenLazy.pow(self, y) }

    /// Clamp values to range [minVal, maxVal]
    public func clip(_ minVal: Double, _ maxVal: Double) -> Tensor {
        return DGenLazy.max(DGenLazy.min(self, maxVal), minVal)
    }

    /// Clamp values to range [minVal, maxVal]
    public func clip(_ minVal: Tensor, _ maxVal: Tensor) -> Tensor {
        return DGenLazy.max(DGenLazy.min(self, maxVal), minVal)
    }
}

extension Signal {
    public func abs() -> Signal { DGenLazy.abs(self) }
    public func exp() -> Signal { DGenLazy.exp(self) }
    public func log() -> Signal { DGenLazy.log(self) }
    public func sqrt() -> Signal { DGenLazy.sqrt(self) }
    public func sin() -> Signal { DGenLazy.sin(self) }
    public func cos() -> Signal { DGenLazy.cos(self) }
    public func tanh() -> Signal { DGenLazy.tanh(self) }
    public func relu() -> Signal { DGenLazy.relu(self) }
    public func sigmoid() -> Signal { DGenLazy.sigmoid(self) }
    public func pow(_ y: Float) -> Signal { DGenLazy.pow(self, y) }

    /// Clamp value to range [minVal, maxVal]
    public func clip(_ minVal: Double, _ maxVal: Double) -> Signal {
        return DGenLazy.max(DGenLazy.min(self, maxVal), minVal)
    }

    /// Clamp value to range [minVal, maxVal]
    public func clip(_ minVal: Signal, _ maxVal: Signal) -> Signal {
        return DGenLazy.max(DGenLazy.min(self, maxVal), minVal)
    }
}

extension SignalTensor {
    public func sin() -> SignalTensor { DGenLazy.sin(self) }
    public func cos() -> SignalTensor { DGenLazy.cos(self) }
    public func exp() -> SignalTensor { DGenLazy.exp(self) }
    public func tanh() -> SignalTensor { DGenLazy.tanh(self) }
    public func relu() -> SignalTensor { DGenLazy.relu(self) }
}

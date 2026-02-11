// Functions - Math functions for lazy types
//
// Global functions like sin, cos, relu, etc. that work with
// Tensor, Signal, and SignalTensor.

import DGen
import Foundation

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
  return Tensor(
    nodeId: nodeId, graph: x.graph, shape: outShape, requiresGrad: x.requiresGrad || y.requiresGrad)
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
  return Tensor(
    nodeId: nodeId, graph: a.graph, shape: outShape, requiresGrad: a.requiresGrad || b.requiresGrad)
}

public func min(_ a: Tensor, _ b: Double) -> Tensor {
  let bNode = a.graph.node(.constant(Float(b)))
  let nodeId = a.graph.node(.min, [a.nodeId, bNode])
  return Tensor(nodeId: nodeId, graph: a.graph, shape: a.shape, requiresGrad: a.requiresGrad)
}

public func max(_ a: Tensor, _ b: Tensor) -> Tensor {
  let nodeId = a.graph.node(.max, [a.nodeId, b.nodeId])
  let outShape = broadcastShape(a.shape, b.shape)
  return Tensor(
    nodeId: nodeId, graph: a.graph, shape: outShape, requiresGrad: a.requiresGrad || b.requiresGrad)
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
  return Signal(
    nodeId: nodeId, graph: cond.graph, requiresGrad: cond.requiresGrad || a.requiresGrad)
}

public func gswitch(_ cond: Signal, _ a: Double, _ b: Signal) -> Signal {
  let aNode = cond.graph.node(.constant(Float(a)))
  let nodeId = cond.graph.node(.gswitch, [cond.nodeId, aNode, b.nodeId])
  return Signal(
    nodeId: nodeId, graph: cond.graph, requiresGrad: cond.requiresGrad || b.requiresGrad)
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
  return Tensor(
    nodeId: nodeId, graph: cond.graph, shape: outShape,
    requiresGrad: cond.requiresGrad || a.requiresGrad)
}

public func gswitch(_ cond: Tensor, _ a: Double, _ b: Tensor) -> Tensor {
  let aNode = cond.graph.node(.constant(Float(a)))
  let nodeId = cond.graph.node(.gswitch, [cond.nodeId, aNode, b.nodeId])
  let outShape = broadcastShape(cond.shape, b.shape)
  return Tensor(
    nodeId: nodeId, graph: cond.graph, shape: outShape,
    requiresGrad: cond.requiresGrad || b.requiresGrad)
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

  /// Sum along a specific axis, reducing that dimension
  ///
  /// ```swift
  /// let t = Tensor(shape: [2, 3, 4], ...)
  /// let s = t.sum(axis: -1)  // [2, 3] - sum over last dim
  /// let s2 = t.sum(axis: 0)  // [3, 4] - sum over first dim
  /// ```
  public func sum(axis: Int) -> Tensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("sum(axis:) axis \(axis) out of range for \(ndim)D tensor")
    }

    // Output shape: remove the reduced axis
    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      // Reducing to scalar - use full sum
      return sum()
    }

    let nodeId = try! graph.graph.sum(self.nodeId, axis: normalizedAxis)
    return Tensor(nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
  }

  /// Mean of all elements
  public func mean() -> Tensor {
    let sumNode = graph.node(.sum, [self.nodeId])
    let countNode = graph.node(.constant(Float(size)))
    let nodeId = graph.node(.div, [sumNode, countNode])
    return Tensor(nodeId: nodeId, graph: graph, shape: [1], requiresGrad: requiresGrad)
  }

  /// Max along a specific axis, reducing that dimension
  public func max(axis: Int) -> Tensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("max(axis:) axis \(axis) out of range for \(ndim)D tensor")
    }

    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      fatalError("max(axis:) reducing to scalar not yet supported")
    }

    let nodeId = try! graph.graph.max(self.nodeId, axis: normalizedAxis)
    return Tensor(nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
  }

  /// Mean along a specific axis, reducing that dimension
  public func mean(axis: Int) -> Tensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("mean(axis:) axis \(axis) out of range for \(ndim)D tensor")
    }

    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      fatalError("mean(axis:) reducing to scalar not yet supported")
    }

    let nodeId = try! graph.graph.mean(self.nodeId, axis: normalizedAxis)
    return Tensor(nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
  }

  /// Softmax along a specific axis: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
  /// Composed from primitives — no new LazyOp needed.
  public func softmax(axis: Int) -> Tensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("softmax(axis:) axis \(axis) out of range for \(ndim)D tensor")
    }

    // max for numerical stability
    let m = self.max(axis: normalizedAxis)
    // Insert size-1 dim at the reduced axis for broadcasting
    var broadcastShape = m.shape
    broadcastShape.insert(1, at: normalizedAxis)
    let mBroadcast = m.reshape(broadcastShape).expand(shape)

    let shifted = self - mBroadcast
    let e = shifted.exp()

    let s = e.sum(axis: normalizedAxis)
    var sBroadcastShape = s.shape
    sBroadcastShape.insert(1, at: normalizedAxis)
    let sBroadcast = s.reshape(sBroadcastShape).expand(shape)

    return e / sBroadcast
  }
}

extension Tensor {
  /// Matrix multiply: A[M,K] @ B[K,N] -> C[M,N]
  ///
  /// ```swift
  /// let a = Tensor([[1, 2], [3, 4]])           // [2, 2]
  /// let b = Tensor([[5, 6], [7, 8]])           // [2, 2]
  /// let c = a.matmul(b)                        // [2, 2]
  /// ```
  public func matmul(_ other: Tensor) -> Tensor {
    guard shape.count == 2, other.shape.count == 2 else {
      fatalError("matmul requires 2D tensors")
    }
    let M = shape[0]
    let N = other.shape[1]
    let nodeId = try! graph.graph.matmul(self.nodeId, other.nodeId)
    return Tensor(
      nodeId: nodeId, graph: graph, shape: [M, N], requiresGrad: requiresGrad || other.requiresGrad)
  }
}

/// Matrix multiply infix operator
/// Equivalent to a.matmul(b)
infix operator ◦ : MultiplicationPrecedence

public func ◦ (lhs: Tensor, rhs: Tensor) -> Tensor {
  return lhs.matmul(rhs)
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

// MARK: - Tensor FFT / IFFT (Pure Tensor Ops)

/// Compute N-point FFT using only tensor view + arithmetic operations.
/// N must be a power of 2. Returns (real, imaginary) tensors of shape [N].
public func tensorFFT(_ input: Tensor, N: Int) -> (re: Tensor, im: Tensor) {
  let (reId, imId) = input.graph.graph.tensorFFT(input.nodeId, N: N)
  return (
    Tensor(nodeId: reId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad),
    Tensor(nodeId: imId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad)
  )
}

/// Compute N-point IFFT using only tensor view + arithmetic operations.
/// Same Cooley-Tukey butterfly as tensorFFT but with positive twiddle angles and 1/N normalization.
/// N must be a power of 2. Returns real tensor of shape [N] (imaginary part discarded for real signals).
public func tensorIFFT(_ re: Tensor, _ im: Tensor, N: Int) -> Tensor {
  let resultId = re.graph.graph.tensorIFFT(re.nodeId, im.nodeId, N: N)
  return Tensor(
    nodeId: resultId, graph: re.graph, shape: [N],
    requiresGrad: re.requiresGrad || im.requiresGrad)
}

/// SignalTensor FFT variant: same algorithm, but input is a frame-varying tensor (usually buffered and hopped).
/// Twiddle factors stay as static Tensor; mixed arithmetic promotes to SignalTensor.
public func signalTensorFFT(_ input: SignalTensor, N: Int) -> (re: SignalTensor, im: SignalTensor) {
  let (reId, imId) = input.graph.graph.tensorFFT(input.nodeId, N: N)
  return (
    SignalTensor(nodeId: reId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad),
    SignalTensor(nodeId: imId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad)
  )
}

/// SignalTensor IFFT variant: positive twiddle angles, 1/N normalization.
public func signalTensorIFFT(_ re: SignalTensor, _ im: SignalTensor, N: Int) -> SignalTensor {
  let resultId = re.graph.graph.tensorIFFT(re.nodeId, im.nodeId, N: N)
  return SignalTensor(
    nodeId: resultId, graph: re.graph, shape: [N],
    requiresGrad: re.requiresGrad || im.requiresGrad)
}

// MARK: - Overlap-Add

extension SignalTensor {
  /// Overlap-add: scatter-add this tensor window into a ring buffer, emit one sample per frame.
  /// Converts a hop-rate tensor (e.g., IFFT output) back to a frame-rate scalar signal.
  ///
  /// - Parameter hop: How many frames between scatter-adds
  /// - Returns: Signal (one sample per frame)
  public func overlapAdd(hop: Int) -> Signal {
    let windowSize = shape.reduce(1, *)
    let nodeId = graph.graph.overlapAdd(self.nodeId, windowSize: windowSize, hopSize: hop)
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: requiresGrad)
  }
}

// MARK: - Higher-Order Operations

/// Spectral loss using FFT - compares frequency content of two signals
///
/// Computes the sum of squared magnitude differences in the frequency domain.
/// Uses Cooley-Tukey FFT for efficient computation and IFFT for gradient backpropagation.
///
/// ```swift
/// let student = sin(Signal.phasor(learnedFreq) * 2 * .pi)
/// let teacher = sin(Signal.phasor(440) * 2 * .pi)
/// let loss = spectralLossFFT(student, teacher, windowSize: 64)
/// ```
///
/// - Parameters:
///   - sig1: First signal (typically the student/predicted signal)
///   - sig2: Second signal (typically the teacher/target signal)
///   - windowSize: FFT window size (must be power of 2)
///   - useHannWindow: Whether to apply Hann window before FFT (default: true)
/// - Returns: Scalar loss signal (per frame)
public func spectralLossFFT(
  _ sig1: Signal,
  _ sig2: Signal,
  windowSize: Int,
  useHannWindow: Bool = true,
  normalize: Bool = false
) -> Signal {
  let nodeId = sig1.graph.graph.spectralLossFFT(
    sig1.nodeId,
    sig2.nodeId,
    windowSize: windowSize,
    useHannWindow: useHannWindow
  )
  let loss = Signal(
    nodeId: nodeId, graph: sig1.graph, requiresGrad: sig1.requiresGrad || sig2.requiresGrad)
  if normalize {
    let n = Float(sig1.graph.graph.maxFrameCount)
    return loss / Signal.constant(n)
  }
  return loss
}

/// Read a row from a 2D tensor with linear interpolation (frame-based)
///
/// Reads a row from a tensor where the row index can change per frame.
/// Fractional indices interpolate between adjacent rows.
///
/// ```swift
/// let amplitudes = Tensor(shape: [16, 4], ...)  // [controlFrames, numHarmonics]
/// let playhead = Signal.phasor(controlRate) * 15  // 0..15 over time
/// let currentAmps = amplitudes.peekRow(playhead)  // [4] per frame
/// ```
///
/// - Parameters:
///   - tensor: 2D tensor [numRows, numCols]
///   - rowIndex: Scalar signal for row selection (fractional values interpolate)
/// - Returns: SignalTensor [numCols] containing the interpolated row per frame
extension Tensor {
  public func peekRow(_ rowIndex: Signal) -> SignalTensor {
    guard shape.count == 2 else {
      fatalError("peekRow requires 2D tensor, got shape \(shape)")
    }
    let numCols = shape[1]
    let nodeId = try! graph.graph.peekRow(tensor: self.nodeId, rowIndex: rowIndex.nodeId)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: [numCols],
      requiresGrad: requiresGrad || rowIndex.requiresGrad)
  }
}

/// Read a scalar from a tensor at (index, channel) with interpolation
/// For 1D tensors, channel is ignored (auto-promoted to [N, 1])
extension Tensor {
  public func peek(_ index: Signal, channel: Signal? = nil) -> Signal {
    let ch = channel ?? Signal.constant(0.0)
    let nodeId = try! graph.graph.peek(tensor: self.nodeId, index: index.nodeId, channel: ch.nodeId)
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: requiresGrad || index.requiresGrad)
  }

  /// Convert a 1D tensor to a Signal: frame[i] reads tensor[i].
  /// Convenience over peek + accumulator.
  public func toSignal(maxFrames: Int? = nil) -> Signal {
    let len = Float(maxFrames ?? shape[0])
    let counter = Signal.accum(Signal.constant(1.0), reset: 0.0, min: 0.0, max: len)
    return peek(counter)
  }
}

/// Read a row from a 2D SignalTensor with linear interpolation
extension SignalTensor {
  public func peekRow(_ rowIndex: Signal) -> SignalTensor {
    guard shape.count == 2 else {
      fatalError("peekRow requires 2D tensor, got shape \(shape)")
    }
    let numCols = shape[1]
    let nodeId = try! graph.graph.peekRow(tensor: self.nodeId, rowIndex: rowIndex.nodeId)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: [numCols],
      requiresGrad: requiresGrad || rowIndex.requiresGrad)
  }
}

// MARK: - Audio Effects (Signal)

extension Signal {
  /// Biquad filter with multiple filter types.
  ///
  /// ```swift
  /// let audio = Signal.input()
  /// let filtered = audio.biquad(cutoff: 1000, resonance: 1.0, gain: 1.0, mode: 0)
  /// ```
  ///
  /// - Parameters:
  ///   - cutoff: Cutoff/center frequency in Hz
  ///   - resonance: Q factor / resonance
  ///   - gain: Output gain (linear). For shelf filters, controls shelf amount.
  ///   - mode: Filter type: 0=LP, 1=HP, 2=BP(skirt), 3=BP(peak), 4=AP, 5=Notch, 6=HiShelf, 7=LoShelf
  /// - Returns: Filtered signal
  public func biquad(cutoff: Signal, resonance: Signal, gain: Signal, mode: Signal) -> Signal {
    let nodeId = graph.graph.biquad(
      self.nodeId, cutoff.nodeId, resonance.nodeId, gain.nodeId, mode.nodeId)
    let needsGrad = requiresGrad || cutoff.requiresGrad || resonance.requiresGrad
      || gain.requiresGrad || mode.requiresGrad
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad)
  }

  /// Biquad filter with Float parameters
  public func biquad(cutoff: Float, resonance: Float, gain: Float, mode: Int) -> Signal {
    return biquad(
      cutoff: Signal.constant(cutoff),
      resonance: Signal.constant(resonance),
      gain: Signal.constant(gain),
      mode: Signal.constant(Float(mode)))
  }

  /// Compressor with sidechain support.
  ///
  /// ```swift
  /// let audio = Signal.input()
  /// let compressed = audio.compressor(
  ///   ratio: 4.0, threshold: -20.0, knee: 6.0,
  ///   attack: 0.005, release: 0.1)
  /// ```
  ///
  /// - Parameters:
  ///   - ratio: Compression ratio (e.g. 4.0 = 4:1)
  ///   - threshold: Threshold in dB
  ///   - knee: Knee width in dB
  ///   - attack: Attack time in seconds
  ///   - release: Release time in seconds
  ///   - sidechain: Optional sidechain input signal. When provided, the compressor
  ///                uses this signal for level detection instead of the main input.
  /// - Returns: Compressed signal
  public func compressor(
    ratio: Signal, threshold: Signal, knee: Signal,
    attack: Signal, release: Signal,
    sidechain: Signal? = nil
  ) -> Signal {
    let isSideChain = graph.node(.constant(sidechain != nil ? 1.0 : 0.0))
    let sidechainNode = sidechain?.nodeId ?? graph.node(.constant(0.0))
    let nodeId = graph.graph.compressor(
      self.nodeId, ratio.nodeId, threshold.nodeId, knee.nodeId,
      attack.nodeId, release.nodeId, isSideChain, sidechainNode)
    let needsGrad = requiresGrad || ratio.requiresGrad || threshold.requiresGrad
      || knee.requiresGrad || attack.requiresGrad || release.requiresGrad
      || (sidechain?.requiresGrad ?? false)
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad)
  }

  /// Compressor with Float parameters
  public func compressor(
    ratio: Float, threshold: Float, knee: Float,
    attack: Float, release: Float
  ) -> Signal {
    return compressor(
      ratio: Signal.constant(ratio),
      threshold: Signal.constant(threshold),
      knee: Signal.constant(knee),
      attack: Signal.constant(attack),
      release: Signal.constant(release))
  }

  /// Delay with linear interpolation for fractional delay times.
  ///
  /// ```swift
  /// let audio = Signal.input()
  /// let delayed = audio.delay(Signal.constant(4410))  // 100ms at 44.1kHz
  /// ```
  ///
  /// - Parameter delayTimeInSamples: Delay time in samples (0 to 88000)
  /// - Returns: Delayed signal
  public func delay(_ delayTimeInSamples: Signal) -> Signal {
    let nodeId = graph.graph.delay(self.nodeId, delayTimeInSamples.nodeId)
    let needsGrad = requiresGrad || delayTimeInSamples.requiresGrad
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: needsGrad)
  }

  /// Delay with Float parameter
  public func delay(_ delayTimeInSamples: Float) -> Signal {
    return delay(Signal.constant(delayTimeInSamples))
  }
}

// MARK: - Audio Effects (SignalTensor)

extension SignalTensor {
  /// Biquad filter applied element-wise to each signal in the tensor.
  ///
  /// Each element gets its own filter state (history cells).
  /// Cutoff, resonance, gain, and mode are broadcast scalars.
  ///
  /// - Parameters:
  ///   - cutoff: Cutoff/center frequency in Hz
  ///   - resonance: Q factor / resonance
  ///   - gain: Output gain (linear)
  ///   - mode: Filter type: 0=LP, 1=HP, 2=BP(skirt), 3=BP(peak), 4=AP, 5=Notch, 6=HiShelf, 7=LoShelf
  /// - Returns: Filtered signal tensor (same shape)
  public func biquad(cutoff: Signal, resonance: Signal, gain: Signal, mode: Signal) -> SignalTensor {
    let nodeId = graph.graph.biquad(
      self.nodeId, cutoff.nodeId, resonance.nodeId, gain.nodeId, mode.nodeId)
    let needsGrad = requiresGrad || cutoff.requiresGrad || resonance.requiresGrad
      || gain.requiresGrad || mode.requiresGrad
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: needsGrad)
  }

  /// Biquad filter with Float parameters
  public func biquad(cutoff: Float, resonance: Float, gain: Float, mode: Int) -> SignalTensor {
    return biquad(
      cutoff: Signal.constant(cutoff),
      resonance: Signal.constant(resonance),
      gain: Signal.constant(gain),
      mode: Signal.constant(Float(mode)))
  }

  /// Compressor applied element-wise to each signal in the tensor.
  ///
  /// - Parameters:
  ///   - ratio: Compression ratio
  ///   - threshold: Threshold in dB
  ///   - knee: Knee width in dB
  ///   - attack: Attack time in seconds
  ///   - release: Release time in seconds
  ///   - sidechain: Optional sidechain input signal
  /// - Returns: Compressed signal tensor (same shape)
  public func compressor(
    ratio: Signal, threshold: Signal, knee: Signal,
    attack: Signal, release: Signal,
    sidechain: Signal? = nil
  ) -> SignalTensor {
    let isSideChain = graph.node(.constant(sidechain != nil ? 1.0 : 0.0))
    let sidechainNode = sidechain?.nodeId ?? graph.node(.constant(0.0))
    let nodeId = graph.graph.compressor(
      self.nodeId, ratio.nodeId, threshold.nodeId, knee.nodeId,
      attack.nodeId, release.nodeId, isSideChain, sidechainNode)
    let needsGrad = requiresGrad || ratio.requiresGrad || threshold.requiresGrad
      || knee.requiresGrad || attack.requiresGrad || release.requiresGrad
      || (sidechain?.requiresGrad ?? false)
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: needsGrad)
  }

  /// Compressor with Float parameters
  public func compressor(
    ratio: Float, threshold: Float, knee: Float,
    attack: Float, release: Float
  ) -> SignalTensor {
    return compressor(
      ratio: Signal.constant(ratio),
      threshold: Signal.constant(threshold),
      knee: Signal.constant(knee),
      attack: Signal.constant(attack),
      release: Signal.constant(release))
  }

  /// Delay applied element-wise to each signal in the tensor.
  ///
  /// - Parameter delayTimeInSamples: Delay time in samples (0 to 88000)
  /// - Returns: Delayed signal tensor (same shape)
  public func delay(_ delayTimeInSamples: Signal) -> SignalTensor {
    let nodeId = graph.graph.delay(self.nodeId, delayTimeInSamples.nodeId)
    let needsGrad = requiresGrad || delayTimeInSamples.requiresGrad
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: needsGrad)
  }

  /// Delay with Float parameter
  public func delay(_ delayTimeInSamples: Float) -> SignalTensor {
    return delay(Signal.constant(delayTimeInSamples))
  }
}

// MARK: - SignalTensor Reductions

extension SignalTensor {
  /// Sum along a specific axis
  /// - Parameter axis: The axis to reduce (supports negative indexing)
  /// - Returns: SignalTensor with that dimension removed
  ///
  /// ```swift
  /// let windows = state.conv2d([2, 2])         // [outH, outW, kH, kW]
  /// let sumKW = (windows * kernel).sum(axis: -1)  // [outH, outW, kH]
  /// let sumKH = sumKW.sum(axis: -1)               // [outH, outW]
  /// ```
  public func sum(axis: Int) -> SignalTensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis

    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("Axis \(axis) out of bounds for shape \(shape)")
    }

    // If reducing to scalar, use the existing sum() -> Signal
    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      // This would return a Signal, but we need SignalTensor
      // Use sum over the single remaining dimension
      let nodeId = try! graph.graph.sum(self.nodeId, axis: normalizedAxis)
      return SignalTensor(nodeId: nodeId, graph: graph, shape: [1], requiresGrad: requiresGrad)
    }

    let nodeId = try! graph.graph.sum(self.nodeId, axis: normalizedAxis)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
  }

  /// Max along a specific axis
  public func max(axis: Int) -> SignalTensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("Axis \(axis) out of bounds for shape \(shape)")
    }

    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      fatalError("max(axis:) reducing to scalar not yet supported for SignalTensor")
    }

    let nodeId = try! graph.graph.max(self.nodeId, axis: normalizedAxis)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
  }

  /// Mean along a specific axis
  public func mean(axis: Int) -> SignalTensor {
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      fatalError("Axis \(axis) out of bounds for shape \(shape)")
    }

    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      fatalError("mean(axis:) reducing to scalar not yet supported for SignalTensor")
    }

    let nodeId = try! graph.graph.mean(self.nodeId, axis: normalizedAxis)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
  }
}

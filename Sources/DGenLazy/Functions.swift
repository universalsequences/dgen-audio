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
}

// MARK: - Tensor View Operations

extension Tensor {
    /// Reshape tensor to a new shape (must have same total size)
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2, 3]
    /// let flat = t.reshape([6])               // [6]
    /// let square = t.reshape([3, 2])          // [3, 2]
    /// ```
    public func reshape(_ newShape: Shape) -> Tensor {
        let nodeId = try! graph.graph.reshape(self.nodeId, to: newShape)
        return Tensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
    }

    /// Transpose tensor by permuting axes
    /// Default (nil) reverses all axes: [H, W] -> [W, H]
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2, 3]
    /// let tT = t.transpose()                  // [3, 2]
    /// let perm = t.transpose([1, 0])          // Same as above
    /// ```
    public func transpose(_ axes: [Int]? = nil) -> Tensor {
        let nodeId = try! graph.graph.transpose(self.nodeId, axes: axes)
        let newShape: Shape
        if let axes = axes {
            newShape = axes.map { shape[$0] }
        } else {
            newShape = shape.reversed()
        }
        return Tensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
    }

    /// Shrink/slice tensor along each axis
    /// ranges: for each dimension, (start, end) or nil to keep all
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  // [3, 3]
    /// let row = t.shrink([(0, 1), nil])                   // [1, 3] first row
    /// let col = t.shrink([nil, (1, 2)])                   // [3, 1] middle column
    /// let sub = t.shrink([(0, 2), (1, 3)])                // [2, 2] top-right 2x2
    /// ```
    public func shrink(_ ranges: [(Int, Int)?]) -> Tensor {
        let nodeId = try! graph.graph.shrink(self.nodeId, ranges: ranges)
        var newShape = [Int]()
        for (dim, range) in ranges.enumerated() {
            if let (start, end) = range {
                newShape.append(end - start)
            } else {
                newShape.append(shape[dim])
            }
        }
        return Tensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
    }

    /// Pad tensor with zeros along each axis
    /// padding: for each dimension, (left, right) padding amounts
    ///
    /// ```swift
    /// let t = Tensor([[1, 2], [3, 4]])       // [2, 2]
    /// let padded = t.pad([(1, 1), (1, 1)])   // [4, 4] with zeros around edges
    /// ```
    public func pad(_ padding: [(Int, Int)]) -> Tensor {
        let nodeId = try! graph.graph.pad(self.nodeId, padding: padding)
        let newShape = zip(shape, padding).map { dim, pad in
            dim + pad.0 + pad.1
        }
        return Tensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
    }

    /// Repeat/tile tensor along each dimension
    /// repeats: number of times to repeat along each axis
    ///
    /// ```swift
    /// let t = Tensor([[1, 2], [3, 4]])     // [2, 2]
    /// let tiled = t.repeat([2, 3])         // [4, 6] - 2x vertical, 3x horizontal
    /// ```
    public func `repeat`(_ repeats: [Int]) -> Tensor {
        let nodeId = try! graph.graph.repeatView(self.nodeId, repeats: repeats)
        let newShape = zip(shape, repeats).map { $0 * $1 }
        return Tensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
    }

    /// Expand size-1 dimensions to target shape (broadcasting)
    ///
    /// ```swift
    /// let t = Tensor([[1], [2], [3]])      // [3, 1]
    /// let expanded = t.expand([3, 4])      // [3, 4] - column broadcast to 4 cols
    /// ```
    public func expand(_ targetShape: Shape) -> Tensor {
        let nodeId = try! graph.graph.expandView(self.nodeId, to: targetShape)
        return Tensor(nodeId: nodeId, graph: graph, shape: targetShape, requiresGrad: requiresGrad)
    }

    /// Extract sliding windows (im2col) for convolution
    /// Transforms [H, W] -> [outH, outW, kH, kW] where each output position
    /// contains its corresponding input window
    ///
    /// ```swift
    /// let img = Tensor(shape: [5, 5], ...)     // 5x5 image
    /// let windows = img.windows([3, 3])        // [3, 3, 3, 3] - 3x3 output, 3x3 windows
    /// ```
    public func windows(_ kernelShape: Shape) -> Tensor {
        guard shape.count >= 2, kernelShape.count == 2 else {
            fatalError("windows requires 2D input and kernel shape")
        }
        let H = shape[shape.count - 2]
        let W = shape[shape.count - 1]
        let kH = kernelShape[0]
        let kW = kernelShape[1]
        let outH = H - kH + 1
        let outW = W - kW + 1

        // Use asStrided to create the im2col view
        let inputTensor = try! graph.graph.getTensor(self.nodeId)
        let inStrides = inputTensor.strides

        // Output strides: [outH, outW, kH, kW]
        // Moving in outH/outW moves by input strides
        // Moving in kH/kW also moves by input strides (within the window)
        let outStrides = [
            inStrides[inStrides.count - 2],  // outH stride = input row stride
            inStrides[inStrides.count - 1],  // outW stride = 1 (usually)
            inStrides[inStrides.count - 2],  // kH stride = input row stride
            inStrides[inStrides.count - 1],  // kW stride = 1 (usually)
        ]

        let outShape = [outH, outW, kH, kW]
        let nodeId = try! graph.graph.asStrided(self.nodeId, shape: outShape, strides: outStrides)
        return Tensor(nodeId: nodeId, graph: graph, shape: outShape, requiresGrad: requiresGrad)
    }

    /// 2D Convolution with a kernel tensor
    /// Convolves input [H, W] with kernel [kH, kW] to produce [outH, outW]
    ///
    /// ```swift
    /// let laplacian = Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    /// let membrane = state.conv2d(laplacian)   // [H-2, W-2] Laplacian result
    /// ```
    public func conv2d(_ kernel: Tensor) -> Tensor {
        guard shape.count == 2, kernel.shape.count == 2 else {
            fatalError("conv2d requires 2D input and 2D kernel tensor")
        }
        let nodeId = try! graph.graph.conv2dView(self.nodeId, kernel: kernel.nodeId)

        let H = shape[0]
        let W = shape[1]
        let kH = kernel.shape[0]
        let kW = kernel.shape[1]
        let outH = H - kH + 1
        let outW = W - kW + 1

        return Tensor(nodeId: nodeId, graph: graph, shape: [outH, outW], requiresGrad: requiresGrad || kernel.requiresGrad)
    }

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
        return Tensor(nodeId: nodeId, graph: graph, shape: [M, N], requiresGrad: requiresGrad || other.requiresGrad)
    }
}

/// Matrix multiply infix operator
/// Equivalent to a.matmul(b)
infix operator ◦: MultiplicationPrecedence

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
    let loss = Signal(nodeId: nodeId, graph: sig1.graph, requiresGrad: sig1.requiresGrad || sig2.requiresGrad)
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
        return SignalTensor(nodeId: nodeId, graph: graph, shape: [numCols], requiresGrad: requiresGrad || rowIndex.requiresGrad)
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
        return SignalTensor(nodeId: nodeId, graph: graph, shape: [numCols], requiresGrad: requiresGrad || rowIndex.requiresGrad)
    }
}

// MARK: - SignalTensor Views

extension SignalTensor {
    /// Extract sliding windows (im2col) for convolution
    /// Transforms [H, W] -> [outH, outW, kH, kW] where each output position
    /// contains its corresponding input window
    ///
    /// ```swift
    /// let state = history.read()                 // SignalTensor [5, 5]
    /// let windows = state.windows([3, 3])        // SignalTensor [3, 3, 3, 3]
    /// ```
    public func windows(_ kernelShape: Shape) -> SignalTensor {
        guard shape.count >= 2, kernelShape.count == 2 else {
            fatalError("windows requires 2D input and kernel shape")
        }
        let H = shape[shape.count - 2]
        let W = shape[shape.count - 1]
        let kH = kernelShape[0]
        let kW = kernelShape[1]
        let outH = H - kH + 1
        let outW = W - kW + 1

        // Use asStrided to create the im2col view
        let inputTensor = try! graph.graph.getTensor(self.nodeId)
        let inStrides = inputTensor.strides

        // Output strides: [outH, outW, kH, kW]
        // Moving in outH/outW moves by input strides
        // Moving in kH/kW also moves by input strides (within the window)
        let outStrides = [
            inStrides[inStrides.count - 2],  // outH stride = input row stride
            inStrides[inStrides.count - 1],  // outW stride = 1 (usually)
            inStrides[inStrides.count - 2],  // kH stride = input row stride
            inStrides[inStrides.count - 1],  // kW stride = 1 (usually)
        ]

        let outShape = [outH, outW, kH, kW]
        let nodeId = try! graph.graph.asStrided(self.nodeId, shape: outShape, strides: outStrides)
        return SignalTensor(nodeId: nodeId, graph: graph, shape: outShape, requiresGrad: requiresGrad)
    }

    /// 2D Convolution with a kernel tensor
    /// Convolves input [H, W] with kernel [kH, kW] to produce [outH, outW]
    /// Perfect for membrane simulations with Laplacian kernels!
    ///
    /// ```swift
    /// let laplacian = Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    /// let state = history.read()               // SignalTensor [H, W]
    /// let laplacianResult = state.conv2d(laplacian)  // SignalTensor [H-2, W-2]
    /// ```
    public func conv2d(_ kernel: Tensor) -> SignalTensor {
        guard shape.count == 2, kernel.shape.count == 2 else {
            fatalError("conv2d requires 2D input and 2D kernel tensor")
        }
        let nodeId = try! graph.graph.conv2dView(self.nodeId, kernel: kernel.nodeId)

        let H = shape[0]
        let W = shape[1]
        let kH = kernel.shape[0]
        let kW = kernel.shape[1]
        let outH = H - kH + 1
        let outW = W - kW + 1

        return SignalTensor(nodeId: nodeId, graph: graph, shape: [outH, outW], requiresGrad: requiresGrad || kernel.requiresGrad)
    }

    /// Pad tensor with zeros along each axis (same-size convolution helper)
    /// padding: for each dimension, (left, right) padding amounts
    ///
    /// Use this before conv2d to get same-size output (Dirichlet boundary conditions)
    ///
    /// ```swift
    /// let state = history.read()                     // SignalTensor [4, 4]
    /// let padded = state.pad([(1, 1), (1, 1)])       // SignalTensor [6, 6]
    /// let laplacian = padded.conv2d(kernel3x3)      // SignalTensor [4, 4] (same size!)
    /// ```
    public func pad(_ padding: [(Int, Int)]) -> SignalTensor {
        let nodeId = try! graph.graph.pad(self.nodeId, padding: padding)
        let newShape = zip(shape, padding).map { dim, pad in
            dim + pad.0 + pad.1
        }
        return SignalTensor(nodeId: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
    }

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
        return SignalTensor(nodeId: nodeId, graph: graph, shape: outputShape, requiresGrad: requiresGrad)
    }
}

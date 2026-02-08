// TensorOps - Shared view operations for Tensor and SignalTensor
//
// Protocol with default implementations so view ops (reshape, transpose,
// shrink, pad, expand, repeat, windows, conv2d) are written once.

import DGen

// MARK: - TensorOps Protocol

/// Protocol for types that support tensor view operations.
///
/// Conforming types get reshape, transpose, shrink, pad, expand, repeat,
/// windows, and conv2d for free — each returning `Self`.
public protocol TensorOps: LazyValue {
  var shape: Shape { get }

  /// View-only initializer used by default method implementations.
  init(_view nodeId: NodeID, graph: LazyGraph, shape: Shape, requiresGrad: Bool)
}

// MARK: - Default Implementations

extension TensorOps {

  /// Reshape tensor to a new shape (must have same total size)
  public func reshape(_ newShape: Shape) -> Self {
    let nodeId = try! graph.graph.reshape(self.nodeId, to: newShape)
    return Self(_view: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
  }

  /// Transpose tensor by permuting axes.
  /// Default (nil) reverses all axes.
  public func transpose(_ axes: [Int]? = nil) -> Self {
    let nodeId = try! graph.graph.transpose(self.nodeId, axes: axes)
    let newShape: Shape
    if let axes = axes {
      newShape = axes.map { shape[$0] }
    } else {
      newShape = shape.reversed()
    }
    return Self(_view: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
  }

  /// Shrink/slice tensor along each axis.
  /// ranges: for each dimension, (start, end) or nil to keep all.
  public func shrink(_ ranges: [(Int, Int)?]) -> Self {
    let nodeId = try! graph.graph.shrink(self.nodeId, ranges: ranges)
    var newShape = [Int]()
    for (dim, range) in ranges.enumerated() {
      if let (start, end) = range {
        newShape.append(end - start)
      } else {
        newShape.append(shape[dim])
      }
    }
    return Self(_view: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
  }

  /// Pad tensor with zeros along each axis.
  public func pad(_ padding: [(Int, Int)]) -> Self {
    let nodeId = try! graph.graph.pad(self.nodeId, padding: padding)
    let newShape = zip(shape, padding).map { dim, pad in
      dim + pad.0 + pad.1
    }
    return Self(_view: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
  }

  /// Repeat/tile tensor along each dimension.
  public func `repeat`(_ repeats: [Int]) -> Self {
    let nodeId = try! graph.graph.repeatView(self.nodeId, repeats: repeats)
    let newShape = zip(shape, repeats).map { $0 * $1 }
    return Self(_view: nodeId, graph: graph, shape: newShape, requiresGrad: requiresGrad)
  }

  /// Expand size-1 dimensions to target shape (broadcasting).
  public func expand(_ targetShape: Shape) -> Self {
    let nodeId = try! graph.graph.expandView(self.nodeId, to: targetShape)
    return Self(_view: nodeId, graph: graph, shape: targetShape, requiresGrad: requiresGrad)
  }

  /// Extract sliding windows (im2col) for convolution.
  /// Transforms [H, W] -> [outH, outW, kH, kW].
  public func windows(_ kernelShape: Shape) -> Self {
    guard shape.count >= 2, kernelShape.count == 2 else {
      fatalError("windows requires 2D input and kernel shape")
    }
    let H = shape[shape.count - 2]
    let W = shape[shape.count - 1]
    let kH = kernelShape[0]
    let kW = kernelShape[1]
    let outH = H - kH + 1
    let outW = W - kW + 1

    let inputTensor = try! graph.graph.getTensor(self.nodeId)
    let inStrides = inputTensor.strides

    let outStrides = [
      inStrides[inStrides.count - 2],
      inStrides[inStrides.count - 1],
      inStrides[inStrides.count - 2],
      inStrides[inStrides.count - 1],
    ]

    let outShape = [outH, outW, kH, kW]
    let nodeId = try! graph.graph.asStrided(self.nodeId, shape: outShape, strides: outStrides)
    return Self(_view: nodeId, graph: graph, shape: outShape, requiresGrad: requiresGrad)
  }

  /// 2D Convolution with a kernel tensor.
  /// Convolves input [H, W] with kernel [kH, kW] to produce [outH, outW].
  public func conv2d(_ kernel: Tensor) -> Self {
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

    return Self(
      _view: nodeId, graph: graph, shape: [outH, outW],
      requiresGrad: requiresGrad || kernel.requiresGrad)
  }
}

// MARK: - Conformances
// Tensor and SignalTensor declare `required init(_view:...)` in their class bodies.

extension Tensor: TensorOps {}
extension SignalTensor: TensorOps {}

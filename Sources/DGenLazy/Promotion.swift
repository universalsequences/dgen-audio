// Promotion.swift - lifting frame-invariant values into the frame-varying domain
//
// `SignalTensor` is the widest numeric domain in DGenLazy: a tensor whose
// elements vary per audio frame. Every other numeric value (a constant
// `Tensor`, a scalar `Signal`) is a special case of it, so any mixed-type
// expression can be evaluated by lifting the narrower operands and then using
// the SignalTensor⊗SignalTensor operators.
//
// The lifts below are *views*: they reuse the operand's existing graph node and
// only change the Swift-side wrapper. Shape inference in DGen broadcasts scalar
// nodes against tensor nodes, so a lifted scalar Signal combined with a real
// tensor produces a correctly tensor-shaped node.

import DGen

extension SignalTensor {
  /// View a frame-invariant `Tensor` as a `SignalTensor` (same graph node).
  public static func lift(_ t: Tensor) -> SignalTensor {
    SignalTensor(
      nodeId: t.nodeId, graph: t.graph, shape: t.shape,
      requiresGrad: t.requiresGrad, tensorId: t.tensorId)
  }

  /// View a scalar `Signal` as a `SignalTensor` broadcast over `shape`.
  ///
  /// The underlying node stays scalar-shaped; DGen broadcasts it elementwise
  /// against whatever tensor operand it is combined with. Only use this for
  /// operands of an elementwise op whose other side is genuinely tensor-shaped.
  public static func lift(_ s: Signal, shape: Shape) -> SignalTensor {
    SignalTensor(nodeId: s.nodeId, graph: s.graph, shape: shape, requiresGrad: s.requiresGrad)
  }
}

// MARK: - Unary math gaps

public func tan(_ x: SignalTensor) -> SignalTensor {
  let nodeId = x.graph.node(.tan, [x.nodeId])
  return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func sigmoid(_ x: SignalTensor) -> SignalTensor {
  // sigmoid(x) = 1 / (1 + exp(-x))
  let negX = x.graph.node(.mul, [x.nodeId, x.graph.node(.constant(-1.0))])
  let expNeg = x.graph.node(.exp, [negX])
  let onePlus = x.graph.node(.add, [x.graph.node(.constant(1.0)), expNeg])
  let nodeId = x.graph.node(.div, [x.graph.node(.constant(1.0)), onePlus])
  return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

// MARK: - Equality

extension SignalTensor {
  public func eq(_ other: SignalTensor) -> SignalTensor {
    let nodeId = graph.node(.eq, [self.nodeId, other.nodeId])
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: broadcastShape(shape, other.shape),
      requiresGrad: false)
  }
}

// MARK: - Conditional selection

/// Elementwise `cond > 0 ? a : b` over frame-varying tensors.
public func gswitch(_ cond: SignalTensor, _ a: SignalTensor, _ b: SignalTensor) -> SignalTensor {
  let nodeId = cond.graph.node(.gswitch, [cond.nodeId, a.nodeId, b.nodeId])
  return SignalTensor(
    nodeId: nodeId, graph: cond.graph,
    shape: broadcastShape(broadcastShape(cond.shape, a.shape), b.shape),
    requiresGrad: a.requiresGrad || b.requiresGrad)
}

/// Elementwise 1-based multi-way selection over frame-varying tensors.
public func selector(_ mode: SignalTensor, _ options: [SignalTensor]) -> SignalTensor {
  let nodeId = mode.graph.node(.selector, [mode.nodeId] + options.map(\.nodeId))
  let shape = options.reduce(mode.shape) { broadcastShape($0, $1.shape) }
  return SignalTensor(
    nodeId: nodeId, graph: mode.graph, shape: shape,
    requiresGrad: mode.requiresGrad || options.contains(where: \.requiresGrad))
}

// MARK: - Shaping helpers

extension SignalTensor {
  /// Clamp every element to `[minVal, maxVal]`.
  public func clip(_ minVal: SignalTensor, _ maxVal: SignalTensor) -> SignalTensor {
    DGenLazy.max(DGenLazy.min(self, maxVal), minVal)
  }

  /// Convert a per-element phasor ramp (0→1) to a triangle wave (0→1→0).
  ///
  /// Uses the same `Graph.triangle` composite as the scalar `Signal.triangle`;
  /// every primitive it emits (`lt`, `gswitch`, arithmetic) is elementwise and
  /// broadcasts, so the tensor path is the same math as the scalar path.
  public func triangle(duty: Signal? = nil) -> SignalTensor {
    graph.markDirty()
    let nodeId = graph.graph.triangle(self.nodeId, duty?.nodeId)
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: false)
  }

  /// Triangle wave with a per-element duty cycle.
  public func triangle(duty: SignalTensor) -> SignalTensor {
    graph.markDirty()
    let nodeId = graph.graph.triangle(self.nodeId, duty.nodeId)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: broadcastShape(shape, duty.shape),
      requiresGrad: false)
  }
}

// MARK: - Matrix multiply over frame-varying tensors

extension SignalTensor {
  /// Matrix multiply: A[M,K] @ B[K,N] -> C[M,N], evaluated per frame.
  ///
  /// `Graph.matmul` lowers to reshape/transpose views plus a broadcast multiply
  /// and an axis sum — all shape-generic and frame-safe — so a frame-varying
  /// operand needs no new codegen.
  public func matmul(_ other: SignalTensor) throws -> SignalTensor {
    graph.markDirty()
    let nodeId = try graph.graph.matmul(self.nodeId, other.nodeId)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: [shape[0], other.shape[1]],
      requiresGrad: requiresGrad || other.requiresGrad)
  }

  public func matmul(_ other: Tensor) throws -> SignalTensor {
    try matmul(SignalTensor.lift(other))
  }
}

extension Tensor {
  public func matmul(_ other: SignalTensor) throws -> SignalTensor {
    try SignalTensor.lift(self).matmul(other)
  }
}

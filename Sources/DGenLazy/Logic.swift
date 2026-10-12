import DGen

// MARK: - Logical operators
//
// Element-wise boolean logic over the 0.0/1.0 convention the comparison
// operators produce: any nonzero input counts as true, and the result is 1.0
// for true and 0.0 for false. Like the comparisons, these are not
// differentiable.

/// Logical AND: 1.0 iff both inputs are nonzero.
public func logicalAnd(_ a: Signal, _ b: Signal) -> Signal { logical(.and, a, b) }
/// Logical OR: 1.0 iff either input is nonzero.
public func logicalOr(_ a: Signal, _ b: Signal) -> Signal { logical(.or, a, b) }
/// Logical XOR: 1.0 iff exactly one input is nonzero.
public func logicalXor(_ a: Signal, _ b: Signal) -> Signal { logical(.xor, a, b) }

public func logicalAnd(_ a: Tensor, _ b: Tensor) -> Tensor { logical(.and, a, b) }
public func logicalOr(_ a: Tensor, _ b: Tensor) -> Tensor { logical(.or, a, b) }
public func logicalXor(_ a: Tensor, _ b: Tensor) -> Tensor { logical(.xor, a, b) }

public func logicalAnd(_ a: Tensor, _ b: Float) -> Tensor { logical(.and, a, b) }
public func logicalOr(_ a: Tensor, _ b: Float) -> Tensor { logical(.or, a, b) }
public func logicalXor(_ a: Tensor, _ b: Float) -> Tensor { logical(.xor, a, b) }

public func logicalAnd(_ a: SignalTensor, _ b: SignalTensor) -> SignalTensor {
  logical(.and, a, b)
}
public func logicalOr(_ a: SignalTensor, _ b: SignalTensor) -> SignalTensor {
  logical(.or, a, b)
}
public func logicalXor(_ a: SignalTensor, _ b: SignalTensor) -> SignalTensor {
  logical(.xor, a, b)
}

private func logical(_ op: LazyOp, _ a: Signal, _ b: Signal) -> Signal {
  let nodeId = a.graph.node(op, [a.nodeId, b.nodeId])
  return Signal(nodeId: nodeId, graph: a.graph, requiresGrad: false)
}

private func logical(_ op: LazyOp, _ a: Tensor, _ b: Tensor) -> Tensor {
  let nodeId = a.graph.node(op, [a.nodeId, b.nodeId])
  return Tensor(
    nodeId: nodeId, graph: a.graph, shape: broadcastShape(a.shape, b.shape), requiresGrad: false)
}

/// A scalar operand stays a constant node rather than becoming a one-element
/// tensor, matching the comparison operators' Tensor/Float overloads.
private func logical(_ op: LazyOp, _ a: Tensor, _ b: Float) -> Tensor {
  let bNode = a.graph.node(.constant(b))
  let nodeId = a.graph.node(op, [a.nodeId, bNode])
  return Tensor(nodeId: nodeId, graph: a.graph, shape: a.shape, requiresGrad: false)
}

private func logical(_ op: LazyOp, _ a: SignalTensor, _ b: SignalTensor) -> SignalTensor {
  let nodeId = a.graph.node(op, [a.nodeId, b.nodeId])
  return SignalTensor(
    nodeId: nodeId, graph: a.graph, shape: broadcastShape(a.shape, b.shape), requiresGrad: false)
}

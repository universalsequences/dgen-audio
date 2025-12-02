/// NumPy-style broadcasting: computes the output shape when two shapes are broadcast together.
/// Returns nil if the shapes are not broadcastable.
/// Example: [2, 1, 3] + [1, 2, 3] -> [2, 2, 3]
public func broadcastShapes(_ s1: [Int], _ s2: [Int]) -> [Int]? {
    // Pad shorter shape with 1s on the left
    let maxLen = max(s1.count, s2.count)
    let padded1 = Array(repeating: 1, count: maxLen - s1.count) + s1
    let padded2 = Array(repeating: 1, count: maxLen - s2.count) + s2

    var result = [Int]()
    for i in 0..<maxLen {
        let d1 = padded1[i]
        let d2 = padded2[i]
        if d1 == d2 {
            result.append(d1)
        } else if d1 == 1 {
            result.append(d2)
        } else if d2 == 1 {
            result.append(d1)
        } else {
            // Incompatible dimensions
            return nil
        }
    }
    return result
}

public func inferShape(op: LazyOp, inputs: [ValueShape], graph: Graph) throws -> ValueShape {
  switch op {
  case .tensorRef(let tid):
    guard let tensor = graph.tensors[tid] else {
      throw DGenError.missingTensorID
    }
    return .tensor(tensor.shape)

  // History read - returns scalar or tensor shape depending on cell
  case .historyRead(let cellId):
    // O(1) lookup using cellToTensor mapping
    if let tensorId = graph.cellToTensor[cellId], let tensor = graph.tensors[tensorId] {
      return .tensor(tensor.shape)
    }
    // Scalar cell
    return .scalar

  // History write - output shape same as input (passthrough)
  case .historyWrite(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "historyWrite", reason: "missing input")
    }
    return firstInput

  // Conv2d - output shape matches input shape (same padding)
  case .conv2d(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "conv2d", reason: "missing input tensor")
    }
    return firstInput

  // Sum reduce - always outputs scalar
  case .sum:
    return .scalar

  // Sum along axis - reduces one dimension
  case .sumAxis(let axis):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "sumAxis", reason: "requires tensor input")
    }
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.shapeInferenceFailed(op: "sumAxis", reason: "axis \(axis) out of range for \(ndim)D tensor")
    }
    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      return .scalar
    }
    return .tensor(outputShape)

  // Reshape - changes shape, preserves total size
  case .reshape(let newShape):
    return .tensor(newShape)

  // Transpose - permutes axes
  case .transpose(let axes):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "transpose", reason: "requires tensor input")
    }
    let perm = axes.isEmpty ? Array((0..<shape.count).reversed()) : axes
    var newShape = [Int](repeating: 0, count: shape.count)
    for i in 0..<shape.count {
      newShape[i] = shape[perm[i]]
    }
    return .tensor(newShape)

  // Inherited (elementwise) - includes all binary and unary math ops
  case .add, .sub, .mul, .div, .sin, .cos, .exp, .sqrt, .tanh,
    .tan, .log, .log10, .abs, .sign, .floor, .ceil, .round,
    .pow, .mod, .min, .max, .atan2, .gt, .gte, .lt, .lte, .eq,
    .and, .or, .xor, .gswitch, .mix:
    let tensors = inputs.filter { x in
      if case .tensor(_) = x { return true }
      return false
    }
    if tensors.count == 2 {
      if case .tensor(let s1) = tensors[0], case .tensor(let s2) = tensors[1] {
        // Try NumPy-style broadcasting
        if let broadcastShape = broadcastShapes(s1, s2) {
          return .tensor(broadcastShape)
        } else {
          throw DGenError.shapeMismatch(op: "\(op)", shape1: s1, shape2: s2)
        }
      }
    }
    if tensors.count > 0 {
      return tensors[0]  // return the tensor as the shape
    }
    return .scalar
  // everything else is a scalar
  default: return .scalar
  }
}

public func inferShapes(graph: Graph, sortedNodes: [NodeID]) throws {
  for nodeId in sortedNodes {
    if var node = graph.nodes[nodeId] {
      node.shape = try inferShape(
        op: node.op, inputs: node.inputs.compactMap { graph.nodes[$0]?.shape }, graph: graph)
      graph.nodes[nodeId] = node
    }
  }
}

/// Allocate output tensors for nodes that produce tensor results.
/// This is called after shape inference, so all nodes have their shapes assigned.
/// Nodes that already have a tensor (via tensorRef) are skipped.
public func allocateTensorOutputs(graph: Graph, sortedNodes: [NodeID]) {
  for nodeId in sortedNodes {
    guard let node = graph.nodes[nodeId] else { continue }

    // Skip if this node already has a tensor (e.g., tensorRef nodes)
    if graph.nodeToTensor[nodeId] != nil { continue }

    // Only allocate for tensor-shaped outputs
    guard case .tensor(let shape) = node.shape else { continue }

    // Allocate memory for this tensor output
    let size = shape.reduce(1, *)
    let cellId = graph.alloc(vectorWidth: size)

    // Create tensor and register mapping
    let tensorId = graph.nextTensorId
    graph.nextTensorId += 1
    graph.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId)
    graph.nodeToTensor[nodeId] = tensorId
  }
}

// MARK: - Temporality Inference

/// Returns true if the op is intrinsically frame-based (produces different values each frame)
public func isIntrinsicallyFrameBased(_ op: LazyOp) -> Bool {
  switch op {
  case .phasor(_): return true  // oscillator state changes each frame
  case .output(_): return true  // oscillator state changes each frame
  case .accum(_): return true  // accumulator state changes each frame
  case .input(_): return true  // audio input varies each frame
  case .historyRead(_): return true  // reads temporal state (scalar or tensor)
  case .historyWrite(_): return true  // writes temporal state (scalar or tensor)
  case .historyReadWrite(_): return true  // combined temporal operation
  case .latch(_): return true  // conditional state update
  case .click(_): return true  // trigger/event based
  default: return false
  }
}

/// Infer temporality for all nodes. Returns the set of frame-based nodes.
/// Frame-based taint propagates: if any input is frame-based, output is frame-based.
public func inferTemporality(graph: Graph, sortedNodes: [NodeID]) -> Set<NodeID> {
  var frameBasedNodes = Set<NodeID>()

  for nodeId in sortedNodes {
    guard let node = graph.nodes[nodeId] else { continue }

    // Check if intrinsically frame-based
    if isIntrinsicallyFrameBased(node.op) {
      frameBasedNodes.insert(nodeId)
      continue
    }

    // Check if any input is frame-based (taint propagation)
    let hasFrameBasedInput = node.inputs.contains { frameBasedNodes.contains($0) }
    if hasFrameBasedInput {
      frameBasedNodes.insert(nodeId)
    }
  }

  return frameBasedNodes
}

/// Assign temporality to blocks based on their nodes.
/// A block is frame-based if ANY of its nodes is frame-based.
public func assignBlockTemporality(blocks: inout [Block], frameBasedNodes: Set<NodeID>) {
  for i in 0..<blocks.count {
    let hasFrameBasedNode = blocks[i].nodes.contains { frameBasedNodes.contains($0) }
    blocks[i].temporality = hasFrameBasedNode ? .frameBased : .static_
  }
}

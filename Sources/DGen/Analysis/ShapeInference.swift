import Foundation

/// Core shape-inference rules used by both graph building and compilation passes.
///
/// Keep this file free of pipeline-specific concerns so shape rules can be reused from
/// `Graph.n(...)` and from compilation pass orchestration.

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

  // Conv1d - output shape matches input shape (same padding)
  case .conv1d(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "conv1d", reason: "missing input tensor")
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
      throw DGenError.shapeInferenceFailed(
        op: "sumAxis", reason: "axis \(axis) out of range for \(ndim)D tensor")
    }
    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      return .scalar
    }
    return .tensor(outputShape)

  // Max along axis - reduces one dimension (same shape logic as sumAxis)
  case .maxAxis(let axis):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "maxAxis", reason: "requires tensor input")
    }
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.shapeInferenceFailed(
        op: "maxAxis", reason: "axis \(axis) out of range for \(ndim)D tensor")
    }
    var outputShape = shape
    outputShape.remove(at: normalizedAxis)
    if outputShape.isEmpty {
      return .scalar
    }
    return .tensor(outputShape)

  // Mean along axis - reduces one dimension (same shape logic as sumAxis)
  case .meanAxis(let axis):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "meanAxis", reason: "requires tensor input")
    }
    let ndim = shape.count
    let normalizedAxis = axis < 0 ? ndim + axis : axis
    guard normalizedAxis >= 0 && normalizedAxis < ndim else {
      throw DGenError.shapeInferenceFailed(
        op: "meanAxis", reason: "axis \(axis) out of range for \(ndim)D tensor")
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

  // AsStrided - view with custom strides (for pool/im2col)
  case .asStrided(let newShape, _):
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

  // Shrink - slices tensor along each axis
  case .shrink(let ranges):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "shrink", reason: "requires tensor input")
    }
    var newShape = [Int]()
    for (dim, range) in ranges.enumerated() {
      if let (start, end) = range {
        newShape.append(end - start)
      } else {
        newShape.append(shape[dim])
      }
    }
    return .tensor(newShape)

  // Pad - expands tensor with zeros along each axis
  case .pad(let padding):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "pad", reason: "requires tensor input")
    }
    let newShape = zip(shape, padding).map { dim, pad in
      dim + pad.0 + pad.1
    }
    return .tensor(newShape)

  case .expandView(let targetShape):
    // expandView broadcasts size-1 dims to target shape (stride=0 view)
    return .tensor(targetShape)

  case .repeatView(let repeats):
    // repeatView tiles tensor - output shape is input shape * repeats
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "repeatView", reason: "requires tensor input")
    }
    let newShape = zip(shape, repeats).map { $0 * $1 }
    return .tensor(newShape)

  // Peek - reads a scalar from a 2D tensor at (index, channel)
  case .peek:
    // Peek always outputs scalar - it reads one value from the tensor
    return .scalar

  // selectRow - extracts a single row from a 2D tensor using dynamic index
  case .selectRow:
    guard let firstInput = inputs.first,
      case .tensor(let shape) = firstInput,
      shape.count == 2
    else {
      throw DGenError.shapeInferenceFailed(op: "selectRow", reason: "requires 2D tensor input")
    }
    return .tensor([shape[1]])  // Output is [numCols]

  // peekRowInline - interpolated row extraction with frame-indexed storage
  case .peekRowInline(_, let numRows, let numCols):
    return .tensor([numCols])  // Output is [numCols]

  // FFT - outputs [numBins, 2] tensor where numBins = windowSize/2 + 1
  // overlapAdd - outputs scalar (one sample per frame via ring buffer)
  case .overlapAdd(_, _, _, _, _):
    return .scalar

  // overlapAdd gradient ops - side-effect only, output scalar
  case .overlapAddGradStore(_), .overlapAddGradGather(_, _, _, _):
    return .scalar

  // bufferView gradient ops - side-effect only, output scalar
  case .bufferViewGradStore(_, _), .bufferViewGradRead(_, _):
    return .scalar

  // Inherited (elementwise) - includes all binary and unary math ops
  // Also includes stateful ops (phasor, accum, latch) that can operate element-wise on tensors
  case .add, .sub, .mul, .div, .sin, .cos, .exp, .sqrt, .tanh,
    .tan, .log, .log10, .abs, .sign, .floor, .ceil, .round,
    .pow, .mod, .min, .max, .atan2, .gt, .gte, .lt, .lte, .eq,
    .and, .or, .xor, .gswitch, .mix,
    .phasor(_), .accum(_), .latch(_), .deterministicPhasor, .gradPhasor, .gradDeterministicPhasor:
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

  // Seq returns the shape of the last input (the value that's returned)
  case .seq:
    return inputs.last ?? .scalar

  // tensorAccumulate is a side-effect op (output is empty)
  case .tensorAccumulate(_):
    return .scalar

  // Gradient-specific operations
  case .neg:
    // Negation preserves shape
    return inputs.first ?? .scalar

  case .expand(let targetShape):
    // Expand broadcasts scalar to tensor
    return .tensor(targetShape)

  case .expandAxis(let targetShape, _):
    // ExpandAxis broadcasts along an axis
    return .tensor(targetShape)

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

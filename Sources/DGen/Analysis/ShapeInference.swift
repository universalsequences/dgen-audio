import Foundation

/// Core shape-inference rules used by both graph building and compilation passes.
///
/// Keep this file free of pipeline-specific concerns so shape rules can be reused from
/// `Graph.n(...)` and from compilation pass orchestration.

/// NumPy-style broadcasting: computes the output shape when two shapes are broadcast together.
/// Returns nil if the shapes are not broadcastable.
/// Example: [2, 1, 3] + [1, 2, 3] -> [2, 2, 3]
public func broadcastShapes(_ s1: [Int], _ s2: [Int]) -> [Int]? {
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
      return nil
    }
  }
  return result
}

/// Reduces one dimension from a tensor shape along the given axis.
/// Returns `.scalar` if the resulting shape is empty (1D tensor reduced to nothing).
private func inferAxisReduceShape(
  opName: String, axis: Int, inputs: [ValueShape]
) throws -> ValueShape {
  guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
    throw DGenError.shapeInferenceFailed(op: opName, reason: "requires tensor input")
  }
  let ndim = shape.count
  let normalizedAxis = axis < 0 ? ndim + axis : axis
  guard normalizedAxis >= 0 && normalizedAxis < ndim else {
    throw DGenError.shapeInferenceFailed(
      op: opName, reason: "axis \(axis) out of range for \(ndim)D tensor")
  }
  var outputShape = shape
  outputShape.remove(at: normalizedAxis)
  return outputShape.isEmpty ? .scalar : .tensor(outputShape)
}

public func inferShape(op: LazyOp, inputs: [ValueShape], graph: Graph) throws -> ValueShape {
  switch op {
  case .tensorRef(let tid):
    guard let tensor = graph.tensors[tid] else {
      throw DGenError.missingTensorID
    }
    return .tensor(tensor.shape)

  case .historyRead(let cellId):
    if let tensorId = graph.cellToTensor[cellId], let tensor = graph.tensors[tensorId] {
      return .tensor(tensor.shape)
    }
    return .scalar

  case .historyWrite(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "historyWrite", reason: "missing input")
    }
    return firstInput

  // Conv output shape matches input shape (same padding)
  case .conv1d(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "conv1d", reason: "missing input tensor")
    }
    return firstInput

  case .conv2d(_):
    guard let firstInput = inputs.first else {
      throw DGenError.shapeInferenceFailed(op: "conv2d", reason: "missing input tensor")
    }
    return firstInput

  case .sum:
    return .scalar

  // Axis reductions all share the same shape logic: remove the reduced dimension
  case .sumAxis(let axis):
    return try inferAxisReduceShape(opName: "sumAxis", axis: axis, inputs: inputs)

  case .sumMulAxis0:
    guard inputs.count == 2,
      case .tensor(let leftShape) = inputs[0],
      case .tensor(let rightShape) = inputs[1],
      leftShape.count == 2,
      rightShape == leftShape
    else {
      throw DGenError.shapeInferenceFailed(
        op: "sumMulAxis0", reason: "requires two matching 2D tensor inputs")
    }
    return .tensor([leftShape[1]])

  case .maxAxis(let axis):
    return try inferAxisReduceShape(opName: "maxAxis", axis: axis, inputs: inputs)

  case .meanAxis(let axis):
    return try inferAxisReduceShape(opName: "meanAxis", axis: axis, inputs: inputs)

  case .reshape(let newShape):
    return .tensor(newShape)

  case .asStrided(let newShape, _):
    return .tensor(newShape)

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

  case .pad(let padding):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "pad", reason: "requires tensor input")
    }
    let newShape = zip(shape, padding).map { dim, pad in
      dim + pad.0 + pad.1
    }
    return .tensor(newShape)

  case .expandView(let targetShape):
    return .tensor(targetShape)

  case .repeatView(let repeats):
    guard let firstInput = inputs.first, case .tensor(let shape) = firstInput else {
      throw DGenError.shapeInferenceFailed(op: "repeatView", reason: "requires tensor input")
    }
    let newShape = zip(shape, repeats).map { $0 * $1 }
    return .tensor(newShape)

  case .peek:
    return .scalar

  case .selectRow:
    guard let firstInput = inputs.first,
      case .tensor(let shape) = firstInput,
      shape.count == 2
    else {
      throw DGenError.shapeInferenceFailed(op: "selectRow", reason: "requires 2D tensor input")
    }
    return .tensor([shape[1]])

  case .sampleInline(_, _, let remainingShape):
    return .tensor(remainingShape)

  case .overlapAdd(_, _, _, _, _):
    return .scalar

  case .overlapAddGradStore(_), .overlapAddGradGather(_, _, _, _):
    return .scalar

  case .bufferViewGradStore(_, _), .bufferViewGradRead(_, _):
    return .scalar

  case .peekGradWrite(_, _, _, _, _, _, _), .peekGradReduce(_, _, _, _, _, _, _):
    return .scalar

  case .sampleGradWrite(_, _, _, _, _, _, _), .sampleGradReduce(_, _, _, _, _, _, _, _):
    return .scalar

  // Elementwise ops: broadcast tensor shapes, or inherit the single tensor shape, or scalar
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
        if let broadcastShape = broadcastShapes(s1, s2) {
          return .tensor(broadcastShape)
        } else {
          throw DGenError.shapeMismatch(op: "\(op)", shape1: s1, shape2: s2)
        }
      }
    }
    if tensors.count > 0 {
      return tensors[0]
    }
    return .scalar

  case .seq:
    return inputs.last ?? .scalar

  case .tensorAccumulate(_):
    return .scalar

  case .chunkPartialsReduceToCell:
    return .scalar

  case .neg:
    return inputs.first ?? .scalar

  case .expand(let targetShape):
    return .tensor(targetShape)

  case .expandAxis(let targetShape, _):
    return .tensor(targetShape)

  case .gemm(let M, let N, _, _, _):
    return .tensor([M, N])

  case .gemmSmall(let M, let N, _, _, _):
    return .tensor([M, N])

  case .gemmChunkPartials(let M, let N, _, _, _, _, let chunkCount):
    return .tensor([chunkCount, M, N])

  // Batched spectral loss: GradRead ops output [B] tensors
  case .spectralLossFFTBatchedGradRead(_, let batchSize, _, _, _):
    return .tensor([batchSize])
  case .spectralLossFFTBatchedGradRead2(_, let batchSize, _, _):
    return .tensor([batchSize])

  // Scalar ops -- listed explicitly so the compiler catches missing cases when new ops are added.
  case .mse,
    .spectralLossFFT, .spectralLossFFTGradSpec, .spectralLossFFTGradIFFT,
    .spectralLossFFTGradInline, .spectralLossFFTGradRead, .spectralLossFFTGradRead2,
    .spectralLossFFTBatched, .spectralLossFFTBatchedGradSpec, .spectralLossFFTBatchedGradIFFT,
    .selectRowGradWrite, .selectRowGradReduce,
    .selector,
    .memoryRead, .memoryWrite, .memoryAccumulate, .memoryCellSum,
    .historyReadWrite,
    .param, .click, .noise,
    .constant, .output, .input:
    return .scalar
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

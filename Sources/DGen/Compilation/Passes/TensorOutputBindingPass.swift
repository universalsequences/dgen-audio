import Foundation

/// Tensor output binding pass.
///
/// This pass runs after shape inference and binds node outputs to tensor records.
/// For non-view tensor producers it reserves lazy cell IDs; for view ops it creates
/// view tensors that share the source cell and append transforms.
enum TensorOutputBindingPass {}

extension TensorOutputBindingPass {
  /// Binds tensor outputs and reserves lazy cells for non-view tensor producers.
  static func bindTensorOutputsAndReserveLazyCells(graph: Graph, sortedNodes: [NodeID]) {
    for nodeId in sortedNodes {
      guard let node = graph.nodes[nodeId] else { continue }

      // Skip nodes that were already bound (e.g. tensorRef).
      if graph.nodeToTensor[nodeId] != nil {
        continue
      }

      guard case .tensor(let shape) = node.shape else { continue }

      expandStateCellIfNeeded(node: node, outputShape: shape, graph: graph)

      switch node.op {
      case .reshape(let newShape):
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: newShape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            .reshape(outputShape: newShape, inputShape: inputTensor.shape)
          })
        {
          continue
        }

      case .asStrided(let newShape, let newStrides):
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: newShape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            .asStrided(
              outputShape: newShape,
              strides: newStrides,
              offset: 0,
              inputShape: inputTensor.shape)
          })
        {
          continue
        }

      case .transpose(let axes):
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: shape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            let perm = axes.isEmpty ? Array((0..<inputTensor.shape.count).reversed()) : axes
            return .transpose(axes: perm, inputShape: inputTensor.shape)
          })
        {
          continue
        }

      case .shrink(let ranges):
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: shape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            let transformRanges: [(start: Int, end: Int)?] = ranges.map {
              if let (start, end) = $0 {
                return (start: start, end: end)
              }
              return nil
            }
            return .shrink(ranges: transformRanges, inputShape: inputTensor.shape)
          })
        {
          continue
        }

      case .pad(let padding):
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: shape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            .pad(
              padding: padding.map { (left: $0.0, right: $0.1) },
              inputShape: inputTensor.shape)
          })
        {
          continue
        }

      case .expandView(let targetShape):
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: shape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            .expand(targetShape: targetShape, inputShape: inputTensor.shape)
          })
        {
          continue
        }

      case .repeatView:
        if bindViewOutput(
          nodeId: nodeId,
          outputShape: shape,
          graph: graph,
          sourceNodeId: node.inputs.first,
          transformBuilder: { inputTensor in
            .repeatTile(innerShape: inputTensor.shape, outputShape: shape)
          })
        {
          continue
        }

      default:
        break
      }

      reserveLazyTensorOutput(nodeId: nodeId, shape: shape, graph: graph)
    }
  }

  /// Stateful scalar ops can become tensor-shaped; expand their state-cell allocation size.
  private static func expandStateCellIfNeeded(node: Node, outputShape: [Int], graph: Graph) {
    switch node.op {
    case .phasor(let originalCellId), .accum(let originalCellId), .latch(let originalCellId):
      let size = outputShape.reduce(1, *)
      if size > 1 {
        graph.cellAllocationSizes[originalCellId] = size
      }
    default:
      break
    }
  }

  /// Binds a view output tensor by appending a transform to the source tensor.
  @discardableResult
  private static func bindViewOutput(
    nodeId: NodeID,
    outputShape: [Int],
    graph: Graph,
    sourceNodeId: NodeID?,
    transformBuilder: (Tensor) -> ViewTransform
  ) -> Bool {
    guard
      let sourceNodeId,
      let inputTensorId = graph.nodeToTensor[sourceNodeId],
      let inputTensor = graph.tensors[inputTensorId]
    else {
      return false
    }

    var transforms = inputTensor.transforms
    transforms.append(transformBuilder(inputTensor))

    let tensorId = graph.nextTensorId
    graph.nextTensorId += 1
    graph.tensors[tensorId] = Tensor(
      id: tensorId,
      shape: outputShape,
      cellId: inputTensor.cellId,
      baseShape: inputTensor.baseShape,
      baseStrides: inputTensor.baseStrides,
      transforms: transforms
    )
    graph.nodeToTensor[nodeId] = tensorId
    return true
  }

  /// Reserves a lazy cell-backed tensor output for later temporality-aware materialization.
  private static func reserveLazyTensorOutput(nodeId: NodeID, shape: [Int], graph: Graph) {
    let lazyCellId = graph.reserveLazyCellId()

    let tensorId = graph.nextTensorId
    graph.nextTensorId += 1
    graph.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: lazyCellId, isLazy: true)
    graph.nodeToTensor[nodeId] = tensorId
  }
}

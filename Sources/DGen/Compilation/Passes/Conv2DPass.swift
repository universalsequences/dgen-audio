import Foundation

extension GraphPrepPasses {
  /// Annotates `.conv2d` nodes whose shape and kernel qualify for the SIMD-unrolled
  /// emission path. Analogous to `gemmPass` in structure: iterate candidate nodes,
  /// check eligibility, record metadata on `Graph`.
  ///
  /// Does NOT rewrite the graph — the conv2d node survives; only its emission differs.
  /// The actual NEON code comes from `emitOptimizedConv2D` which is gated by the
  /// annotation set.
  static func conv2dPass(graph: Graph) {
    for (nodeId, node) in graph.nodes {
      guard case .conv2d = node.op else { continue }
      guard isSIMDEligible(node: node, graph: graph) else { continue }

      // TODO(phase1-step3): allocate mask tensor and populate graph.conv2dMaskNodes.
      graph.simdOptimizedConv2Ds.insert(nodeId)
    }
  }

  /// Eligibility rules for the SIMD conv2d emission path:
  /// - 2D input with inW divisible by 4 and inW ≥ 4 (so 4-wide NEON loads align to rows)
  /// - Kernel is a compile-time constant tensor (its data is baked into `Tensor.data`)
  /// - Kernel size fits within the unroll cap (kH*kW ≤ 49)
  private static func isSIMDEligible(node: Node, graph: Graph) -> Bool {
    guard node.inputs.count >= 2 else { return false }
    guard case .conv2d(let kernelShape) = node.op, kernelShape.count == 2 else { return false }
    let (kH, kW) = (kernelShape[0], kernelShape[1])
    guard kH * kW <= 49 else { return false }

    guard let inputNode = graph.nodes[node.inputs[0]],
      case .tensor(let inShape) = inputNode.shape, inShape.count == 2
    else { return false }
    let inW = inShape[1]
    guard inW >= 4 && inW % 4 == 0 else { return false }

    // Kernel must be a compile-time constant (data baked in).
    guard let kTensor = graph.nodeToTensor[node.inputs[1]].flatMap({ graph.tensors[$0] }),
      kTensor.data != nil
    else { return false }

    return true
  }
}

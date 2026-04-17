import Foundation

/// Flip to `true` to log Conv2DPass eligibility decisions per node.
public enum DGenConv2DPassDebug {
  public static var enabled: Bool = false
}

extension GraphPrepPasses {
  /// Annotates `.conv2d` nodes whose shape and kernel qualify for the SIMD-unrolled
  /// emission path. Analogous to `gemmPass` in structure: iterate candidate nodes,
  /// check eligibility, record metadata on `Graph`.
  ///
  /// Does NOT rewrite the graph — the conv2d node survives; only its emission differs.
  /// The actual NEON code comes from `emitOptimizedConv2D` which is gated by the
  /// annotation set.
  static func conv2dPass(graph: Graph) {
    // 4-lane column masks for SIMD conv2d edge handling: left, full, right.
    // Concatenated into a single 12-float buffer indexed by 4×{0,1,2}.
    let maskData: [Float] = [
      0, 1, 1, 1,   // left edge  — lane 0 zero
      1, 1, 1, 1,   // fully in bounds
      1, 1, 1, 0,   // right edge — lane 3 zero
    ]

    for (nodeId, node) in graph.nodes {
      guard case .conv2d = node.op else { continue }
      let (eligible, reason) = eligibilityReport(node: node, graph: graph)
      if DGenConv2DPassDebug.enabled {
        print("[Conv2DPass] node=\(nodeId) eligible=\(eligible) — \(reason)")
      }
      guard eligible else { continue }

      let maskCellId = graph.alloc(vectorWidth: maskData.count)
      let maskTensorId = graph.nextTensorId
      graph.nextTensorId += 1
      graph.tensors[maskTensorId] = Tensor(
        id: maskTensorId, shape: [3, 4], cellId: maskCellId, data: maskData)
      graph.cellToTensor[maskCellId] = maskTensorId

      graph.simdOptimizedConv2Ds.insert(nodeId)
      graph.conv2dMaskCells[nodeId] = maskCellId
    }
  }

  /// Eligibility rules for the SIMD conv2d emission path:
  /// - 2D input with inW divisible by 4 and inW ≥ 4 (so 4-wide NEON loads align to rows)
  /// - Kernel is a tensor-backed input (cellId known); runtime-variable values
  ///   are fine — they're loaded lane-uniform via `simdBroadcastLoad`.
  /// - kW ≤ 3 — our edge masks (left [0,1,1,1] / right [1,1,1,0]) only cover
  ///   single-lane OOB. Wider kernels would put 2+ lanes OOB at the row edges
  ///   and need a richer mask table; deferred until a test exercises it.
  /// - kH unconstrained — row bounds are resolved at Swift emit time, so the
  ///   kernel can be any height.
  ///
  /// Returns `(eligible, humanReadableReason)` so the DGenConv2DPassDebug log
  /// can surface exactly which gate the caller tripped.
  private static func eligibilityReport(node: Node, graph: Graph) -> (Bool, String) {
    guard node.inputs.count >= 2 else { return (false, "fewer than 2 inputs") }
    guard case .conv2d(let kernelShape) = node.op, kernelShape.count == 2 else {
      return (false, "not a .conv2d([kH,kW]) op")
    }
    let (kH, kW) = (kernelShape[0], kernelShape[1])
    guard kW <= 3 else { return (false, "kW=\(kW) > 3 (mask table only covers 1-lane OOB)") }
    guard kH * kW <= 49 else { return (false, "kernel too large: \(kH)x\(kW)") }

    guard let inputNode = graph.nodes[node.inputs[0]] else {
      return (false, "input[0] node missing")
    }
    guard case .tensor(let inShape) = inputNode.shape else {
      return (false, "input[0] has no tensor shape (got \(inputNode.shape))")
    }
    guard inShape.count == 2 else {
      return (false, "input[0] shape rank \(inShape.count), need 2 (shape=\(inShape))")
    }
    let inW = inShape[1]
    guard inW >= 4 else { return (false, "inW=\(inW) < 4") }
    guard inW % 4 == 0 else { return (false, "inW=\(inW) not divisible by 4") }

    guard let kTensor = graph.nodeToTensor[node.inputs[1]].flatMap({ graph.tensors[$0] }) else {
      return (false, "kernel (input[1]) has no backing tensor")
    }
    let kernelKind = kTensor.data != nil ? "constant" : "runtime"
    return (true, "shape [\(inShape[0]),\(inW)] × [\(kH),\(kW)] kernel=\(kernelKind)")
  }

}

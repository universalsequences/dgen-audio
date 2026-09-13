// Emit+AcceleratedFFT.swift
//
// Emission for .acceleratedFFT / .acceleratedIFFT LazyOps. Both decompose into:
//   1. A loop that copies the input tensor(s) into owned re/im scratch cells
//   2. A single `.acceleratedFFTCall` UOp that the CRenderer renders as a call
//      to the runtime host FFT service
//   3. For IFFT only, a scale-by-1/N pass over the real scratch cell
//   4. Copy the result into temporality-aware published tensor storage
//
// Downstream tensorRef consumers read published results, never mutable scratch.
// Fails with a clear error on the Metal backend
// via MetalRenderer's unsupported-op path.

import Foundation

extension LazyOp {
  func emitAcceleratedFFT(
    b: IRBuilder, ctx: IRContext, g: Graph, node: Node,
    inputs: [Lazy], nodeId: NodeID
  ) throws {
    switch self {
    case .acceleratedFFT(let N, let reCell, let imCell, let reOutput, let imOutput):
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "acceleratedFFT", expected: 1, actual: inputs.count)
      }
      guard let inputNodeId = node.inputs.first,
        let inputTensorId = g.nodeToTensor[inputNodeId],
        let inputTensor = g.tensors[inputTensorId]
      else {
        throw DGenError.tensorError(
          op: "acceleratedFFT", reason: "input must be a tensor of shape [N]")
      }

      let log2N = Int(Foundation.log2(Double(N)))
      guard 1 << log2N == N else {
        throw DGenError.tensorError(
          op: "acceleratedFFT", reason: "N (\(N)) must be a power of 2")
      }

      let zero = b.constant(0.0)

      // Copy input real samples into reCell, zero imCell.
      b.loop(N) { i in
        let sample = b.tensorRead(inputTensor, flatIdx: i, shape: [N])
        _ = b.memoryWrite(reCell, i, sample)
        _ = b.memoryWrite(imCell, i, zero)
      }

      // Forward host FFT call, rendered by CRenderer.
      let callUOp = UOp(
        op: .acceleratedFFTCall(
          log2N: log2N, reCell: reCell, imCell: imCell, inverse: false),
        value: ctx.useVariable(src: nil))
      b.ops.append(callUOp)

      try publishFFTResult(reCell, tensorId: reOutput, count: N, b: b, g: g)
      try publishFFTResult(imCell, tensorId: imOutput, count: N, b: b, g: g)
      b.use(val: zero)

    case .acceleratedIFFT(let N, let reCell, let imCell, let output):
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "acceleratedIFFT", expected: 2, actual: inputs.count)
      }
      guard let reNodeId = node.inputs.first,
        let reTensorId = g.nodeToTensor[reNodeId],
        let reTensor = g.tensors[reTensorId]
      else {
        throw DGenError.tensorError(
          op: "acceleratedIFFT", reason: "first input (re) must be a tensor of shape [N]")
      }
      guard node.inputs.count >= 2,
        let imTensorId = g.nodeToTensor[node.inputs[1]],
        let imTensor = g.tensors[imTensorId]
      else {
        throw DGenError.tensorError(
          op: "acceleratedIFFT", reason: "second input (im) must be a tensor of shape [N]")
      }

      let log2N = Int(Foundation.log2(Double(N)))
      guard 1 << log2N == N else {
        throw DGenError.tensorError(
          op: "acceleratedIFFT", reason: "N (\(N)) must be a power of 2")
      }

      let zero = b.constant(0.0)
      let invN = b.constant(1.0 / Float(N))

      // Copy re and im input tensors into owned scratch cells.
      b.loop(N) { i in
        let reSample = b.tensorRead(reTensor, flatIdx: i, shape: [N])
        let imSample = b.tensorRead(imTensor, flatIdx: i, shape: [N])
        _ = b.memoryWrite(reCell, i, reSample)
        _ = b.memoryWrite(imCell, i, imSample)
      }

      // Inverse host FFT call, rendered by CRenderer.
      let callUOp = UOp(
        op: .acceleratedFFTCall(
          log2N: log2N, reCell: reCell, imCell: imCell, inverse: true),
        value: ctx.useVariable(src: nil))
      b.ops.append(callUOp)

      // The host inverse FFT is unnormalized; normalize the real output by 1/N.
      b.loop(N) { i in
        let v = b.memoryRead(reCell, i)
        _ = b.memoryWrite(reCell, i, v * invN)
      }

      try publishFFTResult(reCell, tensorId: output, count: N, b: b, g: g)
      b.use(val: zero)

    default: break
    }
  }
}

/// Copy host-call scratch into the result's temporality-aware allocation.
private func publishFFTResult(
  _ scratch: CellID, tensorId: TensorID, count: Int, b: IRBuilder, g: Graph
) throws {
  guard let tensor = g.tensors[tensorId], !tensor.isLazy else {
    throw DGenError.tensorError(op: "acceleratedFFT", reason: "result storage was not materialized")
  }
  b.loop(count) { i in
    let value = b.memoryRead(scratch, i)
    if let layout = g.frameAwareCells[tensor.cellId] {
      _ = b.frameAwareTensorWrite(cellId: tensor.cellId, tensorSize: layout.tensorSize,
                                 elemIdx: i, value: value)
    } else {
      _ = b.memoryWrite(tensor.cellId, i, value)
    }
  }
}

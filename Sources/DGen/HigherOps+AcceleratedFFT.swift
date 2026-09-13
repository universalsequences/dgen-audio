// HigherOps+AcceleratedFFT.swift
//
// C-only FFT/IFFT ops backed by the runtime host FFT service.
// API mirrors tensorFFT/tensorIFFT in HigherOps+FFT.swift: input/output shapes
// are identical, but the kernel compiles down to a single host FFT call instead of
// a butterfly decomposition — orders of magnitude smaller generated C code and
// much faster at runtime. Throws `DGenError.compilationFailed` if used with the
// Metal backend.

import Foundation

extension Graph {

  /// N-point forward FFT using the runtime host FFT service.
  ///
  /// Input must have tensor shape [N] where N is a power of 2.
  /// Returns (re, im) NodeIDs, both shape [N].
  ///
  /// C backend only — Metal compilation will throw a clear error.
  public func acceleratedFFT(_ input: NodeID, N: Int) -> (re: NodeID, im: NodeID) {
    let log2N = Int(Foundation.log2(Double(N)))
    precondition(1 << log2N == N, "N must be a power of 2")

    let reCell = alloc(vectorWidth: N)
    let imCell = alloc(vectorWidth: N)

    // Published real result, separate from the in-place host-call scratch.
    let reOutputCell = reserveLazyCellId()
    let reTensorId = nextTensorId
    nextTensorId += 1
    tensors[reTensorId] = Tensor(
      id: reTensorId, shape: [N], cellId: reOutputCell,
      baseShape: [N], transforms: [], isLazy: true)
    cellToTensor[reOutputCell] = reTensorId

    // Published imaginary result.
    let imOutputCell = reserveLazyCellId()
    let imTensorId = nextTensorId
    nextTensorId += 1
    tensors[imTensorId] = Tensor(
      id: imTensorId, shape: [N], cellId: imOutputCell,
      baseShape: [N], transforms: [], isLazy: true)
    cellToTensor[imOutputCell] = imTensorId

    // Scratch is private to the host FFT call. Published results use ordinary
    // lazy tensor storage so consumers in another frame loop retain each hop.
    let fftOp = n(
      .acceleratedFFT(windowSize: N, reCell: reCell, imCell: imCell,
                      reOutput: reTensorId, imOutput: imTensorId),
      [input], shape: .scalar)

    // TensorRef nodes chained after fftOp for ordering.
    let reNode = n(.tensorRef(reTensorId), [fftOp], shape: .tensor([N]))
    let imNode = n(.tensorRef(imTensorId), [fftOp], shape: .tensor([N]))
    nodeToTensor[reNode] = reTensorId
    materializeNodes.insert(reNode)
    nodeToTensor[imNode] = imTensorId
    materializeNodes.insert(imNode)

    if let hopRate = nodeHopRate[input] ?? nodeHopRate[fftOp] {
      nodeHopRate[reNode] = hopRate
      nodeHopRate[imNode] = hopRate
    }

    return (re: reNode, im: imNode)
  }

  /// N-point inverse FFT using the runtime host FFT service.
  ///
  /// Takes (re, im) NodeIDs of shape [N], returns real part of shape [N]
  /// normalized by 1/N. Imaginary part is discarded (correct for real signals).
  ///
  /// C backend only — Metal compilation will throw a clear error.
  public func acceleratedIFFT(_ re: NodeID, _ im: NodeID, N: Int) -> NodeID {
    let log2N = Int(Foundation.log2(Double(N)))
    precondition(1 << log2N == N, "N must be a power of 2")

    let reCell = alloc(vectorWidth: N)
    let imCell = alloc(vectorWidth: N)

    // Only the real result is exposed. Scratch never aliases published results.
    let reOutputCell = reserveLazyCellId()
    let reTensorId = nextTensorId
    nextTensorId += 1
    tensors[reTensorId] = Tensor(
      id: reTensorId, shape: [N], cellId: reOutputCell,
      baseShape: [N], transforms: [], isLazy: true)
    cellToTensor[reOutputCell] = reTensorId

    let ifftOp = n(
      .acceleratedIFFT(windowSize: N, reCell: reCell, imCell: imCell, output: reTensorId),
      [re, im], shape: .scalar)

    let reNode = n(.tensorRef(reTensorId), [ifftOp], shape: .tensor([N]))
    nodeToTensor[reNode] = reTensorId
    materializeNodes.insert(reNode)

    if let hopRate = nodeHopRate[re] ?? nodeHopRate[im] ?? nodeHopRate[ifftOp] {
      nodeHopRate[reNode] = hopRate
    }

    return reNode
  }
}

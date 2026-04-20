// HigherOps+AcceleratedFFT.swift
//
// C-only FFT/IFFT ops backed by Apple's Accelerate framework (vDSP_fft_zip).
// API mirrors tensorFFT/tensorIFFT in HigherOps+FFT.swift: input/output shapes
// are identical, but the kernel compiles down to a single vDSP call instead of
// a butterfly decomposition — orders of magnitude smaller generated C code and
// much faster at runtime. Throws `DGenError.compilationFailed` if used with the
// Metal backend.

import Foundation

extension Graph {

  /// N-point forward FFT using Apple's Accelerate framework (vDSP_fft_zip).
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

    // Side-effect op: reads input tensor, populates re/im cells, calls vDSP.
    let fftOp = n(
      .acceleratedFFT(windowSize: N, reCell: reCell, imCell: imCell),
      [input], shape: .scalar)

    // Re tensor view backed by reCell
    let reTensorId = nextTensorId
    nextTensorId += 1
    tensors[reTensorId] = Tensor(
      id: reTensorId, shape: [N], cellId: reCell,
      baseShape: [N], transforms: [])
    cellToTensor[reCell] = reTensorId

    // Im tensor view backed by imCell
    let imTensorId = nextTensorId
    nextTensorId += 1
    tensors[imTensorId] = Tensor(
      id: imTensorId, shape: [N], cellId: imCell,
      baseShape: [N], transforms: [])
    cellToTensor[imCell] = imTensorId

    // TensorRef nodes chained after fftOp for ordering.
    let reNode = n(.tensorRef(reTensorId), [fftOp], shape: .tensor([N]))
    let imNode = n(.tensorRef(imTensorId), [fftOp], shape: .tensor([N]))
    nodeToTensor[reNode] = reTensorId
    nodeToTensor[imNode] = imTensorId

    return (re: reNode, im: imNode)
  }

  /// N-point inverse FFT using Apple's Accelerate framework (vDSP_fft_zip).
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

    let ifftOp = n(
      .acceleratedIFFT(windowSize: N, reCell: reCell, imCell: imCell),
      [re, im], shape: .scalar)

    // Only re tensor is exposed (result is real-valued).
    let reTensorId = nextTensorId
    nextTensorId += 1
    tensors[reTensorId] = Tensor(
      id: reTensorId, shape: [N], cellId: reCell,
      baseShape: [N], transforms: [])
    cellToTensor[reCell] = reTensorId

    let reNode = n(.tensorRef(reTensorId), [ifftOp], shape: .tensor([N]))
    nodeToTensor[reNode] = reTensorId

    return reNode
  }
}

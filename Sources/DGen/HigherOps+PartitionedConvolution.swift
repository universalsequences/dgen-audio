// HigherOps+PartitionedConvolution.swift
//
// Uniformly Partitioned Overlap-Save/Add (UPOLS) as a single DGen primitive.
//
// Takes live (re, im) spectra of shape [N] arriving at hop rate, plus two static
// [K, N] IR partition tensors (re, im). Maintains a [2K, N] mirror-layout ring
// per channel, keyed off the hop counter produced upstream by `bufferView`.
// Per hop, writes the new spectrum into two ring rows (p and p+K), then sweeps
// K partitions computing Y = Σ_{k=0..K-1} H[k] * X[n-k] as a complex MAC.
// Writes two [N] output tensors, exposed downstream as tensorRef nodes — same
// output contract as `acceleratedFFT`.

import Foundation

extension Graph {

  /// Partitioned spectral convolution (UPOLS).
  ///
  /// - Parameters:
  ///   - reInput, imInput: live `[N]` spectrum (e.g. output of `acceleratedFFT`
  ///     on a hop-gated `bufferView`). These inputs carry hop-based temporality
  ///     automatically, so the whole op runs once per hop — no manual gating
  ///     or hop-counter plumbing needed.
  ///   - irRePartitions, irImPartitions: static `[K, N]` tensors. Row k holds
  ///     the N-point FFT of the k-th IR partition.
  ///   - K, N: partition count and FFT size.
  ///   - hopSize: hop size (informational; scheduling comes from input temporality).
  /// - Returns: (reOut, imOut), both tensorRef nodes of shape `[N]`.
  public func partitionedSpectralConvolve(
    _ reInput: NodeID,
    _ imInput: NodeID,
    _ irRePartitions: NodeID,
    _ irImPartitions: NodeID,
    K: Int, N: Int, hopSize: Int
  ) -> (reOut: NodeID, imOut: NodeID) {
    precondition(K > 0 && N > 0, "partitionedSpectralConvolve: K and N must be positive")
    precondition(hopSize > 0, "partitionedSpectralConvolve: hopSize must be positive")

    // Persistent ring cells: [2K, N] per channel. Mirror layout means writing to
    // both row p and row p+K — then reads at row (p + K - k) land in [1, 2K)
    // without runtime modulo.
    let ringReCell = alloc(vectorWidth: 2 * K * N)
    let ringImCell = alloc(vectorWidth: 2 * K * N)
    persistentCells.insert(ringReCell)
    persistentCells.insert(ringImCell)

    // Frame-rate hop counter (0..hopSize-1, wraps at hopSize) — mirrors the
    // overlapAdd pattern so the op can self-gate with `if (hopCounter == 0)`
    // regardless of block-level scheduling.
    let hopCounterCell = alloc(vectorWidth: 1)
    persistentCells.insert(hopCounterCell)

    // Per-hop partition counter (0..K-1). Advances once per hop inside the gate.
    let partitionCounterCell = alloc(vectorWidth: 1)
    persistentCells.insert(partitionCounterCell)

    // Owned output scratch cells.
    let reOutCell = alloc(vectorWidth: N)
    let imOutCell = alloc(vectorWidth: N)

    // Side-effect op: reads re/im inputs + static IR partitions, writes to
    // output cells. Shape `.scalar` because the op itself has no value —
    // downstream reads via tensorRef nodes (same pattern as acceleratedFFT).
    let convOp = n(
      .partitionedSpectralConvolve(
        K: K, N: N, hopSize: hopSize,
        ringReCell: ringReCell, ringImCell: ringImCell,
        hopCounterCell: hopCounterCell, partitionCounterCell: partitionCounterCell,
        reOutCell: reOutCell, imOutCell: imOutCell),
      [reInput, imInput, irRePartitions, irImPartitions],
      shape: .scalar)

    // Expose outputs as tensorRef nodes, chained after convOp for correct
    // emission ordering.
    let reTensorId = nextTensorId
    nextTensorId += 1
    tensors[reTensorId] = Tensor(
      id: reTensorId, shape: [N], cellId: reOutCell,
      baseShape: [N], transforms: [])
    cellToTensor[reOutCell] = reTensorId

    let imTensorId = nextTensorId
    nextTensorId += 1
    tensors[imTensorId] = Tensor(
      id: imTensorId, shape: [N], cellId: imOutCell,
      baseShape: [N], transforms: [])
    cellToTensor[imOutCell] = imTensorId

    let reNode = n(.tensorRef(reTensorId), [convOp], shape: .tensor([N]))
    let imNode = n(.tensorRef(imTensorId), [convOp], shape: .tensor([N]))
    nodeToTensor[reNode] = reTensorId
    nodeToTensor[imNode] = imTensorId

    return (reOut: reNode, imOut: imNode)
  }
}

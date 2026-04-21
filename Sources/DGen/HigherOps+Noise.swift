// HigherOps+Noise.swift
//
// `noise` — shape-polymorphic pseudo-random noise. Matches scalar
// `noise(cellId)` semantics for `size == 1`; for larger sizes emits a
// `[size]` tensor of N independent random values refreshed every frame.
//
// Use cases:
// - scalar `noise` (existing): one random per sample, audio-rate white
//   noise
// - tensor `noise @size N`: randomized phase spectra (paulstretch), noise
//   fields, per-bin jitter. Wrap in `hopHold` to snapshot at hop rate so
//   downstream FFT chains don't pay the N-per-frame regeneration cost.
//
// Temporality contract: frame-rate by default, just like scalar noise.
// Hop-rate consumers should wrap with `hopHold @hopSize` (extended in
// this branch to handle tensor inputs via `graph.latch`).

import Foundation

extension Graph {

  /// Scalar or tensor-shaped pseudo-random noise.
  ///
  /// - Parameters:
  ///   - size: Output tensor size. `size == 1` (default) returns scalar
  ///     noise matching the original `.noise` op. Larger sizes return a
  ///     `[size]` tensor with independent random values per element.
  ///   - hopSize: Optional. When provided (and `size > 1`), the tensor
  ///     regenerates only on hop boundaries — a fused `noise + hopHold`
  ///     that's dramatically cheaper than composing the two. The output
  ///     is registered as hop-producing so downstream spectral ops stay
  ///     hop-gated. Ignored for `size == 1` (use scalar `noise → hopHold`
  ///     instead).
  /// - Returns: Node whose output is uniform random in `[-1, 1]`.
  public func noise(size: Int = 1, hopSize: Int? = nil) -> NodeID {
    precondition(size >= 1, "noise(size:) requires size >= 1, got \(size)")

    if size == 1 {
      // Classic scalar noise: one xorshift state cell, one value per frame.
      let cell = alloc()
      return n(.noise(cell))
    }

    if let hopSize = hopSize {
      precondition(hopSize > 0, "noise(hopSize:) must be > 0, got \(hopSize)")
      return makeHopTensorNoise(size: size, hopSize: hopSize)
    }

    // One shared xorshift state cell (advances N times per frame, producing
    // N distinct values).
    let stateCell = alloc()
    persistentCells.insert(stateCell)

    // Frame-aware output cell: each frame writes its own `[N]` slice so the
    // current frame's values survive the cross-block read in downstream
    // consumers (latch / hopHold / FFT). Without frame-awareness the single
    // [N] cell is overwritten by every subsequent frame before the next
    // block runs, and downstream latches see only the last frame's values.
    let outputCell = allocFrameAware(tensorSize: size, frameCount: maxFrameCount)

    let tensorId = nextTensorId
    nextTensorId += 1
    let tensorShape: Shape = [size]
    tensors[tensorId] = Tensor(
      id: tensorId, shape: tensorShape, cellId: outputCell,
      baseShape: tensorShape, transforms: [])
    cellToTensor[outputCell] = tensorId

    let node = n(
      .tensorNoise(stateCell, outputCell, size), [],
      shape: .tensor(tensorShape))
    nodeToTensor[node] = tensorId
    return node
  }

  /// Fused `noise @size N → hopHold @hopSize M`. Regenerates the output
  /// tensor only on hop boundaries, holding between hops — so the work is
  /// amortized to `N` xorshift steps per hop instead of `N * frameCount`.
  ///
  /// Implementation detail: the output cell is a single persistent `[N]`
  /// buffer (not frame-aware). Because writes happen only when the hop
  /// counter is 0 and reads between hops return the last snapshot, the
  /// persistent layout is sufficient — downstream `hopBased` consumers
  /// read the current hop's values every frame.
  private func makeHopTensorNoise(size: Int, hopSize: Int) -> NodeID {
    // Hop counter shared with downstream hop gates via `nodeHopRate`.
    let counterCell = alloc(vectorWidth: 1)
    persistentCells.insert(counterCell)
    let hOne = n(.constant(1.0), [])
    let hZero = n(.constant(0.0), [])
    let hopConst = n(.constant(Float(hopSize)), [])
    let counterAccum = n(.accum(counterCell), hOne, hZero, hZero, hopConst)

    // xorshift state + held output buffer.
    let stateCell = alloc()
    persistentCells.insert(stateCell)
    let outputCell = alloc(vectorWidth: size)
    persistentCells.insert(outputCell)

    let tensorId = nextTensorId
    nextTensorId += 1
    let tensorShape: Shape = [size]
    tensors[tensorId] = Tensor(
      id: tensorId, shape: tensorShape, cellId: outputCell,
      baseShape: tensorShape, transforms: [])
    cellToTensor[outputCell] = tensorId

    // Take counterAccum as input so the scheduler sees the dep and the
    // emit handler can read its value for the hop gate.
    let node = n(
      .hopTensorNoise(stateCell, outputCell, size), [counterAccum],
      shape: .tensor(tensorShape))
    nodeToTensor[node] = tensorId
    nodeHopRate[node] = (hopSize, counterAccum)
    return node
  }
}

// HigherOps+SpectrumDelay.swift
//
// `spectrumDelay @hops` — read the spectrum from `hops` hop-boundaries
// ago. Maintains a `[hops+1, N]` ring of recent spectra internally and
// exposes the oldest row as a readable `[N]` tensor.
//
// Unlocks phase vocoder (and thus proper time-stretch): the per-bin
// phase difference between the current and delayed spectrum gives you
// instantaneous frequency per bin, which you can scale or rewarp to
// change time without changing pitch.
//
// Temporality: inherits the input's hop rate (must be hop-producing).
// The delayed output only changes on hop boundaries, so downstream
// consumers stay hop-gated.

import Foundation

extension Graph {

  /// N-bin spectrum read `hops` hop-boundaries ago.
  ///
  /// The delay line needs an external hop counter so it knows when a new
  /// hop has arrived. Pass the counter from whatever op is driving your
  /// hop rate (e.g. the counter returned by `bufferView`'s `nodeHopRate`
  /// entry, or any other op registered as hop-producing).
  ///
  /// - Parameters:
  ///   - input: a hop-rate `[N]` tensor (e.g. the real or imaginary output
  ///     of `acceleratedFFT`).
  ///   - hops: number of hop-boundaries of delay. `1` = previous hop.
  ///   - hopCounter: the hop counter node (fires 0 on hop boundaries).
  ///     Typically `graph.nodeHopRate[someUpstreamNode]?.1`.
  ///   - hopSize: hop size in frames. Used only to register the delay's
  ///     output as hop-producing so downstream stays hop-gated.
  /// - Returns: Node exposing a `[N]` tensor of the delayed spectrum.
  ///   Until `hops` boundaries have passed, the output is zero.
  public func spectrumDelay(
    _ input: NodeID, N: Int, hops: Int, hopSize: Int
  ) -> NodeID {
    precondition(N > 0, "spectrumDelay: N must be > 0, got \(N)")
    precondition(hops >= 1, "spectrumDelay: hops must be >= 1, got \(hops)")
    precondition(hopSize > 0, "spectrumDelay: hopSize must be > 0, got \(hopSize)")

    // Allocate an internal hop counter so we don't depend on cross-block
    // availability of some upstream op's counter. The counter mirrors any
    // upstream counter with the same hopSize (both start at 0 and
    // increment by 1 each frame), so firing is in lockstep.
    let counterCell = alloc(vectorWidth: 1)
    persistentCells.insert(counterCell)
    let one = n(.constant(1.0), [])
    let zero = n(.constant(0.0), [])
    let hopConst = n(.constant(Float(hopSize)), [])
    let counterAccum = n(.accum(counterCell), one, zero, zero, hopConst)

    // Input tensor resolution happens at emit time — arithmetic ops
    // (`mul`, `add`, …) only populate `nodeToTensor` during the compile
    // pipeline's `allocateTensorOutputs` pass, not at graph-build time.
    // Caller passes `N` explicitly so we don't have to walk the DAG.
    let rows = hops + 1

    // Ring that stores the last `rows` spectra flat as `[rows * N]`.
    let ringCell = alloc(vectorWidth: rows * N)
    persistentCells.insert(ringCell)
    // Current write row (0..hops, wraps at `rows`).
    let rowCell = alloc(vectorWidth: 1)
    persistentCells.insert(rowCell)
    // Dedicated output cell that holds the current delayed row, so
    // downstream consumers read from a plain `[N]` tensor instead of
    // having to compute a dynamic offset into the ring.
    let outputCell = alloc(vectorWidth: N)
    persistentCells.insert(outputCell)

    let tensorId = nextTensorId
    nextTensorId += 1
    let tensorShape: Shape = [N]
    tensors[tensorId] = Tensor(
      id: tensorId, shape: tensorShape, cellId: outputCell,
      baseShape: tensorShape, transforms: [])
    cellToTensor[outputCell] = tensorId

    // Emit op takes the input spectrum and the caller's hop counter as
    // dependencies. The counter == 0 check inside the emit handler gates
    // the write+advance to hop boundaries.
    let node = n(
      .spectrumDelay(ringCell, rowCell, outputCell, N, hops),
      [input, counterAccum],
      shape: .tensor(tensorShape))
    nodeToTensor[node] = tensorId
    // Propagate hop rate so downstream stays hop-gated.
    nodeHopRate[node] = (hopSize, counterAccum)
    return node
  }

  /// Fractional-delay variant: same storage layout as `spectrumDelay`, but
  /// the delay amount is a scalar hop-rate input `delay` (0..maxHops).
  /// On every hop boundary, linearly interpolates between ring rows at
  /// `floor(delay)` and `floor(delay)+1` hops ago — giving a smoothly
  /// sweepable spectral delay line.
  ///
  /// Typical usage: drive `delay` with a per-hop scalar (e.g. an LFO via
  /// `phasor → hopHold` scaled into `[0, maxHops]`).
  ///
  /// - Parameters:
  ///   - input: hop-rate `[N]` tensor (e.g. FFT real/imag output).
  ///   - delay: hop-rate scalar in `[0, maxHops]`. Values outside clamp.
  ///   - N: spectrum size.
  ///   - maxHops: ring capacity. Maximum `delay` the op will honour.
  ///     Larger = more memory (`(maxHops+1) * N` floats per output cell).
  ///   - hopSize: frames per hop. Must match the surrounding FFT hop.
  public func spectrumDelayMod(
    _ input: NodeID, delay: NodeID, N: Int, maxHops: Int, hopSize: Int
  ) -> NodeID {
    precondition(N > 0, "spectrumDelayMod: N must be > 0")
    precondition(maxHops >= 1, "spectrumDelayMod: maxHops must be >= 1")
    precondition(hopSize > 0, "spectrumDelayMod: hopSize must be > 0")

    // Internal hop counter (mirrors upstream hop rate via matching hopSize).
    let counterCell = alloc(vectorWidth: 1)
    persistentCells.insert(counterCell)
    let one = n(.constant(1.0), [])
    let zero = n(.constant(0.0), [])
    let hopConst = n(.constant(Float(hopSize)), [])
    let counterAccum = n(.accum(counterCell), one, zero, zero, hopConst)

    let rows = maxHops + 1
    let ringCell = alloc(vectorWidth: rows * N)
    persistentCells.insert(ringCell)
    let rowCell = alloc(vectorWidth: 1)
    persistentCells.insert(rowCell)
    let outputCell = alloc(vectorWidth: N)
    persistentCells.insert(outputCell)

    let tensorId = nextTensorId
    nextTensorId += 1
    let tensorShape: Shape = [N]
    tensors[tensorId] = Tensor(
      id: tensorId, shape: tensorShape, cellId: outputCell,
      baseShape: tensorShape, transforms: [])
    cellToTensor[outputCell] = tensorId

    let node = n(
      .spectrumDelayMod(ringCell, rowCell, outputCell, N, maxHops),
      [input, counterAccum, delay],
      shape: .tensor(tensorShape))
    nodeToTensor[node] = tensorId
    nodeHopRate[node] = (hopSize, counterAccum)
    return node
  }
}

// HigherOps+HopHold.swift
//
// `hopHold` — sample-and-hold of a frame-rate scalar signal at hop rate.
// Runs at frame rate (must, since it reads a frame-rate input), but updates
// a persistent cell ONLY on hop boundaries and always outputs the held
// value. Registers `nodeHopRate` on the result so downstream consumers see
// it as hop-rate and get hop-gated scheduling accordingly.
//
// Pure composition of existing primitives — no new LazyOp:
//   counter  = accum(counterCell, +1, 0, 0, hopSize)   // wraps [0, hopSize)
//   write if = (counter == 0) ? input : existing       // gswitch
//   heldCell := write_if
//   output   = heldCell (sequenced after the write)
//
// Used for things like phase modulation into rectFFT where we want the
// modulation to advance per hop, not per sample, and more importantly keep
// the downstream chain hop-gated instead of demoted to per-sample FFTs.

import Foundation

extension Graph {

  /// Latch the current value of `input` into a persistent cell at every hop
  /// boundary (counter == 0) and emit the latched value. The result carries
  /// `nodeHopRate = (hopSize, counterNode)` so downstream ops inherit
  /// hop-based temporality.
  ///
  /// - Parameters:
  ///   - input: Scalar frame-rate signal.
  ///   - hopSize: Number of frames between latches. Must match the hop of
  ///     the surrounding FFT/bufferView/overlapAdd for consistent scheduling.
  /// - Returns: Scalar NodeID whose value changes only at hop boundaries.
  public func hopHold(_ input: NodeID, hopSize: Int) -> NodeID {
    precondition(hopSize > 0, "hopHold: hopSize must be > 0")

    // Counter wrapping at hopSize — same pattern as bufferView's hop counter.
    let counterCell = alloc(vectorWidth: 1)
    persistentCells.insert(counterCell)

    let hOne = n(.constant(1.0), [])
    let hZero = n(.constant(0.0), [])
    let hopConst = n(.constant(Float(hopSize)), [])
    let counterAccum = n(.accum(counterCell), hOne, hZero, hZero, hopConst)

    // Trigger is 1 at hop boundaries (counter == 0), else 0.
    let trigger = n(.eq, counterAccum, hZero)

    // `.latch` is flagged `isInherentlyScalar` in FeedbackAnalysis — the
    // block it lives in runs scalar per-frame, so the read/write of the
    // latch cell is properly sequenced (no SIMD-4 all-lanes-load-stale bug).
    // `latch(value, cond)` stores `value` when `cond > 0`, else holds.
    let latchCell = alloc(vectorWidth: 1)
    persistentCells.insert(latchCell)
    let held = n(.latch(latchCell), input, trigger)

    // Tag as hop-producing so TemporalityPass treats downstream consumers
    // as hop-based. Same contract bufferView uses.
    nodeHopRate[held] = (hopSize, counterAccum)
    return held
  }
}

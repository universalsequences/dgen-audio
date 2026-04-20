// HigherOps+Latch.swift
//
// `latch` — sample-and-hold that works on both scalar and tensor values.
// Callers pass a `value` node and a scalar `cond` node; on frames where
// `cond > 0` the current `value` is captured into a persistent cell, and
// on every frame the latched value is returned.
//
// Polymorphic over input shape: scalar `value` allocates a single-cell
// latch (same behavior as the original patch-editor `latch`); tensor
// `value` allocates a cell sized to the tensor so every element has its
// own stored position. The cell is registered as persistent so the buffer
// reuse pass leaves it alone.
//
// Mirrors how `phasor` / `hopHold` factor their cell allocation into
// a DGen-owned helper so operator implementations in the patch editor
// don't have to know about sizing / persistence.

import Foundation

extension Graph {

  /// Sample-and-hold. Captures `value` into a persistent cell whenever
  /// `cond > 0`, outputs the latched value every frame.
  ///
  /// Allocates a cell matching the value's shape:
  /// - scalar input → scalar cell (width 1)
  /// - tensor `[N]` → cell of width N (per-element latch)
  ///
  /// Unknown / missing shape is treated as scalar so legacy callers keep
  /// working.
  public func latch(_ value: NodeID, _ cond: NodeID) -> NodeID {
    let valueShape = nodes[value]?.shape ?? .scalar
    let width: Int
    switch valueShape {
    case .scalar:
      width = 1
    case .tensor(let dims):
      let elemCount = dims.reduce(1, *)
      width = elemCount > 0 ? elemCount : 1
    }

    let cell = alloc(vectorWidth: width)
    persistentCells.insert(cell)
    let node = n(.latch(cell), value, cond)

    // When the trigger is already hop-rate (e.g. from `hopHold` or a
    // bufferView's own counter), the latched value only *actually* changes
    // on hop boundaries — between hops the store is a no-op and every frame
    // re-reads the same held cell. Tagging the result as hop-producing lets
    // TemporalityPass hop-gate downstream FFT/IFFT/overlap-add chains
    // instead of demoting them to per-frame execution. Without this a
    // spectral-freeze patch runs the IFFT 256× per hop.
    if let hopRate = nodeHopRate[cond] {
      nodeHopRate[node] = hopRate
    }
    return node
  }
}

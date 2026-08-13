// HigherOps+DelayLine.swift
//
// `tensorDelay` — per-lane interpolated delay lines for tensor signals.
// The multichannel analogue of `Graph.delay`: each tensor lane gets its
// own circular buffer region, all lanes share one write head (every lane
// writes exactly one sample per frame), and the delay time may be a
// broadcast scalar or a per-lane tensor.
//
// Storage layout: one flat cell of `lanes * maxDelay` floats, lane-major
// (`lane * maxDelay + pos`), plus a dedicated `[lanes]` output cell that
// downstream tensor reads resolve against (same pattern as
// `spectrumDelay` — see HigherOps+SpectrumDelay.swift).

import Foundation

extension Graph {

  /// Per-lane delay lines for a tensor signal.
  ///
  /// - Parameters:
  ///   - input: tensor-shaped node to delay (one delay line per element).
  ///   - delayTimeInSamples: scalar node (broadcast to every lane) or a
  ///     tensor node with `elementShape` (per-lane delay times). Values are
  ///     clamped to `[0, maxDelay - 1]` at runtime.
  ///   - elementShape: the input's per-frame shape. Passed explicitly
  ///     because arithmetic nodes only get their `nodeToTensor` entry
  ///     during `allocateTensorOutputs`, not at graph-build time.
  ///   - maxDelay: per-lane buffer length in samples. Memory cost is
  ///     `lanes * maxDelay` floats, so tensor delays default to a smaller
  ///     buffer than the scalar op (see `SignalTensor.defaultMaxDelay`).
  /// - Returns: node exposing the delayed `[elementShape]` tensor.
  public func tensorDelay(
    _ input: NodeID, _ delayTimeInSamples: NodeID, elementShape: Shape, maxDelay: Int
  ) -> NodeID {
    precondition(maxDelay > 1, "tensorDelay: maxDelay must be > 1, got \(maxDelay)")
    let lanes = Swift.max(1, elementShape.reduce(1, *))

    // Lane-major sample ring: lane `e` owns [e*maxDelay, (e+1)*maxDelay).
    let bufferCell = alloc(vectorWidth: lanes * maxDelay)
    persistentCells.insert(bufferCell)
    // Shared write head (every lane writes each frame, so one cursor).
    let writePosCell = alloc(vectorWidth: 1)
    persistentCells.insert(writePosCell)
    // Dedicated frame-aware output cell: each frame writes its own `[lanes]`
    // slice, so the per-frame value survives when a downstream consumer lands
    // in a later block (separate frame loop). A plain `[lanes]` cell would be
    // overwritten every frame and later blocks would only see the final frame
    // (same reasoning as tensorNoise's output cell).
    let outputCell = allocFrameAware(tensorSize: lanes, frameCount: maxFrameCount)
    persistentCells.insert(outputCell)

    let tensorId = nextTensorId
    nextTensorId += 1
    tensors[tensorId] = Tensor(
      id: tensorId, shape: elementShape, cellId: outputCell,
      baseShape: elementShape, transforms: [])
    cellToTensor[outputCell] = tensorId

    let node = n(
      .delayLine(bufferCell, writePosCell, outputCell, elementShape, maxDelay),
      [input, delayTimeInSamples],
      shape: .tensor(elementShape))
    nodeToTensor[node] = tensorId
    return node
  }
}

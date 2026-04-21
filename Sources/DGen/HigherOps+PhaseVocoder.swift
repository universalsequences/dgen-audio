import Foundation

extension Graph {

  /// Wrap `x` into the principal argument range `[-π, π]`.
  public func principalArg(_ x: NodeID) -> NodeID {
    let tau = n(.constant(2.0 * Float.pi))
    let k = n(.round, n(.div, x, tau))
    let wrap = n(.mul, k, tau)
    return n(.sub, x, wrap)
  }

  /// Fixed-duration phase-vocoder pitch shift.
  ///
  /// This is implemented as a dedicated hop-rate spectral op rather than a
  /// composition of generic tensor primitives. Per hop it:
  /// 1. Computes current magnitude and true instantaneous frequency per bin
  /// 2. Interpolates those source-bin values at `k / pitchRatio`
  /// 3. Advances a synthesis phase accumulator for each output bin
  /// 4. Writes remapped `(re, im)` spectra for IFFT
  ///
  /// The result preserves duration while moving spectral energy to new bins,
  /// which is the missing step in the previous “scale phase increment in place”
  /// implementation.
  public func phaseVocoder(
    _ xRe: NodeID, _ xIm: NodeID,
    pitchRatio: NodeID,
    N: Int, hopSize: Int
  ) -> (yRe: NodeID, yIm: NodeID) {
    precondition(N > 0, "phaseVocoder: N must be > 0")
    precondition(hopSize > 0, "phaseVocoder: hopSize must be > 0")

    let prevAnalysisPhaseCell = alloc(vectorWidth: N)
    let synthPhaseCell = alloc(vectorWidth: N)
    let tempMagCell = alloc(vectorWidth: N)
    let tempOmegaCell = alloc(vectorWidth: N)
    let initCell = alloc(vectorWidth: 1)
    let reOutCell = alloc(vectorWidth: N)
    let imOutCell = alloc(vectorWidth: N)
    persistentCells.insert(prevAnalysisPhaseCell)
    persistentCells.insert(synthPhaseCell)
    persistentCells.insert(initCell)

    let counterCell = alloc(vectorWidth: 1)
    persistentCells.insert(counterCell)
    let one = n(.constant(1.0))
    let zero = n(.constant(0.0))
    let hopConst = n(.constant(Float(hopSize)))
    let counterAccum = n(.accum(counterCell), one, zero, zero, hopConst)

    let pvOp = n(
      .phaseVocoderPitchShift(
        windowSize: N, hopSize: hopSize,
        prevAnalysisPhaseCell: prevAnalysisPhaseCell,
        synthPhaseCell: synthPhaseCell,
        tempMagCell: tempMagCell,
        tempOmegaCell: tempOmegaCell,
        initCell: initCell,
        reOutCell: reOutCell,
        imOutCell: imOutCell
      ),
      [xRe, xIm, pitchRatio, counterAccum],
      shape: .scalar
    )

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

    let yRe = n(.tensorRef(reTensorId), [pvOp], shape: .tensor([N]))
    let yIm = n(.tensorRef(imTensorId), [pvOp], shape: .tensor([N]))
    nodeToTensor[yRe] = reTensorId
    nodeToTensor[yIm] = imTensorId
    nodeHopRate[yRe] = (hopSize, counterAccum)
    nodeHopRate[yIm] = (hopSize, counterAccum)

    return (yRe: yRe, yIm: yIm)
  }
}

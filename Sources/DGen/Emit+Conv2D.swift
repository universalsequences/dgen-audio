import Foundation

/// SIMD-unrolled emission for `.conv2d` nodes annotated by `Conv2DPass`.
///
/// Strategy: fully unroll the output grid (outY rows and the ≤3 column groups per
/// row) at Swift emit time. For each group, emit a `parallelRange(4)` loop whose
/// body has no nested loops, no gswitch, no mutate, and only the loop variable
/// as a non-constant offset — so the existing `CBackendUOpSIMDUpgrade` pass can
/// promote it to `vectorWidth=4` NEON.
///
/// Row bounds are resolved at emit time (Swift-level `guard`) — taps whose ky
/// would read out-of-bounds are simply not emitted. Column edges are zeroed by
/// multiplying a contiguous mask vector loaded from a compile-time constant
/// cell allocated by the pass.
func emitOptimizedConv2D(
  b: IRBuilder,
  ctx: IRContext,
  g: Graph,
  node: Node,
  maskCellId: CellID
) throws {
  guard case .conv2d(let kernelShape) = node.op,
    kernelShape.count == 2,
    node.inputs.count >= 2,
    case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape, inShape.count == 2,
    let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
    let kTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] }),
    let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId
  else {
    throw DGenError.tensorError(
      op: "conv2d(optimized)", reason: "missing shape/tensor data")
  }

  let (inH, inW) = (inShape[0], inShape[1])
  let (kH, kW) = (kernelShape[0], kernelShape[1])
  let (padH, padW) = (kH / 2, kW / 2)
  let inCell = inTensor.cellId
  let kernelCell = kTensor.cellId

  // Frame-aware cells store one tensor slice per frame at
  // `cellBase + frameIdx * tensorSize + elem`. This gather builds raw offsets,
  // so the frame term must be folded in explicitly — otherwise every frame of
  // a multi-frame block convolves frame 0's slice. The non-affine offset also
  // (correctly) disqualifies the loop from the NEON upgrade pass.
  let inFrameSize = g.frameAwareCells[inCell]?.tensorSize
  let outFrameSize = g.frameAwareCells[outCell]?.tensorSize
  let frameIdx: Expr? =
    (inFrameSize != nil || outFrameSize != nil) ? b.currentFrameIndex() : nil
  func withFrameOffset(_ offset: Expr, frameSize: Int?) -> Expr {
    guard let frameSize, let frameIdx else { return offset }
    return b.cast(frameIdx * b.intConstant(frameSize) + offset, to: .int)
  }
  // Fast path: kernel data baked at graph-build time → hoist each weight as a
  // preamble-broadcast constant. Runtime path: load and broadcast inside the loop.
  let kernelData = kTensor.data

  // Mask tensor layout: [leftMask[0..3], fullMask[0..3], rightMask[0..3]].
  let leftMaskOffset = 0
  let rightMaskOffset = 8

  // Column groups per row: outX_base ∈ {0, 4, 8, ...} plus, when inW is not a
  // multiple of 4, a final overlapped group at inW-4 (its low lanes recompute
  // values an earlier group already wrote — identical results, benign).
  // A group's kx=0 tap reads lane 0 OOB iff outX_base == 0.
  // A group's kx=kW-1 tap reads lane 3 OOB iff outX_base == inW - 4.
  // For inW == 4, a single group is simultaneously left- and right-edge.
  var groups = stride(from: 0, to: inW - 3, by: 4).map { $0 }
  if groups.last != inW - 4 { groups.append(inW - 4) }

  for outY_i in 0..<inH {
    for outX_base in groups {
      let useLeftMask = (outX_base == 0)
      let useRightMask = (outX_base == inW - 4)

      // rowOffset is within-output-cell. Cell IDs get remapped to physical bases
      // at compile time, so we must put position offsets in the offset arg —
      // never add them into the cell ID directly.
      let rowOffset = outY_i * inW + outX_base

      b.parallelRange(4) { t in
        // Full offset expression: rowOffset + t. The upgrade pass recognizes
        // this as `loopVar + const` (affine) and keeps it scalar int.
        let offset =
          rowOffset == 0 ? t : (b.intConstant(rowOffset) + t)

        // SSA accumulator: chain of `.add` UOps, first-tap seeds it.
        var running: Expr? = nil

        for ky in 0..<kH {
          let inY = outY_i + ky - padH
          // Row bounds resolved at emit time; OOB rows contribute nothing.
          guard inY >= 0 && inY < inH else { continue }

          for kx in 0..<kW {
            // Constant-kernel zero taps contribute nothing — skip them entirely
            // (e.g. the 3x3 laplacian has 4 zero taps, a 1D-in-3x3 kernel 6).
            if let data = kernelData, data[ky * kW + kx] == 0 { continue }
            // Per-tap offset within the input cell: inY*inW + outX_base + (kx-padW)
            // plus the lane offset t. We fold the constant portion into a separate
            // int Expr and add `t` to it.
            let tapRowOffset = inY * inW + outX_base + (kx - padW)
            let tapOffset =
              tapRowOffset == 0 ? t : (b.intConstant(tapRowOffset) + t)
            var v = b.memoryRead(inCell, withFrameOffset(tapOffset, frameSize: inFrameSize))

            // Edge masking: multiply by pre-baked 4-lane mask vector. The
            // offset must include the lane var `t`: SIMD rendering loads the
            // same 4 contiguous lanes either way, but if the loop stays
            // scalar (e.g. frame-aware offsets disqualify the NEON upgrade),
            // a constant offset would apply mask lane 0 to every element.
            // A tap only reads out of bounds horizontally when it actually
            // reaches past the pad: kx < padW (left) / kx > padW (right).
            // `kx == 0` / `kx == kW-1` are only valid proxies for kW == 2·padW+1
            // taps that reach; for a column kernel (kW == 1, padW == 0) both
            // held at once and zeroed the edge columns.
            if useLeftMask && kx < padW {
              let mask = b.memoryRead(maskCellId, b.intConstant(leftMaskOffset) + t)
              v = v * mask
            }
            if useRightMask && kx > padW {
              let mask = b.memoryRead(maskCellId, b.intConstant(rightMaskOffset) + t)
              v = v * mask
            }

            // Kernel weight: constant path broadcasts via the kernel preamble,
            // runtime path loads+broadcasts inline via simdBroadcastLoad.
            let kIdx = ky * kW + kx
            let kVal: Expr
            if let data = kernelData {
              kVal = b.constant(data[kIdx])
            } else {
              kVal = b.simdBroadcastLoad(kernelCell, b.intConstant(kIdx))
            }
            let tap = v * kVal

            if let r = running {
              running = r + tap
            } else {
              running = tap
            }
          }
        }

        // If every (outY, ky) was OOB for this row, running can be nil — fall back to zero.
        let value = running ?? b.constant(0.0)
        _ = b.memoryWrite(outCell, withFrameOffset(offset, frameSize: outFrameSize), value)
      }
    }
  }
}

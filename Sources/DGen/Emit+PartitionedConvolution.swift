// Emit+PartitionedConvolution.swift
//
// Emission for .partitionedSpectralConvolve. Runs every frame; gates its heavy
// work behind `if (hopCounter == 0)` and advances the frame-rate hop counter
// each frame. Same pattern as `.overlapAdd` — self-contained, does not rely on
// block-level hop scheduling.
//
// Per hop (inside the gate):
//   p = partitionCounter                              (already in [0, K))
//   For elem in 0..<N:
//     ring_re[p*N + elem]       = X_re[elem]
//     ring_re[(p+K)*N + elem]   = X_re[elem]          (mirror for wrap-free read)
//     ring_im[p*N + elem]       = X_im[elem]
//     ring_im[(p+K)*N + elem]   = X_im[elem]
//   For elem in 0..<N:
//     Y_re[elem] = Σ_{k=0..K-1} ring_re[(p+K-k)*N + elem]·H_re[k, elem]
//                             − ring_im[(p+K-k)*N + elem]·H_im[k, elem]
//     Y_im[elem] = Σ_{k=0..K-1} ring_re[(p+K-k)*N + elem]·H_im[k, elem]
//                             + ring_im[(p+K-k)*N + elem]·H_re[k, elem]
//   partitionCounter = (p + 1) mod K
// After the gate: hopCounter = (hopCounter + 1) mod hopSize

import Foundation

extension LazyOp {
  func emitPartitionedSpectralConvolve(
    b: IRBuilder, ctx: IRContext, g: Graph, node: Node,
    inputs: [Lazy], nodeId: NodeID
  ) throws {
    guard case .partitionedSpectralConvolve(
      let K, let N, let hopSize,
      let ringReCell, let ringImCell,
      let hopCounterCell, let partitionCounterCell,
      let reOutCell, let imOutCell) = self
    else { return }

    guard inputs.count == 4 else {
      throw DGenError.insufficientInputs(
        operator: "partitionedSpectralConvolve", expected: 4, actual: inputs.count)
    }

    guard node.inputs.count >= 4,
      let reInputTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
      let imInputTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] }),
      let irReTensor = g.nodeToTensor[node.inputs[2]].flatMap({ g.tensors[$0] }),
      let irImTensor = g.nodeToTensor[node.inputs[3]].flatMap({ g.tensors[$0] })
    else {
      throw DGenError.tensorError(
        op: "partitionedSpectralConvolve",
        reason: "all four tensor inputs (reInput, imInput, irRe, irIm) must be tensors")
    }

    let zero = b.constant(0.0)
    let oneInt = b.intConstant(1)
    let kInt = b.intConstant(K)
    let nInt = b.intConstant(N)
    _ = hopSize
    _ = hopCounterCell  // deprecated — kept in the case payload for ABI stability

    // This op's enclosing block is already hop-gated via temporality
    // propagation from the upstream `bufferView` → `acceleratedFFT` → (re, im)
    // chain. So every time this emit runs, we are AT a hop boundary: do the
    // work unconditionally. An earlier attempt maintained a private hop
    // counter for internal gating; that counter lived inside the bufferView
    // hop-gate too (since its memoryWrite got scheduled inside the gated
    // block), so after the first hop it would freeze at 1 forever and the
    // MAC would silently stop firing.
    do {
      let p = b.cast(b.memoryRead(partitionCounterCell, zero), to: .int)
      let pTimesN = p * nInt
      let pPlusKTimesN = (p + kInt) * nInt

      // Mirror-write new spectrum into rows p and p+K of the [2K, N] ring.
      b.loop(N) { elem in
        let elemInt = b.cast(elem, to: .int)
        let reSample = b.tensorRead(reInputTensor, flatIdx: elem, shape: [N])
        let imSample = b.tensorRead(imInputTensor, flatIdx: elem, shape: [N])
        _ = b.memoryWrite(ringReCell, pTimesN + elemInt, reSample)
        _ = b.memoryWrite(ringReCell, pPlusKTimesN + elemInt, reSample)
        _ = b.memoryWrite(ringImCell, pTimesN + elemInt, imSample)
        _ = b.memoryWrite(ringImCell, pPlusKTimesN + elemInt, imSample)
      }

      // Complex MAC: Y = Σ H[k] * X[n-k]. Ring row for partition k is
      // (p + K − k), always in [1, 2K) thanks to the mirror.
      //
      // Loop order is K-outer / N-inner so both the ring buffer and the IR
      // partition reads are unit-stride over N within each inner iteration
      // — cache-friendly and auto-vectorizable. We zero the output cells
      // up-front so every partition is a plain read-modify-accumulate
      // without a first-partition branch (the branch blocks clang from
      // vectorizing the inner loop).
      // Hand off the complex MAC to a single vDSP-backed UOp. The CRenderer
      // renders this as `memset(Y, 0)` + K calls to `vDSP_zvma`, which is
      // orders of magnitude faster than a scalar C double-loop for large K.
      let macCall = UOp(
        op: .partitionedSpectralMACCall(
          K: K, N: N,
          partitionIdxCell: partitionCounterCell,
          ringReCell: ringReCell, ringImCell: ringImCell,
          irReCell: irReTensor.cellId, irImCell: irImTensor.cellId,
          reOutCell: reOutCell, imOutCell: imOutCell),
        value: ctx.useVariable(src: nil))
      b.ops.append(macCall)

      // Advance partition counter: p_next = (p + 1) mod K.
      let pPlusOne = p + oneInt
      let wrappedP = b.gswitch(pPlusOne >= kInt, b.intConstant(0), pPlusOne)
      _ = b.memoryWrite(partitionCounterCell, zero, b.cast(wrappedP, to: .float))
    }

    b.use(val: zero)
  }
}

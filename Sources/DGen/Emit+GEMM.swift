import Foundation

extension LazyOp {
  /// Emits a tiled GEMM kernel using 8x8x8 simdgroup matrix (WMMA) operations.
  /// Frame-aware tensors use gid.z to index into per-frame memory.
  ///
  /// When transA/transB are true, the source tensor is stored transposed in memory
  /// (e.g., [K,M] instead of [M,K]). Metal's simdgroup_load transpose parameter
  /// reads the tile as if it were transposed, avoiding a separate transpose kernel.
  func emitGemm(
    b: IRBuilder, ctx: IRContext, g: Graph, node: Node, nodeId: NodeID, ops: inout [UOp]
  ) throws {
    guard case .gemm(let M, let N, let K, let transA, let transB) = self else { return }

    guard node.inputs.count == 2,
      let leftTensorId = g.nodeToTensor[node.inputs[0]],
      let leftTensor = g.tensors[leftTensorId],
      let rightTensorId = g.nodeToTensor[node.inputs[1]],
      let rightTensor = g.tensors[rightTensorId],
      let outCell = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] })?.cellId
    else {
      throw DGenError.tensorError(op: "gemm", reason: "could not resolve input/output tensors")
    }

    let leftCell = leftTensor.cellId
    let rightCell = rightTensor.cellId
    let kSteps = K / 8

    let tileRow = b.threadgroupPositionY()
    let tileCol = b.threadgroupPositionX()
    let frameIndex = b.threadgroupPositionZ()

    /// Returns the per-frame memory offset for a cell, or zero for static (non-frame-aware) cells.
    func frameBase(cell: CellID, tensorSize: Int) -> Expr {
      ctx.frameAwareTensorCells.contains(cell)
        ? frameIndex * b.intConstant(tensorSize) : b.intConstant(0)
    }

    // M*K == K*M and K*N == N*K, so frame base size is the same regardless of transpose
    let leftFrameBase = frameBase(cell: leftCell, tensorSize: M * K)
    let rightFrameBase = frameBase(cell: rightCell, tensorSize: K * N)
    let outFrameBase = frameBase(cell: outCell, tensorSize: M * N)

    // Zero-initialized accumulator
    let acc = b.simdgroupMatrixZero()

    // K-loop: load A and B tiles, multiply-accumulate
    b.loop(kSteps) { k in
      // Left (A) tile load
      let aOffset: Expr
      let aStride: Int
      if transA {
        // Source is [K, M] row-major; transposed load reads as [M, K]
        aOffset = leftFrameBase + k * b.intConstant(8 * M) + tileRow * b.intConstant(8)
        aStride = M
      } else {
        // Source is [M, K] row-major
        aOffset = leftFrameBase + tileRow * b.intConstant(8 * K) + k * b.intConstant(8)
        aStride = K
      }
      let aTile = b.simdgroupLoad(leftCell, offset: aOffset, stride: aStride, transpose: transA)

      // Right (B) tile load
      let bOffset: Expr
      let bStride: Int
      if transB {
        // Source is [N, K] row-major; transposed load reads as [K, N]
        bOffset = rightFrameBase + tileCol * b.intConstant(8 * K) + k * b.intConstant(8)
        bStride = K
      } else {
        // Source is [K, N] row-major
        bOffset = rightFrameBase + k * b.intConstant(8 * N) + tileCol * b.intConstant(8)
        bStride = N
      }
      let bTile = b.simdgroupLoad(rightCell, offset: bOffset, stride: bStride, transpose: transB)

      b.simdgroupMultiplyAccumulate(aTile, bTile, acc)
    }

    // Store result tile at (tileRow*8, tileCol*8) in C[M,N]
    let cOffset = outFrameBase + tileRow * b.intConstant(8 * N) + tileCol * b.intConstant(8)
    b.simdgroupStore(acc, cellId: outCell, offset: cOffset, stride: N)

    // Mark output as memory-resident so downstream blocks read via tload
    // instead of scalar global communication (defineGlobal/loadGlobal).
    ctx.values[nodeId] = .empty
  }
}

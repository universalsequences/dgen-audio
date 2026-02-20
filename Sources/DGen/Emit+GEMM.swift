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
    let params: (M: Int, N: Int, K: Int, transA: Bool, transB: Bool, chunkSize: Int?, opName: String)
    switch self {
    case .gemm(let M, let N, let K, let transA, let transB):
      params = (M, N, K, transA, transB, nil, "gemm")
    case .gemmChunkPartials(let M, let N, let K, let transA, let transB, let chunkSize, _):
      params = (M, N, K, transA, transB, chunkSize, "gemmChunkPartials")
    default:
      return
    }

    guard node.inputs.count == 2,
      let leftTensorId = g.nodeToTensor[node.inputs[0]],
      let leftTensor = g.tensors[leftTensorId],
      let rightTensorId = g.nodeToTensor[node.inputs[1]],
      let rightTensor = g.tensors[rightTensorId],
      let outCell = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] })?.cellId
    else {
      throw DGenError.tensorError(
        op: params.opName, reason: "could not resolve input/output tensors")
    }

    let M = params.M
    let N = params.N
    let K = params.K
    let transA = params.transA
    let transB = params.transB

    let leftCell = leftTensor.cellId
    let rightCell = rightTensor.cellId
    let kSteps = K / 8
    let leftIsFrameAware = ctx.frameAwareTensorCells.contains(leftCell)
    let rightIsFrameAware = ctx.frameAwareTensorCells.contains(rightCell)

    let tileRow = b.threadgroupPositionY()
    let tileCol = b.threadgroupPositionX()
    let zIndex = b.threadgroupPositionZ()

    // Zero-initialized accumulator
    let acc = b.simdgroupMatrixZero()

    // Shared tile MAC body (used by both per-frame GEMM and chunked partial GEMM).
    func emitTileMac(leftFrameBase: Expr, rightFrameBase: Expr) {
      b.loop(kSteps) { k in
        let aOffset: Expr
        let aStride: Int
        if transA {
          aOffset = leftFrameBase + k * b.intConstant(8 * M) + tileRow * b.intConstant(8)
          aStride = M
        } else {
          aOffset = leftFrameBase + tileRow * b.intConstant(8 * K) + k * b.intConstant(8)
          aStride = K
        }
        let aTile = b.simdgroupLoad(leftCell, offset: aOffset, stride: aStride, transpose: transA)

        let bOffset: Expr
        let bStride: Int
        if transB {
          bOffset = rightFrameBase + tileCol * b.intConstant(8 * K) + k * b.intConstant(8)
          bStride = K
        } else {
          bOffset = rightFrameBase + k * b.intConstant(8 * N) + tileCol * b.intConstant(8)
          bStride = N
        }
        let bTile = b.simdgroupLoad(rightCell, offset: bOffset, stride: bStride, transpose: transB)
        b.simdgroupMultiplyAccumulate(aTile, bTile, acc)
      }
    }

    if let chunkSize = params.chunkSize {
      // Pass 1 of split-frame reduction:
      // - gid.z is CHUNK index (not frame index)
      // - each chunk covers `chunkSize` frames
      // - each (chunk, tileRow, tileCol) writes one partial tile into [chunk, M, N]
      //
      // Dispatch depth is chunkCount (set by scheduler), not frameCount.
      let runtimeFrameCount = b.cast(b.frameCount(), to: .int)
      let chunkBaseFrame = zIndex * b.intConstant(chunkSize)
      let chunkEndFrame = chunkBaseFrame + b.intConstant(chunkSize)

      func emitFrameMac(_ frameIdx: Expr) {
        let leftFrameBase =
          leftIsFrameAware
          ? frameIdx * b.intConstant(M * K) : b.intConstant(0)
        let rightFrameBase =
          rightIsFrameAware
          ? frameIdx * b.intConstant(K * N) : b.intConstant(0)
        emitTileMac(leftFrameBase: leftFrameBase, rightFrameBase: rightFrameBase)
      }

      if leftIsFrameAware || rightIsFrameAware {
        // At least one GEMM input varies per frame, so we must reduce over frames.
        // Fast path: full chunk in bounds. Avoid per-frame branch in the hot loop.
        b.if_(chunkEndFrame <= runtimeFrameCount) {
          b.loop(chunkSize) { localFrame in
            let frameIdx = chunkBaseFrame + localFrame
            emitFrameMac(frameIdx)
          }
        }

        // Tail path: only needed for the final partially-filled chunk.
        b.if_(chunkEndFrame > runtimeFrameCount) {
          b.loop(chunkSize) { localFrame in
            let frameIdx = chunkBaseFrame + localFrame
            b.if_(frameIdx < runtimeFrameCount) {
              emitFrameMac(frameIdx)
            }
          }
        }
      } else {
        // Both inputs are frame-invariant. Preserve tensorAccumulate semantics by computing
        // the static GEMM exactly once and placing it in chunk 0; other chunks remain zero.
        b.if_(zIndex == b.intConstant(0)) {
          emitTileMac(leftFrameBase: b.intConstant(0), rightFrameBase: b.intConstant(0))
        }
      }

      let chunkOffset = zIndex * b.intConstant(M * N)
      let outOffset = chunkOffset + tileRow * b.intConstant(8 * N) + tileCol * b.intConstant(8)
      b.simdgroupStore(acc, cellId: outCell, offset: outOffset, stride: N)
    } else {
      // Standard per-frame GEMM: z is frame index.
      func frameBase(cell: CellID, tensorSize: Int) -> Expr {
        ctx.frameAwareTensorCells.contains(cell)
          ? zIndex * b.intConstant(tensorSize) : b.intConstant(0)
      }
      let leftFrameBase = frameBase(cell: leftCell, tensorSize: M * K)
      let rightFrameBase = frameBase(cell: rightCell, tensorSize: K * N)
      let outFrameBase = frameBase(cell: outCell, tensorSize: M * N)

      emitTileMac(leftFrameBase: leftFrameBase, rightFrameBase: rightFrameBase)

      let cOffset = outFrameBase + tileRow * b.intConstant(8 * N) + tileCol * b.intConstant(8)
      b.simdgroupStore(acc, cellId: outCell, offset: cOffset, stride: N)
    }

    ctx.values[nodeId] = .empty
  }
}

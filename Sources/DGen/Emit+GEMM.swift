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

  /// Emits a threadgroup-staged GEMM kernel. Each threadgroup owns a
  /// (blockM × blockN) output region computed by (blockM/8) × (blockN/8) SIMD groups.
  /// A and B strips of size (blockM × blockK) and (blockK × blockN) are cooperatively
  /// staged into threadgroup memory once per K-iteration, then reused by all SIMD
  /// groups in the threadgroup.
  ///
  /// Handles both the per-frame `.gemmStaged` and the chunked-partials
  /// `.gemmStagedChunkPartials` shapes.
  func emitGemmStaged(
    b: IRBuilder, ctx: IRContext, g: Graph, node: Node, nodeId: NodeID, ops: inout [UOp]
  ) throws {
    let params: (M: Int, N: Int, K: Int, transA: Bool, transB: Bool,
                 blockM: Int, blockN: Int, blockK: Int, chunkSize: Int?)
    switch self {
    case .gemmStaged(let M, let N, let K, let transA, let transB, let bM, let bN, let bK):
      params = (M, N, K, transA, transB, bM, bN, bK, nil)
    case .gemmStagedChunkPartials(let M, let N, let K, let transA, let transB, let chunkSize, _, let bM, let bN, let bK):
      params = (M, N, K, transA, transB, bM, bN, bK, chunkSize)
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
        op: "gemmStaged", reason: "could not resolve input/output tensors")
    }

    let M = params.M
    let N = params.N
    let K = params.K
    let transA = params.transA
    let transB = params.transB
    let blockM = params.blockM
    let blockN = params.blockN
    let blockK = params.blockK

    let leftCell = leftTensor.cellId
    let rightCell = rightTensor.cellId
    let leftIsFrameAware = ctx.frameAwareTensorCells.contains(leftCell)
    let rightIsFrameAware = ctx.frameAwareTensorCells.contains(rightCell)
    let outIsFrameAware = ctx.frameAwareTensorCells.contains(outCell)

    let simdGroupsPerRow = blockN / 8  // SIMD groups across N
    let kSteps = K / blockK

    // Threadgroup scratch: row-major. aScratch is (blockM × blockK), bScratch is (blockK × blockN).
    let aScratch = b.threadgroupScratch(blockM * blockK)
    let bScratch = b.threadgroupScratch(blockK * blockN)

    let tileRow = b.threadgroupPositionY()  // M-tile index
    let tileCol = b.threadgroupPositionX()  // N-tile index
    let zIndex = b.threadgroupPositionZ()   // per-frame: frame index; chunked: chunk index
    let tid = b.threadIndexInThreadgroup()
    let sgid = b.simdgroupIndexInThreadgroup()

    let sgRow = b.div(sgid, b.intConstant(simdGroupsPerRow))
    let sgCol = b.mod(sgid, b.intConstant(simdGroupsPerRow))

    // Per-thread coordinates within the staging strips.
    let aStripRow = b.div(tid, b.intConstant(blockK))   // 0 .. blockM-1
    let aStripCol = b.mod(tid, b.intConstant(blockK))   // 0 .. blockK-1
    let bStripRow = b.div(tid, b.intConstant(blockN))   // 0 .. blockK-1
    let bStripCol = b.mod(tid, b.intConstant(blockN))   // 0 .. blockN-1

    let acc = b.simdgroupMatrixZero()

    /// Emit the full K-loop MAC, reading A from `leftFrameBase` and B from `rightFrameBase`.
    /// Accumulates into `acc`.
    func emitKLoopMac(leftFrameBase: Expr, rightFrameBase: Expr) {
      b.loop(kSteps) { kStrip in
        // --- Cooperative load A strip ---
        let aGlobalRow = tileRow * b.intConstant(blockM) + aStripRow
        let aGlobalCol = kStrip * b.intConstant(blockK) + aStripCol
        let aSrcOffset: Expr
        if transA {
          aSrcOffset = leftFrameBase + aGlobalCol * b.intConstant(M) + aGlobalRow
        } else {
          aSrcOffset = leftFrameBase + aGlobalRow * b.intConstant(K) + aGlobalCol
        }
        let aValue = b.memoryRead(leftCell, aSrcOffset)
        let aDestOffset = aStripRow * b.intConstant(blockK) + aStripCol
        b.scratchWrite(aScratch, aDestOffset, aValue)

        // --- Cooperative load B strip ---
        let bGlobalRow = kStrip * b.intConstant(blockK) + bStripRow
        let bGlobalCol = tileCol * b.intConstant(blockN) + bStripCol
        let bSrcOffset: Expr
        if transB {
          bSrcOffset = rightFrameBase + bGlobalCol * b.intConstant(K) + bGlobalRow
        } else {
          bSrcOffset = rightFrameBase + bGlobalRow * b.intConstant(N) + bGlobalCol
        }
        let bValue = b.memoryRead(rightCell, bSrcOffset)
        let bDestOffset = bStripRow * b.intConstant(blockN) + bStripCol
        b.scratchWrite(bScratch, bDestOffset, bValue)

        b.threadgroupBarrier()

        // --- Per-SIMD-group MAC reading from scratch ---
        let aTileOffset = sgRow * b.intConstant(8 * blockK)
        let aTile = b.simdgroupLoadScratch(aScratch, offset: aTileOffset, stride: blockK, transpose: false)
        let bTileOffset = sgCol * b.intConstant(8)
        let bTile = b.simdgroupLoadScratch(bScratch, offset: bTileOffset, stride: blockN, transpose: false)
        b.simdgroupMultiplyAccumulate(aTile, bTile, acc)

        b.threadgroupBarrier()
      }
    }

    if let chunkSize = params.chunkSize {
      // Chunked-partials mode: z is CHUNK index. Each chunk covers `chunkSize` frames.
      // Accumulator persists across frames within the chunk; output written once to
      // the chunk's [chunkIdx, M, N] slot.
      let runtimeFrameCount = b.cast(b.frameCount(), to: .int)
      let chunkBaseFrame = zIndex * b.intConstant(chunkSize)
      let chunkEndFrame = chunkBaseFrame + b.intConstant(chunkSize)

      func emitFrameMac(_ frameIdx: Expr) {
        let leftFrameBase =
          leftIsFrameAware ? frameIdx * b.intConstant(M * K) : b.intConstant(0)
        let rightFrameBase =
          rightIsFrameAware ? frameIdx * b.intConstant(K * N) : b.intConstant(0)
        emitKLoopMac(leftFrameBase: leftFrameBase, rightFrameBase: rightFrameBase)
      }

      if leftIsFrameAware || rightIsFrameAware {
        // Fast path: full chunk in bounds.
        b.if_(chunkEndFrame <= runtimeFrameCount) {
          b.loop(chunkSize) { localFrame in
            let frameIdx = chunkBaseFrame + localFrame
            emitFrameMac(frameIdx)
          }
        }
        // Tail path: final partially-filled chunk.
        b.if_(chunkEndFrame > runtimeFrameCount) {
          b.loop(chunkSize) { localFrame in
            let frameIdx = chunkBaseFrame + localFrame
            b.if_(frameIdx < runtimeFrameCount) {
              emitFrameMac(frameIdx)
            }
          }
        }
      } else {
        // Both inputs frame-invariant: compute once, place in chunk 0.
        b.if_(zIndex == b.intConstant(0)) {
          emitKLoopMac(leftFrameBase: b.intConstant(0), rightFrameBase: b.intConstant(0))
        }
      }

      // Store to the chunk's partial slot. No tileRow/tileCol in output address — each
      // TG already covers a full (blockM × blockN) block via the grid dimensions.
      let chunkOffset = zIndex * b.intConstant(M * N)
      let cRowBase = tileRow * b.intConstant(blockM) + sgRow * b.intConstant(8)
      let cColBase = tileCol * b.intConstant(blockN) + sgCol * b.intConstant(8)
      let cOffset = chunkOffset + cRowBase * b.intConstant(N) + cColBase
      b.simdgroupStore(acc, cellId: outCell, offset: cOffset, stride: N)
    } else {
      // Per-frame mode: z is FRAME index.
      let leftFrameBase = leftIsFrameAware ? zIndex * b.intConstant(M * K) : b.intConstant(0)
      let rightFrameBase = rightIsFrameAware ? zIndex * b.intConstant(K * N) : b.intConstant(0)
      let outFrameBase = outIsFrameAware ? zIndex * b.intConstant(M * N) : b.intConstant(0)

      emitKLoopMac(leftFrameBase: leftFrameBase, rightFrameBase: rightFrameBase)

      let cRowBase = tileRow * b.intConstant(blockM) + sgRow * b.intConstant(8)
      let cColBase = tileCol * b.intConstant(blockN) + sgCol * b.intConstant(8)
      let cOffset = outFrameBase + cRowBase * b.intConstant(N) + cColBase
      b.simdgroupStore(acc, cellId: outCell, offset: cOffset, stride: N)
    }

    ctx.values[nodeId] = .empty
  }
}

import Foundation

extension LazyOp {
  /// Emit a tiled GEMM kernel using simdgroup matrix operations.
  ///
  /// For A[M,K] @ B[K,N] → C[M,N] with 8×8×8 WMMA tiles:
  /// - Each SIMD group (32 threads) handles one 8×8 output tile
  /// - K-loop iterates in steps of 8
  /// - One A tile and one B tile loaded per K-step
  /// - One WMMA call per K-step
  ///
  /// For frame-based blocks, gid.z indexes into per-frame tensor memory.
  /// Frame-aware tensors are addressed as memory[cell + frame*tensorSize + offset].
  func emitGemm(
    b: IRBuilder, ctx: IRContext, g: Graph, node: Node, nodeId: NodeID, ops: inout [UOp]
  ) throws {
    guard case .gemm(let M, let N, let K) = self else { return }

    // Resolve memory cells for A, B, and C
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

    // Threadgroup position determines which 8×8 output tile this SIMD group handles
    let tileRow = b.threadgroupPositionY()  // gid.y
    let tileCol = b.threadgroupPositionX()  // gid.x

    // Frame offset for per-frame GEMM (gid.z = frame index, 0 for static)
    let frameIndex = b.threadgroupPositionZ()  // gid.z
    let leftFrameBase =
      ctx.frameAwareTensorCells.contains(leftCell)
      ? frameIndex * b.intConstant(M * K) : b.intConstant(0)
    let rightFrameBase =
      ctx.frameAwareTensorCells.contains(rightCell)
      ? frameIndex * b.intConstant(K * N) : b.intConstant(0)
    let outFrameBase =
      ctx.frameAwareTensorCells.contains(outCell)
      ? frameIndex * b.intConstant(M * N) : b.intConstant(0)

    // 1. Zero-initialized accumulator
    let acc = b.simdgroupMatrixZero()

    // 2. K-loop: load A tile, B tile, multiply-accumulate (in-place on acc)
    b.loop(kSteps) { k in
      // A[M,K] row-major: tile at row tileRow*8, col k*8 → offset = tileRow*8*K + k*8
      let aOffset = leftFrameBase + tileRow * b.intConstant(8 * K) + k * b.intConstant(8)
      let aTile = b.simdgroupLoad(leftCell, offset: aOffset, stride: K)

      // B[K,N] row-major: tile at row k*8, col tileCol*8 → offset = k*8*N + tileCol*8
      let bOffset = rightFrameBase + k * b.intConstant(8 * N) + tileCol * b.intConstant(8)
      let bTile = b.simdgroupLoad(rightCell, offset: bOffset, stride: N)

      b.simdgroupMultiplyAccumulate(aTile, bTile, acc)
    }

    // 3. Store result: C[M,N] row-major at (tileRow*8, tileCol*8)
    let cOffset = outFrameBase + tileRow * b.intConstant(8 * N) + tileCol * b.intConstant(8)
    b.simdgroupStore(acc, cellId: outCell, offset: cOffset, stride: N)

    // Mark GEMM output as "tensor in memory" so downstream blocks read via tload,
    // not via scalar global communication (defineGlobal/loadGlobal).
    // simdgroupStore sets ctx.values[nodeId] to a variable via useVariable —
    // override that to prevent MetalRenderer from promoting it to a global.
    ctx.values[nodeId] = .empty

    _ = (M, N)  // used later for dispatch grid sizing
  }
}

import Foundation

public typealias NodeID = Int
public typealias VarID = Int
public typealias ConstantID = Int
public typealias CellID = Int
public typealias GradID = Int
public typealias ChannelNumber = Int

// MARK: - Tensor Emit Helpers

/// Emit a binary op for scalars or tensors.
func emitBinaryOp(
  b: IRBuilder,
  g: Graph,
  node: Node,
  inputs: [Lazy],
  op: (Expr, Expr) -> Expr
) throws {
  let a = try b.readInput(node, inputs, at: 0)
  let c = try b.readInput(node, inputs, at: 1)
  let result = op(a, c)

  try b.writeOutput(node, result)
}

/// Emit a unary op for scalars or tensors.
func emitUnaryOp(
  b: IRBuilder,
  g: Graph,
  node: Node,
  inputs: [Lazy],
  op: (Expr) -> Expr
) throws {
  let a = try b.readInput(node, inputs, at: 0)
  let result = op(a)
  try b.writeOutput(node, result)
}

/// Emit a ternary op for scalars or tensors.
func emitTernaryOp(
  b: IRBuilder,
  g: Graph,
  node: Node,
  inputs: [Lazy],
  op: (Expr, Expr, Expr) -> Expr
) throws {
  let a = try b.readInput(node, inputs, at: 0)
  let c = try b.readInput(node, inputs, at: 1)
  let d = try b.readInput(node, inputs, at: 2)

  let result = op(a, c, d)

  try b.writeOutput(node, result)
}

// frontend
public enum LazyOp {
  case add, sub, div, mul, abs, sign, sin, cos, tan, tanh, exp, log, log10, sqrt, atan2, gt, gte,
    lte,
    lt, eq,
    gswitch, mix, pow, floor, ceil, round, mod, min, max, and, or, xor
  case mse  // mean squared error per-sample: (a-b)^2

  // FFT-based spectral loss with backprop support
  case spectralLossFFT(
    windowSize: Int,
    hop: Int,
    useHann: Bool,
    useLogMagnitude: Bool,
    windowCell: CellID,
    fft1Cell: CellID,
    fft2Cell: CellID,
    mag1Cell: CellID,
    mag2Cell: CellID,
    scratchCell: CellID
  )
  case spectralLossFFTGradSpec(
    windowSize: Int,
    hop: Int,
    useLogMagnitude: Bool,
    fft1Cell: CellID,
    fft2Cell: CellID,
    mag1Cell: CellID,
    mag2Cell: CellID,
    gradSpec1Cell: CellID,
    gradSpec2Cell: CellID
  )
  case spectralLossFFTGradIFFT(
    windowSize: Int,
    hop: Int,
    gradSpec1Cell: CellID,
    gradSpec2Cell: CellID,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID,
    windowCell: CellID
  )
  // Inline gradient computation for spectralLossFFT - recomputes DFT to avoid race conditions
  case spectralLossFFTGradInline(
    windowSize: Int,
    hop: Int,
    useHann: Bool,
    windowCell: CellID,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID
  )
  // Read gradient from frame-indexed storage (returns grad1)
  case spectralLossFFTGradRead(
    windowSize: Int,
    hop: Int,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID
  )
  // Read second gradient from frame-indexed storage (returns grad2)
  case spectralLossFFTGradRead2(
    windowSize: Int,
    hop: Int,
    gradTime2Cell: CellID
  )

  // Batched FFT-based spectral loss: processes [B] SignalTensors independently per batch element
  case spectralLossFFTBatched(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    useHann: Bool,
    useLogMagnitude: Bool,
    windowCell: CellID,
    fft1Cell: CellID,
    fft2Cell: CellID,
    mag1Cell: CellID,
    mag2Cell: CellID,
    scratchCell: CellID
  )
  // Reduce per-batch spectral losses written by spectralLossFFTBatched into scalar mean.
  case spectralLossFFTBatchedReduce(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    scratchCell: CellID
  )
  case spectralLossFFTBatchedGradSpec(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    useLogMagnitude: Bool,
    fft1Cell: CellID,
    fft2Cell: CellID,
    mag1Cell: CellID,
    mag2Cell: CellID,
    gradSpec1Cell: CellID,
    gradSpec2Cell: CellID
  )
  case spectralLossFFTBatchedGradIFFT(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    gradSpec1Cell: CellID,
    gradSpec2Cell: CellID,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID,
    windowCell: CellID
  )
  case spectralLossFFTBatchedGradRead(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID,
    outputCell: CellID
  )
  case spectralLossFFTBatchedGradRead2(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    gradTime2Cell: CellID,
    outputCell: CellID
  )

  // selectRow: extract a single row from a 2D tensor using dynamic index
  // Input: [tensor2D, rowIndex] where rowIndex is floored to int
  // Output: 1D tensor [numCols]
  case selectRow
  // peekGradWrite: write per-frame scalar gradient and interpolation metadata for peek backward
  // Input: [gradOutput (scalar), index, channel]
  case peekGradWrite(
    gradWriteCell: CellID, floorPosCell: CellID, nextPosCell: CellID, fracCell: CellID,
    channelSize: Int, numChannels: Int, maxFrameCount: Int)
  // peekGradReduce: sum per-frame peek contributions into tensor gradient
  // Input: [peekGradWritePass] (for ordering)
  case peekGradReduce(
    gradWriteCell: CellID, floorPosCell: CellID, nextPosCell: CellID, fracCell: CellID,
    gradCell: CellID, totalSize: Int, maxFrameCount: Int)
  // selectRowGradWrite: write gradient to frame-indexed storage (deterministic, no atomics)
  // Input: [gradOutput (1D tensor), rowIndex]
  // Writes to gradWriteCell[frame * numCols + col] and rowIdxCell[frame]
  // sampleInline: interpolated sampling along axis 0 for any-rank tensor (N >= 2)
  // Input: [tensorND, index] where index is in [0, D0)
  // Output: tensor with shape.dropFirst() — uses frame-indexed storage for SIMD safety
  case sampleInline(scratchCell: CellID, numRows: Int, remainingShape: [Int])
  // sampleGradWrite: write interpolation-weighted gradients to frame-indexed storage
  case sampleGradWrite(
    floorGradCell: CellID, ceilGradCell: CellID, rowIdxCell: CellID, fracCell: CellID,
    numRows: Int, remainingShape: [Int], maxFrameCount: Int)
  // sampleGradReduce: sum gradient contributions from all frames for each tensor position
  case sampleGradReduce(
    floorGradCell: CellID, ceilGradCell: CellID, rowIdxCell: CellID, fracCell: CellID,
    gradCell: CellID, numRows: Int, remainingShape: [Int], maxFrameCount: Int)
  case selectRowGradWrite(gradWriteCell: CellID, rowIdxCell: CellID, numRows: Int, numCols: Int)
  // selectRowGradReduce: sum contributions from all frames for each tensor position
  // Input: [gradWritePass] (for ordering)
  // Reads from frame-indexed storage and accumulates to gradCell
  case selectRowGradReduce(
    gradWriteCell: CellID, rowIdxCell: CellID, gradCell: CellID, numRows: Int, numCols: Int,
    maxFrameCount: Int)
  case selector  // selector(mode, options[])
  case memoryRead(CellID)
  case memoryWrite(CellID)
  case memoryAccumulate(CellID)  // Atomic add to memory cell
  case memoryCellSum(CellID, Int)  // Sum all elements in a memory cell (cell, size)
  case tensorAccumulate(CellID)  // Atomic add tensor elements to memory region
  // Two-pass deterministic cross-frame GEMM reduction:
  // 1) chunked GEMM writes partial sums [chunkCount, M, N]
  // 2) chunk reduction accumulates those partials into target cell.
  // Triggered by GEMMPass when matching tensorAccumulate(view* -> gemm(...)).
  case gemmChunkPartials(Int, Int, Int, Bool, Bool, Int, Int)  // (M, N, K, transA, transB, chunkSize, chunkCount)
  case chunkPartialsReduceToCell(CellID, Int, Int, Int, Bool)  // (targetCell, M, N, chunkCount, outputTransposed)
  case historyWrite(CellID)
  case historyReadWrite(CellID)
  case param(CellID)
  case latch(CellID)
  case click(CellID)
  case historyRead(CellID)
  case phasor(CellID)
  case deterministicPhasor  // Stateless phasor for constant frequency - parallelizable
  case accum(CellID)
  case noise(CellID)
  case constant(Float)
  case output(Int)
  case input(Int)
  case tensorRef(TensorID)
  case seq  // Sequential execution - returns value of last input

  // Tensor operations (historyRead/historyWrite handle tensors automatically based on cell size)
  case conv1d(Int)  // 1D convolution, Int is kernel size
  case conv2d(Shape)  // 2D convolution, Shape is kernel shape [kH, kW]
  case sum  // Reduce tensor to scalar by summing all elements
  case sumAxis(Int)  // Reduce along a specific axis
  case sumMulAxis0  // Fused reduction: sum over axis 0 of elementwise mul for 2D tensors
  case gemm(Int, Int, Int, Bool, Bool)  // Matrix multiply via tensor cores: gemm(M, N, K, transA, transB)
  /// Element-parallel matmul for non-8-aligned M/N/K.
  /// Dispatches perFrameScaled(M*N): one thread per output element, inner K-loop.
  case gemmSmall(Int, Int, Int, Bool, Bool)  // M, N, K, transA, transB
  case maxAxis(Int)  // Reduce along axis keeping maximum
  case meanAxis(Int)  // Reduce along axis computing mean
  case reshape(Shape)  // Reshape tensor (metadata only, no data movement)
  case transpose([Int])  // Transpose/permute axes (metadata only)
  case shrink([(Int, Int)?])  // Shrink/slice tensor (metadata only, no data movement)
  case pad([(Int, Int)])  // Pad tensor with zeros (virtual view, conditional reads)
  case expandView(Shape)  // Broadcast size-1 dims to target shape (stride=0 view, no data copy)
  case repeatView([Int])  // Tile tensor by repeating along each dim (modular index view, no data copy)
  case asStrided(Shape, [Int])  // View with custom strides (for pool/im2col operations)
  case peek  // Read from 2D tensor at (index, channel) with interpolation - lazy version
  case overlapAdd(Int, Int, CellID, CellID, CellID)  // Overlap-add: windowSize, hopSize, outputRingCell, readPosCell, counterCell
  // overlapAddGradStore: store per-frame output gradient to shared memory
  case overlapAddGradStore(gradStoreCell: CellID)
  // overlapAddGradGather: gather stored gradients into per-frame gradient tensor
  case overlapAddGradGather(
    windowSize: Int, hopSize: Int,
    gradStoreCell: CellID, gradInputCell: CellID)

  // bufferViewGradStore: store per-frame tensor gradient to frame-indexed memory
  case bufferViewGradStore(gradCell: CellID, windowSize: Int)
  // bufferViewGradRead: sum overlapping window contributions → scalar gradient
  case bufferViewGradRead(gradCell: CellID, windowSize: Int)

  // Gradient-specific operations (used by Gradients.swift)
  case neg  // Unary negation: -x
  case expand(Shape)  // Broadcast scalar to tensor shape (sum backward)
  case expandAxis(Shape, Int)  // Broadcast along a specific axis (sumAxis backward)
  case gradPhasor(NodeID)  // Gradient for phasor: needs frame index context
  case gradDeterministicPhasor  // Gradient for deterministic phasor

  /// View-only ops: metadata transforms that emit no compute code.
  /// Used to skip these ops during shape transition detection, tensor block
  /// splitting, and scalar node extraction.
  public var isViewOnly: Bool {
    switch self {
    case .reshape, .transpose, .shrink, .pad, .expandView:
      return true
    default:
      return false
    }
  }

  /// Inherently scalar stateful ops with single-cell state.
  /// These must not receive a tensorIndex (which would cause indexed memory
  /// access on single-cell state, corrupting adjacent memory).
  public var isInherentlyScalar: Bool {
    switch self {
    case .accum, .phasor, .click, .latch, .noise:
      return true
    default:
      return false
    }
  }
}

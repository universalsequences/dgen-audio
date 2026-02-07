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
    useHann: Bool,
    windowCell: CellID,
    fft1Cell: CellID,
    fft2Cell: CellID,
    mag1Cell: CellID,
    mag2Cell: CellID,
    scratchCell: CellID
  )
  case spectralLossFFTGradSpec(
    windowSize: Int,
    fft1Cell: CellID,
    fft2Cell: CellID,
    mag1Cell: CellID,
    mag2Cell: CellID,
    gradSpec1Cell: CellID,
    gradSpec2Cell: CellID
  )
  case spectralLossFFTGradIFFT(
    windowSize: Int,
    gradSpec1Cell: CellID,
    gradSpec2Cell: CellID,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID,
    windowCell: CellID
  )
  // Inline gradient computation for spectralLossFFT - recomputes DFT to avoid race conditions
  case spectralLossFFTGradInline(
    windowSize: Int,
    useHann: Bool,
    windowCell: CellID,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID
  )
  // Read gradient from frame-indexed storage (returns grad1)
  case spectralLossFFTGradRead(
    windowSize: Int,
    gradTime1Cell: CellID,
    gradTime2Cell: CellID
  )
  // Read second gradient from frame-indexed storage (returns grad2)
  case spectralLossFFTGradRead2(
    windowSize: Int,
    gradTime2Cell: CellID
  )
  // selectRow: extract a single row from a 2D tensor using dynamic index
  // Input: [tensor2D, rowIndex] where rowIndex is floored to int
  // Output: 1D tensor [numCols]
  case selectRow
  // peekRowInline: interpolated row extraction with frame-local computation
  // Input: [tensor2D, rowIndex] where rowIndex is interpolated between floor and ceil
  // Output: 1D tensor [numCols] - uses frame-indexed storage for SIMD safety
  // scratchCell stores frame-indexed outputs: frame * numCols + col
  case peekRowInline(scratchCell: CellID, numRows: Int, numCols: Int)
  // peekRowGradWrite: write gradients for both floor and ceil rows to frame-indexed storage
  // Input: [gradOutput (1D tensor), rowIndex]
  case peekRowGradWrite(
    floorGradCell: CellID, ceilGradCell: CellID, rowIdxCell: CellID, fracCell: CellID,
    numRows: Int, numCols: Int, maxFrameCount: Int)
  // peekRowGradReduce: sum gradient contributions from all frames for each tensor position
  case peekRowGradReduce(
    floorGradCell: CellID, ceilGradCell: CellID, rowIdxCell: CellID, fracCell: CellID,
    gradCell: CellID, numRows: Int, numCols: Int, maxFrameCount: Int)
  // selectRowGradWrite: write gradient to frame-indexed storage (deterministic, no atomics)
  // Input: [gradOutput (1D tensor), rowIndex]
  // Writes to gradWriteCell[frame * numCols + col] and rowIdxCell[frame]
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
  case reshape(Shape)  // Reshape tensor (metadata only, no data movement)
  case transpose([Int])  // Transpose/permute axes (metadata only)
  case shrink([(Int, Int)?])  // Shrink/slice tensor (metadata only, no data movement)
  case pad([(Int, Int)])  // Pad tensor with zeros (virtual view, conditional reads)
  case expandView(Shape)  // Broadcast size-1 dims to target shape (stride=0 view, no data copy)
  case repeatView([Int])  // Tile tensor by repeating along each dim (modular index view, no data copy)
  case asStrided(Shape, [Int])  // View with custom strides (for pool/im2col operations)
  case peek  // Read from 2D tensor at (index, channel) with interpolation - lazy version
  case fft(Int, Int, CellID, CellID, CellID, CellID)  // FFT transform: windowSize, hopSize, scratchCell, ringBufferCell, writePosCell, counterCell
  case ifft(Int, Int, CellID, CellID, CellID, CellID)  // IFFT transform: windowSize, hopSize, scratchCell, outputRingCell, readPosCell, counterCell

  // Gradient-specific operations (used by Gradients.swift)
  case neg  // Unary negation: -x
  case expand(Shape)  // Broadcast scalar to tensor shape (sum backward)
  case expandAxis(Shape, Int)  // Broadcast along a specific axis (sumAxis backward)
  case gradPhasor(NodeID)  // Gradient for phasor: needs frame index context
  case gradDeterministicPhasor  // Gradient for deterministic phasor
}

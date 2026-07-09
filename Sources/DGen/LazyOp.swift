import Foundation

public typealias NodeID = Int
public typealias VarID = Int
public typealias ConstantID = Int
public typealias CellID = Int
public typealias GradID = Int
public typealias ChannelNumber = Int

/// Distance mode used by FFT spectral loss in magnitude (or log-magnitude) space.
public enum SpectralLossMode: String, Codable, Sendable {
  /// Squared difference `(a - b)^2` (default).
  case l2
  /// Absolute difference `|a - b|`.
  case l1
}

public enum ModulatedParamMode: String, Codable, Sendable {
  case additive
  case multiplicative
  case semitone
}

public struct ModulatedParamLane: Hashable, Codable, Sendable {
  public let modulatorChannel: Int
  public let depthCellId: CellID

  public init(modulatorChannel: Int, depthCellId: CellID) {
    self.modulatorChannel = modulatorChannel
    self.depthCellId = depthCellId
  }
}

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
  case add, sub, div, mul, abs, sign, sin, cos, tan, atan, tanh, exp, log, log10, sqrt, atan2, gt, gte,
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
    lossMode: SpectralLossMode,
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
    lossMode: SpectralLossMode,
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

  // Accelerate-framework FFT (C backend only). Calls vDSP_fft_zip for in-place
  // complex FFT. Writes result to reCell/imCell, exposed downstream as two [N]
  // tensorRef views. Forward variant takes a real [N] input (im cleared internally).
  case acceleratedFFT(windowSize: Int, reCell: CellID, imCell: CellID)
  // Accelerate-framework IFFT (C backend only). Takes two [N] inputs (re, im),
  // calls vDSP_fft_zip with FFT_INVERSE, normalizes by 1/N, returns real [N].
  case acceleratedIFFT(windowSize: Int, reCell: CellID, imCell: CellID)

  // Dedicated hop-rate phase vocoder pitch shifter. Inputs:
  //   [xRe, xIm, pitchRatio, hopCounter]
  // Maintains previous analysis phase and synthesis phase internally, writes
  // remapped complex output to reOutCell/imOutCell, exposed downstream as
  // tensorRef views.
  case phaseVocoderPitchShift(
    windowSize: Int, hopSize: Int,
    prevAnalysisPhaseCell: CellID,
    synthPhaseCell: CellID,
    tempMagCell: CellID,
    tempOmegaCell: CellID,
    initCell: CellID,
    reOutCell: CellID,
    imOutCell: CellID
  )

  // Partitioned spectral convolution (UPOLS). Inputs: live (re, im) spectra of
  // shape [N] at hop-rate, and two static [K, N] IR partition tensors (re, im).
  // Writes last K input spectra into a mirror-layout [2K, N] ring per channel,
  // computes Y[n] = Σ_{k=0..K-1} H[k] * X[n-k] as a complex MAC, writes to the
  // two owned output cells. Exposed downstream as two [N] tensorRef views.
  case partitionedSpectralConvolve(
    K: Int, N: Int, hopSize: Int,
    ringReCell: CellID, ringImCell: CellID,
    hopCounterCell: CellID, partitionCounterCell: CellID,
    reOutCell: CellID, imOutCell: CellID)

  // Batched FFT-based spectral loss: processes [B] SignalTensors independently per batch element
  case spectralLossFFTBatched(
    windowSize: Int,
    batchSize: Int,
    hop: Int,
    useHann: Bool,
    useLogMagnitude: Bool,
    lossMode: SpectralLossMode,
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
    lossMode: SpectralLossMode,
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
  /// Host-routed modulation destination. Inputs are:
  /// [baseParam].
  /// Scalar parameter cells are loaded as lane-uniform values so the same op is
  /// safe in scalar and SIMD C frame loops.
  case modulatedParam(
    mode: ModulatedParamMode,
    min: Float,
    max: Float,
    baseCellId: CellID,
    activeCellId: CellID,
    lanes: [ModulatedParamLane])
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
  /// Threadgroup-staged variant of gemmChunkPartials: same two-pass reduction
  /// shape, but the inner per-frame GEMM uses cooperative threadgroup-memory staging.
  /// (M, N, K, transA, transB, chunkSize, chunkCount, blockM, blockN, blockK)
  case gemmStagedChunkPartials(Int, Int, Int, Bool, Bool, Int, Int, Int, Int, Int)
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
  case tensorNoise(CellID, CellID, Int)  // stateCell, outputCell, size — N independent random values per frame
  case hopTensorNoise(CellID, CellID, Int)  // stateCell, outputCell, size — regenerates only on hop boundaries (counter input == 0), holds between hops
  case spectrumDelay(CellID, CellID, CellID, Int, Int)  // ringCell, rowCell, outputCell, N, hops — N-bin spectrum from `hops` hops ago
  case spectrumDelayMod(CellID, CellID, CellID, Int, Int)  // ringCell, rowCell, outputCell, N, maxHops — fractional delay driven by a scalar `delay` input (0..maxHops)
  case constant(Float)
  case hostSampleRate
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
  /// Threadgroup-staged matmul: cooperatively stages A/B strips into threadgroup
  /// memory and computes a (blockM × blockN) output region per threadgroup.
  /// Tuple: (M, N, K, transA, transB, blockM, blockN, blockK).
  case gemmStaged(Int, Int, Int, Bool, Bool, Int, Int, Int)
  /// Element-parallel matmul for non-8-aligned M/N/K.
  /// Dispatches perFrameScaled(M*N): one thread per output element, inner K-loop.
  case gemmSmall(Int, Int, Int, Bool, Bool)  // M, N, K, transA, transB
  case maxAxis(Int)  // Reduce along axis keeping maximum
  case meanAxis(Int)  // Reduce along axis computing mean
  case cumsum(Int)  // Cumulative (prefix) sum along an axis. Output shape == input shape.
  case gather  // Indexed read: gather(source, indices). Output shape == indices shape.
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
  case gradDeterministicPhasor  // Gradient for deterministic phasor
  /// Stores a frame's upstream gradient and reset gate for a temporal reverse scan.
  case temporalGradStore(
    gradCell: CellID, resetCell: CellID, elementCount: Int)
  /// Computes an exclusive, reset-aware suffix sum over stored frame gradients.
  case temporalGradScan(
    gradCell: CellID, resetCell: CellID, outputCell: CellID, elementCount: Int)
  /// Reads one frame/element from a completed temporal gradient scan.
  case temporalGradRead(
    outputCell: CellID, shape: Shape, scaleBySampleRate: Bool)

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

  /// GEMM variants that own their dispatch grid (excludes `.gemmSmall`, which
  /// uses `perFrameScaled` and carries a shape-driven `tensorIndex`).
  public var isSelfDispatchedGemm: Bool {
    switch self {
    case .gemm, .gemmStaged, .gemmChunkPartials, .gemmStagedChunkPartials:
      return true
    default:
      return false
    }
  }

  /// Ops whose emit handles its own iteration (internal `b.loop` /
  /// `b.parallelRange` / explicit vDSP calls) and does NOT consume the
  /// block-level `tensorIndex`. Such ops must not force a block's body to be
  /// wrapped in `parallelRange(block.shape.reduce(1,*))` — doing so makes the
  /// emit run `block.shape` times with no benefit. `.tensorRef` also counts:
  /// it emits no code at all. `.seq` just returns its last input's value.
  ///
  /// Note: forward `.selectRow`, `.peek`, `.sampleInline` still rely on the
  /// block's tensorIndex and must NOT be listed here. GEMM and conv variants
  /// are explicitly isolated into their own blocks by block formation, so
  /// their self-iteration does not interact with a sibling compute op's wrapper.
  public var emitsInternalIteration: Bool {
    switch self {
    case .tensorRef,
      .seq,
      .acceleratedFFT, .acceleratedIFFT,
      .phaseVocoderPitchShift,
      .overlapAdd, .overlapAddGradStore, .overlapAddGradGather,
      .bufferViewGradStore, .bufferViewGradRead,
      .temporalGradStore, .temporalGradScan,
      .partitionedSpectralConvolve,
      .tensorNoise,
      .hopTensorNoise,
      .spectrumDelay,
      .spectrumDelayMod,
      .gemm, .gemmStaged, .gemmChunkPartials, .gemmStagedChunkPartials,
      .gemmSmall,
      .conv1d, .conv2d, .cumsum, .gather,
      .tensorAccumulate, .chunkPartialsReduceToCell,
      .spectralLossFFT, .spectralLossFFTGradSpec, .spectralLossFFTGradIFFT,
      .spectralLossFFTGradInline, .spectralLossFFTGradRead, .spectralLossFFTGradRead2,
      .spectralLossFFTBatched, .spectralLossFFTBatchedReduce,
      .spectralLossFFTBatchedGradSpec, .spectralLossFFTBatchedGradIFFT,
      .spectralLossFFTBatchedGradRead2:
      return true
    default:
      return false
    }
  }
}

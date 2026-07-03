public enum Lazy: Hashable {
  case constant(ConstantID, Float)
  case global(VarID)
  case variable(VarID, NodeID?)
  case gradient(GradID)
  case empty
}

public enum CastType {
  case int
  case float
}

// IR (intermediate representation) is called UOp and consists of an
// operator (Op) and value (the variable it's result is bound to)
public enum Op {
  case load(CellID)
  case store(CellID, Lazy)
  case delay1(CellID, Lazy)
  case mse(Lazy, Lazy)
  case mutate(Lazy, Lazy)
  case add(Lazy, Lazy)
  case sub(Lazy, Lazy)
  case mul(Lazy, Lazy)
  case div(Lazy, Lazy)
  case abs(Lazy)
  case sign(Lazy)
  case sin(Lazy)
  case cos(Lazy)
  case and(Lazy, Lazy)
  case or(Lazy, Lazy)
  case xor(Lazy, Lazy)
  case tan(Lazy)
  case atan(Lazy)
  case tanh(Lazy)
  case exp(Lazy)
  case log(Lazy)
  case log10(Lazy)
  case sqrt(Lazy)
  case pow(Lazy, Lazy)
  case atan2(Lazy, Lazy)
  case mod(Lazy, Lazy)
  case gt(Lazy, Lazy)
  case gte(Lazy, Lazy)
  case lte(Lazy, Lazy)
  case lt(Lazy, Lazy)
  case eq(Lazy, Lazy)
  case min(Lazy, Lazy)
  case max(Lazy, Lazy)
  case floor(Lazy)
  case ceil(Lazy)
  case round(Lazy)
  case noise(CellID)
  case memoryRead(CellID, Lazy)
  case memoryWrite(CellID, Lazy, Lazy)
  case memoryAccumulate(CellID, Lazy, Lazy)  // Atomic add to memory cell
  /// Read one float from `memory[cell + offset]` and, in a SIMD block, broadcast
  /// it across all 4 lanes (`vdupq_n_f32`). In scalar mode this is equivalent
  /// to `memoryRead`. Used for runtime-variable kernel coefficients and other
  /// lane-uniform loads inside NEON inner loops.
  case simdBroadcastLoad(CellID, Lazy)
  /// Broadcast a loop-invariant scalar float variable across all 4 SIMD lanes
  /// (`vdupq_n_f32(t<id>)` / `vdupq_n_f32(t<id>[i])` for frame-scoped globals).
  /// Inserted by the C-backend region-loop SIMD upgrade so element loops can
  /// reference frame-scope scalars. Scalar mode degrades to a plain copy.
  case broadcastScalar(Lazy)
  case latch(Lazy, Lazy)
  case beginIf(Lazy)
  case gswitch(Lazy, Lazy, Lazy)
  case selector(Lazy, [Lazy])  // selector(mode, options[])
  case endIf
  case defineGlobal(VarID)
  case defineConstant(ConstantID, Float)
  case loadGlobal(VarID)
  case beginLoop(Lazy, Int)
  case beginForLoop(Lazy, Lazy)  // (loopVariable, count) - step is always 1
  case beginReverseLoop(Lazy)  // reverse loop: for (int i = count-1; i >= 0; i--)
  case endLoop
  case beginRange(Lazy, Lazy)
  case endRange
  case beginParallelRange(Int, Int)  // count - iterations are independent, can be parallelized
  case endParallelRange
  case setThreadCountScale(Int)  // dispatch threads = frameCount * scale
  case setFrameIndex(Lazy)  // override frame index used for outputs/gradients
  case output(ChannelNumber, Lazy)
  case input(ChannelNumber)
  case frameCount
  case hostSampleRate
  case frameIndex
  case threadIndex
  case loadTape(Lazy, Lazy)
  case cast(Lazy, CastType)
  case identity(Lazy)  // Identity copy: float t_new = t_old; Used when folding x*1, x+0, etc.
  case declareVar(Lazy)  // Declares and initializes a variable: float t = value;
  case reshape([Int])  // View op: reshape to new shape - renders to nothing but prevents SIMD
  case transpose([Int])  // View op: transpose with permutation - renders to nothing but prevents SIMD
  case shrink([(Int, Int)?])  // View op: shrink/slice - renders to nothing but prevents SIMD
  case pad([(Int, Int)])  // View op: pad with zeros - renders to nothing but prevents SIMD
  case expandView([Int])  // View op: broadcast size-1 dims via stride=0 - renders to nothing but prevents SIMD
  case repeatView([Int])  // View op: tile tensor via modular indexing - renders to nothing but prevents SIMD
  case broadcastAccess  // Marker: broadcast indexing used - renders to nothing but prevents SIMD
  case sumAxisMarker(Int, Int, [Int], [Int], Bool, Bool)  // Marker: sumAxis(nodeId, axis, inShape, outShape, inFrameAware, outFrameAware)
  case maxAxisMarker(Int, Int, [Int], [Int], Bool, Bool)  // Marker: maxAxis(nodeId, axis, inShape, outShape, inFrameAware, outFrameAware)
  case meanAxisMarker(Int, Int, [Int], [Int], Bool, Bool)  // Marker: meanAxis(nodeId, axis, inShape, outShape, inFrameAware, outFrameAware)
  case expandAxisMarker(Int, Int, [Int], [Int], Bool, Bool)  // Marker: expandAxis(nodeId, axis, inShape, outShape, inFrameAware, outFrameAware)

  // Hop-based execution control (for FFT/spectral processing)
  case beginHopCheck(Lazy)  // if (counter == 0.0f) { - runs block only when counter is 0
  case endHopCheck  // } - closes the hop check conditional

  // Threadgroup position (for GEMM 2D/3D dispatch)
  case threadgroupPositionX  // gid.x — column tile index
  case threadgroupPositionY  // gid.y — row tile index
  case threadgroupPositionZ  // gid.z — frame index (for per-frame GEMM)
  case threadIndexInThreadgroup  // [[thread_index_in_threadgroup]] — flat tid within TG
  case simdgroupIndexInThreadgroup  // [[simdgroup_index_in_threadgroup]] — SIMD group ID within TG

  // GEMM / simdgroup matrix operations (Metal tensor cores)
  case simdgroupMatrixZero  // declare simdgroup_float8x8, zero-initialized
  case simdgroupLoad(CellID, Lazy, Int, Bool)  // simdgroup_load(dest, memory[cell] + offset, stride, transpose)
  /// Load an 8×8 matrix tile from threadgroup scratch memory rather than device memory.
  /// (scratchId, offset, stride, transpose).
  case simdgroupLoadScratch(scratchId: Int, Lazy, Int, Bool)
  case simdgroupStore(Lazy, CellID, Lazy, Int)  // simdgroup_store(src, memory[cell] + offset, stride)
  case simdgroupMultiplyAccumulate(Lazy, Lazy, Lazy)  // acc = a * b + acc

  // Accelerate-framework FFT call (C backend only). Emits vDSP_fft_zip on the
  // [reCell, imCell] buffers. For inverse, also scales by 1/N.
  case acceleratedFFTCall(log2N: Int, reCell: CellID, imCell: CellID, inverse: Bool)

  // Partitioned spectral complex-MAC (C backend only). Emits a tight loop
  // calling vDSP_zvma per partition to compute
  //   Y[n] = Σ_{k=0..K-1} X_ring[(p+K-k)*N .. +N] * H[k*N .. +N]
  // where p is read from partitionIdxCell. X_ring lives in ringReCell/ringImCell
  // (mirror-layout [2K, N]), H in irReCell/irImCell ([K, N]), Y written to
  // reOutCell/imOutCell (both [N]). Y is zeroed first.
  case partitionedSpectralMACCall(
    K: Int, N: Int,
    partitionIdxCell: CellID,
    ringReCell: CellID, ringImCell: CellID,
    irReCell: CellID, irImCell: CellID,
    reOutCell: CellID, imOutCell: CellID)

  // Threadgroup shared memory (on-chip SRAM for FFT scratch)
  case threadgroupArrayDecl(scratchId: Int, size: Int)  // declare threadgroup float scratch_N[size]
  case threadgroupRead(scratchId: Int, Lazy)    // read scratch_N[offset]
  case threadgroupWrite(scratchId: Int, Lazy, Lazy)   // scratch_N[offset] = value
  case threadgroupBarrier  // threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  public var isDefineGlobal: Bool {
    if case .defineGlobal = self { return true }
    return false
  }

  /// Returns the memory cell ID if this operation accesses memory, nil otherwise.
  public var memoryCellId: CellID? {
    switch self {
    case .load(let cellId), .store(let cellId, _), .delay1(let cellId, _),
      .memoryRead(let cellId, _), .memoryWrite(let cellId, _, _),
      .memoryAccumulate(let cellId, _, _),
      .simdBroadcastLoad(let cellId, _),
      .noise(let cellId),
      .simdgroupLoad(let cellId, _, _, _), .simdgroupStore(_, let cellId, _, _):
      return cellId
    default:
      return nil
    }
  }

  /// Returns a new Op with Lazy inputs remapped. Used for BPTT to redirect
  /// backward ops from forward-loop variables to per-frame stored values.
  public func remapLazyInputs(_ remap: [Lazy: Lazy]) -> Op {
    func r(_ l: Lazy) -> Lazy { remap[l] ?? l }
    switch self {
    case .store(let c, let v): return .store(c, r(v))
    case .delay1(let c, let a): return .delay1(c, r(a))
    case .mse(let a, let b): return .mse(r(a), r(b))
    case .mutate(let a, let b): return .mutate(r(a), r(b))
    case .add(let a, let b): return .add(r(a), r(b))
    case .sub(let a, let b): return .sub(r(a), r(b))
    case .mul(let a, let b): return .mul(r(a), r(b))
    case .div(let a, let b): return .div(r(a), r(b))
    case .abs(let a): return .abs(r(a))
    case .sign(let a): return .sign(r(a))
    case .sin(let a): return .sin(r(a))
    case .cos(let a): return .cos(r(a))
    case .and(let a, let b): return .and(r(a), r(b))
    case .or(let a, let b): return .or(r(a), r(b))
    case .xor(let a, let b): return .xor(r(a), r(b))
    case .tan(let a): return .tan(r(a))
    case .atan(let a): return .atan(r(a))
    case .tanh(let a): return .tanh(r(a))
    case .exp(let a): return .exp(r(a))
    case .log(let a): return .log(r(a))
    case .log10(let a): return .log10(r(a))
    case .sqrt(let a): return .sqrt(r(a))
    case .pow(let a, let b): return .pow(r(a), r(b))
    case .atan2(let a, let b): return .atan2(r(a), r(b))
    case .mod(let a, let b): return .mod(r(a), r(b))
    case .gt(let a, let b): return .gt(r(a), r(b))
    case .gte(let a, let b): return .gte(r(a), r(b))
    case .lte(let a, let b): return .lte(r(a), r(b))
    case .lt(let a, let b): return .lt(r(a), r(b))
    case .eq(let a, let b): return .eq(r(a), r(b))
    case .min(let a, let b): return .min(r(a), r(b))
    case .max(let a, let b): return .max(r(a), r(b))
    case .floor(let a): return .floor(r(a))
    case .ceil(let a): return .ceil(r(a))
    case .round(let a): return .round(r(a))
    case .memoryRead(let c, let o): return .memoryRead(c, r(o))
    case .memoryWrite(let c, let o, let v): return .memoryWrite(c, r(o), r(v))
    case .simdBroadcastLoad(let c, let o): return .simdBroadcastLoad(c, r(o))
    case .broadcastScalar(let a): return .broadcastScalar(r(a))
    case .memoryAccumulate(let c, let o, let v): return .memoryAccumulate(c, r(o), r(v))
    case .latch(let a, let b): return .latch(r(a), r(b))
    case .gswitch(let c, let a, let b): return .gswitch(r(c), r(a), r(b))
    case .selector(let m, let opts): return .selector(r(m), opts.map { r($0) })
    case .beginForLoop(let v, let c): return .beginForLoop(r(v), r(c))
    case .beginLoop(let i, let s): return .beginLoop(r(i), s)
    case .beginReverseLoop(let i): return .beginReverseLoop(r(i))
    case .beginRange(let s, let e): return .beginRange(r(s), r(e))
    case .beginParallelRange(let c, let s): return .beginParallelRange(c, s)
    case .output(let ch, let v): return .output(ch, r(v))
    case .cast(let e, let t): return .cast(r(e), t)
    case .identity(let a): return .identity(r(a))
    case .declareVar(let v): return .declareVar(r(v))
    case .setFrameIndex(let i): return .setFrameIndex(r(i))
    case .loadTape(let v, let o): return .loadTape(r(v), r(o))
    case .beginIf(let c): return .beginIf(r(c))
    case .beginHopCheck(let c): return .beginHopCheck(r(c))
    case .simdgroupLoad(let c, let o, let s, let t): return .simdgroupLoad(c, r(o), s, t)
    case .simdgroupLoadScratch(let id, let o, let s, let t):
      return .simdgroupLoadScratch(scratchId: id, r(o), s, t)
    case .simdgroupStore(let src, let c, let o, let s): return .simdgroupStore(r(src), c, r(o), s)
    case .simdgroupMultiplyAccumulate(let a, let b, let acc):
      return .simdgroupMultiplyAccumulate(r(a), r(b), r(acc))
    case .threadgroupRead(let id, let offset): return .threadgroupRead(scratchId: id, r(offset))
    case .threadgroupWrite(let id, let offset, let value):
      return .threadgroupWrite(scratchId: id, r(offset), r(value))
    default: return self  // ops without Lazy inputs (load, endLoop, frameIndex, etc.)
    }
  }

  /// Returns a new Op with the cell ID remapped, or nil if no remapping is needed.
  public func withRemappedCellId(_ remapping: [CellID: CellID]) -> Op? {
    // Ops that reference multiple cells need explicit remapping.
    if case .acceleratedFFTCall(let log2N, let reCell, let imCell, let inverse) = self {
      let newRe = remapping[reCell] ?? reCell
      let newIm = remapping[imCell] ?? imCell
      if newRe == reCell && newIm == imCell { return nil }
      return .acceleratedFFTCall(log2N: log2N, reCell: newRe, imCell: newIm, inverse: inverse)
    }
    if case .partitionedSpectralMACCall(
      let K, let N, let pIdx, let rRe, let rIm, let iRe, let iIm, let oRe, let oIm) = self
    {
      let nPIdx = remapping[pIdx] ?? pIdx
      let nRRe = remapping[rRe] ?? rRe
      let nRIm = remapping[rIm] ?? rIm
      let nIRe = remapping[iRe] ?? iRe
      let nIIm = remapping[iIm] ?? iIm
      let nORe = remapping[oRe] ?? oRe
      let nOIm = remapping[oIm] ?? oIm
      if nPIdx == pIdx && nRRe == rRe && nRIm == rIm && nIRe == iRe
        && nIIm == iIm && nORe == oRe && nOIm == oIm { return nil }
      return .partitionedSpectralMACCall(
        K: K, N: N,
        partitionIdxCell: nPIdx,
        ringReCell: nRRe, ringImCell: nRIm,
        irReCell: nIRe, irImCell: nIIm,
        reOutCell: nORe, imOutCell: nOIm)
    }
    guard let cellId = memoryCellId, let newCellId = remapping[cellId] else {
      return nil
    }
    switch self {
    case .load: return .load(newCellId)
    case .store(_, let val): return .store(newCellId, val)
    case .delay1(_, let a): return .delay1(newCellId, a)
    case .noise: return .noise(newCellId)
    case .memoryRead(_, let offset): return .memoryRead(newCellId, offset)
    case .memoryWrite(_, let offset, let value): return .memoryWrite(newCellId, offset, value)
    case .memoryAccumulate(_, let offset, let value):
      return .memoryAccumulate(newCellId, offset, value)
    case .simdBroadcastLoad(_, let offset): return .simdBroadcastLoad(newCellId, offset)
    case .simdgroupLoad(_, let offset, let stride, let transpose):
      return .simdgroupLoad(newCellId, offset, stride, transpose)
    case .simdgroupStore(let src, _, let offset, let stride):
      return .simdgroupStore(src, newCellId, offset, stride)
    default: return nil
    }
  }
}

public struct UOp {
  public let op: Op
  public let value: Lazy
  public var vectorWidth: Int = 1  // 1 = scalar, 4 = SIMD (C NEON)
  public var tensorIndex: Lazy? = nil
  public var scalarType: CastType = .float  // int or float for variable declarations

  /// Whether this UOp uses SIMD vectorization (vectorWidth > 1).
  public var isSimd: Bool { vectorWidth > 1 }

  public init(
    op: Op, value: Lazy, vectorWidth: Int = 1,
    tensorIndex: Lazy? = nil,
    scalarType: CastType = .float
  ) {
    self.op = op
    self.value = value
    self.vectorWidth = vectorWidth
    self.tensorIndex = tensorIndex
    self.scalarType = scalarType
  }
}

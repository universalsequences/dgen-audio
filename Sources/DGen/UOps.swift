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

  // Threadgroup position (for GEMM 2D dispatch)
  case threadgroupPositionX  // gid.x — column tile index
  case threadgroupPositionY  // gid.y — row tile index

  // GEMM / simdgroup matrix operations (Metal tensor cores)
  case simdgroupMatrixZero  // declare simdgroup_float8x8, zero-initialized
  case simdgroupLoad(CellID, Lazy, Int)  // simdgroup_load(dest, memory[cell] + offset, stride)
  case simdgroupStore(Lazy, CellID, Lazy, Int)  // simdgroup_store(src, memory[cell] + offset, stride)
  case simdgroupMultiplyAccumulate(Lazy, Lazy, Lazy)  // acc = a * b + acc

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
      .simdgroupLoad(let cellId, _, _), .simdgroupStore(_, let cellId, _, _):
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
    case .simdgroupLoad(let c, let o, let s): return .simdgroupLoad(c, r(o), s)
    case .simdgroupStore(let src, let c, let o, let s): return .simdgroupStore(r(src), c, r(o), s)
    case .simdgroupMultiplyAccumulate(let a, let b, let acc):
      return .simdgroupMultiplyAccumulate(r(a), r(b), r(acc))
    default: return self  // ops without Lazy inputs (load, endLoop, frameIndex, etc.)
    }
  }

  /// Returns a new Op with the cell ID remapped, or nil if no remapping is needed.
  public func withRemappedCellId(_ remapping: [CellID: CellID]) -> Op? {
    guard let cellId = memoryCellId, let newCellId = remapping[cellId] else {
      return nil
    }
    switch self {
    case .load: return .load(newCellId)
    case .store(_, let val): return .store(newCellId, val)
    case .delay1(_, let a): return .delay1(newCellId, a)
    case .memoryRead(_, let offset): return .memoryRead(newCellId, offset)
    case .memoryWrite(_, let offset, let value): return .memoryWrite(newCellId, offset, value)
    case .memoryAccumulate(_, let offset, let value):
      return .memoryAccumulate(newCellId, offset, value)
    case .simdgroupLoad(_, let offset, let stride):
      return .simdgroupLoad(newCellId, offset, stride)
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

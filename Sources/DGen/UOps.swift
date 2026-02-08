public enum Lazy: Equatable {
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
  case beginHopCheck(CellID)  // if (memory[counterCell] == 0.0f) { - runs block only when counter is 0
  case endHopCheck  // } - closes the hop check conditional
  case hopCounterIncrement(CellID, Int)  // counter increment + wrap: memory[cell] = (memory[cell]+1) >= hopSize ? 0 : memory[cell]+1

  public var isDefineGlobal: Bool {
    if case .defineGlobal = self { return true }
    return false
  }

  /// Returns the memory cell ID if this operation accesses memory, nil otherwise.
  public var memoryCellId: CellID? {
    switch self {
    case .load(let cellId), .store(let cellId, _), .delay1(let cellId, _),
      .memoryRead(let cellId, _), .memoryWrite(let cellId, _, _),
      .memoryAccumulate(let cellId, _, _):
      return cellId
    default:
      return nil
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
    default: return nil
    }
  }
}

public struct UOp {
  public let op: Op
  public let value: Lazy
  public var kind: Kind? = nil  // SIMD or Scalar
  public var kindOverride: Kind? = nil
  public var tensorIndex: Lazy? = nil
  public var scalarType: CastType = .float  // int or float for variable declarations
}

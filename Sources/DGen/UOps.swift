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
  case concatShift(Lazy, Lazy, Int)  // used in vectorized history
  case delay1(CellID, Lazy)
  case loadGradMemory(CellID)
  case storeGradMemory(CellID, Lazy)
  case accumulateGrad(GradID, Lazy)
  case loadGrad(GradID)
  case loadTensorGrad(GradID, Lazy)  // Load gradient at baseGradId + index
  case accumulateTensorGrad(GradID, Lazy, Lazy)  // Accumulate to baseGradId + index
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
  case defineMemory(Int)
  case loadGlobal(VarID)
  case beginLoop(Lazy, Int)
  case beginForLoop(Lazy, Lazy)  // (loopVariable, count) - step is always 1
  case endLoop
  case beginRange(Lazy, Lazy)
  case endRange
  case beginParallelRange(Int, Int)  // count - iterations are independent, can be parallelized
  case endParallelRange
  case parallelIndex  // current index within parallel range
  case setThreadCountScale(Int)  // dispatch threads = frameCount * scale
  case setFrameIndex(Lazy)        // override frame index used for outputs/gradients
  case output(ChannelNumber, Lazy)
  case input(ChannelNumber)

  // Tensor reduction operations
  case beginReduce(Int)  // (size) - start reduction over tensor elements
  case endReduce  // end reduction
  case reduceAccumulate(Lazy)  // accumulate value into reduction result
  case frameCount
  case frameIndex
  case threadIndex
  case loadTape(Lazy, Lazy)
  case cast(Lazy, CastType)
  case declareVar(Lazy)  // Declares and initializes a variable: float t = value;
  case reshape([Int])  // View op: reshape to new shape - renders to nothing but prevents SIMD
  case transpose([Int])  // View op: transpose with permutation - renders to nothing but prevents SIMD
  case shrink([(Int, Int)?])  // View op: shrink/slice - renders to nothing but prevents SIMD
  case pad([(Int, Int)])      // View op: pad with zeros - renders to nothing but prevents SIMD
  case broadcastAccess  // Marker: broadcast indexing used - renders to nothing but prevents SIMD
  case requiresScalar   // Marker: stateful accumulation requires scalar (sample-by-sample) execution

  // Hop-based execution control (for FFT/spectral processing)
  case beginHopCheck(CellID)  // if (memory[counterCell] == 0.0f) { - runs block only when counter is 0
  case endHopCheck            // } - closes the hop check conditional

  // Local tensor operations for SIMD-across-frames optimization
  // These enable thread-local tensor storage for frame-dependent tensor chains
  case declareLocalTensor(VarID, Int)        // float localT<id>[size] - thread-local array
  case localTensorRead(VarID, Lazy)          // localT<id>[idx] - read from local tensor
  case localTensorWrite(VarID, Lazy, Lazy)   // localT<id>[idx] = val - write to local tensor
  case beginInlineLoop(Lazy, Int)            // for (int j = 0; j < count; j++) - non-parallel loop
  case endInlineLoop                         // } - closes inline loop

  // Marker for SIMD-across-frames optimization (frame-tensor chain detected)
  case frameTensorChainMarker([Int])         // Marks block as frame-tensor chain with tensor shape

  public var isDefineGlobal: Bool {
    if case .defineGlobal = self { return true }
    return false
  }
}

public struct UOp {
  public let op: Op
  public let value: Lazy
  public var kind: Kind? = nil  // SIMD or Scalar
  public var kindOverride: Kind? = nil
  public var tensorIndex: Lazy? = nil
}

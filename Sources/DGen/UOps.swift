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
  case memoryRead(CellID, Lazy)
  case memoryWrite(CellID, Lazy, Lazy)
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
  case broadcastAccess  // Marker: broadcast indexing used - renders to nothing but prevents SIMD
}

public struct UOp {
  public let op: Op
  public let value: Lazy
  public var kind: Kind? = nil  // SIMD or Scalar
  public var kindOverride: Kind? = nil
  public var tensorIndex: Lazy? = nil
}

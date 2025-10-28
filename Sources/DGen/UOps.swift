public enum Lazy {
  case constant(ConstantID, Float)
  case global(VarID)
  case variable(VarID, NodeID?)
  case empty
}

// IR (intermediate representation) is called UOp and consists of an
// operator (Op) and value (the variable it's result is bound to)
public enum Op {
  case load(CellID)
  case store(CellID, Lazy)
  case concatShift(Lazy, Lazy, Int)  // used in vectorized history
  case delay1(CellID, Lazy)
  case loadGrad(CellID)
  case storeGrad(CellID, Lazy)
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
  case endLoop
  case beginRange(Lazy, Lazy)
  case endRange
  case output(ChannelNumber, Lazy)
  case input(ChannelNumber)
  case frameCount
  case frameIndex
  case loadTape(Int)  // load forward intermediate from tape at offset + frameIndex
}

public struct UOp {
  public let op: Op
  public let value: Lazy
  public var kind: Kind? = nil
}

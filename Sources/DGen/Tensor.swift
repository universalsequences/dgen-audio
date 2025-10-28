public typealias TensorID = Int
public typealias Shape = [Int]
public struct Tensor {
  let id: TensorID
  let shape: Shape
  let cellId: CellID

  init(id: TensorID, shape: Shape, cellId: CellID) {
    self.id = id
    self.shape = shape
    self.cellId = cellId
  }
}

extension Graph {
  public func tensor(shape: Shape) -> TensorID {
    // allocate shape size
    // store in map
    // return tensorID
    var size = 1
    for dim in shape {
      size *= dim
    }

    let cellId = alloc(vectorWidth: size)
    let tensorId = nextTensorId
    nextTensorId += 1
    self.tensors[tensorId] = Tensor(id: tensorId, shape: shape, cellId: cellId)
    return tensorId
  }

  public func poke(tensorId: TensorID, index: NodeID, channel: NodeID, value: NodeID) throws
    -> NodeID
  {
    print("poke called wiith tensor:\(tensorId), index:\(index), channel:\(channel) val=\(value)")
    guard let tensor = tensors[tensorId] else {
      throw DGenError.missingTensorID
    }

    let zero = n(.constant(0.0))
    let channelSizeFloat = n(.constant(Float(tensor.shape[0])))

    // Properly wrap the index within the channel using modulo for true wrapping
    // This handles cases where index might be very negative or very positive
    let wrappedIndex = n(.mod, index, channelSizeFloat)

    // Handle negative modulo results: if wrappedIndex < 0, add channelSize
    let isNegative = n(.lt, wrappedIndex, zero)
    let positiveIndex = n(
      .gswitch, isNegative,
      n(.add, wrappedIndex, channelSizeFloat),
      wrappedIndex)

    // Calculate channel offset: floor(channel) * channelSize
    // Clamp channel to valid range [0, numChannels-1]
    let clampedChannel = n(.floor,
      n(.max, zero,
        n(.min, channel, n(.constant(Float(tensor.shape[1] - 1))))))

    let channelOffset = n(.mul, channelSizeFloat, clampedChannel)

    // Calculate final write position within the tensor buffer
    let finalWritePos = n(.floor, n(.add, channelOffset, positiveIndex))

    let bufferBase = tensor.cellId
    return n(.memoryWrite(bufferBase), finalWritePos, value)
  }

  public func peek(tensorId: TensorID, index: NodeID, channel: NodeID) throws -> NodeID {
    print("peek called wiith tensor:\(tensorId), index:\(index), channel:\(channel)")
    guard let tensor = tensors[tensorId] else {
      throw DGenError.missingTensorID
    }

    let one = n(.constant(1.0))
    let zero = n(.constant(0.0))
    let channelSizeFloat = n(.constant(Float(tensor.shape[0])))

    // Properly wrap the index within the channel using modulo for true wrapping
    // This handles cases where index might be very negative or very positive
    let wrappedIndex = n(.mod, index, channelSizeFloat)

    // Handle negative modulo results: if wrappedIndex < 0, add channelSize
    let isNegative = n(.lt, wrappedIndex, zero)
    let positiveIndex = n(
      .gswitch, isNegative,
      n(.add, wrappedIndex, channelSizeFloat),
      wrappedIndex)

    // Calculate channel offset: floor(channel) * channelSize
    // Clamp channel to valid range [0, numChannels-1]
    let clampedChannel = n(.floor,
      n(.max, zero,
        n(.min, channel, n(.constant(Float(tensor.shape[1] - 1))))))

    let channelOffset = n(.mul, channelSizeFloat, clampedChannel)

    // Calculate final read position within the channel
    let finalReadPos = n(.add, channelOffset, positiveIndex)

    let bufferBase = tensor.cellId

    // Read with linear interpolation for fractional indices
    let flooredPos = n(.floor, finalReadPos)
    let frac = n(.sub, finalReadPos, flooredPos)

    // Read two samples for interpolation
    let sample1 = n(.memoryRead(bufferBase), flooredPos)
    let nextPos = n(.add, flooredPos, one)

    // Calculate the boundary for wrapping: channelOffset + channelSize
    let nextChannelOffset = n(.add, channelOffset, channelSizeFloat)

    // Wrap nextPos if it crosses into the next channel
    // If nextPos >= nextChannelOffset, wrap back to channelOffset
    let nextPosWrapped = n(.gswitch, n(.gte, nextPos, nextChannelOffset), channelOffset, nextPos)

    let sample2 = n(.memoryRead(bufferBase), nextPosWrapped)

    // Linear interpolation: (1-frac)*sample1 + frac*sample2
    let interpolated = n(.mix, sample1, sample2, frac)
    return interpolated
  }
}

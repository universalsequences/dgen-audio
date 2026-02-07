import Foundation

extension LazyOp {
  func emitTensorOp(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy], nodeId: NodeID, ops: inout [UOp]) throws {
    switch self {
    case .conv1d(let kernelSize):
      guard node.inputs.count >= 2,
        case .tensor(let outShape) = node.shape,
        case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape, inShape.count == 1,
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let kTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] })
      else {
        throw DGenError.tensorError(
          op: "conv1d", reason: "requires 1D input/output tensors")
      }

      let inLen = inShape[0]
      let pad = kernelSize / 2

      b.parallelRange(outShape.reduce(1, *)) { flatIdx in
        let outX = b.cast(flatIdx, to: .int)
        let acc = b.float(0.0)

        b.loop(kernelSize) { kx in
          let inX = outX + b.cast(kx, to: .float) - b.constant(Float(pad))
          let inBounds = (inX >= b.constant(0)) * (inX < b.constant(Float(inLen)))

          let rawIdx = b.tensorMemoryIndex(inTensor, indices: [b.cast(inX, to: .int)])
          let safeIdx = b.gswitch(inBounds, rawIdx, b.constant(0))
          let inVal = b.gswitch(
            inBounds, b.memoryRead(inTensor.cellId, safeIdx), b.constant(0))

          let kMemIdx = b.tensorMemoryIndex(kTensor, indices: [b.cast(kx, to: .int)])
          let kVal = b.memoryRead(kTensor.cellId, kMemIdx)

          acc.accumulate(inVal * kVal)
        }
        _ = b.memoryWrite(outCell, b.cast(flatIdx, to: .int), acc.value)
      }

    case .conv2d(let kernelShape):
      guard node.inputs.count >= 2,
        case .tensor(let outShape) = node.shape,
        case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape, inShape.count == 2,
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let kTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] })
      else {
        throw DGenError.tensorError(
          op: "conv2d", reason: "requires 2D input/output tensors")
      }

      let (inH, inW) = (inShape[0], inShape[1])
      let (kH, kW) = (kernelShape[0], kernelShape[1])
      let (padH, padW) = (kH / 2, kW / 2)

      b.parallelRange(outShape.reduce(1, *)) { flatIdx in
        let outY = b.cast(flatIdx, to: .int) / b.constant(Float(inW))
        let outX = b.cast(flatIdx, to: .int) % b.constant(Float(inW))
        let acc = b.float(0.0)

        b.loop(kH) { ky in
          b.loop(kW) { kx in
            let inY = outY + b.cast(ky, to: .float) - b.constant(Float(padH))
            let inX = outX + b.cast(kx, to: .float) - b.constant(Float(padW))

            let inBounds =
              (inY >= b.constant(0)) * (inY < b.constant(Float(inH)))
              * (inX >= b.constant(0)) * (inX < b.constant(Float(inW)))

            let rawIdx = b.tensorMemoryIndex(
              inTensor, indices: [b.cast(inY, to: .int), b.cast(inX, to: .int)])
            let safeIdx = b.gswitch(inBounds, rawIdx, b.constant(0))
            let inVal = b.gswitch(
              inBounds, b.memoryRead(inTensor.cellId, safeIdx), b.constant(0))

            let kMemIdx = b.tensorMemoryIndex(
              kTensor, indices: [b.cast(ky, to: .int), b.cast(kx, to: .int)])
            let kVal = b.memoryRead(kTensor.cellId, kMemIdx)

            acc.accumulate(inVal * kVal)
          }
        }
        _ = b.memoryWrite(outCell, b.cast(flatIdx, to: .int), acc.value)
      }

    case .sum:
      if let scratch = ctx.frameTensorChainScratch[nodeId] {
        let acc = b.float(0.0)
        // Use currentFrameIndex for correct behavior in frame-aware tensor blocks
        let frameIdx = b.currentFrameIndex()
        let sizeExpr = b.constant(Float(scratch.tensorSize))
        b.loop(scratch.tensorSize) { i in
          let idx = frameIdx * sizeExpr + b.cast(i, to: .float)
          let val = b.memoryRead(scratch.cellId, b.cast(idx, to: .int))
          acc.accumulate(val)
        }
        b.use(val: acc.value)
        return
      }
      guard case .tensor(let shape) = g.nodes[node.inputs[0]]?.shape,
        let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] })
      else {
        if let s = inputs.first { b.use(val: b.value(s)) }
        return
      }
      let acc = b.float(0.0)
      b.loop(shape.reduce(1, *)) { i in
        let val = b.tensorRead(inTensor, flatIdx: i, shape: shape)
        acc.accumulate(val)
      }
      b.use(val: acc.value)

    case .sumAxis(let axis):
      guard case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape,
        case .tensor(let outShape) = node.shape,
        let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let loopIdx = b.ctx.tensorIndices[node.id],
        axis >= 0 && axis < inShape.count
      else {
        throw DGenError.tensorError(op: "sumAxis", reason: "invalid input")
      }

      // Check if input/output cells are frame-aware
      let inIsFrameAware = ctx.frameAwareTensorCells.contains(inTensor.cellId)
      let outIsFrameAware = ctx.frameAwareTensorCells.contains(outCell)
      let inTensorSize = inShape.reduce(1, *)
      let outTensorSize = outShape.reduce(1, *)

      // Debug marker for tracing sumAxis operations
      b.ops.append(
        UOp(
          op: .sumAxisMarker(nodeId, axis, inShape, outShape, inIsFrameAware, outIsFrameAware),
          value: .empty))

      let outIdx = b.value(loopIdx, scalarType: .int)
      let sumAcc = b.float(0.0)

      // Compute input strides for flat index calculation
      var inStrides = [Int](repeating: 1, count: inShape.count)
      for i in stride(from: inShape.count - 2, through: 0, by: -1) {
        inStrides[i] = inStrides[i + 1] * inShape[i + 1]
      }

      // Compute output strides for index conversion
      var outStrides = [Int](repeating: 1, count: outShape.count)
      for i in stride(from: outShape.count - 2, through: 0, by: -1) {
        outStrides[i] = outStrides[i + 1] * outShape[i + 1]
      }

      b.loop(inShape[axis]) { reduceIdx in
        let rIdx = b.cast(reduceIdx, to: .float)

        // Convert flat outIdx to multi-dimensional output indices
        var outIndices = [Expr]()
        var remaining = b.cast(outIdx, to: .int)
        for i in 0..<outShape.count {
          let stride = b.intConstant(outStrides[i])
          let idx = remaining / stride
          outIndices.append(idx)
          remaining = remaining - idx * stride
        }

        // Build input indices by inserting rIdx at the reduction axis
        var inIndices = [Expr]()
        var outDim = 0
        for i in 0..<inShape.count {
          if i == axis {
            inIndices.append(rIdx)
          } else {
            inIndices.append(outIndices[outDim])
            outDim += 1
          }
        }

        // Compute flat input index from multi-dimensional indices
        var inFlatIdx: Expr = b.intConstant(0)
        for i in 0..<inShape.count {
          inFlatIdx = inFlatIdx + inIndices[i] * b.intConstant(inStrides[i])
        }

        // Read from input tensor
        // Always use tensorRead for padded tensors - it handles padding bounds checking
        // tensorRead also handles frame-aware tensors internally
        let val: Expr
        if inTensor.padding != nil {
          // Padded tensor: must use tensorRead for padding bounds checking
          val = b.tensorRead(inTensor, indices: inIndices)
        } else if inIsFrameAware {
          // Non-padded frame-aware tensor: use direct frame-aware read
          val = b.frameAwareTensorRead(
            cellId: inTensor.cellId, tensorSize: inTensorSize, elemIdx: inFlatIdx)
        } else {
          // Non-padded, non-frame-aware: use tensorRead
          val = b.tensorRead(inTensor, indices: inIndices)
        }

        sumAcc.accumulate(val)

        // Write with frame-aware addressing if needed
        if outIsFrameAware {
          _ = b.frameAwareTensorWrite(
            cellId: outCell, tensorSize: outTensorSize, elemIdx: outIdx, value: sumAcc.value)
        } else {
          _ = b.memoryWrite(outCell, b.cast(outIdx, to: .int), sumAcc.value)
        }
      }

    case .reshape(let newShape):
      // Reshape is metadata-only - the data stays in place
      // Just register that this node produces a tensor view
      // The actual shape change is handled by the tensor metadata
      ctx.values[nodeId] = .empty
      // Emit marker UOp for debugging and to signal SIMD should be disabled
      ops.append(UOp(op: .reshape(newShape), value: .empty))

    case .asStrided(let newShape, _):
      // asStrided is metadata-only (view with custom strides for pool/im2col)
      // The actual stride change is handled by tensor metadata
      ctx.values[nodeId] = .empty
      ops.append(UOp(op: .reshape(newShape), value: .empty))  // Use reshape UOp as marker

    case .transpose(let axes):
      // Transpose is metadata-only for contiguous layouts
      // For non-trivial transposes, we may need to copy data
      // For now, just register as a view - emit will use strides
      ctx.values[nodeId] = .empty
      // Emit marker UOp for debugging and to signal SIMD should be disabled
      ops.append(UOp(op: .transpose(axes), value: .empty))

    case .shrink(let ranges):
      // Shrink is metadata-only - uses offset + strides to access slice
      ctx.values[nodeId] = .empty
      // Emit marker UOp for debugging and to signal SIMD should be disabled
      ops.append(UOp(op: .shrink(ranges), value: .empty))

    case .pad(let padding):
      // Pad is a virtual view - reads return 0 for padded regions
      ctx.values[nodeId] = .empty
      // Emit marker UOp for debugging and to signal SIMD should be disabled
      ops.append(UOp(op: .pad(padding), value: .empty))

    case .expandView(let targetShape):
      // expandView is metadata-only - broadcasts size-1 dims via stride=0
      ctx.values[nodeId] = .empty
      // Emit marker UOp for debugging and to signal SIMD should be disabled
      ops.append(UOp(op: .expandView(targetShape), value: .empty))

    case .repeatView(let repeats):
      // repeatView is metadata-only - tiles tensor via modular indexing
      ctx.values[nodeId] = .empty
      // Emit marker UOp for debugging and to signal SIMD should be disabled
      ops.append(UOp(op: .repeatView(repeats), value: .empty))

    case .peek:
      // Lazy peek: read from 2D tensor at (index, channel) with linear interpolation
      // Inputs: [tensor, index, channel]
      guard node.inputs.count == 3 else {
        throw DGenError.insufficientInputs(
          operator: "peek", expected: 3, actual: node.inputs.count)
      }

      let tensorInput = node.inputs[0]

      // Get tensor shape from the input node (auto-promote 1D to [N, 1])
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let originalShape) = inputNode.shape
      else {
        throw DGenError.tensorError(op: "peek", reason: "requires tensor input")
      }
      let shape = originalShape.count == 1 ? [originalShape[0], 1] : originalShape

      // Try to get concrete tensor, or use shape info to compute access
      let channelSize = shape[0]
      let numChannels = shape[1]

      // Read index and channel inputs
      let index = try b.readInput(node, inputs, at: 1)
      let channel = try b.readInput(node, inputs, at: 2)

      let one = b.constant(1.0)
      let zero = b.constant(0.0)
      let channelSizeFloat = b.constant(Float(channelSize))

      // Wrap index within channel using modulo
      let wrappedIndex = b.mod(index, channelSizeFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + channelSizeFloat, wrappedIndex)

      // Clamp channel to valid range [0, numChannels-1]
      let clampedChannel = b.floor(
        b.max(zero, b.min(channel, b.constant(Float(numChannels - 1)))))
      let channelOffset = channelSizeFloat * clampedChannel

      // Calculate final read position
      let finalReadPos = channelOffset + positiveIndex

      // Linear interpolation for fractional indices
      let flooredPos = b.floor(finalReadPos)
      let frac = finalReadPos - flooredPos

      // Get tensor cellId - either from concrete tensor or from input tensor
      let cellId: CellID
      if let tensorId = g.nodeToTensor[tensorInput],
        let tensor = g.tensors[tensorId]
      {
        cellId = tensor.cellId
      } else {
        throw DGenError.tensorError(
          op: "peek",
          reason: "frame-based tensor peek requires tensor context - not yet implemented")
      }

      // Prepare positions for interpolation
      let nextPos = flooredPos + one

      // Wrap nextPos if it crosses channel boundary
      let nextChannelOffset = channelOffset + channelSizeFloat
      let nextPosWrapped = b.gswitch(nextPos >= nextChannelOffset, channelOffset, nextPos)

      // Check if tensor is frame-aware (per-frame storage)
      // Frame-aware tensors store each frame's data at frameIndex * tensorSize
      let readPos1: Expr
      let readPos2: Expr
      if ctx.frameAwareTensorCells.contains(cellId) {
        // Frame-aware tensor: add frameIndex * tensorSize to read positions
        let tensorSizeFloat = b.constant(Float(channelSize * numChannels))
        let frameIdx = b.currentFrameIndex()
        let frameBase = frameIdx * tensorSizeFloat
        readPos1 = frameBase + flooredPos
        readPos2 = frameBase + nextPosWrapped
      } else {
        readPos1 = flooredPos
        readPos2 = nextPosWrapped
      }

      // Read two samples for interpolation
      let sample1 = b.memoryRead(cellId, b.cast(readPos1, to: .int))
      let sample2 = b.memoryRead(cellId, b.cast(readPos2, to: .int))

      // Linear interpolation: (1-frac)*sample1 + frac*sample2
      let interpolated = b.mix(sample1, sample2, frac)
      b.use(val: interpolated)

    case .expand(let targetShape):
      // Broadcast scalar to tensor shape (for sum backward)
      // Uses block-level parallelism via tensorIndices (like binary ops)
      // Input is scalar, output is tensor where all elements = input
      guard inputs.count == 1 else { fatalError("expand requires 1 input") }

      // Get or create output tensor
      guard let outTensor = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] }),
        let loopIdx = ctx.tensorIndices[nodeId]
      else {
        // Fallback: just pass through the scalar
        b.use(val: b.value(inputs[0]))
        return
      }

      let size = targetShape.reduce(1, *)
      let inputNodeId = node.inputs[0]

      // Check if output tensor is frame-aware (needs per-frame storage)
      let isFrameAware = ctx.frameAwareTensorCells.contains(outTensor.cellId)

      // Use block's tensor index - each thread handles ONE element
      let idx = b.value(loopIdx, scalarType: .int)

      // Get scalar value to broadcast
      let scalarVal: Expr
      if let inputTensorId = g.nodeToTensor[inputNodeId],
        let inputTensor = g.tensors[inputTensorId]
      {
        // Input is a tensor - read element 0 (assumes scalar stored in tensor)
        scalarVal = b.memoryRead(inputTensor.cellId, b.int(0))
      } else {
        // Input is a scalar computed in current scope
        scalarVal = b.value(inputs[0])
      }

      // Write to output position
      if isFrameAware {
        // Frame-aware output: write to frame-indexed position (integer arithmetic)
        let frameIdx = b.currentFrameIndex()
        let frameBase = frameIdx * b.intConstant(size)
        let writePos = frameBase + b.cast(idx, to: .int)
        _ = b.memoryWrite(outTensor.cellId, writePos, scalarVal)
      } else {
        // Non-frame-aware: direct write
        _ = b.memoryWrite(outTensor.cellId, b.cast(idx, to: .int), scalarVal)
      }

    case .expandAxis(let targetShape, let axis):
      // Broadcast along a specific axis (for sumAxis backward)
      // Uses block-level parallelism via tensorIndices (like binary ops)
      guard inputs.count == 1 else { fatalError("expandAxis requires 1 input") }

      guard let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let outTensor = g.nodeToTensor[nodeId].flatMap({ g.tensors[$0] }),
        let loopIdx = ctx.tensorIndices[nodeId]
      else {
        // Fallback for scalar case
        if let s = inputs.first { b.use(val: b.value(s)) }
        return
      }

      let outSize = targetShape.reduce(1, *)
      let normalizedAxis = axis < 0 ? targetShape.count + axis : axis

      // Check if input/output cells are frame-aware
      let inIsFrameAware = ctx.frameAwareTensorCells.contains(inTensor.cellId)
      let outIsFrameAware = ctx.frameAwareTensorCells.contains(outTensor.cellId)

      // Compute input shape (output shape with the expanded axis removed)
      var inputShape = targetShape
      inputShape.remove(at: normalizedAxis)

      // Debug marker for tracing expandAxis operations
      b.ops.append(
        UOp(
          op: .expandAxisMarker(
            nodeId, normalizedAxis, inputShape, targetShape, inIsFrameAware, outIsFrameAware),
          value: .empty))
      let inSize = inputShape.reduce(1, *)
      let inputStrides = Tensor.computeRowMajorStrides(inputShape)
      let outputStrides = Tensor.computeRowMajorStrides(targetShape)

      // Use block's tensor index (like binary ops) - each thread handles ONE element
      let outIdx = b.value(loopIdx, scalarType: .int)

      // Map output index to input index (skip the expanded axis dimension)
      var inputFlatIdx: Expr = b.int(0)
      var inDim = 0
      for dim in 0..<targetShape.count {
        if dim == normalizedAxis { continue }
        let coord = b.mod(
          b.floorDiv(outIdx, b.int(outputStrides[dim])), b.int(targetShape[dim]))
        inputFlatIdx = b.add(inputFlatIdx, b.mul(coord, b.int(inputStrides[inDim])))
        inDim += 1
      }

      // Read with frame-aware addressing if needed
      let val: Expr
      if inIsFrameAware {
        val = b.frameAwareTensorRead(
          cellId: inTensor.cellId, tensorSize: inSize, elemIdx: b.cast(inputFlatIdx, to: .float))
      } else {
        val = b.memoryRead(inTensor.cellId, inputFlatIdx)
      }

      // Write with frame-aware addressing if needed
      if outIsFrameAware {
        _ = b.frameAwareTensorWrite(
          cellId: outTensor.cellId, tensorSize: outSize, elemIdx: b.cast(outIdx, to: .float),
          value: val)
      } else {
        _ = b.memoryWrite(outTensor.cellId, b.cast(outIdx, to: .int), val)
      }

    case .gradPhasor(_):
      // Gradient for phasor: d(phase)/d(freq) = frameIndex / sampleRate
      // inputs: [gradOutput, sampleRate]
      // Use threadIndex() - the actual sample index, not decomposed frame index
      guard inputs.count == 2 else { fatalError("gradPhasor requires 2 inputs") }
      let gradOut = b.value(inputs[0])
      let sampleRate = b.value(inputs[1])
      let frameIdx = b.threadIndex()
      let gradFreq = gradOut * frameIdx / sampleRate
      b.use(val: gradFreq)

    default: break
    }
  }
}

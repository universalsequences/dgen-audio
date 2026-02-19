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

      // outIdx is invariant across reduction iterations: decode once.
      var outIndices = [Expr]()
      var remaining = b.cast(outIdx, to: .int)
      for i in 0..<outShape.count {
        let stride = b.intConstant(outStrides[i])
        let idx = remaining / stride
        outIndices.append(idx)
        remaining = remaining - idx * stride
      }

      // Build the non-reduction input indices once and precompute their flat contribution.
      var inStaticIndices = [Expr](repeating: b.intConstant(0), count: inShape.count)
      var outDim = 0
      for i in 0..<inShape.count {
        if i == axis { continue }
        inStaticIndices[i] = outIndices[outDim]
        outDim += 1
      }
      let axisStride = b.intConstant(inStrides[axis])
      var inBaseFlatIdx: Expr = b.intConstant(0)
      for i in 0..<inShape.count where i != axis {
        inBaseFlatIdx = inBaseFlatIdx + inStaticIndices[i] * b.intConstant(inStrides[i])
      }

      func readInlineReduceTensor(_ tensor: Tensor, indices: [Expr]) -> Expr {
        if let source = ctx.inlineReduceSources[tensor.cellId] {
          switch source {
          case .expandAxis(let sourceTensor, let axis):
            var sourceIndices: [Expr] = []
            sourceIndices.reserveCapacity(Swift.max(0, indices.count - 1))
            for dim in 0..<indices.count where dim != axis {
              sourceIndices.append(indices[dim])
            }
            return readInlineReduceTensor(sourceTensor, indices: sourceIndices)
          case .mul:
            break
          }
        }

        if tensor.padding != nil {
          return b.tensorRead(tensor, indices: indices)
        }
        if ctx.frameAwareTensorCells.contains(tensor.cellId) {
          let elemIdx = b.tensorMemoryIndex(tensor, indices: indices)
          return b.frameAwareTensorRead(
            cellId: tensor.cellId,
            tensorSize: tensor.shape.reduce(1, *),
            elemIdx: elemIdx
          )
        }
        if tensor.transforms.isEmpty {
          let elemIdx = b.tensorMemoryIndex(tensor, indices: indices)
          return b.memoryRead(tensor.cellId, elemIdx)
        }
        return b.tensorRead(tensor, indices: indices)
      }

      b.loop(inShape[axis]) { reduceIdx in
        let rIdx = reduceIdx
        let inFlatIdx = inBaseFlatIdx + rIdx * axisStride

        // Read from input tensor — or compute inline if fused with skipped producers.
        let val: Expr
        // Fusion fast-path: if this input cell was produced by a skipped mul region,
        // compute the product from its original source tensors inline while reducing.
        if let source = ctx.inlineReduceSources[inTensor.cellId],
          case .mul(let aTensor, let bTensor) = source
        {
          var inIndices = inStaticIndices
          inIndices[axis] = rIdx
          // Fused path: compute A * B inline instead of reading from memory.
          // The mul's input tensors have view transforms (reshape, expand) that
          // handle broadcasting — tensorRead walks the transform chain to map
          // [M,N,K] indices back to the correct base memory locations.
          let broadcastedA = b.broadcastIndices(
            outputIndices: inIndices, outputShape: inShape, inputTensor: aTensor)
          let broadcastedB = b.broadcastIndices(
            outputIndices: inIndices, outputShape: inShape, inputTensor: bTensor)
          let aVal = readInlineReduceTensor(aTensor, indices: broadcastedA)
          let bVal = readInlineReduceTensor(bTensor, indices: broadcastedB)
          val = aVal * bVal
        } else if let source = ctx.inlineReduceSources[inTensor.cellId],
          case .expandAxis(let sourceTensor, let expandedAxis) = source
        {
          var inIndices = inStaticIndices
          inIndices[axis] = rIdx

          var sourceIndices: [Expr] = []
          sourceIndices.reserveCapacity(Swift.max(0, inIndices.count - 1))
          for dim in 0..<inIndices.count where dim != expandedAxis {
            sourceIndices.append(inIndices[dim])
          }
          val = readInlineReduceTensor(sourceTensor, indices: sourceIndices)
        } else if inTensor.padding != nil {
          var inIndices = inStaticIndices
          inIndices[axis] = rIdx
          // Padded tensor: must use tensorRead for padding bounds checking
          val = b.tensorRead(inTensor, indices: inIndices)
        } else if inIsFrameAware {
          // Non-padded frame-aware tensor: use direct frame-aware read
          val = b.frameAwareTensorRead(
            cellId: inTensor.cellId, tensorSize: inTensorSize, elemIdx: inFlatIdx)
        } else if inTensor.transforms.isEmpty {
          // Contiguous non-frame-aware tensor: direct linear read.
          val = b.memoryRead(inTensor.cellId, inFlatIdx)
        } else {
          var inIndices = inStaticIndices
          inIndices[axis] = rIdx
          // Non-padded, non-frame-aware: use tensorRead
          val = b.tensorRead(inTensor, indices: inIndices)
        }

        sumAcc.accumulate(val)
      }

      // Write final sum once after the reduction loop.
      if outIsFrameAware {
        _ = b.frameAwareTensorWrite(
          cellId: outCell, tensorSize: outTensorSize, elemIdx: outIdx, value: sumAcc.value)
      } else {
        _ = b.memoryWrite(outCell, b.cast(outIdx, to: .int), sumAcc.value)
      }

    case .sumMulAxis0:
      guard node.inputs.count == 2,
        case .tensor(let leftShape) = g.nodes[node.inputs[0]]?.shape,
        case .tensor(let rightShape) = g.nodes[node.inputs[1]]?.shape,
        leftShape.count == 2,
        rightShape == leftShape,
        case .tensor(let outShape) = node.shape,
        outShape == [leftShape[1]],
        let leftTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let rightTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] }),
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let loopIdx = b.ctx.tensorIndices[node.id]
      else {
        throw DGenError.tensorError(op: "sumMulAxis0", reason: "invalid input")
      }

      let rows = leftShape[0]
      let cols = leftShape[1]
      let colIdx = b.value(loopIdx, scalarType: .int)
      let sumAcc = b.float(0.0)

      let leftIsFrameAware = ctx.frameAwareTensorCells.contains(leftTensor.cellId)
      let rightIsFrameAware = ctx.frameAwareTensorCells.contains(rightTensor.cellId)
      let outIsFrameAware = ctx.frameAwareTensorCells.contains(outCell)
      let leftTensorSize = leftShape.reduce(1, *)
      let rightTensorSize = rightShape.reduce(1, *)
      let outTensorSize = outShape.reduce(1, *)

      func readFusedInput(_ tensor: Tensor, row: Expr, col: Expr, frameAware: Bool, tensorSize: Int)
        -> Expr
      {
        if tensor.padding != nil || !tensor.transforms.isEmpty {
          return b.tensorRead(tensor, indices: [row, col])
        }

        let flatIdx = row * b.intConstant(cols) + col
        if frameAware {
          return b.frameAwareTensorRead(
            cellId: tensor.cellId,
            tensorSize: tensorSize,
            elemIdx: flatIdx
          )
        }
        return b.memoryRead(tensor.cellId, flatIdx)
      }

      b.loop(rows) { row in
        let leftVal = readFusedInput(
          leftTensor, row: row, col: colIdx, frameAware: leftIsFrameAware,
          tensorSize: leftTensorSize)
        let rightVal = readFusedInput(
          rightTensor, row: row, col: colIdx, frameAware: rightIsFrameAware,
          tensorSize: rightTensorSize)
        sumAcc.accumulate(leftVal * rightVal)
      }

      if outIsFrameAware {
        _ = b.frameAwareTensorWrite(
          cellId: outCell, tensorSize: outTensorSize, elemIdx: colIdx, value: sumAcc.value)
      } else {
        _ = b.memoryWrite(outCell, colIdx, sumAcc.value)
      }

    case .gemmSmall(let M, let N, let K, let transA, let transB):
      guard node.inputs.count == 2,
        let leftTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let rightTensor = g.nodeToTensor[node.inputs[1]].flatMap({ g.tensors[$0] }),
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let loopIdx = b.ctx.tensorIndices[node.id]
      else {
        throw DGenError.tensorError(op: "gemmSmall", reason: "invalid input")
      }

      let leftIsFrameAware = ctx.frameAwareTensorCells.contains(leftTensor.cellId)
      let rightIsFrameAware = ctx.frameAwareTensorCells.contains(rightTensor.cellId)
      let outIsFrameAware = ctx.frameAwareTensorCells.contains(outCell)
      // Physical sizes: A is [M,K] or [K,M], B is [K,N] or [N,K]
      let leftTensorSize = M * K
      let rightTensorSize = K * N
      let outTensorSize = M * N

      // One thread per output element: elemIdx = m * N + n
      let elemIdx = b.value(loopIdx, scalarType: .int)
      let m = elemIdx / b.intConstant(N)
      let n = elemIdx % b.intConstant(N)
      let sumAcc = b.float(0.0)

      func readGemmInput(_ tensor: Tensor, flatIdx: Expr, frameAware: Bool, tensorSize: Int) -> Expr
      {
        if tensor.padding != nil || !tensor.transforms.isEmpty {
          // Fallback: use generic tensor read for complex views
          return b.memoryRead(tensor.cellId, flatIdx)
        }
        if frameAware {
          return b.frameAwareTensorRead(
            cellId: tensor.cellId, tensorSize: tensorSize, elemIdx: flatIdx)
        }
        return b.memoryRead(tensor.cellId, flatIdx)
      }

      b.loop(K) { k in
        // A[m,k]: stored as [M,K] (row-major) or transposed [K,M]
        let aFlat: Expr = transA
          ? k * b.intConstant(M) + m    // A stored as [K,M]
          : m * b.intConstant(K) + k    // A stored as [M,K]
        // B[k,n]: stored as [K,N] (row-major) or transposed [N,K]
        let bFlat: Expr = transB
          ? n * b.intConstant(K) + k    // B stored as [N,K]
          : k * b.intConstant(N) + n    // B stored as [K,N]

        let leftVal = readGemmInput(
          leftTensor, flatIdx: aFlat, frameAware: leftIsFrameAware, tensorSize: leftTensorSize)
        let rightVal = readGemmInput(
          rightTensor, flatIdx: bFlat, frameAware: rightIsFrameAware, tensorSize: rightTensorSize)
        sumAcc.accumulate(leftVal * rightVal)
      }

      let outFlat = m * b.intConstant(N) + n
      if outIsFrameAware {
        _ = b.frameAwareTensorWrite(
          cellId: outCell, tensorSize: outTensorSize, elemIdx: outFlat, value: sumAcc.value)
      } else {
        _ = b.memoryWrite(outCell, outFlat, sumAcc.value)
      }

    case .maxAxis(let axis):
      guard case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape,
        case .tensor(let outShape) = node.shape,
        let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let loopIdx = b.ctx.tensorIndices[node.id],
        axis >= 0 && axis < inShape.count
      else {
        throw DGenError.tensorError(op: "maxAxis", reason: "invalid input")
      }

      let inIsFrameAwareMax = ctx.frameAwareTensorCells.contains(inTensor.cellId)
      let outIsFrameAwareMax = ctx.frameAwareTensorCells.contains(outCell)
      let inTensorSizeMax = inShape.reduce(1, *)
      let outTensorSizeMax = outShape.reduce(1, *)

      b.ops.append(
        UOp(
          op: .maxAxisMarker(nodeId, axis, inShape, outShape, inIsFrameAwareMax, outIsFrameAwareMax),
          value: .empty))

      let outIdxMax = b.value(loopIdx, scalarType: .int)
      let maxAcc = b.float(-Float.greatestFiniteMagnitude)

      var inStridesMax = [Int](repeating: 1, count: inShape.count)
      for i in stride(from: inShape.count - 2, through: 0, by: -1) {
        inStridesMax[i] = inStridesMax[i + 1] * inShape[i + 1]
      }
      var outStridesMax = [Int](repeating: 1, count: outShape.count)
      for i in stride(from: outShape.count - 2, through: 0, by: -1) {
        outStridesMax[i] = outStridesMax[i + 1] * outShape[i + 1]
      }

      b.loop(inShape[axis]) { reduceIdx in
        let rIdx = reduceIdx

        var outIndices = [Expr]()
        var remaining = b.cast(outIdxMax, to: .int)
        for i in 0..<outShape.count {
          let stride = b.intConstant(outStridesMax[i])
          let idx = remaining / stride
          outIndices.append(idx)
          remaining = remaining - idx * stride
        }

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

        var inFlatIdx: Expr = b.intConstant(0)
        for i in 0..<inShape.count {
          inFlatIdx = inFlatIdx + inIndices[i] * b.intConstant(inStridesMax[i])
        }

        let val: Expr
        if inTensor.padding != nil {
          val = b.tensorRead(inTensor, indices: inIndices)
        } else if inIsFrameAwareMax {
          val = b.frameAwareTensorRead(
            cellId: inTensor.cellId, tensorSize: inTensorSizeMax, elemIdx: inFlatIdx)
        } else {
          val = b.tensorRead(inTensor, indices: inIndices)
        }

        maxAcc.mutate(to: b.max(maxAcc.value, val))
      }

      // Write final max value after the loop
      if outIsFrameAwareMax {
        _ = b.frameAwareTensorWrite(
          cellId: outCell, tensorSize: outTensorSizeMax, elemIdx: outIdxMax, value: maxAcc.value)
      } else {
        _ = b.memoryWrite(outCell, b.cast(outIdxMax, to: .int), maxAcc.value)
      }

    case .meanAxis(let axis):
      guard case .tensor(let inShape) = g.nodes[node.inputs[0]]?.shape,
        case .tensor(let outShape) = node.shape,
        let inTensor = g.nodeToTensor[node.inputs[0]].flatMap({ g.tensors[$0] }),
        let outCell = g.nodeToTensor[node.id].flatMap({ g.tensors[$0] })?.cellId,
        let loopIdx = b.ctx.tensorIndices[node.id],
        axis >= 0 && axis < inShape.count
      else {
        throw DGenError.tensorError(op: "meanAxis", reason: "invalid input")
      }

      let inIsFrameAwareMean = ctx.frameAwareTensorCells.contains(inTensor.cellId)
      let outIsFrameAwareMean = ctx.frameAwareTensorCells.contains(outCell)
      let inTensorSizeMean = inShape.reduce(1, *)
      let outTensorSizeMean = outShape.reduce(1, *)

      b.ops.append(
        UOp(
          op: .meanAxisMarker(nodeId, axis, inShape, outShape, inIsFrameAwareMean, outIsFrameAwareMean),
          value: .empty))

      let outIdxMean = b.value(loopIdx, scalarType: .int)
      let meanAcc = b.float(0.0)

      var inStridesMean = [Int](repeating: 1, count: inShape.count)
      for i in stride(from: inShape.count - 2, through: 0, by: -1) {
        inStridesMean[i] = inStridesMean[i + 1] * inShape[i + 1]
      }
      var outStridesMean = [Int](repeating: 1, count: outShape.count)
      for i in stride(from: outShape.count - 2, through: 0, by: -1) {
        outStridesMean[i] = outStridesMean[i + 1] * outShape[i + 1]
      }

      b.loop(inShape[axis]) { reduceIdx in
        let rIdx = reduceIdx

        var outIndices = [Expr]()
        var remaining = b.cast(outIdxMean, to: .int)
        for i in 0..<outShape.count {
          let stride = b.intConstant(outStridesMean[i])
          let idx = remaining / stride
          outIndices.append(idx)
          remaining = remaining - idx * stride
        }

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

        var inFlatIdx: Expr = b.intConstant(0)
        for i in 0..<inShape.count {
          inFlatIdx = inFlatIdx + inIndices[i] * b.intConstant(inStridesMean[i])
        }

        let val: Expr
        if inTensor.padding != nil {
          val = b.tensorRead(inTensor, indices: inIndices)
        } else if inIsFrameAwareMean {
          val = b.frameAwareTensorRead(
            cellId: inTensor.cellId, tensorSize: inTensorSizeMean, elemIdx: inFlatIdx)
        } else {
          val = b.tensorRead(inTensor, indices: inIndices)
        }

        meanAcc.accumulate(val)
      }

      // Divide by axis size and write
      meanAcc.mutate(to: meanAcc.value / b.constant(Float(inShape[axis])))
      if outIsFrameAwareMean {
        _ = b.frameAwareTensorWrite(
          cellId: outCell, tensorSize: outTensorSizeMean, elemIdx: outIdxMean, value: meanAcc.value)
      } else {
        _ = b.memoryWrite(outCell, b.cast(outIdxMean, to: .int), meanAcc.value)
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

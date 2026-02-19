import Foundation

extension LazyOp {
  func emitRowSelection(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy], nodeId: NodeID) throws {
    switch self {
    case .selectRow:
      // Extract a single row from a 2D tensor using dynamic index.
      // Uses block-level parallelism via tensorIndices (like binary ops)
      // Inputs: [tensor2D, rowIndex], Output: 1D tensor [numCols]
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "selectRow", expected: 2, actual: node.inputs.count)
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape,
        shape.count == 2
      else {
        throw DGenError.tensorError(op: "selectRow", reason: "requires 2D tensor input")
      }

      let numRows = shape[0]

      guard let inTensorId = g.nodeToTensor[tensorInput],
        let inTensor = g.tensors[inTensorId],
        let outTensorId = g.nodeToTensor[node.id],
        let outTensor = g.tensors[outTensorId],
        let loopIdx = ctx.tensorIndices[nodeId]
      else {
        throw DGenError.tensorError(op: "selectRow", reason: "missing tensor")
      }

      let rowIndex = try b.readInput(node, inputs, at: 1)
      let numRowsFloat = b.constant(Float(numRows))
      let zero = b.constant(0.0)

      // Wrap rowIndex with modulo and ensure positive
      let wrappedIndex = b.mod(rowIndex, numRowsFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)
      let floorIndex = b.floor(positiveIndex)

      // Use block's tensor index - each thread handles ONE column
      let colIdx = b.value(loopIdx, scalarType: .int)
      let colIdxFloat = b.cast(colIdx, to: .float)
      // Row-major read: tensorRead handles both transformed and base tensors
      let value = b.tensorRead(inTensor, indices: [floorIndex, colIdxFloat])
      _ = b.memoryWrite(outTensor.cellId, b.cast(colIdx, to: .int), value)

      ctx.values[nodeId] = .empty

    case .peekRowInline(let scratchCell, let numRows, let numCols):
      // Interpolated row extraction with frame-indexed storage for SIMD safety.
      // Uses block-level parallelism via tensorIndices (like binary ops)
      // Inputs: [tensor2D, rowIndex], Output: 1D tensor [numCols] at scratchCell[frame * numCols + col]
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "peekRowInline", expected: 2, actual: node.inputs.count)
      }

      let tensorInput = node.inputs[0]
      guard let inTensorId = g.nodeToTensor[tensorInput],
        let inTensor = g.tensors[inTensorId],
        let outTensorId = g.nodeToTensor[node.id],
        let outTensor = g.tensors[outTensorId],
        let loopIdx = ctx.tensorIndices[nodeId]
      else {
        throw DGenError.tensorError(op: "peekRowInline", reason: "missing tensor")
      }

      let rowIndex = try b.readInput(node, inputs, at: 1)
      let numRowsFloat = b.constant(Float(numRows))
      let numColsFloat = b.constant(Float(numCols))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      // Wrap rowIndex with modulo and ensure positive
      let wrappedIndex = b.mod(rowIndex, numRowsFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)

      // Compute floor/ceil indices for interpolation
      let floorIndex = b.floor(positiveIndex)
      let ceilIndex = floorIndex + one
      let ceilWrapped = b.gswitch(ceilIndex >= numRowsFloat, zero, ceilIndex)
      let frac = positiveIndex - floorIndex
      let oneMinusFrac = one - frac

      // Use currentFrameIndex for correct frame index in frame-aware tensor blocks
      let frameIdx = b.currentFrameIndex()
      let frameBase = frameIdx * numColsFloat

      // Use block's tensor index - each thread handles ONE column
      let colIdx = b.value(loopIdx, scalarType: .int)
      let colIdxFloat = b.cast(colIdx, to: .float)
      // Row-major read: tensorRead handles both transformed and base tensors
      let floorValue = b.tensorRead(inTensor, indices: [floorIndex, colIdxFloat])
      let ceilValue = b.tensorRead(inTensor, indices: [ceilWrapped, colIdxFloat])

      // Interpolate: (1 - frac) * floor + frac * ceil
      let interpolated = oneMinusFrac * floorValue + frac * ceilValue
      let writePos = frameBase + colIdxFloat
      _ = b.memoryWrite(scratchCell, b.cast(writePos, to: .int), interpolated)

      // Write to output tensor with appropriate addressing
      if ctx.frameAwareTensorCells.contains(outTensor.cellId) {
        _ = b.memoryWrite(outTensor.cellId, b.cast(writePos, to: .int), interpolated)
      } else {
        _ = b.memoryWrite(outTensor.cellId, b.cast(colIdx, to: .int), interpolated)
      }

      ctx.values[nodeId] = .empty

    case .selectRowGradWrite(let gradWriteCell, let rowIdxCell, let numRows, let numCols):
      // Write gradient to frame-indexed storage (deterministic, no atomics)
      // Inputs: [gradOutput (1D tensor), rowIndex]
      // Writes to gradWriteCell[frame * numCols + col] and rowIdxCell[frame]
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "selectRowGradWrite", expected: 2, actual: node.inputs.count)
      }

      let gradTensorInput = node.inputs[0]

      // Get the gradient tensor's cell
      guard let gradTensorId = g.nodeToTensor[gradTensorInput],
        let gradTensor = g.tensors[gradTensorId]
      else {
        throw DGenError.tensorError(
          op: "selectRowGradWrite", reason: "missing gradient tensor")
      }

      // Read rowIndex input and floor it
      let rowIndex = try b.readInput(node, inputs, at: 1)
      let numRowsFloat = b.constant(Float(numRows))
      let numColsFloat = b.constant(Float(numCols))
      let zero = b.constant(0.0)

      // Wrap rowIndex using modulo for wrapping behavior, then floor
      let wrappedIndex = b.mod(rowIndex, numRowsFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)
      let floorIndex = b.floor(positiveIndex)

      // Get frame index for frame-indexed storage
      // Use currentFrameIndex for correct behavior in frame-aware tensor blocks
      let frameIdx = b.currentFrameIndex()

      // Write the floored row index for this frame
      _ = b.memoryWrite(rowIdxCell, b.cast(frameIdx, to: .int), floorIndex)

      // Write each gradient element to frame-indexed position
      // Layout: gradWriteCell[frame * numCols + col]
      let frameBase = frameIdx * numColsFloat
      b.parallelRange(numCols) { colIdx in
        let colIdxFloat = b.cast(colIdx, to: .float)
        // Read gradient element from gradOutput tensor
        let gradValue = b.memoryRead(gradTensor.cellId, b.cast(colIdx, to: .int))
        // Write to frame-indexed position
        let writePos = frameBase + colIdxFloat
        _ = b.memoryWrite(gradWriteCell, b.cast(writePos, to: .int), gradValue)
      }

      b.use(val: zero)  // Side-effect only

    case .selectRowGradReduce(
      let gradWriteCell, let rowIdxCell, let gradCell,
      let numRows, let numCols, let maxFrameCount):
      // Sum gradient contributions from all frames for each tensor position.
      // Input: [gradWritePass] (dependency ordering only)
      guard node.inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "selectRowGradReduce", expected: 1, actual: node.inputs.count)
      }

      _ = b.value(inputs[0])  // Force dependency on write pass

      let numColsFloat = b.constant(Float(numCols))
      let numColsInt = b.intConstant(numCols)
      let zero = b.constant(0.0)
      let totalElems = numRows * numCols
      b.parallelRange(totalElems) { flatIdx in
        let rowIdx = flatIdx / numColsInt
        let colIdx = flatIdx - rowIdx * numColsInt
        let rowFloat = b.cast(rowIdx, to: .float)
        let colFloat = b.cast(colIdx, to: .float)
        let gradSum = b.float(0.0)

        b.loop(maxFrameCount) { frameIdx in
          let frameFloat = b.cast(frameIdx, to: .float)
          let selectedRow = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
          let isMatch = b.abs(selectedRow - rowFloat) < b.constant(0.5)
          let readPos = frameFloat * numColsFloat + colFloat
          let gradValue = b.memoryRead(gradWriteCell, b.cast(readPos, to: .int))
          let contribution = b.gswitch(isMatch, gradValue, zero)
          gradSum.accumulate(contribution)
        }

        // Row-major layout to match tensorRead: offset = row * numCols + col
        let destPos = rowFloat * numColsFloat + colFloat
        _ = b.memoryAccumulate(gradCell, b.cast(destPos, to: .int), gradSum.value)
      }

      b.use(val: zero)

    case .peekRowGradWrite(
      let floorGradCell, let ceilGradCell, let rowIdxCell, let fracCell,
      let numRows, let numCols, let maxFrameCount):
      // Write gradients for both floor and ceil rows to frame-indexed storage
      // Inputs: [gradOutput (scalar or 1D tensor), rowIndex]
      // Note: gradOutput can be scalar (from sum reduction) - same value for all elements
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "peekRowGradWrite", expected: 2, actual: node.inputs.count)
      }

      let gradTensorInput = node.inputs[0]

      // Check if gradient input is a tensor or scalar
      let gradCellId: CellID?
      if let gradTensorId = g.nodeToTensor[gradTensorInput],
        let gradTensor = g.tensors[gradTensorId]
      {
        gradCellId = gradTensor.cellId
      } else {
        gradCellId = nil  // Scalar gradient - will use b.value()
      }

      // Read gradient as scalar (works for both scalar and will be used as broadcast value)
      let scalarGrad = b.value(inputs[0])

      // Read rowIndex input
      let rowIndex = try b.readInput(node, inputs, at: 1)
      let numRowsFloat = b.constant(Float(numRows))
      let numColsFloat = b.constant(Float(numCols))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      // Wrap rowIndex using modulo
      let wrappedIndex = b.mod(rowIndex, numRowsFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)

      // Compute floor and ceil indices
      let floorIndex = b.floor(positiveIndex)
      let ceilIndex = floorIndex + one
      let ceilWrapped = b.gswitch(ceilIndex >= numRowsFloat, zero, ceilIndex)

      // Compute frac
      let frac = positiveIndex - floorIndex

      // Get frame index - use frameIndex() which respects setFrameIndex
      let frameIdx = b.frameIndex()

      // Write row indices and frac for this frame
      _ = b.memoryWrite(rowIdxCell, b.cast(frameIdx, to: .int), floorIndex)
      _ = b.memoryWrite(fracCell, b.cast(frameIdx, to: .int), frac)
      // Write ceil index to a separate slot (frame + maxFrameCount)
      let ceilSlot = frameIdx + b.constant(Float(maxFrameCount))
      _ = b.memoryWrite(rowIdxCell, b.cast(ceilSlot, to: .int), ceilWrapped)

      // Write weighted gradients for floor and ceil
      // floor gets grad * (1 - frac), ceil gets grad * frac
      let oneMinusFrac = one - frac
      let frameBase = frameIdx * numColsFloat

      b.parallelRange(numCols) { colIdx in
        let colIdxFloat = b.cast(colIdx, to: .float)
        // Get gradient value - from tensor cell if available, otherwise use scalar
        let gradValue: Expr
        if let cellId = gradCellId {
          // Frame-aware tensor: read from frameIdx * numCols + colIdx
          let readPos = frameBase + colIdxFloat
          gradValue = b.memoryRead(cellId, b.cast(readPos, to: .int))
        } else {
          gradValue = scalarGrad  // Broadcast scalar to all elements
        }
        let writePos = frameBase + colIdxFloat
        // Floor gradient: grad * (1 - frac)
        let floorGrad = gradValue * oneMinusFrac
        _ = b.memoryWrite(floorGradCell, b.cast(writePos, to: .int), floorGrad)
        // Ceil gradient: grad * frac
        let ceilGrad = gradValue * frac
        _ = b.memoryWrite(ceilGradCell, b.cast(writePos, to: .int), ceilGrad)
      }

      b.use(val: zero)

    case .peekRowGradReduce(
      let floorGradCell, let ceilGradCell, let rowIdxCell, _,
      let gradCell, let numRows, let numCols, let maxFrameCount):
      // Input: [gradWritePass] (dependency ordering only)
      guard node.inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "peekRowGradReduce", expected: 1, actual: node.inputs.count)
      }

      _ = b.value(inputs[0])  // Force dependency on write pass

      let numColsFloat = b.constant(Float(numCols))
      let numColsInt = b.intConstant(numCols)
      let zero = b.constant(0.0)
      let maxFrameCountFloat = b.constant(Float(maxFrameCount))
      let frameCount = b.frameCount()
      if DGenGradientConfig.useFastPeekRowGradReduce {
        // Fast scatter-add path:
        // parallelize over (frame, col), then atomically add to floor/ceil row bins.
        let maxRow = b.constant(Float(Swift.max(0, numRows - 1)))
        let totalElems = maxFrameCount * numCols
        b.parallelRange(totalElems) { flatIdx in
          let frameIdx = flatIdx / numColsInt
          let colIdx = flatIdx - frameIdx * numColsInt
          let frameFloat = b.cast(frameIdx, to: .float)
          let colFloat = b.cast(colIdx, to: .float)
          let inBounds = frameFloat < frameCount
          let readPos = frameFloat * numColsFloat + colFloat

          // Floor contribution.
          let floorRowRaw = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
          // Match legacy "< 0.5" row compare semantics by rounding to nearest row.
          let floorRowRounded = b.floor(floorRowRaw + b.constant(0.5))
          let floorRow = b.min(b.max(floorRowRounded, zero), maxRow)
          let floorGrad = b.memoryRead(floorGradCell, b.cast(readPos, to: .int))
          let floorContrib = b.gswitch(inBounds > zero, floorGrad, zero)
          let floorDest = floorRow * numColsFloat + colFloat
          _ = b.memoryAccumulate(gradCell, b.cast(floorDest, to: .int), floorContrib)

          // Ceil contribution (index stored at frame + maxFrameCount).
          let ceilSlot = frameFloat + maxFrameCountFloat
          let ceilRowRaw = b.memoryRead(rowIdxCell, b.cast(ceilSlot, to: .int))
          let ceilRowRounded = b.floor(ceilRowRaw + b.constant(0.5))
          let ceilRow = b.min(b.max(ceilRowRounded, zero), maxRow)
          let ceilGrad = b.memoryRead(ceilGradCell, b.cast(readPos, to: .int))
          let ceilContrib = b.gswitch(inBounds > zero, ceilGrad, zero)
          let ceilDest = ceilRow * numColsFloat + colFloat
          _ = b.memoryAccumulate(gradCell, b.cast(ceilDest, to: .int), ceilContrib)
        }
      } else {
        // Legacy row-bin scan path:
        // for each (row,col), scan all frames and sum matching floor/ceil contributions.
        let totalElems = numRows * numCols
        b.parallelRange(totalElems) { flatIdx in
          let rowIdx = flatIdx / numColsInt
          let colIdx = flatIdx - rowIdx * numColsInt
          let rowFloat = b.cast(rowIdx, to: .float)
          let colFloat = b.cast(colIdx, to: .float)
          let gradSum = b.float(0.0)

          b.loop(maxFrameCount) { frameIdx in
            let frameFloat = b.cast(frameIdx, to: .float)
            let inBounds = frameFloat < frameCount
            let readPos = frameFloat * numColsFloat + colFloat

            // Floor row contribution
            let floorRow = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
            let isFloorMatch = b.abs(floorRow - rowFloat) < b.constant(0.5)
            let floorGrad = b.memoryRead(floorGradCell, b.cast(readPos, to: .int))
            let floorValid = inBounds * isFloorMatch
            let floorContrib = b.gswitch(floorValid > zero, floorGrad, zero)
            gradSum.accumulate(floorContrib)

            // Ceil row contribution (index stored at frame + maxFrameCount)
            let ceilSlot = frameFloat + maxFrameCountFloat
            let ceilRow = b.memoryRead(rowIdxCell, b.cast(ceilSlot, to: .int))
            let isCeilMatch = b.abs(ceilRow - rowFloat) < b.constant(0.5)
            let ceilGrad = b.memoryRead(ceilGradCell, b.cast(readPos, to: .int))
            let ceilValid = inBounds * isCeilMatch
            let ceilContrib = b.gswitch(ceilValid > zero, ceilGrad, zero)
            gradSum.accumulate(ceilContrib)
          }

          // Row-major layout to match tensorRead: offset = row * numCols + col
          let destPos = rowFloat * numColsFloat + colFloat
          _ = b.memoryAccumulate(gradCell, b.cast(destPos, to: .int), gradSum.value)
        }
      }

      b.use(val: zero)

    case .sampleInline(let scratchCell, let numRows, let remainingShape):
      // Interpolated sampling along axis 0 for any-rank tensor.
      // Inputs: [tensorND, index], Output: tensor with remainingShape
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "sampleInline", expected: 2, actual: node.inputs.count)
      }

      let tensorInput = node.inputs[0]
      guard let inTensorId = g.nodeToTensor[tensorInput],
        let inTensor = g.tensors[inTensorId],
        let outTensorId = g.nodeToTensor[node.id],
        let outTensor = g.tensors[outTensorId],
        let loopIdx = ctx.tensorIndices[nodeId]
      else {
        throw DGenError.tensorError(op: "sampleInline", reason: "missing tensor")
      }

      let remainingSize = remainingShape.reduce(1, *)
      let rowIndex = try b.readInput(node, inputs, at: 1)
      let numRowsFloat = b.constant(Float(numRows))
      let remainingSizeFloat = b.constant(Float(remainingSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      // Wrap rowIndex with modulo and ensure positive
      let wrappedIndex = b.mod(rowIndex, numRowsFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)

      // Compute floor/ceil indices for interpolation
      let floorIndex = b.floor(positiveIndex)
      let ceilIndex = floorIndex + one
      let ceilWrapped = b.gswitch(ceilIndex >= numRowsFloat, zero, ceilIndex)
      let frac = positiveIndex - floorIndex
      let oneMinusFrac = one - frac

      // Use currentFrameIndex for correct frame index in frame-aware tensor blocks
      let frameIdx = b.currentFrameIndex()
      let frameBase = frameIdx * remainingSizeFloat

      // Use block's tensor index - each thread handles ONE element of remainingShape
      let elemIdx = b.value(loopIdx, scalarType: .int)
      // Decompose flat elem index to multi-dim indices for tensorRead
      let remainingIndices = b.flatToMultiIndex(elemIdx, remainingShape)
      let remainingIndicesFloat = remainingIndices.map { b.cast($0, to: .float) }

      // tensorRead with [rowIndex] + remaining dims
      let floorValue = b.tensorRead(inTensor, indices: [floorIndex] + remainingIndicesFloat)
      let ceilValue = b.tensorRead(inTensor, indices: [ceilWrapped] + remainingIndicesFloat)

      // Interpolate: (1 - frac) * floor + frac * ceil
      let interpolated = oneMinusFrac * floorValue + frac * ceilValue
      let elemIdxFloat = b.cast(elemIdx, to: .float)
      let writePos = frameBase + elemIdxFloat
      _ = b.memoryWrite(scratchCell, b.cast(writePos, to: .int), interpolated)

      // Write to output tensor with appropriate addressing
      if ctx.frameAwareTensorCells.contains(outTensor.cellId) {
        _ = b.memoryWrite(outTensor.cellId, b.cast(writePos, to: .int), interpolated)
      } else {
        _ = b.memoryWrite(outTensor.cellId, b.cast(elemIdx, to: .int), interpolated)
      }

      ctx.values[nodeId] = .empty

    case .sampleGradWrite(
      let floorGradCell, let ceilGradCell, let rowIdxCell, let fracCell,
      let numRows, let remainingShape, let maxFrameCount):
      // Write gradients for both floor and ceil rows to frame-indexed storage
      // Inputs: [gradOutput (tensor or scalar), index]
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "sampleGradWrite", expected: 2, actual: node.inputs.count)
      }

      let remainingSize = remainingShape.reduce(1, *)
      let gradTensorInput = node.inputs[0]

      // Check if gradient input is a tensor or scalar
      let gradCellId: CellID?
      if let gradTensorId = g.nodeToTensor[gradTensorInput],
        let gradTensor = g.tensors[gradTensorId]
      {
        gradCellId = gradTensor.cellId
      } else {
        gradCellId = nil
      }

      let scalarGrad = b.value(inputs[0])

      // Read index and compute interpolation params
      let rowIndex = try b.readInput(node, inputs, at: 1)
      let numRowsFloat = b.constant(Float(numRows))
      let remainingSizeFloat = b.constant(Float(remainingSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      let wrappedIndex = b.mod(rowIndex, numRowsFloat)
      let isNegative = wrappedIndex < zero
      let positiveIndex = b.gswitch(isNegative, wrappedIndex + numRowsFloat, wrappedIndex)

      let floorIndex = b.floor(positiveIndex)
      let ceilIndex = floorIndex + one
      let ceilWrapped = b.gswitch(ceilIndex >= numRowsFloat, zero, ceilIndex)
      let frac = positiveIndex - floorIndex

      let frameIdx = b.frameIndex()

      // Write row indices and frac for this frame
      _ = b.memoryWrite(rowIdxCell, b.cast(frameIdx, to: .int), floorIndex)
      _ = b.memoryWrite(fracCell, b.cast(frameIdx, to: .int), frac)
      let ceilSlot = frameIdx + b.constant(Float(maxFrameCount))
      _ = b.memoryWrite(rowIdxCell, b.cast(ceilSlot, to: .int), ceilWrapped)

      // Write weighted gradients for floor and ceil
      let oneMinusFrac = one - frac
      let frameBase = frameIdx * remainingSizeFloat

      b.parallelRange(remainingSize) { elemIdx in
        let elemIdxFloat = b.cast(elemIdx, to: .float)
        let gradValue: Expr
        if let cellId = gradCellId {
          let readPos = frameBase + elemIdxFloat
          gradValue = b.memoryRead(cellId, b.cast(readPos, to: .int))
        } else {
          gradValue = scalarGrad
        }
        let writePos = frameBase + elemIdxFloat
        let floorGrad = gradValue * oneMinusFrac
        _ = b.memoryWrite(floorGradCell, b.cast(writePos, to: .int), floorGrad)
        let ceilGrad = gradValue * frac
        _ = b.memoryWrite(ceilGradCell, b.cast(writePos, to: .int), ceilGrad)
      }

      b.use(val: zero)

    case .sampleGradReduce(
      let floorGradCell, let ceilGradCell, let rowIdxCell, _,
      let gradCell, let numRows, let remainingShape, let maxFrameCount):
      // Sum gradient contributions from all frames for each tensor position
      guard node.inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "sampleGradReduce", expected: 1, actual: node.inputs.count)
      }

      _ = b.value(inputs[0])  // Force dependency on write pass

      let remainingSize = remainingShape.reduce(1, *)
      let remainingSizeInt = b.intConstant(remainingSize)
      let remainingSizeFloat = b.constant(Float(remainingSize))
      let zero = b.constant(0.0)
      let maxFrameCountFloat = b.constant(Float(maxFrameCount))
      let frameCount = b.frameCount()

      if DGenGradientConfig.useFastPeekRowGradReduce {
        // Fast scatter-add path: parallelize over (frame, elem)
        let maxRow = b.constant(Float(Swift.max(0, numRows - 1)))
        let totalElems = maxFrameCount * remainingSize
        b.parallelRange(totalElems) { flatIdx in
          let frameIdx = flatIdx / remainingSizeInt
          let elemIdx = flatIdx - frameIdx * remainingSizeInt
          let frameFloat = b.cast(frameIdx, to: .float)
          let elemFloat = b.cast(elemIdx, to: .float)
          let inBounds = frameFloat < frameCount
          let readPos = frameFloat * remainingSizeFloat + elemFloat

          // Floor contribution
          let floorRowRaw = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
          let floorRowRounded = b.floor(floorRowRaw + b.constant(0.5))
          let floorRow = b.min(b.max(floorRowRounded, zero), maxRow)
          let floorGrad = b.memoryRead(floorGradCell, b.cast(readPos, to: .int))
          let floorContrib = b.gswitch(inBounds > zero, floorGrad, zero)
          let floorDest = floorRow * remainingSizeFloat + elemFloat
          _ = b.memoryAccumulate(gradCell, b.cast(floorDest, to: .int), floorContrib)

          // Ceil contribution
          let ceilSlot = frameFloat + maxFrameCountFloat
          let ceilRowRaw = b.memoryRead(rowIdxCell, b.cast(ceilSlot, to: .int))
          let ceilRowRounded = b.floor(ceilRowRaw + b.constant(0.5))
          let ceilRow = b.min(b.max(ceilRowRounded, zero), maxRow)
          let ceilGrad = b.memoryRead(ceilGradCell, b.cast(readPos, to: .int))
          let ceilContrib = b.gswitch(inBounds > zero, ceilGrad, zero)
          let ceilDest = ceilRow * remainingSizeFloat + elemFloat
          _ = b.memoryAccumulate(gradCell, b.cast(ceilDest, to: .int), ceilContrib)
        }
      } else {
        // Legacy row-bin scan path
        let totalElems = numRows * remainingSize
        b.parallelRange(totalElems) { flatIdx in
          let rowIdx = flatIdx / remainingSizeInt
          let elemIdx = flatIdx - rowIdx * remainingSizeInt
          let rowFloat = b.cast(rowIdx, to: .float)
          let elemFloat = b.cast(elemIdx, to: .float)
          let gradSum = b.float(0.0)

          b.loop(maxFrameCount) { frameIdx in
            let frameFloat = b.cast(frameIdx, to: .float)
            let inBounds = frameFloat < frameCount
            let readPos = frameFloat * remainingSizeFloat + elemFloat

            // Floor row contribution
            let floorRow = b.memoryRead(rowIdxCell, b.cast(frameIdx, to: .int))
            let isFloorMatch = b.abs(floorRow - rowFloat) < b.constant(0.5)
            let floorGrad = b.memoryRead(floorGradCell, b.cast(readPos, to: .int))
            let floorValid = inBounds * isFloorMatch
            let floorContrib = b.gswitch(floorValid > zero, floorGrad, zero)
            gradSum.accumulate(floorContrib)

            // Ceil row contribution
            let ceilSlot = frameFloat + maxFrameCountFloat
            let ceilRow = b.memoryRead(rowIdxCell, b.cast(ceilSlot, to: .int))
            let isCeilMatch = b.abs(ceilRow - rowFloat) < b.constant(0.5)
            let ceilGrad = b.memoryRead(ceilGradCell, b.cast(readPos, to: .int))
            let ceilValid = inBounds * isCeilMatch
            let ceilContrib = b.gswitch(ceilValid > zero, ceilGrad, zero)
            gradSum.accumulate(ceilContrib)
          }

          let destPos = rowFloat * remainingSizeFloat + elemFloat
          _ = b.memoryAccumulate(gradCell, b.cast(destPos, to: .int), gradSum.value)
        }
      }

      b.use(val: zero)

    case .peekGradWrite(
      let gradWriteCell, let floorPosCell, let nextPosCell, let fracCell,
      let channelSize, let numChannels, _):
      // Write per-frame scalar grad and interpolation metadata for peek backward.
      // Inputs: [gradOutput (scalar), index, channel]
      guard node.inputs.count == 3 else {
        throw DGenError.insufficientInputs(
          operator: "peekGradWrite", expected: 3, actual: node.inputs.count)
      }

      let gradOutput = b.value(inputs[0])
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
      let floorPos = b.floor(finalReadPos)
      let frac = finalReadPos - floorPos

      // Next position with wrapping at channel boundary
      let nextPos = floorPos + one
      let nextChannelOffset = channelOffset + channelSizeFloat
      let nextPosWrapped = b.gswitch(nextPos >= nextChannelOffset, channelOffset, nextPos)

      // Frame-indexed storage
      let frameIdx = b.frameIndex()
      let slot = b.cast(frameIdx, to: .int)
      _ = b.memoryWrite(gradWriteCell, slot, gradOutput)
      _ = b.memoryWrite(floorPosCell, slot, floorPos)
      _ = b.memoryWrite(nextPosCell, slot, nextPosWrapped)
      _ = b.memoryWrite(fracCell, slot, frac)

      b.use(val: zero)

    case .peekGradReduce(
      let gradWriteCell, let floorPosCell, let nextPosCell, let fracCell,
      let gradCell, let totalSize, let maxFrameCount):
      // Sum per-frame peek contributions into tensor gradient.
      // Input: [peekGradWritePass] (dependency ordering only)
      guard node.inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "peekGradReduce", expected: 1, actual: node.inputs.count)
      }

      _ = b.value(inputs[0])  // Force dependency on write pass

      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      let half = b.constant(0.5)
      let frameCount = b.frameCount()

      b.parallelRange(totalSize) { tensorPos in
        let tensorPosFloat = b.cast(tensorPos, to: .float)
        let gradSum = b.float(0.0)

        b.loop(maxFrameCount) { frameIdx in
          let frameFloat = b.cast(frameIdx, to: .float)
          let inBounds = frameFloat < frameCount
          let slot = b.cast(frameIdx, to: .int)

          let gradOutput = b.memoryRead(gradWriteCell, slot)
          let floorPos = b.memoryRead(floorPosCell, slot)
          let nextPos = b.memoryRead(nextPosCell, slot)
          let frac = b.memoryRead(fracCell, slot)
          let oneMinusFrac = one - frac

          let isFloorMatch = b.abs(floorPos - tensorPosFloat) < half
          let isNextMatch = b.abs(nextPos - tensorPosFloat) < half

          let floorValid = inBounds * isFloorMatch
          let nextValid = inBounds * isNextMatch

          let floorContrib = b.gswitch(floorValid > zero, gradOutput * oneMinusFrac, zero)
          let nextContrib = b.gswitch(nextValid > zero, gradOutput * frac, zero)

          gradSum.accumulate(floorContrib)
          gradSum.accumulate(nextContrib)
        }

        _ = b.memoryAccumulate(gradCell, b.cast(tensorPos, to: .int), gradSum.value)
      }

      b.use(val: zero)

    default: break
    }
  }
}

import Foundation

extension LazyOp {
  func emitStateOp(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy], nodeId: NodeID) throws {
    switch self {
    case .memoryRead(let cellId):
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "memoryRead", expected: 1, actual: inputs.count)
      }
      b.use(val: b.memoryRead(cellId, b.value(inputs[0])))
    case .memoryWrite(let cellId):
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "memoryWrite", expected: 2, actual: inputs.count)
      }
      b.use(val: b.memoryWrite(cellId, b.value(inputs[0]), b.value(inputs[1])))
    case .memoryAccumulate(let cellId):
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "memoryAccumulate", expected: 2, actual: inputs.count)
      }
      b.use(val: b.memoryAccumulate(cellId, b.value(inputs[0]), b.value(inputs[1])))
    case .memoryCellSum(let cellId, let size):
      // Sum all elements in a memory cell
      let acc = b.float(0.0)
      b.loop(size) { i in
        let val = b.memoryRead(cellId, b.cast(i, to: .int))
        acc.accumulate(val)
      }
      b.use(val: acc.value)
    case .tensorAccumulate(let cellId):
      // Atomically add each tensor element to a memory cell
      guard node.inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "tensorAccumulate", expected: 1, actual: node.inputs.count)
      }

      let tensorInput = node.inputs[0]
      guard let inputNode = g.nodes[tensorInput],
        case .tensor(let shape) = inputNode.shape,
        let tensorId = g.nodeToTensor[tensorInput],
        let tensor = g.tensors[tensorId]
      else {
        throw DGenError.tensorError(op: "tensorAccumulate", reason: "requires tensor input")
      }

      let size = shape.reduce(1, *)

      if ctx.frameAwareTensorCells.contains(tensor.cellId),
        let (tensorSize, frameCount) = ctx.g.frameAwareCells[tensor.cellId]
      {
        // Fast frame-aware reduction:
        // 1) zero destination cell in parallel
        // 2) reduce frame axis in chunks and atomically accumulate chunk partials
        //
        // This shortens the inner loop from frameCount to chunkSize and increases
        // parallelism for large-frame training workloads.
        let reductionChunkSize = 256
        let chunkCount = Swift.max(1, (frameCount + reductionChunkSize - 1) / reductionChunkSize)
        let tensorSizeInt = b.intConstant(tensorSize)
        let sizeInt = b.intConstant(size)
        let frameCountInt = b.intConstant(frameCount)
        let zero = b.constant(0.0)

        b.parallelRange(size) { elemIdx in
          let elemIdxInt = b.cast(elemIdx, to: .int)
          _ = b.memoryWrite(cellId, elemIdxInt, zero)
        }

        b.parallelRange(size * chunkCount) { laneIdx in
          let laneIdxInt = b.cast(laneIdx, to: .int)
          let elemIdxInt = laneIdxInt % sizeInt
          let chunkIdxInt = laneIdxInt / sizeInt
          let frameStart = chunkIdxInt * b.intConstant(reductionChunkSize)
          let localSum = b.float(0.0)

          b.loop(reductionChunkSize) { chunkOffset in
            let frameIdxInt = frameStart + chunkOffset
            let inBounds = frameIdxInt < frameCountInt
            let readPos = frameIdxInt * tensorSizeInt + elemIdxInt
            let val = b.gswitch(inBounds, b.memoryRead(tensor.cellId, readPos), zero)
            localSum.accumulate(val)
          }

          _ = b.memoryAccumulate(cellId, elemIdxInt, localSum.value)
        }
      } else if !tensor.transforms.isEmpty {
        // Use tensorRead to walk the transform chain
        b.parallelRange(size) { idx in
          let indices = b.flatToMultiIndex(b.cast(idx, to: .float), shape)
          let val = b.tensorRead(tensor, indices: indices)
          _ = b.memoryAccumulate(cellId, b.cast(idx, to: .int), val)
        }
      } else {
        // Direct linear read (no transforms, no frame awareness)
        b.parallelRange(size) { idx in
          let val = b.memoryRead(tensor.cellId, b.cast(idx, to: .int))
          _ = b.memoryAccumulate(cellId, b.cast(idx, to: .int), val)
        }
      }

      ctx.values[nodeId] = .empty

    case .historyWrite(let cellId):
      // Unified history write - handles both scalar and tensor based on cellToTensor mapping
      // historyWrite is pass-through: stores value AND outputs it for downstream use.
      // This ensures historyWrite is in the computation graph for correct BPTT.
      if let tensorId = g.cellToTensor[cellId], g.tensors[tensorId] != nil {
        // Tensor write: copy from input tensor to history cell
        guard node.inputs.count >= 1 else {
          throw DGenError.insufficientInputs(
            operator: "historyWrite", expected: 1, actual: node.inputs.count)
        }
        let inputTensorId = g.nodeToTensor[node.inputs[0]]!
        let inputCellId = g.tensors[inputTensorId]!.cellId

        guard let index = ctx.tensorIndices[nodeId] else {
          throw DGenError.insufficientInputs(
            operator: "historyWrite", expected: 1, actual: node.inputs.count)
        }
        let idx = b.value(index, scalarType: .int)
        // tload for cached register, but ALWAYS write - history persists across frames
        let value = b.tload(inputCellId, idx)
        _ = b.memoryWrite(cellId, b.cast(idx, to: .int), value)
        // Pass-through: output is same as input tensor
        ctx.values[nodeId] = .empty
      } else {
        // Scalar write + pass-through
        guard inputs.count == 1 else {
          throw DGenError.insufficientInputs(
            operator: "history write", expected: 1, actual: inputs.count)
        }
        let inputVal = b.value(inputs[0])
        _ = b.store(cellId, inputVal)
        // Pass-through: output the input value so downstream ops can use it
        b.use(val: inputVal)
      }
    case .param(let cellId):
      b.use(val: b.load(cellId))
    case .historyReadWrite(let cellId):
      // for simd its beyond just this -- we need to ensure that we shift the results 1
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "history write", expected: 1, actual: inputs.count)
      }
      b.use(val: u_historyWrite(cellId: cellId, b.value(inputs[0]))(b))
    case .historyRead(let cellId):
      // Unified history read - handles both scalar and tensor based on cellToTensor mapping
      if let tensorId = g.cellToTensor[cellId], g.tensors[tensorId] != nil {
        // Tensor read: copy from history cell to output tensor
        let outputTensorId = g.nodeToTensor[node.id]!
        let outputCellId = g.tensors[outputTensorId]!.cellId

        guard let index = ctx.tensorIndices[nodeId] else {
          throw DGenError.insufficientInputs(
            operator: "historyWrite", expected: 1, actual: node.inputs.count)
        }
        let idx = b.value(index, scalarType: .int)
        let value = b.tload(cellId, idx)
        _ = b.tstore(outputCellId, idx, value)
        // Register placeholder for downstream ops
        ctx.values[nodeId] = .empty
      } else {
        // Scalar read
        b.use(val: b.load(cellId))
      }
    case .latch(let cellId):
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "latch", expected: 2, actual: inputs.count)
      }

      // Check if we're in a tensor context
      if let tensorIndex = ctx.tensorIndices[nodeId] {
        // Tensor latch: read inputs from tensors, use indexed state
        // Mark as requiring scalar execution - latch state updates sample-by-sample

        let value = try b.readInput(node, inputs, at: 0)
        let cond = try b.readInput(node, inputs, at: 1)
        let idx = b.value(tensorIndex, scalarType: .int)

        let zero = b.constant(0.0)

        // Load current latched value from indexed position
        let latched = b.memoryRead(cellId, b.cast(idx, to: .int))

        // If cond > 0, store new value; else keep latched
        // Use gswitch for SIMD compatibility
        let newLatched = b.gswitch(cond > zero, value, latched)

        // Store the result
        _ = b.memoryWrite(cellId, b.cast(idx, to: .int), newLatched)

        // Output the latched value (returns old value, like scalar latch)
        try b.writeOutput(node, latched)
      } else {
        // Scalar latch: use original implementation
        let value = b.value(inputs[0])
        let cond = b.value(inputs[1])
        b.use(val: u_latch(cellId, value: value, cond: cond)(b))
      }
    case .accum(let cellId):
      guard inputs.count == 4 else {
        throw DGenError.insufficientInputs(
          operator: "accum", expected: 4, actual: inputs.count)
      }

      // Check if we're in a tensor context
      if let tensorIndex = ctx.tensorIndices[nodeId] {
        // Tensor accum: read inputs from tensors, use indexed state
        // Mark as requiring scalar execution - accum state accumulates sample-by-sample

        let incr = try b.readInput(node, inputs, at: 0)
        let reset = try b.readInput(node, inputs, at: 1)
        let min = try b.readInput(node, inputs, at: 2)
        let max = try b.readInput(node, inputs, at: 3)
        let idx = b.value(tensorIndex, scalarType: .int)

        let zero = b.constant(0.0)
        let span = max - min

        // Load current state from indexed position
        let prevAcc = b.memoryRead(cellId, b.cast(idx, to: .int))

        let nextCand = prevAcc + incr
        let next = b.gswitch(reset > zero, min, nextCand)

        // Modulo wrap to [min, max)
        let rel = next - min
        let k = b.floor(rel / span)
        let wBase = next - (k * span)

        // Correct if >= max, using gswitch for SIMD compatibility
        let corrected = b.gswitch(wBase >= max, wBase - span, wBase)

        // Reset override using gswitch
        let finalValue = b.gswitch(reset > zero, min, corrected)

        // Store final value
        _ = b.memoryWrite(cellId, b.cast(idx, to: .int), finalValue)

        // Output the previous value (like scalar accum)
        try b.writeOutput(node, prevAcc)
      } else {
        // Scalar accum: use original implementation
        let (incr, reset, min, max) = b.values(inputs, count: 4)
        b.use(val: u_accum(cellId, incr: incr, reset: reset, min: min, max: max)(b))
      }
    case .click(let cellId):
      guard inputs.count == 0 else {
        throw DGenError.insufficientInputs(
          operator: "click", expected: 0, actual: inputs.count)
      }
      b.use(val: u_click(cellId)(b))
    case .noise(let cellId):
      guard inputs.count == 0 else {
        throw DGenError.insufficientInputs(
          operator: "noise", expected: 0, actual: inputs.count)
      }
      b.use(val: u_noise(cellId)(b))
    case .phasor(let cellId):
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "phasor", expected: 2, actual: inputs.count)
      }

      // Check if we're in a tensor context
      if let tensorIndex = ctx.tensorIndices[nodeId] {
        // Tensor phasor: read freq from tensor, use indexed state
        // Mark as requiring scalar execution - phasor state accumulates sample-by-sample

        let freq = try b.readInput(node, inputs, at: 0)
        let reset = try b.readInput(node, inputs, at: 1)
        let idx = b.value(tensorIndex, scalarType: .int)

        // Phasor accumulator logic with indexed state
        // Uses gswitch instead of if statements for SIMD compatibility
        let sampleRate = b.constant(b.ctx.g.sampleRate)
        let incr = freq / sampleRate
        let zero = b.constant(0.0)
        let one = b.constant(1.0)

        // Load current state from indexed position
        let prevPhase = b.memoryRead(cellId, b.cast(idx, to: .int))

        let nextCand = prevPhase + incr
        let next = b.gswitch(reset > zero, zero, nextCand)

        // Modulo wrap to [0, 1)
        let k = b.floor(next)
        let wBase = next - k

        // Correct if >= 1, using gswitch for SIMD compatibility
        let corrected = b.gswitch(wBase >= one, wBase - one, wBase)

        // Reset override using gswitch
        let finalValue = b.gswitch(reset > zero, zero, corrected)

        // Store final value
        _ = b.memoryWrite(cellId, b.cast(idx, to: .int), finalValue)

        // Output the previous value (like scalar phasor)
        try b.writeOutput(node, prevPhase)
      } else {
        // Scalar phasor: use original implementation
        let (freq, reset) = b.values(inputs, count: 2)
        b.use(val: u_phasor(cellId, freq: freq, reset: reset)(b))
      }
    case .deterministicPhasor:
      // Stateless phasor for constant frequency - fully parallelizable
      // phase = fmod(freq / sampleRate * threadIndex, 1.0)
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "deterministicPhasor", expected: 1, actual: inputs.count)
      }
      // Use emitUnaryOp to handle both scalar and tensor cases
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { freq in
        let sampleRate = b.constant(g.sampleRate)
        // Use currentFrameIndex which returns the correct frame index in both normal
        // and frame-aware tensor blocks
        let frameIdx = b.currentFrameIndex()
        // Compute phase directly: (freq / sampleRate * frameIndex) mod 1.0
        let phaseIncrement = freq / sampleRate
        let rawPhase = phaseIncrement * frameIdx
        return rawPhase - b.floor(rawPhase)  // fmod equivalent for positive values
      }
    case .gradDeterministicPhasor:
      // Gradient for deterministic phasor: d(phase)/d(freq) = frameIndex / sampleRate
      // inputs: [gradOutput, sampleRate]
      // Use currentFrameIndex() - matches forward pass which uses currentFrameIndex() for tensor contexts
      guard inputs.count == 2 else { fatalError("gradDeterministicPhasor requires 2 inputs") }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { gradOut, sampleRate in
        let frameIdx = b.currentFrameIndex()
        let gradFreq = gradOut * frameIdx / sampleRate
        return gradFreq
      }
    case .output(let outputNumber):
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "output", expected: 1, actual: inputs.count)
      }
      // Walk through seq nodes to find the actual value source.
      // backward() chains gradient side effects via seq(sideEffect, lossNode),
      // so the output's direct input may be a seq rather than the loss tensor.
      var sourceId = node.inputs[0]
      while let srcNode = ctx.g.nodes[sourceId], case .seq = srcNode.op,
        let lastInput = srcNode.inputs.last
      {
        sourceId = lastInput
      }

      // If the source is a shape-[1] tensor, read from its memory cell
      // instead of referencing a variable that may live in another kernel scope.
      let outputValue: Expr
      if let srcNode = ctx.g.nodes[sourceId],
        case .tensor(let shape) = srcNode.shape,
        shape.reduce(1, *) == 1,
        let tensorId = ctx.g.nodeToTensor[sourceId],
        let tensor = ctx.g.tensors[tensorId]
      {
        outputValue = b.memoryRead(tensor.cellId, b.cast(b.constant(0), to: .int))
      } else {
        outputValue = b.value(inputs[0])
      }
      b.use(val: b.output(outputNumber, outputValue))
    case .input(let inputNumber):
      b.use(val: b.input(inputNumber))
    case .seq:
      // Sequential execution - assumes all input nodes have been emitted in topological order
      // This operator simply returns the value of the last input
      guard node.inputs.count >= 2 else {
        throw DGenError.insufficientInputs(
          operator: "seq", expected: 2, actual: node.inputs.count)
      }

      // Use the value of the last input node (all inputs should already be emitted)
      if let lastInputId = node.inputs.last,
        let lastValue = ctx.values[lastInputId]
      {
        b.use(val: b.value(lastValue))
      } else {
        throw DGenError.insufficientInputs(
          operator: "seq", expected: node.inputs.count,
          actual: node.inputs.compactMap { ctx.values[$0] }.count)
      }

    default: break
    }
  }
}

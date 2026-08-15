import Foundation

extension LazyOp {
  /// Emits the three temporal-adjoint phases used by phasor and accumulator:
  /// frame-parallel store, static reverse scan, then frame-parallel read.
  func emitTemporalGradient(
    b: IRBuilder,
    ctx: IRContext,
    g: Graph,
    node: Node,
    inputs: [Lazy],
    nodeId: NodeID
  ) throws {
    switch self {
    case .temporalGradStore(let gradCell, let resetCell, let elementCount):
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "temporalGradStore", expected: 2, actual: inputs.count)
      }

      let frame = b.currentFrameIndex()
      let frameInt = b.cast(frame, to: .int)
      let reset = b.value(inputs[1])
      _ = b.memoryWrite(resetCell, frameInt, reset)

      // A shape-[1] tensor phasor also has elementCount == 1, but its grad
      // lives in tensor storage — the scalar fast path would read an unwritten
      // scalar slot and silently store zeros.
      let gradInputIsTensor: Bool
      if case .tensor = g.nodes[node.inputs[0]]?.shape {
        gradInputIsTensor = true
      } else {
        gradInputIsTensor = false
      }
      if elementCount == 1 && !gradInputIsTensor {
        _ = b.memoryWrite(gradCell, frameInt, b.value(inputs[0]))
      } else {
        _ = b.value(inputs[0])  // preserve the graph dependency
        guard let tensorId = g.nodeToTensor[node.inputs[0]],
          let tensor = g.tensors[tensorId]
        else {
          throw DGenError.missingTensorID
        }
        let width = b.intConstant(elementCount)
        b.parallelRange(elementCount) { element in
          let elementInt = b.cast(element, to: .int)
          let grad = b.tensorRead(tensor, flatIdx: elementInt, shape: tensor.shape)
          let offset = frameInt * width + elementInt
          _ = b.memoryWrite(gradCell, offset, grad)
        }
      }
      b.use(val: b.constant(0.0))

    case .temporalGradScan(let gradCell, let resetCell, let outputCell, let elementCount):
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "temporalGradScan", expected: 1, actual: inputs.count)
      }
      _ = b.value(inputs[0])  // hard dependency on the completed store pass

      let frameCount = b.frameCount()
      let width = b.intConstant(elementCount)
      let zero = b.constant(0.0)
      b.parallelRange(elementCount) { element in
        let elementInt = b.cast(element, to: .int)
        let carry = b.float(0.0)
        b.ops.append(UOp(op: .beginReverseLoop(frameCount.lazy), value: .empty))

        let frame = b.frameIndex()
        let frameInt = b.cast(frame, to: .int)
        let reset = b.memoryRead(resetCell, frameInt)
        let activeCarry = b.gswitch(reset > zero, zero, carry.value)
        let offset = frameInt * width + elementInt
        _ = b.memoryWrite(outputCell, offset, activeCarry)

        // y[n] is the pre-update state. Its gradient contributes to the state
        // entering n even when reset[n] cuts the later recurrence.
        let upstream = b.memoryRead(gradCell, offset)
        carry.mutate(to: upstream + activeCarry)
        b.ops.append(UOp(op: .endLoop, value: .empty))
      }
      b.use(val: zero)

    case .temporalGradRead(let outputCell, let shape, let scaleBySampleRate):
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "temporalGradRead", expected: 1, actual: inputs.count)
      }
      _ = b.value(inputs[0])  // hard dependency on the completed scan pass

      let elementCount = Swift.max(1, shape.reduce(1, *))
      let frame = b.currentFrameIndex()
      let frameInt = b.cast(frame, to: .int)
      let elementInt: Expr
      if let tensorIndex = ctx.tensorIndices[nodeId] {
        elementInt = b.value(tensorIndex, scalarType: .int)
      } else {
        elementInt = b.intConstant(0)
      }
      let offset = frameInt * b.intConstant(elementCount) + elementInt
      var result = b.memoryRead(outputCell, offset)
      if scaleBySampleRate {
        result = result / b.hostSampleRate()
      }
      try b.writeOutput(node, result)

    default:
      break
    }
  }
}

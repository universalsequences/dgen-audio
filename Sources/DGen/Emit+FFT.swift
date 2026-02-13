import Foundation

extension LazyOp {
  func emitFFT(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy], nodeId: NodeID) throws {
    switch self {
    case .overlapAdd(let windowSize, let hopSize, let outputRingCell, let readPosCell, let counterCell):
      // Overlap-add: scatter-add a window into a ring buffer, emit one sample per frame
      // Input: tensor of size windowSize (time-domain samples from IFFT)
      // Output: scalar (one sample per frame)

      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "overlapAdd", expected: 1, actual: inputs.count)
      }

      // Get input tensor
      guard let inputNodeId = node.inputs.first,
        let inputTensorId = g.nodeToTensor[inputNodeId],
        let inputTensor = g.tensors[inputTensorId]
      else {
        throw DGenError.tensorError(op: "overlapAdd", reason: "missing input tensor")
      }

      let winSizeFloat = b.constant(Float(windowSize))
      let hopSizeFloat = b.constant(Float(hopSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      let readPos = b.memoryRead(readPosCell, zero)
      let counter = b.memoryRead(counterCell, zero)

      // Scatter-add FIRST: when counter == 0, add input window to ring buffer
      // Must happen before read so hop boundary frames include the current window's contribution
      let shouldScatter = counter == zero
      b.if_(shouldScatter) {
        b.loop(windowSize) { i in
          let sample = b.tensorRead(inputTensor, flatIdx: i, shape: [windowSize])
          let outPos = readPos + i
          let wrappedOutPos = b.gswitch(
            outPos >= winSizeFloat, outPos - winSizeFloat, outPos)
          let outPosInt = b.cast(wrappedOutPos, to: .int)
          let existing = b.memoryRead(outputRingCell, outPosInt)
          _ = b.memoryWrite(outputRingCell, outPosInt, existing + sample)
        }
      }

      // Ring buffer read + clear (after scatter so current window is included)
      let outputSample = b.memoryRead(outputRingCell, b.cast(readPos, to: .int))
      _ = b.memoryWrite(outputRingCell, b.cast(readPos, to: .int), zero)

      // ReadPos advance
      let nextReadPos = readPos + one
      let wrappedReadPos = b.gswitch(nextReadPos >= winSizeFloat, zero, nextReadPos)
      _ = b.memoryWrite(readPosCell, zero, wrappedReadPos)

      // Counter advance
      let nextCounter = counter + one
      let wrappedCounter = b.gswitch(nextCounter >= hopSizeFloat, zero, nextCounter)
      _ = b.memoryWrite(counterCell, zero, wrappedCounter)

      // Use the output sample (ring buffer read from current position)
      b.use(val: outputSample)

    case .overlapAddGradStore(let gradStoreCell):
      // Store per-frame output gradient to shared memory for later gathering
      let gradOutput = b.value(inputs[0])
      let frameIdx = b.currentFrameIndex()
      _ = b.memoryWrite(gradStoreCell, frameIdx, gradOutput)
      b.use(val: b.constant(0.0))

    case .overlapAddGradGather(let windowSize, let hopSize, let gradStoreCell, let gradInputCell):
      // Gather stored gradients into per-frame gradient tensor
      // For hop frame h: grad_input[h][i] = grad_output[h + offset(i)]
      //   where offset(0) = windowSize, offset(i>0) = i
      // For non-hop frames: grad_input = 0
      _ = b.value(inputs[0])  // force dependency on store phase
      let frameIdx = b.currentFrameIndex()
      let frameCount = b.frameCount()
      let hopSizeFloat = b.constant(Float(hopSize))
      let winSizeFloat = b.constant(Float(windowSize))
      let winSizeInt = b.intConstant(windowSize)
      let zero = b.constant(0.0)

      let frameFloat = b.cast(frameIdx, to: .float)
      let frameInt = b.cast(frameIdx, to: .int)
      let frameCountFloat = b.cast(frameCount, to: .float)
      let maxReadFrame = frameCountFloat - b.constant(1.0)

      // Check if this is a hop frame: frameIdx % hopSize == 0
      let modResult = frameFloat - b.floor(frameFloat / hopSizeFloat) * hopSizeFloat
      let isHopFrame = modResult == zero

      b.loop(windowSize) { i in
        let iFloat = b.cast(i, to: .float)
        // offset(0) = windowSize, offset(i>0) = i (all in float to avoid select ambiguity)
        let offset = b.gswitch(iFloat == zero, winSizeFloat, iFloat)
        let readFrame = frameFloat + offset
        let inBounds = readFrame < frameCountFloat

        // Read stored gradient (clamped to avoid OOB)
        let clampedFrame = b.min(readFrame, maxReadFrame)
        let gradVal = b.memoryRead(gradStoreCell, b.cast(clampedFrame, to: .int))

        // Only use if hop frame AND in bounds
        let validGrad = b.gswitch(isHopFrame, b.gswitch(inBounds, gradVal, zero), zero)

        // Write to frame-indexed gradient tensor cell
        let iInt = b.cast(i, to: .int)
        let writeIdx = frameInt * winSizeInt + iInt
        _ = b.memoryWrite(gradInputCell, writeIdx, validGrad)
      }
      b.use(val: zero)

    case .bufferViewGradStore(let gradCell, let windowSize):
      // Store per-frame tensor gradient to frame-indexed memory
      // Input: gradient tensor (from autodiff)
      // Output: side effect (stores gradCell[frame * windowSize + elem])
      _ = b.value(inputs[0])  // force dependency
      let inputNodeId = node.inputs[0]
      guard let tensorId = g.nodeToTensor[inputNodeId],
        let tensor = g.tensors[tensorId]
      else {
        throw DGenError.missingTensorID
      }

      let bvFrameIdx = b.currentFrameIndex()
      let bvFrameInt = b.cast(bvFrameIdx, to: .int)
      let bvWinSizeInt = b.intConstant(windowSize)

      b.loop(windowSize) { j in
        let jInt = b.cast(j, to: .int)
        let gradElem = b.tensorRead(tensor, flatIdx: jInt, shape: tensor.shape)
        let writeIdx = bvFrameInt * bvWinSizeInt + jInt
        _ = b.memoryWrite(gradCell, writeIdx, gradElem)
      }
      b.use(val: b.constant(0.0))

    case .bufferViewGradRead(let gradCell, let windowSize):
      // Sum overlapping window contributions to produce scalar gradient per frame
      // Sample at position p appears in windows at frames p, p+1, ..., p+windowSize-1
      // In window frame w, sample p is at element offset (windowSize - 1 - (w - p))
      _ = b.value(inputs[0])  // force dependency on store phase

      let p = b.threadIndex()  // absolute sample position
      let bvrWinSizeInt = b.intConstant(windowSize)
      let bvrFrameCount = b.frameCount()

      let gradSum = b.float(0.0)
      b.loop(windowSize) { i in
        let iInt = b.cast(i, to: .int)
        let iFloat = b.cast(i, to: .float)
        let pFloat = b.cast(p, to: .float)
        let w = pFloat + iFloat  // window frame index (float for comparisons)
        let offsetInt = bvrWinSizeInt - b.intConstant(1) - iInt  // offset in that window
        let clampedW = b.min(w, b.cast(bvrFrameCount, to: .float) - b.constant(1.0))
        let idx = b.cast(clampedW, to: .int) * bvrWinSizeInt + offsetInt
        let contrib = b.memoryRead(gradCell, idx)
        let inBounds = w < b.cast(bvrFrameCount, to: .float)
        let safeContrib = b.gswitch(inBounds, contrib, b.constant(0.0))
        gradSum.accumulate(safeContrib)
      }
      b.use(val: gradSum.value)

    default: break
    }
  }
}

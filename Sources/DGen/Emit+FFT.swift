import Foundation

extension LazyOp {
  func emitFFT(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy], nodeId: NodeID) throws {
    switch self {
    case .fft(
      let windowSize, let hopSize, let scratchCell, let ringBufferCell, let writePosCell,
      let counterCell):
      // FFT using Cooley-Tukey algorithm with ring buffer for sample history
      // Input: signal (scalar per frame)
      // Output: tensor [numBins, 2] where numBins = windowSize/2 + 1
      // hopSize: only compute FFT every hopSize frames (reduces CPU)

      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "fft", expected: 1, actual: inputs.count)
      }
      guard windowSize > 0 && (windowSize & (windowSize - 1)) == 0 else {
        throw DGenError.tensorError(op: "fft", reason: "windowSize must be a power of 2")
      }

      let numBins = windowSize / 2 + 1
      let numStages = Int(log2(Double(windowSize)))

      // Get output tensor
      guard let outTensorId = g.nodeToTensor[node.id],
        let outTensor = g.tensors[outTensorId]
      else {
        throw DGenError.tensorError(op: "fft", reason: "missing output tensor")
      }

      let sig = b.value(inputs[0])

      // Scratch memory layout:
      // scratch[0..<windowSize] = real components
      // scratch[windowSize..<windowSize*2] = imaginary components
      let imagOffset = windowSize

      let winSizeFloat = b.constant(Float(windowSize))
      let hopSizeFloat = b.constant(Float(hopSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      // Load write position and hop counter
      let writePos = b.memoryRead(writePosCell, zero)
      let counter = b.memoryRead(counterCell, zero)

      // Write current sample to ring buffer at writePos
      _ = b.memoryWrite(ringBufferCell, b.cast(writePos, to: .int), sig)

      // Update write position: (writePos + 1) % windowSize
      let nextWritePos = writePos + one
      let wrappedWritePos = b.gswitch(nextWritePos >= winSizeFloat, zero, nextWritePos)
      _ = b.memoryWrite(writePosCell, zero, wrappedWritePos)

      // Update counter: (counter + 1) % hopSize
      let nextCounter = counter + one
      let wrappedCounter = b.gswitch(nextCounter >= hopSizeFloat, zero, nextCounter)
      _ = b.memoryWrite(counterCell, zero, wrappedCounter)

      // Only compute FFT when counter == 0 (every hopSize frames)
      let shouldCompute = counter == zero

      // Wrap entire FFT computation in if-statement for efficiency
      // This skips all expensive loops when shouldCompute is false
      b.if_(shouldCompute) {
        // 1. Load samples from ring buffer into scratch (imaginary = 0)
        // Read from oldest to newest: start from wrappedWritePos (oldest) and go around
        b.loop(windowSize) { n in
          let nFloat = b.cast(n, to: .float)
          // Read position: (writePos + n) % windowSize gives oldest to newest
          // Use wrappedWritePos which now points to oldest sample
          let readIdx = wrappedWritePos + nFloat
          let wrappedReadIdx = b.gswitch(
            readIdx >= winSizeFloat, readIdx - winSizeFloat, readIdx)
          let sample = b.memoryRead(ringBufferCell, b.cast(wrappedReadIdx, to: .int))
          _ = b.memoryWrite(scratchCell, b.cast(n, to: .int), sample)
          _ = b.memoryWrite(
            scratchCell, b.cast(n, to: .int) + b.constant(Float(imagOffset)),
            b.constant(0.0))
        }

        // 2. Bit-reversal permutation
        b.loop(windowSize) { i in
          // Compute bit-reversed index
          var rev = b.constant(0.0)
          var n = b.cast(i, to: .float)
          for _ in 0..<numStages {
            rev = rev * b.constant(2.0) + (n % b.constant(2.0))
            n = b.floor(n / b.constant(2.0))
          }

          let iFloat = b.cast(i, to: .float)
          // Swap if i < rev (avoid double-swap)
          let shouldSwap = iFloat < rev
          let iInt = b.cast(i, to: .int)
          let revInt = b.cast(rev, to: .int)

          // Load values at i and rev
          let tempRealI = b.memoryRead(scratchCell, iInt)
          let tempImagI = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
          let tempRealRev = b.memoryRead(scratchCell, revInt)
          let tempImagRev = b.memoryRead(
            scratchCell, revInt + b.constant(Float(imagOffset)))

          // Conditionally swap
          let newRealI = b.gswitch(shouldSwap, tempRealRev, tempRealI)
          let newImagI = b.gswitch(shouldSwap, tempImagRev, tempImagI)
          let newRealRev = b.gswitch(shouldSwap, tempRealI, tempRealRev)
          let newImagRev = b.gswitch(shouldSwap, tempImagI, tempImagRev)

          _ = b.memoryWrite(scratchCell, iInt, newRealI)
          _ = b.memoryWrite(scratchCell, iInt + b.constant(Float(imagOffset)), newImagI)
          _ = b.memoryWrite(scratchCell, revInt, newRealRev)
          _ = b.memoryWrite(
            scratchCell, revInt + b.constant(Float(imagOffset)), newImagRev)
        }

        // 3. Butterfly stages
        for stage in 0..<numStages {
          let butterflySize = 1 << (stage + 1)  // 2, 4, 8, ...
          let halfSize = butterflySize / 2
          let numGroups = windowSize / butterflySize

          b.loop(numGroups) { group in
            b.loop(halfSize) { k in
              let groupFloat = b.cast(group, to: .float)
              let kFloat = b.cast(k, to: .float)
              let butterflySizeFloat = b.constant(Float(butterflySize))
              let halfSizeFloat = b.constant(Float(halfSize))

              let i = groupFloat * butterflySizeFloat + kFloat
              let j = i + halfSizeFloat

              // Twiddle factor: W = e^(-2*pi*i*k/butterflySize)
              let angle = b.constant(-2.0) * b.pi * kFloat / butterflySizeFloat
              let wr = b.cos(angle)
              let wi = b.sin(angle)

              let iInt = b.cast(i, to: .int)
              let jInt = b.cast(j, to: .int)

              // Load values
              let ar = b.memoryRead(scratchCell, iInt)
              let ai = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
              let br = b.memoryRead(scratchCell, jInt)
              let bi = b.memoryRead(scratchCell, jInt + b.constant(Float(imagOffset)))

              // Butterfly: (ar,ai) + W*(br,bi) and (ar,ai) - W*(br,bi)
              // W*(br,bi) = (wr*br - wi*bi, wr*bi + wi*br)
              let tr = wr * br - wi * bi
              let ti = wr * bi + wi * br

              _ = b.memoryWrite(scratchCell, iInt, ar + tr)
              _ = b.memoryWrite(
                scratchCell, iInt + b.constant(Float(imagOffset)), ai + ti)
              _ = b.memoryWrite(scratchCell, jInt, ar - tr)
              _ = b.memoryWrite(
                scratchCell, jInt + b.constant(Float(imagOffset)), ai - ti)
            }
          }
        }

        // 4. Copy first numBins to output tensor [numBins, 2]
        // Output layout: column-major for peek compatibility
        // Channel 0 (real): offsets 0, 1, 2, ... numBins-1
        // Channel 1 (imag): offsets numBins, numBins+1, ... 2*numBins-1
        b.loop(numBins) { k in
          let kInt = b.cast(k, to: .int)
          let real = b.memoryRead(scratchCell, kInt)
          let imag = b.memoryRead(scratchCell, kInt + b.constant(Float(imagOffset)))
          _ = b.memoryWrite(outTensor.cellId, kInt, real)
          _ = b.memoryWrite(outTensor.cellId, kInt + b.constant(Float(numBins)), imag)
        }
      }

      // Register output for downstream ops
      ctx.values[nodeId] = .empty

    case .ifft(
      let windowSize, let hopSize, let scratchCell, let outputRingCell, let readPosCell,
      let counterCell):
      // IFFT using Cooley-Tukey algorithm with overlap-add for reconstruction
      // Input: spectrum tensor [numBins, 2] where numBins = windowSize/2 + 1
      // Output: scalar (one sample per frame via overlap-add)

      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "ifft", expected: 1, actual: inputs.count)
      }
      guard windowSize > 0 && (windowSize & (windowSize - 1)) == 0 else {
        throw DGenError.tensorError(op: "ifft", reason: "windowSize must be a power of 2")
      }

      let numBins = windowSize / 2 + 1
      let numStages = Int(log2(Double(windowSize)))
      let imagOffset = windowSize  // Scratch layout: real[0..<N], imag[N..<2N]

      // Get input tensor (spectrum)
      guard let inputNodeId = node.inputs.first,
        let inputTensorId = g.nodeToTensor[inputNodeId],
        let inputTensor = g.tensors[inputTensorId]
      else {
        throw DGenError.tensorError(op: "ifft", reason: "missing input spectrum tensor")
      }

      let winSizeFloat = b.constant(Float(windowSize))
      let hopSizeFloat = b.constant(Float(hopSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)

      // Load read position and hop counter
      let readPos = b.memoryRead(readPosCell, zero)
      let counter = b.memoryRead(counterCell, zero)

      // Output current sample from ring buffer, then clear it for next overlap-add cycle
      let outputSample = b.memoryRead(outputRingCell, b.cast(readPos, to: .int))
      _ = b.memoryWrite(outputRingCell, b.cast(readPos, to: .int), zero)

      // Update read position: (readPos + 1) % windowSize
      let nextReadPos = readPos + one
      let wrappedReadPos = b.gswitch(nextReadPos >= winSizeFloat, zero, nextReadPos)
      _ = b.memoryWrite(readPosCell, zero, wrappedReadPos)

      // Update counter: (counter + 1) % hopSize
      let nextCounter = counter + one
      let wrappedCounter = b.gswitch(nextCounter >= hopSizeFloat, zero, nextCounter)
      _ = b.memoryWrite(counterCell, zero, wrappedCounter)

      // Only compute IFFT when counter == 0
      let shouldCompute = counter == zero
      b.if_(shouldCompute) {
        // 1. Load spectrum into scratch with conjugate symmetry to get full N points
        // Input layout (column-major): real[0..<numBins], imag[numBins..<2*numBins]
        // For real signal: X[N-k] = conj(X[k]) for k = 1 to N/2-1
        // DC (k=0) and Nyquist (k=N/2) have no imaginary part in output

        b.loop(windowSize) { n in
          let nInt = b.cast(n, to: .int)
          let halfN = b.constant(Float(windowSize / 2))

          // Determine which input bin to read from
          let isFirstHalf = n <= halfN
          let inputBin = b.gswitch(isFirstHalf, n, winSizeFloat - n)
          let inputBinInt = b.cast(inputBin, to: .int)

          // Read real and imag from input tensor (column-major layout)
          let inReal = b.memoryRead(inputTensor.cellId, inputBinInt)
          let inImag = b.memoryRead(
            inputTensor.cellId, inputBinInt + b.constant(Float(numBins)))

          // For second half (conjugate), negate imaginary part
          let realVal = inReal
          let imagVal = b.gswitch(isFirstHalf, inImag, zero - inImag)

          // Write to scratch
          _ = b.memoryWrite(scratchCell, nInt, realVal)
          _ = b.memoryWrite(scratchCell, nInt + b.constant(Float(imagOffset)), imagVal)
        }

        // 2. Bit-reversal permutation (same as FFT)
        b.loop(windowSize) { i in
          var rev = b.constant(0.0)
          var n = i
          for _ in 0..<numStages {
            rev = rev * b.constant(2.0) + b.mod(n, b.constant(2.0))
            n = b.floor(n / b.constant(2.0))
          }

          let iFloat = i
          let shouldSwap = b.and(iFloat < rev, shouldCompute)

          let iInt = b.cast(i, to: .int)
          let revInt = b.cast(rev, to: .int)

          let tempR = b.memoryRead(scratchCell, iInt)
          let tempI = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
          let revR = b.memoryRead(scratchCell, revInt)
          let revI = b.memoryRead(scratchCell, revInt + b.constant(Float(imagOffset)))

          let newIR = b.gswitch(shouldSwap, revR, tempR)
          let newII = b.gswitch(shouldSwap, revI, tempI)
          let newRevR = b.gswitch(shouldSwap, tempR, revR)
          let newRevI = b.gswitch(shouldSwap, tempI, revI)

          _ = b.memoryWrite(scratchCell, iInt, newIR)
          _ = b.memoryWrite(scratchCell, iInt + b.constant(Float(imagOffset)), newII)
          _ = b.memoryWrite(scratchCell, revInt, newRevR)
          _ = b.memoryWrite(scratchCell, revInt + b.constant(Float(imagOffset)), newRevI)
        }

        // 3. Butterfly stages (IFFT uses POSITIVE twiddle angles)
        var butterflySize = 2
        for _ in 0..<numStages {
          let halfSize = butterflySize / 2
          let numGroups = windowSize / butterflySize

          b.loop(numGroups) { group in
            b.loop(halfSize) { k in
              let i = group * b.constant(Float(butterflySize)) + k
              let j = i + b.constant(Float(halfSize))

              // IFFT twiddle: W = e^(+2πi*k/butterflySize) - POSITIVE angle
              let angle =
                b.constant(2.0) * b.constant(Float.pi) * k
                / b.constant(Float(butterflySize))
              let wr = b.cos(angle)
              let wi = b.sin(angle)

              let iInt = b.cast(i, to: .int)
              let jInt = b.cast(j, to: .int)

              let ar = b.memoryRead(scratchCell, iInt)
              let ai = b.memoryRead(scratchCell, iInt + b.constant(Float(imagOffset)))
              let br = b.memoryRead(scratchCell, jInt)
              let bi = b.memoryRead(scratchCell, jInt + b.constant(Float(imagOffset)))

              // Complex multiply: (wr + i*wi) * (br + i*bi)
              let tr = wr * br - wi * bi
              let ti = wr * bi + wi * br

              // Butterfly
              _ = b.memoryWrite(scratchCell, iInt, ar + tr)
              _ = b.memoryWrite(
                scratchCell, iInt + b.constant(Float(imagOffset)), ai + ti)
              _ = b.memoryWrite(scratchCell, jInt, ar - tr)
              _ = b.memoryWrite(
                scratchCell, jInt + b.constant(Float(imagOffset)), ai - ti)
            }
          }
          butterflySize *= 2
        }

        // 4. Divide by N and add to output ring buffer (overlap-add)
        let invN = b.constant(1.0 / Float(windowSize))
        b.loop(windowSize) { n in
          let nInt = b.cast(n, to: .int)
          let realVal = b.memoryRead(scratchCell, nInt) * invN

          // Calculate output position with wrap-around
          let outPos = readPos + n
          let wrappedOutPos = b.gswitch(
            outPos >= winSizeFloat, outPos - winSizeFloat, outPos)
          let outPosInt = b.cast(wrappedOutPos, to: .int)

          // Overlap-add: accumulate into output buffer
          let existing = b.memoryRead(outputRingCell, outPosInt)
          _ = b.memoryWrite(outputRingCell, outPosInt, existing + realVal)
        }
      }

      // Use the output sample
      b.use(val: outputSample)

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

    default: break
    }
  }
}

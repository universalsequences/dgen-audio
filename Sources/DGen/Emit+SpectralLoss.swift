import Foundation

extension LazyOp {
  func emitSpectralLoss(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy]) throws {
    switch self {
    case .spectralLossFFT(
      let windowSize, _, _,
      let fft1Cell, let fft2Cell, let mag1Cell, let mag2Cell, _):
      // FFT-based spectral loss: forward pass (SIMD-parallel across frames)
      // Uses threadgroup shared memory for butterfly stages (~25-50x faster than device).
      // Precomputed tables (Hann, twiddle, bit-reversal) are passed as tensor inputs.
      guard inputs.count >= 6 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFT", expected: 6, actual: inputs.count)
      }

      // Helper to resolve tensor input NodeID → cell ID
      func tensorCellId(_ nodeId: NodeID) -> CellID {
        let tensorId = g.nodeToTensor[nodeId]!
        return g.tensors[tensorId]!.cellId
      }

      let numBins = windowSize / 2 + 1
      let numStages = Int(log2(Double(windowSize)))
      let fftSize = windowSize * 2  // real + imag
      let imagOffset = windowSize  // Device layout: real[0..<N], imag[N..<2N]

      let sig1 = b.value(inputs[0])
      let sig2 = b.value(inputs[1])
      let hannCellId = tensorCellId(node.inputs[2])
      let twReCellId = tensorCellId(node.inputs[3])
      let twImCellId = tensorCellId(node.inputs[4])
      let bitRevCellId = tensorCellId(node.inputs[5])
      let hopCounter: Expr? = inputs.count > 6 ? b.value(inputs[6]) : nil
      let winSizeFloat = b.constant(Float(windowSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      let frameIdx = b.frameIndex()

      // Per-frame base offsets for device memory (integer to avoid float32 precision loss)
      let fftBaseOffset = frameIdx * b.intConstant(fftSize)
      let magBaseOffset = frameIdx * b.intConstant(numBins)

      // Threadgroup scratch arrays for FFT butterflies (on-chip SRAM, ~2-4 cycle latency).
      // Reused sequentially for both signals.
      // Budget: scratchRe(N) + scratchIm(N) + twRe(N-1) + twIm(N-1) = 4N-2 floats.
      // N=2048 → 32760 bytes < 32KB limit. Hann/bitrev stay in device (O(N) reads, negligible).
      let scratchRe = b.threadgroupScratch(windowSize)
      let scratchIm = b.threadgroupScratch(windowSize)
      // Twiddle factors loaded from device once, then read from scratch during butterflies
      let scratchTwRe = b.threadgroupScratch(windowSize - 1)
      let scratchTwIm = b.threadgroupScratch(windowSize - 1)

      let shouldRun = hopCounter.map { $0 == zero } ?? (one > zero)
      let loss = b.float(0.0)
      b.if_(shouldRun) {
        // Load twiddle tables from device → scratch (once for both signals)
        let twiddleSize = windowSize - 1
        b.loop(twiddleSize) { n in
          let nInt = b.cast(n, to: .int)
          b.scratchWrite(scratchTwRe, nInt, b.memoryRead(twReCellId, nInt))
          b.scratchWrite(scratchTwIm, nInt, b.memoryRead(twImCellId, nInt))
        }

        // Helper: FFT one signal in scratch, copy result to device, compute magnitude
        func emitFFTInScratch(signal: Expr, fftCell: CellID, magCell: CellID) {
          // 1. Load windowed samples into scratch (Hann read from device — O(N), negligible)
          b.loop(windowSize) { n in
            let nInt = b.cast(n, to: .int)
            let nFloat = b.cast(n, to: .float)
            let w = b.memoryRead(hannCellId, nInt)
            let j = b.cast(frameIdx, to: .float) - (winSizeFloat - one) + nFloat
            let s = b.tapeLoad(signal, at: j)
            b.scratchWrite(scratchRe, nInt, s * w)
            b.scratchWrite(scratchIm, nInt, zero)
          }

          // 2. Bit-reversal permutation in scratch (bitrev indices from device — O(N), negligible)
          b.loop(windowSize) { i in
            let iInt = b.cast(i, to: .int)
            let rev = b.memoryRead(bitRevCellId, iInt)
            let iFloat = b.cast(i, to: .float)
            let shouldSwap = iFloat < rev
            let revInt = b.cast(rev, to: .int)

            let tempReI = b.scratchRead(scratchRe, iInt)
            let tempImI = b.scratchRead(scratchIm, iInt)
            let tempReRev = b.scratchRead(scratchRe, revInt)
            let tempImRev = b.scratchRead(scratchIm, revInt)

            let newReI = b.gswitch(shouldSwap, tempReRev, tempReI)
            let newImI = b.gswitch(shouldSwap, tempImRev, tempImI)
            let newReRev = b.gswitch(shouldSwap, tempReI, tempReRev)
            let newImRev = b.gswitch(shouldSwap, tempImI, tempImRev)

            b.scratchWrite(scratchRe, iInt, newReI)
            b.scratchWrite(scratchIm, iInt, newImI)
            b.scratchWrite(scratchRe, revInt, newReRev)
            b.scratchWrite(scratchIm, revInt, newImRev)
          }

          // 3. Butterfly stages entirely in scratch (all reads from threadgroup SRAM)
          for stage in 0..<numStages {
            let butterflySize = 1 << (stage + 1)
            let halfSize = butterflySize / 2
            let numGroups = windowSize / butterflySize
            let numButterflies = numGroups * halfSize

            b.loop(numButterflies) { flatIdx in
              let flatFloat = b.cast(flatIdx, to: .float)
              let halfSizeFloat = b.constant(Float(halfSize))
              let butterflySizeFloat = b.constant(Float(butterflySize))

              let group = b.floor(flatFloat / halfSizeFloat)
              let k = flatFloat - (group * halfSizeFloat)

              let i = group * butterflySizeFloat + k
              let j = i + halfSizeFloat

              let twiddleOffset = b.intConstant(halfSize - 1)
              let kInt = b.cast(k, to: .int)
              let twiddleIdx = twiddleOffset + kInt
              let wr = b.scratchRead(scratchTwRe, twiddleIdx)
              let wi = zero - b.scratchRead(scratchTwIm, twiddleIdx)  // negate for forward FFT

              let iInt = b.cast(i, to: .int)
              let jInt = b.cast(j, to: .int)

              let ar = b.scratchRead(scratchRe, iInt)
              let ai = b.scratchRead(scratchIm, iInt)
              let br = b.scratchRead(scratchRe, jInt)
              let bi = b.scratchRead(scratchIm, jInt)

              let tr = wr * br - wi * bi
              let ti = wr * bi + wi * br

              b.scratchWrite(scratchRe, iInt, ar + tr)
              b.scratchWrite(scratchIm, iInt, ai + ti)
              b.scratchWrite(scratchRe, jInt, ar - tr)
              b.scratchWrite(scratchIm, jInt, ai - ti)
            }
          }

          // 4. Copy FFT result to device (for downstream GradSpec) + compute magnitudes
          let imagOff = b.intConstant(imagOffset)
          b.loop(windowSize) { n in
            let nInt = b.cast(n, to: .int)
            let re = b.scratchRead(scratchRe, nInt)
            let im = b.scratchRead(scratchIm, nInt)
            _ = b.memoryWrite(fftCell, fftBaseOffset + nInt, re)
            _ = b.memoryWrite(fftCell, fftBaseOffset + nInt + imagOff, im)
          }
          b.loop(numBins) { k in
            let kInt = b.cast(k, to: .int)
            let re = b.scratchRead(scratchRe, kInt)
            let im = b.scratchRead(scratchIm, kInt)
            let mag = b.sqrt(re * re + im * im)
            _ = b.memoryWrite(magCell, magBaseOffset + kInt, mag)
          }
        }

        // Process both signals sequentially (scratch arrays reused)
        emitFFTInScratch(signal: sig1, fftCell: fft1Cell, magCell: mag1Cell)
        emitFFTInScratch(signal: sig2, fftCell: fft2Cell, magCell: mag2Cell)

        // 5. Compute loss: sum of squared magnitude differences
        b.loop(numBins) { k in
          let kInt = b.cast(k, to: .int)
          let mag1 = b.memoryRead(mag1Cell, magBaseOffset + kInt)
          let mag2 = b.memoryRead(mag2Cell, magBaseOffset + kInt)
          let diff = mag1 - mag2
          loss.accumulate(diff * diff)
        }
      }

      b.use(val: loss.value)

    case .spectralLossFFTGradSpec(
      let windowSize, let fft1Cell, let fft2Cell,
      let mag1Cell, let mag2Cell, let gradSpec1Cell, let gradSpec2Cell):
      // Compute gradient w.r.t. complex spectrum (frame-aware)
      // Reads from forward pass's per-frame FFT/magnitude cells
      // ∂L/∂X.real = ∂L/∂mag * (real / mag)
      // ∂L/∂X.imag = ∂L/∂mag * (imag / mag)
      // inputs[0] = gradOutput, inputs[1..2] = sig1/sig2 (ordering-only dependencies)
      guard inputs.count >= 1 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradSpec", expected: 1, actual: inputs.count)
      }

      let fftSize = windowSize * 2
      let numBins = windowSize / 2 + 1
      let imagOffset = windowSize
      let gradOutput = b.value(inputs[0])
      let hopCounter: Expr? = inputs.count > 3 ? b.value(inputs[3]) : nil
      let eps = b.constant(1e-8)
      let frameIdx = b.frameIndex()
      let zero = b.constant(0.0)

      // Per-frame base offsets (integer arithmetic for precision)
      let fftBase = frameIdx * b.intConstant(fftSize)
      let magBase = frameIdx * b.intConstant(numBins)
      let gradSpecBase = frameIdx * b.intConstant(fftSize)
      let imagOff = b.intConstant(imagOffset)

      let shouldRun = hopCounter.map { $0 == zero } ?? (zero == zero)
      b.if_(shouldRun) {
        // Compute gradient spectrum in parallel (each bin is independent)
        b.parallelRange(numBins) { k in
          let kInt = b.cast(k, to: .int)

          // Read stored values from forward pass's per-frame cells
          let mag1 = b.memoryRead(mag1Cell, magBase + kInt)
          let mag2 = b.memoryRead(mag2Cell, magBase + kInt)
          let real1 = b.memoryRead(fft1Cell, fftBase + kInt)
          let imag1 = b.memoryRead(fft1Cell, fftBase + kInt + imagOff)
          let real2 = b.memoryRead(fft2Cell, fftBase + kInt)
          let imag2 = b.memoryRead(fft2Cell, fftBase + kInt + imagOff)

          // ∂L/∂mag = 2 * (mag1 - mag2) * gradOutput
          let gradMag1 = b.constant(2.0) * (mag1 - mag2) * gradOutput
          let gradMag2 = b.constant(-2.0) * (mag1 - mag2) * gradOutput

          // Handle division by zero with epsilon
          let safeMag1 = b.max(mag1, eps)
          let safeMag2 = b.max(mag2, eps)

          // ∂L/∂X = gradMag * (X / |X|) = gradMag * (real/mag, imag/mag)
          let gradReal1 = gradMag1 * real1 / safeMag1
          let gradImag1 = gradMag1 * imag1 / safeMag1
          let gradReal2 = gradMag2 * real2 / safeMag2
          let gradImag2 = gradMag2 * imag2 / safeMag2

          // Store gradient spectrum directly (no conjugation needed)
          // IFFT with positive twiddle: Re[Σ X[k]·e^{+jθ}] = Σ [X_real·cos - X_imag·sin]
          // which matches the DFT transpose scatter formula exactly
          _ = b.memoryWrite(gradSpec1Cell, gradSpecBase + kInt, gradReal1)
          _ = b.memoryWrite(gradSpec1Cell, gradSpecBase + kInt + imagOff, gradImag1)
          _ = b.memoryWrite(gradSpec2Cell, gradSpecBase + kInt, gradReal2)
          _ = b.memoryWrite(gradSpec2Cell, gradSpecBase + kInt + imagOff, gradImag2)
        }

        // Zero upper bins (numBins to windowSize-1) — no conjugate fill needed.
        // The loss only depends on bins 0..numBins-1, so the gradient has no
        // contribution from mirror bins. Zeroing prevents garbage from affecting IFFT.
        if windowSize / 2 - 1 > 0 {
          b.parallelRange(windowSize / 2 - 1) { k in
            let kInt = b.cast(k, to: .int) + b.intConstant(numBins)
            _ = b.memoryWrite(gradSpec1Cell, gradSpecBase + kInt, b.constant(0.0))
            _ = b.memoryWrite(gradSpec1Cell, gradSpecBase + kInt + imagOff, b.constant(0.0))
            _ = b.memoryWrite(gradSpec2Cell, gradSpecBase + kInt, b.constant(0.0))
            _ = b.memoryWrite(gradSpec2Cell, gradSpecBase + kInt + imagOff, b.constant(0.0))
          }
        }
      }

      b.use(val: zero)  // Side-effect only

    case .spectralLossFFTGradIFFT(
      let windowSize, let gradSpec1Cell, let gradSpec2Cell,
      let gradTime1Cell, let gradTime2Cell, let windowCell):
      // IFFT to scatter frequency-domain gradients back to time domain (frame-aware)
      // Uses threadgroup shared memory for butterfly stages (~25-50x faster than device).
      // Then multiply by window coefficients for Hann backprop
      // Inputs: [gradSpec, twReNode, twImNode, bitRevNode, ?hopCounter]
      guard inputs.count >= 4 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradIFFT", expected: 4, actual: inputs.count)
      }

      // Helper to resolve tensor input NodeID → cell ID
      func tensorCellId(_ nodeId: NodeID) -> CellID {
        let tensorId = g.nodeToTensor[nodeId]!
        return g.tensors[tensorId]!.cellId
      }

      let twReCellId = tensorCellId(node.inputs[1])
      let twImCellId = tensorCellId(node.inputs[2])
      let bitRevCellId = tensorCellId(node.inputs[3])

      let fftSize = windowSize * 2
      let numStages = Int(log2(Double(windowSize)))
      let imagOffset = windowSize
      let numBins = windowSize / 2 + 1
      let invNBins = b.constant(1.0 / Float(windowSize * numBins))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      let hopCounter: Expr? = inputs.count > 4 ? b.value(inputs[4]) : nil
      let frameIdx = b.frameIndex()

      // Per-frame base offsets (integer arithmetic for precision)
      let gradSpecBase = frameIdx * b.intConstant(fftSize)
      let gradTimeBase = frameIdx * b.intConstant(windowSize)
      let imagOff = b.intConstant(imagOffset)

      // Threadgroup scratch arrays for IFFT butterflies (reused for both signals)
      // Budget: scratchRe(N) + scratchIm(N) + twRe(N-1) + twIm(N-1) = 4N-2 floats.
      // N=2048 → 32760 bytes < 32KB limit. Bitrev/window stay in device (O(N) reads, negligible).
      let scratchRe = b.threadgroupScratch(windowSize)
      let scratchIm = b.threadgroupScratch(windowSize)
      // Twiddle factors loaded from device once, then read from scratch during butterflies
      let scratchTwRe = b.threadgroupScratch(windowSize - 1)
      let scratchTwIm = b.threadgroupScratch(windowSize - 1)

      let shouldRun = hopCounter.map { $0 == zero } ?? (one > zero)
      b.if_(shouldRun) {
        // Load twiddle tables from device → scratch (once for both signals)
        let twiddleSize = windowSize - 1
        b.loop(twiddleSize) { n in
          let nInt = b.cast(n, to: .int)
          b.scratchWrite(scratchTwRe, nInt, b.memoryRead(twReCellId, nInt))
          b.scratchWrite(scratchTwIm, nInt, b.memoryRead(twImCellId, nInt))
        }

        // Helper: IFFT one gradient spectrum in scratch, scale+window → device
        func emitIFFTInScratch(_ gradSpecCell: CellID, _ gradTimeCell: CellID) {
          // 1. Load gradient spectrum from device into scratch
          b.loop(windowSize) { n in
            let nInt = b.cast(n, to: .int)
            let re = b.memoryRead(gradSpecCell, gradSpecBase + nInt)
            let im = b.memoryRead(gradSpecCell, gradSpecBase + nInt + imagOff)
            b.scratchWrite(scratchRe, nInt, re)
            b.scratchWrite(scratchIm, nInt, im)
          }

          // 2. Bit-reversal permutation in scratch (bitrev indices from device — O(N), negligible)
          b.loop(windowSize) { i in
            let iInt = b.cast(i, to: .int)
            let rev = b.memoryRead(bitRevCellId, iInt)
            let iFloat = b.cast(i, to: .float)
            let shouldSwap = iFloat < rev
            let revInt = b.cast(rev, to: .int)

            let tempR = b.scratchRead(scratchRe, iInt)
            let tempI = b.scratchRead(scratchIm, iInt)
            let revR = b.scratchRead(scratchRe, revInt)
            let revI = b.scratchRead(scratchIm, revInt)

            let newIR = b.gswitch(shouldSwap, revR, tempR)
            let newII = b.gswitch(shouldSwap, revI, tempI)
            let newRevR = b.gswitch(shouldSwap, tempR, revR)
            let newRevI = b.gswitch(shouldSwap, tempI, revI)

            b.scratchWrite(scratchRe, iInt, newIR)
            b.scratchWrite(scratchIm, iInt, newII)
            b.scratchWrite(scratchRe, revInt, newRevR)
            b.scratchWrite(scratchIm, revInt, newRevI)
          }

          // 3. Butterfly stages entirely in scratch with POSITIVE twiddle (IFFT)
          var butterflySize = 2
          for _ in 0..<numStages {
            let halfSize = butterflySize / 2
            let numGroups = windowSize / butterflySize
            let numButterflies = numGroups * halfSize

            b.loop(numButterflies) { flatIdx in
              let flatFloat = b.cast(flatIdx, to: .float)
              let halfSizeFloat = b.constant(Float(halfSize))
              let butterflySizeFloat = b.constant(Float(butterflySize))

              let group = b.floor(flatFloat / halfSizeFloat)
              let k = flatFloat - (group * halfSizeFloat)

              let i = group * butterflySizeFloat + k
              let j = i + halfSizeFloat

              let twiddleOffset = b.intConstant(halfSize - 1)
              let kInt = b.cast(k, to: .int)
              let twiddleIdx = twiddleOffset + kInt
              let wr = b.scratchRead(scratchTwRe, twiddleIdx)
              let wi = b.scratchRead(scratchTwIm, twiddleIdx)  // positive for IFFT

              let iInt = b.cast(i, to: .int)
              let jInt = b.cast(j, to: .int)

              let ar = b.scratchRead(scratchRe, iInt)
              let ai = b.scratchRead(scratchIm, iInt)
              let br = b.scratchRead(scratchRe, jInt)
              let bi = b.scratchRead(scratchIm, jInt)

              let tr = wr * br - wi * bi
              let ti = wr * bi + wi * br

              b.scratchWrite(scratchRe, iInt, ar + tr)
              b.scratchWrite(scratchIm, iInt, ai + ti)
              b.scratchWrite(scratchRe, jInt, ar - tr)
              b.scratchWrite(scratchIm, jInt, ai - ti)
            }
            butterflySize *= 2
          }

          // 4. Scale by 1/(N*numBins), multiply by window → write to device
          b.loop(windowSize) { n in
            let nInt = b.cast(n, to: .int)
            let realVal = b.scratchRead(scratchRe, nInt) * invNBins
            let w = b.memoryRead(windowCell, nInt)
            _ = b.memoryWrite(gradTimeCell, gradTimeBase + nInt, realVal * w)
          }
        }

        // Apply IFFT to both gradient cells (scratch reused sequentially)
        emitIFFTInScratch(gradSpec1Cell, gradTime1Cell)
        emitIFFTInScratch(gradSpec2Cell, gradTime2Cell)

      }  // end b.if_(shouldRun)

      // On non-hop frames, clear this frame's time-domain gradient slice so
      // grad-read kernels don't see stale values from previous runs.
      if let hopCounter {
        b.if_(hopCounter > zero) {
          b.loop(windowSize) { n in
            let nInt = b.cast(n, to: .int)
            _ = b.memoryWrite(gradTime1Cell, gradTimeBase + nInt, zero)
            _ = b.memoryWrite(gradTime2Cell, gradTimeBase + nInt, zero)
          }
        }
      }

      b.use(val: zero)  // Side-effect only

    case .spectralLossFFTGradInline(
      let windowSize, let useHann, _,
      let gradTime1Cell, let gradTime2Cell):
      // Inline gradient computation that recomputes DFT to avoid race conditions
      // Uses frame-indexed storage to prevent race conditions between parallel frames
      // Inputs: [gradOutput, sig1, sig2]
      guard inputs.count == 3 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradInline", expected: 3, actual: inputs.count)
      }

      let numBins = windowSize / 2 + 1
      let gradOutput = b.value(inputs[0])
      let sig1 = b.value(inputs[1])
      let sig2 = b.value(inputs[2])
      let winSizeFloat = b.constant(Float(windowSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      let eps = b.constant(1e-8)
      let frameIdx = b.frameIndex()

      // Frame-indexed base offset for this frame's gradient storage (integer arithmetic)
      let frameBase = frameIdx * b.intConstant(windowSize)

      // Helper to compute Hann coefficient inline (avoids shared memory race)
      func hannCoeff(_ nFloat: Expr) -> Expr {
        if useHann {
          let angle = b.constant(2.0) * b.pi * nFloat / b.constant(Float(windowSize - 1))
          return b.constant(0.5) * (one - b.cos(angle))
        } else {
          return one
        }
      }

      // Use if_ to force scalar mode
      let alwaysTrue = one > zero
      b.if_(alwaysTrue) {
        // 1. Zero the gradient cells at frame-indexed positions
        b.loop(windowSize) { n in
          let idx = frameBase + b.cast(n, to: .int)
          _ = b.memoryWrite(gradTime1Cell, idx, zero)
          _ = b.memoryWrite(gradTime2Cell, idx, zero)
        }

        // 2. For each bin, recompute DFT and accumulate gradients to time domain
        // This is the key: we compute the DFT inline using tapeLoad, avoiding shared cells
        b.loop(numBins) { kIdx in
          let k = b.cast(kIdx, to: .float)

          // Recompute DFT for this bin
          let real1 = b.float(0.0)
          let imag1 = b.float(0.0)
          let real2 = b.float(0.0)
          let imag2 = b.float(0.0)

          b.loop(windowSize) { n in
            let nFloat = b.cast(n, to: .float)
            let j = b.cast(frameIdx, to: .float) - (winSizeFloat - one) + nFloat
            let w = hannCoeff(nFloat)  // Compute Hann inline

            let s1 = b.tapeLoad(sig1, at: j) * w
            let s2 = b.tapeLoad(sig2, at: j) * w

            // DFT basis: e^(-2πi*k*n/N)
            let angle = b.constant(-2.0) * b.pi * k * nFloat / winSizeFloat
            let c = b.cos(angle)
            let s = b.sin(angle)

            real1.accumulate(s1 * c)
            imag1.accumulate(s1 * s)
            real2.accumulate(s2 * c)
            imag2.accumulate(s2 * s)
          }

          // Compute magnitudes
          let mag1 = b.sqrt(real1.value * real1.value + imag1.value * imag1.value)
          let mag2 = b.sqrt(real2.value * real2.value + imag2.value * imag2.value)

          // Gradient: ∂L/∂mag = 2 * (mag1 - mag2) * gradOutput
          let gradMag1 = b.constant(2.0) * (mag1 - mag2) * gradOutput
          let gradMag2 = b.constant(-2.0) * (mag1 - mag2) * gradOutput

          // Handle division by zero
          let safeMag1 = b.max(mag1, eps)
          let safeMag2 = b.max(mag2, eps)

          // ∂L/∂X = gradMag * (real/mag, imag/mag)
          let gradReal1 = gradMag1 * real1.value / safeMag1
          let gradImag1 = gradMag1 * imag1.value / safeMag1
          let gradReal2 = gradMag2 * real2.value / safeMag2
          let gradImag2 = gradMag2 * imag2.value / safeMag2

          // Scatter gradient to time domain: ∂L/∂x[n] += gradReal * cos + gradImag * sin
          // This is the transpose of the DFT with normalization
          // Uses frame-indexed storage to avoid race conditions
          //
          // Normalization: 1/(windowSize * numBins) - less aggressive to allow learning
          let normFactor = b.constant(1.0 / Float(windowSize * numBins))

          b.loop(windowSize) { n in
            let nFloat = b.cast(n, to: .float)
            let angle = b.constant(-2.0) * b.pi * k * nFloat / winSizeFloat
            let c = b.cos(angle)
            let s = b.sin(angle)
            let w = hannCoeff(nFloat)  // Compute Hann inline

            // Frame-indexed position (integer arithmetic)
            let idx = frameBase + b.cast(n, to: .int)

            // Accumulate gradient (window backprop included, with normalization)
            let grad1Contrib = (gradReal1 * c + gradImag1 * s) * w * normFactor
            let grad2Contrib = (gradReal2 * c + gradImag2 * s) * w * normFactor
            _ = b.memoryAccumulate(gradTime1Cell, idx, grad1Contrib)
            _ = b.memoryAccumulate(gradTime2Cell, idx, grad2Contrib)
          }
        }
      }

      b.use(val: zero)  // Side-effect only

    case .spectralLossFFTGradRead(let windowSize, let gradTime1Cell, _):
      // Read gradient for signal 1 from frame-indexed storage
      // Sample at position p appears in windows at frames p, p+1, ..., p+windowSize-1
      // We must sum contributions from all these windows (but only if the window exists)
      //
      // NOTE: GradRead must NOT be hop-gated. Every sample contributes to the loss
      // through nearby hop-aligned windows. The gradTime cells already have zeros on
      // non-hop frames (GradIFFT handles this), so the summation naturally includes
      // only valid contributions without explicit gating.
      guard inputs.count >= 1 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradRead", expected: 1, actual: inputs.count)
      }
      // Force dependency on gradPass by reading its value (should be 0, just for ordering)
      let _ = b.value(inputs[0])
      // Read hop counter to maintain dependency, but don't gate on it
      if inputs.count > 1 { let _ = b.value(inputs[1]) }

      let frameIdx = b.frameIndex()
      let winSizeInt = b.intConstant(windowSize)
      let winSizeFloat = b.constant(Float(windowSize))
      let frameCount = b.frameCount()
      let p = frameIdx  // absolute sample position (int)

      let gradSum = b.float(0.0)
      // Sum contributions from all windows that contain sample p
      b.loop(windowSize) { i in
        let iInt = b.cast(i, to: .int)
        let iFloat = b.cast(i, to: .float)
        let pFloat = b.cast(p, to: .float)
        let w = pFloat + iFloat  // window frame index (float for comparisons)
        let offsetInt = winSizeInt - b.intConstant(1) - iInt  // offset in that window (int)
        // Clamp index to valid range to prevent out-of-bounds read
        let clampedW = b.min(w, b.cast(frameCount, to: .float) - b.constant(1.0))
        let idx = b.cast(clampedW, to: .int) * winSizeInt + offsetInt
        // Read is now safe, but only accumulate if in bounds
        let contrib = b.memoryRead(gradTime1Cell, idx)
        let inBounds = w < b.cast(frameCount, to: .float)
        let safeContrib = b.gswitch(inBounds, contrib, b.constant(0.0))
        gradSum.accumulate(safeContrib)
      }
      // Normalize to prevent gradient explosion while allowing learning
      let numBinsFloat = b.constant(Float(windowSize / 2 + 1))
      let normFactor = b.sqrt(numBinsFloat * winSizeFloat)
      let normalizedGrad = gradSum.value / normFactor
      b.use(val: normalizedGrad)

    case .spectralLossFFTGradRead2(let windowSize, let gradTime2Cell):
      // Read gradient for signal 2 from frame-indexed storage
      // Same as GradRead: must NOT be hop-gated (see comment above)
      guard inputs.count >= 1 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradRead2", expected: 1, actual: inputs.count)
      }
      let _ = b.value(inputs[0])
      // Read hop counter to maintain dependency, but don't gate on it
      if inputs.count > 1 { let _ = b.value(inputs[1]) }

      let frameIdx = b.frameIndex()
      let winSizeInt = b.intConstant(windowSize)
      let winSizeFloat = b.constant(Float(windowSize))
      let frameCount = b.frameCount()
      let p = frameIdx  // absolute sample position (int)

      let gradSum = b.float(0.0)
      b.loop(windowSize) { i in
        let iInt = b.cast(i, to: .int)
        let iFloat = b.cast(i, to: .float)
        let pFloat = b.cast(p, to: .float)
        let w = pFloat + iFloat  // window frame index (float for comparisons)
        let offsetInt = winSizeInt - b.intConstant(1) - iInt  // offset in that window (int)
        let clampedW = b.min(w, b.cast(frameCount, to: .float) - b.constant(1.0))
        let idx = b.cast(clampedW, to: .int) * winSizeInt + offsetInt
        let contrib = b.memoryRead(gradTime2Cell, idx)
        let inBounds = w < b.cast(frameCount, to: .float)
        let safeContrib = b.gswitch(inBounds, contrib, b.constant(0.0))
        gradSum.accumulate(safeContrib)
      }
      let numBinsFloat2 = b.constant(Float(windowSize / 2 + 1))
      let normFactor2 = b.sqrt(numBinsFloat2 * winSizeFloat)
      let normalizedGrad2 = gradSum.value / normFactor2
      b.use(val: normalizedGrad2)

    default: break
    }
  }
}

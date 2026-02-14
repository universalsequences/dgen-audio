import Foundation

extension LazyOp {
  func emitSpectralLoss(b: IRBuilder, ctx: IRContext, g: Graph, node: Node, inputs: [Lazy]) throws {
    switch self {
    case .spectralLossFFT(
      let windowSize, let useHann, let windowCell,
      let fft1Cell, let fft2Cell, let mag1Cell, let mag2Cell, let scratchCell):
      // FFT-based spectral loss: forward pass (SIMD-parallel across frames)
      // Each frame gets its own scratch space to avoid race conditions
      // 1. Apply optional Hann window (shared across frames)
      // 2. Compute FFT of both signals via Cooley-Tukey (per-frame scratch)
      // 3. Compute magnitudes (per-frame storage)
      // 4. Sum squared differences as loss
      guard inputs.count >= 2 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFT", expected: 2, actual: inputs.count)
      }

      let numBins = windowSize / 2 + 1
      let numStages = Int(log2(Double(windowSize)))
      let fftSize = windowSize * 2  // real + imag
      let imagOffset = windowSize  // Scratch layout: real[0..<N], imag[N..<2N]

      let sig1 = b.value(inputs[0])
      let sig2 = b.value(inputs[1])
      let hopCounter: Expr? = inputs.count > 2 ? b.value(inputs[2]) : nil
      let winSizeFloat = b.constant(Float(windowSize))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      // Use logical frame index (respects setFrameIndex in scaled-thread kernels).
      let frameIdx = b.frameIndex()

      // Per-frame base offsets for thread-local scratch memory (integer to avoid float32 precision loss)
      let fftBaseOffset = frameIdx * b.intConstant(fftSize)
      let magBaseOffset = frameIdx * b.intConstant(numBins)

      // Helper to compute Hann coefficient inline (avoids shared memory race)
      func hannCoeff(_ nFloat: Expr) -> Expr {
        if useHann {
          let angle = b.constant(2.0) * b.pi * nFloat / b.constant(Float(windowSize - 1))
          return b.constant(0.5) * (one - b.cos(angle))
        } else {
          return one
        }
      }

      // Optional hop gating: when a hop counter input is present, execute only on counter==0 frames.
      // Keep computation in a single if-block to preserve scalar-mode lowering in nested loops.
      let shouldRun = hopCounter.map { $0 == zero } ?? (one > zero)
      let loss = b.float(0.0)
      b.if_(shouldRun) {
        // 1. Load windowed samples into per-frame FFT scratch cells
        // Compute Hann coefficient inline to avoid shared memory race
        b.parallelRange(windowSize) { n in
          let nInt = n
          let nFloat = b.cast(nInt, to: .float)
          let w = hannCoeff(nFloat)

          // Load from tape: samples at position frameIdx - windowSize + 1 + n
          let j = b.cast(frameIdx, to: .float) - (winSizeFloat - one) + nFloat

          let s1 = b.tapeLoad(sig1, at: j)
          let s2 = b.tapeLoad(sig2, at: j)

          // Store in per-frame scratch: baseOffset + index (all int arithmetic)
          let imagOff = b.intConstant(imagOffset)
          _ = b.memoryWrite(fft1Cell, fftBaseOffset + nInt, s1 * w)
          _ = b.memoryWrite(fft1Cell, fftBaseOffset + nInt + imagOff, zero)
          _ = b.memoryWrite(fft2Cell, fftBaseOffset + nInt, s2 * w)
          _ = b.memoryWrite(fft2Cell, fftBaseOffset + nInt + imagOff, zero)

          // Store window coefficients for backward pass (GradIFFT reads these)
          // All frames write the same values — benign race
          _ = b.memoryWrite(windowCell, nInt, w)
        }

        // 3. In-place FFT via Cooley-Tukey (per-frame scratch)
        func emitFFTInPlace(_ fftCell: CellID, _ baseOffset: Expr) {
          let imagOff = b.intConstant(imagOffset)

          // Bit-reversal permutation
          b.loop(windowSize) { i in
            var rev = b.constant(0.0)
            var bits = b.cast(i, to: .float)
            for _ in 0..<numStages {
              rev = rev * b.constant(2.0) + (bits % b.constant(2.0))
              bits = b.floor(bits / b.constant(2.0))
            }

            let iFloat = b.cast(i, to: .float)
            let shouldSwap = iFloat < rev
            let iInt = b.cast(i, to: .int)
            let revInt = b.cast(rev, to: .int)

            let tempRealI = b.memoryRead(fftCell, baseOffset + iInt)
            let tempImagI = b.memoryRead(fftCell, baseOffset + iInt + imagOff)
            let tempRealRev = b.memoryRead(fftCell, baseOffset + revInt)
            let tempImagRev = b.memoryRead(fftCell, baseOffset + revInt + imagOff)

            let newRealI = b.gswitch(shouldSwap, tempRealRev, tempRealI)
            let newImagI = b.gswitch(shouldSwap, tempImagRev, tempImagI)
            let newRealRev = b.gswitch(shouldSwap, tempRealI, tempRealRev)
            let newImagRev = b.gswitch(shouldSwap, tempImagI, tempImagRev)

            _ = b.memoryWrite(fftCell, baseOffset + iInt, newRealI)
            _ = b.memoryWrite(fftCell, baseOffset + iInt + imagOff, newImagI)
            _ = b.memoryWrite(fftCell, baseOffset + revInt, newRealRev)
            _ = b.memoryWrite(fftCell, baseOffset + revInt + imagOff, newImagRev)
          }

          // Butterfly stages
          for stage in 0..<numStages {
            let butterflySize = 1 << (stage + 1)
            let halfSize = butterflySize / 2
            let numGroups = windowSize / butterflySize
            let numButterflies = numGroups * halfSize

            b.parallelRange(numButterflies) { flatIdx in
              let flatFloat = b.cast(flatIdx, to: .float)
              let halfSizeFloat = b.constant(Float(halfSize))
              let butterflySizeFloat = b.constant(Float(butterflySize))

              let group = b.floor(flatFloat / halfSizeFloat)
              let k = flatFloat - (group * halfSizeFloat)

              let i = group * butterflySizeFloat + k
              let j = i + halfSizeFloat

              let angle = b.constant(-2.0) * b.pi * k / butterflySizeFloat
              let wr = b.cos(angle)
              let wi = b.sin(angle)

              let iInt = b.cast(i, to: .int)
              let jInt = b.cast(j, to: .int)

              let ar = b.memoryRead(fftCell, baseOffset + iInt)
              let ai = b.memoryRead(fftCell, baseOffset + iInt + imagOff)
              let br = b.memoryRead(fftCell, baseOffset + jInt)
              let bi = b.memoryRead(fftCell, baseOffset + jInt + imagOff)

              let tr = wr * br - wi * bi
              let ti = wr * bi + wi * br

              _ = b.memoryWrite(fftCell, baseOffset + iInt, ar + tr)
              _ = b.memoryWrite(fftCell, baseOffset + iInt + imagOff, ai + ti)
              _ = b.memoryWrite(fftCell, baseOffset + jInt, ar - tr)
              _ = b.memoryWrite(fftCell, baseOffset + jInt + imagOff, ai - ti)
            }
          }
        }

        // Apply FFT to both cells with per-frame offsets
        emitFFTInPlace(fft1Cell, fftBaseOffset)
        emitFFTInPlace(fft2Cell, fftBaseOffset)

        // 4. Compute magnitudes and store in per-frame storage
        b.parallelRange(numBins) { k in
          let kInt = b.cast(k, to: .int)
          let imagOff = b.intConstant(imagOffset)

          let real1 = b.memoryRead(fft1Cell, fftBaseOffset + kInt)
          let imag1 = b.memoryRead(fft1Cell, fftBaseOffset + kInt + imagOff)
          let mag1 = b.sqrt(real1 * real1 + imag1 * imag1)
          _ = b.memoryWrite(mag1Cell, magBaseOffset + kInt, mag1)

          let real2 = b.memoryRead(fft2Cell, fftBaseOffset + kInt)
          let imag2 = b.memoryRead(fft2Cell, fftBaseOffset + kInt + imagOff)
          let mag2 = b.sqrt(real2 * real2 + imag2 * imag2)
          _ = b.memoryWrite(mag2Cell, magBaseOffset + kInt, mag2)

          // Store squared difference in per-frame scratch
          let diff = mag1 - mag2
          _ = b.memoryWrite(scratchCell, magBaseOffset + kInt, diff * diff)
        }

        // 5. Sequential reduction of loss
        b.loop(numBins) { k in
          let kInt = b.cast(k, to: .int)
          let diffSq = b.memoryRead(scratchCell, magBaseOffset + kInt)
          loss.accumulate(diffSq)
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
      // Then multiply by window coefficients for Hann backprop
      guard inputs.count >= 1 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradIFFT", expected: 1, actual: inputs.count)
      }

      let fftSize = windowSize * 2
      let numStages = Int(log2(Double(windowSize)))
      let imagOffset = windowSize
      // Scale by 1/(N * numBins) to match GradInline normalization
      let numBins = windowSize / 2 + 1
      let invNBins = b.constant(1.0 / Float(windowSize * numBins))
      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      let hopCounter: Expr? = inputs.count > 1 ? b.value(inputs[1]) : nil
      let frameIdx = b.frameIndex()

      // Per-frame base offsets (integer arithmetic for precision)
      let gradSpecBase = frameIdx * b.intConstant(fftSize)
      let gradTimeBase = frameIdx * b.intConstant(windowSize)
      let imagOff = b.intConstant(imagOffset)

      // Force scalar mode (same pattern as forward FFT), with optional hop gating.
      let shouldRun = hopCounter.map { $0 == zero } ?? (one > zero)
      b.if_(shouldRun) {

        // Helper function to emit IFFT for a single gradient spectrum cell
        func emitIFFTInPlace(_ gradSpecCell: CellID, _ gradTimeCell: CellID) {
          // Bit-reversal permutation (has data dependencies - keep sequential)
          b.loop(windowSize) { i in
            var rev = b.constant(0.0)
            var n = b.cast(i, to: .float)
            for _ in 0..<numStages {
              rev = rev * b.constant(2.0) + (n % b.constant(2.0))
              n = b.floor(n / b.constant(2.0))
            }

            let iFloat = b.cast(i, to: .float)
            let shouldSwap = iFloat < rev
            let iInt = b.cast(i, to: .int)
            let revInt = b.cast(rev, to: .int)

            let tempR = b.memoryRead(gradSpecCell, gradSpecBase + iInt)
            let tempI = b.memoryRead(gradSpecCell, gradSpecBase + iInt + imagOff)
            let revR = b.memoryRead(gradSpecCell, gradSpecBase + revInt)
            let revI = b.memoryRead(gradSpecCell, gradSpecBase + revInt + imagOff)

            let newIR = b.gswitch(shouldSwap, revR, tempR)
            let newII = b.gswitch(shouldSwap, revI, tempI)
            let newRevR = b.gswitch(shouldSwap, tempR, revR)
            let newRevI = b.gswitch(shouldSwap, tempI, revI)

            _ = b.memoryWrite(gradSpecCell, gradSpecBase + iInt, newIR)
            _ = b.memoryWrite(gradSpecCell, gradSpecBase + iInt + imagOff, newII)
            _ = b.memoryWrite(gradSpecCell, gradSpecBase + revInt, newRevR)
            _ = b.memoryWrite(gradSpecCell, gradSpecBase + revInt + imagOff, newRevI)
          }

          // Butterfly stages with POSITIVE twiddle angles (IFFT)
          var butterflySize = 2
          for _ in 0..<numStages {
            let halfSize = butterflySize / 2
            let numGroups = windowSize / butterflySize
            let numButterflies = numGroups * halfSize

            b.parallelRange(numButterflies) { flatIdx in
              let flatFloat = b.cast(flatIdx, to: .float)
              let halfSizeFloat = b.constant(Float(halfSize))
              let butterflySizeFloat = b.constant(Float(butterflySize))

              let group = b.floor(flatFloat / halfSizeFloat)
              let k = flatFloat - (group * halfSizeFloat)

              let i = group * butterflySizeFloat + k
              let j = i + halfSizeFloat

              // IFFT twiddle: W = e^(+2πi*k/butterflySize) - POSITIVE angle
              let angle = b.constant(2.0) * b.pi * k / butterflySizeFloat
              let wr = b.cos(angle)
              let wi = b.sin(angle)

              let iInt = b.cast(i, to: .int)
              let jInt = b.cast(j, to: .int)

              let ar = b.memoryRead(gradSpecCell, gradSpecBase + iInt)
              let ai = b.memoryRead(gradSpecCell, gradSpecBase + iInt + imagOff)
              let br = b.memoryRead(gradSpecCell, gradSpecBase + jInt)
              let bi = b.memoryRead(gradSpecCell, gradSpecBase + jInt + imagOff)

              // Complex multiply and butterfly
              let tr = wr * br - wi * bi
              let ti = wr * bi + wi * br

              _ = b.memoryWrite(gradSpecCell, gradSpecBase + iInt, ar + tr)
              _ = b.memoryWrite(gradSpecCell, gradSpecBase + iInt + imagOff, ai + ti)
              _ = b.memoryWrite(gradSpecCell, gradSpecBase + jInt, ar - tr)
              _ = b.memoryWrite(gradSpecCell, gradSpecBase + jInt + imagOff, ai - ti)
            }
            butterflySize *= 2
          }

          // Scale by 1/(N*numBins) and multiply by window (Hann backprop)
          b.parallelRange(windowSize) { n in
            let nInt = b.cast(n, to: .int)
            let realVal = b.memoryRead(gradSpecCell, gradSpecBase + nInt) * invNBins
            let w = b.memoryRead(windowCell, nInt)
            _ = b.memoryWrite(gradTimeCell, gradTimeBase + nInt, realVal * w)
          }
        }

        // Apply IFFT to both gradient cells
        emitIFFTInPlace(gradSpec1Cell, gradTime1Cell)
        emitIFFTInPlace(gradSpec2Cell, gradTime2Cell)

      }  // end b.if_(shouldRun)

      // On non-hop frames, clear this frame's time-domain gradient slice so
      // grad-read kernels don't see stale values from previous runs.
      if let hopCounter {
        b.if_(hopCounter > zero) {
          b.parallelRange(windowSize) { n in
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
      guard inputs.count >= 1 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradRead", expected: 1, actual: inputs.count)
      }
      // Force dependency on gradPass by reading its value (should be 0, just for ordering)
      let _ = b.value(inputs[0])
      let hopCounter: Expr? = inputs.count > 1 ? b.value(inputs[1]) : nil

      let frameIdx = b.frameIndex()
      let winSizeInt = b.intConstant(windowSize)
      let winSizeFloat = b.constant(Float(windowSize))
      let frameCount = b.frameCount()
      let p = frameIdx  // absolute sample position (int)

      let gradSum = b.float(0.0)
      let shouldRun = hopCounter.map { $0 == b.constant(0.0) } ?? (b.constant(1.0) > b.constant(0.0))
      b.if_(shouldRun) {
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
      }
      // Normalize to prevent gradient explosion while allowing learning
      let numBinsFloat = b.constant(Float(windowSize / 2 + 1))
      let normFactor = b.sqrt(numBinsFloat * winSizeFloat)
      let normalizedGrad = gradSum.value / normFactor
      b.use(val: normalizedGrad)

    case .spectralLossFFTGradRead2(let windowSize, let gradTime2Cell):
      // Read gradient for signal 2 from frame-indexed storage
      guard inputs.count >= 1 else {
        throw DGenError.insufficientInputs(
          operator: "spectralLossFFTGradRead2", expected: 1, actual: inputs.count)
      }
      let _ = b.value(inputs[0])
      let hopCounter: Expr? = inputs.count > 1 ? b.value(inputs[1]) : nil

      let frameIdx = b.frameIndex()
      let winSizeInt = b.intConstant(windowSize)
      let winSizeFloat = b.constant(Float(windowSize))
      let frameCount = b.frameCount()
      let p = frameIdx  // absolute sample position (int)

      let gradSum = b.float(0.0)
      let shouldRun2 = hopCounter.map { $0 == b.constant(0.0) } ?? (b.constant(1.0) > b.constant(0.0))
      b.if_(shouldRun2) {
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
      }
      let numBinsFloat2 = b.constant(Float(windowSize / 2 + 1))
      let normFactor2 = b.sqrt(numBinsFloat2 * winSizeFloat)
      let normalizedGrad2 = gradSum.value / normFactor2
      b.use(val: normalizedGrad2)

    default: break
    }
  }
}

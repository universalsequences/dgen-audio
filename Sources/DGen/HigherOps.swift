extension Graph {
  /// Smart phasor that chooses between deterministic (parallelizable) and stateful versions.
  /// If freq is constant and reset is constant 0, uses deterministicPhasor for parallel execution.
  /// Otherwise uses the stateful phasor with memory cell.
  public func phasor(freq: NodeID, reset: NodeID) -> NodeID {
    // Check if we can use the deterministic (parallelizable) version
    guard let freqNode = nodes[freq] else {
      let cellId = alloc()
      return n(.phasor(cellId), freq, reset)
    }
    guard let resetNode = nodes[reset] else {
      let cellId = alloc()
      return n(.phasor(cellId), freq, reset)
    }

    // If freq is constant and reset is constant 0, use deterministic phasor
    if case .constant(_) = freqNode.op,
      case .constant(let resetVal) = resetNode.op,
      resetVal == 0.0
    {
      // Use deterministic phasor - no memory needed, fully parallelizable
      return n(.deterministicPhasor, freq)
    }

    // Otherwise use stateful phasor
    let cellId = alloc()
    return n(.phasor(cellId), freq, reset)
  }

  /// Spectral loss: compute DFT-based magnitude MSE between two signals over a window
  /// Uses tape-based compute; no ring buffers are required.
  public func spectralLoss(_ sig1: NodeID, _ sig2: NodeID, windowSize: Int) -> NodeID {
    // Allocate scratch memory for two-pass backward gradient distribution
    // Size: maxFrameCount * windowSize * 2 (for sig1 and sig2 contributions)
    let scratchSize = maxFrameCount * windowSize * 2
    let scratchCell = alloc(vectorWidth: scratchSize)

    // Create Pass1: computes spectral loss and stores DFT contributions (for backward)
    let pass1 = n(.spectralLossPass1(windowSize, scratchCell), sig1, sig2)

    // Create Pass2: no-op in forward, reduces contributions to gradients in backward
    // Depends on pass1 to ensure correct kernel ordering
    let pass2 = n(.spectralLossPass2(windowSize, scratchCell), pass1)

    return pass2  // Return Pass2 result (forwards Pass1's loss value)
  }

  /// FFT-based spectral loss with backpropagation support.
  ///
  /// Computes spectral loss using Cooley-Tukey FFT and provides gradients via IFFT.
  /// The backward pass uses IFFT (the adjoint of FFT) to scatter frequency-domain
  /// gradients back to time domain - fully parallelizable via butterfly stages.
  ///
  /// Mathematical formulation:
  /// - Forward: Apply optional Hann window, compute FFT, compute magnitudes, sum squared differences
  /// - Backward: Chain rule through magnitude, scale by unit vector, IFFT to time domain, multiply by window
  ///
  /// - Parameters:
  ///   - sig1: First input signal
  ///   - sig2: Second input signal (reference/target)
  ///   - windowSize: FFT window size (must be power of 2)
  ///   - useHannWindow: Whether to apply Hann window before FFT (default: true)
  /// - Returns: Scalar loss value per frame
  public func spectralLossFFT(
    _ sig1: NodeID,
    _ sig2: NodeID,
    windowSize: Int,
    useHannWindow: Bool = true
  ) -> NodeID {
    precondition(windowSize > 0 && (windowSize & (windowSize - 1)) == 0,
                 "windowSize must be a power of 2")

    let numBins = windowSize / 2 + 1

    // Allocate window coefficients (shared between all frames - same Hann window)
    let windowCell = alloc(vectorWidth: windowSize)

    // Allocate FFT scratch cells for storing complex spectrum - PER FRAME
    // Layout per frame: real[0..<windowSize], imag[windowSize..<windowSize*2]
    // Total: (windowSize * 2) * maxFrameCount
    let fftSize = windowSize * 2
    let fft1Cell = alloc(vectorWidth: fftSize * maxFrameCount)
    let fft2Cell = alloc(vectorWidth: fftSize * maxFrameCount)

    // Allocate magnitude cells (only positive frequencies: numBins) - PER FRAME
    let mag1Cell = alloc(vectorWidth: numBins * maxFrameCount)
    let mag2Cell = alloc(vectorWidth: numBins * maxFrameCount)

    // Allocate scratch for per-frame squared differences
    let scratchCell = alloc(vectorWidth: numBins * maxFrameCount)

    // Create the spectralLossFFT node with all resources
    return n(.spectralLossFFT(
      windowSize: windowSize,
      useHann: useHannWindow,
      windowCell: windowCell,
      fft1Cell: fft1Cell,
      fft2Cell: fft2Cell,
      mag1Cell: mag1Cell,
      mag2Cell: mag2Cell,
      scratchCell: scratchCell
    ), [sig1, sig2])
  }

  /// Parallel map2D test: writes per-bin values using flattened (frame, bin) threads,
  /// then reduces to a scalar per frame. Used to validate parallelMap2D semantics.
  public func parallelMap2DTest(bins: Int) -> NodeID {
    let scratchSize = maxFrameCount * bins
    let scratchCell = alloc(vectorWidth: scratchSize)

    let pass1 = n(.parallelMap2DTestPass1(bins, scratchCell))
    let pass2 = n(.parallelMap2DTestPass2(bins, scratchCell), pass1)
    return pass2
  }

  /// Delta: returns the difference between current and previous input value
  /// delta(input) = input - history(input)
  public func delta(_ input: NodeID) -> NodeID {
    let cellId = alloc()
    let historyRead = n(.historyRead(cellId))
    _ = n(.historyWrite(cellId), input)
    return n(.sub, input, historyRead)
  }

  /// Change: returns the sign of the difference between current and previous input
  /// change(input) = sign(input - history(input))
  public func change(_ input: NodeID) -> NodeID {
    let deltaNode = delta(input)
    return n(.sign, deltaNode)
  }

  /// RampToTrig: detects discontinuities in a ramp signal and outputs triggers
  /// Detects when a ramp resets (wraps around) by analyzing the rate of change
  public func rampToTrig(_ ramp: NodeID) -> NodeID {
    let cellId = alloc()
    let historyRead = n(.historyRead(cellId))
    _ = n(.historyWrite(cellId), ramp)

    // Calculate abs(ramp - history) / (ramp + history)
    let diff = n(.sub, ramp, historyRead)
    let sum = n(.add, ramp, historyRead)
    let ratio = n(.div, diff, sum)
    let absRatio = n(.abs, ratio)

    // Check if ratio > 0.5 (indicates a wrap-around)
    let halfConst = n(.constant(0.5))
    let isLargeJump = n(.gt, absRatio, halfConst)

    // Get the change/sign of this boolean result and convert to positive triggers only
    let changeResult = change(isLargeJump)
    // Note: Using abs() here loses distinction between rising/falling edges,
    // but provides clean 0/1 triggers suitable for accumulation and most trigger uses
    return n(.abs, changeResult)
  }

  /// Scale: maps a value from one range to another with optional exponential curve
  /// Based on Max MSP gen~ scale function
  /// scale(value, min1, max1, min2, max2, exponent = 1.0)
  /// - value: input value to scale
  /// - min1: minimum of input range
  /// - max1: maximum of input range
  /// - min2: minimum of output range
  /// - max2: maximum of output range
  /// - exponent: exponential curve factor (1.0 = linear, >1.0 = exponential, <1.0 = logarithmic)
  public func scale(
    _ value: NodeID, _ min1: NodeID, _ max1: NodeID, _ min2: NodeID, _ max2: NodeID,
    _ exponent: NodeID? = nil
  ) -> NodeID {
    let exp = exponent ?? n(.constant(1.0))

    // Calculate range1 = max1 - min1
    let range1 = n(.sub, max1, min1)

    // Calculate range2 = max2 - min2
    let range2 = n(.sub, max2, min2)

    // Calculate normalized value = (value - min1) / range1
    // Handle division by zero by checking if range1 == 0
    let valueMinusMin1 = n(.sub, value, min1)
    let zero = n(.constant(0.0))
    let range1IsZero = n(.eq, range1, zero)
    let normVal = n(.gswitch, range1IsZero, zero, n(.div, valueMinusMin1, range1))

    // Apply exponential curve: pow(normVal, exponent)
    let curvedVal = n(.pow, normVal, exp)

    // Scale to output range: min2 + range2 * curvedVal
    let scaledRange = n(.mul, range2, curvedVal)
    return n(.add, min2, scaledRange)
  }

  /// Triangle: converts a ramp signal to a triangle wave
  /// Based on Max MSP gen~ triangle function
  /// triangle(ramp, duty = 0.5)
  /// - ramp: input ramp signal (0-1)
  /// - duty: duty cycle (0-1), determines the peak position (0.5 = symmetric triangle)
  /// When ramp < duty: output scales from 0 to 1
  /// When ramp >= duty: output scales from 1 to 0
  public func triangle(_ ramp: NodeID, _ duty: NodeID? = nil) -> NodeID {
    let dutyValue = duty ?? n(.constant(0.5))

    // Check if ramp < duty
    let isRising = n(.lt, ramp, dutyValue)

    // Rising portion: scale(ramp, 0, duty, 0, 1)
    let zero = n(.constant(0.0))
    let one = n(.constant(1.0))
    let risingOutput = scale(ramp, zero, dutyValue, zero, one)

    // Falling portion: scale(ramp, duty, 1, 1, 0)
    let fallingOutput = scale(ramp, dutyValue, one, one, zero)

    // Switch between rising and falling based on comparison
    return n(.gswitch, isRising, risingOutput, fallingOutput)
  }

  public func wrap(_ input: NodeID, _ min: NodeID, _ max: NodeID) -> NodeID {
    let range = n(.sub, max, min)
    let normalized = n(.mod, n(.sub, input, min), range)
    return n(
      .gswitch, n(.gte, normalized, n(.constant(0))), n(.add, normalized, min),
      n(.add, range, n(.add, normalized, min)))
  }

  public func clip(_ input: NodeID, _ min: NodeID, _ max: NodeID) -> NodeID {
    return n(.max, min, n(.max, input, max))
  }

  public func selector(_ cond: NodeID, _ args: NodeID...) -> NodeID {
    // Use the new first-class selector operation
    return n(.selector, [cond] + args)
  }

  /// Biquad filter with multiple filter types.
  ///
  /// - Parameters:
  ///   - in1: Input signal
  ///   - cutoff: Cutoff/center frequency in Hz
  ///   - resonance: Q factor / resonance (higher = narrower bandwidth)
  ///   - gain: Output gain (linear). For shelf filters, this controls the shelf boost/cut amount.
  ///   - mode: Filter type:
  ///     - 0: Lowpass - passes frequencies below cutoff
  ///     - 1: Highpass - passes frequencies above cutoff
  ///     - 2: Bandpass (constant skirt gain) - passes frequencies around cutoff
  ///     - 3: Bandpass (constant peak gain) - passes frequencies around cutoff
  ///     - 4: Allpass - passes all frequencies, only affects phase
  ///     - 5: Notch - rejects frequencies around cutoff
  ///     - 6: High shelf - boosts/cuts frequencies above cutoff (gain > 1 boosts, < 1 cuts)
  ///     - 7: Low shelf - boosts/cuts frequencies below cutoff (gain > 1 boosts, < 1 cuts)
  ///
  /// - Returns: Filtered output signal
  public func biquad(
    _ in1: NodeID, _ cutoff: NodeID, _ resonance: NodeID, _ gain: NodeID, _ mode: NodeID
  ) -> NodeID {
    // Allocate history cells
    let history0Cell = alloc()
    let history1Cell = alloc()
    let history2Cell = alloc()
    let history3Cell = alloc()

    // History reads
    let history0Read = n(.historyRead(history0Cell))
    let history1Read = n(.historyRead(history1Cell))
    let history2Read = n(.historyRead(history2Cell))
    let history3Read = n(.historyRead(history3Cell))

    // History writes and chaining
    _ = n(.historyWrite(history0Cell), history1Read)

    _ = n(.historyWrite(history2Cell), in1)

    _ = n(.historyWrite(history3Cell), history2Read)

    // Core filter computation following TypeScript exactly
    let param3 = mode
    let add4 = n(.add, param3, n(.constant(1)))
    let param5 = cutoff
    let abs6 = n(.abs, param5)
    let mult7 = n(.mul, abs6, n(.constant(0.00014247585730565955)))
    let cos8 = n(.cos, mult7)
    let mult9 = n(.mul, cos8, n(.constant(-1)))
    let add10 = n(.add, mult9, n(.constant(1)))
    let div11 = n(.div, add10, n(.constant(2)))
    let sub12 = n(.sub, mult9, n(.constant(1)))
    let div13 = n(.div, sub12, n(.constant(-2)))
    let sin14 = n(.sin, mult7)
    let mult15 = n(.mul, sin14, n(.constant(0.5)))
    let param16 = resonance
    let abs17 = n(.abs, param16)
    let div18 = n(.div, mult15, abs17)
    let mult19 = n(.mul, div18, n(.constant(-1)))
    let mult20 = n(.mul, div18, abs17)
    let mult21 = n(.mul, mult20, n(.constant(-1)))
    let add22 = n(.add, div18, n(.constant(1)))

    // Shelf filter coefficients (mode 6 = high shelf, mode 7 = low shelf)
    // A = sqrt(gain) for amplitude, using gain param directly as linear amplitude
    let param25 = gain
    let A = n(.sqrt, n(.abs, param25))
    let cosW0Pos = n(.mul, mult9, n(.constant(-1)))  // cos(w0) = -(-cos(w0))
    let sinW0 = sin14

    // alpha for shelf: sin(w0)/2 * sqrt((A + 1/A)*(1/S - 1) + 2)
    // Using S = resonance (slope), simplified: alpha = sin(w0)/(2*Q) where Q relates to slope
    let alphaShelf = n(.div, sinW0, n(.mul, n(.constant(2.0)), abs17))

    // Common shelf terms
    let Ap1 = n(.add, A, n(.constant(1)))  // A + 1
    let Am1 = n(.sub, A, n(.constant(1)))  // A - 1
    let Ap1_cosW0 = n(.mul, Ap1, cosW0Pos)  // (A+1)*cos(w0)
    let Am1_cosW0 = n(.mul, Am1, cosW0Pos)  // (A-1)*cos(w0)
    let twoSqrtA_alpha = n(.mul, n(.mul, n(.constant(2.0)), n(.sqrt, A)), alphaShelf)  // 2*sqrt(A)*alpha

    // High shelf (mode 6):
    // b0 =    A*( (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha )
    // b1 = -2*A*( (A-1) + (A+1)*cos(w0)                   )
    // b2 =    A*( (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha )
    // a0 =        (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha
    // a1 =    2*( (A-1) - (A+1)*cos(w0)                   )
    // a2 =        (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha
    let hs_b0 = n(.mul, A, n(.add, n(.add, Ap1, Am1_cosW0), twoSqrtA_alpha))
    let hs_b1 = n(.mul, n(.constant(-2.0)), n(.mul, A, n(.add, Am1, Ap1_cosW0)))
    let hs_b2 = n(.mul, A, n(.sub, n(.add, Ap1, Am1_cosW0), twoSqrtA_alpha))
    let hs_a0 = n(.add, n(.sub, Ap1, Am1_cosW0), twoSqrtA_alpha)

    // Low shelf (mode 7):
    // b0 =    A*( (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha )
    // b1 =  2*A*( (A-1) - (A+1)*cos(w0)                   )
    // b2 =    A*( (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha )
    // a0 =        (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha
    // a1 =   -2*( (A-1) + (A+1)*cos(w0)                   )
    // a2 =        (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha
    let ls_b0 = n(.mul, A, n(.add, n(.sub, Ap1, Am1_cosW0), twoSqrtA_alpha))
    let ls_b1 = n(.mul, n(.constant(2.0)), n(.mul, A, n(.sub, Am1, Ap1_cosW0)))
    let ls_b2 = n(.mul, A, n(.sub, n(.sub, Ap1, Am1_cosW0), twoSqrtA_alpha))
    let ls_a0 = n(.add, n(.add, Ap1, Am1_cosW0), twoSqrtA_alpha)

    // Normalize shelf coefficients by a0
    let hs_b0_norm = n(.div, hs_b0, hs_a0)
    let hs_b1_norm = n(.div, hs_b1, hs_a0)
    let hs_b2_norm = n(.div, hs_b2, hs_a0)
    let ls_b0_norm = n(.div, ls_b0, ls_a0)
    let ls_b1_norm = n(.div, ls_b1, ls_a0)
    let ls_b2_norm = n(.div, ls_b2, ls_a0)

    // a1, a2 normalized for feedback (modes 0-5 use different normalization)
    let hs_a1_norm = n(.div, n(.mul, n(.constant(2.0)), n(.sub, Am1, Ap1_cosW0)), hs_a0)
    let hs_a2_norm = n(.div, n(.sub, n(.sub, Ap1, Am1_cosW0), twoSqrtA_alpha), hs_a0)
    let ls_a1_norm = n(.div, n(.mul, n(.constant(-2.0)), n(.add, Am1, Ap1_cosW0)), ls_a0)
    let ls_a2_norm = n(.div, n(.sub, n(.add, Ap1, Am1_cosW0), twoSqrtA_alpha), ls_a0)

    let selector23 = selector(
      add4, div11, div13, mult19, mult21, n(.constant(1)), add22, hs_b2_norm, ls_b2_norm)
    let reciprical24 = n(.div, n(.constant(1)), add22)
    let mult26 = n(.mul, reciprical24, param25)
    // For modes 0-5, apply gain scaling; for modes 6-7, coefficients already include gain via A
    let b2_scaled = n(.mul, selector23, mult26)
    // Use selector to pick between scaled (modes 0-5) and shelf (modes 6-7)
    let isShelfMode = n(.gte, param3, n(.constant(6)))
    let b2 = n(.gswitch, isShelfMode, selector23, b2_scaled)
    let mult28 = n(.mul, history3Read, b2)
    let mult31 = n(.mul, mult9, n(.constant(2)))
    let selector32 = selector(
      add4, add10, sub12, n(.constant(0)), n(.constant(0)), mult31, mult31, hs_b1_norm, ls_b1_norm)
    let b1_scaled = n(.mul, selector32, mult26)
    let b1 = n(.gswitch, isShelfMode, selector32, b1_scaled)
    let mult34 = n(.mul, history2Read, b1)
    let not_sub35 = n(.sub, n(.constant(1)), div18)
    let selector36 = selector(
      add4, div11, div13, div18, mult20, n(.constant(1)), not_sub35, hs_b0_norm, ls_b0_norm)
    let b0_scaled = n(.mul, selector36, mult26)
    let b0 = n(.gswitch, isShelfMode, selector36, b0_scaled)
    let mult38 = n(.mul, in1, b0)

    // Final computation - following TypeScript s() pattern
    let add39 = n(.add, n(.add, mult28, mult34), mult38)

    // Feedback coefficients (a1, a2 normalized by a0)
    // For modes 0-5: use original coefficients
    // For modes 6-7: use shelf-specific a1, a2
    let history040 = history0Read
    let a2_orig = n(.mul, not_sub35, reciprical24)  // Original a2/a0
    let a1_orig = n(.mul, mult31, reciprical24)  // Original a1/a0

    // Select appropriate feedback coefficients based on mode
    let a2 = n(
      .gswitch, isShelfMode,
      n(.gswitch, n(.gte, param3, n(.constant(7))), ls_a2_norm, hs_a2_norm),
      a2_orig)
    let a1 = n(
      .gswitch, isShelfMode,
      n(.gswitch, n(.gte, param3, n(.constant(7))), ls_a1_norm, hs_a1_norm),
      a1_orig)

    let mult42 = n(.mul, history040, a2)
    let mult44 = n(.mul, history1Read, a1)
    let add45 = n(.add, mult42, mult44)
    let sub46 = n(.sub, add39, add45)

    // Write final result to history1
    _ = n(.historyWrite(history1Cell), sub46)

    return sub46
  }

  private func amp2db(_ amp: NodeID) -> NodeID {
    let absAmp = n(.abs, amp)
    let maxAmp = n(.max, absAmp, n(.constant(0.00001)))
    let log10Val = n(.log10, maxAmp)
    return n(.mul, n(.constant(20.0)), log10Val)
  }

  private func db2amp(_ db: NodeID) -> NodeID {
    let divBy20 = n(.div, db, n(.constant(20.0)))
    return n(.pow, n(.constant(10.0)), divBy20)
  }

  public func compressor(
    _ in1: NodeID, _ ratio: NodeID, _ threshold: NodeID, _ knee: NodeID, _ attack: NodeID,
    _ release: NodeID, _ isSideChain: NodeID, _ sidechainIn: NodeID
  ) -> NodeID {

    // Detect level from either main or sidechain input
    let detectorDb = amp2db(n(.gswitch, isSideChain, sidechainIn, in1))

    // But always process the main input
    let inDb = amp2db(in1)

    // Attack and release coefficient calculations
    let log001 = n(.log, n(.constant(0.01)))
    let sampleRate = n(.constant(self.sampleRate))

    let attackSamples = n(.mul, attack, sampleRate)
    let attackCoef = n(.exp, n(.div, log001, attackSamples))

    let releaseSamples = n(.mul, release, sampleRate)
    let releaseCoef = n(.exp, n(.div, log001, releaseSamples))

    // Knee calculations
    let kneeHalf = n(.div, knee, n(.constant(2.0)))
    let kneeStart = n(.sub, threshold, kneeHalf)
    let kneeEnd = n(.add, threshold, kneeHalf)

    // Position within knee and interpolated ratio
    let positionWithinKnee = n(.div, n(.sub, detectorDb, kneeStart), knee)
    let ratioMinus1 = n(.sub, ratio, n(.constant(1.0)))
    let interpolatedRatio = n(.add, n(.constant(1.0)), n(.mul, positionWithinKnee, ratioMinus1))

    let thresholdMinusKneeStart = n(.sub, threshold, kneeStart)
    let overThreshold = n(
      .sub, detectorDb,
      n(.add, kneeStart, n(.mul, positionWithinKnee, thresholdMinusKneeStart)))

    // Gain reduction calculation using nested gswitch (zswitch equivalent)
    let belowKneeStart = n(.lte, detectorDb, kneeStart)
    let aboveKneeEnd = n(.gte, detectorDb, kneeEnd)

    let hardRatio = n(.sub, n(.constant(1.0)), n(.div, n(.constant(1.0)), ratio))
    let softRatio = n(.sub, n(.constant(1.0)), n(.div, n(.constant(1.0)), interpolatedRatio))

    let hardCompression = n(.mul, n(.sub, detectorDb, threshold), hardRatio)
    let softCompression = n(.mul, overThreshold, softRatio)

    let gr = n(
      .gswitch, belowKneeStart, n(.constant(0.0)),
      n(.gswitch, aboveKneeEnd, hardCompression, softCompression))

    // History cell for previous gain reduction (equivalent to history() in TypeScript)
    let grHistoryCell = alloc()
    let prevGr = n(.historyRead(grHistoryCell))

    // Smooth the gain reduction (attack/release)
    let grIsIncreasing = n(.gt, gr, prevGr)
    let attackSmooth = n(
      .add, n(.mul, attackCoef, prevGr), n(.mul, gr, n(.sub, n(.constant(1.0)), attackCoef)))
    let releaseSmooth = n(
      .add, n(.mul, releaseCoef, prevGr), n(.mul, gr, n(.sub, n(.constant(1.0)), releaseCoef))
    )

    let smoothedGr = n(.gswitch, grIsIncreasing, attackSmooth, releaseSmooth)

    // Write smoothed gain reduction to history
    _ = n(.historyWrite(grHistoryCell), smoothedGr)

    // Apply gain reduction to main input
    let signIn1 = n(.sign, in1)
    let attenuatedDb = n(.sub, inDb, smoothedGr)
    let attenuatedAmp = db2amp(attenuatedDb)

    return n(.mul, signIn1, attenuatedAmp)
  }

  /// Delay: delays the input signal by a variable amount of time with linear interpolation
  /// delay(input, delayTimeInSamples)
  /// - input: input signal to delay
  /// - delayTimeInSamples: delay time in samples (0 to MAX_DELAY)
  /// Implements a circular buffer with linear interpolation for fractional delay times
  public func delay(_ input: NodeID, _ delayTimeInSamples: NodeID) -> NodeID {
    let MAX_DELAY = 88000
    let bufferBase = alloc(vectorWidth: MAX_DELAY)
    let writePosCellId = alloc()

    // Constants
    let one = n(.constant(1.0))
    let zero = n(.constant(0.0))
    let maxDelay = n(.constant(Float(MAX_DELAY)))

    // Write head: wraps to [0, MAX_DELAY)
    // (your original accum+floor style is fine as long as it stays bounded)
    let writePos = n(.floor, n(.accum(writePosCellId), one, zero, zero, maxDelay))

    // Write first (important for feedback)
    let writeOp = n(.memoryWrite(bufferBase), writePos, input)

    // ---- Robust wrap helper (mod) in-node form ----
    // wrap(x, L) = x - floor(x / L) * L  -> [0, L) for any finite x
    func wrap0ToL(_ x: NodeID, _ L: NodeID) -> NodeID {
      let q = n(.div, x, L)
      let qFloor = n(.floor, q)
      return n(.sub, x, n(.mul, qFloor, L))
    }

    // Read position in samples (float)
    let rawReadPos = n(.sub, writePos, delayTimeInSamples)

    // True modulo wrap (robust vs huge modulation)
    let readPos = wrap0ToL(rawReadPos, maxDelay)

    // Linear interpolation
    let i0 = n(.floor, readPos)
    let frac = n(.sub, readPos, i0)

    // i1 = (i0 + 1) wrapped
    let i1raw = n(.add, i0, one)
    let i1 = wrap0ToL(i1raw, maxDelay)

    let s0 = n(.memoryRead(bufferBase), i0)
    let s1 = n(.memoryRead(bufferBase), i1)

    // (1-frac)*s0 + frac*s1
    let out = n(.mix, s0, s1, frac)

    // Ensure ordering: write then read path
    return n(.seq, writeOp, out)
  }

  /// Compute FFT of a signal using Cooley-Tukey algorithm.
  /// - Parameters:
  ///   - signal: Input signal (scalar per frame)
  ///   - windowSize: FFT window size (MUST be power of 2)
  ///   - hopSize: Compute FFT every hopSize frames (default: windowSize/4 for efficiency)
  ///              Use hopSize=1 to compute every frame (slow but most responsive)
  ///              Use hopSize=windowSize for one FFT per window (most efficient)
  /// - Returns: Tensor [numBins, 2] where numBins = windowSize/2 + 1
  ///            [k, 0] = Real component, [k, 1] = Imaginary component
  ///
  /// The FFT operation has hop-based temporality: the FFT computation only runs every
  /// hopSize frames, while ring buffer management runs every frame. Operations downstream
  /// of FFT (like conv2d, mul) inherit this hop-based temporality and also only run
  /// every hopSize frames, avoiding redundant computation.
  public func fft(_ signal: NodeID, windowSize: Int, hopSize: Int? = nil) -> NodeID {
    precondition(
      windowSize > 0 && (windowSize & (windowSize - 1)) == 0,
      "windowSize must be a power of 2")

    // Default hopSize to windowSize/4 (75% overlap) - good balance of efficiency and responsiveness
    let actualHopSize = hopSize ?? max(1, windowSize / 4)
    precondition(actualHopSize > 0, "hopSize must be positive")

    let numBins = windowSize / 2 + 1
    let outputSize = numBins * 2

    // Allocate output tensor
    let cellId = alloc(vectorWidth: outputSize)
    let tensorId = nextTensorId
    nextTensorId += 1
    tensors[tensorId] = Tensor(id: tensorId, shape: [numBins, 2], cellId: cellId)
    cellToTensor[cellId] = tensorId

    // Allocate scratch for in-place FFT computation: real[N] + imag[N]
    let scratchCell = alloc(vectorWidth: windowSize * 2)

    // Allocate ring buffer for storing last windowSize samples
    let ringBufferCell = alloc(vectorWidth: windowSize)

    // Allocate write position cell (single float, 0.0 to windowSize-1)
    let writePosCell = alloc(vectorWidth: 1)

    // Allocate counter cell for hop timing (single float, 0 to hopSize-1)
    let counterCell = alloc(vectorWidth: 1)

    let fftNode = n(
      .fft(windowSize, actualHopSize, scratchCell, ringBufferCell, writePosCell, counterCell),
      [signal], shape: .tensor([numBins, 2]))
    nodeToTensor[fftNode] = tensorId

    // Register hop-based temporality for this FFT node
    // This allows downstream operations to inherit hop-based execution
    nodeHopRate[fftNode] = (actualHopSize, counterCell)

    return fftNode
  }

  /// Compute magnitude spectrum from FFT output.
  /// - Parameter fftOutput: FFT tensor output [numBins, 2]
  /// - Returns: Tensor [numBins] with magnitude = sqrt(real^2 + imag^2) for each bin
  public func fftMagnitude(_ fftOutput: NodeID) -> NodeID {
    // Get the FFT output tensor info
    guard let fftTensorId = nodeToTensor[fftOutput],
      let fftTensor = tensors[fftTensorId],
      fftTensor.shape.count == 2 && fftTensor.shape[1] == 2
    else {
      fatalError("fftMagnitude requires an FFT output tensor with shape [numBins, 2]")
    }

    let numBins = fftTensor.shape[0]

    // Allocate output tensor for magnitude
    let magCellId = alloc(vectorWidth: numBins)
    let magTensorId = nextTensorId
    nextTensorId += 1
    tensors[magTensorId] = Tensor(id: magTensorId, shape: [numBins], cellId: magCellId)
    cellToTensor[magCellId] = magTensorId

    // Create a tensor reference node for the magnitude output
    let magRef = n(.tensorRef(magTensorId), [], shape: .tensor([numBins]))
    nodeToTensor[magRef] = magTensorId

    // For each bin, compute sqrt(real^2 + imag^2)
    // We need to read from the FFT output tensor and write to the magnitude tensor
    // This is done element-wise: for k in 0..<numBins: mag[k] = sqrt(fft[k,0]^2 + fft[k,1]^2)

    // Create element-wise computation using peek operations
    // First, create a loop counter using phasor or similar mechanism
    // Actually, we need to use the tensor operation infrastructure

    // For now, we implement this by creating individual peek operations
    // This is inefficient but works; a better approach would be a dedicated op

    // Create nodes that will compute magnitude for each bin at read time
    // Using peek to read from the FFT tensor and compute magnitude

    // Create constant nodes for indices
    var lastMagNode: NodeID = magRef

    for k in 0..<numBins {
      let kConst = n(.constant(Float(k)))
      let realChannel = n(.constant(0.0))
      let imagChannel = n(.constant(1.0))

      // Peek real and imaginary parts
      let realVal = n(.peek, fftOutput, kConst, realChannel)
      let imagVal = n(.peek, fftOutput, kConst, imagChannel)

      // Compute magnitude = sqrt(real^2 + imag^2)
      let realSq = n(.mul, realVal, realVal)
      let imagSq = n(.mul, imagVal, imagVal)
      let sumSq = n(.add, realSq, imagSq)
      let mag = n(.sqrt, sumSq)

      // Write to magnitude tensor
      let writeOp = n(.memoryWrite(magCellId), kConst, mag)
      lastMagNode = n(.seq, lastMagNode, writeOp)
    }

    return lastMagNode
  }

  /// Compute Inverse FFT to reconstruct time-domain signal from spectrum.
  /// Uses overlap-add for smooth reconstruction.
  /// - Parameters:
  ///   - spectrum: Input spectrum tensor [numBins, 2] (typically from FFT or modified FFT output)
  ///   - windowSize: IFFT window size (MUST match the FFT windowSize that produced the spectrum)
  ///   - hopSize: How often to compute IFFT (default windowSize/4 for 75% overlap)
  /// - Returns: Scalar output (one sample per frame)
  ///
  /// The IFFT operation has hop-based temporality for the IFFT computation itself,
  /// while the overlap-add output runs every frame. This allows the expensive IFFT
  /// to run only when needed (when counter == 0).
  public func ifft(_ spectrum: NodeID, windowSize: Int, hopSize: Int? = nil) -> NodeID {
    precondition(
      windowSize > 0 && (windowSize & (windowSize - 1)) == 0,
      "windowSize must be a power of 2")

    // Default hopSize to windowSize/4 (75% overlap) for smooth reconstruction
    let actualHopSize = hopSize ?? max(1, windowSize / 4)
    precondition(actualHopSize > 0, "hopSize must be positive")

    // Allocate scratch for in-place IFFT computation: real[N] + imag[N]
    let scratchCell = alloc(vectorWidth: windowSize * 2)

    // Allocate output ring buffer for overlap-add reconstruction
    let outputRingCell = alloc(vectorWidth: windowSize)

    // Allocate read position cell (single float, 0.0 to windowSize-1)
    let readPosCell = alloc(vectorWidth: 1)

    // Allocate counter cell for hop timing (single float, 0 to hopSize-1)
    let counterCell = alloc(vectorWidth: 1)

    let ifftNode = n(
      .ifft(windowSize, actualHopSize, scratchCell, outputRingCell, readPosCell, counterCell),
      [spectrum], shape: .scalar)

    // Register hop-based temporality for this IFFT node
    // Note: IFFT output is scalar (one sample per frame via overlap-add),
    // but the expensive IFFT computation only runs every hopSize frames
    nodeHopRate[ifftNode] = (actualHopSize, counterCell)

    return ifftNode
  }

  /// Select a single row from a 2D tensor using a dynamic row index.
  /// The rowIndex is floored to an integer and wrapped using modulo.
  /// - Parameters:
  ///   - tensor: 2D tensor [numRows, numCols]
  ///   - rowIndex: Scalar row index (will be floored and wrapped)
  /// - Returns: 1D tensor [numCols] containing the selected row
  public func selectRow(tensor: NodeID, rowIndex: NodeID) throws -> NodeID {
    guard let tensorNode = nodes[tensor],
          case .tensor(let shape) = tensorNode.shape,
          shape.count == 2 else {
      throw DGenError.tensorError(op: "selectRow", reason: "requires 2D tensor input")
    }
    let numCols = shape[1]
    return n(.selectRow, [tensor, rowIndex], shape: .tensor([numCols]))
  }

  /// Read a row from a 2D tensor with linear interpolation between adjacent rows.
  /// Uses peekRowInline for SIMD-safe parallel execution with frame-indexed storage.
  /// - Parameters:
  ///   - tensor: 2D tensor [numRows, numCols]
  ///   - rowIndex: Scalar row index (fractional values interpolate between rows)
  /// - Returns: 1D tensor [numCols] containing the interpolated row
  public func peekRow(tensor: NodeID, rowIndex: NodeID) throws -> NodeID {
    guard let tensorNode = nodes[tensor],
          case .tensor(let shape) = tensorNode.shape,
          shape.count == 2 else {
      throw DGenError.tensorError(op: "peekRow", reason: "requires 2D tensor input")
    }
    let numRows = shape[0]
    let numCols = shape[1]

    // Allocate frame-indexed scratch cell for SIMD safety
    // Size: maxFrameCount * numCols
    let scratchCell = alloc(vectorWidth: maxFrameCount * numCols)

    return n(.peekRowInline(scratchCell: scratchCell, numRows: numRows, numCols: numCols),
             [tensor, rowIndex], shape: .tensor([numCols]))
  }

}

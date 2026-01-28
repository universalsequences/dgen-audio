extension Graph {
  /// Spectral loss: compute DFT-based magnitude MSE between two signals over a window
  /// Uses tape-based compute; no ring buffers are required.
  public func spectralLoss(_ sig1: NodeID, _ sig2: NodeID, windowSize: Int) -> NodeID {
    // Allocate scratch memory for two-pass backward gradient distribution
    // Conservative max: 4096 frames * windowSize * 2 (for sig1 and sig2 contributions)
    let maxFrameCount = 4096
    let scratchSize = maxFrameCount * windowSize * 2
    let scratchCell = alloc(vectorWidth: scratchSize)

    // Create Pass1: computes spectral loss and stores DFT contributions (for backward)
    let pass1 = n(.spectralLossPass1(windowSize, scratchCell), sig1, sig2)

    // Create Pass2: no-op in forward, reduces contributions to gradients in backward
    // Depends on pass1 to ensure correct kernel ordering
    let pass2 = n(.spectralLossPass2(windowSize, scratchCell), pass1)

    return pass2  // Return Pass2 result (forwards Pass1's loss value)
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
    let sampleRate = n(.constant(44100.0))

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

}

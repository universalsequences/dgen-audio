import Accelerate
import Foundation

/// Fixed CPU-side selection metric for best-checkpoint tracking.
///
/// The training loss mixes scheduled aux terms (loudness, regularizers) whose
/// weights ramp across steps, so its value is not comparable between steps and
/// min-picking it selects pre-ramp checkpoints. Checkpoint selection instead
/// scores rendered audio against the target with this schedule-independent
/// multi-resolution spectral distance, mirroring the spectral portion of the
/// training objective (linear + smooth-log magnitude L1, Hann-windowed).
enum ModalBestCheckpointScorer {

  /// Lower is better. Returns nil if no usable window size fits the signal.
  ///
  /// Uses vDSP: this runs inside the training loop, and a naive Swift FFT here
  /// cost ~1s per evaluation — a ~40% tax on step time at the default cadence.
  /// `referenceScore` is the same metric written plainly, kept as the
  /// differential oracle for `BestMetricTests`.
  static func multiScaleSpectralScore(
    prediction: [Float],
    target: [Float],
    windowSizes: [Int],
    hopDivisor: Int,
    logEpsilon: Float
  ) -> Float? {
    let n = min(prediction.count, target.count)
    let usable = windowSizes.filter { $0 > 1 && $0 <= n && $0 & ($0 - 1) == 0 }
    guard !usable.isEmpty, n > 0 else { return nil }

    var total: Float = 0
    for w in usable {
      guard let setup = fftSetup(forWindow: w) else { continue }
      let hop = max(1, w / max(1, hopDivisor))
      let window = hannWindow(w)
      let half = w / 2
      let numBins = half + 1
      let epsSq = logEpsilon * logEpsilon

      var predMag = [Float](repeating: 0, count: numBins)
      var targetMag = [Float](repeating: 0, count: numBins)
      var linSum: Float = 0
      var logSum: Float = 0
      var count = 0

      var start = 0
      while start + w <= n {
        magnitudes(
          of: prediction, at: start, window: window, setup: setup, log2n: log2Size(w),
          into: &predMag)
        magnitudes(
          of: target, at: start, window: window, setup: setup, log2n: log2Size(w),
          into: &targetMag)

        for b in 0..<numBins {
          let p = predMag[b]
          let t = targetMag[b]
          linSum += Swift.abs(sqrtf(p) - sqrtf(t))
          logSum += Swift.abs(0.5 * logf(p + epsSq) - 0.5 * logf(t + epsSq))
        }
        count += numBins
        start += hop
      }
      if count > 0 {
        total += (linSum + logSum) / Float(count)
      }
    }
    return total / Float(usable.count)
  }

  // MARK: - vDSP plumbing

  // Single-threaded use inside the training loop; setups and windows are cached
  // because creating them per frame dominates the measurement.
  private static var setups: [Int: FFTSetup] = [:]
  private static var windows: [Int: [Float]] = [:]

  private static func log2Size(_ w: Int) -> vDSP_Length {
    vDSP_Length(round(log2(Double(w))))
  }

  private static func fftSetup(forWindow w: Int) -> FFTSetup? {
    if let existing = setups[w] { return existing }
    guard let created = vDSP_create_fftsetup(log2Size(w), FFTRadix(kFFTRadix2)) else {
      return nil
    }
    setups[w] = created
    return created
  }

  private static func hannWindow(_ w: Int) -> [Float] {
    if let existing = windows[w] { return existing }
    let values = (0..<w).map { i in
      Float(0.5 * (1.0 - Foundation.cos(2.0 * Double.pi * Double(i) / Double(w))))
    }
    windows[w] = values
    return values
  }

  /// Squared magnitudes of one windowed frame, in `out` (length w/2 + 1).
  private static func magnitudes(
    of samples: [Float],
    at offset: Int,
    window: [Float],
    setup: FFTSetup,
    log2n: vDSP_Length,
    into out: inout [Float]
  ) {
    let w = window.count
    let half = w / 2
    var windowed = [Float](repeating: 0, count: w)
    samples.withUnsafeBufferPointer { buffer in
      let base = buffer.baseAddress! + offset
      vDSP_vmul(base, 1, window, 1, &windowed, 1, vDSP_Length(w))
    }

    var real = [Float](repeating: 0, count: half)
    var imag = [Float](repeating: 0, count: half)
    real.withUnsafeMutableBufferPointer { realPtr in
      imag.withUnsafeMutableBufferPointer { imagPtr in
        var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
        windowed.withUnsafeBufferPointer { input in
          input.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { typed in
            vDSP_ctoz(typed, 2, &split, 1, vDSP_Length(half))
          }
        }
        vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
      }
    }

    // vDSP packs DC in real[0] and Nyquist in imag[0], and scales results by 2.
    let scale: Float = 0.5
    out[0] = (real[0] * scale) * (real[0] * scale)
    out[half] = (imag[0] * scale) * (imag[0] * scale)
    for k in 1..<half {
      let re = real[k] * scale
      let im = imag[k] * scale
      out[k] = re * re + im * im
    }
  }

  // MARK: - Reference implementation (test oracle)

  /// Same metric, written plainly. Not used in training — kept so the vDSP path
  /// has something to be checked against.
  static func referenceScore(
    prediction: [Float],
    target: [Float],
    windowSizes: [Int],
    hopDivisor: Int,
    logEpsilon: Float
  ) -> Float? {
    let n = min(prediction.count, target.count)
    let usable = windowSizes.filter { $0 > 1 && $0 <= n && $0 & ($0 - 1) == 0 }
    guard !usable.isEmpty, n > 0 else { return nil }

    var total: Float = 0
    for w in usable {
      let hop = max(1, w / max(1, hopDivisor))
      let hann = (0..<w).map { i in
        Float(0.5 * (1.0 - Foundation.cos(2.0 * Double.pi * Double(i) / Double(w))))
      }
      let numBins = w / 2 + 1
      var linSum: Float = 0
      var logSum: Float = 0
      var count = 0
      var re = [Float](repeating: 0, count: w)
      var im = [Float](repeating: 0, count: w)
      let epsSq = logEpsilon * logEpsilon

      var start = 0
      while start + w <= n {
        for i in 0..<w {
          re[i] = prediction[start + i] * hann[i]
          im[i] = 0
        }
        fftInPlace(re: &re, im: &im)
        var predMag = [Float](repeating: 0, count: numBins)
        for b in 0..<numBins {
          predMag[b] = re[b] * re[b] + im[b] * im[b]
        }
        for i in 0..<w {
          re[i] = target[start + i] * hann[i]
          im[i] = 0
        }
        fftInPlace(re: &re, im: &im)
        for b in 0..<numBins {
          let tMagSq = re[b] * re[b] + im[b] * im[b]
          linSum += Swift.abs(sqrtf(predMag[b]) - sqrtf(tMagSq))
          logSum += Swift.abs(0.5 * logf(predMag[b] + epsSq) - 0.5 * logf(tMagSq + epsSq))
        }
        count += numBins
        start += hop
      }
      if count > 0 {
        total += (linSum + logSum) / Float(count)
      }
    }
    return total / Float(usable.count)
  }

  /// Iterative in-place radix-2 Cooley-Tukey FFT. Length must be a power of 2.
  private static func fftInPlace(re: inout [Float], im: inout [Float]) {
    let n = re.count
    guard n > 1 else { return }

    var j = 0
    for i in 0..<(n - 1) {
      if i < j {
        re.swapAt(i, j)
        im.swapAt(i, j)
      }
      var m = n >> 1
      while m >= 1 && j & m != 0 {
        j ^= m
        m >>= 1
      }
      j |= m
    }

    var len = 2
    while len <= n {
      let ang = -2.0 * Double.pi / Double(len)
      let wRe = Float(Foundation.cos(ang))
      let wIm = Float(Foundation.sin(ang))
      var i = 0
      while i < n {
        var curRe: Float = 1
        var curIm: Float = 0
        for k in 0..<(len / 2) {
          let a = i + k
          let b = i + k + len / 2
          let tRe = re[b] * curRe - im[b] * curIm
          let tIm = re[b] * curIm + im[b] * curRe
          re[b] = re[a] - tRe
          im[b] = im[a] - tIm
          re[a] += tRe
          im[a] += tIm
          let nRe = curRe * wRe - curIm * wIm
          curIm = curRe * wIm + curIm * wRe
          curRe = nRe
        }
        i += len
      }
      len <<= 1
    }
  }
}

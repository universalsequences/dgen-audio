import Foundation

/// Fixed CPU-side selection metric for best-checkpoint tracking.
///
/// The training loss mixes scheduled aux terms (loudness, regularizers) whose
/// weights ramp across steps, so its value is not comparable between steps and
/// min-picking it selects pre-ramp checkpoints. Checkpoint selection instead
/// scores rendered audio against the target with this schedule-independent
/// multi-resolution spectral distance, mirroring the spectral portion of the
/// training objective (linear + smooth-log magnitude L1, Hann-windowed).
enum BestCheckpointScorer {

  /// Lower is better. Returns nil if no usable window size fits the signal.
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
        // prediction spectrum
        for i in 0..<w {
          re[i] = prediction[start + i] * hann[i]
          im[i] = 0
        }
        fftInPlace(re: &re, im: &im)
        var predMag = [Float](repeating: 0, count: numBins)
        for b in 0..<numBins {
          predMag[b] = re[b] * re[b] + im[b] * im[b]
        }
        // target spectrum
        for i in 0..<w {
          re[i] = target[start + i] * hann[i]
          im[i] = 0
        }
        fftInPlace(re: &re, im: &im)
        for b in 0..<numBins {
          let tMagSq = re[b] * re[b] + im[b] * im[b]
          let pMag = Foundation.sqrtf(predMag[b])
          let tMag = Foundation.sqrtf(tMagSq)
          linSum += Swift.abs(pMag - tMag)
          let pLog = 0.5 * Foundation.logf(predMag[b] + epsSq)
          let tLog = 0.5 * Foundation.logf(tMagSq + epsSq)
          logSum += Swift.abs(pLog - tLog)
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

    // Bit-reversal permutation
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

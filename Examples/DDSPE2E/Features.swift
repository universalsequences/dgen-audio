import Accelerate
import Foundation

struct ChunkFeatures: Codable {
  var f0Hz: [Float]
  var loudnessDB: [Float]
  var uvMask: [Float]
}

enum FeatureExtractor {
  static func extract(
    samples: [Float],
    sampleRate: Float,
    config: DDSPE2EConfig
  ) -> ChunkFeatures {
    let frameStarts = makeFrameStarts(
      sampleCount: samples.count,
      frameSize: config.frameSize,
      frameHop: config.frameHop
    )

    var f0Hz = [Float]()
    var loudnessDB = [Float]()
    var uvMask = [Float]()
    f0Hz.reserveCapacity(frameStarts.count)
    loudnessDB.reserveCapacity(frameStarts.count)
    uvMask.reserveCapacity(frameStarts.count)

    for start in frameStarts {
      let frame = frameAt(samples: samples, start: start, frameSize: config.frameSize)
      let rms = rootMeanSquare(frame)
      let loudness = max(-120.0, 20.0 * log10(max(rms, 1e-7)))
      loudnessDB.append(loudness)

      let (pitch, voiced) = estimateF0Autocorr(
        frame: frame,
        sampleRate: sampleRate,
        minF0Hz: config.minF0Hz,
        maxF0Hz: config.maxF0Hz,
        silenceRMS: config.silenceRMS,
        voicedThreshold: config.voicedThreshold
      )
      f0Hz.append(pitch)
      uvMask.append(voiced ? 1.0 : 0.0)
    }

    return ChunkFeatures(f0Hz: f0Hz, loudnessDB: loudnessDB, uvMask: uvMask)
  }

  /// MFCC-style reference descriptor. Averaged log-mel spectra preserve the
  /// broad spectral envelope and noise character while the DCT suppresses the
  /// fine harmonic comb that would otherwise mostly encode reference pitch.
  static func timbreDescriptor(
    samples: [Float],
    sampleRate: Float,
    features: ChunkFeatures,
    frameSize: Int,
    frameHop: Int,
    count: Int
  ) -> [Float] {
    guard count > 0, frameSize > 1, frameSize & (frameSize - 1) == 0 else {
      return [Float](repeating: 0, count: max(0, count))
    }
    let voiced = features.f0Hz.indices.filter { features.uvMask[$0] > 0.5 }
    guard !voiced.isEmpty else { return [Float](repeating: 0, count: count) }
    let picks = [voiced.count / 4, voiced.count / 2, (3 * voiced.count) / 4]
      .map { voiced[min($0, voiced.count - 1)] }
    let log2n = vDSP_Length(round(log2(Double(frameSize))))
    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
      return [Float](repeating: 0, count: count)
    }
    defer { vDSP_destroy_fftsetup(setup) }

    let half = frameSize / 2
    var averagePower = [Float](repeating: 0, count: half + 1)
    for frameIndex in picks {
      var windowed = frameAt(
        samples: samples, start: frameIndex * frameHop, frameSize: frameSize)
      let mean = windowed.reduce(0, +) / Float(frameSize)
      for n in windowed.indices {
        let hann = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(frameSize - 1))
        windowed[n] = (windowed[n] - mean) * hann
      }
      var real = [Float](repeating: 0, count: half)
      var imag = [Float](repeating: 0, count: half)
      real.withUnsafeMutableBufferPointer { rp in
        imag.withUnsafeMutableBufferPointer { ip in
          var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
          windowed.withUnsafeBufferPointer { input in
            input.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
              vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
            }
          }
          vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
        }
      }
      averagePower[0] += real[0] * real[0]
      averagePower[half] += imag[0] * imag[0]
      for bin in 1..<half {
        averagePower[bin] += real[bin] * real[bin] + imag[bin] * imag[bin]
      }
    }
    let invFrames = 1 / Float(max(1, picks.count))
    averagePower = averagePower.map { $0 * invFrames }

    let melBands = max(32, count * 2)
    func mel(_ hz: Float) -> Float { 2595 * log10(1 + hz / 700) }
    func hz(_ mel: Float) -> Float { 700 * (pow(10, mel / 2595) - 1) }
    let melMin = mel(30)
    let melMax = mel(sampleRate * 0.5)
    let points = (0..<(melBands + 2)).map { i -> Int in
      let value = melMin + (melMax - melMin) * Float(i) / Float(melBands + 1)
      return min(half, max(0, Int((hz(value) / sampleRate * Float(frameSize)).rounded())))
    }
    var logBands = [Float](repeating: 0, count: melBands)
    for band in 0..<melBands {
      let left = points[band]
      let center = max(left + 1, points[band + 1])
      let right = max(center + 1, points[band + 2])
      var energy: Float = 0
      var weightSum: Float = 0
      if left < min(center, half + 1) {
        for bin in left..<min(center, half + 1) {
          let weight = Float(bin - left) / Float(max(1, center - left))
          energy += averagePower[bin] * weight
          weightSum += weight
        }
      }
      if center < min(right, half + 1) {
        for bin in center..<min(right, half + 1) {
          let weight = Float(right - bin) / Float(max(1, right - center))
          energy += averagePower[bin] * weight
          weightSum += weight
        }
      }
      logBands[band] = log(max(energy / max(weightSum, 1e-6), 1e-12))
    }

    // Drop coefficient zero (overall level); shared target loudness controls it.
    return (1...count).map { coefficient in
      var value: Float = 0
      for band in 0..<melBands {
        value += logBands[band]
          * cos(Float.pi * Float(coefficient) * (Float(band) + 0.5) / Float(melBands))
      }
      return tanh(value / Float(melBands * 4))
    }
  }

  /// Time-varying reference representation: `timeFrames` log-mel frames of
  /// `melBins` bins each, row-major [timeFrames, melBins]. Unlike
  /// `timbreDescriptor` there is NO averaging across time and no DCT — the
  /// learned temporal encoder sees attack/noise/spectral-envelope evolution
  /// directly. The chunk-wide mean log energy is subtracted so overall level
  /// (controlled by the target loudness features) does not dominate `z`.
  static func timbreLogMelFrames(
    samples: [Float],
    sampleRate: Float,
    frameSize: Int,
    frameHop: Int,
    timeFrames: Int,
    melBins: Int
  ) -> [Float] {
    guard timeFrames > 0, melBins > 0, frameSize > 1,
      frameSize & (frameSize - 1) == 0
    else {
      return [Float](repeating: 0, count: max(0, timeFrames * melBins))
    }
    let frameStarts = makeFrameStarts(
      sampleCount: samples.count, frameSize: frameSize, frameHop: frameHop)
    let log2n = vDSP_Length(round(log2(Double(frameSize))))
    guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
      return [Float](repeating: 0, count: timeFrames * melBins)
    }
    defer { vDSP_destroy_fftsetup(setup) }
    let half = frameSize / 2

    // Mel triangle bank edges shared by every frame.
    func mel(_ hz: Float) -> Float { 2595 * log10(1 + hz / 700) }
    func hz(_ mel: Float) -> Float { 700 * (pow(10, mel / 2595) - 1) }
    let melMin = mel(30)
    let melMax = mel(sampleRate * 0.5)
    let points = (0..<(melBins + 2)).map { i -> Int in
      let value = melMin + (melMax - melMin) * Float(i) / Float(melBins + 1)
      return min(half, max(0, Int((hz(value) / sampleRate * Float(frameSize)).rounded())))
    }

    var out = [Float](repeating: 0, count: timeFrames * melBins)
    for t in 0..<timeFrames {
      // Uniform coverage of the chunk from its first frame (attack) to its
      // last, independent of how many analysis frames the chunk has.
      let pick = timeFrames == 1
        ? 0
        : (t * (frameStarts.count - 1)) / (timeFrames - 1)
      var windowed = frameAt(
        samples: samples, start: frameStarts[min(pick, frameStarts.count - 1)],
        frameSize: frameSize)
      let mean = windowed.reduce(0, +) / Float(frameSize)
      for n in windowed.indices {
        let hann = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(frameSize - 1))
        windowed[n] = (windowed[n] - mean) * hann
      }
      var real = [Float](repeating: 0, count: half)
      var imag = [Float](repeating: 0, count: half)
      real.withUnsafeMutableBufferPointer { rp in
        imag.withUnsafeMutableBufferPointer { ip in
          var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
          windowed.withUnsafeBufferPointer { input in
            input.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
              vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
            }
          }
          vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
        }
      }
      var power = [Float](repeating: 0, count: half + 1)
      power[0] = real[0] * real[0]
      power[half] = imag[0] * imag[0]
      for bin in 1..<half {
        power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin]
      }

      for band in 0..<melBins {
        let left = points[band]
        let center = Swift.max(left + 1, points[band + 1])
        let right = Swift.max(center + 1, points[band + 2])
        var energy: Float = 0
        var weightSum: Float = 0
        if left < min(center, half + 1) {
          for bin in left..<min(center, half + 1) {
            let weight = Float(bin - left) / Float(Swift.max(1, center - left))
            energy += power[bin] * weight
            weightSum += weight
          }
        }
        if center < min(right, half + 1) {
          for bin in center..<min(right, half + 1) {
            let weight = Float(right - bin) / Float(Swift.max(1, right - center))
            energy += power[bin] * weight
            weightSum += weight
          }
        }
        out[t * melBins + band] = log(Swift.max(energy / Swift.max(weightSum, 1e-6), 1e-12))
      }
    }

    // Remove overall level; softly bound the remaining shape/contrast values.
    let globalMean = out.reduce(0, +) / Float(out.count)
    for i in out.indices {
      out[i] = tanh((out[i] - globalMean) / 4.0)
    }
    return out
  }

  /// Canonical target controls for reference-conditioned training. TinySOL's
  /// per-recording f0/loudness microstructure can reveal instrument identity,
  /// allowing the decoder to ignore its reference. Sustained-note training
  /// therefore uses median pitch and a shared dynamic level; the synthesizer
  /// still receives the original f0 trajectory for oscillator accuracy.
  static func canonicalReferenceControls(
    features: ChunkFeatures,
    sourceFile: String
  ) -> ChunkFeatures {
    let voicedF0 = zip(features.f0Hz, features.uvMask)
      .filter { $0.1 > 0.5 && $0.0 > 0 }
      .map(\.0)
      .sorted()
    let medianF0 = voicedF0.isEmpty ? Float(440) : voicedF0[voicedF0.count / 2]
    let dynamicDB: Float
    if sourceFile.contains("-pp-") {
      dynamicDB = -40.5
    } else if sourceFile.contains("-ff-") {
      dynamicDB = -16.5
    } else {
      dynamicDB = -27.0
    }
    return ChunkFeatures(
      f0Hz: [Float](repeating: medianF0, count: features.f0Hz.count),
      loudnessDB: [Float](repeating: dynamicDB, count: features.loudnessDB.count),
      uvMask: [Float](repeating: 1, count: features.uvMask.count))
  }

  static func makeFrameStarts(sampleCount: Int, frameSize: Int, frameHop: Int) -> [Int] {
    if sampleCount <= 0 {
      return [0]
    }
    if sampleCount <= frameSize {
      return [0]
    }

    var starts = [Int]()
    var start = 0
    while start + frameSize <= sampleCount {
      starts.append(start)
      start += frameHop
    }

    if let last = starts.last, last + frameSize < sampleCount {
      starts.append(max(0, sampleCount - frameSize))
    }

    return starts
  }

  private static func frameAt(samples: [Float], start: Int, frameSize: Int) -> [Float] {
    var frame = [Float](repeating: 0, count: frameSize)
    let end = min(samples.count, start + frameSize)
    if end > start {
      frame.replaceSubrange(0..<(end - start), with: samples[start..<end])
    }
    return frame
  }

  private static func rootMeanSquare(_ frame: [Float]) -> Float {
    if frame.isEmpty { return 0 }
    var sum: Float = 0
    for value in frame {
      sum += value * value
    }
    return sqrt(sum / Float(frame.count))
  }

  private static func estimateF0Autocorr(
    frame: [Float],
    sampleRate: Float,
    minF0Hz: Float,
    maxF0Hz: Float,
    silenceRMS: Float,
    voicedThreshold: Float
  ) -> (f0Hz: Float, voiced: Bool) {
    let rms = rootMeanSquare(frame)
    if rms < silenceRMS {
      return (0.0, false)
    }

    let mean = frame.reduce(0, +) / Float(frame.count)
    var centered = frame
    for i in centered.indices {
      centered[i] -= mean
    }

    var energy: Float = 0
    for x in centered {
      energy += x * x
    }
    if energy <= 1e-8 {
      return (0.0, false)
    }

    let minLag = max(1, Int(sampleRate / maxF0Hz))
    let maxLag = min(centered.count - 2, Int(sampleRate / minF0Hz))
    if maxLag <= minLag {
      return (0.0, false)
    }

    var corr = [Float](repeating: 0, count: maxLag + 1)
    var bestLag = minLag
    var bestCorr: Float = -Float.greatestFiniteMagnitude

    for lag in minLag...maxLag {
      let upper = centered.count - lag
      if upper <= 0 { continue }
      var sum: Float = 0
      centered.withUnsafeBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        vDSP_dotpr(
          base,
          1,
          base.advanced(by: lag),
          1,
          &sum,
          vDSP_Length(upper)
        )
      }
      let normalized = sum / (energy + 1e-8)
      corr[lag] = normalized

      if normalized > bestCorr {
        bestCorr = normalized
        bestLag = lag
      }
    }

    if bestCorr < voicedThreshold {
      return (0.0, false)
    }

    // The global autocorrelation maximum often lands on an integer multiple
    // of the period (notably for very high flute notes). Prefer the earliest
    // strong local peak; this is the fundamental-period peak in a periodic
    // monophonic frame and avoids subharmonic/octave errors.
    if maxLag - minLag >= 2 {
      let peakThreshold = max(voicedThreshold, bestCorr * 0.9)
      for lag in (minLag + 1)..<maxLag {
        if corr[lag] >= peakThreshold,
          corr[lag] > corr[lag - 1],
          corr[lag] >= corr[lag + 1]
        {
          bestLag = lag
          break
        }
      }
    }

    var refinedLag = Float(bestLag)
    if bestLag > minLag && bestLag < maxLag {
      let y0 = corr[bestLag - 1]
      let y1 = corr[bestLag]
      let y2 = corr[bestLag + 1]
      let denom = y0 - (2.0 * y1) + y2
      if abs(denom) > 1e-8 {
        let delta = 0.5 * (y0 - y2) / denom
        refinedLag += max(-1.0, min(1.0, delta))
      }
    }

    let f0 = sampleRate / max(refinedLag, 1.0)
    return (f0, true)
  }
}

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
      var sum: Float = 0
      let upper = centered.count - lag
      if upper <= 0 { continue }
      for i in 0..<upper {
        sum += centered[i] * centered[i + lag]
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

import Foundation

struct PitchPoint: Codable {
  var time: Float
  var hz: Float
  var confidence: Float
}

struct PitchFit: Codable {
  var fStart: Float
  var fEnd: Float
  var pitchDecay: Float
  var error: Float
}

enum PitchTrack {
  static func extract(
    samples: [Float],
    sampleRate: Float,
    windowSize: Int = 1024,
    hop: Int = 256,
    minHz: Float = 30,
    maxHz: Float = 300
  ) -> [PitchPoint] {
    guard samples.count >= windowSize else { return [] }
    let lagMin = max(1, Int(sampleRate / maxHz))
    let lagMax = min(windowSize - 2, Int(sampleRate / minHz))
    guard lagMin < lagMax else { return [] }

    var points: [PitchPoint] = []
    var start = 0
    while start + windowSize <= samples.count {
      var frame = [Float](repeating: 0, count: windowSize)
      var mean: Float = 0
      for i in 0..<windowSize { mean += samples[start + i] }
      mean /= Float(windowSize)

      var energy: Float = 0
      for i in 0..<windowSize {
        let w = 0.5 - 0.5 * Foundation.cos(2.0 * Double.pi * Double(i) / Double(windowSize - 1))
        let value = (samples[start + i] - mean) * Float(w)
        frame[i] = value
        energy += value * value
      }
      if energy < 1e-7 {
        start += hop
        continue
      }

      var bestLag = lagMin
      var best: Float = -.infinity
      var previous: Float = 0
      var bestPrev: Float = 0
      var bestNext: Float = 0

      for lag in lagMin...lagMax {
        var corr: Float = 0
        var lagEnergy: Float = 0
        for i in 0..<(windowSize - lag) {
          corr += frame[i] * frame[i + lag]
          lagEnergy += frame[i + lag] * frame[i + lag]
        }
        let normalized = corr / max(Foundation.sqrt(energy * lagEnergy), 1e-12)
        if normalized > best {
          best = normalized
          bestLag = lag
          bestPrev = previous
        }
        if lag == bestLag + 1 {
          bestNext = normalized
        }
        previous = normalized
      }

      let denom = bestPrev - 2.0 * best + bestNext
      let offset = abs(denom) > 1e-9 ? 0.5 * (bestPrev - bestNext) / denom : 0
      let lag = Float(bestLag) + Swift.max(-0.5, Swift.min(0.5, offset))
      let hz = sampleRate / lag
      let time = Float(start + windowSize / 2) / sampleRate
      if best > 0.2, hz >= minHz, hz <= maxHz {
        points.append(PitchPoint(time: time, hz: hz, confidence: best))
      }
      start += hop
    }
    return points
  }

  static func fit(points: [PitchPoint]) -> PitchFit {
    let candidates = points
      .filter { $0.time <= 0.25 && $0.hz >= 30 && $0.hz <= 300 && $0.confidence > 0.2 }
      .prefix(64)
    var monotonic: [PitchPoint] = []
    for point in candidates {
      if let previous = monotonic.last {
        let largeUpwardJump = point.hz > previous.hz * 1.12
        let highRidgeAfterBody = point.hz > 250 && previous.hz < 180
        if largeUpwardJump || highRidgeAfterBody {
          continue
        }
      }
      monotonic.append(point)
    }
    let usable = monotonic.count >= 3 ? monotonic : Array(candidates)

    guard usable.count >= 3 else {
      return PitchFit(fStart: 120, fEnd: 45, pitchDecay: -35, error: .infinity)
    }

    var best = PitchFit(fStart: 120, fEnd: 45, pitchDecay: -35, error: .infinity)
    let fEndSteps = stride(from: Float(35), through: Float(60), by: Float(0.5))
    let decaySteps = stride(from: Float(-80), through: Float(-15), by: Float(1.0))

    for fEnd in fEndSteps {
      for decay in decaySteps {
        var num: Float = 0
        var den: Float = 0
        for point in usable {
          let e = Foundation.exp(decay * point.time)
          let w = point.confidence * point.confidence
          num += w * e * (point.hz - fEnd)
          den += w * e * e
        }
        guard den > 1e-9 else { continue }
        let amplitude = Swift.min(Swift.max(num / den, 80 - fEnd), 180 - fEnd)
        let fStart = fEnd + amplitude
        var error: Float = 0
        var weightSum: Float = 0
        for point in usable {
          let predicted = fEnd + amplitude * Foundation.exp(decay * point.time)
          let delta = predicted - point.hz
          let w = point.confidence * point.confidence
          error += w * delta * delta
          weightSum += w
        }
        error /= max(weightSum, 1e-9)
        if error < best.error {
          best = PitchFit(fStart: fStart, fEnd: fEnd, pitchDecay: decay, error: error)
        }
      }
    }
    return best
  }

  static func fit(samples: [Float], sampleRate: Float) -> PitchFit {
    let primary = fit(points: extract(samples: samples, sampleRate: sampleRate))
    if primary.error.isFinite,
      primary.error < 400,
      primary.fStart > 80.1,
      primary.fStart < 179.9,
      primary.fEnd > 35.1,
      primary.fEnd < 59.9,
      primary.pitchDecay > -79.9,
      primary.pitchDecay < -15.1
    {
      return primary
    }

    let lowFrequencyFallback = fit(
      points: extract(
        samples: samples,
        sampleRate: sampleRate,
        windowSize: 2048,
        hop: 256,
        minHz: 30,
        maxHz: 300))
    if lowFrequencyFallback.error < primary.error {
      return lowFrequencyFallback
    }
    return primary
  }
}

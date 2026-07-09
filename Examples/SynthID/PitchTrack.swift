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
  /// Mean-square contour fit error, or nil when the signal has too few usable
  /// pitch points. Optional keeps fallback diagnostics JSON-encodable.
  var error: Float?
}

enum PitchTrack {
  // Window must cover ~2 periods of minHz (44100/30 = 1470-sample lag needs
  // >= 4096-sample windows); 1024 windows cannot correlate the 35-60 Hz fEnd
  // band at all and latch onto spurious high-frequency peaks.
  static func extract(
    samples: [Float],
    sampleRate: Float,
    windowSize: Int = 4096,
    hop: Int = 512,
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

  /// Estimate fEnd from the tail of the signal, where the body has settled to a
  /// quasi-stationary sine at fEnd for hundreds of milliseconds. Long-window
  /// autocorrelation there is far more reliable than any point of the swept
  /// contour (sub-1% vs several %).
  static func tailFEnd(
    samples: [Float],
    sampleRate: Float,
    minHz: Float = 30,
    maxHz: Float = 70
  ) -> PitchPoint? {
    let n = samples.count
    let windowSize = Swift.min(16384, n / 2)
    guard Float(windowSize) > 2.2 * sampleRate / minHz else { return nil }
    let startAt = Swift.max(0, Int(Float(n) * 0.5))
    let tail = Array(samples[startAt...])
    let points = extract(
      samples: tail,
      sampleRate: sampleRate,
      windowSize: windowSize,
      hop: 2048,
      minHz: minHz,
      maxHz: maxHz)
    guard !points.isEmpty else { return nil }
    let sorted = points.map(\.hz).sorted()
    let median = sorted[sorted.count / 2]
    let confidence = points.map(\.confidence).max() ?? 0
    return PitchPoint(time: Float(startAt) / sampleRate, hz: median, confidence: confidence)
  }

  static func fit(
    points: [PitchPoint],
    fEndRange: ClosedRange<Float>? = nil,
    fEndStep: Float = 0.5
  ) -> PitchFit {
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
      return PitchFit(fStart: 120, fEnd: 45, pitchDecay: -35, error: nil)
    }

    var best = PitchFit(fStart: 120, fEnd: 45, pitchDecay: -35, error: nil)
    var bestError = Float.infinity
    let fEndLo = Swift.max(Float(35), fEndRange?.lowerBound ?? 35)
    let fEndHi = Swift.min(Float(60), fEndRange?.upperBound ?? 60)
    let fEndSteps = stride(from: fEndLo, through: fEndHi, by: fEndStep)
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
        if error < bestError {
          bestError = error
          best = PitchFit(fStart: fStart, fEnd: fEnd, pitchDecay: decay, error: error)
        }
      }
    }
    return best
  }

  static func fit(samples: [Float], sampleRate: Float) -> PitchFit {
    // Contour extraction favors time resolution for the fast early sweep
    // (fStart/pitchDecay); 2048-sample windows only support >= ~50 Hz, which is
    // fine because fEnd comes from the long-window tail anchor below.
    let points = extract(
      samples: samples,
      sampleRate: sampleRate,
      windowSize: 2048,
      hop: 256,
      minHz: 50,
      maxHz: 300)
    let tail = tailFEnd(samples: samples, sampleRate: sampleRate)
    if let tail, tail.confidence > 0.5 {
      // The tail anchor is accurate to ~0.1%; the swept-contour error metric is
      // biased upward near fEnd (window smearing), so give it almost no say.
      let anchored = fit(
        points: points,
        fEndRange: (tail.hz - 0.25)...(tail.hz + 0.25),
        fEndStep: 0.1)
      if anchored.error?.isFinite == true {
        return anchored
      }
      // No usable contour points: still trust the tail measurement for fEnd.
      return PitchFit(
        fStart: 120,
        fEnd: Swift.min(Swift.max(tail.hz, 35), 60),
        pitchDecay: -35,
        error: nil)
    }
    return fit(points: points)
  }
}

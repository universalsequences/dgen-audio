// EncoderFeatures.swift — host log-mel features for the latent encoder
// (docs/LATENT_ENCODER_SPEC.md §2 "Features").
//
// The recipe is chosen so a compiled patch can rebuild it from audio with existing ops:
//
//   frame t = x[t·hop ..< t·hop + window]      (frames from sample 0, no centring)
//   X = DFT(frame · hann)                       periodic Hann, sum convention (no 1/N)
//   mel = |X[0 ... window/2]| @ M               M: [window/2 + 1, bands] triangle bank
//   f = (log(mel + eps) − mean[band]) / std[band]
//
// Output layout is [bands, frames] row-major (channels first, time last), flattened per row.

import Accelerate
import DGenTrainProtocol
import Foundation

struct EncoderFeatureConfig: Equatable {
  var sampleRate: Float
  var window = 1024
  var hop = 256
  var bands = 64
  /// Seconds of onset-aligned audio analysed.
  var seconds: Float = 0.5
  var eps: Float = 1e-5
  var fMin: Float = 30

  var cropSamples: Int { Int((seconds * sampleRate).rounded()) }
  var frames: Int { max(0, (cropSamples - window) / hop + 1) }
  var bins: Int { window / 2 + 1 }
  /// Per-row feature width.
  var width: Int { bands * frames }

  var json: [String: Any] {
    ["sample_rate": Double(sampleRate), "window": window, "hop": hop, "bands": bands,
     "seconds": Double(seconds), "eps": Double(eps), "f_min": Double(fMin), "frames": frames,
     "hann": "periodic", "layout": "[bands, frames]", "magnitude": "abs(dft), sum convention"]
  }
}

final class EncoderFeatureExtractor {
  let config: EncoderFeatureConfig
  /// [bins, bands] row-major.
  let melFilters: [Float]
  private let hann: [Float]
  private let log2n: vDSP_Length
  private let setup: FFTSetup

  init(_ config: EncoderFeatureConfig) {
    precondition(config.window > 1 && config.window & (config.window - 1) == 0,
                 "feature window must be a power of two")
    self.config = config
    let n = config.window
    hann = (0..<n).map { 0.5 - 0.5 * Foundation.cos(2 * Float.pi * Float($0) / Float(n)) }
    melFilters = Self.melBank(config)
    log2n = vDSP_Length(Foundation.log2(Double(n)).rounded())
    setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
  }

  deinit { vDSP_destroy_fftsetup(setup) }

  /// HTK-mel triangles evaluated at each bin's exact frequency (unnormalised peak 1), so
  /// narrow low bands still get weight from the nearest bins.
  static func melBank(_ c: EncoderFeatureConfig) -> [Float] {
    func mel(_ hz: Float) -> Float { 2595 * Foundation.log10(1 + hz / 700) }
    func hz(_ m: Float) -> Float { 700 * (Foundation.pow(10, m / 2595) - 1) }
    let lo = mel(c.fMin), hi = mel(c.sampleRate / 2)
    let edges = (0..<(c.bands + 2)).map { hz(lo + (hi - lo) * Float($0) / Float(c.bands + 1)) }
    var m = [Float](repeating: 0, count: c.bins * c.bands)
    for b in 0..<c.bands {
      let l = edges[b], ctr = edges[b + 1], r = edges[b + 2]
      var any = false
      for k in 0..<c.bins {
        let f = Float(k) * c.sampleRate / Float(c.window)
        let w = Swift.max(0, Swift.min((f - l) / (ctr - l), (r - f) / (r - ctr)))
        m[k * c.bands + b] = w
        any = any || w > 0
      }
      if !any {
        // narrower than a bin: take the nearest bin
        let k = Swift.min(c.bins - 1, Int((ctr * Float(c.window) / c.sampleRate).rounded()))
        m[k * c.bands + b] = 1
      }
    }
    return m
  }

  /// Unnormalised log-mel features of onset-aligned audio, [bands, frames] flattened.
  /// Thread-safe (the FFT setup is only read).
  func logMel(_ x: [Float]) -> [Float] {
    let c = config
    let n = c.window, half = n / 2, T = c.frames
    let avail = min(x.count, c.cropSamples)
    var frame = [Float](repeating: 0, count: n)
    var real = [Float](repeating: 0, count: half)
    var imag = [Float](repeating: 0, count: half)
    var mags = [Float](repeating: 0, count: T * c.bins)  // [frames, bins]
    for t in 0..<T {
      let start = t * c.hop
      let m = max(0, min(n, avail - start))
      for i in 0..<n { frame[i] = i < m ? x[start + i] : 0 }
      vDSP.multiply(frame, hann, result: &frame)
      real.withUnsafeMutableBufferPointer { rp in
        imag.withUnsafeMutableBufferPointer { ip in
          var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
          frame.withUnsafeBufferPointer { input in
            input.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
              vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
            }
          }
          vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
        }
      }
      // zrip packs DC in real[0] and Nyquist in imag[0], and scales by 2.
      let row = t * c.bins
      mags[row] = Swift.abs(real[0]) / 2
      mags[row + half] = Swift.abs(imag[0]) / 2
      for k in 1..<half { mags[row + k] = (real[k] * real[k] + imag[k] * imag[k]).squareRoot() / 2 }
    }
    // [frames, bins] @ [bins, bands] → [frames, bands], then log and transpose to [bands, frames]
    var mel = [Float](repeating: 0, count: T * c.bands)
    vDSP_mmul(mags, 1, melFilters, 1, &mel, 1, vDSP_Length(T), vDSP_Length(c.bands), vDSP_Length(c.bins))
    var out = [Float](repeating: 0, count: c.width)
    for t in 0..<T {
      for b in 0..<c.bands { out[b * T + t] = Foundation.log(mel[t * c.bands + b] + c.eps) }
    }
    return out
  }
}

/// Per-band normalisation statistics over a set of feature rows.
struct EncoderFeatureStats {
  var mean: [Float]
  var std: [Float]

  static func fit(_ rows: [[Float]], bands: Int, frames: Int) -> EncoderFeatureStats {
    var mean = [Double](repeating: 0, count: bands)
    var sq = [Double](repeating: 0, count: bands)
    let n = Double(max(1, rows.count * frames))
    for r in rows {
      for b in 0..<bands {
        for t in 0..<frames {
          let v = Double(r[b * frames + t])
          mean[b] += v
          sq[b] += v * v
        }
      }
    }
    var m = [Float](repeating: 0, count: bands), s = m
    for b in 0..<bands {
      let mu = mean[b] / n
      m[b] = Float(mu)
      s[b] = Float(Swift.max(sq[b] / n - mu * mu, 1e-8).squareRoot())
    }
    return EncoderFeatureStats(mean: m, std: s)
  }

  func apply(_ row: [Float], bands: Int, frames: Int) -> [Float] {
    var y = row
    for b in 0..<bands {
      for t in 0..<frames { y[b * frames + t] = (row[b * frames + t] - mean[b]) / std[b] }
    }
    return y
  }

  func write(_ url: URL, config: EncoderFeatureConfig) throws {
    var d = config.json
    d["mean"] = mean.map(Double.init)
    d["std"] = std.map(Double.init)
    try JSONSerialization.data(withJSONObject: d, options: [.prettyPrinted, .sortedKeys]).write(to: url)
  }

  static func load(_ url: URL) throws -> (EncoderFeatureStats, EncoderFeatureConfig) {
    guard let d = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
      let mean = d["mean"] as? [NSNumber], let std = d["std"] as? [NSNumber],
      let sr = d["sample_rate"] as? NSNumber
    else { throw TrainProtocolError("feature stats \(url.path): expected {mean, std, sample_rate, …}") }
    var c = EncoderFeatureConfig(sampleRate: sr.floatValue)
    c.window = (d["window"] as? NSNumber)?.intValue ?? c.window
    c.hop = (d["hop"] as? NSNumber)?.intValue ?? c.hop
    c.bands = (d["bands"] as? NSNumber)?.intValue ?? c.bands
    c.seconds = (d["seconds"] as? NSNumber)?.floatValue ?? c.seconds
    c.eps = (d["eps"] as? NSNumber)?.floatValue ?? c.eps
    c.fMin = (d["f_min"] as? NSNumber)?.floatValue ?? c.fMin
    return (EncoderFeatureStats(mean: mean.map(\.floatValue), std: std.map(\.floatValue)), c)
  }
}

/// Host-side training augmentation of one prepared hit (§3): gain, onset jitter, pitch
/// (resampling), spectral tilt, additive noise. Deterministic given the RNG.
enum EncoderAugment {
  static func apply(_ x: [Float], sampleRate: Float, rng: inout SplitMix64) -> (audio: [Float], tiltDB: Float) {
    func u(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * rng.uniform() }
    let gain = Foundation.pow(10, u(-6, 6) / 20)
    let jitter = Int((u(-0.003, 0.003) * sampleRate).rounded())
    let ratio = Foundation.pow(2, u(-1, 1) / 12)  // ±1 semitone: read position step
    let noise = Foundation.pow(10, u(-60, -35) / 20)
    let tilt = u(-6, 6)
    let n = x.count
    var y = [Float](repeating: 0, count: n)
    for i in 0..<n {
      let p = Float(i - jitter) * ratio
      let k = Int(Foundation.floor(p))
      guard k >= 0, k + 1 < n else { continue }
      let f = p - Float(k)
      y[i] = (x[k] * (1 - f) + x[k + 1] * f) * gain
    }
    for i in 0..<n { y[i] += noise * rng.gaussian() }
    return (y, tilt)
  }

  /// Spectral tilt applied in the log-mel domain: tiltDB across the band range (linear in
  /// band index), as a log-magnitude offset.
  static func tilt(_ f: inout [Float], dB: Float, bands: Int, frames: Int) {
    guard bands > 1 else { return }
    for b in 0..<bands {
      let off = (Float(b) / Float(bands - 1) - 0.5) * dB / 20 * Foundation.log(10)
      for t in 0..<frames { f[b * frames + t] += off }
    }
  }
}

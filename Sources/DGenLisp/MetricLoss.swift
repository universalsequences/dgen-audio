// MetricLoss.swift — the kick track's acceptance metrics as differentiable CPU losses.
//
// band:  32 log bands 25 Hz–20 kHz (bin edges of a 48 kHz, 2048-point grid, as the reference
//        metric defines them), Hann W=2048 hop 256 after a W/2 zero pre-pad, band power =
//        mean of |X|² over the band's bins, 4-frame box smoothing ('same'), 10·log10(·+1e-12),
//        both clamped at (target max − 70 dB); score = mean |P − T| over frames × bands.
// mr:    multi-resolution STFT, Hann, hop w/4, mean over bins of |√P − √T| + |½log(P+ε²) −
//        ½log(T+ε²)|, ε = 1e-3, averaged over the windows.
//
// `lossAndGradient` returns both scores and d(wBand·band + wMR·mr)/dy. The trainer injects the
// gradient into the voice graph as Σ y·g, so any loss computed here trains exactly.

import Accelerate
import Foundation

struct MetricLoss {
  let windows: [Int]
  let bandWeight: Double
  let mrWeight: Double
  var waveWeight: Double = 0
  var waveCutoff: Double = 600
  var sampleRate: Double = 44100
  static let eps = 1e-3

  // MARK: transforms

  /// Complex DFT of length w (forward or inverse, unnormalised) on split real/imag arrays.
  final class DFT {
    let w: Int
    private let fwd: vDSP.DFT<Double>
    private let inv: vDSP.DFT<Double>
    let hann: [Double]
    init(_ w: Int) {
      self.w = w
      fwd = vDSP.DFT(count: w, direction: .forward, transformType: .complexComplex, ofType: Double.self)!
      inv = vDSP.DFT(count: w, direction: .inverse, transformType: .complexComplex, ofType: Double.self)!
      hann = (0..<w).map { 0.5 * (1 - cos(2 * Double.pi * Double($0) / Double(w))) }
    }
    func forward(_ re: [Double]) -> (re: [Double], im: [Double]) {
      var oR = [Double](repeating: 0, count: w), oI = [Double](repeating: 0, count: w)
      fwd.transform(inputReal: re, inputImaginary: [Double](repeating: 0, count: w),
                    outputReal: &oR, outputImaginary: &oI)
      return (oR, oI)
    }
    /// Unnormalised inverse DFT (real part) of a full spectrum.
    func adjointFull(_ gR: [Double], _ gI: [Double]) -> [Double] {
      var oR = [Double](repeating: 0, count: w), oI = [Double](repeating: 0, count: w)
      inv.transform(inputReal: gR, inputImaginary: gI, outputReal: &oR, outputImaginary: &oI)
      return oR
    }
    /// Re(Σ_k G_k e^{+2πikm/w}) for G given on k = 0…w/2 (zero above).
    func adjoint(_ gR: [Double], _ gI: [Double]) -> [Double] {
      var iR = [Double](repeating: 0, count: w), iI = [Double](repeating: 0, count: w)
      for k in 0..<gR.count { iR[k] = gR[k]; iI[k] = gI[k] }
      var oR = [Double](repeating: 0, count: w), oI = [Double](repeating: 0, count: w)
      inv.transform(inputReal: iR, inputImaginary: iI, outputReal: &oR, outputImaginary: &oI)
      return oR
    }
  }

  private static var cache: [Int: DFT] = [:]
  private static let lock = NSLock()
  static func dft(_ w: Int) -> DFT {
    lock.lock(); defer { lock.unlock() }
    if let d = cache[w] { return d }
    let d = DFT(w)
    cache[w] = d
    return d
  }

  // MARK: band metric constants

  static let bandW = 2048
  static let bandHop = 256
  /// [bands][bins] membership (empty bands dropped), and bin counts.
  static let bandMatrix: (rows: [[Int]], count: Int) = {
    let binHz: Double = 48000 / Double(bandW)
    let fr: [Double] = (0...(bandW / 2)).map { (k: Int) -> Double in Double(k) * binHz }
    let l0: Double = log(25.0), l1: Double = log(20000.0)
    let edges: [Double] = (0...32).map { (k: Int) -> Double in exp(l0 + (l1 - l0) * Double(k) / 32) }
    var rows: [[Int]] = []
    for b in 0..<32 {
      let bins = fr.indices.filter { fr[$0] >= edges[b] && fr[$0] < edges[b + 1] }
      if !bins.isEmpty { rows.append(bins) }
    }
    return (rows, rows.count)
  }()

  // MARK: spectra

  /// Power spectra of frames starting at `starts`, plus the complex bins (for the gradient).
  struct Spectra {
    var P: [[Double]]  // [J][w/2+1]
    var re: [[Double]]
    var im: [[Double]]
  }

  static func spectra(_ x: [Double], w: Int, starts: [Int]) -> Spectra {
    let d = dft(w)
    let nb = w / 2 + 1
    var s = Spectra(P: [], re: [], im: [])
    s.P.reserveCapacity(starts.count)
    for st in starts {
      var frame = [Double](repeating: 0, count: w)
      for m in 0..<w { frame[m] = x[st + m] * d.hann[m] }
      let (r, i) = d.forward(frame)
      s.re.append(Array(r[0..<nb]))
      s.im.append(Array(i[0..<nb]))
      s.P.append((0..<nb).map { r[$0] * r[$0] + i[$0] * i[$0] })
    }
    return s
  }

  // MARK: MR-STFT

  /// MR score and its gradient w.r.t. y.
  static func mr(_ y: [Double], _ t: [Double], windows: [Int]) -> (Double, [Double]) {
    let n = min(y.count, t.count)
    var grad = [Double](repeating: 0, count: n)
    var total = 0.0
    let ws = windows.filter { $0 <= n }
    for w in ws {
      let hop = w / 4
      let starts = Array(stride(from: 0, through: n - w, by: hop))
      let nb = w / 2 + 1
      let sy = spectra(y, w: w, starts: starts)
      let stt = spectra(t, w: w, starts: starts)
      let norm = 1.0 / Double(starts.count * nb) / Double(ws.count)
      let d = dft(w)
      let e2 = eps * eps
      for (j, st) in starts.enumerated() {
        var gR = [Double](repeating: 0, count: nb), gI = [Double](repeating: 0, count: nb)
        for k in 0..<nb {
          let P = sy.P[j][k], T = stt.P[j][k]
          let A = P.squareRoot(), TA = T.squareRoot()
          let lg = 0.5 * log(P + e2) - 0.5 * log(T + e2)
          total += (abs(A - TA) + abs(lg)) * norm
          // dL/dA, then G = dL/dA · X/A ; the log term's dL/dX = sign · X/(P+ε²)
          let sLin = A > TA ? 1.0 : (A < TA ? -1.0 : 0.0)
          let sLog = lg > 0 ? 1.0 : (lg < 0 ? -1.0 : 0.0)
          let cLin = A > 0 ? sLin / A : 0
          let c = (cLin + sLog / (P + e2)) * norm
          gR[k] = c * sy.re[j][k]
          gI[k] = c * sy.im[j][k]
        }
        let back = d.adjoint(gR, gI)
        for m in 0..<w { grad[st + m] += back[m] * d.hann[m] }
      }
    }
    return (total, grad)
  }

  // MARK: band envelope

  static func bandEnvDB(_ x: [Double]) -> (D: [[Double]], E: [[Double]], Es: [[Double]], sp: Spectra, starts: [Int]) {
    let W = bandW, hop = bandHop
    let xp = [Double](repeating: 0, count: W / 2) + x
    let starts = Array(stride(from: 0, through: xp.count - W, by: hop))
    let sp = spectra(xp, w: W, starts: starts)
    let (rows, nbands) = bandMatrix
    let E: [[Double]] = sp.P.map { P in rows.map { bins in bins.reduce(0) { $0 + P[$1] } / Double(bins.count) } }
    let J = E.count
    var Es = [[Double]](repeating: [Double](repeating: 0, count: nbands), count: J)
    for j in 0..<J {
      for b in 0..<nbands {
        var s = 0.0
        for jj in [j + 1, j, j - 1, j - 2] where jj >= 0 && jj < J { s += E[jj][b] }
        Es[j][b] = s / 4
      }
    }
    let D = Es.map { $0.map { 10 * log10($0 + 1e-12) } }
    return (D, E, Es, sp, starts)
  }

  static func band(_ y: [Double], _ t: [Double], floorDB: Double = 70) -> (Double, [Double]) {
    let n = min(y.count, t.count)
    let Y = bandEnvDB(Array(y[0..<n]))
    let T = bandEnvDB(Array(t[0..<n]))
    let lo = (T.D.flatMap { $0 }.max() ?? 0) - floorDB
    let J = Y.D.count, nbands = bandMatrix.count
    let norm = 1.0 / Double(J * nbands)
    var score = 0.0
    var gEs = [[Double]](repeating: [Double](repeating: 0, count: nbands), count: J)
    for j in 0..<J {
      for b in 0..<nbands {
        let p = max(Y.D[j][b], lo), q = max(T.D[j][b], lo)
        score += abs(p - q) * norm
        guard Y.D[j][b] > lo, p != q else { continue }
        let s = p > q ? 1.0 : -1.0
        gEs[j][b] = s * norm * 10 / (log(10.0) * (Y.Es[j][b] + 1e-12))
      }
    }
    // smoothing transpose: Es[j] uses E[j+1], E[j], E[j-1], E[j-2]
    var gE = [[Double]](repeating: [Double](repeating: 0, count: nbands), count: J)
    for j in 0..<J {
      for b in 0..<nbands {
        var s = 0.0
        for jj in [j - 1, j, j + 1, j + 2] where jj >= 0 && jj < J { s += gEs[jj][b] }
        gE[j][b] = s / 4
      }
    }
    let W = bandW
    let d = dft(W)
    let nb = W / 2 + 1
    var gpad = [Double](repeating: 0, count: n + W / 2)
    let rows = bandMatrix.rows
    for (j, st) in Y.starts.enumerated() {
      var gP = [Double](repeating: 0, count: nb)
      for (b, bins) in rows.enumerated() where gE[j][b] != 0 {
        let c = gE[j][b] / Double(bins.count)
        for k in bins { gP[k] += c }
      }
      var gR = [Double](repeating: 0, count: nb), gI = [Double](repeating: 0, count: nb)
      for k in 0..<nb where gP[k] != 0 {
        gR[k] = 2 * gP[k] * Y.sp.re[j][k]
        gI[k] = 2 * gP[k] * Y.sp.im[j][k]
      }
      let back = d.adjoint(gR, gI)
      for m in 0..<W { gpad[st + m] += back[m] * d.hann[m] }
    }
    return (score, Array(gpad[(W / 2)...]))
  }

  // MARK: low-band waveform

  /// Phase-aware low-band fit: sum((LP y - LP t)^2) / sum((LP t)^2) with a zero-phase FFT low-pass
  /// (brick-wall at `cutoff`, raised-cosine edge one octave wide). LP is linear and self-adjoint,
  /// so the gradient is 2 LP(LP y - LP t) / sum((LP t)^2).
  static func lowWave(_ y: [Double], _ t: [Double], cutoff: Double, sampleRate: Double) -> (Double, [Double]) {
    let n = min(y.count, t.count)
    var m = 1
    while m < 2 * n { m <<= 1 }
    let d = dft(m)
    let mask: [Double] = (0..<m).map { k in
      let f = Double(min(k, m - k)) * sampleRate / Double(m)
      if f <= cutoff { return 1 }
      if f >= 2 * cutoff { return 0 }
      return 0.5 * (1 + cos(Double.pi * (f - cutoff) / cutoff))
    }
    func lp(_ x: [Double]) -> [Double] {
      var pad = [Double](repeating: 0, count: m)
      for i in 0..<n { pad[i] = x[i] }
      let (r, im) = d.forward(pad)
      let gr = (0..<m).map { r[$0] * mask[$0] }, gi = (0..<m).map { im[$0] * mask[$0] }
      return Array(d.adjointFull(gr, gi).prefix(n)).map { $0 / Double(m) }
    }
    let ly = lp(y), lt = lp(t)
    let e = max(lt.reduce(0) { $0 + $1 * $1 }, 1e-12)
    let diff = zip(ly, lt).map { $0 - $1 }
    let loss = diff.reduce(0) { $0 + $1 * $1 } / e
    let g = lp(diff).map { 2 * $0 / e }
    return (loss, g)
  }

  // MARK: combined

  struct Result {
    var band: Double
    var mr: Double
    var loss: Double
    var grad: [Float]
    var wave: Double = 0
  }

  func lossAndGradient(_ y: [Float], _ t: [Float]) -> Result {
    let yd = y.map(Double.init), td = t.map(Double.init)
    let (b, gb) = bandWeight > 0 ? MetricLoss.band(yd, td) : (0, [Double](repeating: 0, count: y.count))
    let (m, gm) = mrWeight > 0 ? MetricLoss.mr(yd, td, windows: windows) : (0, [Double](repeating: 0, count: y.count))
    let (w, gw) = waveWeight > 0
      ? MetricLoss.lowWave(yd, td, cutoff: waveCutoff, sampleRate: sampleRate)
      : (0, [Double](repeating: 0, count: y.count))
    var g = [Float](repeating: 0, count: y.count)
    for i in 0..<min(gb.count, g.count) { g[i] += Float(bandWeight * gb[i]) }
    for i in 0..<min(gm.count, g.count) { g[i] += Float(mrWeight * gm[i]) }
    for i in 0..<min(gw.count, g.count) { g[i] += Float(waveWeight * gw[i]) }
    return Result(band: b, mr: m, loss: bandWeight * b + mrWeight * m + waveWeight * w, grad: g, wave: w)
  }

  /// Scores only (both terms), for evaluation.
  static func scores(_ y: [Float], _ t: [Float], windows: [Int]) -> (band: Double, mr: Double) {
    let yd = y.map(Double.init), td = t.map(Double.init)
    return (band(yd, td).0, mr(yd, td, windows: windows).0)
  }
}

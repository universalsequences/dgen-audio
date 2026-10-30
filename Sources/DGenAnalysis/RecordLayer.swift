// The v3 voice's RECORD layer pieces that the v1/v2 analysis did not have: crackle (sparse
// heavy-tailed impulses through 4 band-passes) and noise events (bursts on the hiss bands).
// CPU references match voice3.lisp exactly (up to the random realisation):
//
//   crackle  imp[n] = [u_n < rate/sr] · min(30, 1/sqrt(max(a_n, 0.001))) · (s_n > 0.5 ? 1 : -1)
//            band_k = RBJ band-pass (constant 0 dB peak) at fc_k ∈ {1000, 2500, 6000, 12000} Hz,
//                     alpha = sin(w0)/1.6 (Q = 0.8), direct form I, zero state
//            y_cr   = cr_lvl · e^{-t/cr_dec} · Σ_k cr_col_k band_k(imp)       (after the clipper)
//   event e  env_e(t) = ev_lvl · [t > ev_t] (1 - e^{-(t-ev_t)/ev_rise}) e^{-(t-ev_t)/ev_dec}
//            y_ev   = env_e(t) · Σ_b ev_col_eb band_b(t)    (band_b: the hiss bands, same white)
//
// Crackle amplitude law: P(a > x) = 1/x² on [1, 30] (a Pareto tail with index 2, capped), so
// E[a²] = E[min(900, 1/u)] = 1 + ln 900 and the expected crackle power at t = 0 is
//   cr_lvl² · (rate/sr) · (1 + ln 900) · Σ_n h_col[n]²,   h_col = Σ_k cr_col_k h_k.

import Foundation

public struct NoiseEventParams: Codable, Equatable {
    public var t: Double
    public var level: Double
    public var decay: Double
    public var rise: Double
    public var colour: [Double]  // one weight per hiss band, max 1
    /// "secondary" (a later onset) or "onset" (at the hit, spectrally distinct from the hiss)
    public var kind: String
    /// relative fall of the band-power fit's dB RMS when the event was admitted
    public var gain: Double
    public init(t: Double, level: Double, decay: Double, rise: Double, colour: [Double], kind: String, gain: Double) {
        self.t = t; self.level = level; self.decay = decay; self.rise = rise; self.colour = colour
        self.kind = kind; self.gain = gain
    }
    @inline(__always) public func envelope(_ tt: Double) -> Double {
        let d = tt - t
        if d <= 0 { return 0 }
        return level * (1 - exp(-d / max(rise, 1e-7))) * exp(-d / decay)
    }
}

public struct CrackleParams: Codable, Equatable {
    public var level: Double
    public var rate: Double      // impulses per second
    public var colour: [Double]  // 4 band weights (1, 2.5, 6, 12 kHz), max 1
    public var decay: Double
    public var present: Bool
    public var detected: Int
    public var scannedSeconds: Double
    /// detection threshold in local robust σ
    public var threshold: Double
    public var clickTimes: [Double]?
    public var clickPeaks: [Double]?
    /// clicks counted for the rate (peak >= θ*), over eligibleSeconds (Pareto-corrected detectable time)
    public var counted: Int? = nil
    public var eligibleSeconds: Double? = nil
    /// level that would match the counted clicks' energy in expectation (diagnostic)
    public var levelFromEnergy: Double? = nil
    /// clicks rejected as phase-locked to the fundamental
    public var phaseLocked: Int? = nil
    public init(level: Double, rate: Double, colour: [Double], decay: Double, present: Bool, detected: Int,
                scannedSeconds: Double, threshold: Double, clickTimes: [Double]? = nil, clickPeaks: [Double]? = nil) {
        self.level = level; self.rate = rate; self.colour = colour; self.decay = decay; self.present = present
        self.detected = detected; self.scannedSeconds = scannedSeconds; self.threshold = threshold
        self.clickTimes = clickTimes; self.clickPeaks = clickPeaks
    }
}

public enum Crackle {
    public static let centres = [1000.0, 2500.0, 6000.0, 12000.0]
    public static let meanSquareAmplitude = 1 + log(900.0)
    public static let silentLevel = 1e-5
    public static let nominalRate = 20.0
    public static let defaultDecay = 10.0

    public static func silent(scanned: Double = 0) -> CrackleParams {
        CrackleParams(level: silentLevel, rate: nominalRate, colour: [1, 1, 1, 1], decay: defaultDecay,
                      present: false, detected: 0, scannedSeconds: scanned, threshold: 0)
    }

    public static func biquad(_ fc: Double, sampleRate sr: Double) -> NoiseBands.Biquad {
        let w0 = 2 * Double.pi * min(fc, 0.49 * sr) / sr
        let alpha = sin(w0) / 1.6
        let a0 = 1 + alpha
        return NoiseBands.Biquad(b0: alpha / a0, b1: 0, b2: -alpha / a0, a1: -2 * cos(w0) / a0, a2: (1 - alpha) / a0)
    }

    /// h_col = Σ_k c_k h_k (first n samples).
    public static func impulseResponse(colour: [Double], sampleRate sr: Double, count n: Int = 4096) -> [Double] {
        var imp = [Double](repeating: 0, count: n)
        imp[0] = 1
        var h = [Double](repeating: 0, count: n)
        for (k, fc) in centres.enumerated() where k < colour.count && colour[k] != 0 {
            let y = NoiseBands.filter(imp, biquad(fc, sampleRate: sr))
            for i in 0..<n { h[i] += colour[k] * y[i] }
        }
        return h
    }

    /// Σ h_col² weighted by an optional zero-phase gain |g(f)|² (frequency domain).
    public static func powerGain(colour: [Double], sampleRate sr: Double, weight: ((Double) -> Double)? = nil) -> Double {
        let n = 8192
        let h = impulseResponse(colour: colour, sampleRate: sr, count: n)
        guard let g = weight else { return h.reduce(0) { $0 + $1 * $1 } }
        var re = h, im = [Double](repeating: 0, count: n)
        FFT.transform(&re, &im)
        var s = 0.0
        for k in 0..<n {
            let kk = k <= n / 2 ? k : k - n
            let w = g(abs(Double(kk)) * sr / Double(n))
            s += (re[k] * re[k] + im[k] * im[k]) * w * w
        }
        return s / Double(n)
    }

    /// Expected crackle power at t = 0 (per sample).
    public static func expectedPower(_ p: CrackleParams, sampleRate sr: Double) -> Double {
        p.level * p.level * p.rate / sr * meanSquareAmplitude * powerGain(colour: p.colour, sampleRate: sr)
    }

    /// CPU reference of voice3's crackle (own xorshift64* stream; the realisation differs from dgen's).
    public static func render(_ p: CrackleParams, count n: Int, sampleRate sr: Double, seed: UInt64 = 0x51ED) -> [Double] {
        var s = seed == 0 ? 1 : seed
        @inline(__always) func uni() -> Double {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27
            return Double((s &* 2685821657736338717) >> 11) / Double(1 << 53)
        }
        let prob = p.rate / sr
        var imp = [Double](repeating: 0, count: n)
        for i in 0..<n {
            let u = uni(), a = uni(), sg = uni()
            if u < prob { imp[i] = min(30, 1 / max(a, 0.001).squareRoot()) * (sg > 0.5 ? 1 : -1) }
        }
        var y = [Double](repeating: 0, count: n)
        for (k, fc) in centres.enumerated() where k < p.colour.count && p.colour[k] != 0 {
            let b = NoiseBands.filter(imp, biquad(fc, sampleRate: sr))
            for i in 0..<n { y[i] += p.colour[k] * b[i] }
        }
        for i in 0..<n { y[i] *= p.level * exp(-Double(i) / sr / p.decay) }
        return y
    }

    public struct FitOptions {
        public var highpassHz = 800.0
        /// detection threshold in local robust σ (MAD of the high-passed residual, ±scaleWindow)
        public var threshold = 5.5
        public var scaleWindow = 0.01
        /// scan from onset + this (the attack's own transients and strikes live before it)
        public var startAfterOnset = 0.03
        public var endMargin = 0.01
        public var minSeparation = 0.001
        public var maskBefore = 0.0005
        public var maskAfter = 0.0015
        public var minClicks = 3
        public var minQuietClicks = 2
        public var minRate = 3.0
        public init() {}
    }

    /// Detect crackle impulses in the residual and estimate (rate, colour, level).
    ///
    /// Model identity: θ* = the median over the scan of the local threshold (threshold · robust σ of
    /// the high-passed residual). Clicks with peak >= θ* are the generator's impulses and the lowest
    /// of them, s, its unit peak: rate = their count over the time they were detectable
    /// (Σ_t min(1, (s/θ(t))²) under the Pareto tail where the local threshold θ(t) exceeds s), and
    /// the level puts
    /// the generator's unit impulse (a = 1) at peak s after the same high-pass, so its Pareto tail
    /// (P(a > x) = 1/x²) reproduces the peak distribution above s; weaker crackle is buried in the
    /// hiss either way. Colour: a 4-weight fit of |Σ c_k H_k|² to the clicks' average power spectrum
    /// (minus nearby non-click spectra), in dB.
    /// Returns a sample mask (0 on the clicks, tapered) to gate them out of the hiss measurement.
    public static func fit(residual r: [Double], sr: Double, onset: Double, end: Double,
                           fundamental: ModeParams? = nil,
                           options o: FitOptions = FitOptions()) -> (params: CrackleParams, mask: [Double])
    {
        let n = r.count
        var mask = [Double](repeating: 1, count: n)
        let i0 = Int((onset + o.startAfterOnset) * sr)
        // stop short of the end: a chopped sample (or a residual step there) rings through the high-pass
        let i1 = min(n, Int((end - o.endMargin) * sr))
        guard i1 - i0 > Int(0.02 * sr) else { return (silent(), mask) }
        let hp = FFT.bandGain(lo: o.highpassHz, hi: 0, taper: 0.25)
        let y = FFT.spectralFilter(r, sampleRate: sr, analytic: false, gain: hp).re
        // local robust scale on a 1 ms grid
        let step = max(1, Int(0.001 * sr)), W = Int(o.scaleWindow * sr)
        let grid = Swift.stride(from: i0, to: i1, by: step).map { $0 }
        let sig = grid.map { c -> Double in
            let a = max(0, c - W), b = min(n, c + W)
            let v = y[a..<b].map { abs($0) }.sorted()
            return max(v[v.count / 2] / 0.6745, 1e-12)
        }
        func sigma(_ i: Int) -> Double {
            let g = Double(i - i0) / Double(step)
            let k = min(max(Int(g), 0), grid.count - 1), k1 = min(k + 1, grid.count - 1)
            let f = min(max(g - Double(k), 0), 1)
            return sig[k] * (1 - f) + sig[k1] * f
        }
        let sep = max(1, Int(o.minSeparation * sr))
        var clicks: [Int] = []
        var i = i0
        while i < i1 {
            if abs(y[i]) > o.threshold * sigma(i) {
                // the click is the largest |y| within the next `sep` samples
                var j = i
                for k in i..<min(n, i + sep + 1) where abs(y[k]) > abs(y[j]) { j = k }
                if clicks.last.map({ j - $0 >= sep }) ?? true { clicks.append(j) }
                i = j + sep
                continue
            }
            i += 1
        }
        let scanned = Double(i1 - i0) / sr
        // clicks phase-locked to the fundamental (once or twice per cycle: the record's clipping /
        // the kick's own wave edges) are not crackle: Rayleigh test on the slot-0 phase at the
        // clicks, drop those within ±60° of the mean phase when the lock is strong
        var lockedOut = 0
        if let f0 = fundamental, clicks.count >= 4 {
            for order in [1.0, 2.0] {
                let ph = clicks.map { order * (f0.phaseLaw(Double($0) / sr) + f0.phi) }
                let c = ph.reduce(0) { $0 + cos($1) } / Double(ph.count)
                let sn = ph.reduce(0) { $0 + sin($1) } / Double(ph.count)
                guard (c * c + sn * sn).squareRoot() > 0.5 else { continue }
                let mean = atan2(sn, c)
                let before = clicks.count
                clicks = clicks.indices.filter { abs(remainder(ph[$0] - mean, 2 * Double.pi)) > Double.pi / 3 }
                    .map { clicks[$0] }
                lockedOut += before - clicks.count
                if clicks.count < 4 { break }
            }
        }
        guard clicks.count >= o.minClicks else {
            var p = silent(scanned: scanned)
            p.phaseLocked = lockedOut
            return (p, mask)
        }
        let peaks = clicks.map { abs(y[$0]) }
        // common absolute threshold θ* = median of the local thresholds. A record's crackle is
        // stationary, so it must show in the quiet half of the scan (local threshold <= θ*), where
        // it is easiest to see: clicks only in the loud early part are the attack's own transients.
        // The generator's unit peak s is the lowest quiet-half click (>= θ*): = θ* when crackle
        // continues under the hiss, higher when it all stands clear of it.
        let thGrid = sig.map { o.threshold * $0 }
        let thStar = thGrid.sorted()[thGrid.count / 2]
        let quiet = clicks.indices.filter { o.threshold * sigma(clicks[$0]) <= thStar }
        guard quiet.count >= o.minQuietClicks else {
            var p = silent(scanned: scanned)
            p.phaseLocked = lockedOut
            p.detected = clicks.count
            return (p, mask)
        }
        let sUnit = max(thStar, quiet.map { peaks[$0] }.min()!)
        let above = clicks.indices.filter { peaks[$0] >= sUnit }
        guard above.count >= o.minClicks else { return (silent(scanned: scanned), mask) }
        // impulses with peak >= s are detectable at t when also >= θ(t): under the Pareto tail
        // a fraction min(1, (s/θ(t))²) of them
        var eligible = 0.0
        for th in thGrid { eligible += min(1, (sUnit / th) * (sUnit / th)) }
        eligible *= Double(step) / sr
        let counted = above
        let rate = Double(counted.count) / max(eligible, 1e-3)
        guard rate >= o.minRate else { return (silent(scanned: scanned), mask) }
        // click energy (noise-floor corrected) and spectra
        let wb = Int(o.maskBefore * sr), wa = Int(o.maskAfter * sr)
        var eClicks = 0.0
        let nfft = 256, seg = 128
        var specC = [Double](repeating: 0, count: nfft / 2 + 1), specN = specC
        let win = hann(seg)
        func spectrum(at c: Int) -> [Double]? {
            let a = c - seg / 4
            guard a >= 0 && a + seg <= n else { return nil }
            var re = [Double](repeating: 0, count: nfft), im = re
            for k in 0..<seg { re[k] = y[a + k] * win[k] }
            FFT.transform(&re, &im)
            return (0...(nfft / 2)).map { re[$0] * re[$0] + im[$0] * im[$0] }
        }
        var nSpec = 0, nNoise = 0
        for c in clicks {
            let a = max(0, c - wb), b = min(n, c + wa + 1)
            var e = 0.0
            for k in a..<b { e += y[k] * y[k] }
            let s = sigma(c)
            eClicks += max(0, e - Double(b - a) * s * s)
            if let sp = spectrum(at: c) { for k in sp.indices { specC[k] += sp[k] }; nSpec += 1 }
            for off in [-Int(0.005 * sr), Int(0.005 * sr)] {
                let c2 = c + off
                if clicks.contains(where: { abs($0 - c2) < 2 * wa }) { continue }
                if let sp = spectrum(at: c2) { for k in sp.indices { specN[k] += sp[k] }; nNoise += 1 }
            }
        }
        var P = [Double](repeating: 0, count: specC.count)
        for k in P.indices {
            let c = specC[k] / Double(max(nSpec, 1)), nz = nNoise > 0 ? specN[k] / Double(nNoise) : 0
            P[k] = max(c - nz, 0.02 * c)
        }
        // colour: |Σ c_k H_k(f)|² hp(f)² ≈ P(f) over 600 Hz .. 0.45 sr, in dB
        let bqs = centres.map { biquad($0, sampleRate: sr) }
        let fk = (0..<P.count).map { Double($0) * sr / Double(nfft) }
        let use = fk.indices.filter { fk[$0] >= 600 && fk[$0] <= 0.45 * sr }
        func H(_ b: NoiseBands.Biquad, _ f: Double) -> (Double, Double) {
            let w = 2 * Double.pi * f / sr
            let nr = b.b0 + b.b1 * cos(w) + b.b2 * cos(2 * w), ni = -(b.b1 * sin(w) + b.b2 * sin(2 * w))
            let dr = 1 + b.a1 * cos(w) + b.a2 * cos(2 * w), di = -(b.a1 * sin(w) + b.a2 * sin(2 * w))
            let d = dr * dr + di * di
            return ((nr * dr + ni * di) / d, (ni * dr - nr * di) / d)
        }
        let Hk = use.map { k in bqs.map { H($0, fk[k]) } }
        let pmax = use.map { P[$0] }.max() ?? 1
        let floorP = pmax * 1e-4
        let fit = LinAlg.levenbergMarquardt(x0: [Double](repeating: log(pmax.squareRoot()), count: 4),
                                            lower: [Double](repeating: log(pmax.squareRoot()) - 12, count: 4),
                                            upper: [Double](repeating: log(pmax.squareRoot()) + 6, count: 4),
                                            maxIter: 60) { q in
            let c = q.map { exp($0) }
            return use.indices.map { j in
                var re = 0.0, im = 0.0
                for k in 0..<4 { re += c[k] * Hk[j][k].0; im += c[k] * Hk[j][k].1 }
                let g = hp(fk[use[j]])
                return 10 * log10((re * re + im * im) * g * g + floorP) - 10 * log10(P[use[j]] + floorP)
            }
        }
        var colour = fit.x.map { exp($0) }
        let cmax = colour.max() ?? 1
        colour = colour.map { min(max($0 / cmax, 0.001), 1) }
        // level: the generator's unit impulse (a = 1) peaks at s after the same high-pass, so its
        // Pareto tail reproduces the counted clicks' peak distribution above s. (The energy-matched
        // level is kept as a diagnostic: the heavy tail makes the measured energy unreliable.)
        let hcol = impulseResponse(colour: colour, sampleRate: sr, count: 4096)
        let hpk = FFT.spectralFilter(hcol, sampleRate: sr, analytic: false, gain: hp).re.map { abs($0) }.max() ?? 1
        let level = sUnit / max(hpk, 1e-12)
        let Pmeas = eClicks / Double(i1 - i0)
        let G = powerGain(colour: colour, sampleRate: sr, weight: hp)
        let energyLevel = (Pmeas / max(rate / sr * meanSquareAmplitude * G, 1e-30)).squareRoot()
        for c in clicks {
            let a = c - wb, b = c + wa
            let taper = max(1, Int(0.00025 * sr))
            for k in (a - taper)...(b + taper) where k >= 0 && k < n {
                var m = 0.0
                if k < a { m = 0.5 + 0.5 * cos(Double.pi * Double(k - (a - taper)) / Double(taper)) }
                if k > b { m = 0.5 - 0.5 * cos(Double.pi * Double(k - b) / Double(taper)) }
                mask[k] = min(mask[k], m)
            }
        }
        var params = CrackleParams(level: max(level, silentLevel), rate: min(max(rate, 1), 4000), colour: colour,
                                   decay: defaultDecay, present: true, detected: clicks.count, scannedSeconds: scanned,
                                   threshold: o.threshold,
                                   clickTimes: clicks.map { Double($0) / sr }, clickPeaks: peaks)
        params.counted = counted.count
        params.phaseLocked = lockedOut
        params.eligibleSeconds = eligible
        params.levelFromEnergy = energyLevel
        return (params, mask)
    }
}

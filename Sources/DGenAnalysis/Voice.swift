// CPU reference of the SPEC §3.1 kick voice (v1), used by the analysis for
// subtraction and for its own resynthesis check, and by the tests to render
// synthetic kicks from known parameters. Not a product voice: the DGenLisp
// voice is the product; this file documents the exact equations the
// analysis parameters are expressed in.
//
// t = seconds since the hit (analysis: prepared-signal sample 0).
//   f_k(t)   = F_k (1 + G_k e^{-t/T_k})              [slot 0: + F_0 G2 e^{-t/T2}]
//   phase_k  = 2π[F_k t + F_k G_k T_k (1 - e^{-t/T_k}) (+ F_0 G2 T2 (1 - e^{-t/T2}))] + φ_k
//   env_k    = A_k e^{-t/D_k} (1 - e^{-t/R_k})
//   y_modal  = Σ_k env_k sin(phase_k)
//   N_b(t)   = L_b [m_b e^{-t/d1_b} + (1 - m_b) e^{-t/d2_b}] (1 - e^{-t/r_n})
//   x        = y_modal + Σ_b N_b(t) band_b(t)
//   y        = level · gate(t) · clip(x)
//   clip(x)  = mix · c tanh(drive x / c) / drive + (1 - mix) x
//   gate(t)  = 1 - smoothstep((t - E + Fd) / Fd)

import Foundation

public struct ModeParams: Codable, Equatable {
    public var F: Double      // Hz (settled frequency)
    public var G: Double      // drop depth (relative)
    public var T: Double      // drop time constant, s
    public var A: Double      // amplitude at t = 0
    public var D: Double      // decay time constant, s
    public var R: Double      // rise time constant, s
    public var phi: Double    // phase, rad
    public var G2: Double = 0 // slot 0 only: second drop depth
    public var T2: Double = 0 // slot 0 only: second drop time constant

    public init(F: Double, G: Double = 0, T: Double = 0.03, A: Double, D: Double, R: Double = 0.0005,
                phi: Double = 0, G2: Double = 0, T2: Double = 0) {
        self.F = F; self.G = G; self.T = T; self.A = A; self.D = D; self.R = R
        self.phi = phi; self.G2 = G2; self.T2 = T2
    }

    /// Phase without φ (the drop-law integral), at time t.
    @inline(__always) public func phaseLaw(_ t: Double) -> Double {
        var p = F * t
        if G != 0 && T > 0 { p += F * G * T * (1 - exp(-t / T)) }
        if G2 != 0 && T2 > 0 { p += F * G2 * T2 * (1 - exp(-t / T2)) }
        return 2 * Double.pi * p
    }

    @inline(__always) public func frequency(_ t: Double) -> Double {
        var f = F
        if G != 0 && T > 0 { f += F * G * exp(-t / T) }
        if G2 != 0 && T2 > 0 { f += F * G2 * exp(-t / T2) }
        return f
    }

    @inline(__always) public func envelope(_ t: Double) -> Double {
        let rise = R > 0 ? 1 - exp(-t / R) : 1
        return A * exp(-t / D) * rise
    }
}

public struct NoiseBandParams: Codable, Equatable {
    public var lo: Double
    public var hi: Double
    public var L: Double
    public var m: Double
    public var d1: Double
    public var d2: Double
    public init(lo: Double, hi: Double, L: Double, m: Double, d1: Double, d2: Double) {
        self.lo = lo; self.hi = hi; self.L = L; self.m = m; self.d1 = d1; self.d2 = d2
    }
    @inline(__always) public func envelope(_ t: Double, rn: Double) -> Double {
        let rise = rn > 0 ? 1 - exp(-t / rn) : 1
        return L * (m * exp(-t / d1) + (1 - m) * exp(-t / d2)) * rise
    }
}

public struct GateParams: Codable, Equatable {
    public var E: Double
    public var Fd: Double
    public init(E: Double, Fd: Double) { self.E = E; self.Fd = Fd }
    @inline(__always) public func value(_ t: Double) -> Double {
        let u = (t - E + Fd) / max(Fd, 1e-6)
        if u <= 0 { return 1 }
        if u >= 1 { return 0 }
        return 1 - u * u * (3 - 2 * u)
    }
}

public struct ChainParams: Codable, Equatable {
    public var level: Double = 1
    public var drive: Double = 1
    public var ceiling: Double = 1
    public var mix: Double = 0
    public init(level: Double = 1, drive: Double = 1, ceiling: Double = 1, mix: Double = 0) {
        self.level = level; self.drive = drive; self.ceiling = ceiling; self.mix = mix
    }
    @inline(__always) public func clip(_ x: Double) -> Double {
        if mix == 0 { return x }
        return mix * ceiling * tanh(drive * x / ceiling) / drive + (1 - mix) * x
    }
}

public struct KickVoiceParams: Codable, Equatable {
    public var modes: [ModeParams]          // modes[0] = fundamental (slot 0)
    public var noiseBands: [NoiseBandParams]
    public var rn: Double
    public var gate: GateParams
    public var chain: ChainParams
    /// true: every band filters the same white (SPEC §3.1); false: one
    /// independent white per band (seed + band index).
    public var sharedWhite: Bool
    /// cascaded band-pass sections per noise band (1 = 2nd order, SPEC §3.1)
    public var noiseSections: Int
    /// v3: noise events on the same band signals (coefficient N_b(t) + Σ_e env_e(t) col_eb)
    public var events: [NoiseEventParams] = []
    /// v3: crackle, added after the clipper (it is the record, not the recorded drum)
    public var crackle: CrackleParams? = nil
    public init(modes: [ModeParams], noiseBands: [NoiseBandParams], rn: Double, gate: GateParams,
                chain: ChainParams = ChainParams(), sharedWhite: Bool = true, noiseSections: Int = 1) {
        self.modes = modes; self.noiseBands = noiseBands; self.rn = rn; self.gate = gate; self.chain = chain
        self.sharedWhite = sharedWhite
        self.noiseSections = noiseSections
    }
}

public enum KickVoice {
    /// Modal part only (no gate, no chain).
    public static func renderModes(_ modes: [ModeParams], count n: Int, sampleRate sr: Double) -> [Double] {
        var y = [Double](repeating: 0, count: n)
        for m in modes {
            if m.A == 0 { continue }
            for i in 0..<n {
                let t = Double(i) / sr
                let e = m.envelope(t)
                if e == 0 { continue }
                y[i] += e * sin(m.phaseLaw(t) + m.phi)
            }
        }
        return y
    }

    /// Noise part Σ N_b band_b with the deterministic band signals.
    public static func renderNoise(_ bands: [NoiseBandParams], rn: Double, count n: Int, sampleRate sr: Double,
                                   seed: UInt64 = NoiseBands.defaultSeed, sharedWhite: Bool = true,
                                   sections: Int = 1, events: [NoiseEventParams] = []) -> [Double] {
        var y = [Double](repeating: 0, count: n)
        if bands.isEmpty { return y }
        let white = NoiseBands.white(count: n, seed: seed)
        let evEnv = events.map { e in (0..<n).map { e.envelope(Double($0) / sr) } }
        for (k, b) in bands.enumerated() {
            let evUsed = events.indices.contains { k < events[$0].colour.count && events[$0].colour[k] != 0 && events[$0].level != 0 }
            if b.L == 0 && !evUsed { continue }
            let w = sharedWhite ? white : NoiseBands.white(count: n, seed: seed &+ UInt64(k + 1) &* 0x2545F4914F6CDD1D)
            let s = NoiseBands.bandSignal(white: w, lo: b.lo, hi: b.hi, sampleRate: sr, sections: sections)
            for i in 0..<n {
                var c = b.envelope(Double(i) / sr, rn: rn)
                for (e, ev) in events.enumerated() where k < ev.colour.count { c += evEnv[e][i] * ev.colour[k] }
                y[i] += c * s[i]
            }
        }
        return y
    }

    public static func render(_ p: KickVoiceParams, count n: Int, sampleRate sr: Double,
                              includeNoise: Bool = true, seed: UInt64 = NoiseBands.defaultSeed) -> [Double] {
        var x = renderModes(p.modes, count: n, sampleRate: sr)
        if includeNoise {
            let nz = renderNoise(p.noiseBands, rn: p.rn, count: n, sampleRate: sr, seed: seed,
                                 sharedWhite: p.sharedWhite, sections: p.noiseSections, events: p.events)
            for i in 0..<n { x[i] += nz[i] }
        }
        let cr = includeNoise ? p.crackle.map { Crackle.render($0, count: n, sampleRate: sr, seed: seed ^ 0xC4AC) } : nil
        for i in 0..<n {
            let t = Double(i) / sr
            x[i] = p.chain.level * p.gate.value(t) * (p.chain.clip(x[i]) + (cr?[i] ?? 0))
        }
        return x
    }
}

/// Noise band definition shared by the analysis and the voice.
///   edges: NB+1 log-spaced edges from lo to hi;
///   band b: RBJ band-pass (constant 0 dB peak gain), fc = sqrt(lo_b hi_b),
///           Q = fc / (hi_b - lo_b), direct form I, zero initial state;
///           optionally `sections` cascaded copies (2 = 4th order);
///   input: one uniform white in [-1, 1) (xorshift64*, seed fixed);
///   normalised to unit expected RMS: band_b = (h_b * w) / (σ_w sqrt(Σ h_b²)),
///           σ_w = 1/sqrt(3).
public enum NoiseBands {
    public static let defaultSeed: UInt64 = 0x9E3779B97F4A7C15

    public static func edges(count nb: Int, lo: Double, hi: Double) -> [Double] {
        (0...nb).map { lo * pow(hi / lo, Double($0) / Double(nb)) }
    }

    public struct Biquad {
        public var b0, b1, b2, a1, a2: Double
    }

    public static func bandpass(lo: Double, hi: Double, sampleRate sr: Double) -> Biquad {
        let fc = sqrt(lo * hi)
        let q = fc / (hi - lo)
        let w0 = 2 * Double.pi * min(fc, 0.49 * sr) / sr
        let alpha = sin(w0) / (2 * q)
        let a0 = 1 + alpha
        return Biquad(b0: alpha / a0, b1: 0, b2: -alpha / a0, a1: -2 * cos(w0) / a0, a2: (1 - alpha) / a0)
    }

    public static func filter(_ x: [Double], _ c: Biquad) -> [Double] {
        var y = [Double](repeating: 0, count: x.count)
        var x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0
        for i in 0..<x.count {
            let v = c.b0 * x[i] + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2
            x2 = x1; x1 = x[i]; y2 = y1; y1 = v
            y[i] = v
        }
        return y
    }

    /// Σ h² of the impulse response of `sections` cascaded copies of the
    /// band-pass (noise power gain).
    public static func powerGain(_ c: Biquad, sections: Int = 1) -> Double {
        var imp = [Double](repeating: 0, count: 1 << 16)
        imp[0] = 1
        var len = 1 << 16
        while true {
            var h = imp
            for _ in 0..<sections { h = filter(h, c) }
            // long enough when the tail has died out
            let tail = h[(len - 64)..<len].reduce(0) { $0 + $1 * $1 }
            if tail < 1e-24 || len >= 1 << 21 { return h.reduce(0) { $0 + $1 * $1 } }
            len *= 2
            imp = [Double](repeating: 0, count: len)
            imp[0] = 1
        }
    }

    /// |H(f)|² of the biquad at frequency f.
    public static func powerResponse(_ c: Biquad, f: Double, sampleRate sr: Double) -> Double {
        let w = 2 * Double.pi * f / sr
        let (c1, s1, c2, s2) = (cos(w), sin(w), cos(2 * w), sin(2 * w))
        let nr = c.b0 + c.b1 * c1 + c.b2 * c2, ni = -(c.b1 * s1 + c.b2 * s2)
        let dr = 1 + c.a1 * c1 + c.a2 * c2, di = -(c.a1 * s1 + c.a2 * s2)
        return (nr * nr + ni * ni) / max(dr * dr + di * di, 1e-300)
    }

    public static func white(count n: Int, seed: UInt64 = defaultSeed) -> [Double] {
        var s = seed == 0 ? 1 : seed
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27
            let r = s &* 2685821657736338717
            out[i] = Double(r >> 11) / Double(1 << 53) * 2 - 1
        }
        return out
    }

    /// Unit-expected-RMS band signal of the given white (`sections`
    /// cascaded copies of the band-pass: 1 = 2nd order, 2 = 4th order).
    public static func bandSignal(white: [Double], lo: Double, hi: Double, sampleRate sr: Double,
                                  sections: Int = 1) -> [Double] {
        let c = bandpass(lo: lo, hi: hi, sampleRate: sr)
        let g = 1 / (sqrt(1.0 / 3.0) * sqrt(powerGain(c, sections: sections)))
        var y = white
        for _ in 0..<sections { y = filter(y, c) }
        return y.map { $0 * g }
    }
}

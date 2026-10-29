// Per-sample kick analysis: prepare → gate → pitch drop → slot 0 envelope →
// damped modes (matrix pencil per region) → noise bands → check.
// Every parameter is expressed in the SPEC §3.1 voice terms (Voice.swift)
// with t = 0 at sample 0 of the prepared target.

import Foundation

public struct KickAnalysisOptions: Codable {
    public var noiseBandCount = 16
    public var noiseLoHz = 40.0
    public var noiseHiHz = 16000.0
    public var maxModes = 24
    public var lowRegion = [30.0, 1000.0]
    public var midRegion = [1000.0, 6000.0]
    public var lowDecimation = 8
    public var midDecimation = 2
    public var lowSegmentSeconds = 0.3
    public var midSegmentSeconds = 0.12
    public var minDecayLow = 0.004
    public var minDecayMid = 0.0015
    public var modeEnergyFloorDB = 60.0   // vs slot 0 energy
    public var fundamentalExclusion = 0.07
    public var secondDrop = true
    public var modeDropTest = true
    public var varproRefine = true
    /// relative weighting floor for the slot-0 fit and the joint LS
    /// (0 = plain least squares; 0.1 = errors 20 dB down weigh up to 10x)
    public var relativeWeightFloor = 0.0
    /// allow a second mode on the fundamental's law with a slower decay
    /// (two-stage decay of the fundamental; §3.1 has one exponential per mode)
    public var fundamentalTail = true
    public var fundamentalTailGain = 0.85  // accept when the dB RMS falls below this fraction
    public var stabilize = true           // keep only matrix-pencil modes stable across segment/pencil variants
    public var stabilityRelF = 0.03
    public var stabilityRelD = 0.8
    public var modeDropRatio = 2.0
    public var defaultRise = 0.0007
    public var sharedNoiseWhite = true    // SPEC §3.1: one white for all bands
    public var noiseSections = 1          // 1 = 2nd-order bands (SPEC §3.1), 2 = 4th order
    public var checkLength = 32768
    public var noiseSeed: UInt64 = NoiseBands.defaultSeed
    public init() {}
}

public struct ModeEstimate: Codable {
    public var F: Double
    public var G: Double
    public var T: Double
    public var A: Double
    public var D: Double
    public var R: Double
    public var phi: Double
    public var G2: Double
    public var T2: Double
    public var region: String       // fundamental | low | mid
    public var energy: Double       // ∫ env² sin² ≈ A² D / 4 (prepared units)
    public var dropTestRatio: Double?
    public var amplitudeCapped: Bool

    var params: ModeParams {
        ModeParams(F: F, G: G, T: T, A: A, D: D, R: R, phi: phi, G2: G2, T2: T2)
    }
}

public struct KickCheck: Codable {
    public var band: Double
    public var mr: Double
    public var bandModal: Double
    public var mrModal: Double
    public var tailBandModal: Double
    public var tailRelRMSModal: Double
    public var noiseShare: Double
    public var tailStartSeconds: Double
}

public struct PrepInfo: Codable {
    public var cropStart: Int
    public var prerollSamples: Int
    public var onsetRawSample: Double
    public var onsetSeconds: Double
    public var rawPeak: Double
    public var gain: Double
    public var length: Int
    public var channels: Int
    public var rawFrames: Int
}

public struct KickAnalysis: Codable {
    public var name: String
    public var sampleRate: Double
    public var prep: PrepInfo
    public var drop: PitchDropFit
    public var dropTail: DampedSinusoid?   // MP mode used to pin F0 (nil: IF law F kept)
    public var modes: [ModeEstimate]       // modes[0] = slot 0 (fundamental)
    public var noise: NoiseFitResult
    public var gate: GateFit
    public var chain: ChainParams
    public var check: KickCheck?
    public var warnings: [String]

    public var voiceParams: KickVoiceParams {
        KickVoiceParams(modes: modes.map { $0.params }, noiseBands: noise.bands, rn: noise.rn,
                        gate: GateParams(E: gate.E, Fd: gate.Fd), chain: chain, sharedWhite: noise.sharedWhite,
                        noiseSections: noise.sections)
    }
}

public struct KickDebug: Codable {
    public var ifTrack: IFTrack?
    public var slot0Envelope: IFTrack?   // t, |baseband|, unused
}

public enum KickAnalyzer {
    public static func analyze(name: String, audio: WavAudio, options o: KickAnalysisOptions = KickAnalysisOptions(),
                               debug: inout KickDebug) -> (KickAnalysis, PreparedTarget)
    {
        let prep = Prepare.prepare(name: name, audio: audio)
        let x = prep.signal
        let sr = prep.sampleRate
        let onset = prep.onsetSeconds
        var warnings: [String] = []

        // gate
        let gate = GateEstimator.fit(x, sr: sr)
        let tEnd = max(onset + 0.02, gate.E - gate.Fd)

        // pitch drop
        var dopt = PitchDrop.Options()
        dopt.allowSecondDrop = o.secondDrop
        var track: IFTrack? = nil
        var drop = PitchDrop.fit(x, sr: sr, onset: onset, end: tEnd, options: dopt, track: &track)
        debug.ifTrack = track
        // tail start: where the IF law has settled to within 3% of F
        // (clamped to [20, 120] ms after the onset and >= 60 ms before tEnd)
        let gTot = drop.G + (drop.secondDrop ? drop.G2 : 0)
        let tSettle = gTot > 0.03 ? drop.T * log(gTot / 0.03) : 0.02
        let t1 = onset + max(0.02, min(tSettle, 0.12, tEnd - onset - 0.06))

        // MP on the raw low region tail pins F0 (to the tail mode nearest the
        // late IF median) and gives the tail decay
        var tailIF = drop.F
        if let tr = track {
            let late = tr.t.indices.filter { tr.t[$0] > t1 }
            if late.count >= 8 { tailIF = PitchDrop.weightedMedian(late.map { tr.f[$0] }, late.map { tr.w[$0] }) }
        }
        let lowRaw = regionSegment(x, sr: sr, band: o.lowRegion, decim: o.lowDecimation, from: t1,
                                   to: min(tEnd, t1 + o.lowSegmentSeconds))
        var dropTail: DampedSinusoid? = nil
        if let seg = lowRaw {
            let mp = MatrixPencil.estimate(seg.x, sampleRate: seg.fs)
            let near = mp.modes.filter { abs($0.frequency / tailIF - 1) < 0.12 && $0.decay > 0.005 && $0.frequency > 25 }
            if let m = near.max(by: { $0.amplitude < $1.amplitude }) {
                dropTail = m
                if let tr = track { drop = PitchDrop.refit(drop, track: tr, fixedF: m.frequency, onset: onset) }
            } else {
                warnings.append(String(format: "no tail mode within 12%% of the late IF median %.1f Hz", tailIF))
            }
        } else {
            warnings.append("low-region tail segment too short for matrix pencil")
        }

        // without a pinned tail mode, F0 is held near the late IF median
        // (the IF law alone can trade F against a slow T)
        var fRange = [0.97 * drop.F, 1.03 * drop.F]
        if dropTail == nil {
            if abs(drop.F / tailIF - 1) > 0.08, let tr = track {
                drop = PitchDrop.refit(drop, track: tr, fixedF: tailIF, onset: onset)
            }
            fRange = [0.92 * tailIF, 1.08 * tailIF]
        }
        // slot 0: phase-accurate fit of the demodulated fundamental
        // the early frequency may not exceed what the IF track supports
        var fCap = 1.1 * dopt.lowpassHz
        if let tr = track, let wmax = tr.w.max() {
            let strong = tr.t.indices.filter { tr.w[$0] > 0.1 * wmax }
            if let fmax = strong.map({ tr.f[$0] }).max() { fCap = min(fCap, 1.15 * fmax) }
        }
        var slot0 = fitSlot0(x, sr: sr, drop: drop, onset: onset, tEnd: tEnd, fRange: fRange, fCap: fCap,
                             tailDecay: dropTail?.decay, allowSecond: o.secondDrop,
                             relativeFloor: o.relativeWeightFloor, debug: &debug)
        // two-stage decay of the fundamental: a second mode on the same law
        var slot0Tail: ModeEstimate? = nil
        if o.fundamentalTail, let two = fitSlot0TwoStage(x, sr: sr, slot0: slot0, onset: onset, tEnd: tEnd,
                                                                   tailGain: o.fundamentalTailGain) {
            slot0 = two.0
            slot0Tail = two.1
        }
        let y0 = KickVoice.renderModes([slot0.params] + (slot0Tail.map { [$0.params] } ?? []), count: x.count,
                                       sampleRate: sr)
        let r0 = zip(x, y0).map { $0 - $1 }

        // candidate modes: matrix pencil on the slot-0 residual, per region
        var cands: [ModeEstimate] = []
        if let seg = regionSegment(r0, sr: sr, band: o.lowRegion, decim: o.lowDecimation, from: t1,
                                   to: min(tEnd, t1 + o.lowSegmentSeconds)) {
            var lows = modesOf(seg, o).filter {
                $0.frequency >= o.lowRegion[0] * 0.9 && $0.frequency <= o.lowRegion[1] * 1.1 && $0.decay >= o.minDecayLow
            }
            if o.varproRefine { lows = MatrixPencil.refine(seg.x, sampleRate: seg.fs, modes: lows, minDecay: o.minDecayLow) }
            for m in lows {
                if m.frequency < o.lowRegion[0] * 0.9 || m.frequency > o.lowRegion[1] * 1.1 { continue }
                if abs(m.frequency / slot0.F - 1) < o.fundamentalExclusion { continue }
                if m.decay < o.minDecayLow || m.decay > 5 { continue }
                cands.append(candidate(m, t0: seg.t0, region: "low", rise: o.defaultRise))
            }
        }
        let midFrom = onset + 0.003
        if let seg = regionSegment(r0, sr: sr, band: o.midRegion, decim: o.midDecimation, from: midFrom,
                                   to: min(tEnd, midFrom + o.midSegmentSeconds)) {
            var mids = modesOf(seg, o).filter {
                $0.frequency >= o.midRegion[0] * 0.95 && $0.frequency <= o.midRegion[1] * 1.05 && $0.decay >= o.minDecayMid
            }
            if o.varproRefine { mids = MatrixPencil.refine(seg.x, sampleRate: seg.fs, modes: mids, minDecay: o.minDecayMid) }
            for m in mids {
                if m.frequency < o.midRegion[0] * 0.95 || m.frequency > o.midRegion[1] * 1.05 { continue }
                if m.decay < o.minDecayMid || m.decay > 5 { continue }
                cands.append(candidate(m, t0: seg.t0, region: "mid", rise: o.defaultRise))
            }
        }
        // de-duplicate (within 2%: keep the stronger at its own segment start)
        cands.sort { $0.energy > $1.energy }
        var kept: [ModeEstimate] = []
        for m in cands {
            if kept.contains(where: { abs($0.F / m.F - 1) < 0.02 }) { continue }
            if kept.count >= o.maxModes - 1 { break }
            kept.append(m)
        }
        kept.sort { $0.F < $1.F }

        // joint least-squares amplitudes/phases of all modes over [0, tEnd]
        let nFit = min(x.count, Int(tEnd * sr))
        if let tl = slot0Tail {
            if kept.count >= o.maxModes - 2 { kept.removeLast() }
            kept.insert(tl, at: 0)
        }
        var modes = [slot0] + kept
        let lsW = relativeWeights(x, sr: sr, count: nFit, floor: o.relativeWeightFloor)
        modes = jointAmplitudes(modes, x: x, count: nFit, sr: sr, weights: lsW)
        let floorE = modes[0].energy * pow(10, -o.modeEnergyFloorDB / 10)
        let before = modes.count
        modes = [modes[0]] + modes.dropFirst().filter { $0.energy >= floorE }
        if modes.count != before { modes = jointAmplitudes(modes, x: x, count: nFit, sr: sr, weights: lsW) }

        // per-mode drop test (low modes only)
        if o.modeDropTest && drop.G > 0 {
            modes = dropTest(modes, x: x, count: min(nFit, Int((t1 + 0.05) * sr)), fullCount: nFit, sr: sr,
                             drop: drop, ratio: o.modeDropRatio, weights: lsW)
        }

        // noise from the residual of the full modal part
        let ym = KickVoice.renderModes(modes.map { $0.params }, count: x.count, sampleRate: sr)
        let resid = zip(x, ym).map { $0 - $1 }
        let edges = NoiseBands.edges(count: o.noiseBandCount, lo: o.noiseLoHz, hi: min(o.noiseHiHz, 0.45 * sr))
        var nopt = NoiseFit.Options()
        nopt.sharedWhite = o.sharedNoiseWhite
        nopt.sections = o.noiseSections
        let noise = NoiseFit.fit(residual: resid, sr: sr, onset: onset, end: gate.E, edges: edges, options: nopt)

        var a = KickAnalysis(name: name, sampleRate: sr,
                             prep: PrepInfo(cropStart: prep.cropStart, prerollSamples: prep.prerollSamples,
                                            onsetRawSample: prep.onsetRawSample, onsetSeconds: onset,
                                            rawPeak: prep.rawPeak, gain: prep.gain, length: x.count,
                                            channels: prep.channels, rawFrames: prep.rawFrames),
                             drop: drop, dropTail: dropTail, modes: modes, noise: noise, gate: gate,
                             chain: ChainParams(), check: nil, warnings: warnings)
        a.check = check(a, target: x, residual: resid, modal: ym, options: o)
        return (a, prep)
    }

    // MARK: pieces

    struct Segment {
        var x: [Double]
        var fs: Double
        var t0: Double
    }

    /// w(t) = max|z| / (|z|_smooth(t) + floor·max|z|): least squares in
    /// (roughly) relative terms, so the quiet tail counts like the band
    /// metric counts it. floor = 0 → nil (plain least squares).
    static func relativeWeights(_ x: [Double], sr: Double, count n: Int, floor: Double) -> [Double]? {
        guard floor > 0 else { return nil }
        let z = FFT.spectralFilter(x, sampleRate: sr, analytic: true, gain: FFT.bandGain(lo: 20, hi: 0))
        let mag = (0..<n).map { (z.re[$0] * z.re[$0] + z.im[$0] * z.im[$0]).squareRoot() }
        // 5 ms moving average
        let h = max(1, Int(0.0025 * sr))
        var pre = [Double](repeating: 0, count: n + 1)
        for i in 0..<n { pre[i + 1] = pre[i] + mag[i] }
        let sm = (0..<n).map { i -> Double in
            let a = max(0, i - h), b = min(n, i + h + 1)
            return (pre[b] - pre[a]) / Double(b - a)
        }
        let mx = sm.max() ?? 1
        return sm.map { mx / ($0 + floor * mx) }
    }

    static func modesOf(_ seg: Segment, _ o: KickAnalysisOptions) -> [DampedSinusoid] {
        o.stabilize ? MatrixPencil.stableModes(seg.x, sampleRate: seg.fs, relF: o.stabilityRelF, relD: o.stabilityRelD)
            : MatrixPencil.estimate(seg.x, sampleRate: seg.fs).modes
    }

    /// Band-limit (zero phase), decimate, and cut [from, to).
    static func regionSegment(_ x: [Double], sr: Double, band: [Double], decim: Int, from: Double, to: Double)
        -> Segment?
    {
        let a = Int((from * sr).rounded())
        let b = min(x.count, Int(to * sr))
        guard b - a > decim * 48 else { return nil }
        let y = FFT.spectralFilter(x, sampleRate: sr, analytic: false,
                                   gain: FFT.bandGain(lo: band[0], hi: band[1], taper: 0.2)).re
        let seg = Swift.stride(from: a, to: b, by: decim).map { y[$0] }
        return Segment(x: seg, fs: sr / Double(decim), t0: Double(a) / sr)
    }

    /// Candidate mode from a segment estimate: F, D, and the energy at its
    /// own segment start (for ranking); amplitudes come from the joint LS.
    static func candidate(_ m: DampedSinusoid, t0: Double, region: String, rise: Double) -> ModeEstimate {
        ModeEstimate(F: m.frequency, G: 0, T: 0.03, A: m.amplitude, D: m.decay, R: rise, phi: 0, G2: 0, T2: 0,
                     region: region, energy: m.amplitude * m.amplitude * m.decay / 4, dropTestRatio: nil,
                     amplitudeCapped: false)
    }

    /// Unit-amplitude cos/sin columns of a mode: env(t)/A · {cos, sin}(phaseLaw).
    static func columns(_ m: ModeEstimate, count n: Int, sr: Double) -> ([Double], [Double]) {
        var p = m.params
        p.A = 1
        var c = [Double](repeating: 0, count: n), s = c
        for i in 0..<n {
            let t = Double(i) / sr
            let e = p.envelope(t), th = p.phaseLaw(t)
            c[i] = e * cos(th)
            s[i] = e * sin(th)
        }
        return (c, s)
    }

    /// Solve all modes' (A, φ) jointly: x ≈ Σ a_k c_k + b_k s_k on [0, n).
    /// Least squares guarantees the modal part never adds residual energy
    /// over the fitted window. Returns the SSE in `sse`.
    static func jointAmplitudes(_ modes: [ModeEstimate], x x0: [Double], count n: Int, sr: Double,
                                weights: [Double]? = nil,
                                sse: UnsafeMutablePointer<Double>? = nil) -> [ModeEstimate]
    {
        let cols = modes.map { columns($0, count: n, sr: sr) }
        let k = modes.count
        // normal equations (2k x 2k), small ridge for collinear pairs
        var basis: [[Double]] = []
        for (c, s) in cols { basis.append(c); basis.append(s) }
        var x = x0
        if let w = weights {
            for i in 0..<n { x[i] *= w[i] }
            for c in basis.indices { for i in 0..<n { basis[c][i] *= w[i] } }
        }
        let m = 2 * k
        var G = [[Double]](repeating: [Double](repeating: 0, count: m), count: m)
        var rhs = [Double](repeating: 0, count: m)
        for a in 0..<m {
            basis[a].withUnsafeBufferPointer { pa in
                x.withUnsafeBufferPointer { px in
                    var r = 0.0
                    for i in 0..<n { r += pa[i] * px[i] }
                    rhs[a] = r
                }
                for b in a..<m {
                    basis[b].withUnsafeBufferPointer { pb in
                        var g = 0.0
                        for i in 0..<n { g += pa[i] * pb[i] }
                        G[a][b] = g
                        G[b][a] = g
                    }
                }
            }
        }
        let tr = (0..<m).reduce(0.0) { $0 + G[$1][$1] } / Double(max(m, 1))
        for a in 0..<m { G[a][a] += 1e-9 * tr }
        let coef = LinAlg.leastSquares(G, rhs)
        var out = modes
        for j in 0..<k {
            let a = coef[2 * j], b = coef[2 * j + 1]  // a cos + b sin = A sin(θ + φ)
            out[j].A = (a * a + b * b).squareRoot()
            out[j].phi = atan2(a, b)
            out[j].energy = out[j].A * out[j].A * out[j].D / 4
        }
        if let p = sse {
            var e = 0.0
            for i in 0..<n {
                var y = 0.0
                for c in 0..<m { y += coef[c] * basis[c][i] }
                e += (x[i] - y) * (x[i] - y)
            }
            p.pointee = e
        }
        return out
    }

    /// Slot 0 by variable projection on the demodulated fundamental:
    /// u(t) = LP40( z(t) e^{-iθ_ref(t)} ), z = analytic signal of x in
    /// [25, 400] Hz; model u ≈ c · env(t; D, R) · e^{i(θ(t; F,G,T,G2,T2) - θ_ref(t))}
    /// with the complex c solved in closed form; LM over the nonlinear
    /// parameters, two re-referencing passes. A = |c|, φ = arg(i c).
    static func fitSlot0(_ x: [Double], sr: Double, drop: PitchDropFit, onset: Double, tEnd: Double,
                         fRange: [Double], fCap: Double, tailDecay: Double?, allowSecond: Bool,
                         relativeFloor: Double = 0, debug: inout KickDebug) -> ModeEstimate
    {
        let z = PitchDrop.analytic(x, sr: sr, lo: 25, hi: 400)
        let a = fitSlot0Core(x, z: z, sr: sr, drop: drop, onset: onset, tEnd: tEnd, fRange: fRange, fCap: fCap,
                             tailDecay: tailDecay, relativeFloor: relativeFloor, debug: &debug)
        guard allowSecond && !drop.secondDrop && a.G > 0.1 else { return a }
        // the phase-sensitive fit can see a fast initial drop the IF fit missed:
        // try a second term and keep it if the low-band residual falls >= 10%
        var d2 = drop
        d2.secondDrop = true
        d2.F = a.F; d2.G = a.G * 0.7; d2.T = a.T * 1.3; d2.G2 = a.G * 0.5; d2.T2 = max(0.0008, a.T * 0.2)
        var dbg2 = KickDebug()
        let b = fitSlot0Core(x, z: z, sr: sr, drop: d2, onset: onset, tEnd: tEnd, fRange: fRange, fCap: fCap,
                             tailDecay: tailDecay, relativeFloor: relativeFloor, debug: &dbg2)
        guard b.G2 > 0 else { return a }
        let i0 = Int(onset * sr), i1 = min(x.count, Int(tEnd * sr))
        func lowResidual(_ m: ModeEstimate) -> Double {
            let y = KickVoice.renderModes([m.params], count: i1, sampleRate: sr)
            var e = 0.0
            for i in i0..<i1 { e += (z.re[i] - y[i]) * (z.re[i] - y[i]) }
            return e
        }
        if lowResidual(b) < 0.9 * lowResidual(a) {
            debug.slot0Envelope = dbg2.slot0Envelope
            return b
        }
        return a
    }

    /// Envelope-only refit of slot 0 with two decay stages on its fixed law,
    /// fitted in dB (the check metrics are logarithmic, and a tail 30 dB
    /// down is invisible to a linear fit):
    ///   |i·u| ≈ A [(1-q) e^{-t/D} + q e^{-t/Db}] (1 - e^{-t/R}),  Db > D,
    /// 60 dB floor. Returned as two modes (slot 0 and a "fundamental-tail"
    /// mode with the same law and phase) when the dB RMS falls by >= 15%
    /// vs. the single-stage envelope.
    static func fitSlot0TwoStage(_ x: [Double], sr: Double, slot0: ModeEstimate, onset: Double, tEnd: Double,
                                 tailGain: Double)
        -> (ModeEstimate, ModeEstimate)?
    {
        let z = PitchDrop.analytic(x, sr: sr, lo: 25, hi: 400)
        let n = x.count
        var law = slot0.params
        law.A = 1
        law.phi = 0
        var dr = [Double](repeating: 0, count: n), di = dr
        for i in 0..<n {
            let th = law.phaseLaw(Double(i) / sr)
            let c = cos(th), s = sin(th)
            dr[i] = z.re[i] * c + z.im[i] * s
            di[i] = z.im[i] * c - z.re[i] * s
        }
        let u = FFT.complexFilter(re: dr, im: di, sampleRate: sr, gain: FFT.bandGain(lo: 0, hi: 40, taper: 0.5))
        let hop = max(1, Int(0.001 * sr))
        let last = min(n, Int(tEnd * sr))
        let idx = Swift.stride(from: Int((onset + 0.002) * sr), to: last, by: hop).map { $0 }
        guard idx.count > 40 else { return nil }
        // from the envelope peak on: the 40 Hz baseband smears the attack
        let mag0 = idx.map { (u.re[$0] * u.re[$0] + u.im[$0] * u.im[$0]).squareRoot() }
        let ipk = mag0.indices.max { mag0[$0] < mag0[$1] } ?? 0
        let ts = Array(idx[ipk...]).map { Double($0) / sr }
        let mag = Array(mag0[ipk...])
        guard ts.count > 30 else { return nil }
        let floorA = (mag.max() ?? 1) * 1e-3
        let target = mag.map { 20 * log10($0 + floorA) }
        let R = slot0.R
        func model(_ p: [Double], _ t: Double) -> Double {
            let (A, D, Db, q) = (p[0], p[1], p[2], p[3])
            return A * ((1 - q) * exp(-t / D) + q * exp(-t / Db)) * (1 - exp(-t / R))
        }
        func res(_ p: [Double]) -> [Double] { ts.indices.map { 20 * log10(model(p, ts[$0]) + floorA) - target[$0] } }
        func rms(_ r: [Double]) -> Double { (r.reduce(0) { $0 + $1 * $1 } / Double(r.count)).squareRoot() }
        let A0 = slot0.A, D0 = slot0.D
        let one = LinAlg.levenbergMarquardt(x0: [A0, D0, D0, 0], lower: [0, 0.003, 0.003, 0],
                                            upper: [20, 5, 5, 0], maxIter: 60, residuals: res)
        var best: [Double]? = nil
        var bestCost = Double.infinity
        for (dbMul, q0) in [(3.0, 0.1), (8.0, 0.03)] {
            let f = LinAlg.levenbergMarquardt(
                x0: [A0, D0 * 0.8, min(5, D0 * dbMul), q0],
                lower: [0, 0.003, 0.003, 0.0], upper: [20, 5, 5, 0.95], maxIter: 80, residuals: res)
            if f.cost < bestCost {
                bestCost = f.cost
                best = f.x
            }
        }
        guard var p = best else { return nil }
        if p[2] < p[1] { p = [p[0], p[2], p[1], 1 - p[3]] }
        if ProcessInfo.processInfo.environment["DGEN_ANALYSIS_TRACE"] != nil {
            fputs(String(format: "two-stage: one rms %.2f dB (D %.3f) two rms %.2f dB p %@\n", rms(res(one.x)), one.x[1],
                         rms(res(p)), p.description), stderr)
        }
        guard p[2] > 1.5 * p[1], p[3] > 0.002, rms(res(p)) < tailGain * rms(res(one.x)) else { return nil }
        var a = slot0
        a.A = p[0] * (1 - p[3]); a.D = p[1]; a.energy = a.A * a.A * a.D / 4
        var b = slot0
        b.A = p[0] * p[3]; b.D = p[2]; b.energy = b.A * b.A * b.D / 4
        b.region = "fundamental-tail"
        return (a, b)
    }

    static func fitSlot0Core(_ x: [Double], z: PitchDrop.Analytic, sr: Double, drop: PitchDropFit, onset: Double,
                             tEnd: Double, fRange: [Double], fCap: Double, tailDecay: Double?,
                             relativeFloor: Double = 0, debug: inout KickDebug) -> ModeEstimate
    {
        let second = drop.secondDrop
        let n = x.count
        let hop = max(1, Int(0.0005 * sr))
        let last = min(n, Int(tEnd * sr))
        let idx = Swift.stride(from: Int(onset * sr), to: last, by: hop).map { $0 }
        let ts = idx.map { Double($0) / sr }
        var lawP: [Double] = second ? [drop.F, drop.G, drop.T, drop.G2, drop.T2] : [drop.F, drop.G, drop.T]
        var D = tailDecay ?? 0.1, R = 0.001
        var ur: [Double] = [], ui: [Double] = []
        var wv: [Double] = []   // residual weights (relative weighting, 1 = linear)
        var refP = lawP
        func law(_ p: [Double]) -> ModeParams {
            ModeParams(F: p[0], G: p[1], T: p[2], A: 1, D: 1, R: 0, phi: 0,
                       G2: p.count > 3 ? p[3] : 0, T2: p.count > 3 ? p[4] : 0)
        }
        func project(_ p: [Double], D: Double, R: Double) -> (cr: Double, ci: Double, res: [Double]) {
            let lp = law(p), lr = law(refP)
            let env = ModeParams(F: 1, A: 1, D: D, R: R)
            var br = [Double](repeating: 0, count: ts.count), bi = br
            var nr = 0.0, ni = 0.0, dd = 0.0
            for (j, t) in ts.enumerated() {
                let e = env.envelope(t), dth = lp.phaseLaw(t) - lr.phaseLaw(t)
                br[j] = e * cos(dth)
                bi[j] = e * sin(dth)
                // Σ w² u conj(b)
                let w2 = wv[j] * wv[j]
                nr += w2 * (ur[j] * br[j] + ui[j] * bi[j])
                ni += w2 * (ui[j] * br[j] - ur[j] * bi[j])
                dd += w2 * (br[j] * br[j] + bi[j] * bi[j])
            }
            let cr = nr / max(dd, 1e-300), ci = ni / max(dd, 1e-300)
            var res = [Double](repeating: 0, count: 2 * ts.count + 1)
            var scale = 0.0
            for j in 0..<ts.count {
                res[2 * j] = wv[j] * (ur[j] - (cr * br[j] - ci * bi[j]))
                res[2 * j + 1] = wv[j] * (ui[j] - (cr * bi[j] + ci * br[j]))
                scale = max(scale, wv[j] * (ur[j] * ur[j] + ui[j] * ui[j]).squareRoot())
            }
            // keep f(onset) within what the IF track supports
            let excess = max(0, lp.frequency(onset) / fCap - 1)
            res[2 * ts.count] = 10 * scale * Double(ts.count).squareRoot() * excess
            return (cr, ci, res)
        }
        var cr = 0.0, ci = 0.0
        for _ in 0..<3 {
            refP = lawP
            let lr = law(refP)
            var dr = [Double](repeating: 0, count: n), di = dr
            for i in 0..<n {
                let th = lr.phaseLaw(Double(i) / sr)
                let c = cos(th), s = sin(th)
                dr[i] = z.re[i] * c + z.im[i] * s
                di[i] = z.im[i] * c - z.re[i] * s
            }
            let u = FFT.complexFilter(re: dr, im: di, sampleRate: sr, gain: FFT.bandGain(lo: 0, hi: 40, taper: 0.5))
            ur = idx.map { u.re[$0] }
            ui = idx.map { u.im[$0] }
            let mag = ts.indices.map { (ur[$0] * ur[$0] + ui[$0] * ui[$0]).squareRoot() }
            let mmax = mag.max() ?? 1
            wv = relativeFloor > 0 ? mag.map { mmax / ($0 + relativeFloor * mmax) } : mag.map { _ in 1 }
            let np = lawP.count
            var lo = (second ? PitchDrop.lo5 : PitchDrop.lo3) + [0.003, 0.0001]
            var hi = (second ? PitchDrop.hi5 : PitchDrop.hi3) + [5.0, 0.01]
            lo[0] = max(lo[0], fRange[0])
            hi[0] = min(hi[0], fRange[1])
            let fit = LinAlg.levenbergMarquardt(x0: lawP + [D, R], lower: lo, upper: hi, maxIter: 60) { q in
                project(Array(q[0..<np]), D: q[np], R: q[np + 1]).res
            }
            lawP = Array(fit.x[0..<np])
            D = fit.x[np]
            R = fit.x[np + 1]
            let pr = project(lawP, D: D, R: R)
            cr = pr.cr
            ci = pr.ci
        }
        if second {
            // keep (G, T) the slower term; a degenerate pair (T/T2 < 2) is one drop
            if lawP[2] < lawP[4] { lawP = [lawP[0], lawP[3], lawP[4], lawP[1], lawP[2]] }
            if lawP[2] < 2 * lawP[4] || min(lawP[1], lawP[3]) < 0.03 {
                var single = drop
                single.secondDrop = false
                single.F = lawP[0]
                single.G = lawP[1] + lawP[3]
                single.T = (lawP[1] * lawP[2] + lawP[3] * lawP[4]) / max(lawP[1] + lawP[3], 1e-9)
                return fitSlot0Core(x, z: z, sr: sr, drop: single, onset: onset, tEnd: tEnd, fRange: fRange,
                                    fCap: fCap, tailDecay: tailDecay, relativeFloor: relativeFloor, debug: &debug)
            }
        }
        debug.slot0Envelope = IFTrack(t: ts, f: ts.indices.map { (ur[$0] * ur[$0] + ui[$0] * ui[$0]).squareRoot() },
                                      w: [])
        // i c = A e^{iφ}  →  (−ci) + i cr
        let A = (cr * cr + ci * ci).squareRoot()
        let phi = atan2(cr, -ci)
        return ModeEstimate(F: lawP[0], G: lawP[1], T: lawP[2], A: A, D: D, R: R, phi: phi,
                            G2: second ? lawP[3] : 0, T2: second ? lawP[4] : 0,
                            region: "fundamental", energy: A * A * D / 4, dropTestRatio: nil,
                            amplitudeCapped: false)
    }

    /// For each low mode, refit the joint amplitudes on the early window
    /// with that mode sharing the fundamental's drop (G0, T0) instead of
    /// G = 0. Adopt the drop when the early-window SSE falls by more than
    /// (1 - 1/ratio) of the mode's own early energy. Amplitudes are then
    /// re-solved on the full window.
    static func dropTest(_ modes: [ModeEstimate], x: [Double], count n: Int, fullCount: Int, sr: Double,
                         drop: PitchDropFit, ratio: Double, weights: [Double]?) -> [ModeEstimate]
    {
        var out = modes
        var base = 0.0
        let fitted = jointAmplitudes(out, x: x, count: n, sr: sr, weights: weights, sse: &base)
        var changed = false
        for k in out.indices where out[k].region == "low" {
            var trial = out
            trial[k].G = drop.G
            trial[k].T = drop.T
            var e = 0.0
            _ = jointAmplitudes(trial, x: x, count: n, sr: sr, weights: weights, sse: &e)
            // early energy of mode k in the base fit
            let (c, s) = columns(fitted[k], count: n, sr: sr)
            let a = fitted[k].A * sin(fitted[k].phi), b = fitted[k].A * cos(fitted[k].phi)
            var ek = 0.0
            for i in 0..<n {
                let y = (a * c[i] + b * s[i]) * (weights?[i] ?? 1)
                ek += y * y
            }
            let gain = (base - e) / max(ek, 1e-30)
            out[k].dropTestRatio = gain
            if gain > 1 - 1 / ratio {
                out[k].G = drop.G
                out[k].T = drop.T
                base = e
                changed = true
            }
        }
        return changed ? jointAmplitudes(out, x: x, count: fullCount, sr: sr, weights: weights) : out
    }

    // MARK: check

    public static func pad(_ x: [Double], _ n: Int) -> [Double] {
        x.count >= n ? Array(x[0..<n]) : x + [Double](repeating: 0, count: n - x.count)
    }

    static func check(_ a: KickAnalysis, target x: [Double], residual: [Double], modal: [Double],
                      options o: KickAnalysisOptions) -> KickCheck
    {
        let sr = a.sampleRate
        let n = x.count
        let full = KickVoice.render(a.voiceParams, count: n, sampleRate: sr, seed: o.noiseSeed)
        let gate = GateParams(E: a.gate.E, Fd: a.gate.Fd)
        let modalG = (0..<n).map { modal[$0] * gate.value(Double($0) / sr) }
        let L = max(o.checkLength, n)
        let T = pad(x, L), P = pad(full, L), Mo = pad(modalG, L)
        let envT = KickMetrics.bandEnvDB(T)
        let envP = KickMetrics.bandEnvDB(P)
        let envM = KickMetrics.bandEnvDB(Mo)
        let tailStart = a.prep.onsetSeconds + 0.06
        let fa = KickMetrics.bandFrame(forSample: Int(tailStart * sr))
        let fb = KickMetrics.bandFrame(forSample: Int(a.gate.E * sr))
        let ia = min(n, Int(tailStart * sr)), ib = min(n, Int(a.gate.E * sr))
        let tailRel = ib > ia ? KickMetrics.relRMS(Array(modalG[ia..<ib]), Array(x[ia..<ib])) : .nan
        var er = 0.0, ex = 0.0
        for i in 0..<min(n, Int(a.gate.E * sr)) {
            er += residual[i] * residual[i]
            ex += x[i] * x[i]
        }
        return KickCheck(band: KickMetrics.bandScore(envP: envP, envT: envT),
                         mr: KickMetrics.mrScore(P, T),
                         bandModal: KickMetrics.bandScore(envP: envM, envT: envT),
                         mrModal: KickMetrics.mrScore(Mo, T),
                         tailBandModal: fb > fa ? KickMetrics.bandScore(envP: envM, envT: envT, frames: fa..<fb) : .nan,
                         tailRelRMSModal: tailRel, noiseShare: er / max(ex, 1e-30), tailStartSeconds: tailStart)
    }
}

import Foundation
import XCTest

@testable import DGenAnalysis

final class LinAlgTests: XCTestCase {
    func testFFTMatchesDFT() {
        let n = 16
        let x = (0..<n).map { sin(Double($0) * 0.7) + 0.3 * cos(Double($0) * 2.1) }
        var re = x, im = [Double](repeating: 0, count: n)
        FFT.transform(&re, &im)
        for k in 0..<n {
            var a = 0.0, b = 0.0
            for i in 0..<n {
                let w = -2 * Double.pi * Double(k * i) / Double(n)
                a += x[i] * cos(w)
                b += x[i] * sin(w)
            }
            XCTAssertEqual(re[k], a, accuracy: 1e-9)
            XCTAssertEqual(im[k], b, accuracy: 1e-9)
        }
        FFT.transform(&re, &im, inverse: true)
        for i in 0..<n { XCTAssertEqual(re[i], x[i], accuracy: 1e-12) }
    }

    func testEigenvaluesOfRotationAndTriangular() {
        // rotation by 0.3 rad scaled by 0.9 → 0.9 e^{±0.3i}; plus a real 0.5
        let c = 0.9 * cos(0.3), s = 0.9 * sin(0.3)
        let m = [[c, -s, 0.1], [s, c, 0.2], [0, 0, 0.5]]
        let ev = LinAlg.eigenvalues(m).sorted { $0.im < $1.im }
        XCTAssertEqual(ev[0].re, c, accuracy: 1e-10)
        XCTAssertEqual(ev[0].im, -s, accuracy: 1e-10)
        XCTAssertEqual(ev[1].re, 0.5, accuracy: 1e-10)
        XCTAssertEqual(ev[2].im, s, accuracy: 1e-10)
    }

    func testNNLS() {
        let A = [[1.0, 0.5], [0.5, 1.0], [0.2, 0.1]]
        let xTrue = [2.0, 0.0]
        let b = A.map { $0[0] * xTrue[0] + $0[1] * xTrue[1] - 0.0 }
        let x = LinAlg.nnls(A, b.map { $0 - 0.3 * 0 })
        XCTAssertEqual(x[0], 2, accuracy: 1e-9)
        XCTAssertEqual(x[1], 0, accuracy: 1e-9)
        // negative unconstrained solution must clamp at 0
        let x2 = LinAlg.nnls([[1, 0], [0, 1]], [1, -1])
        XCTAssertEqual(x2[0], 1, accuracy: 1e-12)
        XCTAssertEqual(x2[1], 0, accuracy: 1e-12)
    }

    func testMatrixPencilRecoversDampedSinusoids() {
        let fs = 5512.5
        let truth = [(62.0, 0.12, 0.8, 0.3), (143.0, 0.05, 0.3, -1.2), (311.0, 0.02, 0.2, 2.0)]
        let x = (0..<1200).map { n -> Double in
            let t = Double(n) / fs
            return truth.reduce(0) { $0 + $1.2 * exp(-t / $1.1) * sin(2 * Double.pi * $1.0 * t + $1.3) }
        }
        let r = MatrixPencil.estimate(x, sampleRate: fs)
        for (f, d, a, p) in truth {
            guard let m = r.modes.min(by: { abs($0.frequency - f) < abs($1.frequency - f) }) else {
                return XCTFail("no modes")
            }
            XCTAssertEqual(m.frequency, f, accuracy: f * 1e-4)
            XCTAssertEqual(m.decay, d, accuracy: d * 1e-3)
            XCTAssertEqual(m.amplitude, a, accuracy: a * 1e-3)
            XCTAssertEqual(cos(m.phase - p), 1, accuracy: 1e-5)
        }
    }

    func testMetricsZeroForIdentical() {
        let x = (0..<8192).map { sin(Double($0) * 0.05) * exp(-Double($0) / 2000) }
        XCTAssertEqual(KickMetrics.bandScore(x, x), 0, accuracy: 1e-12)
        XCTAssertEqual(KickMetrics.mrScore(x, x), 0, accuracy: 1e-12)
        XCTAssertGreaterThan(KickMetrics.mrScore(x.map { $0 * 0.5 }, x), 0)
    }
}

/// Synthetic kicks rendered from known §3.1 parameters.
enum Synth {
    static let sr = 44100.0

    static func kick(noise: Bool, E: Double = 0.4, Fd: Double = 0.0005, second: Bool = false) -> KickVoiceParams {
        var modes = [
            ModeParams(F: 52, G: 2.2, T: 0.018, A: 0.75, D: 0.16, R: 0.0006, phi: 0.4,
                       G2: second ? 1.5 : 0, T2: second ? 0.003 : 0),
            ModeParams(F: 97, A: 0.12, D: 0.07, R: 0.0007, phi: 1.1),
            ModeParams(F: 183, A: 0.08, D: 0.045, R: 0.0007, phi: -2.0),
            ModeParams(F: 1530, A: 0.03, D: 0.012, R: 0.0007, phi: 0.7),
            ModeParams(F: 2710, A: 0.02, D: 0.008, R: 0.0007, phi: 2.6),
        ]
        if !noise { modes[0].A = 0.8 }
        let edges = NoiseBands.edges(count: 16, lo: 40, hi: 16000)
        var bands: [NoiseBandParams] = []
        if noise {
            for b in 0..<16 {
                let lvl = 0.02 * pow(0.85, Double(b))
                bands.append(NoiseBandParams(lo: edges[b], hi: edges[b + 1], L: lvl, m: 0.8, d1: 0.004, d2: 0.06))
            }
        }
        return KickVoiceParams(modes: modes, noiseBands: bands, rn: 0.0004, gate: GateParams(E: E, Fd: Fd))
    }

    /// Prepared-equivalent audio: 2 ms of silence is NOT added; the voice
    /// starts at t = 0, so the analysis' prepared origin sits a little
    /// before the true start (onset is a -40 dB threshold).
    static func audio(_ p: KickVoiceParams, seconds: Double) -> WavAudio {
        let n = Int(seconds * sr)
        let y = KickVoice.render(p, count: n, sampleRate: sr, seed: 12345)
        return WavAudio(sampleRate: sr, channels: [y, y])
    }
}

final class KickAnalysisTests: XCTestCase {
    func analyze(_ p: KickVoiceParams, seconds: Double) -> (KickAnalysis, PreparedTarget) {
        var dbg = KickDebug()
        return KickAnalyzer.analyze(name: "synthetic", audio: Synth.audio(p, seconds: seconds), debug: &dbg)
    }

    /// Analysis parameters live in prepared time; shift truth into it.
    func shiftT(_ prep: PreparedTarget) -> Double { -Double(prep.cropStart) / Synth.sr }

    func checkRecovery(_ truth: KickVoiceParams, _ a: KickAnalysis, _ prep: PreparedTarget,
                       decayTol: Double = 0.10, ampTol: Double = 0.25,
                       file: StaticString = #filePath, line: UInt = #line) {
        let dt = shiftT(prep)  // prepared t = truth t + dt  (dt <= 0)
        let f0 = truth.modes[0]
        let s0 = a.modes[0]
        // drop law compared as a curve in prepared time (G, T are shift-covariant via G e^{dt/T})
        XCTAssertEqual(s0.F, f0.F, accuracy: f0.F * 0.01, "F0", file: file, line: line)
        for tt in [0.003, 0.01, 0.02, 0.04, 0.08] {
            let fTrue = f0.frequency(tt)
            let fEst = s0.params.frequency(tt - dt)
            XCTAssertEqual(fEst, fTrue, accuracy: fTrue * 0.03, "f(t=\(tt))", file: file, line: line)
        }
        XCTAssertEqual(s0.T, f0.T, accuracy: f0.T * 0.05, "T0", file: file, line: line)
        XCTAssertEqual(s0.G, f0.G * exp(dt / f0.T), accuracy: f0.G * 0.05, "G0", file: file, line: line)
        for m in truth.modes.dropFirst() {
            guard let e = a.modes.dropFirst().min(by: { abs($0.F - m.F) < abs($1.F - m.F) }) else {
                XCTFail("no mode near \(m.F)", file: file, line: line)
                continue
            }
            XCTAssertEqual(e.F, m.F, accuracy: m.F * 0.005, "mode F \(m.F)", file: file, line: line)
            XCTAssertEqual(e.D, m.D, accuracy: m.D * decayTol, "mode D \(m.F)", file: file, line: line)
            XCTAssertEqual(e.A, m.A, accuracy: m.A * ampTol, "mode A \(m.F)", file: file, line: line)
        }
        XCTAssertEqual(a.gate.E - dt, truth.gate.E, accuracy: 0.002, "gate E", file: file, line: line)
    }

    func testCleanSyntheticKick() {
        let truth = Synth.kick(noise: false)
        let (a, prep) = analyze(truth, seconds: 0.4)
        checkRecovery(truth, a, prep)
        XCTAssertLessThan(a.check!.band, 1.0)
    }

    /// Decaying broadband noise biases damped-mode decays a little (it is
    /// itself a decaying narrowband process inside each mode's band), so
    /// decays get 15% and amplitudes 40% here; frequencies keep 0.5%.
    func testNoisyChoppedSyntheticKick() {
        let truth = Synth.kick(noise: true, E: 0.3)
        let (a, prep) = analyze(truth, seconds: 0.3)
        checkRecovery(truth, a, prep, decayTol: 0.15, ampTol: 0.4)
        XCTAssertTrue(a.gate.chopped)
        // modes not matching a true one (0.5%) must be weak: < 1/3 of the weakest true amplitude
        let weakest = truth.modes.map { $0.A }.min()!
        for m in a.modes where !truth.modes.contains(where: { abs($0.F / m.F - 1) < 0.005 }) {
            XCTAssertLessThan(m.A, weakest / 3, "spurious mode \(m.F) Hz")
        }
        // the full resynthesis (modes + fitted noise) is close to the target
        XCTAssertLessThan(a.check!.band, 1.5)
        // noise power above 1 kHz at 5 and 30 ms within 3 dB. Per-band levels
        // of overlapping 2nd-order bands are only weakly identifiable, and
        // below ~500 Hz the noise also absorbs the modal misfit of the chirp.
        for tt in [0.005, 0.03] {
            let pT = truth.noiseBands.filter { $0.lo >= 1000 }.reduce(0.0) { $0 + pow($1.envelope(tt, rn: truth.rn), 2) }
            let pE = a.noise.bands.filter { $0.lo >= 1000 }.reduce(0.0) { $0 + pow($1.envelope(tt, rn: a.noise.rn), 2) }
            XCTAssertEqual(10 * log10(pE), 10 * log10(pT), accuracy: 3, "noise power > 1 kHz at \(tt)")
        }
    }

    func testFadedEnd() {
        let truth = Synth.kick(noise: false, E: 0.35, Fd: 0.03)
        let (a, prep) = analyze(truth, seconds: 0.4)
        XCTAssertEqual(a.gate.E - shiftT(prep), 0.35, accuracy: 0.004)
        XCTAssertEqual(a.gate.Fd, 0.03, accuracy: 0.008)
        XCTAssertFalse(a.gate.chopped)
    }

    /// The fast term (T2 = 3 ms) is resolved only to the time resolution of
    /// the 40 Hz baseband, so the law is checked from 5 ms on (4%) and
    /// loosely at 2 ms.
    func testSecondDrop() {
        let truth = Synth.kick(noise: false, second: true)
        let (a, prep) = analyze(truth, seconds: 0.4)
        let dt = shiftT(prep)
        XCTAssertGreaterThan(a.modes[0].G2, 0, "second drop detected")
        XCTAssertEqual(a.modes[0].F, truth.modes[0].F, accuracy: truth.modes[0].F * 0.01)
        for (tt, tol) in [(0.002, 0.08), (0.005, 0.04), (0.01, 0.04), (0.02, 0.04), (0.05, 0.03)] {
            let fTrue = truth.modes[0].frequency(tt)
            XCTAssertEqual(a.modes[0].params.frequency(tt - dt), fTrue, accuracy: fTrue * tol, "f(\(tt))")
        }
    }

    /// Overlapping 2nd-order bands make per-band levels only weakly
    /// identifiable (and fast stages in narrow low bands not at all), so the
    /// criterion is the one that matters downstream: the band-envelope error
    /// of a resynthesis vs. the truth, relative to the realisation floor
    /// (truth rendered with another seed).
    func testNoiseOnlyEnvelopeFit() {
        let edges = NoiseBands.edges(count: 16, lo: 40, hi: 16000)
        let bands = (0..<16).map { b in
            NoiseBandParams(lo: edges[b], hi: edges[b + 1], L: 0.05 * pow(0.8, Double(b)), m: 0.7, d1: 0.005, d2: 0.08)
        }
        let n = Int(0.3 * Synth.sr)
        for shared in [true, false] {
            let x = KickVoice.renderNoise(bands, rn: 0.0005, count: n, sampleRate: Synth.sr, seed: 777,
                                          sharedWhite: shared)
            let x2 = KickVoice.renderNoise(bands, rn: 0.0005, count: n, sampleRate: Synth.sr, seed: 4242,
                                           sharedWhite: shared)
            var o = NoiseFit.Options()
            o.sharedWhite = shared
            let r = NoiseFit.fit(residual: x, sr: Synth.sr, onset: 0, end: 0.3, edges: edges, options: o)
            let y = KickVoice.renderNoise(r.bands, rn: r.rn, count: n, sampleRate: Synth.sr, seed: 4242,
                                          sharedWhite: shared)
            let floor = KickMetrics.bandScore(x2, x)
            let err = KickMetrics.bandScore(y, x)
            XCTAssertLessThan(err, floor * 1.25 + 0.3, "shared \(shared): fit \(err) vs realisation floor \(floor)")
            // broadband level and decay: total power at 10 ms and 100 ms within 2 dB
            for tt in [0.01, 0.1] {
                let pT = bands.reduce(0.0) { $0 + pow($1.envelope(tt, rn: 0.0005), 2) }
                let pE = r.bands.reduce(0.0) { $0 + pow($1.envelope(tt, rn: r.rn), 2) }
                // 3.5 dB: the 5 ms fast stage of the narrow low bands is below their time resolution
                XCTAssertEqual(10 * log10(pE), 10 * log10(pT), accuracy: 3.5, "shared \(shared) total power at \(tt)")
            }
        }
    }
}

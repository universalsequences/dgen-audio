import Foundation
import XCTest

@testable import DGenAnalysis

/// v3 analysis pieces on synthetic signals rendered with the CPU references of voice3:
/// crackle, noise events, strike sines.
final class RecordLayerTests: XCTestCase {
    let sr = Synth.sr

    /// Decaying hiss in 16 bands (independent of the crackle) + a decaying kick underneath.
    func floorSignal(count n: Int, level: Double, seed: UInt64) -> [Double] {
        let edges = NoiseBands.edges(count: 16, lo: 40, hi: 16000)
        let bands = (0..<16).map { b in
            NoiseBandParams(lo: edges[b], hi: edges[b + 1], L: level * pow(0.85, Double(b)), m: 0.6, d1: 0.01, d2: 0.3)
        }
        var y = KickVoice.renderNoise(bands, rn: 0.0005, count: n, sampleRate: sr, seed: seed, sections: 2)
        let kick = KickVoice.renderModes([ModeParams(F: 55, G: 2, T: 0.02, A: 0.8, D: 0.12, R: 0.0006)],
                                         count: n, sampleRate: sr)
        for i in 0..<n { y[i] += kick[i] }
        return y
    }

    func testCrackleRateLevelColourRecovery() {
        let n = Int(0.6 * sr)
        let truth = CrackleParams(level: 0.01, rate: 60, colour: [0.15, 1, 0.4, 0.05], decay: 10, present: true,
                                  detected: 0, scannedSeconds: 0, threshold: 0)
        let cr = Crackle.render(truth, count: n, sampleRate: sr, seed: 99)
        let floor = floorSignal(count: n, level: 0.002, seed: 7)
        // the kick stays in (its drop sweeps 165 -> 55 Hz, under the detector's 800 Hz high-pass)
        let resid = (0..<n).map { floor[$0] + cr[$0] }
        let (p, mask) = Crackle.fit(residual: resid, sr: sr, onset: 0.002, end: 0.6)
        XCTAssertTrue(p.present)
        XCTAssertEqual(p.rate, truth.rate, accuracy: 0.35 * truth.rate, "rate")
        // colour: the dominant band and the ordering of the two strongest
        XCTAssertEqual(p.colour.firstIndex(of: p.colour.max()!), 1, "dominant colour band \(p.colour)")
        XCTAssertGreaterThan(p.colour[2], p.colour[3], "colour \(p.colour)")
        // level vs the true peak scale: the generator's expected power within 4 dB of the truth's
        let pe = 10 * log10(Crackle.expectedPower(p, sampleRate: sr))
        let pt = 10 * log10(Crackle.expectedPower(truth, sampleRate: sr))
        XCTAssertEqual(pe, pt, accuracy: 4, "expected crackle power")
        XCTAssertLessThan(mask.reduce(0, +), Double(n), "clicks gated")
        // no crackle: hiss + kick only
        let (q, _) = Crackle.fit(residual: floor, sr: sr, onset: 0.002, end: 0.6)
        XCTAssertFalse(q.present, "spurious crackle: rate \(q.rate) detected \(q.detected)")
        XCTAssertEqual(q.level, Crackle.silentLevel)
    }

    func testCrackleGeneratorExpectedPower() {
        // E[a²] = 1 + ln 900 and Σh² of the colour: long renders converge to the closed form
        let p = CrackleParams(level: 1, rate: 800, colour: [1, 0.5, 0.25, 0.1], decay: 1e6, present: true,
                              detected: 0, scannedSeconds: 0, threshold: 0)
        var acc = 0.0
        let n = 4 * Int(sr)
        for seed in [1, 2, 3] as [UInt64] {
            let y = Crackle.render(p, count: n, sampleRate: sr, seed: seed)
            acc += y.reduce(0) { $0 + $1 * $1 } / Double(n)
        }
        let measured = acc / 3, expected = Crackle.expectedPower(p, sampleRate: sr)
        // heavy tail: the realised power is a noisy estimate (3 x 4 s)
        XCTAssertEqual(10 * log10(measured), 10 * log10(expected), accuracy: 1.5)
    }

    func testNoiseEventTimeLevelRecovery() {
        let n = Int(0.5 * sr)
        let edges = NoiseBands.edges(count: 8, lo: 40, hi: 16000)
        let bands = (0..<8).map { b in
            NoiseBandParams(lo: edges[b], hi: edges[b + 1], L: 0.03 * pow(0.8, Double(b)), m: 0.8, d1: 0.006, d2: 0.1)
        }
        // a hat-like burst at 300 ms on the top bands
        let colour = [0.001, 0.001, 0.001, 0.01, 0.05, 0.3, 1, 0.6]
        let ev = NoiseEventParams(t: 0.3, level: 0.02, decay: 0.04, rise: 0.001, colour: colour, kind: "secondary",
                                  gain: 0)
        let x = KickVoice.renderNoise(bands, rn: 0.0005, count: n, sampleRate: sr, seed: 31, sections: 2, events: [ev])
        var o = NoiseFit.Options()
        o.sections = 2
        o.maxEvents = 2
        let r = NoiseFit.fit(residual: x, sr: sr, onset: 0.0, end: 0.5, edges: edges, options: o)
        let evs = r.events ?? []
        guard let e = evs.min(by: { abs($0.t - 0.3) < abs($1.t - 0.3) }) else {
            return XCTFail("no event found; candidates \(r.eventCandidates ?? [])")
        }
        XCTAssertEqual(e.t, 0.3, accuracy: 0.01, "event time")
        // level at the colour peak band (the event's loudest band) within 4 dB
        let peakTruth = ev.level * colour.max()!
        let peakEst = e.level * e.colour.max()!
        XCTAssertEqual(20 * log10(peakEst), 20 * log10(peakTruth), accuracy: 4, "event level")
        XCTAssertEqual(e.colour.firstIndex(of: e.colour.max()!)!, 6, accuracy: 1, "event colour peak band")
        XCTAssertEqual(log(e.decay), log(ev.decay), accuracy: log(1.6), "event decay")
        XCTAssertLessThanOrEqual(evs.filter { abs($0.t - 0.3) > 0.03 }.count, 0, "spurious events \(evs)")
    }

    func testStrikeSineRecovery() {
        let n = Int(0.12 * sr)
        let onset = 0.002
        let truth = [
            ModeParams(F: 2600, G: 0.4, T: 0.004, A: 0.25, D: 0.006, R: StrikeFit.rise, phi: 0.5),
            ModeParams(F: 420, G: 0, T: 0.003, A: 0.4, D: 0.009, R: StrikeFit.rise, phi: -1.0),
        ]
        var x = KickVoice.renderModes(truth, count: n, sampleRate: sr)
        let w = NoiseBands.white(count: n, seed: 5)
        for i in 0..<n { x[i] += 0.002 * w[i] }
        var found: [ModeEstimate] = []
        for band in StrikeFit.regions {
            var o = StrikeFit.Options()
            o.count = 2
            o.fRange = band
            o.highpassHz = band[0]
            found += StrikeFit.fit(residual: x, sr: sr, onset: onset, options: o)
        }
        for m in truth {
            guard let e = found.min(by: { abs($0.F - m.F) < abs($1.F - m.F) }) else { return XCTFail("no strikes") }
            // settled frequency, the glide's start frequency and the decay
            XCTAssertEqual(e.F, m.F, accuracy: 0.03 * m.F, "strike F \(m.F)")
            for tt in [0.001, 0.003] {
                XCTAssertEqual(e.params.frequency(tt), m.frequency(tt), accuracy: 0.04 * m.frequency(tt),
                               "strike f(\(tt)) \(m.F)")
            }
            XCTAssertEqual(e.D, m.D, accuracy: 0.25 * m.D, "strike D \(m.F)")
            XCTAssertEqual(e.A, m.A, accuracy: 0.25 * m.A, "strike A \(m.F)")
        }
    }

    func testRowsGlideFloorKeepsTheSettledPhase() {
        // g = 0 plays as the box minimum: the replacement glide's phase integral is compensated
        let F = 1234.0
        let (g, T, dph) = KickRowsV3.glideFloor(F: F, g: 0, T: 0.03, gMin: 0.08, tMin: 0.00106)
        let a = ModeParams(F: F, G: 0, T: 0.03, A: 1, D: 1, phi: 0.3)
        let b = ModeParams(F: F, G: g, T: T, A: 1, D: 1, phi: 0.3 + dph)
        for t in [0.02, 0.05, 0.2] {
            XCTAssertEqual(remainder(a.phaseLaw(t) + a.phi - b.phaseLaw(t) - b.phi, 2 * Double.pi), 0, accuracy: 1e-6)
        }
        XCTAssertEqual(KickRowsV3.effectiveBox("g")!.0, 0.08, accuracy: 1e-12)
        XCTAssertEqual(KickRowsV3.effectiveBox("td")!.0, 0.001 * pow(500, 0.01), accuracy: 1e-12)
        XCTAssertEqual(KickRowsV3.layout.reduce(0) { $0 + $1.1 }, 251, "voice3 row width (with the onset delay t0)")
    }
}

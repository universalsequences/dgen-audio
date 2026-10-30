import Foundation
import XCTest

@testable import DGenAnalysis

/// Synthetic snares rendered from known parameters: a membrane head (Bessel-like mode ratios on one
/// shared glide), wires (two-stage noise bands) and a stationary record-hiss floor.
enum SnareSynth {
    static let sr = 44100.0
    /// circular membrane mode ratios (01, 11, 21, 02, 31, 12)
    static let ratios = [1.0, 1.594, 2.136, 2.296, 2.653, 2.918]

    static func edges() -> [Double] { NoiseBands.edges(count: 16, lo: 40, hi: 16000) }

    static func wires(hiss: Double) -> [NoiseBandParams] {
        let e = edges()
        return (0..<16).map { b in
            let fc = (e[b] * e[b + 1]).squareRoot()
            // wires live above ~1 kHz; the hiss floor is flat-ish
            let lvl = fc > 900 ? 0.06 * pow(0.9, Double(b - 9)) : 0.004
            return NoiseBandParams(lo: e[b], hi: e[b + 1], L: lvl, m: 0.6, d1: 0.006, d2: 0.12, H: hiss)
        }
    }

    static func snare(F0: Double = 200, G: Double = 0.05, T: Double = 0.01, hiss: Double = 0.002) -> KickVoiceParams {
        let amps = [0.6, 0.35, 0.25, 0.2, 0.15, 0.12], decs = [0.09, 0.07, 0.06, 0.05, 0.045, 0.04]
        var modes = zip(ratios, zip(amps, decs)).enumerated().map { k, v in
            ModeParams(F: F0 * v.0, G: G, T: T, A: v.1.0, D: v.1.1, R: 0.0006, phi: 0.3 + Double(k))
        }
        modes.append(ModeParams(F: 2150, G: G, T: T, A: 0.08, D: 0.02, R: 0.0006, phi: -1.0))
        return KickVoiceParams(modes: modes, noiseBands: wires(hiss: hiss), rn: 0.0008,
                               gate: GateParams(E: 0.6, Fd: 0.0005), noiseSections: 2)
    }

    static func audio(_ p: KickVoiceParams, seconds: Double) -> WavAudio {
        let y = KickVoice.render(p, count: Int(seconds * sr), sampleRate: sr, seed: 777)
        return WavAudio(sampleRate: sr, channels: [y, y])
    }
}

final class SnareAnalysisTests: XCTestCase {
    func db(_ a: Double, _ b: Double) -> Double { 20 * log10(max(a, 1e-12) / max(b, 1e-12)) }

    /// NoiseFit's floor term recovers the wires (decaying) and the hiss (stationary) of the same bands.
    func testNoiseFloorTermSeparatesWiresFromHiss() {
        var p = SnareSynth.snare()
        p.modes = []
        let n = Int(0.6 * SnareSynth.sr)
        let x = KickVoice.render(p, count: n, sampleRate: SnareSynth.sr, seed: 777)
        var o = NoiseFit.Options()
        o.floorTerm = true
        o.maxSlowDecay = 1.5
        o.sections = 2
        let r = NoiseFit.fit(residual: x, sr: SnareSynth.sr, onset: 0, end: 0.6, edges: SnareSynth.edges(), options: o)
        for (b, (fit, truth)) in zip(r.bands, p.noiseBands).enumerated() {
            let fc = (truth.lo * truth.hi).squareRoot()
            guard fc > 1500 && fc < 12000 else { continue }  // away from the synthetic 24 dB step at 900 Hz (the band prior smooths it)
            XCTAssertLessThan(abs(db(fit.L, truth.L)), 3, "band \(b) wire level")
            XCTAssertLessThan(abs(log(fit.d2 / truth.d2)), log(1.5), "band \(b) slow decay")
            XCTAssertLessThan(abs(db(fit.H ?? 0, truth.H!)), 6, "band \(b) hiss floor")
        }
        // without the floor term the stationary part must hide in the slow decay
        o.floorTerm = false
        o.maxSlowDecay = 3
        let r0 = NoiseFit.fit(residual: x, sr: SnareSynth.sr, onset: 0, end: 0.6, edges: SnareSynth.edges(), options: o)
        XCTAssertNil(r0.bands[12].H)
    }

    /// The snare preset recovers the head: slot 0 and the membrane modes, a small shared glide, and the
    /// hiss floor under the wires.
    func testSnarePresetRecoversHeadWiresAndHiss() {
        let p = SnareSynth.snare()
        var dbg = KickDebug()
        let (a, prep) = KickAnalyzer.analyze(name: "snare", audio: SnareSynth.audio(p, seconds: 0.6),
                                             options: KickAnalysisOptions.snare(), debug: &dbg)
        XCTAssertGreaterThan(prep.signal.count, 0)
        XCTAssertEqual(a.modes[0].F, 200, accuracy: 2)
        XCTAssertLessThanOrEqual(a.modes[0].G, 0.25 + 1e-9)
        // every mode follows slot 0's glide
        for m in a.modes { XCTAssertEqual(m.G, a.modes[0].G, accuracy: 1e-12) }
        for truth in p.modes {
            let near = a.modes.min { abs($0.F / truth.F - 1) < abs($1.F / truth.F - 1) }!
            XCTAssertLessThan(abs(near.F / truth.F - 1), 0.015, "mode \(truth.F) Hz not found (nearest \(near.F))")
        }
        // the floor term is on, and the high bands' hiss is found within 6 dB
        for (fit, truth) in zip(a.noise.bands, p.noiseBands) where (truth.lo * truth.hi).squareRoot() > 2000 {
            XCTAssertLessThan(abs(db((fit.H ?? 0) / prep.gain, truth.H!)), 6, "hiss \(truth.lo)-\(truth.hi) Hz")
        }
    }

    /// The snare rows writer fills the snare1 layout (every segment, the right widths, inside its box).
    func testSnareRowsLayout() {
        var dbg = KickDebug()
        let (a, _) = KickAnalyzer.analyze(name: "snare", audio: SnareSynth.audio(SnareSynth.snare(), seconds: 0.6),
                                          options: KickAnalysisOptions.snare(), debug: &dbg)
        let template = KickSlotTemplate.estimate([a], slots: 8)
        let strikes = KickSlotTemplate.estimateStrikes([a], perRegion: 2)
        let row = SnareRows.row(a, slots: 8, template: template, assignment: KickSlots.assignV3(a, template: template),
                                strikeTemplate: strikes, strikeSlots: KickSlots.assignStrikes(a, template: strikes))
        let layout = SnareRows.layout(slots: 8)
        XCTAssertEqual(Set(row.keys), Set(layout.map { $0.0 }))
        for (name, width, _, _) in layout {
            XCTAssertEqual(row[name]?.count, width, name)
            if let (lo, hi) = SnareRows.effectiveBox(name, slots: 8) {
                for v in row[name]! { XCTAssert(v >= lo && v <= hi, "\(name) \(v) outside [\(lo), \(hi)]") }
            }
        }
        XCTAssertEqual(row["f"]![0], 200, accuracy: 2)
    }
}

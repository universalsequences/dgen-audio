import XCTest

@testable import DGenTrainProtocol

/// Excitation measurement (spec §6) against synthesized known assets:
/// a drum-style one-shot (exponentially decaying low sine) and a
/// sustained tone.
final class ExcitationTests: XCTestCase {
    let sampleRate: Float = 44100

    func drumOneShot(hz: Float = 55, seconds: Float = 1.2, tau: Float = 0.12) -> [Float] {
        let n = Int(seconds * sampleRate)
        return (0..<n).map { i in
            let t = Float(i) / sampleRate
            return exp(-t / tau) * sin(2 * .pi * hz * t)
        }
    }

    func sustainedTone(hz: Float = 220, seconds: Float = 1.0, fade: Float = 0.05) -> [Float] {
        let n = Int(seconds * sampleRate)
        let fadeSamples = Int(fade * sampleRate)
        return (0..<n).map { i in
            let t = Float(i) / sampleRate
            let env = i >= n - fadeSamples ? Float(n - i) / Float(fadeSamples) : 1.0
            return 0.8 * env * sin(2 * .pi * hz * t)
        }
    }

    func testPitchEstimateDrumOneShot() throws {
        let pitch = try XCTUnwrap(
            Excitation.estimatePitchHz(samples: drumOneShot(), sampleRate: sampleRate))
        XCTAssertEqual(pitch, 55, accuracy: 2.0)
    }

    func testPitchEstimateSustained() throws {
        let pitch = try XCTUnwrap(
            Excitation.estimatePitchHz(samples: sustainedTone(), sampleRate: sampleRate))
        XCTAssertEqual(pitch, 220, accuracy: 2.0)
    }

    func testPitchEstimateSilenceIsNil() {
        XCTAssertNil(
            Excitation.estimatePitchHz(
                samples: [Float](repeating: 0, count: 44100), sampleRate: sampleRate))
    }

    func testGateFramesDrumReleasesEarly() {
        let samples = drumOneShot()
        let gate = Excitation.gateFrames(samples: samples)
        // Envelope hits 0.15x peak at t = tau*ln(1/0.15) ~ 0.228 s ~ 10K samples.
        XCTAssertGreaterThan(gate, 5000)
        XCTAssertLessThan(gate, 20000)
        XCTAssertLessThan(Float(gate), 0.3 * Float(samples.count),
                          "one-shot gate must release well before the sample ends")
    }

    func testGateFramesSustainedStaysOpen() {
        let samples = sustainedTone()
        let gate = Excitation.gateFrames(samples: samples)
        XCTAssertGreaterThan(Float(gate), 0.8 * Float(samples.count),
                             "sustained gate should stay open until near the end")
    }

    func testGateFramesSilenceFallsBackToFullLength() {
        let samples = [Float](repeating: 0, count: 4096)
        XCTAssertEqual(Excitation.gateFrames(samples: samples), 4096)
    }

    func testCropFramesClamps() {
        XCTAssertEqual(Excitation.cropFrames(sampleCount: 200_000), 65536)
        XCTAssertEqual(Excitation.cropFrames(sampleCount: 500), 1024)
        XCTAssertEqual(Excitation.cropFrames(sampleCount: 32768), 32768)
    }
}

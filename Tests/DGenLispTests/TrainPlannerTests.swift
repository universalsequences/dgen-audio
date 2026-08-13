import DGen
import DGenLazy
import DGenTrainProtocol
import XCTest

@testable import DGenLisp

/// Plan-event tests for the training lowering pass (spec §7):
/// (a) fully learnable, (b) frozen f0, (c) unsupported sync node,
/// (d) missing bounds — plus macro transparency of the freeze policy.
final class TrainPlannerTests: XCTestCase {
    override func setUpWithError() throws {
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 44100
        LazyGraphContext.reset()
    }

    func makeOptions(pitchHz: Double? = 110, gateFrames: Int? = 4000) -> TrainOptions {
        var options = TrainOptions(
            patchPath: "unused", targetPath: "unused",
            seedParamsPath: "unused", jobDirPath: "unused")
        options.pitchHz = pitchHz
        options.gateFrames = gateFrames
        return options
    }

    func plan(
        _ source: String,
        seed: [String: Double] = [:],
        options: TrainOptions? = nil,
        targetLength: Int = 8192
    ) throws -> PatchPlan {
        LazyGraphContext.reset()
        let target = (0..<targetLength).map { i in
            Float(sin(2.0 * Double.pi * 110.0 * Double(i) / 44100.0))
        }
        let (patchPlan, _) = try TrainPlanner.makePlan(
            patchSource: source,
            assetBase: URL(fileURLWithPath: NSTemporaryDirectory()),
            seed: SeedParams(params: seed),
            targetSamples: target,
            targetSampleRate: 44100,
            options: options ?? makeOptions())
        return patchPlan
    }

    // (a) fully learnable patch — golden line for the exact plan JSON.
    func testFullyLearnablePatchGolden() throws {
        let source = """
            (def cutoff (param cutoff @default 1000 @min 100 @max 8000))
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (+ (phasor 110) cutoff)) 0)
            """
        let patchPlan = try plan(source, seed: ["cutoff": 1200, "amp": 0.9])
        XCTAssertEqual(patchPlan.plan.learnable, ["amp", "cutoff"])
        XCTAssertTrue(patchPlan.plan.frozen.isEmpty)
        XCTAssertTrue(patchPlan.plan.unsupported.isEmpty)
        XCTAssertTrue(patchPlan.fatalUnsupported.isEmpty)

        // Seed echo is verbatim and start values come from the seed.
        XCTAssertEqual(patchPlan.plan.seedEcho, ["cutoff": 1200, "amp": 0.9])
        XCTAssertEqual(
            patchPlan.learnable.first { $0.name == "cutoff" }?.seedValue, 1200)
        XCTAssertEqual(patchPlan.learnable.first { $0.name == "amp" }?.seedValue, 0.9)

        let line = try TrainEventCoding.encodeLine(.plan(patchPlan.plan))
        XCTAssertEqual(
            line,
            #"{"crop_frames":8192,"frozen":[],"gate_frames":4000,"learnable":["amp","cutoff"],"pitch_hz":110,"seed_echo":{"amp":0.9,"cutoff":1200},"type":"plan","unsupported":[]}"#
        )
    }

    // (b) param feeding a phasor frequency is frozen.
    func testPhasorFrequencyParamFrozen() throws {
        let source = """
            (def freq (param freq @default 110 @min 50 @max 400))
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor freq)) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(patchPlan.plan.learnable, ["amp"])
        XCTAssertEqual(
            patchPlan.plan.frozen,
            [ParamVerdict(name: "freq", reason: TrainPlanner.reasonPitchDetached)])
    }

    // (b') the freeze policy sees through defs AND macro expansion.
    func testPhasorFrequencyFrozenThroughMacro() throws {
        let source = """
            (defmacro myosc (f) (phasor (* f 1.0)))
            (def freq (param freq @default 110 @min 50 @max 400))
            (out (myosc freq) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(
            patchPlan.plan.frozen,
            [ParamVerdict(name: "freq", reason: TrainPlanner.reasonPitchDetached)])
        XCTAssertTrue(patchPlan.plan.learnable.isEmpty)
    }

    // (b'') stateful-phasor frequency params freeze identically.
    func testStatefulPhasorFrequencyFrozen() throws {
        let source = """
            (def freq (param freq @default 110 @min 50 @max 400))
            (out (stateful-phasor freq) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(
            patchPlan.plan.frozen,
            [ParamVerdict(name: "freq", reason: TrainPlanner.reasonPitchDetached)])
    }

    // (b''') mixed-path params stay learnable: the pitch path is severed
    // by stop-gradient, but the amplitude path still carries gradient.
    func testMixedPathParamStaysLearnable() throws {
        let source = """
            (def bright (param bright @default 1.0 @min 0.1 @max 5.0))
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (* bright (phasor (* 110 bright)))) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(patchPlan.plan.learnable, ["amp", "bright"])
        XCTAssertTrue(patchPlan.plan.frozen.isEmpty)
    }

    // Params with no path to the output at all are frozen distinctly.
    func testUnusedParamFrozen() throws {
        let source = """
            (def unused (param unused @default 0.5 @min 0 @max 1))
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor 110)) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(patchPlan.plan.learnable, ["amp"])
        XCTAssertEqual(
            patchPlan.plan.frozen,
            [ParamVerdict(name: "unused", reason: TrainPlanner.reasonNoGradPath)])
    }

    // (c) oscillator-driven phasor reset = hard sync -> unsupported + fatal.
    func testOscillatorSyncUnsupported() throws {
        let source = """
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor 110 (phasor 55))) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(patchPlan.plan.unsupported.count, 1)
        XCTAssertEqual(patchPlan.plan.unsupported.first?.reason, TrainPlanner.reasonSync)
        XCTAssertFalse(patchPlan.fatalUnsupported.isEmpty)
    }

    // (c') a click-reset phasor is ordinary voice behavior, NOT sync.
    func testClickResetIsNotSync() throws {
        let source = """
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor 110 (click))) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertTrue(patchPlan.plan.unsupported.isEmpty)
    }

    // (d) params without @min/@max are refused-and-reported, not guessed.
    func testMissingBoundsFrozen() throws {
        let source = """
            (def gain (param gain @default 0.5))
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (* gain (phasor 110))) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(patchPlan.plan.learnable, ["amp"])
        XCTAssertEqual(
            patchPlan.plan.frozen,
            [ParamVerdict(name: "gain", reason: TrainPlanner.reasonMissingBounds)])
    }

    // Seed values are clamped into declared bounds.
    func testSeedClampedToBounds() throws {
        let source = """
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor 110)) 0)
            """
        let patchPlan = try plan(source, seed: ["amp": 3.5])
        XCTAssertEqual(patchPlan.learnable.first?.seedValue, 1.0)
        // But the echo stays verbatim — the host must see what was parsed.
        XCTAssertEqual(patchPlan.plan.seedEcho, ["amp": 3.5])
    }

    // Excitation defaults are measured from the target when not overridden.
    func testExcitationDefaultsMeasured() throws {
        let source = """
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor 110)) 0)
            """
        let patchPlan = try plan(
            source, options: makeOptions(pitchHz: nil, gateFrames: nil),
            targetLength: 16384)
        XCTAssertEqual(patchPlan.plan.pitchHz, 110, accuracy: 2.0)
        XCTAssertEqual(patchPlan.plan.cropFrames, 16384)
        XCTAssertGreaterThan(patchPlan.plan.gateFrames, 0)
        XCTAssertLessThanOrEqual(patchPlan.plan.gateFrames, 16384)
    }

    // Patches without outputs fail with a clear message.
    func testNoOutputsRejected() {
        XCTAssertThrowsError(try plan("(param amp @default 0.5 @min 0 @max 1)")) { error in
            XCTAssertTrue("\(error)".contains("no outputs"))
        }
    }

    // lowered.lisp carries the verdict annotation header.
    func testLoweredSourceAnnotation() throws {
        let source = """
            (def freq (param freq @default 110 @min 50 @max 400))
            (out (phasor freq) 0)
            """
        let patchPlan = try plan(source)
        let lowered = TrainPlanner.loweredSource(patchPlan: patchPlan)
        XCTAssertTrue(lowered.contains("; frozen: freq (pitch-path-detached)"))
        XCTAssertTrue(lowered.contains("(out (phasor freq) 0)"))
        // lowered.lisp must be re-parseable (train-render consumes it).
        XCTAssertNoThrow(try parseSource(lowered))
    }

    // `(in ...)` inlets are rewritten to the excitation convention; the
    // driven graph is what gets classified and trained.
    func testInputInletsDrivenByExcitation() throws {
        let source = """
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* (in 1 @name gate) (* amp (* (in 2 @name velocity) (phasor (in 3 @name pitch))))) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertTrue(patchPlan.plan.unsupported.isEmpty)
        XCTAssertEqual(patchPlan.plan.learnable, ["amp"])
        let lowered = TrainPlanner.loweredSource(patchPlan: patchPlan)
        XCTAssertFalse(lowered.contains("(in "), "all inlets should be rewritten")
        XCTAssertTrue(lowered.contains("(accum 1.0 0.0 0.0 1000000000.0)"), lowered)
        XCTAssertTrue(lowered.contains("110.0"), "pitch inlet becomes the frozen pitch constant")
    }

    // Inlets outside the convention are refused, not silently zeroed.
    func testUnknownInletRefused() throws {
        let source = """
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (in 1 @name aftertouch)) 0)
            """
        let patchPlan = try plan(source)
        XCTAssertEqual(
            patchPlan.plan.unsupported,
            [ParamVerdict(name: "in#aftertouch", reason: TrainPlanner.reasonUndrivenInput)])
        XCTAssertFalse(patchPlan.fatalUnsupported.isEmpty)
    }
}

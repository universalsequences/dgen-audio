import DGen
import DGenLazy
import DGenTrainProtocol
import XCTest

@testable import DGenLisp

final class AnalyticADSRLoweringTests: XCTestCase {
    func testRewritesCallsOnlyAndLeavesRenderTreeExact() throws {
        let nodes = try parseSource("""
            (defmacro adsr (gate trigger attack decay sustain release) gate)
            (def env (adsr gate trigger attack decay sustain release))
            """)
        let lowered = try AnalyticADSRLowering.lower(nodes: nodes, gateFrames: 24000)
        let source = lowered.map(ExcitationLowering.printAST).joined(separator: "\n")
        let rendering = nodes.map(ExcitationLowering.printAST).joined(separator: "\n")

        XCTAssertTrue(source.contains("(def __dgen_train_sample_index (accum 1.0"))
        XCTAssertTrue(source.contains("(defmacro adsr "))
        XCTAssertTrue(source.contains("(__dgen_train_analytic_adsr attack decay sustain release)"))
        XCTAssertFalse(source.contains("(__dgen_train_analytic_adsr gate trigger"))
        XCTAssertTrue(rendering.contains("(adsr gate trigger attack decay sustain release)"))
        XCTAssertFalse(rendering.contains("__dgen_train_analytic_adsr"))
    }

    func testPlannerUsesAnalyticADSRForTrainingButNotRendering() throws {
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 8_000
        LazyGraphContext.reset()
        let source = """
            (defmacro adsr (gate trigger attack decay sustain release) gate)
            (def gate (in 0 @name gate))
            (def trigger (in 1 @name trigger))
            (param attack @default 0 @min 0 @max 1000)
            (param decay @default 100 @min 0 @max 1000)
            (param sustain @default 0.5 @min 0 @max 1)
            (param release @default 100 @min 0 @max 1000)
            (out (* (phasor 220) (adsr gate trigger attack decay sustain release)) 0)
            """
        var options = TrainOptions(
            patchPath: "unused", targetPath: "unused",
            seedParamsPath: "unused", jobDirPath: "unused")
        options.pitchHz = 220
        options.gateFrames = 128
        options.filterSurrogate = "none"
        let target = (0..<256).map { Float(sin(2 * Double.pi * 220 * Double($0) / 8_000)) }

        let (plan, evaluator) = try TrainPlanner.makePlan(
            patchSource: source, assetBase: URL(fileURLWithPath: NSTemporaryDirectory()),
            seed: SeedParams(params: [:]), targetSamples: target,
            targetSampleRate: 8_000, options: options)
        let samples = try XCTUnwrap(evaluator.outputs.first?.signal).realize(frames: 256)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        let training = TrainPlanner.loweredSource(patchPlan: plan)
        let rendering = TrainPlanner.renderSource(patchPlan: plan)

        XCTAssertTrue(training.contains("__dgen_train_analytic_adsr"))
        XCTAssertFalse(training.contains("make-history"))
        XCTAssertFalse(rendering.contains("__dgen_train_analytic_adsr"))
        XCTAssertTrue(rendering.contains("(adsr gate trigger attack decay sustain release)"))
        XCTAssertEqual(Set(plan.plan.learnable), Set(["attack", "decay", "sustain", "release"]))
    }
}

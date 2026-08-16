import DGen
import DGenLazy
import DGenTrainProtocol
import XCTest

@testable import DGenLisp

final class FilterSurrogateTests: XCTestCase {
    func testLoweringRewritesCallsButKeepsMacroAndRenderAST() throws {
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 8_000
        LazyGraphContext.reset()
        let source = """
            (defmacro svf (input cutoff q mode) input)
            (def cutoff (param cutoff @default 900 @min 100 @max 3000))
            (out (svf (phasor 220) cutoff 1 0) 0)
            """
        var options = TrainOptions(
            patchPath: "unused", targetPath: "unused",
            seedParamsPath: "unused", jobDirPath: "unused")
        options.pitchHz = 220
        options.gateFrames = 1024
        // Surrogate mode defaults off; this test exercises the opt-in rewrite.
        options.filterSurrogate = "freq"
        options.surrogateWindow = 64
        options.surrogateHop = 16
        let target = (0..<2048).map {
            Float(sin(2 * Double.pi * 220 * Double($0) / 8_000))
        }
        let (plan, _) = try TrainPlanner.makePlan(
            patchSource: source, assetBase: URL(fileURLWithPath: NSTemporaryDirectory()),
            seed: SeedParams(params: [:]), targetSamples: target,
            targetSampleRate: 8_000, options: options)

        let training = TrainPlanner.loweredSource(patchPlan: plan)
        let rendering = TrainPlanner.renderSource(patchPlan: plan)
        XCTAssertTrue(training.contains("(defmacro svf "))
        XCTAssertTrue(training.contains("(svf-freq (phasor 220) cutoff 1 0 @window 64 @hop 16)"))
        XCTAssertFalse(rendering.contains("svf-freq"))
        XCTAssertTrue(rendering.contains("(svf (phasor 220) cutoff 1 0)"))
    }

    func testLispSVFFreqEvaluates() throws {
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 8_000
        DGenConfig.maxFrameCount = 512
        LazyGraphContext.reset()
        let evaluator = LispEvaluator()
        try evaluator.evaluate(source: "(out (svf-freq (phasor 220) 900 1 5 @window 64 @hop 16) 0)")
        let output = try XCTUnwrap(evaluator.outputs.first?.signal)
        let samples = try output.realize(frames: 512)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(samples.map(abs).max() ?? 0, 0.1)
    }
}

// RealTrainer.swift — the non-fake `dgenlisp train` backend.
//
// Phase B: loads the target, runs the lowering pass, emits the real plan
// event, writes lowered.lisp, and fails fast on unsupported nodes or
// --plan-only. Phase C wires the E4 direction-mode trainer after the plan.

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

enum RealTrainer {
    static func run(
        options: TrainOptions, sink: TrainEventSink, jobDir: JobDir
    ) throws -> ResultEvent {
        let seed = try SeedParams.load(url: URL(fileURLWithPath: options.seedParamsPath))
        let patchURL = URL(fileURLWithPath: options.patchPath)
        let patchSource: String
        do {
            patchSource = try String(contentsOf: patchURL, encoding: .utf8)
        } catch {
            throw TrainProtocolError("cannot read patch: \(options.patchPath)")
        }

        // v1 target policy: mono (AudioFile mono-sums multichannel input).
        let target: (samples: [Float], sampleRate: Float)
        do {
            target = try AudioFile.load(
                url: URL(fileURLWithPath: options.targetPath), mono: true)
        } catch {
            throw TrainProtocolError("cannot load target wav: \(options.targetPath) (\(error))")
        }
        guard !target.samples.isEmpty else {
            throw TrainProtocolError("target wav is empty: \(options.targetPath)")
        }

        // Plan/lowering runs on the C backend — no GPU time before the plan.
        DGenConfig.backend = .c
        DGenConfig.sampleRate = target.sampleRate
        LazyGraphContext.reset()

        let (patchPlan, _) = try TrainPlanner.makePlan(
            patchSource: patchSource,
            assetBase: patchURL.deletingLastPathComponent(),
            seed: seed,
            targetSamples: target.samples,
            targetSampleRate: target.sampleRate,
            options: options)
        try sink.emit(.plan(patchPlan.plan))

        try TrainPlanner.loweredSource(patchPlan: patchPlan)
            .data(using: .utf8)!
            .write(to: jobDir.loweredLisp, options: .atomic)

        if !patchPlan.fatalUnsupported.isEmpty {
            let described = patchPlan.fatalUnsupported
                .map { "\($0.name) (\($0.reason))" }
                .joined(separator: ", ")
            throw TrainProtocolError("patch contains unsupported nodes: \(described)")
        }
        guard !patchPlan.learnable.isEmpty else {
            throw TrainProtocolError("no learnable params (all frozen or unbounded)")
        }
        if options.planOnly {
            throw TrainProtocolError("plan-only: no training performed")
        }

        return try DirectionTrainer.train(
            options: options,
            patchPlan: patchPlan,
            targetSamples: target.samples,
            targetSampleRate: target.sampleRate,
            sink: sink,
            jobDir: jobDir)
    }
}

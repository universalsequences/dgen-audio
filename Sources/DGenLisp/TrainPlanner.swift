// TrainPlanner.swift — the training lowering pass's verdict (spec §7) plus
// excitation measurement (§6), producing the plan event (§4).
//
// Policy, not smoothing: params are transcribed verbatim; known-bad
// gradient paths are frozen; genuinely hard discontinuities (oscillator
// sync) are refused. Classification runs on the evaluated lazy graph, so
// defs and macro expansion are seen through.

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

struct LearnableParam {
    let name: String
    let min: Float
    let max: Float
    /// Training start value: the host's seed when present, else @default.
    let seedValue: Float
}

struct PatchPlan {
    let plan: PlanEvent
    let learnable: [LearnableParam]
    /// Non-empty means the job must fail fast right after emitting the plan.
    let fatalUnsupported: [ParamVerdict]
    /// Modulation-lowered AST with `(in ...)` inlets rewritten to the
    /// excitation convention — the graph training actually runs on.
    let loweredNodes: [ASTNode]
}

enum TrainPlanner {
    /// Freeze/refuse reasons (stable strings — the host displays them).
    static let reasonF0 = "f0-adjoint-unreliable"
    static let reasonMissingBounds = "missing-bounds"
    static let reasonGenerated = "generated-param"
    static let reasonHidden = "hidden-param"
    static let reasonSync = "oscillator-sync"
    static let reasonUndrivenInput = "input-not-in-excitation-convention"

    /// Evaluates the patch into the CURRENT lazy graph (caller must have
    /// reset the context and configured DGenConfig) and classifies params.
    static func makePlan(
        patchSource: String,
        assetBase: URL,
        seed: SeedParams,
        targetSamples: [Float],
        targetSampleRate: Float,
        options: TrainOptions
    ) throws -> (PatchPlan, LispEvaluator) {
        // Excitation (spec §6) is measured before evaluation because the
        // inlet rewrite needs the chosen values. CLI overrides win.
        let cropFrames = Excitation.cropFrames(sampleCount: targetSamples.count)
        let pitchHz =
            options.pitchHz
            ?? Excitation.estimatePitchHz(samples: targetSamples, sampleRate: targetSampleRate)
            ?? 0
        guard pitchHz > 0 else {
            throw TrainProtocolError(
                "no confident pitch estimate for target; pass --pitch-hz explicitly")
        }
        let gateFrames = min(
            options.gateFrames ?? Excitation.gateFrames(samples: targetSamples),
            cropFrames)

        let evaluator = LispEvaluator(sourceDirectory: assetBase)
        let rewrite: ExcitationLowering.Rewrite
        do {
            let nodes = ExcitationLowering.stripModulation(nodes: try parseSource(patchSource))
            let lowered = try lowerModulation(in: nodes)
            rewrite = ExcitationLowering.drive(
                nodes: lowered, pitchHz: pitchHz, gateFrames: gateFrames)
            try evaluator.evaluate(nodes: rewrite.nodes)
        } catch let error as LispError {
            throw TrainProtocolError("patch parse/eval failed: \(error.message)")
        }
        guard !evaluator.outputs.isEmpty else {
            throw TrainProtocolError("patch has no outputs; add (out <signal> <channel>)")
        }

        let analysis = LazyGraphContext.current.analyzePhasorFrequencies()

        var learnable: [LearnableParam] = []
        var frozen: [ParamVerdict] = []
        for param in evaluator.params {
            if param.generatedKind != nil {
                frozen.append(ParamVerdict(name: param.name, reason: reasonGenerated))
                continue
            }
            if param.hidden {
                frozen.append(ParamVerdict(name: param.name, reason: reasonHidden))
                continue
            }
            if let cell = param.cellId, analysis.frequencyParamCells.contains(cell) {
                // Swept f0 fails fdcheck; trainable statefulPhasor frequency
                // corrupts other params' gradients ~10x. Freeze, don't guess.
                frozen.append(ParamVerdict(name: param.name, reason: reasonF0))
                continue
            }
            guard let minBound = param.min, let maxBound = param.max, minBound < maxBound else {
                // @min/@max ARE the search space; refuse to invent one.
                frozen.append(ParamVerdict(name: param.name, reason: reasonMissingBounds))
                continue
            }
            let seedValue = seed.params[param.name].map(Float.init) ?? param.defaultValue
            learnable.append(
                LearnableParam(
                    name: param.name, min: minBound, max: maxBound,
                    seedValue: Swift.min(Swift.max(seedValue, minBound), maxBound)))
        }

        var unsupported = analysis.syncPhasorNodeIds.map {
            ParamVerdict(name: "phasor#\($0)", reason: reasonSync)
        }
        unsupported += rewrite.undriven.map {
            ParamVerdict(name: $0, reason: reasonUndrivenInput)
        }

        for name in seed.params.keys.sorted()
        where !evaluator.params.contains(where: { $0.name == name }) {
            FileHandle.standardError.write(
                Data("[train] seed param '\(name)' not present in patch; ignored\n".utf8))
        }

        let plan = PlanEvent(
            learnable: learnable.map(\.name).sorted(),
            frozen: frozen.sorted { $0.name < $1.name },
            unsupported: unsupported,
            seedEcho: seed.params,
            pitchHz: pitchHz,
            gateFrames: gateFrames,
            cropFrames: cropFrames)
        return (
            PatchPlan(
                plan: plan, learnable: learnable, fatalUnsupported: unsupported,
                loweredNodes: rewrite.nodes),
            evaluator
        )
    }

    /// lowered.lisp: the patch the trainer actually trained on (inlets
    /// rewritten to the excitation convention), annotated with the lowering
    /// verdict (artifact-trail contract, spec §5). Re-parseable lisp.
    static func loweredSource(patchPlan: PatchPlan) -> String {
        let plan = patchPlan.plan
        var header = "; dgenlisp train lowering verdict\n"
        header += "; learnable: \(plan.learnable.joined(separator: " "))\n"
        for v in plan.frozen {
            header += "; frozen: \(v.name) (\(v.reason))\n"
        }
        for v in plan.unsupported {
            header += "; unsupported: \(v.name) (\(v.reason))\n"
        }
        header += "; excitation: pitch_hz=\(plan.pitchHz) gate_frames=\(plan.gateFrames) crop_frames=\(plan.cropFrames)\n"
        let body = patchPlan.loweredNodes.map(ExcitationLowering.printAST).joined(separator: "\n")
        return header + body + "\n"
    }
}

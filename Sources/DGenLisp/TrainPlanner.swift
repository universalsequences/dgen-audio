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
    /// Excitation/modulation lowering only. Checkpoints and final renders use
    /// these nodes so the real patch SVF is always rendered.
    let renderNodes: [ASTNode]
}

enum TrainPlanner {
    /// Freeze/refuse reasons (stable strings — the host displays them).
    /// Pitch-only params: training runs with stop-gradient at phasor
    /// frequency inputs (DGenGradientConfig.detachPhasorFrequency), so a
    /// param whose every path to the output crosses a frequency input
    /// provably receives zero gradient. Pitch is never learned.
    static let reasonPitchDetached = "pitch-path-detached"
    static let reasonNoGradPath = "no-gradient-path"
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
        let trainingNodes: [ASTNode]
        do {
            let nodes = ExcitationLowering.stripModulation(nodes: try parseSource(patchSource))
            let lowered = try lowerModulation(in: nodes)
            rewrite = ExcitationLowering.drive(
                nodes: lowered, pitchHz: pitchHz, gateFrames: gateFrames)
            let analyticEnvelopeNodes = try AnalyticADSRLowering.lower(
                nodes: rewrite.nodes, gateFrames: gateFrames)
            trainingNodes = options.filterSurrogate == "freq"
                ? FilterSurrogateLowering.lower(
                    nodes: analyticEnvelopeNodes, window: options.surrogateWindow,
                    hop: options.surrogateHop)
                : analyticEnvelopeNodes
            try evaluator.evaluate(nodes: trainingNodes)
        } catch let error as LispError {
            throw TrainProtocolError("patch parse/eval failed: \(error.message)")
        }
        guard !evaluator.outputs.isEmpty else {
            throw TrainProtocolError("patch has no outputs; add (out <signal> <channel>)")
        }

        let analysis = LazyGraphContext.current.analyzePhasorFrequencies()
        let gradReachable: Set<CellID>
        if let output = (evaluator.outputs.first { $0.channel == 0 } ?? evaluator.outputs.first) {
            gradReachable = LazyGraphContext.current.gradientReachableParamCells(
                from: output.signal)
        } else {
            gradReachable = []
        }

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
            if let cell = param.cellId, !gradReachable.contains(cell) {
                // Zero gradient by construction. Distinguish "all paths
                // cross a detached phasor frequency" from "not connected".
                let reason =
                    analysis.frequencyParamCells.contains(cell)
                    ? reasonPitchDetached : reasonNoGradPath
                frozen.append(ParamVerdict(name: param.name, reason: reason))
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
                loweredNodes: trainingNodes,
                renderNodes: rewrite.nodes),
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

    /// Re-parseable excitation-lowered source used only by render subprocesses.
    static func renderSource(patchPlan: PatchPlan) -> String {
        patchPlan.renderNodes.map(ExcitationLowering.printAST).joined(separator: "\n") + "\n"
    }
}

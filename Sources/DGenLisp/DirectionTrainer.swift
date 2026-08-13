// DirectionTrainer.swift — E4 direction-finding mode (SUBTRACTIVE_SPEC E4,
// patch-learn-spec §3): seeded short run + one cold restart as basin check.
//
// Loss: multi-resolution log-magnitude STFT L1 exactly as frozen in
// SynthID SPEC.md §4 (windows 256-2048, hop w/4, normalize, +0.1 linear
// term, log-eps 1e-3). No smoothing surrogates. Adam runs CPU-side in
// transformed coordinates (log for wide positive ranges), with bounds
// projection and a global LR at 2x the legacy production tone LR per
// BATCH_REFINE_FINDING.
//
// Each epoch rebuilds the whole graph (LazyGraphContext.reset + re-evaluate
// the lowered patch): parameter values live in the CPU-side optimizer state
// and are written into the fresh graph before backward, which sidesteps the
// stale-nodeId class of bugs entirely. Preview WAVs are rendered by
// re-invoking this executable (`train-render`) — realize() must never
// interleave with backward() in the same process (SPEC.md §5).

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

enum DirectionTrainer {
    static let defaultEpochs = 300
    static let logEvery = 10
    static let defaultCheckpointEvery = 25
    /// Range-normalized coordinates: 0.5% of a knob's range per Adam step
    /// (2e-2 railed wide params in ~50 steps and slammed gain into the
    /// zero-amplitude dead-start trap on the monologue fit).
    static let transformedLR: Float = 5e-3
    static let gradClip: Float = 1.0
    /// Cold restart beats the seeded run "decisively" below this ratio.
    static let basinDecisiveRatio: Float = 0.75

    static let spectralWindows = [256, 512, 1024, 2048]
    static let linearMagnitudeWeight: Float = 0.1
    static let logEpsilon: Float = 1e-3

    struct TransformedParam {
        let name: String
        let min: Float
        let max: Float
        let useLog: Bool

        init(_ p: LearnableParam) {
            name = p.name
            min = p.min
            max = p.max
            useLog = p.min > 0 && p.max / p.min >= 8
        }

        // All params train span-normalized: z in [0,1] across the declared
        // range (in log space for wide positive ranges, linear otherwise),
        // so an Adam step of `lr` uniformly means "lr fraction of the
        // range". Raw natural coordinates make wide knobs untrainable and
        // narrow knobs hot under Adam's normalized steps.
        var span: Float { useLog ? log(max) - log(min) : max - min }

        func toZ(_ natural: Float) -> Float {
            useLog ? (log(natural) - log(min)) / span : (natural - min) / span
        }
        func fromZ(_ z: Float) -> Float {
            let natural = useLog ? exp(log(min) + z * span) : min + z * span
            return Swift.min(Swift.max(natural, min), max)
        }
        func dNaturalDZ(_ z: Float) -> Float {
            useLog ? fromZ(z) * span : span
        }
        var zMin: Float { 0 }
        var zMax: Float { 1 }
    }

    static func train(
        options: TrainOptions,
        patchPlan: PatchPlan,
        targetSamples: [Float],
        targetSampleRate: Float,
        sink: TrainEventSink,
        jobDir: JobDir
    ) throws -> ResultEvent {
        let epochs = min(max(options.epochs ?? defaultEpochs, 1), 2000)
        let crop = patchPlan.plan.cropFrames
        let transforms = patchPlan.learnable.map(TransformedParam.init)

        // Target: peak-normalize to 0.9, fit to the crop (SPEC.md §4).
        let prepared = preparedTarget(targetSamples, frames: crop)

        let seedZ = zip(transforms, patchPlan.learnable).map { t, p in t.toZ(p.seedValue) }
        // Cold restart: deterministic transformed-midpoint init (never on
        // bounds; the honest generic init per SPEC.md §3/§5).
        let coldZ = transforms.map { ($0.zMin + $0.zMax) / 2 }

        let seeded = try runPhase(
            name: "train", initialZ: seedZ, epochs: epochs,
            transforms: transforms, patchPlan: patchPlan, target: prepared,
            sampleRate: targetSampleRate, crop: crop, options: options,
            sink: sink, jobDir: jobDir, emitCheckpoints: true)

        let cold = try runPhase(
            name: "basin-check", initialZ: coldZ, epochs: epochs,
            transforms: transforms, patchPlan: patchPlan, target: prepared,
            sampleRate: targetSampleRate, crop: crop, options: options,
            sink: sink, jobDir: jobDir, emitCheckpoints: false)

        let basinCheck =
            cold.bestLoss < basinDecisiveRatio * seeded.bestLoss ? "wrong_neighborhood" : "ok"

        try renderViaSubprocess(
            params: naturalValues(z: seeded.bestZ, transforms: transforms),
            jobDir: jobDir, out: jobDir.finalWav, frames: crop,
            sampleRate: targetSampleRate, options: options)

        var deltas: [String: ParamDelta] = [:]
        let finalNatural = naturalValues(z: seeded.bestZ, transforms: transforms)
        for p in patchPlan.learnable {
            deltas[p.name] = ParamDelta(
                from: Double(p.seedValue), to: Double(finalNatural[p.name] ?? p.seedValue))
        }
        let improvement =
            seeded.initLoss > 0
            ? 100.0 * Double(seeded.initLoss - seeded.bestLoss) / Double(seeded.initLoss)
            : 0
        return ResultEvent(
            improvementPct: improvement,
            absDistance: Double(seeded.bestLoss),
            basinCheck: basinCheck,
            deltas: deltas,
            finalWav: jobDir.finalWav.path)
    }

    // MARK: - One training phase (seeded run or cold basin check)

    struct PhaseResult {
        var initLoss: Float
        var bestLoss: Float
        var bestZ: [Float]
    }

    private static func runPhase(
        name: String, initialZ: [Float], epochs: Int,
        transforms: [TransformedParam], patchPlan: PatchPlan,
        target: [Float], sampleRate: Float, crop: Int,
        options: TrainOptions, sink: TrainEventSink, jobDir: JobDir,
        emitCheckpoints: Bool
    ) throws -> PhaseResult {
        try sink.emit(.stage(StageEvent(name: name, total: epochs)))

        var z = initialZ
        var m = [Float](repeating: 0, count: z.count)
        var v = [Float](repeating: 0, count: z.count)
        var result = PhaseResult(initLoss: .infinity, bestLoss: .infinity, bestZ: z)
        let checkpointEvery = options.checkpointEvery ?? defaultCheckpointEvery
        var deadEpochs = 0

        for epoch in 1...epochs {
            configureRuntime(options: options, sampleRate: sampleRate, crop: crop)
            LazyGraphContext.reset()

            let evaluator = LispEvaluator()
            do {
                try evaluator.evaluate(nodes: patchPlan.loweredNodes)
            } catch let error as LispError {
                throw TrainProtocolError("epoch re-evaluation failed: \(error.message)")
            }
            let paramSignals = try learnableSignals(evaluator: evaluator, transforms: transforms)
            for (i, t) in transforms.enumerated() {
                paramSignals[i].updateDataLazily(t.fromZ(z[i]))
            }

            guard let output = outputSignal(evaluator: evaluator) else {
                throw TrainProtocolError("patch lost its channel-0 output during re-evaluation")
            }
            let targetSignal = Tensor(target).toSignal(maxFrames: crop)
            let loss = multiResolutionSpectralLoss(
                synth: output, target: targetSignal, frames: crop)
            let lossValues = try loss.backward(frames: crop)
            let epochLoss = lossValues.reduce(0, +)
            guard epochLoss.isFinite else {
                throw TrainProtocolError("\(name) loss diverged (non-finite) at epoch \(epoch)")
            }
            if epoch == 1 { result.initLoss = epochLoss }
            if epochLoss < result.bestLoss {
                result.bestLoss = epochLoss
                result.bestZ = z
            }

            // Zero-amplitude dead-start trap: if every gradient is exactly
            // zero (e.g. gain hit 0 and the whole voice is silent), no step
            // can ever recover — stop the phase instead of idling.
            let naturalGrads = paramSignals.map { $0.grad?.data ?? 0 }
            if naturalGrads.allSatisfy({ $0 == 0 }) {
                deadEpochs += 1
                if deadEpochs >= 3 {
                    FileHandle.standardError.write(
                        Data(
                            "[train] \(name): all gradients zero for \(deadEpochs) epochs (silent voice?); stopping phase at epoch \(epoch)\n"
                                .utf8))
                    break
                }
            } else {
                deadEpochs = 0
            }

            // Adam in transformed coordinates, cosine LR decay
            // (Trainer.swift convention), per-param clip, bounds projection.
            let progress = Float(epoch - 1) / Float(max(epochs - 1, 1))
            let lr = transformedLR * (0.05 + 0.95 * 0.5 * (1 + cos(.pi * progress)))
            for (i, t) in transforms.enumerated() {
                let gNatural = naturalGrads[i]
                var g = gNatural * t.dNaturalDZ(z[i])
                if !g.isFinite { g = 0 }
                g = Swift.min(Swift.max(g, -gradClip), gradClip)
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                let mHat = m[i] / (1 - pow(0.9, Float(epoch)))
                let vHat = v[i] / (1 - pow(0.999, Float(epoch)))
                z[i] -= lr * mHat / (vHat.squareRoot() + 1e-8)
                z[i] = Swift.min(Swift.max(z[i], t.zMin), t.zMax)
            }

            if epoch % logEvery == 0 || epoch == epochs {
                try sink.emit(
                    .epoch(
                        EpochEvent(
                            epoch: epoch, total: epochs, loss: Double(epochLoss),
                            params: naturalValues(z: z, transforms: transforms)
                                .mapValues(Double.init))))
            }
            if emitCheckpoints, epoch % checkpointEvery == 0, epoch < epochs {
                // (cadence: --checkpoint-every, default 25)
                let wav = jobDir.epochWav(epoch)
                do {
                    try renderViaSubprocess(
                        params: naturalValues(z: z, transforms: transforms),
                        jobDir: jobDir, out: wav, frames: crop,
                        sampleRate: sampleRate, options: options)
                    try sink.emit(.checkpoint(CheckpointEvent(epoch: epoch, wav: wav.path)))
                } catch {
                    FileHandle.standardError.write(
                        Data("[train] checkpoint render failed at epoch \(epoch): \(error)\n".utf8))
                }
            }
        }
        return result
    }

    // MARK: - Loss (frozen SPEC.md §4 config)

    static func multiResolutionSpectralLoss(
        synth: Signal, target: Signal, frames: Int
    ) -> Signal {
        var total = Signal.constant(0.0)
        for window in spectralWindows where window <= frames {
            let hop = max(1, window / 4)
            total =
                total
                + spectralLossFFT(
                    synth, target, windowSize: window, useHannWindow: true,
                    useLogMagnitude: true, lossMode: .l1, hop: hop, normalize: true)
            total =
                total
                + spectralLossFFT(
                    synth, target, windowSize: window, useHannWindow: true,
                    useLogMagnitude: false, lossMode: .l1, hop: hop, normalize: true)
                * linearMagnitudeWeight
        }
        return total
    }

    // MARK: - Helpers

    static func configureRuntime(options: TrainOptions, sampleRate: Float, crop: Int) {
        DGenConfig.backend = options.backend == "c" ? .c : .metal
        DGenConfig.sampleRate = sampleRate
        DGenConfig.maxFrameCount = crop
        DGenConfig.defaultFrameCount = crop
        DGenSpectralConfig.logMagnitudeEpsilon = logEpsilon
        // Stop-gradient at phasor frequency inputs: forward unchanged,
        // no gradient through pitch (matches the plan's freeze verdict).
        DGenGradientConfig.detachPhasorFrequency = true
    }

    static func learnableSignals(
        evaluator: LispEvaluator, transforms: [TransformedParam]
    ) throws -> [Signal] {
        try transforms.map { t in
            guard case .signal(let signal)? = evaluator.definitions[t.name] else {
                throw TrainProtocolError("learnable param '\(t.name)' missing after re-evaluation")
            }
            return signal
        }
    }

    static func outputSignal(evaluator: LispEvaluator) -> Signal? {
        (evaluator.outputs.first { $0.channel == 0 } ?? evaluator.outputs.first)?.signal
    }

    static func naturalValues(z: [Float], transforms: [TransformedParam]) -> [String: Float] {
        var out: [String: Float] = [:]
        for (i, t) in transforms.enumerated() {
            out[t.name] = t.fromZ(z[i])
        }
        return out
    }

    static func preparedTarget(_ samples: [Float], frames: Int) -> [Float] {
        let peak = samples.map(abs).max() ?? 0
        let scale = peak > 0 ? 0.9 / peak : 1
        var out = samples.map { $0 * scale }
        if out.count > frames {
            out = Array(out[0..<frames])
        } else if out.count < frames {
            out += [Float](repeating: 0, count: frames - out.count)
        }
        return out
    }

    /// Preview renders re-invoke this executable's `train-render` mode:
    /// realize() must not interleave with backward() in this process.
    static func renderViaSubprocess(
        params: [String: Float], jobDir: JobDir, out: URL, frames: Int,
        sampleRate: Float, options: TrainOptions
    ) throws {
        let paramsURL = jobDir.file("render_params.json")
        let json = try JSONSerialization.data(
            withJSONObject: params.mapValues(Double.init), options: [.sortedKeys])
        try json.write(to: paramsURL, options: .atomic)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        process.arguments = [
            "train-render",
            "--patch", jobDir.loweredLisp.path,
            "--params-json", paramsURL.path,
            "--out", out.path,
            "--frames", String(frames),
            "--sample-rate", String(sampleRate),
            "--backend", options.backend,
        ]
        process.standardOutput = FileHandle.standardError
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw TrainProtocolError("train-render exited \(process.terminationStatus)")
        }
        guard FileManager.default.fileExists(atPath: out.path) else {
            throw TrainProtocolError("train-render produced no file at \(out.path)")
        }
    }
}

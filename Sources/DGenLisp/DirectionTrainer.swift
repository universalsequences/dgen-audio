// DirectionTrainer.swift — E4 direction-finding mode (SUBTRACTIVE_SPEC E4,
// patch-learn-spec §3): seeded short run + one cold restart as basin check.
//
// Loss: multi-resolution log-magnitude STFT L1 exactly as frozen in
// SynthID SPEC.md §4 (windows 256-2048, hop w/4, normalize, +0.1 linear
// term, log-eps 1e-3). Training may substitute the frequency-sampled SVF
// surrogate; rendering remains exact. Adam runs CPU-side in transformed coordinates (log for wide positive ranges), with bounds
// projection and a global LR at 2x the legacy production tone LR per
// BATCH_REFINE_FINDING.
//
// Each phase resets the lazy graph once. Every epoch re-evaluates the lowered
// patch in that graph; named parameters survive the backward graph clear while
// voice/state nodes are rebuilt from silence. Preview WAVs are rendered by
// re-invoking this executable (`train-render`) — realize() must never interleave with backward() in the
// same process (SPEC.md §5).

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

enum DirectionTrainer {
    static let defaultEpochs = 300
    static let defaultReportEvery = 10
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
        let previousCachePolicy = LazyGraphContext.preserveCompilationCaches
        let previousTimingPolicy = LazyGraphContext.collectExecutionTiming
        LazyGraphContext.preserveCompilationCaches = true
        LazyGraphContext.collectExecutionTiming =
            ProcessInfo.processInfo.environment["DGENLISP_TRAIN_TIMING"] == "1"
        defer {
            LazyGraphContext.preserveCompilationCaches = previousCachePolicy
            LazyGraphContext.collectExecutionTiming = previousTimingPolicy
        }
        let transforms = patchPlan.learnable.map(TransformedParam.init)

        // Target: peak-normalize to 0.9, fit to the crop (SPEC.md §4).
        let prepared = preparedTarget(targetSamples, frames: crop)

        let seedZ = zip(transforms, patchPlan.learnable).map { t, p in t.toZ(p.seedValue) }
        // Cold restart: deterministic transformed-midpoint init (never on
        // bounds; the honest generic init per SPEC.md §3/§5).
        let coldZ = transforms.map { ($0.zMin + $0.zMax) / 2 }

        // Debug: DGENLISP_TRAIN_FDCHECK=<param,param,...|all> — compare the
        // trainer's autograd gradients against central finite differences
        // in transformed coordinates at the seed point, then stop.
        if let fdParams = ProcessInfo.processInfo.environment["DGENLISP_TRAIN_FDCHECK"] {
            try fdcheck(
                names: fdParams, transforms: transforms, seedZ: seedZ,
                patchPlan: patchPlan, target: prepared,
                sampleRate: targetSampleRate, crop: crop, options: options)
            throw TrainProtocolError("fdcheck-only run (DGENLISP_TRAIN_FDCHECK)")
        }

        try renderViaSubprocess(
            params: naturalValues(z: seedZ, transforms: transforms),
            jobDir: jobDir, out: jobDir.seededWav, frames: crop,
            sampleRate: targetSampleRate, options: options)

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

        // Optional handoff to the real recurrent SVF. Parameters are shared,
        // so no projection or decoding is required.
        var finalPhase = seeded
        if options.polishEpochs > 0, options.filterSurrogate == "freq" {
            let trueSVFPlan = PatchPlan(
                plan: patchPlan.plan, learnable: patchPlan.learnable,
                fatalUnsupported: patchPlan.fatalUnsupported,
                loweredNodes: patchPlan.renderNodes, renderNodes: patchPlan.renderNodes)
            finalPhase = try runPhase(
                name: "polish", initialZ: seeded.bestZ,
                epochs: min(options.polishEpochs, 2000),
                transforms: transforms, patchPlan: trueSVFPlan, target: prepared,
                sampleRate: targetSampleRate, crop: crop, options: options,
                sink: sink, jobDir: jobDir, emitCheckpoints: false)
        }

        try renderViaSubprocess(
            params: naturalValues(z: finalPhase.bestZ, transforms: transforms),
            jobDir: jobDir, out: jobDir.finalWav, frames: crop,
            sampleRate: targetSampleRate, options: options)

        var deltas: [String: ParamDelta] = [:]
        let finalNatural = naturalValues(z: finalPhase.bestZ, transforms: transforms)
        for p in patchPlan.learnable {
            deltas[p.name] = ParamDelta(
                from: Double(p.seedValue), to: Double(finalNatural[p.name] ?? p.seedValue))
        }
        let improvement =
            seeded.initLoss > 0
            ? 100.0 * Double(seeded.initLoss - finalPhase.bestLoss) / Double(seeded.initLoss)
            : 0
        return ResultEvent(
            improvementPct: improvement,
            absDistance: Double(finalPhase.bestLoss),
            basinCheck: basinCheck,
            deltas: deltas,
            seededWav: jobDir.seededWav.path,
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
        let reportEvery = options.reportEvery ?? defaultReportEvery
        let checkpointEvery = options.checkpointEvery ?? defaultCheckpointEvery
        var deadEpochs = 0
        let timingEnabled =
            ProcessInfo.processInfo.environment["DGENLISP_TRAIN_TIMING"] == "1"

        configureRuntime(options: options, sampleRate: sampleRate, crop: crop)
        LazyGraphContext.reset()

        // Discover and register every Lisp parameter before the first measured
        // build. Named parameters refresh lazily at their AST positions after
        // this clear, preserving fresh-evaluation node and cell ordering.
        do {
            let evaluator = LispEvaluator(reusesRegisteredParameters: true)
            try evaluator.evaluate(nodes: patchPlan.loweredNodes)
        } catch let error as LispError {
            throw TrainProtocolError("parameter registration failed: \(error.message)")
        }
        LazyGraphContext.current.clearComputationGraph()

        var expectedBuildShape: (nodes: Int, tensors: Int, cells: Int)?
        for epoch in 1...epochs {
            let evalStart = CFAbsoluteTimeGetCurrent()
            let (paramSignals, output) = try evaluateTrainingPatch(
                nodes: patchPlan.loweredNodes, transforms: transforms)
            for (i, t) in transforms.enumerated() {
                paramSignals[i].updateDataLazily(t.fromZ(z[i]))
            }
            let lispEvalMS = (CFAbsoluteTimeGetCurrent() - evalStart) * 1000

            let graphBuildStart = CFAbsoluteTimeGetCurrent()
            let targetSignal = trainingTargetSignal(target, crop: crop, patchPlan: patchPlan)
            let loss = multiResolutionSpectralLoss(
                synth: output, target: targetSignal, frames: crop)
            let graphBuildMS = (CFAbsoluteTimeGetCurrent() - graphBuildStart) * 1000
            let epochGraph = LazyGraphContext.current
            let buildShape = (
                nodes: epochGraph.debugNodeCount,
                tensors: epochGraph.debugTensorCount,
                cells: epochGraph.debugMemoryCellCount)
            if let expected = expectedBuildShape,
                buildShape.nodes != expected.nodes || buildShape.tensors != expected.tensors
                    || buildShape.cells != expected.cells
            {
                throw TrainProtocolError(
                    "\(name) graph grew while rebuilding epoch \(epoch): "
                        + "expected \(expected.nodes) nodes/\(expected.tensors) tensors/"
                        + "\(expected.cells) cells, got \(buildShape.nodes)/"
                        + "\(buildShape.tensors)/\(buildShape.cells)")
            }
            expectedBuildShape = buildShape
            let lossValues = try loss.backward(frames: crop)
            if epoch == 1,
                ProcessInfo.processInfo.environment["DGENLISP_TRAIN_PROFILE"] == "1"
            {
                epochGraph.profileGPU(frames: crop)
                if let dir = ProcessInfo.processInfo.environment["DGENLISP_TRAIN_KERNEL_DUMP"] {
                    epochGraph.dumpKernelSources(to: dir)
                }
            }
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
            let optimizerStart = CFAbsoluteTimeGetCurrent()
            let naturalGrads = paramSignals.map { $0.grad?.data ?? 0 }
            if naturalGrads.allSatisfy({ $0 == 0 }) {
                deadEpochs += 1
                if deadEpochs >= 3 {
                    FileHandle.standardError.write(
                        Data(
                            "[train] \(name): all gradients zero for \(deadEpochs) epochs (silent voice?); stopping phase at epoch \(epoch)\n"
                                .utf8))
                    for signal in paramSignals { signal.grad = nil }
                    break
                }
            } else {
                deadEpochs = 0
            }

            // Adam in transformed coordinates, cosine LR decay
            // (Trainer.swift convention), per-param clip, bounds projection.
            let progress = Float(epoch - 1) / Float(max(epochs - 1, 1))
            let lr = transformedLR * (0.05 + 0.95 * 0.5 * (1 + cos(.pi * progress)))
            var normalizedSteps: [String: Double] = [:]
            for (i, t) in transforms.enumerated() {
                let gNatural = naturalGrads[i]
                var g = gNatural * t.dNaturalDZ(z[i])
                if !g.isFinite { g = 0 }
                g = Swift.min(Swift.max(g, -gradClip), gradClip)
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                let mHat = m[i] / (1 - pow(0.9, Float(epoch)))
                let vHat = v[i] / (1 - pow(0.999, Float(epoch)))
                let previousZ = z[i]
                z[i] -= lr * mHat / (vHat.squareRoot() + 1e-8)
                z[i] = Swift.min(Swift.max(z[i], t.zMin), t.zMax)
                // Adam's moment ratio already measures this update against
                // the parameter's recent gradient history. Dividing the
                // applied (post-projection) movement by the scheduled LR
                // expresses it in that knob's own normalized coordinate.
                let normalized = (z[i] - previousZ) / lr
                normalizedSteps[t.name] = Double(
                    Swift.min(Swift.max(normalized, -1), 1))
            }
            let optimizerMS = (CFAbsoluteTimeGetCurrent() - optimizerStart) * 1000

            if timingEnabled {
                emitTiming(
                    phase: name, epoch: epoch, lispEvalMS: lispEvalMS,
                    graphBuildMS: graphBuildMS, optimizerMS: optimizerMS,
                    lazy: epochGraph.lastExecutionTiming)
            }

            if epoch % reportEvery == 0 || epoch == epochs {
                try sink.emit(
                    .epoch(
                        EpochEvent(
                            epoch: epoch, total: epochs, loss: Double(epochLoss),
                            params: naturalValues(z: z, transforms: transforms)
                                .mapValues(Double.init),
                            steps: normalizedSteps)))
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

            // Gradients have already been copied into naturalGrads and all
            // epoch metrics above were captured before clearing them.
            for signal in paramSignals { signal.grad = nil }
        }
        return result
    }

    private static func evaluateTrainingPatch(
        nodes: [ASTNode], transforms: [TransformedParam]
    ) throws -> (params: [Signal], output: Signal) {
        let evaluator = LispEvaluator(reusesRegisteredParameters: true)
        do {
            try evaluator.evaluate(nodes: nodes)
        } catch let error as LispError {
            throw TrainProtocolError("epoch re-evaluation failed: \(error.message)")
        }
        let signals = try learnableSignals(evaluator: evaluator, transforms: transforms)
        guard let output = outputSignal(evaluator: evaluator) else {
            throw TrainProtocolError("patch lost its channel-0 output during re-evaluation")
        }
        return (signals, output)
    }

    private static func emitTiming(
        phase: String, epoch: Int, lispEvalMS: Double, graphBuildMS: Double,
        optimizerMS: Double, lazy: LazyExecutionTiming
    ) {
        let hashes = lazy.kernelSourceHashes.map { String($0, radix: 16) }.joined(separator: ",")
        let line = String(
            format: "[train-timing] phase=%@ epoch=%d lisp_eval_ms=%.3f graph_build_ms=%.3f dgen_compile_ms=%.3f pipeline_create_ms=%.3f gpu_execute_ms=%.3f optimizer_ms=%.3f full_cache_hit=%d runtime_cache_hit=%d kernel_hashes=%@\n",
            phase, epoch, lispEvalMS, graphBuildMS, lazy.compilationMS,
            lazy.runtimeCreationMS, lazy.executionMS, optimizerMS,
            lazy.fullCompilationCacheHit ? 1 : 0, lazy.runtimeCacheHit ? 1 : 0, hashes)
        FileHandle.standardError.write(Data(line.utf8))
    }

    // MARK: - Debug fdcheck

    private static func fdcheck(
        names: String, transforms: [TransformedParam], seedZ: [Float],
        patchPlan: PatchPlan, target: [Float], sampleRate: Float, crop: Int,
        options: TrainOptions
    ) throws {
        let wanted = Set(names.split(separator: ",").map(String.init))
        let indices = transforms.indices.filter {
            names == "all" || wanted.contains(transforms[$0].name)
        }

        func lossAt(_ z: [Float]) throws -> Float {
            configureRuntime(options: options, sampleRate: sampleRate, crop: crop)
            LazyGraphContext.reset()
            let evaluator = LispEvaluator()
            try evaluator.evaluate(nodes: patchPlan.loweredNodes)
            let signals = try learnableSignals(evaluator: evaluator, transforms: transforms)
            for (i, t) in transforms.enumerated() { signals[i].updateDataLazily(t.fromZ(z[i])) }
            guard let output = outputSignal(evaluator: evaluator) else {
                throw TrainProtocolError("no output")
            }
            let targetSignal = trainingTargetSignal(target, crop: crop, patchPlan: patchPlan)
            let loss = multiResolutionSpectralLoss(
                synth: output, target: targetSignal, frames: crop)
            return try loss.realize(frames: crop).reduce(0, +)
        }

        // Finite differences FIRST (forward-only), then one backward.
        let eps: Float = 0.005  // 0.5% of range in z
        var fdGrads: [Int: Float] = [:]
        let base = try lossAt(seedZ)
        FileHandle.standardError.write(Data("[fdcheck] loss at seed: \(base)\n".utf8))
        for i in indices {
            var zp = seedZ
            zp[i] += eps
            var zm = seedZ
            zm[i] -= eps
            fdGrads[i] = (try lossAt(zp) - (try lossAt(zm))) / (2 * eps)
        }

        configureRuntime(options: options, sampleRate: sampleRate, crop: crop)
        LazyGraphContext.reset()
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: patchPlan.loweredNodes)
        let signals = try learnableSignals(evaluator: evaluator, transforms: transforms)
        for (i, t) in transforms.enumerated() { signals[i].updateDataLazily(t.fromZ(seedZ[i])) }
        guard let output = outputSignal(evaluator: evaluator) else {
            throw TrainProtocolError("no output")
        }
        let targetSignal = trainingTargetSignal(target, crop: crop, patchPlan: patchPlan)
        let loss = multiResolutionSpectralLoss(synth: output, target: targetSignal, frames: crop)
        let lossValues = try loss.backward(frames: crop)
        FileHandle.standardError.write(
            Data("[fdcheck] backward loss: \(lossValues.reduce(0, +))\n".utf8))

        FileHandle.standardError.write(
            Data("[fdcheck] param                    autograd(z)        fd(z)   sign-match\n".utf8))
        for i in indices {
            let t = transforms[i]
            let autograd = (signals[i].grad?.data ?? 0) * t.dNaturalDZ(seedZ[i])
            let fd = fdGrads[i] ?? 0
            let match = autograd == 0 || fd == 0 ? "?" : (autograd * fd > 0 ? "YES" : "NO")
            FileHandle.standardError.write(
                Data(
                    String(
                        format: "[fdcheck] %-22s %12.5g %12.5g   %@\n",
                        (t.name as NSString).utf8String!, autograd, fd, match).utf8))
        }
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
        // Phasor frequency is trainable: the suffix-scan adjoint composes
        // with history BPTT since the scalar-recurrence consolidation fix.
        // Reset explicitly in case another phase flipped the global.
        DGenGradientConfig.detachPhasorFrequency = false
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

    private static func trainingTargetSignal(
        _ target: [Float], crop: Int, patchPlan: PatchPlan
    ) -> Signal {
        guard containsSVFFrequencySampled(patchPlan.loweredNodes) else {
            return Tensor(target).toSignal(maxFrames: crop)
        }
        let latency = (surrogateWindow(in: patchPlan.loweredNodes) ?? 1024) - 1
        var delayed = [Float](repeating: 0, count: target.count)
        if latency < target.count {
            for i in latency..<target.count { delayed[i] = target[i - latency] }
        }
        return Tensor(delayed).toSignal(maxFrames: crop)
    }

    private static func containsSVFFrequencySampled(_ nodes: [ASTNode]) -> Bool {
        nodes.contains { node in
            guard case .list(let xs) = node else { return false }
            if case .atom("svf-freq") = xs.first ?? .atom("") { return true }
            return containsSVFFrequencySampled(xs)
        }
    }

    private static func surrogateWindow(in nodes: [ASTNode]) -> Int? {
        for node in nodes {
            guard case .list(let xs) = node else { continue }
            if case .atom("svf-freq") = xs.first ?? .atom("") {
                for i in xs.indices where i + 1 < xs.count {
                    if case .atom("@window") = xs[i], case .atom(let raw) = xs[i + 1] {
                        return Int(raw)
                    }
                }
            }
            if let found = surrogateWindow(in: xs) { return found }
        }
        return nil
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
            "--patch", jobDir.renderLisp.path,
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

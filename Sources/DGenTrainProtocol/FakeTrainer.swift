// FakeTrainer.swift — deterministic protocol-exercising stub.
//
// Substituted for the real Metal trainer via --fake-trainer (or the
// DGENLISP_FAKE_TRAINER env var). Emits a realistic event sequence with
// synthetic exponential loss decay and writes every job-dir artifact the
// real trainer would, so the host protocol and mock consumers can be
// tested anywhere, byte-deterministically, with no GPU.

import Foundation

public enum FakeTrainer {
    public static let defaultEpochs = 100
    static let logEvery = 10
    static let checkpointEvery = 25

    /// Runs the fake job: emits plan/stage/epoch/checkpoint events and writes
    /// artifacts, or stops successfully after plan for --plan-only. The
    /// caller emits and persists a returned result completion.
    public static func run(
        options: TrainOptions, sink: TrainEventSink, jobDir: JobDir
    ) throws -> TrainCompletion {
        let seed = try SeedParams.load(url: URL(fileURLWithPath: options.seedParamsPath))
        let epochs = options.epochs ?? defaultEpochs

        // lowered.lisp: the fake "lowering pass" is an annotated copy.
        let patchSource = try String(
            contentsOfFile: options.patchPath, encoding: .utf8)
        let lowered = "; lowered by fake trainer (identity transcription)\n" + patchSource
        try lowered.data(using: .utf8)!.write(to: jobDir.loweredLisp, options: .atomic)

        let plan = PlanEvent(
            learnable: seed.params.keys.sorted(),
            frozen: [ParamVerdict(name: "f0", reason: "f0-adjoint-unreliable")],
            unsupported: [],
            seedEcho: seed.params,
            pitchHz: options.pitchHz ?? 110.0,
            gateFrames: options.gateFrames ?? 8820,
            cropFrames: 32768)
        try sink.emit(.plan(plan))
        if options.planOnly {
            return .planOnly
        }
        try sink.emit(.stage(StageEvent(name: "train", total: epochs)))

        let initLoss = syntheticLoss(epoch: 0, total: epochs)
        var finalLoss = initLoss
        var epoch = 0
        while epoch < epochs {
            epoch += 1
            if options.fakeEpochMs > 0 {
                usleep(useconds_t(options.fakeEpochMs * 1000))
            }
            if let failAt = options.fakeFailAtEpoch, epoch == failAt {
                throw TrainProtocolError("induced fake-trainer failure at epoch \(epoch)")
            }
            finalLoss = syntheticLoss(epoch: epoch, total: epochs)
            if epoch % logEvery == 0 || epoch == epochs {
                try sink.emit(
                    .epoch(
                        EpochEvent(
                            epoch: epoch, total: epochs, loss: finalLoss,
                            params: syntheticParams(seed: seed, epoch: epoch, total: epochs),
                            steps: syntheticSteps(seed: seed, epoch: epoch, total: epochs))))
            }
            if epoch % checkpointEvery == 0 {
                let wav = jobDir.epochWav(epoch)
                try MiniWav.write(url: wav, samples: previewSamples(epoch: epoch))
                try sink.emit(.checkpoint(CheckpointEvent(epoch: epoch, wav: wav.path)))
            }
        }

        try MiniWav.write(url: jobDir.finalWav, samples: previewSamples(epoch: epochs))

        let finalParams = syntheticParams(seed: seed, epoch: epochs, total: epochs)
        var deltas: [String: ParamDelta] = [:]
        for (name, from) in seed.params {
            deltas[name] = ParamDelta(from: from, to: finalParams[name] ?? from)
        }
        let improvement = round6(100.0 * (initLoss - finalLoss) / initLoss)
        return .result(
            ResultEvent(
                improvementPct: improvement,
                absDistance: finalLoss,
                basinCheck: "ok",
                deltas: deltas,
                finalWav: jobDir.finalWav.path))
    }

    // MARK: - Deterministic synthetics

    static func syntheticLoss(epoch: Int, total: Int) -> Double {
        let progress = Double(epoch) / Double(max(total, 1))
        return round6(0.35 * exp(-3.0 * progress) + 0.012)
    }

    static func syntheticParams(seed: SeedParams, epoch: Int, total: Int) -> [String: Double] {
        let progress = Double(epoch) / Double(max(total, 1))
        let drift = 0.1 * (1.0 - exp(-3.0 * progress))
        var out: [String: Double] = [:]
        for (name, value) in seed.params {
            out[name] = round6(value + drift)
        }
        return out
    }

    static func syntheticSteps(seed: SeedParams, epoch: Int, total: Int) -> [String: Double] {
        let progress = Double(epoch) / Double(max(total, 1))
        let magnitude = round6(exp(-3.0 * progress))
        return Dictionary(uniqueKeysWithValues: seed.params.keys.map { ($0, magnitude) })
    }

    static func previewSamples(epoch: Int) -> [Float] {
        // 512-sample sine whose frequency depends on the epoch, so each
        // checkpoint artifact differs (a host can tell them apart).
        let hz = 220.0 + Double(epoch)
        return (0..<512).map { i in
            Float(0.5 * sin(2.0 * Double.pi * hz * Double(i) / 44100.0))
        }
    }

    static func round6(_ x: Double) -> Double {
        (x * 1e6).rounded() / 1e6
    }
}

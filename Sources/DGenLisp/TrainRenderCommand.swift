// TrainRenderCommand.swift — hidden helper mode used by DirectionTrainer.
//
// Renders a lowered patch at given param values into a WAV. Runs as a
// separate process because realize() must never interleave with
// backward() in the training process (SPEC.md §5). Not part of the
// host-facing protocol: stdout is not the event stream here; all output
// goes to stderr and the exit code.

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

enum TrainRenderCommand {
    static func run(arguments: [String]) -> Never {
        do {
            try render(arguments: arguments)
            exit(0)
        } catch {
            fputs("train-render error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func render(arguments: [String]) throws {
        var patch: String?
        var paramsJSON: String?
        var out: String?
        var frames = 8192
        var sampleRate: Float = 44100
        var backend = "metal"

        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < arguments.count else { throw TrainProtocolError("missing value for \(flag)") }
            return arguments[i]
        }
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--patch": patch = try value(arg)
            case "--params-json": paramsJSON = try value(arg)
            case "--out": out = try value(arg)
            case "--frames": frames = Int(try value(arg)) ?? frames
            case "--sample-rate": sampleRate = Float(try value(arg)) ?? sampleRate
            case "--backend": backend = try value(arg)
            default: throw TrainProtocolError("unknown train-render option \(arg)")
            }
            i += 1
        }
        guard let patchPath = patch, let outPath = out else {
            throw TrainProtocolError("train-render requires --patch and --out")
        }

        var params: [String: Float] = [:]
        if let paramsPath = paramsJSON {
            let data = try Data(contentsOf: URL(fileURLWithPath: paramsPath))
            let decoded = try JSONDecoder().decode([String: Double].self, from: data)
            params = decoded.mapValues(Float.init)
        }

        DGenConfig.backend = backend == "c" ? .c : .metal
        DGenConfig.sampleRate = sampleRate
        DGenConfig.maxFrameCount = frames
        DGenConfig.defaultFrameCount = frames
        LazyGraphContext.reset()

        let source = try String(contentsOfFile: patchPath, encoding: .utf8)
        let evaluator = LispEvaluator(
            sourceDirectory: URL(fileURLWithPath: patchPath).deletingLastPathComponent())
        // lowered.lisp is already modulation-lowered and excitation-driven.
        try evaluator.evaluate(nodes: parseSource(source))
        for (name, value) in params {
            if case .signal(let signal)? = evaluator.definitions[name] {
                signal.updateDataLazily(value)
            }
        }
        guard
            let output = (evaluator.outputs.first { $0.channel == 0 } ?? evaluator.outputs.first)?
                .signal
        else {
            throw TrainProtocolError("patch has no outputs")
        }
        let samples = try output.realize(frames: frames)
        try MiniWav.write(
            url: URL(fileURLWithPath: outPath), samples: samples,
            sampleRate: Int(sampleRate))
    }
}

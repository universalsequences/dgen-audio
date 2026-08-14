// TrainOptions.swift — CLI contract for `dgenlisp train` (patch-learn-spec §3).

import Foundation

public struct TrainOptions: Equatable {
    public var patchPath: String
    public var targetPath: String
    public var seedParamsPath: String
    public var jobDirPath: String
    public var mode: String = "direction"
    /// nil = trainer default (fake: 100; direction mode: 300).
    public var epochs: Int?
    /// nil = derive from the target's amplitude envelope (spec §6).
    public var gateFrames: Int?
    /// Preview WAV cadence; nil = trainer default (25).
    public var checkpointEvery: Int?
    /// Epoch event cadence; nil = trainer default (10).
    public var reportEvery: Int?
    /// nil = CPU pitch estimate from the target (spec §6).
    public var pitchHz: Double?

    /// Compute backend for the real trainer: "metal" (default) or "c"
    /// (CPU; used by CI and machines without a GPU).
    public var backend: String = "metal"

    /// Replace `(svf ...)` calls with the differentiable frequency-sampled
    /// training surrogate. Rendering always keeps the real SVF.
    /// Default "none": the surrogate mode was determined to be broken;
    /// opt back in with --filter-surrogate freq.
    public var filterSurrogate: String = "none"
    public var surrogateWindow: Int = 1024
    public var surrogateHop: Int = 256
    /// Optional true-SVF refinement after surrogate training.
    public var polishEpochs: Int = 0

    /// Opt-in tensor-lane multistart search. Zero preserves the legacy
    /// seeded + midpoint policy. Positive values are the number of cheap
    /// forward candidates generated before short-horizon batched Adam.
    public var multistartCandidates: Int = 0
    public var multistartLanes: Int = 64
    public var multistartBatch: Int = 256
    public var multistartSteps: Int = 30
    public var multistartSeed: Int = 1

    /// Emit the plan event and exit successfully (no GPU time and no result
    /// event).
    public var planOnly: Bool = false

    // Hidden/testing knobs (not part of the host-facing contract).
    public var useFakeTrainer: Bool = false
    public var fakeFailAtEpoch: Int?
    public var fakeEpochMs: Int = 0

    public init(patchPath: String, targetPath: String, seedParamsPath: String, jobDirPath: String) {
        self.patchPath = patchPath
        self.targetPath = targetPath
        self.seedParamsPath = seedParamsPath
        self.jobDirPath = jobDirPath
    }

    /// Parse the argv slice after the `train` subcommand word.
    /// The DGENLISP_FAKE_TRAINER env var is an alternative to --fake-trainer.
    public static func parse(
        _ args: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> TrainOptions {
        var patch: String?
        var target: String?
        var seed: String?
        var jobDir: String?
        var mode = "direction"
        var epochs: Int?
        var gateFrames: Int?
        var checkpointEvery: Int?
        var reportEvery: Int?
        var pitchHz: Double?
        var planOnly = false
        var backend = "metal"
        var filterSurrogate = "none"
        var surrogateWindow = 1024
        var surrogateHop = 256
        var polishEpochs = 0
        var multistartCandidates = 0
        var multistartLanes = 64
        var multistartBatch = 256
        var multistartSteps = 30
        var multistartSeed = 1
        var fake = environment["DGENLISP_FAKE_TRAINER"] != nil
        var fakeFailAtEpoch: Int?
        var fakeEpochMs = 0

        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < args.count else { throw TrainProtocolError("Missing value for \(flag)") }
            return args[i]
        }
        func intValue(_ flag: String) throws -> Int {
            let raw = try value(flag)
            guard let v = Int(raw) else { throw TrainProtocolError("Invalid integer for \(flag): \(raw)") }
            return v
        }
        while i < args.count {
            let arg = args[i]
            switch arg {
            case "--patch": patch = try value(arg)
            case "--target": target = try value(arg)
            case "--seed-params": seed = try value(arg)
            case "--job-dir": jobDir = try value(arg)
            case "--mode": mode = try value(arg)
            case "--epochs": epochs = try intValue(arg)
            case "--gate-frames": gateFrames = try intValue(arg)
            case "--checkpoint-every": checkpointEvery = try intValue(arg)
            case "--report-every": reportEvery = try intValue(arg)
            case "--pitch-hz":
                let raw = try value(arg)
                guard let v = Double(raw) else {
                    throw TrainProtocolError("Invalid number for --pitch-hz: \(raw)")
                }
                pitchHz = v
            case "--backend":
                backend = try value(arg)
                guard backend == "metal" || backend == "c" else {
                    throw TrainProtocolError("Invalid --backend: \(backend) (metal|c)")
                }
            case "--filter-surrogate":
                filterSurrogate = try value(arg)
                guard filterSurrogate == "freq" || filterSurrogate == "none" else {
                    throw TrainProtocolError(
                        "Invalid --filter-surrogate: \(filterSurrogate) (freq|none)")
                }
            case "--surrogate-window": surrogateWindow = try intValue(arg)
            case "--surrogate-hop": surrogateHop = try intValue(arg)
            case "--polish-epochs": polishEpochs = try intValue(arg)
            case "--multistart-candidates": multistartCandidates = try intValue(arg)
            case "--multistart-lanes": multistartLanes = try intValue(arg)
            case "--multistart-batch": multistartBatch = try intValue(arg)
            case "--multistart-steps": multistartSteps = try intValue(arg)
            case "--multistart-seed": multistartSeed = try intValue(arg)
            case "--plan-only": planOnly = true
            case "--fake-trainer": fake = true
            case "--fake-fail-at-epoch": fakeFailAtEpoch = try intValue(arg)
            case "--fake-epoch-ms": fakeEpochMs = try intValue(arg)
            default:
                throw TrainProtocolError("Unknown option for train: \(arg)")
            }
            i += 1
        }

        guard let patchPath = patch else { throw TrainProtocolError("--patch is required") }
        guard let targetPath = target else { throw TrainProtocolError("--target is required") }
        guard let seedPath = seed else { throw TrainProtocolError("--seed-params is required") }
        guard let jobDirPath = jobDir else { throw TrainProtocolError("--job-dir is required") }
        guard mode == "direction" else {
            throw TrainProtocolError("Unsupported --mode: \(mode) (v1 supports only 'direction')")
        }
        guard surrogateWindow >= 2 && surrogateWindow.nonzeroBitCount == 1 else {
            throw TrainProtocolError("--surrogate-window must be a power of two >= 2")
        }
        guard surrogateHop > 0, surrogateHop <= surrogateWindow,
              surrogateWindow % surrogateHop == 0 else {
            throw TrainProtocolError("--surrogate-hop must be positive and divide --surrogate-window")
        }
        guard polishEpochs >= 0 else {
            throw TrainProtocolError("--polish-epochs must be >= 0")
        }
        guard multistartCandidates >= 0 else {
            throw TrainProtocolError("--multistart-candidates must be >= 0")
        }
        guard multistartLanes >= 2, multistartBatch >= 1, multistartSteps >= 1 else {
            throw TrainProtocolError(
                "--multistart-lanes must be >= 2 and --multistart-batch/--multistart-steps must be >= 1")
        }
        if multistartCandidates > 0 && multistartCandidates < multistartLanes {
            throw TrainProtocolError("--multistart-candidates must be >= --multistart-lanes")
        }
        if let reportEvery {
            guard reportEvery > 0 else {
                throw TrainProtocolError("--report-every must be > 0")
            }
        }

        var options = TrainOptions(
            patchPath: patchPath, targetPath: targetPath,
            seedParamsPath: seedPath, jobDirPath: jobDirPath)
        options.mode = mode
        options.epochs = epochs
        options.gateFrames = gateFrames
        options.checkpointEvery = checkpointEvery
        options.reportEvery = reportEvery
        options.pitchHz = pitchHz
        options.backend = backend
        options.filterSurrogate = filterSurrogate
        options.surrogateWindow = surrogateWindow
        options.surrogateHop = surrogateHop
        options.polishEpochs = polishEpochs
        options.multistartCandidates = multistartCandidates
        options.multistartLanes = multistartLanes
        options.multistartBatch = multistartBatch
        options.multistartSteps = multistartSteps
        options.multistartSeed = multistartSeed
        options.planOnly = planOnly
        options.useFakeTrainer = fake
        options.fakeFailAtEpoch = fakeFailAtEpoch
        options.fakeEpochMs = fakeEpochMs

        for (label, path) in [
            ("--patch", patchPath), ("--target", targetPath), ("--seed-params", seedPath),
        ] {
            guard FileManager.default.fileExists(atPath: path) else {
                throw TrainProtocolError("\(label) file not found: \(path)")
            }
        }
        return options
    }
}

// MARK: - Seed params (spec §3.1)

public struct SeedParams: Equatable {
    /// Preserved as parsed; echoed verbatim in the plan event.
    public var params: [String: Double]

    public init(params: [String: Double]) {
        self.params = params
    }

    /// Strict load of `{"params": {name: value}}`.
    public static func load(url: URL) throws -> SeedParams {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw TrainProtocolError("Cannot read seed params: \(url.path)")
        }
        struct Wrapper: Codable { var params: [String: Double] }
        do {
            let wrapper = try JSONDecoder().decode(Wrapper.self, from: data)
            return SeedParams(params: wrapper.params)
        } catch {
            throw TrainProtocolError(
                "seed.json must be {\"params\": {name: value}}: \(error.localizedDescription)")
        }
    }
}

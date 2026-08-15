import Metal
import XCTest

import DGenTrainProtocol

/// Rung-1-style self-consistency for `dgenlisp train --mode direction`:
/// the target is a dgen render of the same patch at hidden params; a
/// seeded short run must move toward it and clear a modest improvement
/// gate, and the emitted transcript must be protocol-clean.
///
/// Requires Metal (spectral-loss BPTT does not compile on the C backend —
/// known codegen gap, see PR notes); skipped where no GPU exists.
final class TrainE2ETests: XCTestCase {
    var workDir: URL!

    override func setUpWithError() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal unavailable; direction-mode training requires a GPU")
        }
        workDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dgenlisp-train-e2e-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDir)
    }

    var dgenlispBinary: URL {
        for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
            return bundle.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("DGenLisp")
        }
        fatalError("cannot locate build products directory")
    }

    @discardableResult
    func runProcess(_ arguments: [String]) throws -> (status: Int32, stdout: String, stderr: String)
    {
        let process = Process()
        process.executableURL = dgenlispBinary
        process.arguments = arguments
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        try process.run()
        var outData = Data()
        var errData = Data()
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global().async {
            outData = outPipe.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }
        group.enter()
        DispatchQueue.global().async {
            errData = errPipe.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }
        process.waitUntilExit()
        group.wait()
        return (
            process.terminationStatus,
            String(data: outData, encoding: .utf8) ?? "",
            String(data: errData, encoding: .utf8) ?? ""
        )
    }

    func testDirectionModeSelfConsistency() throws {
        let patch = workDir.appendingPathComponent("patch.lisp")
        try """
            (def amp (param amp @default 0.3 @min 0.05 @max 1.0))
            (def bright (param bright @default 0.5 @min 0.05 @max 1.0))
            (out (+ (* amp (phasor 110.0)) (* bright (phasor 220.0))) 0)
            """.write(to: patch, atomically: true, encoding: .utf8)

        // Hidden truth: amp 0.8, bright 0.1 — rendered through the SAME
        // executable path the trainer's previews use.
        let hidden = workDir.appendingPathComponent("hidden.json")
        try #"{"amp":0.8,"bright":0.1}"#.write(to: hidden, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("target.wav")
        let render = try runProcess([
            "train-render", "--patch", patch.path, "--params-json", hidden.path,
            "--out", target.path, "--frames", "8192", "--sample-rate", "44100",
            "--backend", "metal",
        ])
        XCTAssertEqual(render.status, 0, "target render failed:\n\(render.stderr)")

        let seed = workDir.appendingPathComponent("seed.json")
        try #"{"params":{"amp":0.3,"bright":0.5}}"#.write(to: seed, atomically: true, encoding: .utf8)
        let jobDir = workDir.appendingPathComponent("job")

        let outcome = try runProcess([
            "train",
            "--patch", patch.path, "--target", target.path,
            "--seed-params", seed.path, "--job-dir", jobDir.path,
            "--mode", "direction", "--epochs", "30", "--backend", "metal",
            "--pitch-hz", "110", "--gate-frames", "8192",
        ])
        XCTAssertEqual(outcome.status, 0, "training failed:\n\(outcome.stderr.suffix(2000))")

        // Strict protocol validation, mock-host style.
        let lines = outcome.stdout.split(separator: "\n").map(String.init)
        var events: [TrainEvent] = []
        for line in lines {
            events.append(try TrainEventCoding.decodeLine(line))
        }
        guard case .plan(let plan) = events.first else {
            return XCTFail("first event must be plan")
        }
        XCTAssertEqual(plan.learnable, ["amp", "bright"])
        XCTAssertEqual(plan.seedEcho, ["amp": 0.3, "bright": 0.5])
        for event in events.dropLast() {
            XCTAssertFalse(event.isTerminal)
        }
        guard case .result(let result) = events.last else {
            return XCTFail("last event must be result, got \(String(describing: events.last))")
        }

        // Two stages: seeded train, then the cold basin check.
        let stages = events.compactMap { event -> String? in
            if case .stage(let s) = event { return s.name } else { return nil }
        }
        XCTAssertEqual(stages, ["train", "basin-check"])

        let epochEvents = events.compactMap { event -> EpochEvent? in
            if case .epoch(let epoch) = event { return epoch } else { return nil }
        }
        XCTAssertFalse(epochEvents.isEmpty)
        for epoch in epochEvents {
            XCTAssertEqual(Set(epoch.params.keys), Set(plan.learnable))
            XCTAssertTrue(epoch.params.values.allSatisfy { (0.05...1.0).contains($0) },
                          "epoch params must be natural knob values: \(epoch.params)")
            XCTAssertEqual(Set(epoch.steps?.keys.map { $0 } ?? []), Set(plan.learnable))
            XCTAssertTrue(epoch.steps?.values.allSatisfy { (-1.0...1.0).contains($0) } == true,
                          "epoch steps must be normalized: \(String(describing: epoch.steps))")
        }

        // The seeded run must beat the honest cold midpoint restart here
        // (the seed is in the right neighborhood by construction).
        XCTAssertEqual(result.basinCheck, "ok")

        // Direction-finding gate: modest but real improvement, and both
        // deltas point toward the hidden truth.
        XCTAssertGreaterThan(result.improvementPct, 15)
        XCTAssertGreaterThan(result.absDistance, 0)
        let ampDelta = try XCTUnwrap(result.deltas["amp"])
        let brightDelta = try XCTUnwrap(result.deltas["bright"])
        XCTAssertGreaterThan(ampDelta.to, ampDelta.from, "amp should move up toward 0.8")
        XCTAssertLessThan(brightDelta.to, brightDelta.from, "bright should move down toward 0.1")

        // Artifacts.
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.finalWav))
        for case .checkpoint(let c) in events {
            XCTAssertTrue(FileManager.default.fileExists(atPath: c.wav))
        }
        for name in ["lowered.lisp", "final.wav", "result.json"] {
            XCTAssertTrue(
                FileManager.default.fileExists(
                    atPath: jobDir.appendingPathComponent(name).path),
                "missing artifact \(name)")
        }
        // result.json matches the streamed result byte-for-byte.
        let stored = try String(
            contentsOf: jobDir.appendingPathComponent("result.json"), encoding: .utf8)
            .trimmingCharacters(in: .newlines)
        XCTAssertEqual(stored, lines.last)
    }

    func testCMAFlatObjectiveConsumesTheRequestedGenerationBudget() throws {
        let patch = workDir.appendingPathComponent("flat-cma-patch.lisp")
        try """
            (def unused (param unused @default 0.3 @min 0.05 @max 1.0))
            (out (+ (phasor 110.0) (* unused 0.0)) 0)
            """.write(to: patch, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("flat-cma-target.wav")
        let render = try runProcess([
            "train-render", "--patch", patch.path,
            "--out", target.path, "--frames", "4096", "--sample-rate", "44100",
            "--backend", "metal",
        ])
        XCTAssertEqual(render.status, 0, "target render failed:\n\(render.stderr)")

        let seed = workDir.appendingPathComponent("flat-cma-seed.json")
        try #"{"params":{"unused":0.3}}"#.write(to: seed, atomically: true, encoding: .utf8)
        let jobDir = workDir.appendingPathComponent("flat-cma-job")
        let generations = 7
        let outcome = try runProcess([
            "train", "--patch", patch.path, "--target", target.path,
            "--seed-params", seed.path, "--job-dir", jobDir.path,
            "--mode", "direction", "--backend", "metal",
            "--pitch-hz", "110", "--gate-frames", "4096",
            "--search", "cma-es", "--cma-generations", String(generations),
            "--cma-population", "8", "--local-epochs", "0",
            "--cma-continue", "1", "--cma-refine-epochs", "0",
            "--cma-final-epochs", "0",
        ])
        XCTAssertEqual(outcome.status, 0, "training failed:\n\(outcome.stderr.suffix(2000))")

        var stage = ""
        var progress = [OptimizationProgressEvent]()
        for line in outcome.stdout.split(separator: "\n") {
            switch try TrainEventCoding.decodeLine(String(line)) {
            case .stage(let value): stage = value.name
            case .optimizationProgress(let value) where stage == "cma-es":
                progress.append(value)
            default: break
            }
        }
        XCTAssertEqual(progress.map(\.current), Array(1...generations))

        let reportData = try Data(contentsOf: jobDir.appendingPathComponent("cma_es_report.json"))
        let report = try XCTUnwrap(
            JSONSerialization.jsonObject(with: reportData) as? [String: Any])
        XCTAssertEqual(report["generations_completed"] as? Int, generations)
        XCTAssertEqual(report["stop_reason"] as? String, "generation_limit")
    }

    func testCMAAndBatchedAdamStreamIncrementalProgress() throws {
        let patch = workDir.appendingPathComponent("progress-patch.lisp")
        try """
            (def amp (param amp @default 0.3 @min 0.05 @max 1.0))
            (out (* amp (phasor 110.0)) 0)
            """.write(to: patch, atomically: true, encoding: .utf8)
        let hidden = workDir.appendingPathComponent("progress-hidden.json")
        try #"{"amp":0.8}"#.write(to: hidden, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("progress-target.wav")
        let render = try runProcess([
            "train-render", "--patch", patch.path, "--params-json", hidden.path,
            "--out", target.path, "--frames", "8192", "--sample-rate", "44100",
            "--backend", "metal",
        ])
        XCTAssertEqual(render.status, 0, "target render failed:\n\(render.stderr)")

        let seed = workDir.appendingPathComponent("progress-seed.json")
        try #"{"params":{"amp":0.3}}"#.write(to: seed, atomically: true, encoding: .utf8)
        let outcome = try runProcess([
            "train", "--patch", patch.path, "--target", target.path,
            "--seed-params", seed.path,
            "--job-dir", workDir.appendingPathComponent("progress-job").path,
            "--mode", "direction", "--epochs", "2", "--backend", "metal",
            "--pitch-hz", "110", "--gate-frames", "8192",
            "--search", "cma-es", "--cma-generations", "2",
            "--cma-population", "8", "--local-epochs", "0",
            "--cma-continue", "2", "--cma-refine-epochs", "2",
            "--cma-refine-mode", "batched", "--cma-final-epochs", "0",
        ])
        XCTAssertEqual(outcome.status, 0, "training failed:\n\(outcome.stderr.suffix(2000))")

        var stage = ""
        var cma = [OptimizationProgressEvent]()
        var batched = [OptimizationProgressEvent]()
        for line in outcome.stdout.split(separator: "\n") {
            switch try TrainEventCoding.decodeLine(String(line)) {
            case .stage(let value): stage = value.name
            case .optimizationProgress(let value):
                if stage == "cma-es" { cma.append(value) }
                if stage == "cma-refine-batched" { batched.append(value) }
            default: break
            }
        }
        XCTAssertEqual(cma.map(\.current), [1, 2])
        XCTAssertTrue(cma.allSatisfy {
            !$0.losses.isEmpty && $0.losses.count <= 5
                && $0.losses == $0.losses.sorted()
        })
        XCTAssertEqual(batched.map(\.current), [1, 2])
        XCTAssertTrue(batched.allSatisfy {
            $0.losses.count == 1 && $0.losses[0].isFinite
        })
    }
}

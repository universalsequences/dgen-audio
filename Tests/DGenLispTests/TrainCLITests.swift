import XCTest

import DGenTrainProtocol

/// Black-box tests of `dgenlisp train` as a subprocess: the NDJSON stream
/// contract exactly as the eseq host will consume it.
final class TrainCLITests: XCTestCase {
    var workDir: URL!

    override func setUpWithError() throws {
        workDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dgenlisp-train-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDir)
    }

    // MARK: - Harness

    var dgenlispBinary: URL {
        for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
            return bundle.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("DGenLisp")
        }
        fatalError("cannot locate build products directory")
    }

    struct RunOutcome {
        var exitStatus: Int32
        var terminationReason: Process.TerminationReason
        var stdoutLines: [String]
        var stderr: String
        var events: [TrainEvent]
    }

    func makeInputs(seedJSON: String = #"{"params":{"sinefm":0.06,"ratio":0.05}}"#) throws
        -> (patch: URL, target: URL, seed: URL, jobDir: URL)
    {
        let patch = workDir.appendingPathComponent("patch.lisp")
        try "(out (phasor 110) 0)\n".write(to: patch, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("target.wav")
        try MiniWav.write(url: target, samples: [Float](repeating: 0, count: 64))
        let seed = workDir.appendingPathComponent("seed.json")
        try seedJSON.write(to: seed, atomically: true, encoding: .utf8)
        let jobDir = workDir.appendingPathComponent("job")
        return (patch, target, seed, jobDir)
    }

    @discardableResult
    func runTrain(
        _ extraArgs: [String], env: [String: String] = [:],
        sigtermAfter: TimeInterval? = nil
    ) throws -> RunOutcome {
        let process = Process()
        process.executableURL = dgenlispBinary
        process.arguments = ["train"] + extraArgs
        var environment = ProcessInfo.processInfo.environment
        for (k, v) in env { environment[k] = v }
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        try process.run()

        if let delay = sigtermAfter {
            Thread.sleep(forTimeInterval: delay)
            process.terminate()  // SIGTERM
        }

        // Drain pipes concurrently to avoid deadlock on large output.
        var stdoutData = Data()
        var stderrData = Data()
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global().async {
            stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }
        group.enter()
        DispatchQueue.global().async {
            stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            group.leave()
        }
        process.waitUntilExit()
        group.wait()

        let stdoutText = String(data: stdoutData, encoding: .utf8) ?? ""
        let lines = stdoutText.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { !$0.isEmpty }
        // STRICT: every stdout line must decode as a typed event.
        var events: [TrainEvent] = []
        for line in lines {
            do {
                events.append(try TrainEventCoding.decodeLine(line))
            } catch {
                XCTFail("stdout line is not a valid train event: \(line)")
            }
        }
        return RunOutcome(
            exitStatus: process.terminationStatus,
            terminationReason: process.terminationReason,
            stdoutLines: lines,
            stderr: String(data: stderrData, encoding: .utf8) ?? "",
            events: events)
    }

    func standardArgs(_ io: (patch: URL, target: URL, seed: URL, jobDir: URL)) -> [String] {
        [
            "--patch", io.patch.path, "--target", io.target.path,
            "--seed-params", io.seed.path, "--job-dir", io.jobDir.path,
            "--mode", "direction", "--fake-trainer", "--epochs", "60",
        ]
    }

    // MARK: - Tests

    func testHappyPathTranscript() throws {
        let io = try makeInputs()
        let outcome = try runTrain(standardArgs(io))

        XCTAssertEqual(outcome.exitStatus, 0)
        XCTAssertFalse(outcome.events.isEmpty)

        // plan first, terminal result last, nothing terminal in between.
        guard case .plan(let plan) = outcome.events.first else {
            return XCTFail("first event must be plan")
        }
        XCTAssertEqual(plan.seedEcho, ["sinefm": 0.06, "ratio": 0.05])
        guard case .result(let result) = outcome.events.last else {
            return XCTFail("last event must be result")
        }
        for event in outcome.events.dropLast() {
            XCTAssertFalse(event.isTerminal, "terminal event mid-stream")
        }

        // Referenced artifacts exist and live inside the job dir.
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.finalWav))
        XCTAssertTrue(result.finalWav.hasPrefix(io.jobDir.path))
        for case .checkpoint(let c) in outcome.events {
            XCTAssertTrue(FileManager.default.fileExists(atPath: c.wav))
        }
        for name in ["lowered.lisp", "final.wav", "result.json"] {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: io.jobDir.appendingPathComponent(name).path),
                "missing job-dir artifact \(name)")
        }
        // Trainer must not write the host-owned stream log.
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: io.jobDir.appendingPathComponent("events.jsonl").path))
    }

    func testPoisonedStdoutCannotCorruptProtocol() throws {
        let io = try makeInputs()
        let outcome = try runTrain(standardArgs(io), env: ["DGENLISP_TRAIN_POISON": "1"])

        XCTAssertEqual(outcome.exitStatus, 0)
        for line in outcome.stdoutLines {
            XCTAssertFalse(line.contains("POISON"), "poison reached stdout: \(line)")
            XCTAssertNoThrow(try TrainEventCoding.decodeLine(line))
        }
        XCTAssertTrue(outcome.stderr.contains("POISON"), "poison should surface on stderr")
        guard case .result = outcome.events.last else {
            return XCTFail("poisoned run must still terminate with result")
        }
    }

    func testCrashPathEmitsErrorEventAndNonzeroExit() throws {
        let io = try makeInputs()
        let outcome = try runTrain(standardArgs(io) + ["--fake-fail-at-epoch", "30"])

        XCTAssertNotEqual(outcome.exitStatus, 0)
        guard case .error(let err) = outcome.events.last else {
            return XCTFail("last event must be error, got \(String(describing: outcome.events.last))")
        }
        XCTAssertTrue(err.message.contains("epoch 30"))
        guard case .plan = outcome.events.first else {
            return XCTFail("plan should still be first on the crash path")
        }
    }

    func testMissingSeedFileFailsWithErrorEvent() throws {
        let io = try makeInputs()
        var args = standardArgs(io)
        args[5] = workDir.appendingPathComponent("nonexistent.json").path
        let outcome = try runTrain(args)
        XCTAssertNotEqual(outcome.exitStatus, 0)
        XCTAssertEqual(outcome.events.count, 1)
        guard case .error(let err) = outcome.events.first else {
            return XCTFail("expected a single error event")
        }
        XCTAssertTrue(err.message.contains("not found"))
    }

    func testUnknownFlagFailsWithErrorEvent() throws {
        let io = try makeInputs()
        let outcome = try runTrain(standardArgs(io) + ["--frobnicate"])
        XCTAssertNotEqual(outcome.exitStatus, 0)
        guard case .error = outcome.events.last else {
            return XCTFail("expected error event")
        }
    }

    func testCMAESOptionsAreAcceptedByTrainContract() throws {
        let io = try makeInputs()
        let outcome = try runTrain(standardArgs(io) + [
            "--search", "cma-es", "--cma-generations", "4",
            "--cma-population", "16", "--cma-sigma", "0.15",
            "--cma-seed", "7", "--cma-forward-batch", "8",
            "--cma-continue", "2", "--local-epochs", "0",
            "--cma-refine-epochs", "0", "--cma-final-epochs", "12",
            "--cma-refine-mode", "scalar",
        ])
        XCTAssertEqual(outcome.exitStatus, 0, outcome.stderr)
        guard case .result = outcome.events.last else {
            return XCTFail("expected terminal result")
        }
    }

    func testSigtermCancelsPromptlyWithoutTerminalEvent() throws {
        let io = try makeInputs()
        var args = standardArgs(io)
        args += ["--fake-epoch-ms", "20"]
        // Long job (60 epochs x 20ms = 1.2s); kill it at 0.3s.
        let start = Date()
        let outcome = try runTrain(args, sigtermAfter: 0.3)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertNotEqual(outcome.exitStatus, 0)
        XCTAssertLessThan(elapsed, 5.0, "SIGTERM must terminate promptly")
        // No terminal event required on cancel; whatever WAS emitted is valid
        // (already checked by the strict per-line parse in runTrain) and
        // must not include result.
        XCTAssertFalse(outcome.events.contains {
            if case .result = $0 { return true } else { return false }
        })
    }

    func testPlanOnlyEmitsRealPlanAndExitsSuccessfully() throws {
        let patch = workDir.appendingPathComponent("patch.lisp")
        try """
            (def freq (param freq @default 110 @min 50 @max 400))
            (def amp (param amp @default 0.5 @min 0 @max 1))
            (out (* amp (phasor freq)) 0)
            """.write(to: patch, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("target.wav")
        let sine = (0..<16384).map { i in
            Float(0.5 * sin(2.0 * Double.pi * 110.0 * Double(i) / 44100.0))
        }
        try MiniWav.write(url: target, samples: sine)
        let seed = workDir.appendingPathComponent("seed.json")
        try #"{"params":{"amp":0.8}}"#.write(to: seed, atomically: true, encoding: .utf8)
        let jobDir = workDir.appendingPathComponent("job")

        let outcome = try runTrain([
            "--patch", patch.path, "--target", target.path,
            "--seed-params", seed.path, "--job-dir", jobDir.path,
            "--plan-only",
        ])

        XCTAssertEqual(outcome.exitStatus, 0)
        XCTAssertEqual(outcome.events.count, 1)
        guard case .plan(let plan) = outcome.events.first else {
            return XCTFail("first event must be the real plan")
        }
        XCTAssertEqual(plan.learnable, ["amp", "freq"])
        XCTAssertTrue(plan.frozen.isEmpty)
        XCTAssertEqual(plan.seedEcho, ["amp": 0.8])
        XCTAssertEqual(plan.pitchHz, 110, accuracy: 2.0)
        XCTAssertEqual(plan.cropFrames, 16384)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: jobDir.appendingPathComponent("lowered.lisp").path))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: jobDir.appendingPathComponent("result.json").path))
    }

    func testPlanOnlyReportsNoLearnableParamsWithoutFailing() throws {
        let patch = workDir.appendingPathComponent("fixed.lisp")
        try "(out (phasor 110) 0)\n".write(
            to: patch, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("fixed-target.wav")
        try MiniWav.write(url: target, samples: [Float](repeating: 0, count: 1024))
        let seed = workDir.appendingPathComponent("fixed-seed.json")
        try #"{"params":{}}"#.write(to: seed, atomically: true, encoding: .utf8)
        let jobDir = workDir.appendingPathComponent("fixed-job")

        let outcome = try runTrain([
            "--patch", patch.path, "--target", target.path,
            "--seed-params", seed.path, "--job-dir", jobDir.path,
            "--pitch-hz", "110", "--plan-only",
        ])

        XCTAssertEqual(
            outcome.exitStatus, 0,
            "stderr: \(outcome.stderr); events: \(outcome.events)")
        XCTAssertEqual(outcome.events.count, 1)
        guard case .plan(let plan) = outcome.events.first else {
            return XCTFail("plan-only must emit exactly one plan")
        }
        XCTAssertTrue(plan.learnable.isEmpty)
    }

    func testMockHostScriptAcceptsStream() throws {
        let script = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests/DGenLispTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("scripts/consume_train_stream.py")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw XCTSkip("consume_train_stream.py not found at \(script.path)")
        }
        guard FileManager.default.fileExists(atPath: "/usr/bin/python3") else {
            throw XCTSkip("python3 unavailable")
        }

        let io = try makeInputs()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [script.path, dgenlispBinary.path] + standardArgs(io)
        let out = Pipe()
        process.standardOutput = out
        process.standardError = out
        try process.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertEqual(process.terminationStatus, 0, "mock host rejected the stream:\n\(text)")
    }
}

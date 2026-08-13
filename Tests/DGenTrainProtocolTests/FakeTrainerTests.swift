import XCTest

@testable import DGenTrainProtocol

final class FakeTrainerTests: XCTestCase {
    var workDir: URL!

    override func setUpWithError() throws {
        workDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dgen-train-proto-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDir)
    }

    func makeOptions(epochs: Int = 60) throws -> (TrainOptions, JobDir) {
        let patch = workDir.appendingPathComponent("patch.lisp")
        try "(out (phasor 110) 0)\n".write(to: patch, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("target.wav")
        try MiniWav.write(url: target, samples: [Float](repeating: 0, count: 64))
        let seed = workDir.appendingPathComponent("seed.json")
        try #"{"params":{"sinefm":0.06,"ratio":0.05,"sixteenth":0.36}}"#
            .write(to: seed, atomically: true, encoding: .utf8)
        let jobDirPath = workDir.appendingPathComponent("job").path

        var options = TrainOptions(
            patchPath: patch.path, targetPath: target.path,
            seedParamsPath: seed.path, jobDirPath: jobDirPath)
        options.epochs = epochs
        options.useFakeTrainer = true
        return (options, try JobDir(path: jobDirPath))
    }

    func testEventSequenceAndArtifacts() throws {
        let (options, jobDir) = try makeOptions(epochs: 60)
        let sink = CollectingEventSink()
        let result = try FakeTrainer.run(options: options, sink: sink, jobDir: jobDir)

        // Plan first, stage second.
        guard case .plan(let plan) = sink.events.first else {
            return XCTFail("first event must be plan, got \(String(describing: sink.events.first))")
        }
        guard case .stage(let stage) = sink.events[1] else {
            return XCTFail("second event must be stage")
        }
        XCTAssertEqual(stage.name, "train")
        XCTAssertEqual(stage.total, 60)

        // Seed echoed verbatim.
        XCTAssertEqual(plan.seedEcho, ["sinefm": 0.06, "ratio": 0.05, "sixteenth": 0.36])
        XCTAssertEqual(plan.learnable, ["ratio", "sinefm", "sixteenth"])

        // Epochs strictly increasing, losses non-increasing on the synthetic decay.
        var lastEpoch = 0
        var lastLoss = Double.infinity
        var checkpointCount = 0
        for event in sink.events.dropFirst(2) {
            switch event {
            case .epoch(let e):
                XCTAssertGreaterThan(e.epoch, lastEpoch)
                XCTAssertLessThan(e.loss, lastLoss)
                lastEpoch = e.epoch
                lastLoss = e.loss
            case .checkpoint(let c):
                checkpointCount += 1
                XCTAssertTrue(FileManager.default.fileExists(atPath: c.wav),
                              "checkpoint wav missing: \(c.wav)")
                XCTAssertTrue(c.wav.hasPrefix(jobDir.url.path), "artifact escaped job dir")
            default:
                XCTFail("unexpected mid-stream event \(event.typeName)")
            }
        }
        XCTAssertEqual(checkpointCount, 2)  // epochs 25, 50 at checkpointEvery=25

        // Trainer never emits the terminal event itself (the command does).
        XCTAssertFalse(sink.events.contains { $0.isTerminal })

        // Result payload sanity.
        XCTAssertGreaterThan(result.improvementPct, 50)
        XCTAssertEqual(result.basinCheck, "ok")
        XCTAssertEqual(Set(result.deltas.keys), Set(["sinefm", "ratio", "sixteenth"]))
        XCTAssertEqual(result.deltas["ratio"]?.from, 0.05)

        // Artifacts per spec §5.
        XCTAssertTrue(FileManager.default.fileExists(atPath: jobDir.loweredLisp.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: jobDir.finalWav.path))
        let lowered = try String(contentsOf: jobDir.loweredLisp, encoding: .utf8)
        XCTAssertTrue(lowered.contains("(out (phasor 110) 0)"))
    }

    func testDeterminism() throws {
        let (options, jobDir) = try makeOptions()
        let sinkA = CollectingEventSink()
        let resultA = try FakeTrainer.run(options: options, sink: sinkA, jobDir: jobDir)
        let sinkB = CollectingEventSink()
        let resultB = try FakeTrainer.run(options: options, sink: sinkB, jobDir: jobDir)
        XCTAssertEqual(sinkA.events, sinkB.events)
        XCTAssertEqual(resultA, resultB)
    }

    func testInducedFailureThrows() throws {
        var (options, jobDir) = try makeOptions()
        options.fakeFailAtEpoch = 30
        let sink = CollectingEventSink()
        XCTAssertThrowsError(try FakeTrainer.run(options: options, sink: sink, jobDir: jobDir)) {
            XCTAssertTrue("\($0)".contains("epoch 30"))
        }
        // Events before the failure are intact and non-terminal.
        XCTAssertTrue(sink.events.count >= 2)
        XCTAssertFalse(sink.events.contains { $0.isTerminal })
    }

    func testMalformedSeedRejected() throws {
        var (options, jobDir) = try makeOptions()
        let badSeed = workDir.appendingPathComponent("bad_seed.json")
        try #"{"sinefm":0.06}"#.write(to: badSeed, atomically: true, encoding: .utf8)
        options.seedParamsPath = badSeed.path
        XCTAssertThrowsError(
            try FakeTrainer.run(options: options, sink: CollectingEventSink(), jobDir: jobDir))
    }

    func testResultJSONMatchesStreamEncoding() throws {
        let (options, jobDir) = try makeOptions()
        let result = try FakeTrainer.run(
            options: options, sink: CollectingEventSink(), jobDir: jobDir)
        try jobDir.writeResult(result)
        let fileLine = try String(contentsOf: jobDir.resultJSON, encoding: .utf8)
            .trimmingCharacters(in: .newlines)
        let streamLine = try TrainEventCoding.encodeLine(.result(result))
        XCTAssertEqual(fileLine, streamLine)
        // And it round-trips through the strict decoder.
        guard case .result(let decoded) = try TrainEventCoding.decodeLine(fileLine) else {
            return XCTFail("result.json did not decode as a result event")
        }
        XCTAssertEqual(decoded, result)
    }
}

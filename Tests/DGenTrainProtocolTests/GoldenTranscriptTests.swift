import XCTest

@testable import DGenTrainProtocol

/// Golden-file test of the fake trainer's full NDJSON transcript.
/// Regenerate with: DGEN_UPDATE_GOLDEN=1 swift test --filter GoldenTranscriptTests
final class GoldenTranscriptTests: XCTestCase {
    func testFullTranscriptMatchesGolden() throws {
        let workDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dgen-train-golden-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workDir) }

        let patch = workDir.appendingPathComponent("patch.lisp")
        try "(out (phasor 110) 0)\n".write(to: patch, atomically: true, encoding: .utf8)
        let target = workDir.appendingPathComponent("target.wav")
        try MiniWav.write(url: target, samples: [Float](repeating: 0, count: 64))
        let seed = workDir.appendingPathComponent("seed.json")
        try #"{"params":{"sinefm":0.06,"ratio":0.05}}"#
            .write(to: seed, atomically: true, encoding: .utf8)

        let jobDirPath = workDir.appendingPathComponent("job").path
        var options = TrainOptions(
            patchPath: patch.path, targetPath: target.path,
            seedParamsPath: seed.path, jobDirPath: jobDirPath)
        options.epochs = 60
        options.useFakeTrainer = true
        let jobDir = try JobDir(path: jobDirPath)

        let sink = CollectingEventSink()
        let completion = try FakeTrainer.run(options: options, sink: sink, jobDir: jobDir)
        guard case .result(let result) = completion else {
            return XCTFail("full fake run must return a result")
        }

        // Reconstruct the full stream exactly as the command emits it:
        // progress events then the terminal result line.
        var lines: [String] = []
        for event in sink.events {
            lines.append(try TrainEventCoding.encodeLine(event))
        }
        lines.append(try TrainEventCoding.encodeLine(.result(result)))
        let transcript =
            lines.joined(separator: "\n")
            .replacingOccurrences(of: jobDir.url.path, with: "<job-dir>") + "\n"

        let goldenURL = Bundle.module.url(
            forResource: "fake_transcript.golden", withExtension: "ndjson",
            subdirectory: "Fixtures")

        if ProcessInfo.processInfo.environment["DGEN_UPDATE_GOLDEN"] != nil {
            let sourceGolden = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures/fake_transcript.golden.ndjson")
            try FileManager.default.createDirectory(
                at: sourceGolden.deletingLastPathComponent(), withIntermediateDirectories: true)
            try transcript.write(to: sourceGolden, atomically: true, encoding: .utf8)
            throw XCTSkip("golden updated at \(sourceGolden.path); rerun without DGEN_UPDATE_GOLDEN")
        }

        guard let goldenURL else {
            return XCTFail("missing Fixtures/fake_transcript.golden.ndjson — run with DGEN_UPDATE_GOLDEN=1 to create it")
        }
        let golden = try String(contentsOf: goldenURL, encoding: .utf8)
        XCTAssertEqual(transcript, golden,
                       "transcript drifted from golden; if intentional, regenerate with DGEN_UPDATE_GOLDEN=1")
    }
}

import XCTest

@testable import DGenTrainProtocol

final class EventCodingTests: XCTestCase {
    static let sampleEvents: [TrainEvent] = [
        .plan(
            PlanEvent(
                learnable: ["ratio", "sinefm"],
                frozen: [ParamVerdict(name: "base_note", reason: "f0-adjoint-unreliable")],
                unsupported: [ParamVerdict(name: "sync1", reason: "oscillator-sync")],
                seedEcho: ["sinefm": 0.06, "ratio": 0.05],
                pitchHz: 49.2, gateFrames: 8820, cropFrames: 32768)),
        .stage(StageEvent(name: "train", total: 300)),
        .optimizationProgress(
            OptimizationProgressEvent(current: 3, total: 12, losses: [0.031, 0.044, 0.052])),
        .epoch(
            EpochEvent(
                epoch: 50, total: 300, loss: 0.104,
                params: ["sinefm": 0.061, "ratio": 0.048],
                steps: ["sinefm": 0.7, "ratio": -0.2])),
        .checkpoint(CheckpointEvent(epoch: 100, wav: "/tmp/job/epoch0100.wav")),
        .result(
            ResultEvent(
                improvementPct: 54.2, absDistance: 0.0116, basinCheck: "ok",
                deltas: [
                    "sinefm": ParamDelta(from: 0.06, to: 0.11),
                    "ratio": ParamDelta(from: 0.05, to: 0.02),
                ],
                seededWav: "/tmp/job/seeded.wav", finalWav: "/tmp/job/final.wav")),
        .error(ErrorEvent(message: "boom")),
    ]

    func testRoundTripEveryEventType() throws {
        for event in Self.sampleEvents {
            let line = try TrainEventCoding.encodeLine(event)
            XCTAssertFalse(line.contains("\n"), "encoded event must be a single line")
            let decoded = try TrainEventCoding.decodeLine(line)
            XCTAssertEqual(decoded, event)
        }
    }

    func testSpecFieldNamesAreSnakeCase() throws {
        let planLine = try TrainEventCoding.encodeLine(Self.sampleEvents[0])
        for key in ["\"seed_echo\"", "\"pitch_hz\"", "\"gate_frames\"", "\"crop_frames\""] {
            XCTAssertTrue(planLine.contains(key), "plan missing \(key): \(planLine)")
        }
        let progressLine = try TrainEventCoding.encodeLine(Self.sampleEvents[2])
        for key in ["\"current\"", "\"total\"", "\"losses\""] {
            XCTAssertTrue(progressLine.contains(key), "optimization progress missing \(key): \(progressLine)")
        }
        let resultLine = try TrainEventCoding.encodeLine(Self.sampleEvents[5])
        for key in ["\"improvement_pct\"", "\"abs_distance\"", "\"basin_check\"", "\"seeded_wav\"", "\"final_wav\"", "\"deltas\"", "\"from\"", "\"to\"" ] {
            XCTAssertTrue(resultLine.contains(key), "result missing \(key): \(resultLine)")
        }
        let epochLine = try TrainEventCoding.encodeLine(Self.sampleEvents[3])
        XCTAssertTrue(epochLine.contains("\"steps\""), "epoch missing steps: \(epochLine)")
    }

    func testEpochWithoutStepsRemainsDecodable() throws {
        let event = try TrainEventCoding.decodeLine(
            #"{"epoch":50,"loss":0.104,"params":{"ratio":0.048},"total":300,"type":"epoch"}"#)
        guard case .epoch(let epoch) = event else {
            return XCTFail("expected epoch event")
        }
        XCTAssertNil(epoch.steps)
    }

    func testTypeDiscriminatorPresent() throws {
        for event in Self.sampleEvents {
            let line = try TrainEventCoding.encodeLine(event)
            XCTAssertTrue(line.contains("\"type\":\"\(event.typeName)\""), line)
        }
    }

    func testUnknownTypeRejected() {
        XCTAssertThrowsError(try TrainEventCoding.decodeLine(#"{"type":"telemetry","x":1}"#))
    }

    func testMalformedJSONRejected() {
        XCTAssertThrowsError(try TrainEventCoding.decodeLine("not json at all"))
        XCTAssertThrowsError(try TrainEventCoding.decodeLine(#"{"epoch":50}"#))
    }

    func testTerminalClassification() {
        XCTAssertTrue(TrainEvent.result(
            ResultEvent(improvementPct: 0, absDistance: 0, basinCheck: "ok", deltas: [:], seededWav: "", finalWav: "")
        ).isTerminal)
        XCTAssertTrue(TrainEvent.error(ErrorEvent(message: "x")).isTerminal)
        XCTAssertFalse(TrainEvent.stage(StageEvent(name: "train", total: 1)).isTerminal)
    }

    func testEncodingIsDeterministic() throws {
        for event in Self.sampleEvents {
            let a = try TrainEventCoding.encodeLine(event)
            let b = try TrainEventCoding.encodeLine(event)
            XCTAssertEqual(a, b)
        }
    }
}

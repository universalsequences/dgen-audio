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
        .epoch(
            EpochEvent(
                epoch: 50, total: 300, loss: 0.104,
                params: ["sinefm": 0.061, "ratio": 0.048])),
        .checkpoint(CheckpointEvent(epoch: 100, wav: "/tmp/job/epoch0100.wav")),
        .result(
            ResultEvent(
                improvementPct: 54.2, absDistance: 0.0116, basinCheck: "ok",
                deltas: [
                    "sinefm": ParamDelta(from: 0.06, to: 0.11),
                    "ratio": ParamDelta(from: 0.05, to: 0.02),
                ],
                finalWav: "/tmp/job/final.wav")),
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
        let resultLine = try TrainEventCoding.encodeLine(Self.sampleEvents[4])
        for key in ["\"improvement_pct\"", "\"abs_distance\"", "\"basin_check\"", "\"final_wav\"", "\"deltas\"", "\"from\"", "\"to\"" ] {
            XCTAssertTrue(resultLine.contains(key), "result missing \(key): \(resultLine)")
        }
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
            ResultEvent(improvementPct: 0, absDistance: 0, basinCheck: "ok", deltas: [:], finalWav: "")
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

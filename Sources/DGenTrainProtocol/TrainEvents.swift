// TrainEvents.swift — typed NDJSON event contract for `dgenlisp train`.
//
// Schema authority: eseq repo docs/patch-learn-spec.md §4 (rev 2).
// One JSON object per line on stdout, each with a "type" discriminator.
// This module is dependency-free (Foundation only) so the protocol layer
// can be tested without Metal or the DGen graph runtime.

import Foundation

// MARK: - Payloads

/// First event on the stream, before any compute: the lowering pass's
/// verdict on the patch plus the chosen excitation settings.
public struct PlanEvent: Codable, Equatable {
    public var learnable: [String]
    public var frozen: [ParamVerdict]
    public var unsupported: [ParamVerdict]
    /// The seed exactly as parsed from seed.json — the host diffs this
    /// against what it sent (naturalValues()-bug defense).
    public var seedEcho: [String: Double]
    public var pitchHz: Double
    public var gateFrames: Int
    public var cropFrames: Int

    enum CodingKeys: String, CodingKey {
        case learnable, frozen, unsupported
        case seedEcho = "seed_echo"
        case pitchHz = "pitch_hz"
        case gateFrames = "gate_frames"
        case cropFrames = "crop_frames"
    }

    public init(
        learnable: [String], frozen: [ParamVerdict], unsupported: [ParamVerdict],
        seedEcho: [String: Double], pitchHz: Double, gateFrames: Int, cropFrames: Int
    ) {
        self.learnable = learnable
        self.frozen = frozen
        self.unsupported = unsupported
        self.seedEcho = seedEcho
        self.pitchHz = pitchHz
        self.gateFrames = gateFrames
        self.cropFrames = cropFrames
    }
}

/// A frozen param or unsupported node, with the policy reason.
public struct ParamVerdict: Codable, Equatable {
    public var name: String
    public var reason: String

    public init(name: String, reason: String) {
        self.name = name
        self.reason = reason
    }
}

public struct StageEvent: Codable, Equatable {
    public var name: String
    public var total: Int

    public init(name: String, total: Int) {
        self.name = name
        self.total = total
    }
}

public struct EpochEvent: Codable, Equatable {
    public var epoch: Int
    public var total: Int
    public var loss: Double
    /// Current parameter values in natural/knob units.
    public var params: [String: Double]
    /// Applied post-Adam movement, normalized independently in each
    /// parameter's transformed coordinate system. Optional so older
    /// transcripts remain decodable.
    public var steps: [String: Double]?

    public init(
        epoch: Int, total: Int, loss: Double, params: [String: Double],
        steps: [String: Double]? = nil
    ) {
        self.epoch = epoch
        self.total = total
        self.loss = loss
        self.params = params
        self.steps = steps
    }
}

public struct CheckpointEvent: Codable, Equatable {
    public var epoch: Int
    /// Absolute path to a preview render inside the job dir.
    public var wav: String

    public init(epoch: Int, wav: String) {
        self.epoch = epoch
        self.wav = wav
    }
}

public struct ResultEvent: Codable, Equatable {
    public var improvementPct: Double
    /// Always reported alongside the percentage (corrected-baseline lesson).
    public var absDistance: Double
    /// "ok" | "wrong_neighborhood"
    public var basinCheck: String
    public var deltas: [String: ParamDelta]
    public var seededWav: String
    public var finalWav: String

    enum CodingKeys: String, CodingKey {
        case improvementPct = "improvement_pct"
        case absDistance = "abs_distance"
        case basinCheck = "basin_check"
        case deltas
        case seededWav = "seeded_wav"
        case finalWav = "final_wav"
    }

    public init(
        improvementPct: Double, absDistance: Double, basinCheck: String,
        deltas: [String: ParamDelta], seededWav: String, finalWav: String
    ) {
        self.improvementPct = improvementPct
        self.absDistance = absDistance
        self.basinCheck = basinCheck
        self.deltas = deltas
        self.seededWav = seededWav
        self.finalWav = finalWav
    }
}

public struct ParamDelta: Codable, Equatable {
    public var from: Double
    public var to: Double

    public init(from: Double, to: Double) {
        self.from = from
        self.to = to
    }
}

public struct ErrorEvent: Codable, Equatable {
    public var message: String

    public init(message: String) {
        self.message = message
    }
}

// MARK: - Envelope

public enum TrainEvent: Equatable {
    case plan(PlanEvent)
    case stage(StageEvent)
    case epoch(EpochEvent)
    case checkpoint(CheckpointEvent)
    case result(ResultEvent)
    case error(ErrorEvent)

    public var typeName: String {
        switch self {
        case .plan: return "plan"
        case .stage: return "stage"
        case .epoch: return "epoch"
        case .checkpoint: return "checkpoint"
        case .result: return "result"
        case .error: return "error"
        }
    }

    /// True for the events allowed (and required) to terminate the stream.
    public var isTerminal: Bool {
        switch self {
        case .result, .error: return true
        default: return false
        }
    }
}

extension TrainEvent: Codable {
    private enum TypeKey: String, CodingKey { case type }

    public init(from decoder: Decoder) throws {
        let probe = try decoder.container(keyedBy: TypeKey.self)
        let type = try probe.decode(String.self, forKey: .type)
        switch type {
        case "plan": self = .plan(try PlanEvent(from: decoder))
        case "stage": self = .stage(try StageEvent(from: decoder))
        case "epoch": self = .epoch(try EpochEvent(from: decoder))
        case "checkpoint": self = .checkpoint(try CheckpointEvent(from: decoder))
        case "result": self = .result(try ResultEvent(from: decoder))
        case "error": self = .error(try ErrorEvent(from: decoder))
        default:
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unknown train event type: \(type)"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TypeKey.self)
        try container.encode(typeName, forKey: .type)
        switch self {
        case .plan(let p): try p.encode(to: encoder)
        case .stage(let p): try p.encode(to: encoder)
        case .epoch(let p): try p.encode(to: encoder)
        case .checkpoint(let p): try p.encode(to: encoder)
        case .result(let p): try p.encode(to: encoder)
        case .error(let p): try p.encode(to: encoder)
        }
    }
}

// MARK: - Line coding

public enum TrainEventCoding {
    /// Shared encoder so the emitter, result.json writer, and tests
    /// produce byte-identical JSON (sorted keys, no slash escaping).
    public static func encodeLine(_ event: TrainEvent) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(event)
        guard let string = String(data: data, encoding: .utf8) else {
            throw TrainProtocolError("Event did not encode as UTF-8")
        }
        return string
    }

    public static func decodeLine(_ line: String) throws -> TrainEvent {
        guard let data = line.data(using: .utf8) else {
            throw TrainProtocolError("Line is not UTF-8")
        }
        return try JSONDecoder().decode(TrainEvent.self, from: data)
    }
}

public struct TrainProtocolError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

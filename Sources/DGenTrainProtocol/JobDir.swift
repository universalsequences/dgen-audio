// JobDir.swift — the trainer's sandbox (patch-learn-spec §5).
//
// The host creates learn-jobs/<id>/ and passes it via --job-dir. The
// trainer writes ONLY here: lowered.lisp, seeded.wav, epoch*.wav, final.wav,
// result.json. events.jsonl belongs to the host (it appends the consumed
// stream) — the trainer must never touch it.

import Foundation

public struct JobDir {
    public let url: URL

    public init(path: String) throws {
        let url = URL(fileURLWithPath: path, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true)
        } catch {
            throw TrainProtocolError("Cannot create job dir \(path): \(error.localizedDescription)")
        }
        self.url = url
    }

    public func file(_ name: String) -> URL {
        url.appendingPathComponent(name)
    }

    public var loweredLisp: URL { file("lowered.lisp") }
    public var renderLisp: URL { file("render.lisp") }
    public var seededWav: URL { file("seeded.wav") }
    public var finalWav: URL { file("final.wav") }
    public var resultJSON: URL { file("result.json") }

    public func epochWav(_ epoch: Int) -> URL {
        file(String(format: "epoch%04d.wav", epoch))
    }

    /// result.json holds the terminal result event, byte-identical to the
    /// stream line (same encoder).
    public func writeResult(_ result: ResultEvent) throws {
        let line = try TrainEventCoding.encodeLine(.result(result)) + "\n"
        try line.data(using: .utf8)!.write(to: resultJSON, options: .atomic)
    }
}

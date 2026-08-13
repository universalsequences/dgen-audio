// TrainCommand.swift — orchestrates one `dgenlisp train` invocation and
// owns the termination contract (patch-learn-spec §3/§4):
//
//   - stdout is claimed before anything else runs; all other output paths
//     are redirected to stderr.
//   - The last stdout line is ALWAYS a result or error event.
//   - Exit code 0 iff a result event was emitted; nonzero otherwise.
//   - Any thrown error becomes an error event + exit 1.
//   - SIGTERM = cancel: default disposition (prompt death, no terminal
//     event required, artifacts left in place) — deliberately no handler.

import Foundation

public enum TrainCommand {
    /// A trainer backend: emits progress events on the sink, writes
    /// artifacts into the job dir, returns the terminal result.
    public typealias Trainer = (TrainOptions, TrainEventSink, JobDir) throws -> ResultEvent

    /// Runs the subcommand with `arguments` = argv after the "train" word.
    /// `realTrainer` handles non-fake jobs (nil until Phase C is wired).
    public static func run(arguments: [String], realTrainer: Trainer?) -> Never {
        let emitter = EventEmitter.claimStdout()
        emitPoisonIfRequested("post-claim")  // test hook: must land on stderr

        func fail(_ message: String) -> Never {
            try? emitter.emit(.error(ErrorEvent(message: message)))
            exit(1)
        }

        do {
            let options = try TrainOptions.parse(arguments)
            let jobDir = try JobDir(path: options.jobDirPath)
            let trainer: Trainer
            if options.useFakeTrainer {
                trainer = FakeTrainer.run
            } else if let realTrainer {
                trainer = realTrainer
            } else {
                fail("real trainer not wired; use --fake-trainer")
            }
            let result = try trainer(options, emitter, jobDir)
            emitPoisonIfRequested("pre-result")
            try jobDir.writeResult(result)
            try emitter.emit(.result(result))
            exit(0)
        } catch let error as TrainProtocolError {
            fail(error.message)
        } catch {
            fail("\(error)")
        }
    }

    /// When DGENLISP_TRAIN_POISON is set, deliberately write through every
    /// "wrong" output path. The poisoned-stdout test asserts none of it
    /// reaches the protocol stream.
    private static func emitPoisonIfRequested(_ site: String) {
        guard ProcessInfo.processInfo.environment["DGENLISP_TRAIN_POISON"] != nil else { return }
        print("POISON print() at \(site)")
        fputs("POISON fputs(stdout) at \(site)\n", stdout)
        FileHandle.standardOutput.write(Data("POISON FileHandle at \(site)\n".utf8))
    }
}

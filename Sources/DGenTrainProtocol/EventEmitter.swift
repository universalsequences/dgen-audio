// EventEmitter.swift — the ONLY writer of protocol events.
//
// The train subcommand claims the real stdout file descriptor at startup:
// the original fd 1 is dup()ed for the emitter's exclusive use, and stderr
// is dup2()ed over fd 1. After that, any stray print(), fputs(stdout), or
// FileHandle.standardOutput write in the entire process lands on stderr —
// a stray print can physically never corrupt the NDJSON stream.
//
// Every event is written as one line followed by "\n" with a direct
// write(2) loop (no stdio buffering), so the host reads events live.

import Foundation

public protocol TrainEventSink: AnyObject {
    func emit(_ event: TrainEvent) throws
}

public final class EventEmitter: TrainEventSink {
    private let fd: Int32
    public private(set) var lastEventWasTerminal = false
    public private(set) var resultEmitted = false

    public init(fileDescriptor: Int32) {
        self.fd = fileDescriptor
    }

    /// Take exclusive ownership of the process's real stdout and redirect
    /// fd 1 to stderr so no other code path can reach the protocol stream.
    public static func claimStdout() -> EventEmitter {
        let saved = dup(STDOUT_FILENO)
        precondition(saved >= 0, "dup(stdout) failed")
        dup2(STDERR_FILENO, STDOUT_FILENO)
        // stdout's stdio stream now feeds stderr; make it line-buffered so
        // stray prints show up promptly in diagnostics instead of at exit.
        setvbuf(stdout, nil, _IONBF, 0)
        return EventEmitter(fileDescriptor: saved)
    }

    public func emit(_ event: TrainEvent) throws {
        let line = try TrainEventCoding.encodeLine(event) + "\n"
        try writeFully(Array(line.utf8))
        lastEventWasTerminal = event.isTerminal
        if case .result = event { resultEmitted = true }
    }

    private func writeFully(_ bytes: [UInt8]) throws {
        var offset = 0
        while offset < bytes.count {
            let n = bytes[offset...].withUnsafeBytes { buf -> Int in
                write(fd, buf.baseAddress, buf.count)
            }
            if n < 0 {
                if errno == EINTR { continue }
                throw TrainProtocolError("write to event stream failed: errno \(errno)")
            }
            offset += n
        }
    }
}

/// In-memory sink for unit tests.
public final class CollectingEventSink: TrainEventSink {
    public private(set) var events: [TrainEvent] = []
    public init() {}
    public func emit(_ event: TrainEvent) throws {
        events.append(event)
    }
}

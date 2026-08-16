// MiniWav.swift — minimal 16-bit PCM mono WAV writer.
//
// Exists so the fake trainer (and its tests) can produce real, readable
// artifacts without depending on DGenLazy/AudioFile (which links Metal).

import Foundation

public enum MiniWav {
    public static func write(url: URL, samples: [Float], sampleRate: Int = 44100) throws {
        var data = Data()
        let byteRate = sampleRate * 2
        let dataSize = samples.count * 2

        func append(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        func append16(_ value: UInt16) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + dataSize))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        append(16)  // fmt chunk size
        append16(1)  // PCM
        append16(1)  // mono
        append(UInt32(sampleRate))
        append(UInt32(byteRate))
        append16(2)  // block align
        append16(16)  // bits per sample
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(dataSize))
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            append16(UInt16(bitPattern: Int16(clamped * 32767)))
        }
        try data.write(to: url, options: .atomic)
    }
}

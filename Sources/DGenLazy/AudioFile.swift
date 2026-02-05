// AudioFile - WAV file loading and saving utilities
//
// Provides standalone WAV I/O without requiring a compiled kernel.
// Load: Parses raw WAV bytes (supports PCM int16 and float32)
// Save: Writes 32-bit float mono WAV (adapted from Runtime.swift)

import Foundation
import DGen

// MARK: - AudioFile

public struct AudioFile {

    public enum AudioFileError: Error, CustomStringConvertible {
        case invalidFormat(String)
        case fileNotFound(String)
        case unsupportedFormat(String)

        public var description: String {
            switch self {
            case .invalidFormat(let msg): return "Invalid WAV format: \(msg)"
            case .fileNotFound(let msg): return "File not found: \(msg)"
            case .unsupportedFormat(let msg): return "Unsupported format: \(msg)"
            }
        }
    }

    /// Load a .wav file into a flat [Float] array
    ///
    /// - Parameters:
    ///   - url: Path to the .wav file
    ///   - mono: If true, mix multi-channel audio down to mono (default: true)
    /// - Returns: Tuple of (samples, sampleRate)
    public static func load(url: URL, mono: Bool = true) throws -> (samples: [Float], sampleRate: Float) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioFileError.fileNotFound(url.path)
        }
        let data = try Data(contentsOf: url)
        return try parseWAV(data: data, mono: mono)
    }

    /// Load a .wav file directly into a DGenLazy Tensor
    ///
    /// - Parameters:
    ///   - url: Path to the .wav file
    ///   - mono: If true, mix to mono (default: true)
    /// - Returns: 1D Tensor containing the audio samples
    public static func loadTensor(url: URL, mono: Bool = true) throws -> Tensor {
        let (samples, _) = try load(url: url, mono: mono)
        return Tensor(samples)
    }

    /// Export [Float] samples to a .wav file (32-bit float, mono)
    ///
    /// - Parameters:
    ///   - url: Destination file path
    ///   - samples: Audio sample data
    ///   - sampleRate: Sample rate in Hz (default: DGenConfig.sampleRate)
    public static func save(url: URL, samples: [Float], sampleRate: Float = DGenConfig.sampleRate) throws {
        let bytesPerSample = 4
        let numChannels = 1
        let byteRate = Int(sampleRate) * numChannels * bytesPerSample
        let blockAlign = numChannels * bytesPerSample
        let dataBytes = samples.count * numChannels * bytesPerSample
        let riffSize = 36 + dataBytes

        var data = Data()

        func append(_ s: String) { data.append(s.data(using: .ascii)!) }
        func appendUInt32(_ v: UInt32) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }
        func appendUInt16(_ v: UInt16) {
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        append("RIFF")
        appendUInt32(UInt32(riffSize))
        append("WAVE")
        append("fmt ")
        appendUInt32(16)                         // PCM fmt chunk size
        appendUInt16(3)                          // AudioFormat 3 = IEEE float
        appendUInt16(UInt16(numChannels))
        appendUInt32(UInt32(sampleRate))
        appendUInt32(UInt32(byteRate))
        appendUInt16(UInt16(blockAlign))
        appendUInt16(32)                         // bits per sample
        append("data")
        appendUInt32(UInt32(dataBytes))

        samples.withUnsafeBytes { raw in
            data.append(contentsOf: raw.bindMemory(to: UInt8.self))
        }

        try data.write(to: url, options: .atomic)
    }

    // MARK: - WAV Parsing

    static func parseWAV(data: Data, mono: Bool) throws -> (samples: [Float], sampleRate: Float) {
        guard data.count >= 44 else {
            throw AudioFileError.invalidFormat("File too small for WAV header")
        }

        // Validate RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        let wave = String(data: data[8..<12], encoding: .ascii)
        guard riff == "RIFF", wave == "WAVE" else {
            throw AudioFileError.invalidFormat("Missing RIFF/WAVE header")
        }

        // Walk chunks to find "fmt " and "data"
        var audioFormat: UInt16 = 0
        var numChannels: UInt16 = 0
        var sampleRate: UInt32 = 0
        var bitsPerSample: UInt16 = 0
        var dataOffset = 0
        var dataSize = 0
        var foundFmt = false
        var foundData = false

        var offset = 12  // Skip RIFF header
        while offset + 8 <= data.count {
            let chunkId = String(data: data[offset..<offset+4], encoding: .ascii) ?? ""
            let chunkSize = Int(readUInt32(from: data, at: offset + 4))
            let chunkDataStart = offset + 8

            if chunkId == "fmt " {
                guard chunkSize >= 16 else {
                    throw AudioFileError.invalidFormat("fmt chunk too small")
                }
                audioFormat = readUInt16(from: data, at: chunkDataStart)
                numChannels = readUInt16(from: data, at: chunkDataStart + 2)
                sampleRate = readUInt32(from: data, at: chunkDataStart + 4)
                bitsPerSample = readUInt16(from: data, at: chunkDataStart + 14)
                foundFmt = true
            } else if chunkId == "data" {
                dataOffset = chunkDataStart
                dataSize = chunkSize
                foundData = true
            }

            // Advance to next chunk (chunks are word-aligned)
            offset = chunkDataStart + chunkSize
            if chunkSize % 2 != 0 { offset += 1 }

            if foundFmt && foundData { break }
        }

        guard foundFmt else { throw AudioFileError.invalidFormat("No fmt chunk found") }
        guard foundData else { throw AudioFileError.invalidFormat("No data chunk found") }
        guard numChannels >= 1 else { throw AudioFileError.invalidFormat("Invalid channel count") }

        let channels = Int(numChannels)
        var samples: [Float]

        switch audioFormat {
        case 1:  // PCM integer
            guard bitsPerSample == 16 else {
                throw AudioFileError.unsupportedFormat("PCM \(bitsPerSample)-bit (only 16-bit supported)")
            }
            let bytesPerSample = 2
            let totalSamples = dataSize / bytesPerSample
            let framesCount = totalSamples / channels
            samples = [Float](repeating: 0, count: framesCount * channels)

            data.withUnsafeBytes { buf in
                for i in 0..<totalSamples {
                    let raw = buf.load(fromByteOffset: dataOffset + i * 2, as: Int16.self).littleEndian
                    samples[i] = Float(raw) / 32768.0
                }
            }

        case 3:  // IEEE float
            guard bitsPerSample == 32 else {
                throw AudioFileError.unsupportedFormat("Float \(bitsPerSample)-bit (only 32-bit supported)")
            }
            let totalSamples = dataSize / 4
            let framesCount = totalSamples / channels
            samples = [Float](repeating: 0, count: framesCount * channels)

            data.withUnsafeBytes { buf in
                for i in 0..<totalSamples {
                    samples[i] = buf.load(fromByteOffset: dataOffset + i * 4, as: Float.self)
                }
            }

        default:
            throw AudioFileError.unsupportedFormat("audioFormat \(audioFormat) (only PCM int16 and float32 supported)")
        }

        // Mono mixdown if needed
        if mono && channels > 1 {
            let framesCount = samples.count / channels
            var monoSamples = [Float](repeating: 0, count: framesCount)
            let scale = 1.0 / Float(channels)
            for f in 0..<framesCount {
                var sum: Float = 0
                for ch in 0..<channels {
                    sum += samples[f * channels + ch]
                }
                monoSamples[f] = sum * scale
            }
            samples = monoSamples
        }

        return (samples: samples, sampleRate: Float(sampleRate))
    }

    // MARK: - Binary Helpers

    static func readUInt32(from data: Data, at offset: Int) -> UInt32 {
        data.withUnsafeBytes { buf in
            buf.load(fromByteOffset: offset, as: UInt32.self).littleEndian
        }
    }

    static func readUInt16(from data: Data, at offset: Int) -> UInt16 {
        data.withUnsafeBytes { buf in
            buf.load(fromByteOffset: offset, as: UInt16.self).littleEndian
        }
    }
}

// MARK: - Signal.exportToWav

extension Signal {
    /// Realize this signal and export to a WAV file
    ///
    /// - Parameters:
    ///   - url: Destination file path
    ///   - frames: Number of frames to realize
    ///   - sampleRate: Sample rate in Hz (default: DGenConfig.sampleRate)
    public func exportToWav(url: URL, frames: Int, sampleRate: Float = DGenConfig.sampleRate) throws {
        let samples = try realize(frames: frames)
        try AudioFile.save(url: url, samples: samples, sampleRate: sampleRate)
    }
}

// Minimal RIFF/WAVE reader (PCM 8/16/24/32, IEEE float 32/64,
// WAVE_FORMAT_EXTENSIBLE) and a float32 writer.

import Foundation

public struct WavAudio {
    public var sampleRate: Double
    /// channels[c][i]
    public var channels: [[Double]]
    public var frameCount: Int { channels.first?.count ?? 0 }
    public var bitsPerSample: Int

    public init(sampleRate: Double, channels: [[Double]], bitsPerSample: Int = 32) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.bitsPerSample = bitsPerSample
    }

    /// Mono mix: mean of the channels.
    public var mono: [Double] {
        guard let first = channels.first else { return [] }
        if channels.count == 1 { return first }
        var out = [Double](repeating: 0, count: first.count)
        for ch in channels { for i in 0..<out.count { out[i] += ch[i] } }
        let s = 1.0 / Double(channels.count)
        for i in 0..<out.count { out[i] *= s }
        return out
    }
}

public enum WavError: Error, CustomStringConvertible {
    case notWave(String)
    case unsupported(String)
    public var description: String {
        switch self {
        case .notWave(let s): return "not a WAVE file: \(s)"
        case .unsupported(let s): return "unsupported WAVE format: \(s)"
        }
    }
}

public enum WAV {
    public static func read(_ url: URL) throws -> WavAudio {
        let data = try Data(contentsOf: url)
        return try parse(data, name: url.lastPathComponent)
    }

    public static func parse(_ data: Data, name: String = "") throws -> WavAudio {
        let bytes = [UInt8](data)
        func u32(_ o: Int) -> UInt32 {
            UInt32(bytes[o]) | UInt32(bytes[o + 1]) << 8 | UInt32(bytes[o + 2]) << 16 | UInt32(bytes[o + 3]) << 24
        }
        func u16(_ o: Int) -> UInt16 { UInt16(bytes[o]) | UInt16(bytes[o + 1]) << 8 }
        guard bytes.count >= 12, String(bytes: bytes[0..<4], encoding: .ascii) == "RIFF",
            String(bytes: bytes[8..<12], encoding: .ascii) == "WAVE"
        else { throw WavError.notWave(name) }
        var off = 12
        var fmtTag = 0, chans = 0, sr = 0, bits = 0
        var dataRange: Range<Int>? = nil
        while off + 8 <= bytes.count {
            let id = String(bytes: bytes[off..<(off + 4)], encoding: .ascii) ?? ""
            let size = Int(u32(off + 4))
            let body = off + 8
            if id == "fmt " {
                fmtTag = Int(u16(body))
                chans = Int(u16(body + 2))
                sr = Int(u32(body + 4))
                bits = Int(u16(body + 14))
                if fmtTag == 0xFFFE && size >= 26 { fmtTag = Int(u16(body + 24)) }
            } else if id == "data" {
                dataRange = body..<min(bytes.count, body + size)
            }
            off = body + size + (size & 1)
        }
        guard let dr = dataRange, chans > 0, sr > 0 else { throw WavError.notWave(name) }
        let bps = bits / 8
        let frames = dr.count / (bps * chans)
        var out = [[Double]](repeating: [Double](repeating: 0, count: frames), count: chans)
        for f in 0..<frames {
            for c in 0..<chans {
                let o = dr.lowerBound + (f * chans + c) * bps
                var v: Double
                switch (fmtTag, bits) {
                case (1, 8): v = (Double(bytes[o]) - 128) / 128
                case (1, 16): v = Double(Int16(bitPattern: u16(o))) / 32768
                case (1, 24):
                    var i = Int32(bytes[o]) | Int32(bytes[o + 1]) << 8 | Int32(bytes[o + 2]) << 16
                    if i & 0x800000 != 0 { i |= ~0xFFFFFF }
                    v = Double(i) / 8388608
                case (1, 32): v = Double(Int32(bitPattern: u32(o))) / 2147483648
                case (3, 32): v = Double(Float(bitPattern: u32(o)))
                case (3, 64):
                    let lo = UInt64(u32(o)), hi = UInt64(u32(o + 4))
                    v = Double(bitPattern: lo | hi << 32)
                default: throw WavError.unsupported("\(name): tag \(fmtTag) bits \(bits)")
                }
                out[c][f] = v
            }
        }
        return WavAudio(sampleRate: Double(sr), channels: out, bitsPerSample: bits)
    }

    /// Write mono float32 WAV.
    public static func writeFloat32(_ x: [Double], sampleRate: Double, to url: URL) throws {
        var d = Data()
        func put32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func put16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let n = x.count
        d.append(contentsOf: Array("RIFF".utf8)); put32(UInt32(36 + 4 * n))
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); put32(16); put16(3); put16(1)
        put32(UInt32(sampleRate)); put32(UInt32(sampleRate) * 4); put16(4); put16(32)
        d.append(contentsOf: Array("data".utf8)); put32(UInt32(4 * n))
        for v in x { put32(Float(v).bitPattern) }
        try d.write(to: url)
    }
}

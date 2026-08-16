import XCTest

@testable import DGenLazy

/// Malformed/odd-but-legal WAV files must produce a thrown `AudioFileError`
/// (or correct samples), never a trap or an out-of-bounds read. The train CLI
/// feeds arbitrary user `--target` files straight into this parser, and a trap
/// kills the process without the contracted terminal error event.
final class WAVParsingTests: XCTestCase {

  private func u16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }
  private func u32(_ v: UInt32) -> [UInt8] {
    [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)]
  }

  /// Build a WAV with full control over the fmt chunk size, the declared data
  /// size, and any ancillary chunk placed between fmt and data.
  private func makeWAV(
    format: UInt16, channels: UInt16, sampleRate: UInt32, bits: UInt16,
    fmtChunkSize: UInt32 = 16, extraChunk: (id: String, payload: [UInt8])? = nil,
    payload: [UInt8], declaredDataSize: UInt32? = nil
  ) -> Data {
    var body: [UInt8] = Array("WAVE".utf8)
    body += Array("fmt ".utf8) + u32(fmtChunkSize)
    let blockAlign = UInt16(Int(channels) * Int(bits) / 8)
    var fmt: [UInt8] = []
    fmt += u16(format) + u16(channels) + u32(sampleRate)
    fmt += u32(sampleRate * UInt32(blockAlign)) + u16(blockAlign) + u16(bits)
    while fmt.count < Int(fmtChunkSize) { fmt.append(0) }
    body += fmt
    if let extraChunk {
      body += Array(extraChunk.id.utf8) + u32(UInt32(extraChunk.payload.count))
      body += extraChunk.payload
    }
    body += Array("data".utf8) + u32(declaredDataSize ?? UInt32(payload.count)) + payload
    var out: [UInt8] = Array("RIFF".utf8) + u32(UInt32(body.count))
    out += body
    return Data(out)
  }

  private func floatBytes(_ values: [Float]) -> [UInt8] {
    values.flatMap { value -> [UInt8] in
      let bits = value.bitPattern.littleEndian
      return [
        UInt8(bits & 0xFF), UInt8((bits >> 8) & 0xFF),
        UInt8((bits >> 16) & 0xFF), UInt8((bits >> 24) & 0xFF),
      ]
    }
  }

  func testDataSizeNotMultipleOfBlockAlignFloorsToWholeFrames() throws {
    // Stereo int16 (blockAlign 4) with 6 payload bytes: 1.5 frames. The partial
    // frame is dropped; before the fix the fill loop indexed past the array.
    let wav = makeWAV(
      format: 1, channels: 2, sampleRate: 8000, bits: 16,
      payload: [UInt8](repeating: 1, count: 6))
    let (samples, _) = try AudioFile.parseWAV(data: wav, mono: true)
    XCTAssertEqual(samples.count, 1, "expected exactly one complete stereo frame")
  }

  func testDataChunkWithNoCompleteFrameThrows() {
    // Stereo int16 with only 2 payload bytes: not even one frame.
    let wav = makeWAV(
      format: 1, channels: 2, sampleRate: 8000, bits: 16,
      payload: [UInt8](repeating: 1, count: 2))
    XCTAssertThrowsError(try AudioFile.parseWAV(data: wav, mono: true)) { error in
      XCTAssertTrue(error is AudioFile.AudioFileError, "unexpected error: \(error)")
    }
  }

  func testDeclaredDataSizeBeyondFileIsClamped() throws {
    // Streaming encoders write 0xFFFFFFFF as a data-size placeholder.
    let wav = makeWAV(
      format: 3, channels: 1, sampleRate: 8000, bits: 32,
      payload: floatBytes([0.25, -0.5, 0.75]), declaredDataSize: 0xFFFF_FFFF)
    let (samples, rate) = try AudioFile.parseWAV(data: wav, mono: true)
    XCTAssertEqual(rate, 8000)
    XCTAssertEqual(samples, [0.25, -0.5, 0.75])
  }

  func testTruncatedFmtChunkThrows() {
    var wav = makeWAV(
      format: 1, channels: 1, sampleRate: 8000, bits: 16,
      payload: [UInt8](repeating: 0, count: 8))
    // Chop the file mid-fmt: the header still declares a 16-byte fmt chunk.
    wav = wav.prefix(24)
    XCTAssertThrowsError(try AudioFile.parseWAV(data: wav, mono: true)) { error in
      XCTAssertTrue(error is AudioFile.AudioFileError, "unexpected error: \(error)")
    }
  }

  func testFloat32WithUnalignedDataChunkLoads() throws {
    // fmt size 18 (cbSize) + a 4-byte `fact` chunk puts the data payload at a
    // byte offset that is 2 (mod 4) — an aligned raw load traps here.
    let wav = makeWAV(
      format: 3, channels: 1, sampleRate: 8000, bits: 32, fmtChunkSize: 18,
      extraChunk: ("fact", u32(3)), payload: floatBytes([1.0, -0.25, 0.5]))
    let (samples, rate) = try AudioFile.parseWAV(data: wav, mono: true)
    XCTAssertEqual(rate, 8000)
    XCTAssertEqual(samples, [1.0, -0.25, 0.5])
  }
}

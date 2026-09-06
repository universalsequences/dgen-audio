import Foundation
import XCTest
import DGenHostSupport

@testable import DGen
@testable import DGenLazy
@testable import DGenLisp

final class SpectralSchedulingTests: XCTestCase {
  func testStereoRoundTripWithSharedOutputFeedbackPreservesHopFramesAndScalarCadence() throws {
    let oldBackend = DGenConfig.backend
    let oldRate = DGenConfig.sampleRate
    let oldFrames = DGenConfig.maxFrameCount
    defer {
      DGenConfig.backend = oldBackend
      DGenConfig.sampleRate = oldRate
      DGenConfig.maxFrameCount = oldFrames
      LazyGraphContext.reset()
    }
    DGenConfig.backend = .c
    DGenConfig.sampleRate = 48000
    DGenConfig.maxFrameCount = 512
    let count = 8192
    let n = 64
    let hop = n / 4
    let input: [[Float]] = [997.0, 1409.0].enumerated().map { channel, frequency in
      (0..<count).map { frame in
        Float(channel == 0 ? 0.2 : 0.13) * sin(2 * Float.pi * Float(frequency) * Float(frame) / 48000)
      }
    }
    // A slow gain ramp exposes accidental per-element history advancement,
    // even when a one-hop host block masks the tensor lifetime defect.
    var gain = Float(0)
    let coefficient = 2 * Float.pi * 2 / 48000
    let gains: [Float] = (0..<count).map { _ in
      gain += coefficient * (1 - gain)
      return gain
    }

    for backend in ["accelerated", "manual"] {
      LazyGraphContext.reset()
      let lazy = LazyGraphContext.current
      let evaluator = LispEvaluator()
      try evaluator.evaluate(nodes: parseSource("""
        (def left (in 1))
        (def right (in 2))
        (def win (hann \(n)))
        (def frame-l (* (reshape (buffer left \(n) \(hop)) @shape [\(n)]) win))
        (def frame-r (* (reshape (buffer right \(n) \(hop)) @shape [\(n)]) win))
        (def (re-l im-l) (fft frame-l @N \(n) @backend \(backend)))
        (def (re-r im-r) (fft frame-r @N \(n) @backend \(backend)))
        (def time-l (ifft re-l im-l @N \(n) @backend \(backend)))
        (def time-r (ifft re-r im-r @N \(n) @backend \(backend)))
        (make-history previous)
        (def coefficient (/ (* twopi 2) samplerate))
        (def gain (write-history previous (+ (read-history previous) (* coefficient (- 1 (read-history previous))))))
        (out (* gain (/ (overlap-add (* time-l win) \(hop)) 1.5)) 1)
        (out (* gain (/ (overlap-add (* time-r win) \(hop)) 1.5)) 2)
        """))
      for output in evaluator.outputs { lazy.addOutput(output.signal, channel: output.channel) }
      let compiled = try CompilationPipeline.compile(
        graph: lazy.graph, backend: .c, options: .init(frameCount: 512, debug: false))
      let kernel = CCompiledKernel(
        source: compiled.kernels.map { $0.source }.joined(separator: "\n\n"),
        cellAllocations: compiled.cellAllocations, memorySize: compiled.totalMemorySlots,
        defaultHostSampleRate: 48000)
      try kernel.compileAndLoad()
      defer { kernel.cleanup() }
      let process = try XCTUnwrap(kernel.getProcessFunction())
      let memory = UnsafeMutablePointer<Float>.allocate(capacity: compiled.totalMemorySlots)
      defer { memory.deallocate() }
      let left = UnsafeMutablePointer<Float>.allocate(capacity: 512)
      let right = UnsafeMutablePointer<Float>.allocate(capacity: 512)
      defer { left.deallocate(); right.deallocate() }

      for block in [64, 128, 256, 512] {
        memory.initialize(repeating: 0, count: compiled.totalMemorySlots)
        DGen.injectTensorData(result: compiled, memory: memory)
        var worst = Float(0)
        input[0].withUnsafeBufferPointer { l in
          input[1].withUnsafeBufferPointer { r in
            for start in stride(from: 0, to: count, by: block) {
              let ins: [UnsafePointer<Float>?] = [l.baseAddress! + start, r.baseAddress! + start]
              let outs: [UnsafeMutablePointer<Float>?] = [left, right]
              var context = DGenProcessContextV1(sampleRate: 48000)
              ins.withUnsafeBufferPointer { ins in
                outs.withUnsafeBufferPointer { outs in
                  withUnsafePointer(to: &context) { context in
                    process(ins.baseAddress, outs.baseAddress, UInt32(block), memory,
                            UnsafeRawPointer(context), UnsafeRawPointer(dgen_reference_host_services_v1()))
                  }
                }
              }
              for i in 0..<block where start + i >= 2 * n {
                let frame = start + i
                for (channel, output) in [left, right].enumerated() {
                  XCTAssertTrue(output[i].isFinite)
                  let expected = gains[frame] * input[channel][frame - (n - 1)]
                  worst = max(worst, abs(output[i] - expected))
                }
              }
            }
          }
        }
        XCTAssertLessThan(worst, 2e-5, "backend=\(backend), block=\(block), error=\(worst)")
      }
    }
  }
}

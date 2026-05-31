import XCTest

@testable import DGen

final class RuntimeSampleRateTests: XCTestCase {
  func testPhasorUsesHostSampleRateArgumentInGeneratedC() throws {
    let frameCount = 64
    let graph = Graph(sampleRate: 44_100.0, maxFrameCount: frameCount)
    let frequency = graph.n(.constant(480.0))
    let reset = graph.n(.constant(0.0))
    let phase = graph.n(.phasor(graph.alloc()), frequency, reset)
    _ = graph.n(.output(0), phase)

    let result = try CompilationPipeline.compile(
      graph: graph,
      backend: .c,
      options: .init(frameCount: frameCount, debug: false)
    )

    XCTAssertTrue(result.source.contains("float hostSampleRate"))
    XCTAssertTrue(result.source.contains("hostSampleRate"))

    let runtime = CCompiledKernel(
      source: result.source,
      cellAllocations: result.cellAllocations,
      memorySize: result.totalMemorySlots
    )
    try runtime.compileAndLoad()

    func render(hostSampleRate: Float) throws -> [Float] {
      guard let memory = runtime.allocateNodeMemory() else {
        throw XCTSkip("failed to allocate C runtime memory")
      }
      defer { runtime.deallocateNodeMemory(memory) }

      let input = [Float](repeating: 0.0, count: frameCount)
      var output = [Float](repeating: 0.0, count: frameCount)
      output.withUnsafeMutableBufferPointer { outPtr in
        input.withUnsafeBufferPointer { inPtr in
          runtime.runWithMemory(
            outputs: outPtr.baseAddress!,
            inputs: inPtr.baseAddress!,
            memory: memory,
            frameCount: frameCount,
            hostSampleRate: hostSampleRate
          )
        }
      }
      return output
    }

    let at441 = try render(hostSampleRate: 44_100.0)
    let at480 = try render(hostSampleRate: 48_000.0)

    XCTAssertEqual(at441[1] - at441[0], 480.0 / 44_100.0, accuracy: 0.000_01)
    XCTAssertEqual(at480[1] - at480[0], 480.0 / 48_000.0, accuracy: 0.000_01)
    XCTAssertNotEqual(at441[1] - at441[0], at480[1] - at480[0], accuracy: 0.000_1)
  }
}

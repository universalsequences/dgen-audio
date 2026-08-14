import XCTest

@testable import DGen

final class MetalRendererOptimizationTests: XCTestCase {
  private let frameCount = Lazy.variable(-1, nil)

  private func compile(_ item: ScheduleItem, context: IRContext) -> CompiledKernel {
    MetalRenderer().compile(
      scheduleItems: [item], ctx: context, graph: context.g,
      totalMemorySlots: 1024, name: "optimization_test"
    )[0]
  }

  func testFrameIndependentFinalStoreParallelizesAndKeepsLastFrameWrite() {
    let context = IRContext(g: Graph())
    let frameValue = context.useVariable(src: nil, trackInValues: false)
    let writeResult = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .frameIndex, value: frameValue, scalarType: .int),
      UOp(op: .memoryWrite(7, .constant(0, 0), frameValue), value: writeResult),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let kernel = compile(item, context: context)

    XCTAssertEqual(kernel.dispatchMode, .perFrame)
    XCTAssertEqual(kernel.frameOrder, .parallel)
    XCTAssertFalse(kernel.source.contains("for (uint i = 0; i < frameCount"))
    XCTAssertTrue(kernel.source.contains("frameCount - 1"))
    XCTAssertTrue(kernel.source.contains("if (t"))
    XCTAssertTrue(kernel.source.contains("memory[7 + (int)0.0]"))
  }

  func testReadWriteStatePreventsFrameParallelization() {
    let context = IRContext(g: Graph())
    let prior = context.useVariable(src: nil, trackInValues: false)
    let next = context.useVariable(src: nil, trackInValues: false)
    let writeResult = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .memoryRead(7, .constant(0, 0)), value: prior),
      UOp(op: .add(prior, .constant(0, 1)), value: next),
      UOp(op: .memoryWrite(7, .constant(0, 0), next), value: writeResult),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let kernel = compile(item, context: context)

    XCTAssertEqual(kernel.dispatchMode, .singleThreaded)
    XCTAssertTrue(kernel.source.contains("for (uint i = 0; i < frameCount"))
  }

  func testMultipleFrameLoopsPreventFrameParallelization() {
    let context = IRContext(g: Graph())
    let firstWrite = context.useVariable(src: nil, trackInValues: false)
    let secondWrite = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .memoryWrite(7, .constant(0, 0), .constant(0, 1)), value: firstWrite),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .beginReverseLoop(frameCount), value: .empty),
      UOp(op: .memoryWrite(8, .constant(0, 0), .constant(0, 1)), value: secondWrite),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let kernel = compile(item, context: context)

    XCTAssertEqual(kernel.dispatchMode, .singleThreaded)
    XCTAssertTrue(kernel.source.contains("for (uint i = 0; i < frameCount"))
    XCTAssertTrue(kernel.source.contains("for (int i = frameCount - 1"))
  }

  func testFrameLocalAccumulatorAllowsFrameParallelization() {
    let context = IRContext(g: Graph())
    let accumulator = context.useVariable(src: nil, trackInValues: false)
    let writeResult = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .declareVar(.constant(0, 0)), value: accumulator),
      UOp(op: .mutate(accumulator, .constant(0, 1)), value: .empty),
      UOp(op: .memoryWrite(7, .constant(0, 0), accumulator), value: writeResult),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let kernel = compile(item, context: context)

    XCTAssertEqual(kernel.dispatchMode, .perFrame)
    XCTAssertFalse(kernel.source.contains("for (uint i = 0; i < frameCount"))
  }

  func testGlobalMutationPreventsFrameParallelization() {
    let context = IRContext(g: Graph())
    let state = context.useVariable(src: nil, trackInValues: false)
    context.globals.append(extractVarId(state))
    let writeResult = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .mutate(state, .constant(0, 1)), value: .empty),
      UOp(op: .memoryWrite(7, .constant(0, 0), state), value: writeResult),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let kernel = compile(item, context: context)

    XCTAssertEqual(kernel.dispatchMode, .singleThreaded)
    XCTAssertTrue(kernel.source.contains("for (uint i = 0; i < frameCount"))
  }

  func testSideEffectBeforeFrameLoopPreventsFrameParallelization() {
    let context = IRContext(g: Graph())
    let prefixWrite = context.useVariable(src: nil, trackInValues: false)
    let frameWrite = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .memoryWrite(6, .constant(0, 0), .constant(0, 1)), value: prefixWrite),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .memoryWrite(7, .constant(0, 0), .constant(0, 1)), value: frameWrite),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let kernel = compile(item, context: context)

    XCTAssertEqual(kernel.dispatchMode, .singleThreaded)
    XCTAssertTrue(kernel.source.contains("for (uint i = 0; i < frameCount"))
  }

  func testReadOnlyInnerLoopOmitsDeviceFence() {
    let context = IRContext(g: Graph())
    let accumulator = context.useVariable(src: nil, trackInValues: false)
    let element = context.useVariable(src: nil, trackInValues: false)
    let loopIndex = context.useVariable(src: nil, trackInValues: false)
    let writeResult = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .declareVar(.constant(0, 0)), value: accumulator),
      UOp(op: .beginForLoop(loopIndex, .constant(0, 4)), value: .empty),
      UOp(op: .memoryRead(7, loopIndex), value: element),
      UOp(op: .mutate(accumulator, element), value: .empty),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .memoryWrite(8, .constant(0, 0), accumulator), value: writeResult),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let source = MetalRenderer().render(
      name: "fence_test", scheduleItem: item, ctx: context, graph: context.g,
      totalMemorySlots: 1024)
    let fenceCount = source.components(separatedBy: "atomic_thread_fence").count - 1

    XCTAssertEqual(fenceCount, 1, "only the device-writing outer loop needs a fence")
  }

  func testFixedScalarMemoryUsesLoopLocalCache() {
    let context = IRContext(g: Graph())
    let prior = context.useVariable(src: nil, trackInValues: false)
    let next = context.useVariable(src: nil, trackInValues: false)
    let writeResult = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .memoryRead(7, .constant(0, 0)), value: prior),
      UOp(op: .add(prior, .constant(0, 1)), value: next),
      UOp(op: .memoryWrite(7, .constant(0, 0), next), value: writeResult),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let source = MetalRenderer().render(
      name: "state_cache_test", scheduleItem: item, ctx: context, graph: context.g,
      totalMemorySlots: 1024)

    XCTAssertTrue(source.contains("float m7 = memory[7];"))
    XCTAssertTrue(source.contains("m7 ="))
    XCTAssertTrue(source.contains("memory[7] = m7;"))
  }

  func testInputTapePreloadRejectsMultipleFrameLoops() {
    let context = IRContext(g: Graph())
    let input = context.useVariable(src: nil, trackInValues: false)
    context.globals.append(extractVarId(input))
    let first = context.useVariable(src: nil, trackInValues: false)
    let second = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .add(input, input), value: first),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .beginReverseLoop(frameCount), value: .empty),
      UOp(op: .add(input, input), value: second),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let source = MetalRenderer().render(
      name: "multi_loop_input_test", scheduleItem: item, ctx: context, graph: context.g,
      totalMemorySlots: 1024)

    XCTAssertFalse(source.contains("  float t\(extractVarId(input));\n"))
    XCTAssertEqual(
      source.components(separatedBy: "t[0*frameCount + i]").count - 1, 4,
      "each use must read the current index of its own frame loop")
  }

  func testInputTapePreloadCachesRepeatedUsesInSingleFrameLoop() {
    let context = IRContext(g: Graph())
    let input = context.useVariable(src: nil, trackInValues: false)
    context.globals.append(extractVarId(input))
    let intermediate = context.useVariable(src: nil, trackInValues: false)
    let sum = context.useVariable(src: nil, trackInValues: false)
    let item = ScheduleItem(frameOrder: .sequential, temporality: .frameBased)
    item.dispatchMode = .singleThreaded
    item.ops = [
      UOp(op: .frameCount, value: .empty),
      UOp(op: .beginRange(.constant(0, 0), .constant(0, 1)), value: .empty),
      UOp(op: .beginLoop(frameCount, 1), value: .empty),
      UOp(op: .add(input, .constant(0, 1)), value: intermediate),
      UOp(op: .add(input, intermediate), value: sum),
      UOp(op: .endLoop, value: .empty),
      UOp(op: .endRange, value: .empty),
    ]

    let source = MetalRenderer().render(
      name: "single_loop_input_test", scheduleItem: item, ctx: context, graph: context.g,
      totalMemorySlots: 1024)

    XCTAssertTrue(source.contains("  float t\(extractVarId(input));\n"))
    XCTAssertEqual(source.components(separatedBy: "t[0*frameCount + i]").count - 1, 1)
    XCTAssertTrue(source.contains("t\(extractVarId(input)) + 1.0"))
    XCTAssertTrue(source.contains("t\(extractVarId(input)) + t\(extractVarId(intermediate))"))
  }
}

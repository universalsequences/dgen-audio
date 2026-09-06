import Foundation

/// Merges runs of plain scalar blocks into single sequential loops (C backend).
///
/// Block formation splits the schedule wherever the frame order flips, and
/// every feedback cluster (a one-pole smoother, an envelope stage) is
/// sequential while the arithmetic between clusters is parallel. A patch with
/// many small recurrences therefore renders as `seq / par / seq / par ...`,
/// each boundary costing a 128-float scratch buffer write and read. Measured on
/// an M1 Max a boundary is roughly 0.3 µs per 128-frame block, while the SIMD
/// win of a tiny parallel block between two recurrences is nil.
///
/// A parallel block may always execute inside a sequential loop (sequential is
/// the stricter order), so absorbing it is semantics-preserving. Large parallel
/// blocks keep their own SIMD loop: only blocks with at most
/// `maxAbsorbedParallelNodes` nodes are absorbed. Blocks touching tensors, hop
/// rates, or ops outside the scalar allowlist are never merged.
/// `DGEN_NO_COALESCE=1` disables the pass; `DGEN_COALESCE_MAX_PARALLEL=<n>`
/// overrides the threshold.
enum ScalarBlockCoalescingPass {

  static var isEnabled: Bool {
    ProcessInfo.processInfo.environment["DGEN_NO_COALESCE"] != "1"
  }

  static var maxAbsorbedParallelNodes: Int {
    if let raw = ProcessInfo.processInfo.environment["DGEN_COALESCE_MAX_PARALLEL"],
      let value = Int(raw)
    {
      return value
    }
    return 16
  }

  static func isPlainScalarOp(_ op: LazyOp) -> Bool {
    switch op {
    case .add, .sub, .div, .mul, .abs, .sign, .sin, .cos, .tan, .atan, .tanh, .exp, .log,
      .log10, .sqrt, .atan2, .gt, .gte, .lte, .lt, .eq, .gswitch, .mix, .pow, .floor, .ceil,
      .round, .mod, .min, .max, .and, .or, .xor, .neg, .mse,
      .selector, .modulatedParam, .constant, .hostSampleRate, .param, .input, .output,
      .historyRead, .historyWrite, .historyReadWrite, .phasor, .deterministicPhasor,
      .accum, .noise, .latch, .click, .seq:
      return true
    default:
      return false
    }
  }

  static func isPlain(_ block: Block, graph: Graph, hopBasedNodes: [NodeID: (Int, NodeID)]) -> Bool {
    guard block.tensorIndex == nil, block.shape == nil else { return false }
    return block.nodes.allSatisfy { id in
      guard let node = graph.nodes[id], isPlainScalarOp(node.op),
        graph.nodeToTensor[id] == nil, hopBasedNodes[id] == nil
      else { return false }
      if case .tensor? = node.shape { return false }
      return true
    }
  }

  static func run(
    blocks: [Block], graph: Graph, hopBasedNodes: [NodeID: (Int, NodeID)]
  ) -> [Block] {
    let threshold = maxAbsorbedParallelNodes
    let plain = blocks.map { isPlain($0, graph: graph, hopBasedNodes: hopBasedNodes) }
    var result: [Block] = []
    var run: Block? = nil

    func flush() {
      if let r = run { result.append(r) }
      run = nil
    }

    for (index, block) in blocks.enumerated() {
      guard plain[index] else {
        flush()
        result.append(block)
        continue
      }
      if block.frameOrder == .sequential {
        if run != nil {
          run!.nodes.append(contentsOf: block.nodes)
        } else {
          run = block
        }
        continue
      }
      // Parallel: absorb only when small and adjacent to a sequential run.
      let nextIsSequential =
        index + 1 < blocks.count && plain[index + 1] && blocks[index + 1].frameOrder == .sequential
      let small = block.nodes.count <= threshold
      if small && (run != nil || nextIsSequential) {
        if run != nil {
          run!.nodes.append(contentsOf: block.nodes)
        } else {
          var seeded = block
          seeded.frameOrder = .sequential
          run = seeded
        }
      } else {
        flush()
        result.append(block)
      }
    }
    flush()
    return result
  }
}

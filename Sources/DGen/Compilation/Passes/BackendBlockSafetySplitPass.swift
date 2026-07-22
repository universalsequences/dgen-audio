import Foundation

/// Namespace for backend-specific block safety splits.
enum BackendBlockSafetySplitPass {}

extension BackendBlockSafetySplitPass {
  /// Applies backend-required block splits while preserving execution semantics.
  ///
  /// Metal requires reduction and memory dependency splits to create hard
  /// kernel fences. The C renderer normally owns reduction scheduling itself;
  /// temporal gradient scans are the exception because they iterate over every
  /// frame internally and therefore must run in a static block.
  static func applyIfNeeded(graph: Graph, blocks: [Block], backend: Backend) -> [Block] {
    guard backend == .metal else {
      let hasTemporalGradientScan = graph.nodes.values.contains { node in
        if case .temporalGradScan = node.op { return true }
        return false
      }
      return hasTemporalGradientScan ? splitReduceBlocks(g: graph, blocks: blocks) : blocks
    }

    let afterReduceSplit = splitReduceBlocks(g: graph, blocks: blocks)
    let afterMemorySplit = splitMemoryBlocks(g: graph, blocks: afterReduceSplit)
    return afterMemorySplit
  }
}

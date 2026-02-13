import Foundation

/// Namespace for backend-specific block safety splits.
enum BackendBlockSafetySplitPass {}

extension BackendBlockSafetySplitPass {
  /// Applies backend-required block splits while preserving execution semantics.
  ///
  /// Metal requires additional splits to avoid intra-kernel hazards when reduce or
  /// memory dependency boundaries require a hard kernel fence.
  static func applyIfNeeded(graph: Graph, blocks: [Block], backend: Backend) -> [Block] {
    guard backend == .metal else { return blocks }

    let afterReduceSplit = splitReduceBlocks(g: graph, blocks: blocks)
    let afterMemorySplit = splitMemoryBlocks(g: graph, blocks: afterReduceSplit)
    return afterMemorySplit
  }
}

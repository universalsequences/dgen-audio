import Foundation

/// Compilation pass wrapper for graph shape inference.
///
/// This pass applies core shape rules (`inferShape`) across the sorted graph
/// and writes the inferred shape metadata back onto each node.
enum ShapeInferencePass {}

extension ShapeInferencePass {
  /// Infers and assigns node shapes in topological order.
  static func inferNodeShapes(graph: Graph, sortedNodes: [NodeID]) throws {
    try inferShapes(graph: graph, sortedNodes: sortedNodes)
  }
}

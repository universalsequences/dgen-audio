/// Core block types for the compilation pipeline.
import Foundation

public enum Kind { case simd, scalar }

// corresponds to one kernel (metal backends) or for loop (in C backend)
public struct Block: Equatable {
  public var kind: Kind
  public var nodes: [NodeID] = []
  public var temporality: Temporality = .static_
  public var tensorIndex: Lazy?
  public var shape: Shape?

  public init(kind: Kind) {
    self.kind = kind
  }

  public static func == (lhs: Block, rhs: Block) -> Bool {
    return lhs.kind == rhs.kind && lhs.nodes == rhs.nodes
      && lhs.temporality == rhs.temporality && lhs.tensorIndex == rhs.tensorIndex
      && lhs.shape == rhs.shape
  }
}

extension LazyOp {
  var isOutput: Bool {
    if case .output = self { return true }
    return false
  }
}

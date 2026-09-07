/// Block emission dependency and outbound tensor analysis helpers.
import Foundation

/// Structural index over one frozen block plan.
///
/// Emission asks the same whole-plan questions once per block ("which of my nodes
/// are read elsewhere?", "which tensor cells do later blocks read?"). Answering each
/// by rescanning every block costs O(blocks x nodes) per query, so a plan with a few
/// hundred blocks pays a quadratic price. This index is built once per plan and lets
/// each query cost only what it reports. Semantics match the previous scans exactly,
/// including the first-occurrence-wins rule for nodes that appear in several blocks.
///
/// The index is a snapshot: it is valid only while `blocks` and the graph's node,
/// tensor, and `nodeToTensor` tables are unchanged.
public final class BlockDependencyIndex {
  /// First block index containing each node (first occurrence wins).
  fileprivate let nodeToBlock: [NodeID: Int]
  /// Every block index containing each node, ascending.
  fileprivate let nodeOccurrences: [NodeID: [Int]]
  /// For each node, the nodes that list it in `allDependencies` and appear in some block.
  fileprivate let consumers: [NodeID: [NodeID]]
  /// Highest block index that reads each tensor cell.
  fileprivate let maxCellReaderBlock: [CellID: Int]

  /// Builds the index for `blocks` against `graph`.
  ///
  /// - Parameters:
  ///   - blocks: Ordered blocks in emission order.
  ///   - graph: Graph containing node, tensor, and dependency metadata.
  public init(blocks: [Block], graph g: Graph) {
    var nodeToBlock = [NodeID: Int]()
    var nodeOccurrences = [NodeID: [Int]]()
    var consumers = [NodeID: Set<NodeID>]()
    var maxCellReaderBlock = [CellID: Int]()

    for (blockIndex, block) in blocks.enumerated() {
      for nodeId in block.nodes {
        if nodeToBlock[nodeId] == nil { nodeToBlock[nodeId] = blockIndex }
        if nodeOccurrences[nodeId]?.last != blockIndex {
          nodeOccurrences[nodeId, default: []].append(blockIndex)
        }
        guard let node = g.nodes[nodeId] else { continue }
        for dep in node.allDependencies {
          consumers[dep, default: []].insert(nodeId)
        }
        // Tensor cells this node reads: any tensor-shaped input, plus the history
        // buffer named by a `historyRead`. `historyWrite` reads through its inputs,
        // which the input scan above already covers.
        for inputId in node.inputs {
          guard let inputNode = g.nodes[inputId], case .tensor = inputNode.shape,
            let tensorId = g.nodeToTensor[inputId], let tensor = g.tensors[tensorId]
          else { continue }
          if (maxCellReaderBlock[tensor.cellId] ?? -1) < blockIndex {
            maxCellReaderBlock[tensor.cellId] = blockIndex
          }
        }
        if case .historyRead(let cellId) = node.op,
          (maxCellReaderBlock[cellId] ?? -1) < blockIndex
        {
          maxCellReaderBlock[cellId] = blockIndex
        }
      }
    }

    self.nodeToBlock = nodeToBlock
    self.nodeOccurrences = nodeOccurrences
    self.consumers = consumers.mapValues { $0.sorted() }
    self.maxCellReaderBlock = maxCellReaderBlock
  }

  /// Whether `node` appears in any block other than `blockIndex`.
  fileprivate func appearsOutside(_ node: NodeID, blockIndex: Int) -> Bool {
    guard let occurrences = nodeOccurrences[node] else { return false }
    return occurrences.contains { $0 != blockIndex }
  }
}

/// Returns the caller's index, or builds a throwaway one for callers that have none.
///
/// The caller is responsible for passing an index built from the same `blocks`.
private func resolveIndex(_ index: BlockDependencyIndex?, _ blocks: [Block], _ g: Graph)
  -> BlockDependencyIndex
{
  index ?? BlockDependencyIndex(blocks: blocks, graph: g)
}

/// Finds nodes produced in `block` whose values are consumed by other blocks.
///
/// Special case: when a dependency is a `.seq` node, this helper exports its final input
/// because that is the value consumed downstream.
///
/// - Parameters:
///   - blks: All blocks in emission order.
///   - g: Graph containing dependency metadata.
///   - block: Producer block being analyzed.
///   - index: Prebuilt plan index; built on demand when `nil`.
/// - Returns: Sorted node IDs that must be materialized for downstream blocks.
func findNodesWithOutboundDependencies(
  _ blks: [Block], _ g: Graph, block: Block, index: BlockDependencyIndex? = nil
) -> [NodeID] {
  let idx = resolveIndex(index, blks, g)
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  var needed: Set<NodeID> = []
  // Walk this block's own products and consult the reverse-dependency map, instead of
  // scanning every other block's nodes looking for edges that point back here.
  for producer in blks[thisIdx].nodes where idx.nodeToBlock[producer] == thisIdx {
    guard let consumers = idx.consumers[producer] else { continue }
    guard consumers.contains(where: { idx.appearsOutside($0, blockIndex: thisIdx) }) else {
      continue
    }
    if let producerNode = g.nodes[producer], case .seq = producerNode.op {
      if let lastInput = producerNode.inputs.last { needed.insert(lastInput) }
      // Preserve seq temporal deps (e.g., buffer write-position dependencies)
      // so cross-kernel global wiring can carry them across block boundaries.
      for temporalDep in producerNode.temporalDependencies
      where idx.nodeToBlock[temporalDep] == thisIdx {
        needed.insert(temporalDep)
      }
    } else {
      needed.insert(producer)
    }
  }
  return needed.sorted()  // Stable ordering
}

/// Finds external dependencies that nodes in `block` read from earlier/different blocks.
///
/// - Parameters:
///   - blks: All blocks in emission order.
///   - g: Graph containing dependency metadata.
///   - block: Consumer block being analyzed.
///   - index: Prebuilt plan index; built on demand when `nil`.
/// - Returns: Sorted node IDs that must be loaded before this block runs.
func findNodesAsInboundDependencies(
  _ blks: [Block], _ g: Graph, block: Block, index: BlockDependencyIndex? = nil
) -> [NodeID] {
  let nodeBlock = resolveIndex(index, blks, g).nodeToBlock
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  var needed: Set<NodeID> = []
  for nodeId in block.nodes {
    guard let node = g.nodes[nodeId] else { continue }
    for dep in node.allDependencies {
      guard let producerIdx = nodeBlock[dep], producerIdx != thisIdx else { continue }
      if let depNode = g.nodes[dep], case .seq = depNode.op {
        if let lastInput = depNode.inputs.last,
          let lastInputProducer = nodeBlock[lastInput],
          lastInputProducer != thisIdx
        {
          needed.insert(lastInput)
        }
        for temporalDep in depNode.temporalDependencies {
          if let temporalProducer = nodeBlock[temporalDep], temporalProducer != thisIdx {
            needed.insert(temporalDep)
          }
        }
      } else {
        needed.insert(dep)
      }
    }
  }
  return needed.sorted()  // Stable ordering
}

/// Compute which tensor cells in this block need to be written to memory because
/// they're used by later blocks. Cells only used within this block stay in registers.
///
/// - Parameters:
///   - blks: All blocks in emission order.
///   - g: Graph with node/tensor mappings.
///   - block: Block whose produced tensor cells are analyzed.
///   - index: Prebuilt plan index; built on demand when `nil`.
/// - Returns: Set of tensor cell IDs that must be flushed to memory.
func findOutboundTensorCells(
  _ blks: [Block], _ g: Graph, block: Block, index: BlockDependencyIndex? = nil
) -> Set<CellID> {
  let idx = resolveIndex(index, blks, g)
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  // Collect all tensor cells produced by nodes in this block
  var producedCells: Set<CellID> = []
  for nodeId in block.nodes {
    if let node = g.nodes[nodeId], case .tensor = node.shape {
      if let tensorId = g.nodeToTensor[nodeId], let tensor = g.tensors[tensorId] {
        producedCells.insert(tensor.cellId)
      }
    }
  }

  // A cell escapes this block exactly when some later block reads it, so the
  // plan-wide last reader answers the question without rescanning later blocks.
  return producedCells.filter { (idx.maxCellReaderBlock[$0] ?? -1) > thisIdx }
}

/// Find tensor cells that cross shape region boundaries within a scalar block.
/// These must be written to memory (not kept in registers) because they're computed
/// in one loop and consumed in a different loop.
///
/// - Parameters:
///   - block: Block containing shape transitions.
///   - g: Graph used to resolve tensor-producing inputs.
///   - transitions: Region boundaries as `(nodeIndex, shape)` entries.
/// - Returns: Tensor cell IDs that cross region boundaries and require memory materialization.
func findCrossRegionOutboundCells(
  block: Block, g: Graph, transitions: [(nodeIndex: Int, shape: [Int])]
) -> Set<CellID> {
  guard !transitions.isEmpty else { return [] }

  var outbound: Set<CellID> = []

  // Build map: nodeId -> regionIndex
  var nodeToRegion: [NodeID: Int] = [:]
  for (regionIdx, transition) in transitions.enumerated() {
    let regionEnd =
      regionIdx + 1 < transitions.count
      ? transitions[regionIdx + 1].nodeIndex
      : block.nodes.count
    for nodeIndex in transition.nodeIndex..<regionEnd {
      nodeToRegion[block.nodes[nodeIndex]] = regionIdx
    }
  }

  // For each node, check if any of its inputs come from a different region
  for nodeId in block.nodes {
    guard let node = g.nodes[nodeId] else { continue }
    let myRegion = nodeToRegion[nodeId]  // May be nil for scalar nodes

    for inputId in node.inputs {
      let inputRegion = nodeToRegion[inputId]

      // Case 1: Both have regions and they differ (cross-region)
      // Case 2: Node is scalar (no region) but input has a region (tensor -> scalar)
      let crossesRegion =
        (myRegion != nil && inputRegion != nil && myRegion != inputRegion)
        || (myRegion == nil && inputRegion != nil)

      guard crossesRegion else { continue }

      // This input crosses a region boundary - its cell must be outbound
      if let tensorId = g.nodeToTensor[inputId],
        let tensor = g.tensors[tensorId]
      {
        outbound.insert(tensor.cellId)
      }
    }
  }

  return outbound
}

/// Block emission dependency and outbound tensor analysis helpers.
import Foundation

/// Builds a stable mapping from node ID to owning block index.
///
/// If a node appears multiple times, the first block occurrence wins.
///
/// - Parameter blocks: Ordered blocks in emission order.
/// - Returns: Dictionary mapping each discovered node to its block index.
private func buildNodeToBlockIndex(_ blocks: [Block]) -> [NodeID: Int] {
  var nodeBlock = [NodeID: Int]()
  for (blockIndex, block) in blocks.enumerated() {
    for nodeId in block.nodes where nodeBlock[nodeId] == nil {
      nodeBlock[nodeId] = blockIndex
    }
  }
  return nodeBlock
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
/// - Returns: Sorted node IDs that must be materialized for downstream blocks.
func findNodesWithOutboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> [NodeID] {
  let nodeBlock = buildNodeToBlockIndex(blks)
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  var needed: Set<NodeID> = []
  for (consumerIdx, consumerBlock) in blks.enumerated() {
    if consumerIdx == thisIdx { continue }
    for nodeId in consumerBlock.nodes {
      guard let node = g.nodes[nodeId] else { continue }
      for dep in node.allDependencies {
        guard let producerIdx = nodeBlock[dep], producerIdx == thisIdx else { continue }
        if let depNode = g.nodes[dep], case .seq = depNode.op {
          if let lastInput = depNode.inputs.last { needed.insert(lastInput) }
        } else {
          needed.insert(dep)
        }
      }
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
/// - Returns: Sorted node IDs that must be loaded before this block runs.
func findNodesAsInboundDependencies(_ blks: [Block], _ g: Graph, block: Block) -> [NodeID] {
  let nodeBlock = buildNodeToBlockIndex(blks)
  guard let thisIdx = blks.firstIndex(of: block) else { return [] }

  var needed: Set<NodeID> = []
  for nodeId in block.nodes {
    guard let node = g.nodes[nodeId] else { continue }
    for dep in node.allDependencies {
      if let producerIdx = nodeBlock[dep], producerIdx != thisIdx {
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
/// - Returns: Set of tensor cell IDs that must be flushed to memory.
func findOutboundTensorCells(_ blks: [Block], _ g: Graph, block: Block) -> Set<CellID> {
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

  // Check which cells are consumed by later blocks
  var outboundCells: Set<CellID> = []
  for (blockIdx, otherBlock) in blks.enumerated() {
    if blockIdx <= thisIdx { continue }  // Only look at later blocks

    for nodeId in otherBlock.nodes {
      guard let node = g.nodes[nodeId] else { continue }

      // Check if this node reads from any of our produced cells
      for inputId in node.inputs {
        if let inputNode = g.nodes[inputId], case .tensor = inputNode.shape {
          if let tensorId = g.nodeToTensor[inputId], let tensor = g.tensors[tensorId] {
            if producedCells.contains(tensor.cellId) {
              outboundCells.insert(tensor.cellId)
            }
          }
        }
      }

      // Also check historyRead/historyWrite operations that reference tensor cells
      switch node.op {
      case .historyRead(let cellId):
        // historyRead's cellId is the history buffer - mark if produced here
        if producedCells.contains(cellId) {
          outboundCells.insert(cellId)
        }
      case .historyWrite(_):
        // historyWrite reads from its INPUT tensor, not the history cell parameter
        // If the input tensor's cell was produced in this block, mark it as outbound
        for inputId in node.inputs {
          if let inputNode = g.nodes[inputId], case .tensor = inputNode.shape {
            if let tensorId = g.nodeToTensor[inputId], let tensor = g.tensors[tensorId] {
              if producedCells.contains(tensor.cellId) {
                outboundCells.insert(tensor.cellId)
              }
            }
          }
        }
      default:
        break
      }
    }
  }

  return outboundCells
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

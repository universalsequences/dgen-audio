/// Memory layout remapping and cell allocations.
import Foundation

public struct CellAllocations {
  public let cellKinds: [CellID: Kind]
  public let cellMappings: [CellID: CellID]
  public let totalMemorySlots: Int

  public init(totalMemorySlots: Int, cellMappings: [CellID: CellID], cellKinds: [CellID: Kind]) {
    self.cellKinds = cellKinds
    self.cellMappings = cellMappings
    self.totalMemorySlots = totalMemorySlots
  }
}

/// Remap memory slots to avoid conflicts between scalar and vector operations
/// Returns the total number of memory slots needed after remapping
func remapVectorMemorySlots(
  _ uopBlocks: inout [BlockUOps], cellSizes: [CellID: Int], voiceCellId: CellID?,
  graph: Graph? = nil,
  enableBufferReuse: Bool = false
)
  -> CellAllocations
{
  // Collect all memory operations and their execution modes
  var memoryUsage: [CellID: Kind] = [:]
  var allCellIds: Set<CellID> = []
  var cellUsedInMultipleModes: Set<CellID> = []

  // Helper to register a cell's usage and detect multi-mode access
  func registerCell(_ cellId: CellID, kind: Kind) {
    allCellIds.insert(cellId)

    if let existingKind = memoryUsage[cellId] {
      if existingKind != kind {
        // Cell used in multiple execution modes
        cellUsedInMultipleModes.insert(cellId)
        // Upgrade to SIMD if any block uses it as SIMD
        // (SIMD needs 4x space, scalar can still access first element)
        if kind == .simd || existingKind == .simd {
          memoryUsage[cellId] = .simd
        }
      }
    } else {
      memoryUsage[cellId] = kind
    }
  }

  // First pass: identify which memory cells are used in which execution modes
  for block in uopBlocks {
    for uop in block.ops {
      if let cellId = uop.op.memoryCellId {
        registerCell(cellId, kind: block.kind)
      }
    }
  }

  if let voiceCellId = voiceCellId {
    registerCell(voiceCellId, kind: .simd)
  }

  // Collect cells with injectable initial data (e.g. twiddle factors).
  // These get placed at low physical offsets so their indices stay within
  // Float32 exact integer range when passed through initial_state buffers.
  var injectableCellIds: Set<CellID> = []
  if let graph = graph {
    for (_, tensor) in graph.tensors {
      if tensor.data != nil {
        injectableCellIds.insert(tensor.cellId)
      }
    }
  }

  // Liveness analysis: compute first/last use of each cell across all blocks.
  // Cells with non-overlapping lifetimes can share the same physical memory.
  // Only performed when enableBufferReuse is true.
  struct LivenessInterval {
    let cellId: CellID
    let firstUse: Int
    let lastUse: Int
    let size: Int
    let kind: Kind
    let canReceiveReuse: Bool  // false for memoryAccumulate cells (need zeroed memory)
  }

  var reuseEligible: Set<CellID> = []
  var intervals: [LivenessInterval] = []

  if enableBufferReuse {
    // Persistent cells: survive across frame iterations, must not be reused.
    // Includes load/store/delay1 cells (detected from UOps) AND cells explicitly
    // marked as persistent by graph construction (circular buffers, ring buffers).
    var persistentCells: Set<CellID> = graph?.persistentCells ?? []
    var accumulateCells: Set<CellID> = []  // memoryAccumulate — need zeroed memory
    var cellFirstUse: [CellID: Int] = [:]
    var cellLastUse: [CellID: Int] = [:]

    // Compute liveness at BLOCK granularity, not UOp granularity.
    // A cell used anywhere in a block is live for the entire block's execution,
    // since a kernel reads/writes all its cells concurrently within the loop.
    // Using the block index ensures two cells in the same kernel never share memory.
    for (blockIndex, block) in uopBlocks.enumerated() {
      for uop in block.ops {
        if let cellId = uop.op.memoryCellId {
          switch uop.op {
          case .load, .store, .delay1:
            persistentCells.insert(cellId)
          case .memoryAccumulate:
            accumulateCells.insert(cellId)
          default:
            break
          }
          if cellFirstUse[cellId] == nil {
            cellFirstUse[cellId] = blockIndex
          }
          cellLastUse[cellId] = blockIndex
        }
      }
    }

    // Determine which cells are eligible for buffer reuse.
    // Excluded: injectable (precomputed data), persistent (scalar feedback),
    // voice cell, and dual-use cells.
    // Frame-aware cells ARE eligible — their frame-indexed addressing is an
    // internal layout detail; inter-block liveness is the same as any other cell.
    // Accumulate cells can DONATE memory but cannot RECEIVE reused memory
    // (memoryAccumulate does += which expects zero; stale data corrupts results).
    for cellId in memoryUsage.keys {
      if injectableCellIds.contains(cellId) { continue }
      if persistentCells.contains(cellId) { continue }
      if cellId == voiceCellId { continue }
      reuseEligible.insert(cellId)
    }

    // Build sorted liveness intervals for eligible cells
    for cellId in reuseEligible.sorted() {
      guard let first = cellFirstUse[cellId], let last = cellLastUse[cellId] else { continue }
      let size = cellSizes[cellId] ?? 1
      let kind = memoryUsage[cellId] ?? .scalar
      let canReceiveReuse = !accumulateCells.contains(cellId)
      intervals.append(LivenessInterval(
        cellId: cellId, firstUse: first, lastUse: last, size: size, kind: kind,
        canReceiveReuse: canReceiveReuse))
    }
    intervals.sort { $0.firstUse < $1.firstUse }
  }

  // Second pass: create remapping.
  // Strategy: injectable cells → physical offset 0+,
  // eligible cells → greedy interval-based reuse,
  // remaining cells → packed sequentially.
  var cellRemapping: [CellID: CellID] = [:]
  // When buffer reuse is enabled, pack everything contiguously (no identity mapping)
  // to avoid huge gaps from high cell IDs. Without reuse, use identity mapping for
  // single-element scalars to minimize remapping overhead.
  var nextAvailableSlot = enableBufferReuse ? 0 : (allCellIds.max() ?? -1) + 1

  // Place injectable cells at low offsets starting from 0
  var nextInjectableSlot = 0
  var reservedRange = 0  // physical slots [0, reservedRange) are taken by injectables
  for cellId in injectableCellIds.sorted() {
    let allocSize = cellSizes[cellId] ?? 1
    cellRemapping[cellId] = nextInjectableSlot
    nextInjectableSlot += allocSize
  }
  reservedRange = nextInjectableSlot
  if enableBufferReuse {
    nextAvailableSlot = max(nextAvailableSlot, reservedRange)
  }

  // Greedy interval-based allocation for reuse-eligible cells.
  // freeRegions tracks allocated regions that may become available for reuse.
  var freeRegions: [(offset: Int, size: Int, availableAfter: Int)] = []
  var reusedCount = 0

  for interval in intervals {
    let needsAlignment = interval.kind == .simd
    let allocSize = needsAlignment ? max(4, interval.size) : interval.size

    // Find best-fit free region: must be available and large enough, prefer least waste.
    // Accumulate cells (memoryAccumulate) skip reuse — they do += and need zeroed memory.
    // They still DONATE their region to later cells after their last use.
    var bestIdx: Int? = nil
    var bestWaste = Int.max
    var bestOffset = 0

    if interval.canReceiveReuse {
      for (idx, region) in freeRegions.enumerated() {
        guard region.availableAfter < interval.firstUse else { continue }
        let effectiveOffset = needsAlignment ? ((region.offset + 3) / 4) * 4 : region.offset
        let usableSize = region.size - (effectiveOffset - region.offset)
        guard usableSize >= interval.size else { continue }
        let waste = usableSize - interval.size
        if waste < bestWaste {
          bestWaste = waste
          bestIdx = idx
          bestOffset = effectiveOffset
        }
      }
    }

    if let idx = bestIdx {
      cellRemapping[interval.cellId] = bestOffset
      freeRegions[idx].availableAfter = interval.lastUse
      reusedCount += 1
    } else {
      // No reusable region — allocate fresh
      let offset: Int
      if needsAlignment {
        offset = ((nextAvailableSlot + 3) / 4) * 4
        nextAvailableSlot = offset + allocSize
      } else {
        offset = nextAvailableSlot
        nextAvailableSlot += allocSize
      }
      cellRemapping[interval.cellId] = offset
      freeRegions.append((offset: offset, size: allocSize, availableAfter: interval.lastUse))
    }
  }

  // Remap non-eligible, non-injectable cells with original logic
  for (cellId, kind) in memoryUsage.sorted(by: { $0.key < $1.key }) {
    if injectableCellIds.contains(cellId) || reuseEligible.contains(cellId) { continue }
    let allocSize = cellSizes[cellId] ?? 1

    if kind == .simd {
      let alignedSlot = ((nextAvailableSlot + 3) / 4) * 4
      cellRemapping[cellId] = alignedSlot
      nextAvailableSlot = alignedSlot + max(4, allocSize)
    } else if allocSize > 1 {
      cellRemapping[cellId] = nextAvailableSlot
      nextAvailableSlot += allocSize
    } else if enableBufferReuse || cellId < reservedRange {
      // Pack contiguously when buffer reuse is on (no identity mapping gaps)
      cellRemapping[cellId] = nextAvailableSlot
      nextAvailableSlot += 1
    } else {
      cellRemapping[cellId] = cellId
    }
  }

  // Third pass: apply the remapping to all UOps
  for blockIndex in 0..<uopBlocks.count {
    for uopIndex in 0..<uopBlocks[blockIndex].ops.count {
      let uop = uopBlocks[blockIndex].ops[uopIndex]
      if let remappedOp = uop.op.withRemappedCellId(cellRemapping) {
        uopBlocks[blockIndex].ops[uopIndex] = UOp(
          op: remappedOp,
          value: uop.value,
          kind: uop.kind
        )
      }
    }
  }

  return CellAllocations(
    totalMemorySlots: nextAvailableSlot,
    cellMappings: cellRemapping,
    cellKinds: memoryUsage
  )
}

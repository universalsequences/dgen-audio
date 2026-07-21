import Foundation

public struct HopDomain: Equatable {
  public let hopSize: Int

  public init(hopSize: Int) {
    self.hopSize = hopSize
  }
}

public struct HopIsland: Equatable {
  public let domain: HopDomain
  public let blockIndices: [Int]

  public init(domain: HopDomain, blockIndices: [Int]) {
    self.domain = domain
    self.blockIndices = blockIndices
  }
}

public enum ScheduledRegion: Equatable {
  case block(Int)
  case hopIsland(HopIsland)
}

enum HopIslandPass {}

extension HopIslandPass {
  static func buildRegions(for blocks: [BlockUOps]) -> [ScheduledRegion] {
    var regions: [ScheduledRegion] = []
    var pendingDomain: HopDomain? = nil
    var pendingIndices: [Int] = []
    var deferredRegions: [ScheduledRegion] = []
    var pendingAccess = BlockAccess()

    func flushPending() {
      guard !pendingIndices.isEmpty else { return }
      if pendingIndices.count > 1, let domain = pendingDomain {
        // Independent frame work can run before the shared hop loop. Dependent
        // scalar frame work is kept in pendingIndices and runs inside the island.
        regions.append(contentsOf: deferredRegions)
        regions.append(.hopIsland(HopIsland(domain: domain, blockIndices: pendingIndices)))
      } else {
        for index in pendingIndices {
          regions.append(.block(index))
        }
        regions.append(contentsOf: deferredRegions)
      }
      pendingDomain = nil
      pendingIndices.removeAll(keepingCapacity: true)
      deferredRegions.removeAll(keepingCapacity: true)
      pendingAccess = BlockAccess()
    }

    for (index, block) in blocks.enumerated() {
      guard let domain = hopDomain(for: block), isIslandEligible(block) else {
        if hopDomain(for: block) != nil || block.temporality == .static_ {
          flushPending()
          regions.append(.block(index))
          continue
        }

        let access = BlockAccess(block.ops)
        if access.conflicts(with: pendingAccess) {
          if isFrameCarrierEligible(block) {
            // Example: hop gate -> latch. The latch is frame-rate, but it must
            // stay in the same outer frame loop as the hop producer it reads.
            pendingIndices.append(index)
            pendingAccess.formUnion(access)
            continue
          } else {
            flushPending()
          }
        }
        if pendingIndices.isEmpty {
          regions.append(.block(index))
        } else {
          deferredRegions.append(.block(index))
        }
        continue
      }

      if let pendingDomain, pendingDomain != domain {
        flushPending()
      }

      pendingDomain = domain
      pendingIndices.append(index)
      pendingAccess.formUnion(BlockAccess(block.ops))
    }

    flushPending()
    return regions
  }

  private static func hopDomain(for block: BlockUOps) -> HopDomain? {
    guard case .hopBased(let hopSize, _) = block.temporality else {
      return nil
    }
    return HopDomain(hopSize: hopSize)
  }

  private static func isIslandEligible(_ block: BlockUOps) -> Bool {
    guard block.vectorWidth == 1 else {
      return false
    }

    switch block.dispatchMode {
    case .singleThreaded, .perFrame, .selfManaged:
      return true
    case .fixedWithFrameLoop:
      return !usesFlatThreading(block.ops)
    case .perFrameScaled, .perFrameThreadgroup1, .perFrameScaledThreadgroup1,
         .staticThreads, .selfManagedThreads, .gemm, .gemmStaged:
      return false
    }
  }

  private static func isFrameCarrierEligible(_ block: BlockUOps) -> Bool {
    guard block.temporality == .frameBased else {
      return false
    }
    return isIslandEligible(block)
  }

  private static func usesFlatThreading(_ ops: [UOp]) -> Bool {
    for op in ops {
      switch op.op {
      case .threadIndex, .setFrameIndex:
        return true
      default:
        continue
      }
    }
    return false
  }
}

private struct BlockAccess {
  var readsVariables: Set<VarID> = []
  var writesVariables: Set<VarID> = []
  var readsCells: Set<CellID> = []
  var writesCells: Set<CellID> = []

  init() {}

  init(_ ops: [UOp]) {
    for uop in ops {
      record(uop)
    }
  }

  mutating func formUnion(_ other: BlockAccess) {
    readsVariables.formUnion(other.readsVariables)
    writesVariables.formUnion(other.writesVariables)
    readsCells.formUnion(other.readsCells)
    writesCells.formUnion(other.writesCells)
  }

  func conflicts(with earlier: BlockAccess) -> Bool {
    !earlier.writesVariables.isDisjoint(with: readsVariables)
      || !earlier.readsVariables.isDisjoint(with: writesVariables)
      || !earlier.writesCells.isDisjoint(with: readsCells)
      || !earlier.readsCells.isDisjoint(with: writesCells)
  }

  private mutating func record(_ uop: UOp) {
    recordWrite(uop.value)

    switch uop.op {
    case .load(let cell):
      readsCells.insert(cell)
    case .store(let cell, let value):
      writesCells.insert(cell)
      recordRead(value)
    case .delay1(let cell, let value):
      readsCells.insert(cell)
      writesCells.insert(cell)
      recordRead(value)
    case .memoryRead(let cell, let offset), .simdBroadcastLoad(let cell, let offset):
      readsCells.insert(cell)
      recordRead(offset)
    case .memoryWrite(let cell, let offset, let value):
      writesCells.insert(cell)
      recordRead(offset)
      recordRead(value)
    case .memoryAccumulate(let cell, let offset, let value):
      readsCells.insert(cell)
      writesCells.insert(cell)
      recordRead(offset)
      recordRead(value)
    case .acceleratedFFTCall(_, let reCell, let imCell, _):
      readsCells.insert(reCell)
      readsCells.insert(imCell)
      writesCells.insert(reCell)
      writesCells.insert(imCell)
    case .partitionedSpectralMACCall(_, _, let partitionIdxCell, let ringReCell, let ringImCell,
                                     let irReCell, let irImCell, let reOutCell, let imOutCell):
      readsCells.formUnion([partitionIdxCell, ringReCell, ringImCell, irReCell, irImCell])
      writesCells.formUnion([reOutCell, imOutCell])
    case .simdgroupLoad(let cell, let offset, _, _):
      readsCells.insert(cell)
      recordRead(offset)
    case .simdgroupStore(let source, let cell, let offset, _):
      writesCells.insert(cell)
      recordRead(source)
      recordRead(offset)
    case .noise(let cell):
      readsCells.insert(cell)
      writesCells.insert(cell)
    default:
      for lazy in lazyInputs(of: uop.op) {
        recordRead(lazy)
      }
    }
  }

  private mutating func recordRead(_ lazy: Lazy) {
    switch lazy {
    case .variable(let id, _), .global(let id):
      readsVariables.insert(id)
    default:
      break
    }
  }

  private mutating func recordWrite(_ lazy: Lazy) {
    switch lazy {
    case .variable(let id, _), .global(let id):
      writesVariables.insert(id)
    default:
      break
    }
  }

  private func lazyInputs(of op: Op) -> [Lazy] {
    switch op {
    case .mse(let a, let b), .mutate(let a, let b), .add(let a, let b), .sub(let a, let b),
         .mul(let a, let b), .div(let a, let b), .and(let a, let b), .or(let a, let b),
         .xor(let a, let b), .pow(let a, let b), .atan2(let a, let b), .mod(let a, let b),
         .gt(let a, let b), .gte(let a, let b), .lte(let a, let b), .lt(let a, let b),
         .eq(let a, let b), .min(let a, let b), .max(let a, let b), .latch(let a, let b),
         .beginForLoop(let a, let b), .beginRange(let a, let b), .loadTape(let a, let b),
         .threadgroupWrite(_, let a, let b):
      return [a, b]
    case .abs(let a), .sign(let a), .sin(let a), .cos(let a), .tan(let a), .atan(let a),
         .tanh(let a), .exp(let a), .log(let a), .log10(let a), .sqrt(let a), .floor(let a),
         .ceil(let a), .round(let a), .beginIf(let a), .beginLoop(let a, _),
         .beginReverseLoop(let a), .output(_, let a), .cast(let a, _), .identity(let a),
         .declareVar(let a), .setFrameIndex(let a), .beginHopCheck(let a),
         .threadgroupRead(_, let a):
      return [a]
    case .gswitch(let a, let b, let c), .simdgroupMultiplyAccumulate(let a, let b, let c):
      return [a, b, c]
    case .selector(let mode, let options):
      return [mode] + options
    case .simdgroupLoadScratch(_, let offset, _, _):
      return [offset]
    default:
      return []
    }
  }
}

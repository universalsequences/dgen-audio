import Foundation

/// C-backend post-emission SIMD upgrade for eligible scalar element loops.
///
/// This pass runs after `emitBlockUOps` when the block was emitted in scalar form but
/// still contains loop bodies that can safely execute with SIMD width 4.
public func upgradeElementLoopsToSIMD(_ uops: inout [UOp]) {
  var i = 0
  while i < uops.count {
    // Match both loop types and extract element count + loop variable.
    let elementCount: Int
    let loopVar: Lazy
    let isParallelRange: Bool

    switch uops[i].op {
    case .beginParallelRange(let count, _):
      elementCount = count
      loopVar = uops[i].value
      isParallelRange = true

    case .beginForLoop(let lv, let countLazy):
      guard case .constant(_, let countFloat) = countLazy else {
        i += 1
        continue
      }
      elementCount = Int(countFloat)
      loopVar = lv
      isParallelRange = false

    default:
      i += 1
      continue
    }

    // SIMD lane width is fixed to 4 right now.
    guard elementCount >= 4 && elementCount % 4 == 0 else {
      i += 1
      continue
    }

    // Find matching end while honoring nested control flow.
    let beginIdx = i
    var depth = 1
    var endIdx: Int? = nil
    for j in (beginIdx + 1)..<uops.count {
      switch uops[j].op {
      case .beginForLoop, .beginLoop, .beginReverseLoop, .beginParallelRange:
        depth += 1
      case .endLoop, .endParallelRange:
        depth -= 1
        if depth == 0 {
          endIdx = j
        }
      default:
        break
      }
      if endIdx != nil { break }
    }

    guard let endIdx else {
      i += 1
      continue
    }

    // Capture loop variable id for memory-offset legality checks.
    let loopVarId: VarID?
    if case .variable(let vid, _) = loopVar {
      loopVarId = vid
    } else {
      loopVarId = nil
    }

    // Bail when body contains stateful/control/aliasing blockers.
    var hasBlocker = false
    for k in (beginIdx + 1)..<endIdx {
      switch uops[k].op {
      case .load, .store, .delay1, .memoryAccumulate:
        hasBlocker = true
      case .noise, .latch:
        hasBlocker = true
      case .beginForLoop, .endLoop, .beginParallelRange, .endParallelRange,
        .beginIf, .endIf, .gswitch:
        hasBlocker = true
      case .broadcastAccess:
        hasBlocker = true
      case .beginHopCheck, .endHopCheck:
        hasBlocker = true
      case .mutate:
        hasBlocker = true
      case .add, .sub, .mul, .div:
        if uops[k].scalarType == .int {
          hasBlocker = true
        }
      case .memoryRead(_, let offset), .memoryWrite(_, let offset, _):
        if case .variable(let vid, _) = offset, vid != loopVarId {
          hasBlocker = true
        }
      default:
        break
      }
      if hasBlocker { break }
    }

    guard !hasBlocker else {
      i = endIdx + 1
      continue
    }

    // Upgrade loop and enclosed ops to SIMD kind.
    uops[beginIdx] = UOp(
      op: .beginParallelRange(elementCount, 4),
      value: loopVar,
      kind: .simd
    )

    for k in (beginIdx + 1)..<endIdx {
      uops[k].kind = .simd
    }

    if isParallelRange {
      uops[endIdx].kind = .simd
    } else {
      uops[endIdx] = UOp(op: .endParallelRange, value: uops[endIdx].value, kind: .simd)
    }

    i = endIdx + 1
  }
}

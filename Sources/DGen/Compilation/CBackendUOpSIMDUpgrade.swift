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

    switch uops[i].op {
    case .beginParallelRange(let count, _):
      elementCount = count
      loopVar = uops[i].value

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
          // Int arithmetic in SIMD bodies is rendered as scalar int (one value
          // per iteration, lane-uniform) — safe as long as it's only used to
          // build memoryRead/Write offsets. Allow `loopVar + const` style adds
          // since they're the common pattern for unrolled tap addresses.
          if case .add(let a, let b) = uops[k].op,
            (isLoopVar(a, loopVarId: loopVarId!) && isConstantLazy(b))
              || (isLoopVar(b, loopVarId: loopVarId!) && isConstantLazy(a))
          {
            // allowed
          } else {
            hasBlocker = true
          }
        }
      case .memoryRead(_, let offset), .memoryWrite(_, let offset, _):
        if case .variable(let vid, _) = offset, vid != loopVarId {
          // Allow affine offsets of the form `loopVar + const` so lane j
          // reads at base + C + j — a contiguous vld1q_f32.
          if let lvId = loopVarId,
            isAffineOfLoopVar(offset, loopVarId: lvId, in: uops, upTo: k)
          {
            // affine; not a blocker
          } else {
            hasBlocker = true
          }
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
      vectorWidth: 4
    )

    for k in (beginIdx + 1)..<endIdx {
      uops[k].vectorWidth = 4
    }

    uops[endIdx].vectorWidth = 4

    i = endIdx + 1
  }
}

/// Returns true when `offset` is either the loop variable, a constant, or the sum
/// of the loop variable and a constant. Used by the SIMD upgrade blocker check to
/// allow per-lane contiguous loads at `base + loopVar + C(lane=0)`.
private func isAffineOfLoopVar(
  _ offset: Lazy, loopVarId: VarID, in uops: [UOp], upTo: Int
) -> Bool {
  switch offset {
  case .constant:
    return true
  case .variable(let vid, _):
    if vid == loopVarId { return true }
    for j in 0..<upTo {
      guard case .variable(let producedVid, _) = uops[j].value, producedVid == vid
      else { continue }
      if case .add(let a, let b) = uops[j].op {
        return (isLoopVar(a, loopVarId: loopVarId) && isConstantLazy(b))
          || (isLoopVar(b, loopVarId: loopVarId) && isConstantLazy(a))
      }
      return false
    }
    return false
  default:
    return false
  }
}

private func isLoopVar(_ expr: Lazy, loopVarId: VarID) -> Bool {
  if case .variable(let vid, _) = expr, vid == loopVarId { return true }
  return false
}

private func isConstantLazy(_ expr: Lazy) -> Bool {
  if case .constant = expr { return true }
  return false
}

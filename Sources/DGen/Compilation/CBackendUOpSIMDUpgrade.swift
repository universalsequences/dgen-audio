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

// MARK: - Region element-loop SIMD upgrade

/// Lane classification for the region-loop SIMD legality analysis.
private enum LaneClass {
  case uniform  // same value for all 4 lanes (loop-invariant, or int math thereof)
  case affine  // loopVar + uniform: lanes are base, base+1, base+2, base+3
  case vector  // lane-varying float data
}

/// SIMD-upgrade for shape-aware region element loops (`beginForLoop` over a
/// constant element count, emitted by `RegionEmitter` inside sequential
/// frame-loop blocks). These are elementwise tensor-op loops that the
/// conservative `upgradeElementLoopsToSIMD` never sees because they are not
/// `beginParallelRange` loops.
///
/// Legality model (bails to scalar on anything else):
/// - int ops may combine uniform operands freely; `uniform + loopVar` is affine;
///   any other use of an affine value is rejected (offsets must be stride-1).
/// - memoryRead/memoryWrite offsets must be affine (contiguous vld1q/vst1q).
///   A memoryRead with a *uniform* offset is rewritten to `simdBroadcastLoad`.
/// - float operands defined outside the loop are lane-uniform scalars; each one
///   is hoisted to a `broadcastScalar` (vdupq) inserted before the loop and the
///   body is remapped to the broadcast value.
/// - loops writing to cross-block globals, containing control flow, stateful
///   ops, nested loops, or whose body values are referenced after the loop are
///   left scalar.
public func upgradeRegionElementLoopsToSIMD(
  _ uops: inout [UOp],
  globalVarIds: Set<VarID>,
  makeVar: () -> Lazy
) {
  var i = 0
  while i < uops.count {
    guard case .beginForLoop(let loopVarLazy, let countLazy) = uops[i].op,
      case .constant(_, let countF) = countLazy,
      case .variable(let loopVarId, _) = loopVarLazy
    else {
      i += 1
      continue
    }
    let elementCount = Int(countF)
    guard elementCount >= 4, elementCount % 4 == 0 else {
      i += 1
      continue
    }

    // Find matching endLoop (reject nested loops outright).
    let beginIdx = i
    var endIdx: Int? = nil
    var nested = false
    scan: for j in (beginIdx + 1)..<uops.count {
      switch uops[j].op {
      case .beginForLoop, .beginLoop, .beginReverseLoop, .beginParallelRange, .beginRange:
        nested = true
        break scan
      case .endLoop, .endParallelRange, .endRange:
        endIdx = j
        break scan
      default:
        break
      }
    }
    guard !nested, let endIdx, case .endLoop = uops[endIdx].op else {
      i += 1
      continue
    }

    // ── Body analysis ──
    var laneClass: [VarID: LaneClass] = [loopVarId: .affine]
    var definedInBody: Set<VarID> = [loopVarId]
    var outsideFloatOperands: [VarID: Lazy] = [:]  // varId -> exact Lazy reference
    var legal = true
    var broadcastLoadRewrites: [Int] = []  // memoryRead indices to rewrite

    func cls(_ l: Lazy) -> LaneClass {
      switch l {
      case .constant: return .uniform
      case .variable(let vid, _), .global(let vid):
        if let c = laneClass[vid] { return c }
        // Defined outside the loop: lane-uniform by construction.
        return .uniform
      default: return .uniform
      }
    }
    func destVarId(_ l: Lazy) -> VarID? {
      if case .variable(let vid, _) = l { return vid }
      if case .global(let vid) = l { return vid }
      return nil
    }
    // Record a float operand that lives outside the loop (needs a broadcast).
    func noteFloatOperand(_ l: Lazy) {
      guard case .variable(let vid, _) = l, !definedInBody.contains(vid) else { return }
      outsideFloatOperands[vid] = l
    }

    for k in (beginIdx + 1)..<endIdx {
      let uop = uops[k]
      // Writing a cross-block global from inside an element loop would render
      // as a frame-axis vst1q — wrong axis. Reject.
      if let dv = destVarId(uop.value), globalVarIds.contains(dv) {
        legal = false
        break
      }

      switch uop.op {
      case .add(let a, let b), .sub(let a, let b), .mul(let a, let b), .div(let a, let b),
        .mod(let a, let b):
        if uop.scalarType == .int {
          let ca = cls(a)
          let cb = cls(b)
          let dest = destVarId(uop.value)
          if case .add = uop.op, (ca == .affine && cb == .uniform) || (ca == .uniform && cb == .affine) {
            if let dest { laneClass[dest] = .affine }
          } else if ca == .uniform && cb == .uniform {
            if let dest { laneClass[dest] = .uniform }
          } else {
            legal = false
          }
        } else {
          if cls(a) != .affine, cls(b) != .affine {
            noteFloatOperand(a)
            noteFloatOperand(b)
            if let dest = destVarId(uop.value) { laneClass[dest] = .vector }
          } else {
            legal = false
          }
        }
      case .min(let a, let b), .max(let a, let b), .pow(let a, let b), .atan2(let a, let b):
        guard uop.scalarType != .int, cls(a) != .affine, cls(b) != .affine else {
          legal = false
          break
        }
        noteFloatOperand(a)
        noteFloatOperand(b)
        if let dest = destVarId(uop.value) { laneClass[dest] = .vector }
      case .abs(let a), .sign(let a), .sqrt(let a), .exp(let a), .log(let a), .log10(let a),
        .sin(let a), .cos(let a), .tan(let a), .atan(let a), .tanh(let a),
        .floor(let a), .ceil(let a), .round(let a):
        guard uop.scalarType != .int, cls(a) != .affine else {
          legal = false
          break
        }
        noteFloatOperand(a)
        if let dest = destVarId(uop.value) { laneClass[dest] = .vector }
      case .identity(let a):
        // Copies preserve the operand's lane class; int copies of the frame
        // index (`int tN = i;`) are the common uniform case.
        let ca = cls(a)
        if uop.scalarType == .int {
          guard ca != .vector else {
            legal = false
            break
          }
          if let dest = destVarId(uop.value) { laneClass[dest] = ca }
        } else {
          guard ca != .affine else {
            legal = false
            break
          }
          noteFloatOperand(a)
          if let dest = destVarId(uop.value) { laneClass[dest] = .vector }
        }
      case .cast(let a, let t):
        // int->int / value chains: keep uniform ints uniform; anything affine
        // or float-involving is rejected (rare in region loops).
        if t == .int, cls(a) == .uniform {
          if let dest = destVarId(uop.value) { laneClass[dest] = .uniform }
        } else {
          legal = false
        }
      case .frameIndex, .frameCount, .hostSampleRate:
        // Lane-uniform environment values (`i`, frameCount, sample rate).
        if let dest = destVarId(uop.value) { laneClass[dest] = .uniform }
      case .memoryRead(_, let offset):
        switch cls(offset) {
        case .affine:
          break
        case .uniform:
          broadcastLoadRewrites.append(k)
        case .vector:
          legal = false
        }
        if let dest = destVarId(uop.value) { laneClass[dest] = .vector }
      case .simdBroadcastLoad(_, let offset):
        guard cls(offset) == .uniform else {
          legal = false
          break
        }
        if let dest = destVarId(uop.value) { laneClass[dest] = .vector }
      case .memoryWrite(_, let offset, let value):
        guard cls(offset) == .affine else {
          legal = false
          break
        }
        noteFloatOperand(value)
      default:
        legal = false
      }

      if !legal {
        if ProcessInfo.processInfo.environment["DGEN_DEBUG_REGION_SIMD"] != nil {
          print("[RegionSIMD] reject count=\(elementCount): op \(uop.op) scalarType=\(uop.scalarType)")
        }
        break
      }
      if let dv = destVarId(uop.value) { definedInBody.insert(dv) }
    }

    guard legal else {
      i = endIdx + 1
      continue
    }

    // Reject if any body-defined value is referenced after the loop (its
    // scalar last-iteration semantics would change).
    let interiorDefs = definedInBody.subtracting([loopVarId])
    if !interiorDefs.isEmpty {
      var escaped = false
      outer: for j in (endIdx + 1)..<uops.count {
        for operand in uops[j].op.referencedVarIds() where interiorDefs.contains(operand) {
          escaped = true
          break outer
        }
      }
      if escaped {
        i = endIdx + 1
        continue
      }
    }

    // ── Apply upgrade ──
    // Uniform-offset reads become lane-uniform broadcast loads.
    for k in broadcastLoadRewrites {
      if case .memoryRead(let cell, let offset) = uops[k].op {
        uops[k] = UOp(
          op: .simdBroadcastLoad(cell, offset), value: uops[k].value,
          vectorWidth: uops[k].vectorWidth, tensorIndex: uops[k].tensorIndex,
          scalarType: uops[k].scalarType)
      }
    }

    // Hoist outside float scalars to vdupq broadcasts before the loop.
    var remap: [Lazy: Lazy] = [:]
    var broadcastUOps: [UOp] = []
    for (_, lazyRef) in outsideFloatOperands.sorted(by: { $0.key < $1.key }) {
      let dest = makeVar()
      var b = UOp(op: .broadcastScalar(lazyRef), value: dest)
      b.vectorWidth = 4
      broadcastUOps.append(b)
      remap[lazyRef] = dest
    }
    if !remap.isEmpty {
      for k in (beginIdx + 1)..<endIdx {
        uops[k] = UOp(
          op: uops[k].op.remapLazyInputs(remap), value: uops[k].value,
          vectorWidth: uops[k].vectorWidth, tensorIndex: uops[k].tensorIndex,
          scalarType: uops[k].scalarType)
      }
    }

    uops[beginIdx] = UOp(
      op: .beginParallelRange(elementCount, 4), value: loopVarLazy, vectorWidth: 4)
    uops[endIdx] = UOp(op: .endParallelRange, value: uops[endIdx].value, vectorWidth: 4)
    for k in (beginIdx + 1)..<endIdx {
      uops[k].vectorWidth = 4
    }

    uops.insert(contentsOf: broadcastUOps, at: beginIdx)
    i = endIdx + broadcastUOps.count + 1
  }
}

extension Op {
  /// All variable ids referenced as operands (not destinations) of this op.
  /// Conservative helper for escape analysis in the SIMD region upgrade.
  fileprivate func referencedVarIds() -> [VarID] {
    var ids: [VarID] = []
    func visit(_ l: Lazy) {
      if case .variable(let vid, _) = l { ids.append(vid) }
      if case .global(let vid) = l { ids.append(vid) }
    }
    // remapLazyInputs visits every Lazy operand; abuse it as a traversal.
    _ = remapLazyInputsCollecting(visit)
    return ids
  }

  /// Visits each Lazy operand via the identity remap.
  fileprivate func remapLazyInputsCollecting(_ visit: (Lazy) -> Void) -> Op {
    var seen: [Lazy: Lazy] = [:]
    // Build a remap dictionary that records visits lazily: remapLazyInputs only
    // consults the dictionary, so instead run it twice — once to collect via a
    // proxy is not possible with a plain dictionary. Enumerate cases directly.
    _ = seen
    switch self {
    case .store(_, let v): visit(v)
    case .delay1(_, let a): visit(a)
    case .mse(let a, let b), .mutate(let a, let b), .add(let a, let b), .sub(let a, let b),
      .mul(let a, let b), .div(let a, let b), .and(let a, let b), .or(let a, let b),
      .xor(let a, let b), .pow(let a, let b), .atan2(let a, let b), .mod(let a, let b),
      .gt(let a, let b), .gte(let a, let b), .lte(let a, let b), .lt(let a, let b),
      .eq(let a, let b), .min(let a, let b), .max(let a, let b), .latch(let a, let b),
      .loadTape(let a, let b), .beginRange(let a, let b), .beginForLoop(let a, let b):
      visit(a)
      visit(b)
    case .abs(let a), .sign(let a), .sin(let a), .cos(let a), .tan(let a), .atan(let a),
      .tanh(let a), .exp(let a), .log(let a), .log10(let a), .sqrt(let a), .floor(let a),
      .ceil(let a), .round(let a), .identity(let a), .declareVar(let a), .cast(let a, _),
      .beginIf(let a), .beginHopCheck(let a), .beginLoop(let a, _), .beginReverseLoop(let a),
      .setFrameIndex(let a), .output(_, let a), .broadcastScalar(let a):
      visit(a)
    case .gswitch(let c, let a, let b):
      visit(c)
      visit(a)
      visit(b)
    case .selector(let m, let opts):
      visit(m)
      opts.forEach(visit)
    case .memoryRead(_, let o), .simdBroadcastLoad(_, let o):
      visit(o)
    case .memoryWrite(_, let o, let v), .memoryAccumulate(_, let o, let v):
      visit(o)
      visit(v)
    default:
      break
    }
    return self
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

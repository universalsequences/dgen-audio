import Foundation

extension LazyOp {
  public func emit(ctx: IRContext, g: Graph, nodeId: NodeID) throws -> [UOp] {
    guard let node = g.nodes[nodeId] else { return [] }

    // collect operands
    let inputs: [Lazy] = node.inputs.compactMap { ctx.values[$0] }
    var ops: [UOp] = []
    let b = IRBuilder(ctx: ctx, nodeId: nodeId)

    switch self {
    // MARK: - Early returns
    case .constant(let value):
      _ = ctx.useConstant(src: nodeId, value: value)
      return []
    case .hostSampleRate:
      let dest = ctx.useVariable(src: nodeId)
      ops.append(UOp(op: .hostSampleRate, value: dest))
    case .tensorRef(_):
      // Register a placeholder value so that downstream ops can find this input
      // The actual tensor data is accessed via nodeToTensor lookup
      ctx.values[nodeId] = .empty
      return []

    // MARK: - Arithmetic
    case .add:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "add", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 + $1 }
    case .sub:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "sub", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 - $1 }
    case .mul:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "mul", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 * $1 }
    case .div:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "div", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 / $1 }
    case .mod:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "mod", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 % $1 }
    case .min:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "min", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.min($0, $1) }
    case .max:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "max", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.max($0, $1) }
    case .and:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "and", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.and($0, $1) }
    case .or:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "or", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.or($0, $1) }
    case .xor:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "xor", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.xor($0, $1) }
    case .atan2:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "atan2", expected: 2, actual: inputs.count)
      }
      // Route through emitBinaryOp so tensor inputs (tensorRef, views, etc.)
      // get resolved via readInput → tensorRead using the block's tensorIndex.
      // Previously used `b.value()` directly, which returns `.empty` for
      // tensorRef nodes and renders as `/* unknown lazy */` in the C backend.
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.atan2($0, $1) }
    case .pow:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "pow", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { b.pow($0, $1) }

    // MARK: - Unary math
    case .abs:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "abs", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.abs($0) }
    case .sign:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "sign", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.sign($0) }
    case .sin:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "sin", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.sin($0) }
    case .neg:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "neg", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.neg($0) }
    case .cos:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "cos", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.cos($0) }
    case .tan:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "tan", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.tan($0) }
    case .atan:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "atan", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.atan($0) }
    case .tanh:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "tanh", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.tanh($0) }
    case .exp:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "exp", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.exp($0) }
    case .log:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "log", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.log($0) }
    case .log10:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "log10", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.log10($0) }
    case .sqrt:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "sqrt", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.sqrt($0) }
    case .floor:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "floor", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.floor($0) }
    case .round:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "round", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.round($0) }
    case .ceil:
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "ceil", expected: 1, actual: inputs.count)
      }
      try emitUnaryOp(b: b, g: g, node: node, inputs: inputs) { b.ceil($0) }

    // MARK: - Comparison
    case .gt:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "gt", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 > $1 }
    case .gte:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "gte", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 >= $1 }
    case .lte:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "lte", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 <= $1 }
    case .lt:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "lt", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 < $1 }
    case .eq:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "eq", expected: 2, actual: inputs.count)
      }
      try emitBinaryOp(b: b, g: g, node: node, inputs: inputs) { $0 == $1 }

    // MARK: - Control flow
    case .gswitch:
      guard inputs.count == 3 else {
        throw DGenError.insufficientInputs(
          operator: "gswitch", expected: 3, actual: inputs.count)
      }
      try emitTernaryOp(b: b, g: g, node: node, inputs: inputs) { b.gswitch($0, $1, $2) }
    case .selector:
      guard inputs.count >= 2 else {
        throw DGenError.insufficientInputs(
          operator: "selector", expected: 2, actual: inputs.count)
      }
      let mode = inputs[0]
      let options = Array(inputs.dropFirst())
      b.use(val: b.selector(b.value(mode), options.map { b.value($0) }))
    case .modulatedParam(let mode, let minValue, let maxValue, let baseCellId, let activeCellId, let lanes):
      guard inputs.count == 1 else {
        throw DGenError.insufficientInputs(
          operator: "modulatedParam", expected: 1, actual: inputs.count)
      }
      let zeroOffset = b.intConstant(0)
      let base = b.simdBroadcastLoad(baseCellId, zeroOffset)
      let active = b.simdBroadcastLoad(activeCellId, zeroOffset)
      var modulation = b.constant(0.0)
      for lane in lanes {
        let modulator = b.input(lane.modulatorChannel)
        let depth = b.simdBroadcastLoad(lane.depthCellId, zeroOffset)
        modulation = modulation + (modulator * depth)
      }

      let resolved: Expr
      switch mode {
      case .additive:
        resolved = b.min(b.max(base + modulation, b.constant(minValue)), b.constant(maxValue))
      case .multiplicative:
        resolved = b.min(
          b.max(base * (b.constant(1.0) + modulation), b.constant(minValue)),
          b.constant(maxValue))
      case .semitone:
        resolved = b.min(
          b.max(
            base * b.exp(b.constant(logf(2.0)) * (modulation / b.constant(12.0))),
            b.constant(minValue)),
          b.constant(maxValue))
      }
      b.use(val: b.gswitch(active > b.constant(0.0), resolved, base))
    case .mix:
      guard inputs.count == 3 else {
        throw DGenError.insufficientInputs(
          operator: "mix", expected: 3, actual: inputs.count)
      }
      try emitTernaryOp(b: b, g: g, node: node, inputs: inputs) {
        let val = u_mix($0, $1, lerp: $2)(b)
        b.use(val: val)
        return val
      }
    case .mse:
      guard inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "mse", expected: 2, actual: inputs.count)
      }
      let (a, b2) = b.values(inputs, count: 2)
      b.use(val: u_mse(a, b2)(b))

    // MARK: - Dispatched categories

    case .spectralLossFFT, .spectralLossFFTGradSpec, .spectralLossFFTGradIFFT,
      .spectralLossFFTGradInline, .spectralLossFFTGradRead, .spectralLossFFTGradRead2,
      .spectralLossFFTBatched, .spectralLossFFTBatchedReduce,
      .spectralLossFFTBatchedGradSpec, .spectralLossFFTBatchedGradIFFT,
      .spectralLossFFTBatchedGradRead, .spectralLossFFTBatchedGradRead2:
      try emitSpectralLoss(b: b, ctx: ctx, g: g, node: node, inputs: inputs)

    case .selectRow, .selectRowGradWrite, .selectRowGradReduce,
      .peekGradWrite, .peekGradReduce,
      .sampleInline, .sampleGradWrite, .sampleGradReduce:
      try emitRowSelection(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .overlapAdd, .overlapAddGradStore, .overlapAddGradGather,
      .bufferViewGradStore, .bufferViewGradRead:
      try emitFFT(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .tensorNoise(let stateCell, let outputCell, let size):
      // Sequential loop over N elements, each advancing the shared xorshift
      // state and storing one independent random value into the current
      // frame's slice of the frame-aware outputCell. Downstream consumers
      // read via the nodeToTensor mapping set up in `graph.noise(size:)`.
      let two = b.constant(2.0)
      let one = b.constant(1.0)
      b.loop(size) { i in
        let r = b.noise(stateCell)            // [0, 1) via xorshift, advances shared state
        let scaled = r * two - one            // [-1, 1) to match scalar `.noise`
        _ = b.frameAwareTensorWrite(
          cellId: outputCell, tensorSize: size, elemIdx: i, value: scaled)
      }
      ctx.values[nodeId] = .empty

    case .spectrumDelay(let ringCell, let rowCell, let outputCell, let N, let hops):
      // One-hop delay line for `[N]` spectra. Inputs: [input, hopCounter].
      // On hop boundaries (counter == 0): write current input to ring row
      // `rowCell`, advance rowCell, copy the NEW (= oldest) ring row into
      // `outputCell`. Downstream reads `outputCell` as a static [N] tensor.
      guard node.inputs.count == 2 else {
        throw DGenError.insufficientInputs(
          operator: "spectrumDelay", expected: 2, actual: node.inputs.count)
      }
      guard let inputTensorId = g.nodeToTensor[node.inputs[0]],
        let inputTensor = g.tensors[inputTensorId]
      else {
        throw DGenError.tensorError(
          op: "spectrumDelay", reason: "input 0 must be a tensor node")
      }
      let counter = b.value(inputs[1])
      let zero = b.constant(0.0)
      let one = b.constant(1.0)
      let rows = b.constant(Float(hops + 1))
      let Nint = b.intConstant(N)

      b.if_(counter == zero) {
        let rowF = b.memoryRead(rowCell, zero)
        let rowIdx = b.cast(rowF, to: .int)
        // Write current input spectrum to ring row `rowIdx`.
        b.loop(N) { e in
          let value = b.tensorRead(inputTensor, flatIdx: e, shape: [N])
          _ = b.memoryWrite(ringCell, rowIdx * Nint + e, value)
        }
        // Advance row counter, modulo (hops + 1).
        let nextRow = b.gswitch(rowF + one >= rows, zero, rowF + one)
        let nextRowIdx = b.cast(nextRow, to: .int)
        _ = b.memoryWrite(rowCell, zero, nextRow)
        // Copy the new row (= oldest spectrum, `hops` hops ago) to output.
        b.loop(N) { e in
          let v = b.memoryRead(ringCell, nextRowIdx * Nint + e)
          _ = b.memoryWrite(outputCell, e, v)
        }
      }
      ctx.values[nodeId] = .empty

    case .spectrumDelayMod(let ringCell, let rowCell, let outputCell, let N, let maxHops):
      // Modulated fractional spectrum delay. Inputs: [input, counter, delay].
      // `delay` is a hop-rate scalar in `[0, maxHops]`. On hop boundaries:
      //   1. Write current input to ring[rowCell]
      //   2. Advance rowCell mod (maxHops + 1)
      //   3. Linearly interpolate between the ring rows at `floor(delay)`
      //      and `floor(delay)+1` hops ago; write result to outputCell.
      guard node.inputs.count == 3 else {
        throw DGenError.insufficientInputs(
          operator: "spectrumDelayMod", expected: 3, actual: node.inputs.count)
      }
      guard let inputTensorId2 = g.nodeToTensor[node.inputs[0]],
        let inputTensor2 = g.tensors[inputTensorId2]
      else {
        throw DGenError.tensorError(
          op: "spectrumDelayMod", reason: "input 0 must be a tensor node")
      }
      let counterM = b.value(inputs[1])
      let delayVal = b.value(inputs[2])
      let zeroM = b.constant(0.0)
      let oneM = b.constant(1.0)
      let rowsF = b.constant(Float(maxHops + 1))
      let maxDelayF = b.constant(Float(maxHops))
      let NintM = b.intConstant(N)
      let rowsInt = b.intConstant(maxHops + 1)

      b.if_(counterM == zeroM) {
        // 1. Write current input to ring[rowCell].
        let rowF = b.memoryRead(rowCell, zeroM)
        let rowIdx = b.cast(rowF, to: .int)
        b.loop(N) { e in
          let v = b.tensorRead(inputTensor2, flatIdx: e, shape: [N])
          _ = b.memoryWrite(ringCell, rowIdx * NintM + e, v)
        }
        // 2. Advance row counter.
        let nextRowF = b.gswitch(rowF + oneM >= rowsF, zeroM, rowF + oneM)
        _ = b.memoryWrite(rowCell, zeroM, nextRowF)

        // 3. Clamp the modulated delay to `[0, maxHops]` so the
        // interpolation reads two valid rows (kCeil never exceeds
        // maxHops).
        let dClamped = b.gswitch(
          delayVal > maxDelayF, maxDelayF,
          b.gswitch(delayVal < zeroM, zeroM, delayVal))
        // kFloor = floor(delay), frac = delay - kFloor.
        let kFloorF = b.floor(dClamped)
        let frac = dClamped - kFloorF
        let oneMinusFrac = oneM - frac
        let kFloorInt = b.cast(kFloorF, to: .int)
        let kCeilInt = kFloorInt + b.intConstant(1)
        // `rowCell` now points to the NEXT write row (= oldest).
        // The "just-written" row is `nextRowF - 1` wrapped.
        // `k hops ago` row = (nextRowF - 1 - k + rows) mod rows.
        let newRowIdx = b.cast(nextRowF, to: .int)
        let baseInt = newRowIdx - b.intConstant(1) + rowsInt * b.intConstant(2)
        let rowNear = b.mod(baseInt - kFloorInt, rowsInt)
        let rowFar = b.mod(baseInt - kCeilInt, rowsInt)

        b.loop(N) { e in
          let near = b.memoryRead(ringCell, rowNear * NintM + e)
          let far = b.memoryRead(ringCell, rowFar * NintM + e)
          let mix = near * oneMinusFrac + far * frac
          _ = b.memoryWrite(outputCell, e, mix)
        }
      }
      ctx.values[nodeId] = .empty

    case .hopTensorNoise(let stateCell, let outputCell, let size):
      // Fused noise + hopHold. The single input is the hop counter accum;
      // on frames where counter == 0, generate N fresh random values into
      // the persistent outputCell. Between hops the cell holds, so the
      // tensorRef read path always returns the current hop's snapshot.
      // This avoids the N-per-frame work that `tensorNoise → hopHold`
      // would pay.
      let counter = b.value(inputs[0])
      let zeroScalar = b.constant(0.0)
      let twoCh = b.constant(2.0)
      let oneCh = b.constant(1.0)
      b.if_(counter == zeroScalar) {
        b.loop(size) { i in
          let r = b.noise(stateCell)
          let scaled = r * twoCh - oneCh
          _ = b.memoryWrite(outputCell, i, scaled)
        }
      }
      ctx.values[nodeId] = .empty

    case .acceleratedFFT, .acceleratedIFFT:
      try emitAcceleratedFFT(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .phaseVocoderPitchShift:
      try emitPhaseVocoderPitchShift(
        b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .partitionedSpectralConvolve:
      try emitPartitionedSpectralConvolve(
        b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .gemm, .gemmChunkPartials:
      try emitGemm(b: b, ctx: ctx, g: g, node: node, nodeId: nodeId, ops: &ops)

    case .gemmStaged, .gemmStagedChunkPartials:
      try emitGemmStaged(b: b, ctx: ctx, g: g, node: node, nodeId: nodeId, ops: &ops)

    case .conv1d, .conv2d, .sum, .sumAxis, .sumMulAxis0, .gemmSmall, .maxAxis, .meanAxis, .reshape, .asStrided, .transpose, .shrink,
      .pad, .expandView, .repeatView, .peek, .expand, .expandAxis, .gradPhasor:
      try emitTensorOp(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId, ops: &ops)

    case .memoryRead, .memoryWrite, .memoryAccumulate, .memoryCellSum, .tensorAccumulate,
      .chunkPartialsReduceToCell,
      .historyWrite, .historyReadWrite, .historyRead, .param, .latch, .click, .noise,
      .phasor, .deterministicPhasor, .gradDeterministicPhasor, .accum,
      .output, .input, .seq:
      try emitStateOp(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)
    }
    ops.append(contentsOf: b.ops)
    return ops
  }
}

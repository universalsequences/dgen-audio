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
      b.use(val: b.atan2(b.value(inputs[0]), b.value(inputs[1])))
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
      .spectralLossFFTGradInline, .spectralLossFFTGradRead, .spectralLossFFTGradRead2:
      try emitSpectralLoss(b: b, ctx: ctx, g: g, node: node, inputs: inputs)

    case .selectRow, .peekRowInline, .selectRowGradWrite, .selectRowGradReduce,
      .peekRowGradWrite, .peekRowGradReduce:
      try emitRowSelection(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .fft, .ifft:
      try emitFFT(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)

    case .conv1d, .conv2d, .sum, .sumAxis, .maxAxis, .meanAxis, .reshape, .asStrided, .transpose, .shrink,
      .pad, .expandView, .repeatView, .peek, .expand, .expandAxis, .gradPhasor:
      try emitTensorOp(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId, ops: &ops)

    case .memoryRead, .memoryWrite, .memoryAccumulate, .memoryCellSum, .tensorAccumulate,
      .historyWrite, .historyReadWrite, .historyRead, .param, .latch, .click, .noise,
      .phasor, .deterministicPhasor, .gradDeterministicPhasor, .accum,
      .output, .input, .seq:
      try emitStateOp(b: b, ctx: ctx, g: g, node: node, inputs: inputs, nodeId: nodeId)
    }
    ops.append(contentsOf: b.ops)
    return ops
  }
}

// LispEvaluator - AST → DGenLazy Signal/Tensor graph building
//
// Walks AST nodes from LispParser and calls DGenLazy APIs to build
// the computation graph. Supports arithmetic, math, signal generators,
// stateful ops, effects, I/O, tensors, macros, and history feedback.

import DGen
import DGenLazy
import Foundation

// Resolve Tensor ambiguity: DGenLazy.Tensor is the one we use everywhere
typealias Tensor = DGenLazy.Tensor

// MARK: - Types

enum EvalResult {
  case signal(Signal)
  case tensor(Tensor)
  case signalTensor(SignalTensor)
  case tuple([EvalResult])
  case float(Float)
  case none
}

struct ParamInfo {
  let name: String
  let cellId: CellID?
  let defaultValue: Float
  let min: Float?
  let max: Float?
  let unit: String?
  let hidden: Bool
  let group: String?
  let env: String?
  let role: UIEnvelopeRole?
  let generatedKind: String?
  let generatedFor: String?
  let modulationMode: ModulationMode?
  let modulationDepthMin: Float?
  let modulationDepthMax: Float?
  let modulationActiveParamName: String?
  let modulationResolvedSymbolName: String?
  let generatedModulatorSlot: Int?
}

struct OutputInfo {
  let channel: Int
  let signal: Signal
  let name: String?
  let modulatorSlot: Int?
}

struct InputInfo {
  let channel: Int
  let name: String?
  let modulatorSlot: Int?
}

struct TensorInfo {
  var name: String
  let shape: [Int]
  let kind: String
  let mutable: Bool
  let sourceFile: String?
  let sourceSampleRate: Float?
  let data: [Float]?
}

struct MacroDefinition {
  let params: [String]
  let body: [ASTNode]
}

// MARK: - Evaluator

class LispEvaluator {
  var definitions: [String: EvalResult] = [:]
  var historyBindings: [String: (read: Signal, write: (Signal) -> Signal)] = [:]
  var tensorHistoryBindings: [String: TensorHistory] = [:]
  var macros: [String: MacroDefinition] = [:]
  var params: [ParamInfo] = []
  var outputs: [OutputInfo] = []
  var inputs: [InputInfo] = []
  var tensors: [TensorInfo] = []
  var macroExpansionCounter: Int = 0
  let sourceDirectory: URL

  init(sourceDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)) {
    self.sourceDirectory = sourceDirectory
  }

  // MARK: - Top-level evaluation

  func evaluate(source: String) throws {
    let nodes = try parseSource(source)
    try evaluate(nodes: nodes)
  }

  func evaluate(nodes: [ASTNode]) throws {
    for node in nodes {
      let _ = try evaluateAST(node)
    }
    try validateParamUIMetadata(params)
  }

  // MARK: - AST Evaluation

  func evaluateAST(_ node: ASTNode) throws -> EvalResult {
    switch node {
    case .atom(let value):
      return try evaluateAtom(value)

    case .list(let elements):
      guard !elements.isEmpty else {
        throw LispError.parseError("Empty list")
      }

      guard case .atom(let opName) = elements[0] else {
        throw LispError.parseError("First element must be an operator name")
      }

      switch opName.lowercased() {
      case "def":
        return try evaluateDef(elements)
      case "defmacro":
        return try evaluateDefmacro(elements)
      case "make-history":
        return try evaluateMakeHistory(elements)
      case "read-history":
        return try evaluateReadHistory(elements)
      case "write-history":
        return try evaluateWriteHistory(elements)
      case "make-tensor-history":
        return try evaluateMakeTensorHistory(elements)
      case "read-tensor-history":
        return try evaluateReadTensorHistory(elements)
      case "write-tensor-history":
        return try evaluateWriteTensorHistory(elements)
      default:
        if let macro = macros[opName] {
          return try evaluateMacroCall(
            macro: macro, macroName: opName,
            args: Array(elements.dropFirst()))
        }
        return try evaluateOperator(opName, args: Array(elements.dropFirst()))
      }
    }
  }

  // MARK: - Atom evaluation

  private func evaluateAtom(_ value: String) throws -> EvalResult {
    if value.lowercased() == "samplerate" || value.lowercased() == "sample-rate" {
      return .signal(Signal.hostSampleRate())
    }

    if let result = definitions[value] {
      return result
    }

    if let number = Float(value) {
      return .float(number)
    }

    // Named constants
    switch value.lowercased() {
    case "pi": return .float(.pi)
    case "twopi", "tau": return .float(.pi * 2)
    case "e": return .float(Float(M_E))
    case "true": return .float(1.0)
    case "false": return .float(0.0)
    default:
      throw LispError.unknownSymbol(value)
    }
  }

  // MARK: - Special forms

  private func evaluateDef(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 3 else {
      throw LispError.parseError("def requires name and value: (def name expr)")
    }

    // Evaluate all body expressions, use last as value
    var result: EvalResult = .none
    for i in 2..<elements.count {
      result = try evaluateAST(elements[i])
    }

    if case .list(let targets) = elements[1] {
      guard elements.count == 3 else {
        throw LispError.parseError("destructuring def requires exactly one value expression")
      }
      let names = try targets.map { node -> String in
        guard case .atom(let name) = node else {
          throw LispError.parseError("def destructuring targets must be atoms")
        }
        return name
      }
      guard case .tuple(let values) = result else {
        throw LispError.typeError("destructuring def requires a tuple-producing expression")
      }
      guard values.count == names.count else {
        throw LispError.typeError(
          "destructuring def expected \(names.count) values, got \(values.count)")
      }
      for (name, value) in zip(names, values) {
        definitions[name] = value
      }
      return .none
    }

    guard case .atom(let name) = elements[1] else {
      throw LispError.parseError("def: name must be an atom or a destructuring list")
    }

    if case .tensor = result, let idx = tensors.indices.last, tensors[idx].name.isEmpty {
      tensors[idx].name = name
    }
    definitions[name] = result
    return .none
  }

  private func evaluateDefmacro(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 4 else {
      throw LispError.parseError(
        "defmacro requires at least 3 arguments: (defmacro name (params...) body...)")
    }
    guard case .atom(let macroName) = elements[1] else {
      throw LispError.parseError("defmacro: macro name must be an atom")
    }
    guard case .list(let paramNodes) = elements[2] else {
      throw LispError.parseError(
        "defmacro: second argument must be a parameter list (param1 param2 ...)")
    }

    var paramNames: [String] = []
    for paramNode in paramNodes {
      guard case .atom(let paramName) = paramNode else {
        throw LispError.parseError("defmacro: all parameters must be atoms")
      }
      paramNames.append(paramName)
    }

    let body = Array(elements.dropFirst(3))
    guard !body.isEmpty else {
      throw LispError.parseError("defmacro: body cannot be empty")
    }

    macros[macroName] = MacroDefinition(params: paramNames, body: body)
    return .none
  }

  /// Parses trailing `@attr value` pairs starting at `startIndex` (after the name).
  private func parseTrailingAttributes(
    _ elements: [ASTNode], startIndex: Int, form: String
  ) throws -> [(name: String, value: String)] {
    var attrs: [(name: String, value: String)] = []
    var i = startIndex
    while i < elements.count {
      guard case .atom(let key) = elements[i], key.hasPrefix("@") else {
        throw LispError.parseError("\(form) expects attributes after the name")
      }
      if i + 1 < elements.count, case .atom(let attrValue) = elements[i + 1] {
        attrs.append((key, attrValue))
        i += 2
      } else {
        attrs.append((key, ""))
        i += 1
      }
    }
    return attrs
  }

  /// Creates a tensor history binding (optionally hop-gated) from parsed attributes.
  private func makeTensorHistoryBinding(
    name: String, attrs: [(name: String, value: String)], form: String
  ) throws {
    guard let shapeStr = attrValue(attrs, "@shape") else {
      throw LispError.invalidArgument("\(form) requires @shape [d1,d2,...]")
    }
    let shape = parseShape(shapeStr)
    let hop = attrValue(attrs, "@hop").flatMap { Int($0) }
    let data = attrValue(attrs, "@data").map { parseFloatList($0) }
    tensorHistoryBindings[name] = TensorHistory(shape: shape, hop: hop, data: data)
  }

  /// `(make-history name)` creates a scalar signal feedback cell.
  /// `(make-history name @shape [...] [@hop N] [@data [...]])` creates a tensor
  /// history (the `@shape` form); with `@hop` the feedback advances once per hop
  /// (fs/hop) for STFT-style spectral state. Both attributes are optional; absent
  /// `@shape` yields the scalar form. `read-history`/`write-history` work on either.
  private func evaluateMakeHistory(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 2, case .atom(let name) = elements[1] else {
      throw LispError.parseError("make-history requires a name: (make-history name)")
    }
    let attrs = try parseTrailingAttributes(elements, startIndex: 2, form: "make-history")
    if attrValue(attrs, "@shape") != nil {
      try makeTensorHistoryBinding(name: name, attrs: attrs, form: "make-history")
      return .none
    }
    let history = Signal.history()
    historyBindings[name] = history
    return .none
  }

  private func evaluateReadHistory(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 2, case .atom(let name) = elements[1] else {
      throw LispError.parseError("read-history requires a name")
    }
    if let history = tensorHistoryBindings[name] {
      return .signalTensor(history.read())
    }
    guard let binding = historyBindings[name] else {
      throw LispError.historyNotFound(name)
    }
    return .signal(binding.read)
  }

  private func evaluateWriteHistory(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 3, case .atom(let name) = elements[1] else {
      throw LispError.parseError("write-history requires name and value")
    }
    if let history = tensorHistoryBindings[name] {
      return try writeTensorHistoryValue(history, valueNode: elements[2])
    }
    guard let binding = historyBindings[name] else {
      throw LispError.historyNotFound(name)
    }
    let value = try requireSignal(evaluateAST(elements[2]))
    let result = binding.write(value)
    return .signal(result)
  }

  private func evaluateMakeTensorHistory(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 2, case .atom(let name) = elements[1] else {
      throw LispError.parseError("make-tensor-history requires a name")
    }
    let attrs = try parseTrailingAttributes(elements, startIndex: 2, form: "make-tensor-history")
    try makeTensorHistoryBinding(name: name, attrs: attrs, form: "make-tensor-history")
    return .none
  }

  private func evaluateReadTensorHistory(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 2, case .atom(let name) = elements[1] else {
      throw LispError.parseError("read-tensor-history requires a name")
    }
    guard let history = tensorHistoryBindings[name] else {
      throw LispError.historyNotFound(name)
    }
    return .signalTensor(history.read())
  }

  private func writeTensorHistoryValue(
    _ history: TensorHistory, valueNode: ASTNode
  ) throws -> EvalResult {
    let value = try evaluateAST(valueNode)
    switch value {
    case .tensor(let t):
      return .tensor(history.write(t))
    case .signalTensor(let st):
      return .signalTensor(history.write(st))
    default:
      throw LispError.typeError("write-tensor-history: value must be tensor or signalTensor")
    }
  }

  private func evaluateWriteTensorHistory(_ elements: [ASTNode]) throws -> EvalResult {
    guard elements.count >= 3, case .atom(let name) = elements[1] else {
      throw LispError.parseError("write-tensor-history requires name and value")
    }
    guard let history = tensorHistoryBindings[name] else {
      throw LispError.historyNotFound(name)
    }
    return try writeTensorHistoryValue(history, valueNode: elements[2])
  }

  // MARK: - Macro expansion

  private func evaluateMacroCall(
    macro: MacroDefinition, macroName: String, args: [ASTNode]
  ) throws -> EvalResult {
    guard args.count == macro.params.count else {
      throw LispError.parseError(
        "Macro '\(macroName)' expects \(macro.params.count) arguments, got \(args.count)")
    }

    let scopePrefix = "_m\(macroExpansionCounter)_"
    macroExpansionCounter += 1

    let localDefNames = findDefNamesInBody(macro.body)

    var substitutions: [String: ASTNode] = [:]
    for (param, arg) in zip(macro.params, args) {
      substitutions[param] = arg
    }
    for defName in localDefNames {
      substitutions[defName] = .atom(scopePrefix + defName)
    }

    var lastResult: EvalResult = .none
    for bodyExpr in macro.body {
      var expandedExpr = substituteInAST(bodyExpr, substitutions: substitutions)
      expandedExpr = scopeDefName(expandedExpr, prefix: scopePrefix, localNames: localDefNames)
      lastResult = try evaluateAST(expandedExpr)
    }

    return lastResult
  }

  private func findDefNamesInBody(_ body: [ASTNode]) -> Set<String> {
    var names: Set<String> = []
    for expr in body {
      findDefNamesInAST(expr, into: &names)
    }
    return names
  }

  private func findDefNamesInAST(_ node: ASTNode, into names: inout Set<String>) {
    switch node {
    case .atom:
      break
    case .list(let elements):
      guard elements.count >= 2,
        case .atom(let op) = elements[0]
      else {
        for element in elements { findDefNamesInAST(element, into: &names) }
        return
      }
      let opLower = op.lowercased()
      if opLower == "def" || opLower == "make-history" || opLower == "make-tensor-history" {
        collectBindingNames(elements[1], into: &names)
      }
      for element in elements { findDefNamesInAST(element, into: &names) }
    }
  }

  private func scopeDefName(_ node: ASTNode, prefix: String, localNames: Set<String>) -> ASTNode {
    switch node {
    case .atom:
      return node
    case .list(let elements):
      if elements.count >= 2,
        case .atom(let op) = elements[0],
        bindingTargetContainsLocal(elements[1], localNames: localNames)
      {
        let opLower = op.lowercased()
        if opLower == "def" || opLower == "make-history" || opLower == "make-tensor-history" {
          var newElements = elements
          newElements[1] = scopeBindingTarget(
            newElements[1], prefix: prefix, localNames: localNames)
          for i in 2..<newElements.count {
            newElements[i] = scopeDefName(newElements[i], prefix: prefix, localNames: localNames)
          }
          return .list(newElements)
        }
      }
      return .list(elements.map { scopeDefName($0, prefix: prefix, localNames: localNames) })
    }
  }

  private func collectBindingNames(_ node: ASTNode, into names: inout Set<String>) {
    switch node {
    case .atom(let name):
      names.insert(name)
    case .list(let elements):
      for element in elements {
        collectBindingNames(element, into: &names)
      }
    }
  }

  private func bindingTargetContainsLocal(_ node: ASTNode, localNames: Set<String>) -> Bool {
    switch node {
    case .atom(let name):
      return localNames.contains(name)
    case .list(let elements):
      return elements.contains { bindingTargetContainsLocal($0, localNames: localNames) }
    }
  }

  private func scopeBindingTarget(_ node: ASTNode, prefix: String, localNames: Set<String>)
    -> ASTNode
  {
    switch node {
    case .atom(let name):
      return localNames.contains(name) ? .atom(prefix + name) : node
    case .list(let elements):
      return .list(elements.map { scopeBindingTarget($0, prefix: prefix, localNames: localNames) })
    }
  }

  private func substituteInAST(_ node: ASTNode, substitutions: [String: ASTNode]) -> ASTNode {
    switch node {
    case .atom(let value):
      if let replacement = substitutions[value] {
        return replacement
      }
      return node
    case .list(let elements):
      return .list(elements.map { substituteInAST($0, substitutions: substitutions) })
    }
  }

  // MARK: - Operator dispatch

  private func evaluateOperator(_ opName: String, args: [ASTNode]) throws -> EvalResult {
    // Extract @attribute pairs
    var attributePairs: [(name: String, value: String)] = []
    var regularArgs: [ASTNode] = []

    var i = 0
    while i < args.count {
      if case .atom(let value) = args[i], value.hasPrefix("@") {
        let attrName = value
        if i + 1 < args.count, case .atom(let attrValue) = args[i + 1] {
          attributePairs.append((name: attrName, value: attrValue))
          i += 2
          continue
        } else {
          attributePairs.append((name: attrName, value: ""))
          i += 1
          continue
        }
      }
      regularArgs.append(args[i])
      i += 1
    }

    let op = opName.lowercased()

    switch op {
    // Arithmetic (binary)
    case "+":
      return try evalBinaryArith(regularArgs, op: op)
    case "-":
      if regularArgs.count == 1 {
        return try evalUnaryNegate(regularArgs[0])
      }
      return try evalBinaryArith(regularArgs, op: op)
    case "*":
      return try evalBinaryArith(regularArgs, op: op)
    case "/":
      return try evalBinaryArith(regularArgs, op: op)
    case "%":
      return try evalMod(regularArgs)

    // Unary math
    case "sin", "cos", "tan", "atan", "tanh", "exp", "log", "log10", "sqrt", "abs", "sign",
      "floor", "ceil", "round", "relu", "sigmoid":
      return try evalUnaryMath(regularArgs, fn: op)

    // Binary math
    case "pow":
      return try evalPow(regularArgs)
    case "atan2":
      return try evalAtan2(regularArgs)
    case "min":
      return try evalBinaryArith(regularArgs, op: op)
    case "max":
      return try evalBinaryArith(regularArgs, op: op)
    case "mse":
      return try evalMse(regularArgs)

    // Comparison
    case "gt", ">": return try evalComparison(regularArgs, op: "gt")
    case "lt", "<": return try evalComparison(regularArgs, op: "lt")
    case "gte", ">=": return try evalComparison(regularArgs, op: "gte")
    case "lte", "<=": return try evalComparison(regularArgs, op: "lte")
    case "eq", "==": return try evalComparison(regularArgs, op: "eq")

    // Signal generators
    case "phasor":
      return try evalPhasor(regularArgs)
    case "stateful-phasor":
      return try evalStatefulPhasor(regularArgs)
    case "noise":
      return try evalNoise(regularArgs, attributes: attributePairs)
    case "click":
      return .signal(Signal.click())
    case "ramp2trig":
      guard regularArgs.count == 1 else {
        throw LispError.invalidArgument("ramp2trig requires 1 argument (ramp signal)")
      }
      let ramp = try requireSignal(evaluateAST(regularArgs[0]))
      return .signal(ramp.rampToTrig())

    // Stateful
    case "accum":
      return try evalAccum(regularArgs)
    case "latch":
      return try evalLatch(regularArgs)
    case "hop-hold", "hophold":
      return try evalHopHold(regularArgs)
    case "mix":
      return try evalMix(regularArgs)

    // Effects
    case "biquad":
      return try evalBiquad(regularArgs, attributes: attributePairs)
    case "compressor":
      return try evalCompressor(regularArgs, rawArgs: args, attributes: attributePairs)
    case "delay":
      return try evalDelay(regularArgs)

    // I/O
    case "param":
      return try evalParam(regularArgs, attributes: attributePairs)
    case "in":
      return try evalInput(regularArgs, attributes: attributePairs)
    case "out":
      return try evalOutput(regularArgs, attributes: attributePairs)

    // Tensor creation
    case "tensor":
      return try evalTensor(regularArgs, attributes: attributePairs)
    case "wavetable":
      return try evalWavetable(regularArgs, attributes: attributePairs, mutable: false)
    case "zeros":
      return try evalTensorCreate(regularArgs, fill: .zeros)
    case "ones":
      return try evalTensorCreate(regularArgs, fill: .ones)
    case "full":
      return try evalTensorCreate(regularArgs, fill: .full)
    case "randn":
      return try evalTensorCreate(regularArgs, fill: .randn)
    case "tensor-param":
      return try evalTensorParam(regularArgs, attributes: attributePairs)
    case "wavetable-param":
      return try evalWavetable(regularArgs, attributes: attributePairs, mutable: true)
    case "audio-tensor":
      return try evalAudioTensor(regularArgs, attributes: attributePairs, kind: "audio")
    case "ir":
      return try evalAudioTensor(regularArgs, attributes: attributePairs, kind: "ir")

    // Tensor ops
    case "matmul":
      return try evalMatmul(regularArgs)
    case "peek":
      return try evalPeek(regularArgs)
    case "peek-row", "peekrow":
      return try evalPeekRow(regularArgs)
    case "sample":
      return try evalSample(regularArgs)
    case "to-signal", "tosignal":
      return try evalToSignal(regularArgs, attributes: attributePairs)

    // Tensor shape ops
    case "reshape":
      return try evalReshape(regularArgs, attributes: attributePairs)
    case "transpose":
      return try evalTranspose(regularArgs, attributes: attributePairs)
    case "shrink":
      return try evalShrink(regularArgs, attributes: attributePairs)
    case "pad":
      return try evalPad(regularArgs, attributes: attributePairs)
    case "expand":
      return try evalExpand(regularArgs, attributes: attributePairs)
    case "repeat":
      return try evalRepeat(regularArgs, attributes: attributePairs)
    case "conv2d":
      return try evalConv2d(regularArgs)
    case "conv1d":
      return try evalConv1d(regularArgs)
    case "windows":
      return try evalWindows(regularArgs, attributes: attributePairs)

    // Reductions
    case "sum":
      return try evalSum(regularArgs, attributes: attributePairs)
    case "mean":
      return try evalMean(regularArgs, attributes: attributePairs)
    case "max-axis", "maxaxis":
      return try evalMaxAxis(regularArgs, attributes: attributePairs)
    case "sum-axis", "sumaxis":
      return try evalSumAxis(regularArgs, attributes: attributePairs)
    case "mean-axis", "meanaxis":
      return try evalMeanAxis(regularArgs, attributes: attributePairs)
    case "softmax":
      return try evalSoftmax(regularArgs, attributes: attributePairs)

    // FFT
    case "fft":
      return try evalFFT(regularArgs, attributes: attributePairs)
    case "ifft":
      return try evalIFFT(regularArgs, attributes: attributePairs)
    case "polar-fft", "polarfft":
      return try evalPolarFFT(regularArgs)
    case "rect-fft", "rectfft":
      return try evalRectFFT(regularArgs)
    case "complex-mul", "complexmul":
      return try evalComplexMul(regularArgs)
    case "complex-conj", "complexconj":
      return try evalComplexConj(regularArgs)
    case "hann":
      return try evalHann(regularArgs, attributes: attributePairs)
    case "window":
      return try evalWindow(regularArgs, attributes: attributePairs)
    case "spectrum-delay", "spectrumdelay":
      return try evalSpectrumDelay(regularArgs, attributes: attributePairs)
    case "spectrum-delay-mod", "spectrumdelaymod":
      return try evalSpectrumDelayMod(regularArgs, attributes: attributePairs)
    case "phase-vocoder", "phasevocoder":
      return try evalPhaseVocoder(regularArgs, attributes: attributePairs)
    case "partition-ir", "partitionir":
      return try evalPartitionIR(regularArgs, attributes: attributePairs)
    case "partitioned-spectral-mac", "partitionedspectralmac":
      return try evalPartitionedSpectralMAC(regularArgs, attributes: attributePairs)
    case "partitioned-convolve", "partitionedconvolve":
      return try evalPartitionedConvolve(regularArgs, attributes: attributePairs)

    // Windowing
    case "buffer":
      return try evalBuffer(regularArgs)
    case "overlap-add", "overlapadd":
      return try evalOverlapAdd(regularArgs)

    // Utility
    case "tuple":
      return try evalTuple(regularArgs)
    case "scale":
      return try evalScale(regularArgs)
    case "triangle":
      return try evalTriangle(regularArgs)
    case "wrap":
      return try evalWrap(regularArgs)
    case "clip":
      return try evalClip(regularArgs)
    case "gswitch":
      return try evalGswitch(regularArgs)
    case "selector":
      return try evalSelector(regularArgs)
    case "__modulated-param":
      return try evalModulatedParam(regularArgs, attributes: attributePairs)

    default:
      throw LispError.unknownOperator(opName)
    }
  }

  // MARK: - Arithmetic

  private func evalTuple(_ args: [ASTNode]) throws -> EvalResult {
    guard !args.isEmpty else {
      throw LispError.invalidArgument("tuple requires at least 1 argument")
    }
    return .tuple(try args.map { try evaluateAST($0) })
  }

  private func evalBinaryArith(_ args: [ASTNode], op: String) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("\(op) requires 2 arguments, got \(args.count)")
    }
    let lhs = try evaluateAST(args[0])
    let rhs = try evaluateAST(args[1])
    return try applyBinaryOp(lhs, rhs, op: op)
  }

  private func applyBinaryOp(_ lhs: EvalResult, _ rhs: EvalResult, op: String) throws -> EvalResult
  {
    // Promote floats to signals for operations
    let l = promoteToValue(lhs)
    let r = promoteToValue(rhs)

    switch (l, r) {
    case (.float(let a), .float(let b)):
      switch op {
      case "+": return .float(a + b)
      case "-": return .float(a - b)
      case "*": return .float(a * b)
      case "/": return .float(a / b)
      case "min": return .float(Swift.min(a, b))
      case "max": return .float(Swift.max(a, b))
      default: throw LispError.unknownOperator(op)
      }

    case (.signal(let a), .signal(let b)):
      switch op {
      case "+": return .signal(a + b)
      case "-": return .signal(a - b)
      case "*": return .signal(a * b)
      case "/": return .signal(a / b)
      case "min": return .signal(DGenLazy.min(a, b))
      case "max": return .signal(DGenLazy.max(a, b))
      default: throw LispError.unknownOperator(op)
      }

    case (.tensor(let a), .tensor(let b)):
      switch op {
      case "+": return .tensor(a + b)
      case "-": return .tensor(a - b)
      case "*": return .tensor(a * b)
      case "/": return .tensor(a / b)
      case "min": return .tensor(DGenLazy.min(a, b))
      case "max": return .tensor(DGenLazy.max(a, b))
      default: throw LispError.unknownOperator(op)
      }

    case (.signal(let a), .tensor(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      case "/": return .signalTensor(a / b)
      default: throw LispError.unknownOperator(op)
      }

    case (.tensor(let a), .signal(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      case "/": return .signalTensor(a / b)
      default: throw LispError.unknownOperator(op)
      }

    case (.signalTensor(let a), .signalTensor(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      case "/": return .signalTensor(a / b)
      default: throw LispError.unknownOperator(op)
      }

    case (.signalTensor(let a), .signal(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      case "/": return .signalTensor(a / b)
      default: throw LispError.typeError("Unsupported op \(op) for signalTensor op signal")
      }

    case (.signal(let a), .signalTensor(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      case "/": return .signalTensor(a / b)
      default: throw LispError.typeError("Unsupported op \(op) for signal op signalTensor")
      }

    case (.signalTensor(let a), .tensor(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      default: throw LispError.typeError("Unsupported op \(op) for signalTensor op tensor")
      }

    case (.tensor(let a), .signalTensor(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      default: throw LispError.typeError("Unsupported op \(op) for tensor op signalTensor")
      }

    case (.signal(let a), .float(let b)):
      switch op {
      case "+": return .signal(a + b)
      case "-": return .signal(a - b)
      case "*": return .signal(a * b)
      case "/": return .signal(a / b)
      case "min": return .signal(DGenLazy.min(a, Double(b)))
      case "max": return .signal(DGenLazy.max(a, Double(b)))
      default: throw LispError.unknownOperator(op)
      }

    case (.float(let a), .signal(let b)):
      switch op {
      case "+": return .signal(b + a)
      case "-": return .signal(Signal.constant(a) - b)
      case "*": return .signal(b * a)
      case "/": return .signal(Signal.constant(a) / b)
      case "min": return .signal(DGenLazy.min(b, Double(a)))
      case "max": return .signal(DGenLazy.max(b, Double(a)))
      default: throw LispError.unknownOperator(op)
      }

    case (.tensor(let a), .float(let b)):
      switch op {
      case "+": return .tensor(a + b)
      case "-": return .tensor(a - b)
      case "*": return .tensor(a * b)
      case "/": return .tensor(a / b)
      case "min": return .tensor(DGenLazy.min(a, Double(b)))
      case "max": return .tensor(DGenLazy.max(a, Double(b)))
      default: throw LispError.unknownOperator(op)
      }

    case (.float(let a), .tensor(let b)):
      switch op {
      case "+": return .tensor(b + a)
      case "-": return .tensor(Tensor([a]) - b)
      case "*": return .tensor(b * a)
      case "/": return .tensor(Tensor([a]) / b)
      case "min": return .tensor(DGenLazy.min(b, Double(a)))
      case "max": return .tensor(DGenLazy.max(b, Double(a)))
      default: throw LispError.unknownOperator(op)
      }

    case (.signalTensor(let a), .float(let b)):
      switch op {
      case "+": return .signalTensor(a + b)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(a * b)
      case "/": return .signalTensor(a / b)
      case "min": return .signalTensor(DGenLazy.min(a, Double(b)))
      case "max": return .signalTensor(DGenLazy.max(a, Double(b)))
      default: throw LispError.typeError("Unsupported op \(op) for signalTensor op float")
      }

    case (.float(let a), .signalTensor(let b)):
      switch op {
      case "+": return .signalTensor(b + a)
      case "-": return .signalTensor(a - b)
      case "*": return .signalTensor(b * a)
      case "/": return .signalTensor(a / b)
      case "min": return .signalTensor(DGenLazy.min(Double(a), b))
      case "max": return .signalTensor(DGenLazy.max(Double(a), b))
      default: throw LispError.typeError("Unsupported op \(op) for float op signalTensor")
      }

    default:
      throw LispError.typeError("Cannot apply \(op) to given types")
    }
  }

  private func evalUnaryNegate(_ arg: ASTNode) throws -> EvalResult {
    let val = try evaluateAST(arg)
    switch promoteToValue(val) {
    case .float(let f): return .float(-f)
    case .signal(let s): return .signal(-s)
    case .tensor(let t): return .tensor(-t)
    case .signalTensor(let st): return .signalTensor(-st)
    default: throw LispError.typeError("Cannot negate this type")
    }
  }

  // MARK: - Unary math

  private func evalUnaryMath(_ args: [ASTNode], fn: String) throws -> EvalResult {
    guard args.count == 1 else {
      throw LispError.invalidArgument("\(fn) requires 1 argument")
    }
    let val = try evaluateAST(args[0])
    return try applyUnaryMath(promoteToValue(val), fn: fn)
  }

  private func applyUnaryMath(_ val: EvalResult, fn: String) throws -> EvalResult {
    switch val {
    case .float(let f):
      switch fn {
      case "sin": return .float(Foundation.sin(f))
      case "cos": return .float(Foundation.cos(f))
      case "tan": return .float(Foundation.tan(f))
      case "atan": return .float(Foundation.atan(f))
      case "tanh": return .float(Foundation.tanh(f))
      case "exp": return .float(Foundation.exp(f))
      case "log": return .float(Foundation.log(f))
      case "log10": return .float(Foundation.log10(f))
      case "sqrt": return .float(Foundation.sqrt(f))
      case "abs": return .float(Swift.abs(f))
      case "sign": return .float(f > 0 ? 1 : (f < 0 ? -1 : 0))
      case "floor": return .float(Foundation.floor(f))
      case "ceil": return .float(Foundation.ceil(f))
      case "round": return .float(Foundation.round(f))
      case "relu": return .float(Swift.max(f, 0))
      case "sigmoid": return .float(1.0 / (1.0 + Foundation.exp(-f)))
      default: throw LispError.unknownOperator(fn)
      }

    case .signal(let s):
      switch fn {
      case "sin": return .signal(DGenLazy.sin(s))
      case "cos": return .signal(DGenLazy.cos(s))
      case "tan": return .signal(DGenLazy.tan(s))
      case "atan": return .signal(DGenLazy.atan(s))
      case "tanh": return .signal(DGenLazy.tanh(s))
      case "exp": return .signal(DGenLazy.exp(s))
      case "log": return .signal(DGenLazy.log(s))
      case "log10": return .signal(DGenLazy.log10(s))
      case "sqrt": return .signal(DGenLazy.sqrt(s))
      case "abs": return .signal(DGenLazy.abs(s))
      case "sign": return .signal(DGenLazy.sign(s))
      case "floor": return .signal(s)  // floor not available for Signal, pass through
      case "ceil": return .signal(s)
      case "round": return .signal(s)
      case "relu": return .signal(DGenLazy.relu(s))
      case "sigmoid": return .signal(DGenLazy.sigmoid(s))
      default: throw LispError.unknownOperator(fn)
      }

    case .tensor(let t):
      switch fn {
      case "sin": return .tensor(DGenLazy.sin(t))
      case "cos": return .tensor(DGenLazy.cos(t))
      case "tan": return .tensor(DGenLazy.tan(t))
      case "atan": return .tensor(DGenLazy.atan(t))
      case "tanh": return .tensor(DGenLazy.tanh(t))
      case "exp": return .tensor(DGenLazy.exp(t))
      case "log": return .tensor(DGenLazy.log(t))
      case "log10": return .tensor(DGenLazy.log10(t))
      case "sqrt": return .tensor(DGenLazy.sqrt(t))
      case "abs": return .tensor(DGenLazy.abs(t))
      case "sign": return .tensor(DGenLazy.sign(t))
      case "floor": return .tensor(DGenLazy.floor(t))
      case "ceil": return .tensor(DGenLazy.ceil(t))
      case "round": return .tensor(DGenLazy.round(t))
      case "relu": return .tensor(DGenLazy.relu(t))
      case "sigmoid": return .tensor(DGenLazy.sigmoid(t))
      default: throw LispError.unknownOperator(fn)
      }

    case .signalTensor(let st):
      switch fn {
      case "sin": return .signalTensor(DGenLazy.sin(st))
      case "cos": return .signalTensor(DGenLazy.cos(st))
      case "atan": return .signalTensor(DGenLazy.atan(st))
      case "exp": return .signalTensor(DGenLazy.exp(st))
      case "log": return .signalTensor(DGenLazy.log(st))
      case "log10": return .signalTensor(DGenLazy.log10(st))
      case "sqrt": return .signalTensor(DGenLazy.sqrt(st))
      case "abs": return .signalTensor(DGenLazy.abs(st))
      case "sign": return .signalTensor(DGenLazy.sign(st))
      case "tanh": return .signalTensor(DGenLazy.tanh(st))
      case "relu": return .signalTensor(DGenLazy.relu(st))
      default: throw LispError.typeError("\(fn) not available for SignalTensor")
      }

    default:
      throw LispError.typeError("Cannot apply \(fn) to this type")
    }
  }

  // MARK: - Binary math

  private func evalPow(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("pow requires 2 arguments")
    }
    let base = try promoteToValue(evaluateAST(args[0]))
    let exp = try promoteToValue(evaluateAST(args[1]))

    switch (base, exp) {
    case (.float(let a), .float(let b)):
      return .float(Foundation.pow(a, b))
    case (.signal(let a), .signal(let b)):
      return .signal(DGenLazy.pow(a, b))
    case (.signal(let a), .float(let b)):
      return .signal(DGenLazy.pow(a, b))
    case (.float(let a), .signal(let b)):
      return .signal(DGenLazy.pow(a, b))
    case (.tensor(let a), .tensor(let b)):
      return .tensor(DGenLazy.pow(a, b))
    case (.tensor(let a), .float(let b)):
      return .tensor(DGenLazy.pow(a, b))
    case (.float(let a), .tensor(let b)):
      return .tensor(DGenLazy.pow(a, b))
    case (.signal(let a), .tensor(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.tensor(let a), .signal(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.signalTensor(let a), .signalTensor(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.signalTensor(let a), .signal(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.signal(let a), .signalTensor(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.signalTensor(let a), .tensor(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.tensor(let a), .signalTensor(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.signalTensor(let a), .float(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    case (.float(let a), .signalTensor(let b)):
      return .signalTensor(DGenLazy.pow(a, b))
    default:
      throw LispError.typeError("pow: unsupported type combination")
    }
  }

  private func evalMod(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("% requires 2 arguments")
    }
    let a = try promoteToValue(evaluateAST(args[0]))
    let b = try promoteToValue(evaluateAST(args[1]))

    switch (a, b) {
    case (.float(let x), .float(let y)):
      return .float(x.truncatingRemainder(dividingBy: y))
    case (.signal(let x), .signal(let y)):
      return .signal(DGenLazy.mod(x, y))
    case (.signal(let x), .float(let y)):
      return .signal(DGenLazy.mod(x, Double(y)))
    default:
      throw LispError.typeError("%: unsupported type combination")
    }
  }

  private func evalAtan2(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("atan2 requires 2 arguments (y, x)")
    }
    let y = try promoteToValue(evaluateAST(args[0]))
    let x = try promoteToValue(evaluateAST(args[1]))
    switch (y, x) {
    case (.float(let a), .float(let b)):
      return .float(Foundation.atan2(a, b))
    case (.signal(let a), .signal(let b)):
      return .signal(DGenLazy.atan2(a, b))
    case (.tensor(let a), .tensor(let b)):
      return .tensor(DGenLazy.atan2(a, b))
    case (.signalTensor(let a), .signalTensor(let b)):
      return .signalTensor(DGenLazy.atan2(a, b))
    default:
      throw LispError.typeError("atan2 operands must have matching scalar/tensor kind")
    }
  }

  // MARK: - Comparison

  private func evalComparison(_ args: [ASTNode], op: String) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("\(op) requires 2 arguments")
    }
    let lhs = try promoteToValue(evaluateAST(args[0]))
    let rhs = try promoteToValue(evaluateAST(args[1]))

    switch (lhs, rhs) {
    case (.signal(let a), .signal(let b)):
      switch op {
      case "gt": return .signal(a > b)
      case "lt": return .signal(a < b)
      case "gte": return .signal(a >= b)
      case "lte": return .signal(a <= b)
      case "eq": return .signal(a.eq(b))
      default: throw LispError.unknownOperator(op)
      }
    case (.signal(let a), .float(let b)):
      switch op {
      case "gt": return .signal(a > Double(b))
      case "lt": return .signal(a < Double(b))
      case "gte": return .signal(a >= Double(b))
      case "lte": return .signal(a <= Double(b))
      case "eq": return .signal(a.eq(b))
      default: throw LispError.unknownOperator(op)
      }
    case (.float(let a), .signal(let b)):
      switch op {
      case "gt": return .signal(Double(a) > b)
      case "lt": return .signal(Double(a) < b)
      case "gte": return .signal(Double(a) >= b)
      case "lte": return .signal(Double(a) <= b)
      case "eq": return .signal(b.eq(a))
      default: throw LispError.unknownOperator(op)
      }
    case (.float(let a), .float(let b)):
      switch op {
      case "gt": return .float(a > b ? 1 : 0)
      case "lt": return .float(a < b ? 1 : 0)
      case "gte": return .float(a >= b ? 1 : 0)
      case "lte": return .float(a <= b ? 1 : 0)
      case "eq": return .float(a == b ? 1 : 0)
      default: throw LispError.unknownOperator(op)
      }
    case (.tensor(let a), .tensor(let b)):
      switch op {
      case "gt": return .tensor(a > b)
      case "lt": return .tensor(a < b)
      case "gte": return .tensor(a >= b)
      case "lte": return .tensor(a <= b)
      case "eq": return .tensor(a.eq(b))
      default: throw LispError.unknownOperator(op)
      }
    default:
      throw LispError.typeError("Comparison \(op): unsupported type combination")
    }
  }

  // MARK: - Signal generators

  private func evalPhasor(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("phasor requires at least 1 argument (freq)")
    }
    let freqResult = try evaluateAST(args[0])
    let reset: Signal? = args.count >= 2 ? try requireSignal(evaluateAST(args[1])) : nil

    switch promoteToValue(freqResult) {
    case .signal(let freq):
      return .signal(Signal.phasor(freq, reset: reset))
    case .float(let freq):
      return .signal(Signal.phasor(freq, reset: reset))
    case .tensor(let freqs):
      return .signalTensor(Signal.phasor(freqs, reset: reset))
    default:
      throw LispError.typeError("phasor: freq must be signal, float, or tensor")
    }
  }

  private func evalNoise(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.isEmpty else {
      throw LispError.invalidArgument(
        "noise takes attributes only: (noise), (noise @size N), (noise @size N @hop H)")
    }
    let size = Int(attrValue(attributes, "@size") ?? "1") ?? 1
    let hop =
      attrValue(attributes, "@hop").flatMap { Int($0) }
      ?? attrValue(attributes, "@hopSize").flatMap { Int($0) }
    if size <= 1 {
      return .signal(Signal.noise())
    }
    return .signalTensor(DGenLazy.tensorNoise(size: size, hop: hop))
  }

  // MARK: - Stateful

  private func evalAccum(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("accum requires at least 1 argument (increment)")
    }
    let inc = try requireSignal(evaluateAST(args[0]))
    let reset: Signal? = args.count >= 2 ? try asSignalOrNil(evaluateAST(args[1])) : nil
    let minVal: Signal? = args.count >= 3 ? try asSignalOrNil(evaluateAST(args[2])) : nil
    let maxVal: Signal? = args.count >= 4 ? try asSignalOrNil(evaluateAST(args[3])) : nil
    return .signal(Signal.accum(inc, reset: reset, min: minVal, max: maxVal))
  }

  private func evalLatch(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("latch requires 2 arguments (value, trigger)")
    }
    let value = try requireSignal(evaluateAST(args[0]))
    let trigger = try requireSignal(evaluateAST(args[1]))
    return .signal(Signal.latch(value, when: trigger))
  }

  private func evalHopHold(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("hop-hold requires 2 arguments (value, hop)")
    }
    let value = try evaluateAST(args[0])
    let hop = Int(try requireFloat(evaluateAST(args[1])))
    switch value {
    case .signal(let s): return .signal(s.hopHold(hop: hop))
    case .tensor(let t): return .signalTensor(t.hopHold(hop: hop))
    case .signalTensor(let st): return .signalTensor(st.hopHold(hop: hop))
    default: throw LispError.typeError("hop-hold: value must be signal, tensor, or signalTensor")
    }
  }

  private func evalMix(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 3 else {
      throw LispError.invalidArgument("mix requires 3 arguments (a, b, t)")
    }
    let a = try requireSignal(evaluateAST(args[0]))
    let b = try requireSignal(evaluateAST(args[1]))
    let tResult = try promoteToValue(evaluateAST(args[2]))
    switch tResult {
    case .signal(let t): return .signal(Signal.mix(a, b, t))
    case .float(let t): return .signal(Signal.mix(a, b, t))
    default: throw LispError.typeError("mix: t must be signal or float")
    }
  }

  // MARK: - Effects

  private func evalBiquad(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("biquad requires at least 1 argument (signal)")
    }
    let sigResult = try evaluateAST(args[0])

    // Parse biquad params from remaining args or attributes — accept signals or floats
    let cutoff: Signal =
      args.count >= 2
      ? try requireSignal(evaluateAST(args[1]))
      : Signal.constant(Float(attrValue(attributes, "@cutoff") ?? "1000") ?? 1000)
    let q: Signal =
      args.count >= 3
      ? try requireSignal(evaluateAST(args[2]))
      : Signal.constant(Float(attrValue(attributes, "@q") ?? "0.707") ?? 0.707)
    let gain: Signal =
      args.count >= 4
      ? try requireSignal(evaluateAST(args[3]))
      : Signal.constant(Float(attrValue(attributes, "@gain") ?? "0") ?? 0)
    let mode: Signal =
      args.count >= 5
      ? try requireSignal(evaluateAST(args[4]))
      : Signal.constant(Float(attrValue(attributes, "@mode") ?? "0") ?? 0)

    switch sigResult {
    case .signal(let sig):
      return .signal(sig.biquad(cutoff: cutoff, resonance: q, gain: gain, mode: mode))
    case .signalTensor(let st):
      return .signalTensor(st.biquad(cutoff: cutoff, resonance: q, gain: gain, mode: mode))
    default:
      let sig = try requireSignal(sigResult)
      return .signal(sig.biquad(cutoff: cutoff, resonance: q, gain: gain, mode: mode))
    }
  }

  private func evalDelay(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("delay requires 2 arguments (signal, time)")
    }
    let sig = try requireSignal(evaluateAST(args[0]))
    let timeResult = try promoteToValue(evaluateAST(args[1]))
    switch timeResult {
    case .signal(let t): return .signal(sig.delay(t))
    case .float(let t): return .signal(sig.delay(t))
    default: throw LispError.typeError("delay: time must be signal or float")
    }
  }

  // MARK: - I/O

  private func evalParam(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    // First regular arg is the name
    guard args.count >= 1, case .atom(let name) = args[0] else {
      throw LispError.invalidArgument("param requires a name: (param name @default 440)")
    }

    let defaultVal = Float(attrValue(attributes, "@default") ?? "0") ?? 0
    let minVal = Float(attrValue(attributes, "@min") ?? "")
    let maxVal = Float(attrValue(attributes, "@max") ?? "")
    let unit = attrValue(attributes, "@unit")
    let hidden = parseBoolAttr(attributes, "@hidden")
    let group = try parseUIMetadataSymbol(attrValue(attributes, "@group"), attribute: "@group", paramName: name)
    let env = try parseUIMetadataSymbol(attrValue(attributes, "@env"), attribute: "@env", paramName: name)
    let role = try parseUIEnvelopeRole(attrValue(attributes, "@role"), paramName: name)
    let generatedKind = attrValue(attributes, "@generated")
    let generatedFor = attrValue(attributes, "@generated-for")
    let modulationMode = attrValue(attributes, "@mod-mode").flatMap {
      ModulationMode(rawValue: $0.lowercased())
    }
    let modulationDepthMin = Float(attrValue(attributes, "@mod-depth-min") ?? "")
    let modulationDepthMax = Float(attrValue(attributes, "@mod-depth-max") ?? "")
    let modulationActiveParamName = attrValue(attributes, "@mod-active-param")
    let modulationResolvedSymbolName = attrValue(attributes, "@mod-resolved-symbol")
    let generatedModulatorSlot = Int(attrValue(attributes, "@modulator-slot") ?? "")

    let signal = Signal.param(defaultVal, min: minVal, max: maxVal)

    let info = ParamInfo(
      name: name,
      cellId: signal.memoryCellId,
      defaultValue: defaultVal,
      min: minVal,
      max: maxVal,
      unit: unit,
      hidden: hidden,
      group: group,
      env: env,
      role: role,
      generatedKind: generatedKind,
      generatedFor: generatedFor,
      modulationMode: modulationMode,
      modulationDepthMin: modulationDepthMin,
      modulationDepthMax: modulationDepthMax,
      modulationActiveParamName: modulationActiveParamName,
      modulationResolvedSymbolName: modulationResolvedSymbolName,
      generatedModulatorSlot: generatedModulatorSlot
    )
    params.append(info)
    definitions[name] = .signal(signal)

    return .signal(signal)
  }

  private func evalInput(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    // First arg is channel number (1-indexed in lisp)
    let channelLisp: Int
    if args.count >= 1 {
      let val = try requireFloat(evaluateAST(args[0]))
      channelLisp = Int(val)
    } else {
      channelLisp = 1
    }
    let channel = channelLisp - 1  // Convert to 0-indexed

    let name = attrValue(attributes, "@name")
    let modulatorSlot = Int(attrValue(attributes, "@modulator") ?? "")
    inputs.append(InputInfo(channel: channel, name: name, modulatorSlot: modulatorSlot))

    return .signal(Signal.input(channel))
  }

  private func evalOutput(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("out requires at least 1 argument (signal)")
    }
    let signal = try requireSignal(evaluateAST(args[0]))

    // Second arg is channel number (1-indexed)
    let channelLisp: Int
    if args.count >= 2 {
      let val = try requireFloat(evaluateAST(args[1]))
      channelLisp = Int(val)
    } else {
      channelLisp = 1
    }
    let channel = channelLisp - 1

    let name = attrValue(attributes, "@name")
    let modulatorSlot = try parseOptionalPositiveIntAttribute(attributes, "@modulator")
    if let modulatorSlot,
      outputs.contains(where: { $0.modulatorSlot == modulatorSlot })
    {
      throw LispError.invalidArgument("duplicate output @modulator slot \(modulatorSlot)")
    }
    outputs.append(
      OutputInfo(channel: channel, signal: signal, name: name, modulatorSlot: modulatorSlot)
    )

    return .none
  }

  // MARK: - Tensor ops

  private func evalTensor(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    if let dataStr = attrValue(attributes, "@data") {
      let shape = try parseShapeFromArgsOrAttributes(args, attributes: attributes, op: "tensor")
      let data = parseFloatList(dataStr)
      guard data.count == shape.reduce(1, *) else {
        throw LispError.invalidArgument(
          "tensor @data has \(data.count) values, expected \(shape.reduce(1, *))")
      }
      return .tensor(makeTensor(shape: shape, data: data, mutable: false))
    }
    if attrValue(attributes, "@file") != nil || attrValue(attributes, "@shape") != nil {
      return try evalWavetable(args, attributes: attributes, mutable: false)
    }
    guard args.count >= 2 else {
      throw LispError.invalidArgument("tensor requires at least 2 arguments (rows, cols)")
    }
    let rows = Int(try requireFloat(evaluateAST(args[0])))
    let cols = Int(try requireFloat(evaluateAST(args[1])))
    return .tensor(Tensor.zeros([rows, cols]))
  }

  private func evalAudioTensor(
    _ args: [ASTNode],
    attributes: [(name: String, value: String)],
    kind: String
  ) throws -> EvalResult {
    guard args.isEmpty else {
      throw LispError.invalidArgument("\(kind) takes attributes only")
    }
    guard let fileAttr = attrValue(attributes, "@file") else {
      throw LispError.invalidArgument("\(kind) requires @file")
    }
    let file = unquote(fileAttr)
    let url = URL(fileURLWithPath: file, relativeTo: sourceDirectory).standardizedFileURL
    let requestedChannel = attrValue(attributes, "@channel").flatMap { Int($0) }
    let mono =
      requestedChannel == nil
      ? (attrValue(attributes, "@mono").map { parseBoolString($0, defaultValue: true) } ?? true)
      : false
    let loaded: (samples: [Float], sampleRate: Float)
    do {
      loaded = try AudioFile.load(url: url, mono: mono)
    } catch {
      throw LispError.invalidArgument(
        "failed to load audio tensor '\(file)' relative to \(sourceDirectory.path): \(error)")
    }

    let originalSamples: [Float]
    if let requestedChannel {
      let channelCount = try wavChannelCount(url: url)
      guard requestedChannel >= 0 && requestedChannel < channelCount else {
        throw LispError.invalidArgument(
          "audio-tensor @channel \(requestedChannel) out of range for \(channelCount) channels")
      }
      originalSamples = stride(from: requestedChannel, to: loaded.samples.count, by: channelCount)
        .map {
          loaded.samples[$0]
        }
    } else {
      originalSamples = loaded.samples
    }
    let startFrame =
      attrValue(attributes, "@start").flatMap(Double.init)
      .map { max(0, Int($0 * Double(loaded.sampleRate))) } ?? 0
    let endFrame =
      attrValue(attributes, "@end").flatMap(Double.init)
      .map { max(0, Int($0 * Double(loaded.sampleRate))) } ?? originalSamples.count
    let clampedStart = min(startFrame, originalSamples.count)
    let clampedEnd = min(max(endFrame, clampedStart), originalSamples.count)
    var samples = Array(originalSamples[clampedStart..<clampedEnd])
    if attrValue(attributes, "@normalize")?.lowercased() == "peak" {
      let peak = samples.map { Swift.abs($0) }.max() ?? 0
      if peak > 0 {
        samples = samples.map { $0 / peak }
      }
    }
    let tensor = Tensor(samples)
    tensors.append(
      TensorInfo(
        name: attrValue(attributes, "@name").map(unquote) ?? "",
        shape: [samples.count],
        kind: kind,
        mutable: false,
        sourceFile: file,
        sourceSampleRate: loaded.sampleRate,
        data: samples
      ))
    return .tensor(tensor)
  }

  private func evalMatmul(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("matmul requires 2 arguments")
    }
    let a = try requireTensor(evaluateAST(args[0]))
    let b = try requireTensor(evaluateAST(args[1]))
    return .tensor(a.matmul(b))
  }

  private func evalPeek(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count >= 2 else {
      throw LispError.invalidArgument("peek requires at least 2 arguments (tensor, index)")
    }
    let tensorResult = try evaluateAST(args[0])
    let indexResult = try evaluateAST(args[1])

    let tensor: Tensor
    switch tensorResult {
    case .tensor(let t): tensor = t
    default: throw LispError.typeError("peek: first argument must be a tensor")
    }

    let index = try requireSignal(coerceToSignal(indexResult))

    if args.count >= 3 {
      let channel = try requireSignal(coerceToSignal(evaluateAST(args[2])))
      return .signal(tensor.peek(index, channel: channel))
    } else {
      return .signal(tensor.peek(index))
    }
  }

  private func evalReshape(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("reshape requires at least 1 argument (tensor)")
    }
    let val = try evaluateAST(args[0])

    // Parse shape from @shape attribute like [2,3]
    guard let shapeStr = attrValue(attributes, "@shape") else {
      throw LispError.invalidArgument("reshape requires @shape attribute")
    }
    let shape = parseShape(shapeStr)

    switch val {
    case .tensor(let t): return .tensor(t.reshape(shape))
    case .signalTensor(let st): return .signalTensor(st.reshape(shape))
    default: throw LispError.typeError("reshape: argument must be a tensor or signalTensor")
    }
  }

  // MARK: - MSE

  private func evalMse(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("mse requires 2 arguments (prediction, target)")
    }
    let a = try evaluateAST(args[0])
    let b = try evaluateAST(args[1])
    switch (a, b) {
    case (.tensor(let t1), .tensor(let t2)):
      return .tensor(DGenLazy.mse(t1, t2))
    case (.signal(let s1), .signal(let s2)):
      return .signal(DGenLazy.mse(s1, s2))
    default:
      throw LispError.typeError("mse: both arguments must be same type (signal or tensor)")
    }
  }

  // MARK: - Stateful phasor

  private func evalStatefulPhasor(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("stateful-phasor requires at least 1 argument (freq)")
    }
    let freqResult = try evaluateAST(args[0])
    let reset: Signal? = args.count >= 2 ? try requireSignal(evaluateAST(args[1])) : nil

    switch promoteToValue(freqResult) {
    case .signal(let freq):
      return .signal(Signal.statefulPhasor(freq, reset: reset))
    case .float(let freq):
      return .signal(Signal.statefulPhasor(Signal.constant(freq), reset: reset))
    case .tensor(let freqs):
      return .signalTensor(Signal.statefulPhasor(freqs, reset: reset))
    case .signalTensor(let freqs):
      return .signalTensor(Signal.statefulPhasor(freqs, reset: reset))
    default:
      throw LispError.typeError(
        "stateful-phasor: freq must be signal, float, tensor, or signalTensor")
    }
  }

  // MARK: - Compressor

  private func evalCompressor(
    _ args: [ASTNode], rawArgs: [ASTNode] = [], attributes: [(name: String, value: String)]
  ) throws -> EvalResult {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("compressor requires at least 1 argument (signal)")
    }
    let sigResult = try evaluateAST(args[0])

    // Accept signals or floats for all parameters
    let ratio: Signal =
      args.count >= 2
      ? try requireSignal(evaluateAST(args[1]))
      : Signal.constant(Float(attrValue(attributes, "@ratio") ?? "4") ?? 4)
    let threshold: Signal =
      args.count >= 3
      ? try requireSignal(evaluateAST(args[2]))
      : Signal.constant(Float(attrValue(attributes, "@threshold") ?? "-20") ?? -20)
    let knee: Signal =
      args.count >= 4
      ? try requireSignal(evaluateAST(args[3]))
      : Signal.constant(Float(attrValue(attributes, "@knee") ?? "6") ?? 6)
    let attack: Signal =
      args.count >= 5
      ? try requireSignal(evaluateAST(args[4]))
      : Signal.constant(Float(attrValue(attributes, "@attack") ?? "0.01") ?? 0.01)
    let release: Signal =
      args.count >= 6
      ? try requireSignal(evaluateAST(args[5]))
      : Signal.constant(Float(attrValue(attributes, "@release") ?? "0.1") ?? 0.1)

    // Sidechain detection:
    // 8-arg form: (compressor sig ratio threshold knee attack release isSidechain sidechain)
    // 7-arg form: (compressor sig ratio threshold knee attack release sidechain)
    // attribute:  @sidechain varname  or  @sidechain (expr)
    if args.count >= 8 {
      let isSideChain = try requireSignal(evaluateAST(args[6]))
      let sidechain = try requireSignal(evaluateAST(args[7]))
      switch sigResult {
      case .signal(let sig):
        return .signal(
          sig.compressor(
            ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release,
            isSideChain: isSideChain, sidechain: sidechain))
      case .signalTensor(let st):
        return .signalTensor(
          st.compressor(
            ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release,
            isSideChain: isSideChain, sidechain: sidechain))
      default:
        throw LispError.typeError("compressor: first argument must be signal or signalTensor")
      }
    }

    let sidechain: Signal?
    if args.count >= 7 {
      sidechain = try requireSignal(evaluateAST(args[6]))
    } else if let sidechainName = attrValue(attributes, "@sidechain"), !sidechainName.isEmpty {
      sidechain = try requireSignal(evaluateAST(.atom(sidechainName)))
    } else if attrValue(attributes, "@sidechain") != nil {
      // @sidechain followed by a non-atom expression — scan raw args
      var found: Signal? = nil
      for (idx, arg) in rawArgs.enumerated() {
        if case .atom(let v) = arg, v == "@sidechain", idx + 1 < rawArgs.count {
          found = try requireSignal(evaluateAST(rawArgs[idx + 1]))
          break
        }
      }
      sidechain = found
    } else {
      sidechain = nil
    }

    switch sigResult {
    case .signal(let sig):
      return .signal(
        sig.compressor(
          ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release,
          sidechain: sidechain))
    case .signalTensor(let st):
      return .signalTensor(
        st.compressor(
          ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release,
          sidechain: sidechain))
    default:
      throw LispError.typeError("compressor: first argument must be signal or signalTensor")
    }
  }

  // MARK: - Tensor creation

  private enum TensorFill { case zeros, ones, full, randn }

  private func evalTensorCreate(_ args: [ASTNode], fill: TensorFill) throws -> EvalResult {
    guard args.count >= 1 else {
      throw LispError.invalidArgument(
        "\(fill) requires at least 1 argument (shape as [d1,d2,...] or individual dims)")
    }

    let shape: [Int]
    // If first arg is a bracket-notation shape like [2,3]
    if case .atom(let str) = args[0], str.hasPrefix("[") {
      shape = parseShape(str)
    } else {
      // Individual dimension args
      shape = try args.prefix(fill == .full ? args.count - 1 : args.count).map {
        Int(try requireFloat(evaluateAST($0)))
      }
    }

    switch fill {
    case .zeros:
      return .tensor(Tensor.zeros(shape))
    case .ones:
      return .tensor(Tensor.ones(shape))
    case .full:
      let value = try requireFloat(evaluateAST(args.last!))
      return .tensor(Tensor.full(shape, value: value))
    case .randn:
      return .tensor(Tensor.randn(shape))
    }
  }

  private func evalTensorParam(_ args: [ASTNode], attributes: [(name: String, value: String)])
    throws -> EvalResult
  {
    if attrValue(attributes, "@file") != nil || attrValue(attributes, "@shape") != nil
      || attrValue(attributes, "@default-file") != nil
    {
      return try evalWavetable(args, attributes: attributes, mutable: true)
    }
    guard args.count >= 1 else {
      throw LispError.invalidArgument("tensor-param requires shape argument")
    }
    let shape = try parseShapeArgs(args)
    return .tensor(Tensor.param(shape))
  }

  private func evalWavetable(
    _ args: [ASTNode], attributes: [(name: String, value: String)], mutable: Bool
  ) throws -> EvalResult {
    let shape = try parseShapeFromArgsOrAttributes(
      args, attributes: attributes, op: mutable ? "wavetable-param" : "wavetable")
    let fileAttr = attrValue(attributes, "@file") ?? attrValue(attributes, "@default-file")
    let sourceFile = fileAttr.map(unquote)
    let data =
      try sourceFile.map { try loadTensorData(file: $0, expectedShape: shape) }
      ?? [Float](repeating: 0, count: shape.reduce(1, *))

    let tensor = makeTensor(shape: shape, data: data, mutable: mutable)
    tensors.append(
      TensorInfo(
        name: attrValue(attributes, "@name").map(unquote) ?? "",
        shape: shape,
        kind: "wavetable",
        mutable: mutable,
        sourceFile: sourceFile,
        sourceSampleRate: nil,
        data: data
      ))
    return .tensor(tensor)
  }

  // MARK: - Tensor sampling

  private func evalPeekRow(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("peek-row requires 2 arguments (tensor, rowIndex)")
    }
    let val = try evaluateAST(args[0])
    let index = try requireSignal(evaluateAST(args[1]))
    switch val {
    case .tensor(let t): return .signalTensor(t.peekRow(index))
    case .signalTensor(let st): return .signalTensor(st.peekRow(index))
    default: throw LispError.typeError("peek-row: first argument must be tensor or signalTensor")
    }
  }

  private func evalSample(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("sample requires 2 arguments (tensor, index)")
    }
    let val = try evaluateAST(args[0])
    let index = try requireSignal(evaluateAST(args[1]))
    switch val {
    case .tensor(let t): return .signalTensor(t.sample(index))
    case .signalTensor(let st): return .signalTensor(st.sample(index))
    default: throw LispError.typeError("sample: first argument must be tensor or signalTensor")
    }
  }

  private func evalToSignal(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("to-signal requires 1 argument (tensor)")
    }
    let t = try requireTensor(evaluateAST(args[0]))
    let maxFrames: Int? = attrValue(attributes, "@max-frames").flatMap { Int($0) }
    return .signal(t.toSignal(maxFrames: maxFrames))
  }

  // MARK: - Tensor shape ops

  private func evalTranspose(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("transpose requires at least 1 argument")
    }
    let val = try evaluateAST(args[0])
    let axes: [Int]? = attrValue(attributes, "@axes").map { parseIntList($0) }
    switch val {
    case .tensor(let t): return .tensor(t.transpose(axes))
    case .signalTensor(let st): return .signalTensor(st.transpose(axes))
    default: throw LispError.typeError("transpose: argument must be tensor or signalTensor")
    }
  }

  private func evalShrink(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("shrink requires at least 1 argument")
    }
    let val = try evaluateAST(args[0])
    guard let rangesStr = attrValue(attributes, "@ranges") else {
      throw LispError.invalidArgument("shrink requires @ranges attribute, e.g. @ranges [0:2,1:3]")
    }
    let ranges = parseRanges(rangesStr)
    switch val {
    case .tensor(let t): return .tensor(t.shrink(ranges))
    case .signalTensor(let st): return .signalTensor(st.shrink(ranges))
    default: throw LispError.typeError("shrink: argument must be tensor or signalTensor")
    }
  }

  private func evalPad(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("pad requires at least 1 argument")
    }
    let val = try evaluateAST(args[0])
    guard let padStr = attrValue(attributes, "@padding") else {
      throw LispError.invalidArgument("pad requires @padding attribute, e.g. @padding [1:1,0:0]")
    }
    let padding = parsePadding(padStr)
    switch val {
    case .tensor(let t): return .tensor(t.pad(padding))
    case .signalTensor(let st): return .signalTensor(st.pad(padding))
    default: throw LispError.typeError("pad: argument must be tensor or signalTensor")
    }
  }

  private func evalExpand(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("expand requires at least 1 argument")
    }
    let val = try evaluateAST(args[0])
    guard let shapeStr = attrValue(attributes, "@shape") else {
      throw LispError.invalidArgument("expand requires @shape attribute")
    }
    let targetShape = parseShape(shapeStr)
    switch val {
    case .tensor(let t): return .tensor(t.expand(targetShape))
    case .signalTensor(let st): return .signalTensor(st.expand(targetShape))
    default: throw LispError.typeError("expand: argument must be tensor or signalTensor")
    }
  }

  private func evalRepeat(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("repeat requires at least 1 argument")
    }
    let val = try evaluateAST(args[0])
    guard let repeatsStr = attrValue(attributes, "@repeats") else {
      throw LispError.invalidArgument("repeat requires @repeats attribute, e.g. @repeats [2,3]")
    }
    let repeats = parseIntList(repeatsStr)
    switch val {
    case .tensor(let t): return .tensor(t.repeat(repeats))
    case .signalTensor(let st): return .signalTensor(st.repeat(repeats))
    default: throw LispError.typeError("repeat: argument must be tensor or signalTensor")
    }
  }

  private func evalConv2d(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("conv2d requires 2 arguments (input, kernel)")
    }
    let val = try evaluateAST(args[0])
    let kernel = try requireTensor(evaluateAST(args[1]))
    switch val {
    case .tensor(let t): return .tensor(t.conv2d(kernel))
    case .signalTensor(let st): return .signalTensor(st.conv2d(kernel))
    default: throw LispError.typeError("conv2d: first argument must be tensor or signalTensor")
    }
  }

  private func evalConv1d(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("conv1d requires 2 arguments (input, kernel)")
    }
    let val = try evaluateAST(args[0])
    let kernel = try requireTensor(evaluateAST(args[1]))
    switch val {
    case .tensor(let t): return .tensor(t.conv1d(kernel))
    case .signalTensor(let st): return .signalTensor(st.conv1d(kernel))
    default: throw LispError.typeError("conv1d: first argument must be tensor or signalTensor")
    }
  }

  private func evalWindows(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("windows requires 1 argument")
    }
    guard let shapeStr = attrValue(attributes, "@shape") else {
      throw LispError.invalidArgument("windows requires @shape [kH kW]")
    }
    let shape = parseShape(shapeStr)
    let val = try evaluateAST(args[0])
    switch val {
    case .tensor(let t): return .tensor(t.windows(shape))
    case .signalTensor(let st): return .signalTensor(st.windows(shape))
    default: throw LispError.typeError("windows: argument must be tensor or signalTensor")
    }
  }

  // MARK: - Reductions

  private func evalSum(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("sum requires 1 argument")
    }
    let val = try evaluateAST(args[0])
    if let axisStr = attrValue(attributes, "@axis"), let axis = Int(axisStr) {
      switch val {
      case .tensor(let t): return .tensor(t.sum(axis: axis))
      case .signalTensor(let st): return .signalTensor(st.sum(axis: axis))
      default: throw LispError.typeError("sum: argument must be tensor or signalTensor")
      }
    }
    switch val {
    case .tensor(let t): return .tensor(t.sum())
    case .signalTensor(let st): return .signal(st.sum())
    default: throw LispError.typeError("sum: argument must be tensor or signalTensor")
    }
  }

  private func evalMean(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("mean requires 1 argument")
    }
    let val = try evaluateAST(args[0])
    if let axisStr = attrValue(attributes, "@axis"), let axis = Int(axisStr) {
      switch val {
      case .tensor(let t): return .tensor(t.mean(axis: axis))
      case .signalTensor(let st): return .signalTensor(st.mean(axis: axis))
      default: throw LispError.typeError("mean: argument must be tensor or signalTensor")
      }
    }
    switch val {
    case .tensor(let t): return .tensor(t.mean())
    case .signalTensor(let st): return .signal(st.mean())
    default: throw LispError.typeError("mean: argument must be tensor or signalTensor")
    }
  }

  private func evalMaxAxis(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("max-axis requires 1 argument")
    }
    let val = try evaluateAST(args[0])
    let axis = Int(attrValue(attributes, "@axis") ?? "0") ?? 0
    switch val {
    case .tensor(let t): return .tensor(t.max(axis: axis))
    case .signalTensor(let st): return .signalTensor(st.max(axis: axis))
    default: throw LispError.typeError("max-axis: argument must be tensor or signalTensor")
    }
  }

  private func evalSumAxis(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("sum-axis requires 1 argument")
    }
    let val = try evaluateAST(args[0])
    let axis = Int(attrValue(attributes, "@axis") ?? "0") ?? 0
    switch val {
    case .tensor(let t): return .tensor(t.sum(axis: axis))
    case .signalTensor(let st): return .signalTensor(st.sum(axis: axis))
    default: throw LispError.typeError("sum-axis: argument must be tensor or signalTensor")
    }
  }

  private func evalMeanAxis(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("mean-axis requires 1 argument")
    }
    let val = try evaluateAST(args[0])
    let axis = Int(attrValue(attributes, "@axis") ?? "0") ?? 0
    switch val {
    case .tensor(let t): return .tensor(t.mean(axis: axis))
    case .signalTensor(let st): return .signalTensor(st.mean(axis: axis))
    default: throw LispError.typeError("mean-axis: argument must be tensor or signalTensor")
    }
  }

  private func evalSoftmax(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("softmax requires 1 argument")
    }
    let t = try requireTensor(evaluateAST(args[0]))
    let axis = Int(attrValue(attributes, "@axis") ?? "-1") ?? -1
    return .tensor(t.softmax(axis: axis))
  }

  // MARK: - FFT

  private func evalFFT(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 1 else {
      throw LispError.invalidArgument("fft requires at least 1 argument (input)")
    }
    let val = try evaluateAST(args[0])
    let backend = attrValue(attributes, "@backend")?.lowercased() ?? "tensor"
    switch val {
    case .tensor(let t):
      let n = try fftSize(args: args, attributes: attributes, defaultValue: t.shape.last!)
      let (re, im) = backend == "accelerated" ? acceleratedFFT(t, N: n) : tensorFFT(t, N: n)
      definitions["__fft_re"] = .tensor(re)
      definitions["__fft_im"] = .tensor(im)
      return .tuple([.tensor(re), .tensor(im)])
    case .signalTensor(let st):
      let n = try fftSize(args: args, attributes: attributes, defaultValue: st.shape.last!)
      let (re, im) = backend == "accelerated" ? acceleratedFFT(st, N: n) : signalTensorFFT(st, N: n)
      definitions["__fft_re"] = .signalTensor(re)
      definitions["__fft_im"] = .signalTensor(im)
      return .tuple([.signalTensor(re), .signalTensor(im)])
    default:
      throw LispError.typeError("fft: argument must be tensor or signalTensor")
    }
  }

  private func evalIFFT(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count >= 2 else {
      throw LispError.invalidArgument("ifft requires at least 2 arguments (re, im)")
    }
    let reResult = try evaluateAST(args[0])
    let imResult = try evaluateAST(args[1])
    let backend = attrValue(attributes, "@backend")?.lowercased() ?? "tensor"
    switch (reResult, imResult) {
    case (.tensor(let re), .tensor(let im)):
      let n = try fftSize(
        args: Array(args.dropFirst()), attributes: attributes, defaultValue: re.shape.last!)
      return .tensor(
        backend == "accelerated" ? acceleratedIFFT(re, im, N: n) : tensorIFFT(re, im, N: n))
    case (.signalTensor(let re), .signalTensor(let im)):
      let n = try fftSize(
        args: Array(args.dropFirst()), attributes: attributes, defaultValue: re.shape.last!)
      return .signalTensor(
        backend == "accelerated" ? acceleratedIFFT(re, im, N: n) : signalTensorIFFT(re, im, N: n))
    default:
      throw LispError.typeError("ifft: both arguments must be same type (tensor or signalTensor)")
    }
  }

  private func fftSize(
    args: [ASTNode], attributes: [(name: String, value: String)], defaultValue: Int
  ) throws -> Int {
    if let attr = attrValue(attributes, "@N") ?? attrValue(attributes, "@n"), let n = Int(attr) {
      return n
    }
    if args.count >= 2 {
      return Int(try requireFloat(evaluateAST(args[1])))
    }
    return defaultValue
  }

  private func evalPolarFFT(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else { throw LispError.invalidArgument("polar-fft requires re and im") }
    switch (try evaluateAST(args[0]), try evaluateAST(args[1])) {
    case (.tensor(let re), .tensor(let im)):
      let out = polarFFT(re, im)
      return .tuple([.tensor(out.mag), .tensor(out.phase)])
    case (.signalTensor(let re), .signalTensor(let im)):
      let out = polarFFT(re, im)
      return .tuple([.signalTensor(out.mag), .signalTensor(out.phase)])
    default:
      throw LispError.typeError("polar-fft inputs must both be tensor or signalTensor")
    }
  }

  private func evalRectFFT(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("rect-fft requires mag and phase")
    }
    switch (try evaluateAST(args[0]), try evaluateAST(args[1])) {
    case (.tensor(let mag), .tensor(let phase)):
      let out = rectFFT(mag, phase)
      return .tuple([.tensor(out.re), .tensor(out.im)])
    case (.signalTensor(let mag), .signalTensor(let phase)):
      let out = rectFFT(mag, phase)
      return .tuple([.signalTensor(out.re), .signalTensor(out.im)])
    default:
      throw LispError.typeError("rect-fft inputs must both be tensor or signalTensor")
    }
  }

  private func evalComplexMul(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 4 else {
      throw LispError.invalidArgument("complex-mul requires ar ai br bi")
    }
    let values = try args.map { try evaluateAST($0) }
    switch (values[0], values[1], values[2], values[3]) {
    case (.tensor(let ar), .tensor(let ai), .tensor(let br), .tensor(let bi)):
      let out = complexMul(ar, ai, br, bi)
      return .tuple([.tensor(out.re), .tensor(out.im)])
    case (.signalTensor(let ar), .signalTensor(let ai), .tensor(let br), .tensor(let bi)):
      let out = complexMul(ar, ai, br, bi)
      return .tuple([.signalTensor(out.re), .signalTensor(out.im)])
    case (
      .signalTensor(let ar), .signalTensor(let ai), .signalTensor(let br), .signalTensor(let bi)
    ):
      let out = complexMul(ar, ai, br, bi)
      return .tuple([.signalTensor(out.re), .signalTensor(out.im)])
    default:
      throw LispError.typeError(
        "complex-mul inputs must be all tensor, signalTensor/signalTensor/tensor/tensor, or all signalTensor"
      )
    }
  }

  private func evalComplexConj(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else { throw LispError.invalidArgument("complex-conj requires re im") }
    switch (try evaluateAST(args[0]), try evaluateAST(args[1])) {
    case (.tensor(let re), .tensor(let im)):
      let out = complexConj(re, im)
      return .tuple([.tensor(out.re), .tensor(out.im)])
    case (.signalTensor(let re), .signalTensor(let im)):
      let out = complexConj(re, im)
      return .tuple([.signalTensor(out.re), .signalTensor(out.im)])
    default:
      throw LispError.typeError("complex-conj inputs must both be tensor or signalTensor")
    }
  }

  private func evalHann(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    let n: Int
    if let attr = attrValue(attributes, "@N") ?? attrValue(attributes, "@n") {
      n = Int(attr) ?? 1024
    } else if args.count == 1 {
      n = Int(try requireFloat(evaluateAST(args[0])))
    } else {
      throw LispError.invalidArgument("hann requires N")
    }
    return .tensor(DGenLazy.hann(n))
  }

  private func evalWindow(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    let type = (attrValue(attributes, "@type") ?? "hann").lowercased()
    guard type == "hann" else {
      throw LispError.invalidArgument("window currently only supports @type hann")
    }
    return try evalHann(args, attributes: attributes)
  }

  private func evalSpectrumDelay(_ args: [ASTNode], attributes: [(name: String, value: String)])
    throws -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("spectrum-delay requires spectrum input")
    }
    guard case .signalTensor(let st) = try evaluateAST(args[0]) else {
      throw LispError.typeError("spectrum-delay input must be signalTensor")
    }
    let n =
      Int(attrValue(attributes, "@N") ?? attrValue(attributes, "@n") ?? "\(st.shape.last!)") ?? st
      .shape.last!
    let hops = Int(attrValue(attributes, "@hops") ?? "1") ?? 1
    let hop = Int(attrValue(attributes, "@hop") ?? attrValue(attributes, "@hopSize") ?? "1") ?? 1
    return .signalTensor(st.spectrumDelay(N: n, hops: hops, hop: hop))
  }

  private func evalSpectrumDelayMod(_ args: [ASTNode], attributes: [(name: String, value: String)])
    throws -> EvalResult
  {
    guard args.count == 2 else {
      throw LispError.invalidArgument("spectrum-delay-mod requires spectrum and delay")
    }
    guard case .signalTensor(let st) = try evaluateAST(args[0]) else {
      throw LispError.typeError("spectrum-delay-mod input must be signalTensor")
    }
    let delay = try requireSignal(coerceToSignal(evaluateAST(args[1])))
    let n =
      Int(attrValue(attributes, "@N") ?? attrValue(attributes, "@n") ?? "\(st.shape.last!)") ?? st
      .shape.last!
    let maxHops =
      Int(attrValue(attributes, "@max-hops") ?? attrValue(attributes, "@maxHops") ?? "1") ?? 1
    let hop = Int(attrValue(attributes, "@hop") ?? attrValue(attributes, "@hopSize") ?? "1") ?? 1
    return .signalTensor(st.spectrumDelayMod(delay: delay, N: n, maxHops: maxHops, hop: hop))
  }

  private func evalPhaseVocoder(_ args: [ASTNode], attributes: [(name: String, value: String)])
    throws -> EvalResult
  {
    guard args.count == 3 else {
      throw LispError.invalidArgument("phase-vocoder requires re im ratio")
    }
    guard case .signalTensor(let re) = try evaluateAST(args[0]),
      case .signalTensor(let im) = try evaluateAST(args[1])
    else {
      throw LispError.typeError("phase-vocoder re/im must be signalTensor")
    }
    let ratio = try requireSignal(coerceToSignal(evaluateAST(args[2])))
    let n =
      Int(attrValue(attributes, "@N") ?? attrValue(attributes, "@n") ?? "\(re.shape.last!)") ?? re
      .shape.last!
    let hop = Int(attrValue(attributes, "@hop") ?? attrValue(attributes, "@hopSize") ?? "1") ?? 1
    let out = DGenLazy.phaseVocoder(re, im, ratio: ratio, N: n, hop: hop)
    return .tuple([.signalTensor(out.re), .signalTensor(out.im)])
  }

  private func evalPartitionIR(_ args: [ASTNode], attributes: [(name: String, value: String)])
    throws -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("partition-ir requires an IR tensor")
    }
    let ir = try requireTensor(evaluateAST(args[0]))
    let n = Int(attrValue(attributes, "@N") ?? attrValue(attributes, "@n") ?? "1024") ?? 1024
    let hop =
      Int(attrValue(attributes, "@hop") ?? attrValue(attributes, "@hopSize") ?? "\(n / 2)") ?? n / 2
    let out = DGenLazy.partitionIR(ir, N: n, hop: hop)
    return .tuple([.tensor(out.re), .tensor(out.im)])
  }

  private func evalPartitionedSpectralMAC(
    _ args: [ASTNode], attributes: [(name: String, value: String)]
  ) throws -> EvalResult {
    guard args.count == 4 else {
      throw LispError.invalidArgument("partitioned-spectral-mac requires xre xim irre irim")
    }
    guard case .signalTensor(let xre) = try evaluateAST(args[0]),
      case .signalTensor(let xim) = try evaluateAST(args[1])
    else {
      throw LispError.typeError("partitioned-spectral-mac live inputs must be signalTensor")
    }
    let irre = try requireTensor(evaluateAST(args[2]))
    let irim = try requireTensor(evaluateAST(args[3]))
    let n =
      Int(attrValue(attributes, "@N") ?? attrValue(attributes, "@n") ?? "\(xre.shape.last!)") ?? xre
      .shape.last!
    let out = DGenLazy.partitionedSpectralMAC(xre, xim, irre, irim, N: n)
    return .tuple([.signalTensor(out.re), .signalTensor(out.im)])
  }

  private func evalPartitionedConvolve(
    _ args: [ASTNode], attributes: [(name: String, value: String)]
  ) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("partitioned-convolve requires input and ir")
    }
    let input = try requireSignal(evaluateAST(args[0]))
    let ir = try requireTensor(evaluateAST(args[1]))
    let n = Int(attrValue(attributes, "@N") ?? attrValue(attributes, "@n") ?? "1024") ?? 1024
    let hop =
      Int(attrValue(attributes, "@hop") ?? attrValue(attributes, "@hopSize") ?? "\(n / 2)") ?? n / 2
    let gain = Float(attrValue(attributes, "@gain") ?? "1") ?? 1
    return .signal(DGenLazy.partitionedConvolve(input, ir, N: n, hop: hop, gain: gain))
  }

  // MARK: - Overlap-add

  private func evalOverlapAdd(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("overlap-add requires 2 arguments (signalTensor, hop)")
    }
    let val = try evaluateAST(args[0])
    let hop = Int(try requireFloat(evaluateAST(args[1])))
    switch val {
    case .signalTensor(let st):
      return .signal(st.overlapAdd(hop: hop))
    default:
      throw LispError.typeError("overlap-add: first argument must be signalTensor")
    }
  }

  // MARK: - Utility functions

  private func evalScale(_ args: [ASTNode]) throws -> EvalResult {
    // (scale sig inMin inMax outMin outMax)
    guard args.count == 5 else {
      throw LispError.invalidArgument("scale requires 5 arguments (sig inMin inMax outMin outMax)")
    }
    let sig = try requireSignal(evaluateAST(args[0]))
    let inMin = try requireSignalOrFloat(evaluateAST(args[1]))
    let inMax = try requireSignalOrFloat(evaluateAST(args[2]))
    let outMin = try requireSignalOrFloat(evaluateAST(args[3]))
    let outMax = try requireSignalOrFloat(evaluateAST(args[4]))

    // scale(x, a, b, c, d) = c + (x - a) / (b - a) * (d - c)
    let normalized = (sig - inMin) / (inMax - inMin)
    let result = outMin + normalized * (outMax - outMin)
    return .signal(result)
  }

  private func evalTriangle(_ args: [ASTNode]) throws -> EvalResult {
    // (triangle phase) or (triangle phase duty) - convert phasor (0..1) to triangle wave (0..1..0)
    guard args.count >= 1 else {
      throw LispError.invalidArgument("triangle requires at least 1 argument (phase)")
    }
    let phase = try requireSignal(evaluateAST(args[0]))
    let duty: Signal? = args.count >= 2 ? try requireSignal(evaluateAST(args[1])) : nil
    return .signal(phase.triangle(duty: duty))
  }

  private func evalWrap(_ args: [ASTNode]) throws -> EvalResult {
    // (wrap sig min max) - wrap value to range
    guard args.count >= 1 else {
      throw LispError.invalidArgument("wrap requires at least 1 argument")
    }
    let sig = try requireSignal(evaluateAST(args[0]))
    let minVal: Signal =
      args.count >= 2 ? try requireSignal(evaluateAST(args[1])) : Signal.constant(0)
    let maxVal: Signal =
      args.count >= 3 ? try requireSignal(evaluateAST(args[2])) : Signal.constant(1)

    let range = maxVal - minVal
    // wrap: mod(sig - min, range) + min
    let shifted = sig - minVal
    let wrapped = DGenLazy.mod(shifted, range)
    return .signal(wrapped + minVal)
  }

  private func evalClip(_ args: [ASTNode]) throws -> EvalResult {
    // (clip sig min max)
    guard args.count == 3 else {
      throw LispError.invalidArgument("clip requires 3 arguments (sig, min, max)")
    }
    let sig = try requireSignal(evaluateAST(args[0]))
    let minResult = try promoteToValue(evaluateAST(args[1]))
    let maxResult = try promoteToValue(evaluateAST(args[2]))

    switch (minResult, maxResult) {
    case (.float(let lo), .float(let hi)):
      return .signal(sig.clip(Double(lo), Double(hi)))
    case (.signal(let lo), .signal(let hi)):
      return .signal(sig.clip(lo, hi))
    default:
      let lo = try requireSignal(coerceToSignal(minResult))
      let hi = try requireSignal(coerceToSignal(maxResult))
      return .signal(sig.clip(lo, hi))
    }
  }

  private func evalGswitch(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 3 else {
      throw LispError.invalidArgument("gswitch requires 3 arguments (cond, a, b)")
    }
    let cond = try promoteToValue(evaluateAST(args[0]))
    let a = try promoteToValue(evaluateAST(args[1]))
    let b = try promoteToValue(evaluateAST(args[2]))

    switch (cond, a, b) {
    case (.signal(let c), .signal(let va), .signal(let vb)):
      return .signal(DGenLazy.gswitch(c, va, vb))
    case (.signal(let c), .float(let va), .float(let vb)):
      return .signal(DGenLazy.gswitch(c, Double(va), Double(vb)))
    case (.signal(let c), .signal(let va), .float(let vb)):
      return .signal(DGenLazy.gswitch(c, va, Double(vb)))
    case (.signal(let c), .float(let va), .signal(let vb)):
      return .signal(DGenLazy.gswitch(c, Double(va), vb))
    case (.tensor(let c), .tensor(let va), .tensor(let vb)):
      return .tensor(DGenLazy.gswitch(c, va, vb))
    default:
      throw LispError.typeError("gswitch: unsupported type combination")
    }
  }

  private func evalSelector(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count >= 2 else {
      throw LispError.invalidArgument("selector requires at least 2 arguments (mode, options...)")
    }

    let mode = try requireSignal(coerceToSignal(evaluateAST(args[0])))
    let options = try args.dropFirst().map { arg -> Signal in
      try requireSignal(coerceToSignal(evaluateAST(arg)))
    }
    return .signal(DGenLazy.selector(mode, options))
  }

  private func evalModulatedParam(
    _ args: [ASTNode],
    attributes: [(name: String, value: String)]
  ) throws -> EvalResult {
    guard args.count >= 2, args.count % 2 == 0 else {
      throw LispError.invalidArgument(
        "__modulated-param requires base, active, and modulator/depth pairs")
    }
    guard let modeRaw = attrValue(attributes, "@mode"),
          let mode = ModulatedParamMode(rawValue: modeRaw.lowercased()),
          let minValue = Float(attrValue(attributes, "@min") ?? ""),
          let maxValue = Float(attrValue(attributes, "@max") ?? "")
    else {
      throw LispError.invalidArgument(
        "__modulated-param requires @mode, @min, and @max attributes")
    }

    let base = try requireSignal(coerceToSignal(evaluateAST(args[0])))
    let active = try requireSignal(coerceToSignal(evaluateAST(args[1])))
    var lanes: [(modulator: Signal, depth: Signal)] = []
    var index = 2
    while index < args.count {
      let modulator = try requireSignal(coerceToSignal(evaluateAST(args[index])))
      let depth = try requireSignal(coerceToSignal(evaluateAST(args[index + 1])))
      lanes.append((modulator, depth))
      index += 2
    }

    return .signal(DGenLazy.modulatedParam(
      base,
      active: active,
      lanes: lanes,
      mode: mode,
      min: minValue,
      max: maxValue))
  }

  private func evalBuffer(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count >= 2 else {
      throw LispError.invalidArgument("buffer requires 2 arguments (signal, size)")
    }
    let sig = try requireSignal(evaluateAST(args[0]))
    let size = Int(try requireFloat(evaluateAST(args[1])))
    let hop: Int? = args.count >= 3 ? Int(try requireFloat(evaluateAST(args[2]))) : nil
    return .signalTensor(sig.buffer(size: size, hop: hop))
  }

  // MARK: - Type helpers

  /// Promote float to signal/tensor-ready value, keeping typed results as-is
  private func promoteToValue(_ result: EvalResult) -> EvalResult {
    return result
  }

  /// Coerce any EvalResult to a Signal
  private func coerceToSignal(_ result: EvalResult) throws -> EvalResult {
    switch result {
    case .signal: return result
    case .float(let f): return .signal(Signal.constant(f))
    case .tensor, .signalTensor, .tuple, .none:
      throw LispError.typeError("Expected signal, got other type")
    }
  }

  private func requireSignal(_ result: EvalResult) throws -> Signal {
    switch result {
    case .signal(let s): return s
    case .float(let f): return Signal.constant(f)
    default: throw LispError.typeError("Expected signal, got other type")
    }
  }

  private func requireTensor(_ result: EvalResult) throws -> Tensor {
    switch result {
    case .tensor(let t): return t
    default: throw LispError.typeError("Expected tensor, got other type")
    }
  }

  private func requireFloat(_ result: EvalResult) throws -> Float {
    switch result {
    case .float(let f): return f
    case .signal(let s):
      if let d = s.data { return d }
      throw LispError.typeError("Expected constant float, got dynamic signal")
    default: throw LispError.typeError("Expected float, got other type")
    }
  }

  private func requireSignalOrFloat(_ result: EvalResult) throws -> Signal {
    switch result {
    case .signal(let s): return s
    case .float(let f): return Signal.constant(f)
    default: throw LispError.typeError("Expected signal or float")
    }
  }

  private func asSignalOrNil(_ result: EvalResult) throws -> Signal? {
    switch result {
    case .signal(let s): return s
    case .float(let f): return Signal.constant(f)
    case .none: return nil
    default: throw LispError.typeError("Expected signal, float, or none")
    }
  }

  // MARK: - Attribute helpers

  private func attrValue(_ attrs: [(name: String, value: String)], _ key: String) -> String? {
    attrs.first(where: { $0.name == key })?.value
  }

  private func parseOptionalPositiveIntAttribute(
    _ attrs: [(name: String, value: String)],
    _ key: String
  ) throws -> Int? {
    guard let rawValue = attrValue(attrs, key) else { return nil }
    guard let value = Int(rawValue), value > 0 else {
      throw LispError.invalidArgument("\(key) requires a positive integer")
    }
    return value
  }

  private func parseBoolAttr(_ attrs: [(name: String, value: String)], _ key: String) -> Bool {
    guard let value = attrValue(attrs, key)?.lowercased() else { return false }
    return value == "true" || value == "1"
  }

  private func parseBoolString(_ value: String, defaultValue: Bool) -> Bool {
    switch value.lowercased() {
    case "true", "1", "yes", "on": return true
    case "false", "0", "no", "off": return false
    default: return defaultValue
    }
  }

  private func unquote(_ value: String) -> String {
    if value.count >= 2, value.first == "\"", value.last == "\"" {
      return String(value.dropFirst().dropLast())
    }
    return value
  }

  private func parseShape(_ str: String) -> [Int] {
    // Parse "[2,3]", "[2 3]", "2,3", or "2 3"
    let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    return
      cleaned
      .split(whereSeparator: { $0 == "," || $0.isWhitespace })
      .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
  }

  private func parseIntList(_ str: String) -> [Int] {
    let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    return
      cleaned
      .split(whereSeparator: { $0 == "," || $0.isWhitespace })
      .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
  }

  private func parseFloatList(_ str: String) -> [Float] {
    let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    return
      cleaned
      .split(whereSeparator: { $0 == "," || $0.isWhitespace })
      .compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
  }

  private func parseShapeArgs(_ args: [ASTNode]) throws -> [Int] {
    if args.count == 1, case .atom(let str) = args[0], str.hasPrefix("[") {
      return parseShape(str)
    }
    return try args.map { Int(try requireFloat(evaluateAST($0))) }
  }

  private func parseShapeFromArgsOrAttributes(
    _ args: [ASTNode],
    attributes: [(name: String, value: String)],
    op: String
  ) throws -> [Int] {
    let shape: [Int]
    if let shapeStr = attrValue(attributes, "@shape") {
      shape = parseShape(shapeStr)
    } else {
      guard !args.isEmpty else {
        throw LispError.invalidArgument("\(op) requires @shape [d1,d2,...] or shape arguments")
      }
      shape = try parseShapeArgs(args)
    }
    guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
      throw LispError.invalidArgument("\(op) shape must contain positive dimensions")
    }
    return shape
  }

  private func makeTensor(shape: [Int], data: [Float], mutable: Bool) -> Tensor {
    if mutable {
      return Tensor.param(shape, data: data)
    }
    if shape.count == 1 {
      return Tensor(data)
    }
    if shape.count == 2 {
      let rows = shape[0]
      let cols = shape[1]
      let nested = (0..<rows).map { row in
        Array(data[(row * cols)..<((row + 1) * cols)])
      }
      return Tensor(nested)
    }
    return Tensor.param(shape, data: data)
  }

  private func loadTensorData(file: String, expectedShape: [Int]) throws -> [Float] {
    let url = URL(fileURLWithPath: file, relativeTo: sourceDirectory).standardizedFileURL
    let rawData: Data
    do {
      rawData = try Data(contentsOf: url)
    } catch {
      throw LispError.invalidArgument(
        "failed to read tensor file '\(file)' relative to \(sourceDirectory.path): \(error)")
    }

    let json: Any
    do {
      json = try JSONSerialization.jsonObject(with: rawData)
    } catch {
      throw LispError.invalidArgument("failed to parse tensor file '\(file)' as JSON: \(error)")
    }

    let loaded: (shape: [Int]?, data: [Float])
    if let object = json as? [String: Any] {
      let shape = (object["shape"] as? [Any])?.compactMap { item -> Int? in
        if let int = item as? Int { return int }
        if let number = item as? NSNumber { return number.intValue }
        return nil
      }
      guard let dataValue = object["data"] else {
        throw LispError.invalidArgument("tensor file '\(file)' object must contain a data array")
      }
      loaded = (shape, try flattenJsonFloats(dataValue, file: file))
    } else {
      loaded = (nil, try flattenJsonFloats(json, file: file))
    }

    if let fileShape = loaded.shape, fileShape != expectedShape {
      throw LispError.invalidArgument(
        "tensor file '\(file)' shape \(fileShape) does not match expected shape \(expectedShape)")
    }

    let expectedCount = expectedShape.reduce(1, *)
    guard loaded.data.count == expectedCount else {
      throw LispError.invalidArgument(
        "tensor file '\(file)' has \(loaded.data.count) values, expected \(expectedCount) for shape \(expectedShape)"
      )
    }
    return loaded.data
  }

  private func wavChannelCount(url: URL) throws -> Int {
    let data = try Data(contentsOf: url)
    guard data.count >= 44 else {
      throw LispError.invalidArgument("audio file '\(url.path)' is too small to be a WAV")
    }
    var offset = 12
    while offset + 8 <= data.count {
      let chunkId = String(data: data[offset..<offset + 4], encoding: .ascii) ?? ""
      let chunkSize = Int(readUInt32LE(data, at: offset + 4))
      let chunkStart = offset + 8
      if chunkId == "fmt " {
        guard chunkSize >= 4 else {
          throw LispError.invalidArgument("audio file '\(url.path)' has an invalid fmt chunk")
        }
        return Int(readUInt16LE(data, at: chunkStart + 2))
      }
      offset = chunkStart + chunkSize + (chunkSize % 2 == 0 ? 0 : 1)
    }
    throw LispError.invalidArgument("audio file '\(url.path)' has no fmt chunk")
  }

  private func readUInt16LE(_ data: Data, at offset: Int) -> UInt16 {
    data.withUnsafeBytes { raw in
      raw.load(fromByteOffset: offset, as: UInt16.self).littleEndian
    }
  }

  private func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
    data.withUnsafeBytes { raw in
      raw.load(fromByteOffset: offset, as: UInt32.self).littleEndian
    }
  }

  private func flattenJsonFloats(_ value: Any, file: String) throws -> [Float] {
    if let array = value as? [Any] {
      return try array.flatMap { try flattenJsonFloats($0, file: file) }
    }
    if let number = value as? NSNumber {
      return [number.floatValue]
    }
    throw LispError.invalidArgument("tensor file '\(file)' contains a non-numeric value")
  }

  /// Parse "[0:2,1:3]" into [(Int,Int)?] ranges for shrink
  private func parseRanges(_ str: String) -> [(Int, Int)?] {
    let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    return cleaned.split(separator: ",").map { part in
      let trimmed = part.trimmingCharacters(in: .whitespaces)
      if trimmed == ":" || trimmed == "nil" { return nil }
      let parts = trimmed.split(separator: ":")
      guard parts.count == 2,
        let start = Int(parts[0].trimmingCharacters(in: .whitespaces)),
        let end = Int(parts[1].trimmingCharacters(in: .whitespaces))
      else {
        return nil
      }
      return (start, end)
    }
  }

  /// Parse "[1:1,0:0]" into [(Int,Int)] padding pairs
  private func parsePadding(_ str: String) -> [(Int, Int)] {
    let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    return
      cleaned
      .replacingOccurrences(of: ",", with: " ")
      .split(whereSeparator: { $0.isWhitespace })
      .compactMap { part in
        let parts = part.split(separator: ":")
        guard parts.count == 2,
          let before = Int(parts[0].trimmingCharacters(in: .whitespaces)),
          let after = Int(parts[1].trimmingCharacters(in: .whitespaces))
        else {
          return nil
        }
        return (before, after)
      }
  }
}

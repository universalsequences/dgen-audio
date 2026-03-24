// LispEvaluator - AST → DGenLazy Signal/Tensor graph building
//
// Walks AST nodes from LispParser and calls DGenLazy APIs to build
// the computation graph. Supports arithmetic, math, signal generators,
// stateful ops, effects, I/O, tensors, macros, and history feedback.

import DGenLazy
import DGen
import Foundation

// Resolve Tensor ambiguity: DGenLazy.Tensor is the one we use everywhere
typealias Tensor = DGenLazy.Tensor

// MARK: - Types

enum EvalResult {
    case signal(Signal)
    case tensor(Tensor)
    case signalTensor(SignalTensor)
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
    let generatedKind: String?
    let generatedFor: String?
    let modulationMode: ModulationMode?
    let modulationDepthMin: Float?
    let modulationDepthMax: Float?
    let modulationSourceParamName: String?
    let modulationDepthParamName: String?
    let modulationResolvedSymbolName: String?
}

struct OutputInfo {
    let channel: Int
    let signal: Signal
    let name: String?
}

struct InputInfo {
    let channel: Int
    let name: String?
    let modulatorSlot: Int?
}

struct MacroDefinition {
    let params: [String]
    let body: [ASTNode]
}

// MARK: - Evaluator

class LispEvaluator {
    var definitions: [String: EvalResult] = [:]
    var historyBindings: [String: (read: Signal, write: (Signal) -> Signal)] = [:]
    var macros: [String: MacroDefinition] = [:]
    var params: [ParamInfo] = []
    var outputs: [OutputInfo] = []
    var inputs: [InputInfo] = []
    var macroExpansionCounter: Int = 0

    // MARK: - Top-level evaluation

    func evaluate(source: String) throws {
        let nodes = try parseSource(source)
        try evaluate(nodes: nodes)
    }

    func evaluate(nodes: [ASTNode]) throws {
        for node in nodes {
            let _ = try evaluateAST(node)
        }
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
        guard case .atom(let name) = elements[1] else {
            throw LispError.parseError("def: name must be an atom")
        }

        // Evaluate all body expressions, use last as value
        var result: EvalResult = .none
        for i in 2..<elements.count {
            result = try evaluateAST(elements[i])
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

    private func evaluateMakeHistory(_ elements: [ASTNode]) throws -> EvalResult {
        guard elements.count >= 2, case .atom(let name) = elements[1] else {
            throw LispError.parseError("make-history requires a name: (make-history name)")
        }
        let history = Signal.history()
        historyBindings[name] = history
        return .none
    }

    private func evaluateReadHistory(_ elements: [ASTNode]) throws -> EvalResult {
        guard elements.count >= 2, case .atom(let name) = elements[1] else {
            throw LispError.parseError("read-history requires a name")
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
        guard let binding = historyBindings[name] else {
            throw LispError.historyNotFound(name)
        }
        let value = try requireSignal(evaluateAST(elements[2]))
        let result = binding.write(value)
        return .signal(result)
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
                case .atom(let op) = elements[0],
                case .atom(let name) = elements[1]
            else {
                for element in elements { findDefNamesInAST(element, into: &names) }
                return
            }
            let opLower = op.lowercased()
            if opLower == "def" || opLower == "make-history" {
                names.insert(name)
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
                case .atom(let name) = elements[1],
                localNames.contains(name)
            {
                let opLower = op.lowercased()
                if opLower == "def" || opLower == "make-history" {
                    var newElements = elements
                    newElements[1] = .atom(prefix + name)
                    for i in 2..<newElements.count {
                        newElements[i] = scopeDefName(newElements[i], prefix: prefix, localNames: localNames)
                    }
                    return .list(newElements)
                }
            }
            return .list(elements.map { scopeDefName($0, prefix: prefix, localNames: localNames) })
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
        case "sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "abs", "sign",
            "floor", "ceil", "round", "relu", "sigmoid":
            return try evalUnaryMath(regularArgs, fn: op)

        // Binary math
        case "pow":
            return try evalPow(regularArgs)
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
            return .signal(Signal.noise())
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
            return try evalTensor(regularArgs)
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
            return try evalFFT(regularArgs)
        case "ifft":
            return try evalIFFT(regularArgs)

        // Windowing
        case "buffer":
            return try evalBuffer(regularArgs)
        case "overlap-add", "overlapadd":
            return try evalOverlapAdd(regularArgs)

        // Utility
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

        default:
            throw LispError.unknownOperator(opName)
        }
    }

    // MARK: - Arithmetic

    private func evalBinaryArith(_ args: [ASTNode], op: String) throws -> EvalResult {
        guard args.count == 2 else {
            throw LispError.invalidArgument("\(op) requires 2 arguments, got \(args.count)")
        }
        let lhs = try evaluateAST(args[0])
        let rhs = try evaluateAST(args[1])
        return try applyBinaryOp(lhs, rhs, op: op)
    }

    private func applyBinaryOp(_ lhs: EvalResult, _ rhs: EvalResult, op: String) throws -> EvalResult {
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
            case "*": return .signalTensor(a * b)
            default: throw LispError.typeError("Unsupported op \(op) for signalTensor op signal")
            }

        case (.signal(let a), .signalTensor(let b)):
            switch op {
            case "+": return .signalTensor(a + b)
            case "*": return .signalTensor(a * b)
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
            let bSignal = Signal.constant(b)
            switch op {
            case "+": return .signalTensor(a + bSignal)
            case "*": return .signalTensor(a * bSignal)
            default: throw LispError.typeError("Unsupported op \(op) for signalTensor op float")
            }

        case (.float(let a), .signalTensor(let b)):
            let aSignal = Signal.constant(a)
            switch op {
            case "+": return .signalTensor(aSignal + b)
            case "*": return .signalTensor(aSignal * b)
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
            case "tanh": return .float(Foundation.tanh(f))
            case "exp": return .float(Foundation.exp(f))
            case "log": return .float(Foundation.log(f))
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
            case "tanh": return .signal(DGenLazy.tanh(s))
            case "exp": return .signal(DGenLazy.exp(s))
            case "log": return .signal(DGenLazy.log(s))
            case "sqrt": return .signal(DGenLazy.sqrt(s))
            case "abs": return .signal(DGenLazy.abs(s))
            case "sign": return .signal(DGenLazy.sign(s))
            case "floor": return .signal(s) // floor not available for Signal, pass through
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
            case "tanh": return .tensor(DGenLazy.tanh(t))
            case "exp": return .tensor(DGenLazy.exp(t))
            case "log": return .tensor(DGenLazy.log(t))
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
            case "exp": return .signalTensor(DGenLazy.exp(st))
            case "log": return .signalTensor(DGenLazy.log(st))
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

    private func evalBiquad(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
        guard args.count >= 1 else {
            throw LispError.invalidArgument("biquad requires at least 1 argument (signal)")
        }
        let sigResult = try evaluateAST(args[0])

        // Parse biquad params from remaining args or attributes — accept signals or floats
        let cutoff: Signal = args.count >= 2 ? try requireSignal(evaluateAST(args[1])) :
            Signal.constant(Float(attrValue(attributes, "@cutoff") ?? "1000") ?? 1000)
        let q: Signal = args.count >= 3 ? try requireSignal(evaluateAST(args[2])) :
            Signal.constant(Float(attrValue(attributes, "@q") ?? "0.707") ?? 0.707)
        let gain: Signal = args.count >= 4 ? try requireSignal(evaluateAST(args[3])) :
            Signal.constant(Float(attrValue(attributes, "@gain") ?? "0") ?? 0)
        let mode: Signal = args.count >= 5 ? try requireSignal(evaluateAST(args[4])) :
            Signal.constant(Float(attrValue(attributes, "@mode") ?? "0") ?? 0)

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

    private func evalParam(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
        // First regular arg is the name
        guard args.count >= 1, case .atom(let name) = args[0] else {
            throw LispError.invalidArgument("param requires a name: (param name @default 440)")
        }

        let defaultVal = Float(attrValue(attributes, "@default") ?? "0") ?? 0
        let minVal = Float(attrValue(attributes, "@min") ?? "")
        let maxVal = Float(attrValue(attributes, "@max") ?? "")
        let unit = attrValue(attributes, "@unit")
        let hidden = parseBoolAttr(attributes, "@hidden")
        let generatedKind = attrValue(attributes, "@generated")
        let generatedFor = attrValue(attributes, "@generated-for")
        let modulationMode = attrValue(attributes, "@mod-mode").flatMap {
            ModulationMode(rawValue: $0.lowercased())
        }
        let modulationDepthMin = Float(attrValue(attributes, "@mod-depth-min") ?? "")
        let modulationDepthMax = Float(attrValue(attributes, "@mod-depth-max") ?? "")
        let modulationSourceParamName = attrValue(attributes, "@mod-source-param")
        let modulationDepthParamName = attrValue(attributes, "@mod-depth-param")
        let modulationResolvedSymbolName = attrValue(attributes, "@mod-resolved-symbol")

        let signal = Signal.param(defaultVal, min: minVal, max: maxVal)

        let info = ParamInfo(
            name: name,
            cellId: signal.memoryCellId,
            defaultValue: defaultVal,
            min: minVal,
            max: maxVal,
            unit: unit,
            hidden: hidden,
            generatedKind: generatedKind,
            generatedFor: generatedFor,
            modulationMode: modulationMode,
            modulationDepthMin: modulationDepthMin,
            modulationDepthMax: modulationDepthMax,
            modulationSourceParamName: modulationSourceParamName,
            modulationDepthParamName: modulationDepthParamName,
            modulationResolvedSymbolName: modulationResolvedSymbolName
        )
        params.append(info)
        definitions[name] = .signal(signal)

        return .signal(signal)
    }

    private func evalInput(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalOutput(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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
        outputs.append(OutputInfo(channel: channel, signal: signal, name: name))

        return .none
    }

    // MARK: - Tensor ops

    private func evalTensor(_ args: [ASTNode]) throws -> EvalResult {
        guard args.count >= 2 else {
            throw LispError.invalidArgument("tensor requires at least 2 arguments (rows, cols)")
        }
        let rows = Int(try requireFloat(evaluateAST(args[0])))
        let cols = Int(try requireFloat(evaluateAST(args[1])))
        return .tensor(Tensor.zeros([rows, cols]))
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

    private func evalReshape(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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
            throw LispError.typeError("stateful-phasor: freq must be signal, float, tensor, or signalTensor")
        }
    }

    // MARK: - Compressor

    private func evalCompressor(_ args: [ASTNode], rawArgs: [ASTNode] = [], attributes: [(name: String, value: String)]) throws -> EvalResult {
        guard args.count >= 1 else {
            throw LispError.invalidArgument("compressor requires at least 1 argument (signal)")
        }
        let sigResult = try evaluateAST(args[0])

        // Accept signals or floats for all parameters
        let ratio: Signal = args.count >= 2 ? try requireSignal(evaluateAST(args[1])) :
            Signal.constant(Float(attrValue(attributes, "@ratio") ?? "4") ?? 4)
        let threshold: Signal = args.count >= 3 ? try requireSignal(evaluateAST(args[2])) :
            Signal.constant(Float(attrValue(attributes, "@threshold") ?? "-20") ?? -20)
        let knee: Signal = args.count >= 4 ? try requireSignal(evaluateAST(args[3])) :
            Signal.constant(Float(attrValue(attributes, "@knee") ?? "6") ?? 6)
        let attack: Signal = args.count >= 5 ? try requireSignal(evaluateAST(args[4])) :
            Signal.constant(Float(attrValue(attributes, "@attack") ?? "0.01") ?? 0.01)
        let release: Signal = args.count >= 6 ? try requireSignal(evaluateAST(args[5])) :
            Signal.constant(Float(attrValue(attributes, "@release") ?? "0.1") ?? 0.1)

        // Sidechain detection:
        // 8-arg form: (compressor sig ratio threshold knee attack release isSidechain sidechain)
        // 7-arg form: (compressor sig ratio threshold knee attack release sidechain)
        // attribute:  @sidechain varname  or  @sidechain (expr)
        if args.count >= 8 {
            let isSideChain = try requireSignal(evaluateAST(args[6]))
            let sidechain = try requireSignal(evaluateAST(args[7]))
            switch sigResult {
            case .signal(let sig):
                return .signal(sig.compressor(ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release, isSideChain: isSideChain, sidechain: sidechain))
            case .signalTensor(let st):
                return .signalTensor(st.compressor(ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release, isSideChain: isSideChain, sidechain: sidechain))
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
            return .signal(sig.compressor(ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release, sidechain: sidechain))
        case .signalTensor(let st):
            return .signalTensor(st.compressor(ratio: ratio, threshold: threshold, knee: knee, attack: attack, release: release, sidechain: sidechain))
        default:
            throw LispError.typeError("compressor: first argument must be signal or signalTensor")
        }
    }

    // MARK: - Tensor creation

    private enum TensorFill { case zeros, ones, full, randn }

    private func evalTensorCreate(_ args: [ASTNode], fill: TensorFill) throws -> EvalResult {
        guard args.count >= 1 else {
            throw LispError.invalidArgument("\(fill) requires at least 1 argument (shape as [d1,d2,...] or individual dims)")
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

    private func evalTensorParam(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
        guard args.count >= 1 else {
            throw LispError.invalidArgument("tensor-param requires shape argument")
        }
        let shape: [Int]
        if case .atom(let str) = args[0], str.hasPrefix("[") {
            shape = parseShape(str)
        } else {
            shape = try args.map { Int(try requireFloat(evaluateAST($0))) }
        }
        return .tensor(Tensor.param(shape))
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

    private func evalToSignal(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
        guard args.count == 1 else {
            throw LispError.invalidArgument("to-signal requires 1 argument (tensor)")
        }
        let t = try requireTensor(evaluateAST(args[0]))
        let maxFrames: Int? = attrValue(attributes, "@max-frames").flatMap { Int($0) }
        return .signal(t.toSignal(maxFrames: maxFrames))
    }

    // MARK: - Tensor shape ops

    private func evalTranspose(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalShrink(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalPad(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalExpand(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalRepeat(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    // MARK: - Reductions

    private func evalSum(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalMean(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalMaxAxis(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalSumAxis(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalMeanAxis(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
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

    private func evalSoftmax(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws -> EvalResult {
        guard args.count == 1 else {
            throw LispError.invalidArgument("softmax requires 1 argument")
        }
        let t = try requireTensor(evaluateAST(args[0]))
        let axis = Int(attrValue(attributes, "@axis") ?? "-1") ?? -1
        return .tensor(t.softmax(axis: axis))
    }

    // MARK: - FFT

    private func evalFFT(_ args: [ASTNode]) throws -> EvalResult {
        guard args.count >= 1 else {
            throw LispError.invalidArgument("fft requires at least 1 argument (input)")
        }
        let val = try evaluateAST(args[0])
        // N defaults to tensor size if not given
        switch val {
        case .tensor(let t):
            let n = args.count >= 2 ? Int(try requireFloat(evaluateAST(args[1]))) : t.shape.last!
            let (re, im) = tensorFFT(t, N: n)
            // Store both as a def - return re, user can access via convention
            // For now, return re (real part). Users should use (def fftResult (fft input)) pattern.
            definitions["__fft_re"] = .tensor(re)
            definitions["__fft_im"] = .tensor(im)
            return .tensor(re)
        case .signalTensor(let st):
            let n = args.count >= 2 ? Int(try requireFloat(evaluateAST(args[1]))) : st.shape.last!
            let (re, im) = signalTensorFFT(st, N: n)
            definitions["__fft_re"] = .signalTensor(re)
            definitions["__fft_im"] = .signalTensor(im)
            return .signalTensor(re)
        default:
            throw LispError.typeError("fft: argument must be tensor or signalTensor")
        }
    }

    private func evalIFFT(_ args: [ASTNode]) throws -> EvalResult {
        guard args.count >= 2 else {
            throw LispError.invalidArgument("ifft requires at least 2 arguments (re, im)")
        }
        let reResult = try evaluateAST(args[0])
        let imResult = try evaluateAST(args[1])
        switch (reResult, imResult) {
        case (.tensor(let re), .tensor(let im)):
            let n = args.count >= 3 ? Int(try requireFloat(evaluateAST(args[2]))) : re.shape.last!
            return .tensor(tensorIFFT(re, im, N: n))
        case (.signalTensor(let re), .signalTensor(let im)):
            let n = args.count >= 3 ? Int(try requireFloat(evaluateAST(args[2]))) : re.shape.last!
            return .signalTensor(signalTensorIFFT(re, im, N: n))
        default:
            throw LispError.typeError("ifft: both arguments must be same type (tensor or signalTensor)")
        }
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
        let minVal: Signal = args.count >= 2 ? try requireSignal(evaluateAST(args[1])) : Signal.constant(0)
        let maxVal: Signal = args.count >= 3 ? try requireSignal(evaluateAST(args[2])) : Signal.constant(1)

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
        case .tensor, .signalTensor, .none:
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

    private func parseBoolAttr(_ attrs: [(name: String, value: String)], _ key: String) -> Bool {
        guard let value = attrValue(attrs, key)?.lowercased() else { return false }
        return value == "true" || value == "1"
    }

    private func parseShape(_ str: String) -> [Int] {
        // Parse "[2,3]" or "2,3"
        let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        return cleaned.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    private func parseIntList(_ str: String) -> [Int] {
        let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        return cleaned.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
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
                  let end = Int(parts[1].trimmingCharacters(in: .whitespaces)) else {
                return nil
            }
            return (start, end)
        }
    }

    /// Parse "[1:1,0:0]" into [(Int,Int)] padding pairs
    private func parsePadding(_ str: String) -> [(Int, Int)] {
        let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        return cleaned.split(separator: ",").compactMap { part in
            let parts = part.trimmingCharacters(in: .whitespaces).split(separator: ":")
            guard parts.count == 2,
                  let before = Int(parts[0].trimmingCharacters(in: .whitespaces)),
                  let after = Int(parts[1].trimmingCharacters(in: .whitespaces)) else {
                return nil
            }
            return (before, after)
        }
    }
}

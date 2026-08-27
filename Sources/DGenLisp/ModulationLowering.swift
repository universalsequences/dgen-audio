import Foundation

enum ModulationMode: String, Codable {
    case additive
    case multiplicative
    case semitone
}

struct TopLevelModulator {
    let name: String?
    let channel: Int
    let slot: Int
}

struct TopLevelModulationParam {
    let identity: ParamIdentity
    let mode: ModulationMode
    let min: Float
    let max: Float
    let unit: String?
    let depthMin: Float
    let depthMax: Float
    let activeParamName: String
    let resolvedSymbolName: String
}

private struct PreScanResult {
    let modulatorsBySlot: [Int: TopLevelModulator]
    let modulatableParams: [String: TopLevelModulationParam]
    let paramNamespace: ParamNamespace
}

func lowerModulation(in nodes: [ASTNode]) throws -> [ASTNode] {
    let preScan = try preScanModulation(nodes)

    var lowered: [ASTNode] = []
    for node in nodes {
        if let expandedNodes = try lowerTopLevelNode(node, preScan: preScan) {
            lowered.append(contentsOf: expandedNodes)
        } else {
            lowered.append(try rewriteModExpressions(
                node,
                modulatableParams: preScan.modulatableParams,
                paramNamespace: preScan.paramNamespace
            ))
        }
    }
    return lowered
}

private func preScanModulation(_ nodes: [ASTNode]) throws -> PreScanResult {
    var modulatorsBySlot: [Int: TopLevelModulator] = [:]
    var modulatableParams: [String: TopLevelModulationParam] = [:]
    var paramNamespace = ParamNamespace()

    for node in nodes {
        guard case .list(let elements) = node,
              let op = listHead(elements)
        else { continue }

        switch op {
        case "def":
            guard elements.count >= 3,
                  case .atom(let defName) = elements[1],
                  case .list(let innerElements) = elements[2],
                  listHead(innerElements) == "in"
            else {
                continue
            }

            let regularArgs = regularArgs(from: innerElements)
            let attributes = attributePairs(from: innerElements)
            if let slotString = attributes["@modulator"] {
                let slot = try parsePositiveInt(slotString, context: "@modulator")
                let channel = try parseInputChannel(regularArgs)
                if modulatorsBySlot[slot] != nil {
                    throw LispError.validationError("duplicate @modulator slot \(slot)")
                }
                modulatorsBySlot[slot] = TopLevelModulator(
                    name: defName,
                    channel: channel,
                    slot: slot
                )
            }

        case "in":
            let regularArgs = regularArgs(from: elements)
            let attributes = attributePairs(from: elements)
            if let slotString = attributes["@modulator"] {
                let slot = try parsePositiveInt(slotString, context: "@modulator")
                let channel = try parseInputChannel(regularArgs)
                if modulatorsBySlot[slot] != nil {
                    throw LispError.validationError("duplicate @modulator slot \(slot)")
                }
                modulatorsBySlot[slot] = TopLevelModulator(
                    name: attributes["@name"],
                    channel: channel,
                    slot: slot
                )
            }

        case "param":
            let regularArgs = regularArgs(from: elements)
            let attributes = attributePairs(from: elements)
            guard let name = firstAtom(in: regularArgs) else { continue }
            let identity = try paramNamespace.declare(
                shortName: name, group: attributes["@group"])
            let isModulatable = parseBool(attributes["@mod"], defaultValue: false)
            if !isModulatable {
                if attributes["@mod-mode"] != nil {
                    throw LispError.validationError(
                        "param '\(name)' has @mod-mode but is missing @mod true")
                }
                continue
            }

            guard let modeRaw = attributes["@mod-mode"],
                  let mode = ModulationMode(rawValue: modeRaw.lowercased())
            else {
                throw LispError.validationError(
                    "param '\(name)' must declare a valid @mod-mode")
            }

            guard let min = parseFloat(attributes["@min"]),
                  let max = parseFloat(attributes["@max"])
            else {
                throw LispError.validationError(
                    "modulatable param '\(name)' requires @min and @max")
            }

            let depthRange = try resolveDepthRange(
                mode: mode,
                paramName: name,
                paramMin: min,
                paramMax: max,
                attributes: attributes
            )

            modulatableParams[identity.canonicalName] = TopLevelModulationParam(
                identity: identity,
                mode: mode,
                min: min,
                max: max,
                unit: attributes["@unit"],
                depthMin: depthRange.min,
                depthMax: depthRange.max,
                activeParamName: "__mod__\(identity.canonicalName)__active",
                resolvedSymbolName: "__mod__\(identity.canonicalName)__resolved"
            )

        default:
            continue
        }
    }

    if !modulatableParams.isEmpty && modulatorsBySlot.isEmpty {
        throw LispError.validationError(
            "patch declares modulatable params but no inputs marked with @modulator")
    }

    return PreScanResult(
        modulatorsBySlot: modulatorsBySlot,
        modulatableParams: modulatableParams,
        paramNamespace: paramNamespace
    )
}

private func lowerTopLevelNode(_ node: ASTNode, preScan: PreScanResult) throws -> [ASTNode]? {
    guard case .list(let elements) = node,
          listHead(elements) == "param"
    else {
        return nil
    }

    let regular = regularArgs(from: elements)
    let attrs = attributePairs(from: elements)
    guard let name = firstAtom(in: regular),
          let identity = preScan.paramNamespace.identity(
            shortName: name, group: attrs["@group"])
    else {
        return [node]
    }
    guard let modParam = preScan.modulatableParams[identity.canonicalName] else {
        return [node]
    }
    var rebuiltElements = elements
    let rebuiltAttributes = mergeAttributes(
        attrs,
        additions: [
            ("@mod", "true"),
            ("@mod-active-param", modParam.activeParamName),
            ("@mod-mode", modParam.mode.rawValue),
            ("@mod-depth-min", formatFloat(modParam.depthMin)),
            ("@mod-depth-max", formatFloat(modParam.depthMax)),
            ("@mod-resolved-symbol", modParam.resolvedSymbolName),
        ]
    )
    rebuiltElements = rebuildTopLevelForm(elements: elements, attributes: rebuiltAttributes)

    let generatedActive = makeParamNode(
        name: modParam.activeParamName,
        attributes: [
            ("@default", "0"),
            ("@min", "0"),
            ("@max", "1"),
            ("@hidden", "true"),
            ("@generated", "modulation-active"),
            ("@generated-for", modParam.identity.canonicalName),
        ]
    )

    let generatedDepths = preScan.modulatorsBySlot.keys.sorted().map { slot -> ASTNode in
        var depthAttributes: [(String, String)] = [
            ("@default", "0"),
            ("@min", formatFloat(modParam.depthMin)),
            ("@max", formatFloat(modParam.depthMax)),
            ("@hidden", "true"),
            ("@generated", "modulation-depth"),
            ("@generated-for", modParam.identity.canonicalName),
            ("@modulator-slot", String(slot)),
        ]
        if let unit = modParam.unit {
            depthAttributes.append(("@unit", unit))
        }
        return makeParamNode(
            name: depthParamName(paramName: modParam.identity.canonicalName, slot: slot),
            attributes: depthAttributes
        )
    }

    let resolvedDef = makeResolvedDef(
        param: modParam,
        modulatorsBySlot: preScan.modulatorsBySlot
    )

    return [ASTNode.list(rebuiltElements), generatedActive] + generatedDepths + [resolvedDef]
}

private func rewriteModExpressions(
    _ node: ASTNode,
    modulatableParams: [String: TopLevelModulationParam],
    paramNamespace: ParamNamespace
) throws -> ASTNode {
    switch node {
    case .atom(let value):
        guard value.hasSuffix("~") else { return node }
        let reference = String(value.dropLast())
        guard let identity = try paramNamespace.resolve(reference, requiresParameter: true) else {
            throw LispError.validationError("'\(reference)' does not reference a parameter")
        }
        guard let modParam = modulatableParams[identity.canonicalName] else {
            throw LispError.validationError(
                "mod: parameter '\(identity.canonicalName)' is not declared with @mod true")
        }
        return .atom(modParam.resolvedSymbolName)

    case .list(let elements):
        guard let head = listHead(elements) else {
            return .list(try elements.map {
                try rewriteModExpressions(
                    $0,
                    modulatableParams: modulatableParams,
                    paramNamespace: paramNamespace
                )
            })
        }

        if head == "mod", elements.count == 2, case .atom(let name) = elements[1] {
            guard let identity = try paramNamespace.resolve(name, requiresParameter: true) else {
                throw LispError.validationError("'\(name)' does not reference a parameter")
            }
            guard let modParam = modulatableParams[identity.canonicalName] else {
                throw LispError.validationError(
                    "mod: parameter '\(identity.canonicalName)' is not declared with @mod true")
            }
            return .atom(modParam.resolvedSymbolName)
        }

        // Operator names and binding targets are declarations, not value
        // references. Preserve them while rewriting expression positions.
        let preservedCount: Int
        switch head {
        case "def": preservedCount = min(2, elements.count)
        case "defmacro": preservedCount = min(3, elements.count)
        default: preservedCount = min(1, elements.count)
        }
        let prefix = Array(elements.prefix(preservedCount))
        let expressions = try elements.dropFirst(preservedCount).map {
            try rewriteModExpressions(
                $0,
                modulatableParams: modulatableParams,
                paramNamespace: paramNamespace
            )
        }
        return .list(prefix + expressions)
    }
}

private func makeResolvedDef(
    param: TopLevelModulationParam,
    modulatorsBySlot: [Int: TopLevelModulator]
) -> ASTNode {
    let laneArgs: [ASTNode] = modulatorsBySlot.keys.sorted().flatMap { slot in
        guard let modulator = modulatorsBySlot[slot], let name = modulator.name else {
            return [ASTNode]()
        }
        return [
            .atom(name),
            .atom(depthParamName(paramName: param.identity.canonicalName, slot: slot)),
        ]
    }

    let resolvedExpr = ASTNode.list(
        [
            .atom("__modulated-param"),
            .atom(param.identity.canonicalName),
            .atom(param.activeParamName),
        ] + laneArgs + [
            .atom("@mode"),
            .atom(param.mode.rawValue),
            .atom("@min"),
            .atom(formatFloat(param.min)),
            .atom("@max"),
            .atom(formatFloat(param.max)),
        ]
    )

    return ASTNode.list([
        .atom("def"),
        .atom(param.resolvedSymbolName),
        resolvedExpr,
    ])
}

private func depthParamName(paramName: String, slot: Int) -> String {
    "__mod__\(paramName)__depth__slot\(slot)"
}

private func makeParamNode(name: String, attributes: [(String, String)]) -> ASTNode {
    var elements: [ASTNode] = [.atom("param"), .atom(name)]
    for (key, value) in attributes {
        elements.append(.atom(key))
        elements.append(.atom(value))
    }
    return .list(elements)
}

private func rebuildTopLevelForm(
    elements: [ASTNode],
    attributes: [(String, String)]
) -> [ASTNode] {
    let op = elements.first ?? .atom("")
    let regular = regularArgs(from: elements)
    var rebuilt = [op] + regular
    for (key, value) in attributes {
        rebuilt.append(.atom(key))
        rebuilt.append(.atom(value))
    }
    return rebuilt
}

private func mergeAttributes(
    _ original: [String: String],
    additions: [(String, String)]
) -> [(String, String)] {
    var merged = original
    for (key, value) in additions {
        merged[key] = value
    }
    return merged.sorted { $0.key < $1.key }
}

private func attributePairs(from elements: [ASTNode]) -> [String: String] {
    let args = Array(elements.dropFirst())
    var result: [String: String] = [:]
    var index = 0
    while index < args.count {
        if case .atom(let key) = args[index], key.hasPrefix("@") {
            let value: String
            if index + 1 < args.count, case .atom(let atomValue) = args[index + 1] {
                value = atomValue
                index += 2
            } else {
                value = ""
                index += 1
            }
            result[key] = value
        } else {
            index += 1
        }
    }
    return result
}

private func regularArgs(from elements: [ASTNode]) -> [ASTNode] {
    let args = Array(elements.dropFirst())
    var regular: [ASTNode] = []
    var index = 0
    while index < args.count {
        if case .atom(let key) = args[index], key.hasPrefix("@") {
            if index + 1 < args.count, case .atom = args[index + 1] {
                index += 2
            } else {
                index += 1
            }
            continue
        }
        regular.append(args[index])
        index += 1
    }
    return regular
}

private func listHead(_ elements: [ASTNode]) -> String? {
    guard let first = elements.first, case .atom(let op) = first else { return nil }
    return op.lowercased()
}

private func firstAtom(in nodes: [ASTNode]) -> String? {
    guard let first = nodes.first, case .atom(let value) = first else { return nil }
    return value
}

private func parseFloat(_ string: String?) -> Float? {
    guard let string else { return nil }
    return Float(string)
}

private func parseBool(_ string: String?, defaultValue: Bool) -> Bool {
    guard let string else { return defaultValue }
    switch string.lowercased() {
    case "1", "true":
        return true
    case "0", "false":
        return false
    default:
        return defaultValue
    }
}

private func parsePositiveInt(_ string: String, context: String) throws -> Int {
    guard let value = Int(string), value > 0 else {
        throw LispError.validationError("\(context) must be a positive integer")
    }
    return value
}

private func parseInputChannel(_ regularArgs: [ASTNode]) throws -> Int {
    guard let channelName = firstAtom(in: regularArgs),
          let lispChannel = Int(channelName),
          lispChannel > 0
    else {
        throw LispError.validationError("in requires a positive channel number")
    }
    return lispChannel - 1
}

private func resolveDepthRange(
    mode: ModulationMode,
    paramName: String,
    paramMin: Float,
    paramMax: Float,
    attributes: [String: String]
) throws -> (min: Float, max: Float) {
    if let depthMin = parseFloat(attributes["@mod-depth-min"]),
       let depthMax = parseFloat(attributes["@mod-depth-max"]) {
        return (depthMin, depthMax)
    }

    switch mode {
    case .additive:
        let span = paramMax - paramMin
        return (-span, span)
    case .multiplicative:
        return (-1, 1)
    case .semitone:
        return (-24, 24)
    }
}

private func formatFloat(_ value: Float) -> String {
    if value.rounded() == value {
        return String(Int(value))
    }
    return String(value)
}

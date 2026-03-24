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
    let name: String
    let mode: ModulationMode
    let min: Float
    let max: Float
    let unit: String?
    let depthMin: Float
    let depthMax: Float
    let sourceParamName: String
    let depthParamName: String
    let resolvedSymbolName: String
}

private struct PreScanResult {
    let modulatorsBySlot: [Int: TopLevelModulator]
    let modulatableParams: [String: TopLevelModulationParam]
    let allParamNames: Set<String>
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
                allParamNames: preScan.allParamNames
            ))
        }
    }
    return lowered
}

private func preScanModulation(_ nodes: [ASTNode]) throws -> PreScanResult {
    var modulatorsBySlot: [Int: TopLevelModulator] = [:]
    var modulatableParams: [String: TopLevelModulationParam] = [:]
    var allParamNames: Set<String> = []

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
            allParamNames.insert(name)
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

            modulatableParams[name] = TopLevelModulationParam(
                name: name,
                mode: mode,
                min: min,
                max: max,
                unit: attributes["@unit"],
                depthMin: depthRange.min,
                depthMax: depthRange.max,
                sourceParamName: "__mod__\(name)__source",
                depthParamName: "__mod__\(name)__depth",
                resolvedSymbolName: "__mod__\(name)__resolved"
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
        allParamNames: allParamNames
    )
}

private func lowerTopLevelNode(_ node: ASTNode, preScan: PreScanResult) throws -> [ASTNode]? {
    guard case .list(let elements) = node,
          listHead(elements) == "param"
    else {
        return nil
    }

    let regular = regularArgs(from: elements)
    guard let name = firstAtom(in: regular),
          let modParam = preScan.modulatableParams[name]
    else {
        return nil
    }

    let attrs = attributePairs(from: elements)
    var rebuiltElements = elements
    let rebuiltAttributes = mergeAttributes(
        attrs,
        additions: [
            ("@mod", "true"),
            ("@mod-mode", modParam.mode.rawValue),
            ("@mod-depth-min", formatFloat(modParam.depthMin)),
            ("@mod-depth-max", formatFloat(modParam.depthMax)),
            ("@mod-source-param", modParam.sourceParamName),
            ("@mod-depth-param", modParam.depthParamName),
            ("@mod-resolved-symbol", modParam.resolvedSymbolName),
        ]
    )
    rebuiltElements = rebuildTopLevelForm(elements: elements, attributes: rebuiltAttributes)

    let generatedSource = makeParamNode(
        name: modParam.sourceParamName,
        attributes: [
            ("@default", "0"),
            ("@min", "0"),
            ("@max", formatFloat(Float(preScan.modulatorsBySlot.keys.max() ?? 0))),
            ("@hidden", "true"),
            ("@generated", "modulation-source"),
            ("@generated-for", modParam.name),
        ]
    )

    var depthAttributes: [(String, String)] = [
        ("@default", "0"),
        ("@min", formatFloat(modParam.depthMin)),
        ("@max", formatFloat(modParam.depthMax)),
        ("@hidden", "true"),
        ("@generated", "modulation-depth"),
        ("@generated-for", modParam.name),
    ]
    if let unit = modParam.unit {
        depthAttributes.append(("@unit", unit))
    }
    let generatedDepth = makeParamNode(
        name: modParam.depthParamName,
        attributes: depthAttributes
    )

    let resolvedDef = makeResolvedDef(
        param: modParam,
        modulatorsBySlot: preScan.modulatorsBySlot
    )

    return [ASTNode.list(rebuiltElements), generatedSource, generatedDepth, resolvedDef]
}

private func rewriteModExpressions(
    _ node: ASTNode,
    modulatableParams: [String: TopLevelModulationParam],
    allParamNames: Set<String>
) throws -> ASTNode {
    switch node {
    case .atom:
        return node

    case .list(let elements):
        guard let head = listHead(elements) else {
            return .list(try elements.map {
                try rewriteModExpressions(
                    $0,
                    modulatableParams: modulatableParams,
                    allParamNames: allParamNames
                )
            })
        }

        if head == "mod", elements.count == 2, case .atom(let name) = elements[1] {
            guard allParamNames.contains(name) else {
                throw LispError.validationError(
                    "mod: '\(name)' does not reference a parameter")
            }
            guard let modParam = modulatableParams[name] else {
                throw LispError.validationError(
                    "mod: parameter '\(name)' is not declared with @mod true")
            }
            return .atom(modParam.resolvedSymbolName)
        }

        return .list(try elements.map {
            try rewriteModExpressions(
                $0,
                modulatableParams: modulatableParams,
                allParamNames: allParamNames
            )
        })
    }
}

private func makeResolvedDef(
    param: TopLevelModulationParam,
    modulatorsBySlot: [Int: TopLevelModulator]
) -> ASTNode {
    let selectorOptions: [ASTNode] = (1...(modulatorsBySlot.keys.max() ?? 0)).map { slot in
        if let modulator = modulatorsBySlot[slot], let name = modulator.name {
            return .atom(name)
        }
        return .atom("0")
    }

    let selectorExpr = ASTNode.list(
        [.atom("selector"), .atom(param.sourceParamName)] + selectorOptions
    )

    let productExpr = ASTNode.list([
        .atom("*"),
        selectorExpr,
        .atom(param.depthParamName),
    ])

    let resolvedExpr: ASTNode
    switch param.mode {
    case .additive:
        resolvedExpr = ASTNode.list([
            .atom("clip"),
            ASTNode.list([.atom("+"), .atom(param.name), productExpr]),
            .atom(formatFloat(param.min)),
            .atom(formatFloat(param.max)),
        ])
    case .multiplicative:
        resolvedExpr = ASTNode.list([
            .atom("clip"),
            ASTNode.list([
                .atom("*"),
                .atom(param.name),
                ASTNode.list([
                    .atom("+"),
                    .atom("1"),
                    productExpr,
                ]),
            ]),
            .atom(formatFloat(param.min)),
            .atom(formatFloat(param.max)),
        ])
    case .semitone:
        resolvedExpr = ASTNode.list([
            .atom("*"),
            .atom(param.name),
            ASTNode.list([
                .atom("exp"),
                ASTNode.list([
                    .atom("*"),
                    ASTNode.list([.atom("log"), .atom("2")]),
                    ASTNode.list([
                        .atom("/"),
                        productExpr,
                        .atom("12"),
                    ]),
                ]),
            ]),
        ])
    }

    return ASTNode.list([
        .atom("def"),
        .atom(param.resolvedSymbolName),
        resolvedExpr,
    ])
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

// Manifest - JSON manifest generation for compiled patches
//
// Produces a JSON manifest with all metadata needed by a host application
// (DAW plugin, etc.) to load and interact with the compiled dylib.

import DGen
import DGenLazy
import Foundation

// MARK: - Manifest types

struct PatchManifest: Codable {
    let version: Int
    let processAbi: String
    let dylib: String
    let cSourcePath: String
    let sampleRate: Float
    let maxFrameCount: Int
    let voiceCount: Int
    let voiceCellId: Int?
    let totalMemorySlots: Int
    let params: [ManifestParam]
    let groups: [ManifestGroup]
    let envelopes: [ManifestEnvelope]
    let inputs: [ManifestInput]
    let outputs: [ManifestOutput]
    let modulators: [ManifestModulator]
    let modOutputs: [ManifestModOutput]
    let modDestinations: [ManifestModDestination]
    let tensors: [ManifestTensor]
    let tensorInitData: [ManifestTensorInit]
}

struct ManifestParam: Codable {
    let name: String
    let cellId: Int
    let cellSpan: Int
    let defaultValue: Float  // JSON key: "default"
    let min: Float?
    let max: Float?
    let unit: String?
    let hidden: Bool?
    let group: String?
    let env: String?
    let role: String?

    enum CodingKeys: String, CodingKey {
        case name, cellId, cellSpan
        case defaultValue = "default"
        case min, max, unit, hidden, group, env, role
    }
}

struct ManifestGroup: Codable {
    let name: String
}

struct ManifestEnvelope: Codable {
    let name: String
    let group: String?
    let roles: ManifestEnvelopeRoles

    enum CodingKeys: String, CodingKey {
        case name, group, roles
    }

    init(name: String, group: String?, roles: ManifestEnvelopeRoles) {
        self.name = name
        self.group = group
        self.roles = roles
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)
        group = try container.decodeIfPresent(String.self, forKey: .group)
        roles = try container.decode(ManifestEnvelopeRoles.self, forKey: .roles)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        if let group {
            try container.encode(group, forKey: .group)
        } else {
            try container.encodeNil(forKey: .group)
        }
        try container.encode(roles, forKey: .roles)
    }
}

struct ManifestEnvelopeRoles: Codable {
    let attack: String?
    let decay: String?
    let sustain: String?
    let release: String?

    enum CodingKeys: String, CodingKey {
        case attack, decay, sustain, release
    }

    init(attack: String?, decay: String?, sustain: String?, release: String?) {
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        attack = try container.decodeIfPresent(String.self, forKey: .attack)
        decay = try container.decodeIfPresent(String.self, forKey: .decay)
        sustain = try container.decodeIfPresent(String.self, forKey: .sustain)
        release = try container.decodeIfPresent(String.self, forKey: .release)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try encodeNullable(attack, in: &container, forKey: .attack)
        try encodeNullable(decay, in: &container, forKey: .decay)
        try encodeNullable(sustain, in: &container, forKey: .sustain)
        try encodeNullable(release, in: &container, forKey: .release)
    }

    private func encodeNullable(
        _ value: String?,
        in container: inout KeyedEncodingContainer<CodingKeys>,
        forKey key: CodingKeys
    ) throws {
        if let value {
            try container.encode(value, forKey: key)
        } else {
            try container.encodeNil(forKey: key)
        }
    }
}

struct ManifestInput: Codable {
    let channel: Int
    let name: String?
}

struct ManifestOutput: Codable {
    let channel: Int
    let name: String?
}

struct ManifestModulator: Codable {
    let slot: Int
    let inputChannel: Int
    let name: String?
}

struct ManifestModOutput: Codable {
    let slot: Int
    let channel: Int
    let name: String?
    let range: String
}

struct ManifestModDestination: Codable {
    let name: String
    let paramCellId: Int
    let mode: String
    let activeCellId: Int
    let depthLanes: [ManifestModDepthLane]
    let min: Float
    let max: Float
    let unit: String?
    let depthMin: Float?
    let depthMax: Float?
}

struct ManifestModDepthLane: Codable {
    let slot: Int
    let depthCellId: Int
}

struct ManifestTensorInit: Codable {
    let offset: Int
    let data: [Float]
}

struct ManifestTensor: Codable {
    let name: String
    let cellOffset: Int
    let shape: [Int]
    let kind: String
    let mutable: Bool
    let sourceFile: String?
    let sourceSampleRate: Float?
}

// MARK: - Manifest generation

func generateManifest(
    compilerResult: CompilerResult,
    evaluator: LispEvaluator,
    options: CompilerOptions
) -> PatchManifest {
    let compilation = compilerResult.compilationResult
    let cellMappings = compilation.cellAllocations.cellMappings
    let cellVectorWidths = compilation.cellAllocations.cellVectorWidths
    let cellAllocationSizes = compilation.graph.cellAllocationSizes

    // Map param cell IDs to physical cell IDs
    let manifestParams = evaluator.params.map { param -> ManifestParam in
        let physicalCellId: Int
        let cellSpan: Int
        if let logicalId = param.cellId {
            physicalCellId = cellMappings[logicalId] ?? logicalId
            cellSpan = max(
                cellVectorWidths[logicalId] ?? 1,
                cellAllocationSizes[logicalId] ?? 1
            )
        } else {
            physicalCellId = -1
            cellSpan = 1
        }
        return ManifestParam(
            name: param.name,
            cellId: physicalCellId,
            cellSpan: cellSpan,
            defaultValue: param.defaultValue,
            min: param.min,
            max: param.max,
            unit: param.unit,
            hidden: param.hidden ? true : nil,
            group: param.group,
            env: param.env,
            role: param.role?.rawValue
        )
    }

    let manifestGroups = deriveManifestGroups(from: evaluator.params)
    let manifestEnvelopes = deriveManifestEnvelopes(from: evaluator.params)

    let manifestInputs = evaluator.inputs.map { input in
        ManifestInput(channel: input.channel, name: input.name)
    }

    let manifestOutputs = evaluator.outputs.map { output in
        ManifestOutput(channel: output.channel, name: output.name)
    }

    let manifestModulators = evaluator.inputs.compactMap { input -> ManifestModulator? in
        guard let slot = input.modulatorSlot else { return nil }
        return ManifestModulator(slot: slot, inputChannel: input.channel, name: input.name)
    }
    .sorted { $0.slot < $1.slot }

    let manifestModOutputs = evaluator.outputs.compactMap { output -> ManifestModOutput? in
        guard let slot = output.modulatorSlot else { return nil }
        return ManifestModOutput(
            slot: slot,
            channel: output.channel,
            name: output.name,
            range: "unipolar"
        )
    }
    .sorted { $0.slot < $1.slot }

    let paramsByName = Dictionary(uniqueKeysWithValues: evaluator.params.map { ($0.name, $0) })
    let manifestModDestinations = evaluator.params.compactMap { param -> ManifestModDestination? in
        guard let mode = param.modulationMode,
              let activeName = param.modulationActiveParamName,
              let min = param.min,
              let max = param.max,
              let paramCell = param.cellId,
              let activeParam = paramsByName[activeName],
              let activeCell = activeParam.cellId
        else {
            return nil
        }

        let depthLanes = evaluator.params.compactMap { depthParam -> ManifestModDepthLane? in
            guard depthParam.generatedKind == "modulation-depth",
                  depthParam.generatedFor == param.name,
                  let slot = depthParam.generatedModulatorSlot,
                  let depthCell = depthParam.cellId
            else {
                return nil
            }
            return ManifestModDepthLane(
                slot: slot,
                depthCellId: cellMappings[depthCell] ?? depthCell
            )
        }
        .sorted { $0.slot < $1.slot }

        return ManifestModDestination(
            name: param.name,
            paramCellId: cellMappings[paramCell] ?? paramCell,
            mode: mode.rawValue,
            activeCellId: cellMappings[activeCell] ?? activeCell,
            depthLanes: depthLanes,
            min: min,
            max: max,
            unit: param.unit,
            depthMin: param.modulationDepthMin,
            depthMax: param.modulationDepthMax
        )
    }

    // Collect tensor init data
    let tensorInitPairs = collectTensorInitData(
        graph: compilerResult.compilationResult.graph,
        cellAllocations: compilerResult.compilationResult.cellAllocations
    )
    let manifestTensorInit = tensorInitPairs.map { (offset, data) in
        ManifestTensorInit(offset: offset, data: data)
    }
    let manifestTensors = mapTensorMetadata(
        evaluatorTensors: evaluator.tensors,
        tensorInitPairs: tensorInitPairs
    )

    return PatchManifest(
        version: 2,
        processAbi: "dgen-c-v2-host-sample-rate",
        dylib: "\(options.name).dylib",
        cSourcePath: compilerResult.cSourcePath,
        sampleRate: options.sampleRate,
        maxFrameCount: options.maxFrames,
        voiceCount: options.voiceCount,
        voiceCellId: compilerResult.compilationResult.voiceCellId.flatMap { cellMappings[$0] ?? $0 },
        totalMemorySlots: compilerResult.compilationResult.totalMemorySlots,
        params: manifestParams,
        groups: manifestGroups,
        envelopes: manifestEnvelopes,
        inputs: manifestInputs,
        outputs: manifestOutputs,
        modulators: manifestModulators,
        modOutputs: manifestModOutputs,
        modDestinations: manifestModDestinations,
        tensors: manifestTensors,
        tensorInitData: manifestTensorInit
    )
}

private func deriveManifestGroups(from params: [ParamInfo]) -> [ManifestGroup] {
    var seen = Set<String>()
    var groups: [ManifestGroup] = []
    for param in params {
        guard let group = param.group, !seen.contains(group) else { continue }
        seen.insert(group)
        groups.append(ManifestGroup(name: group))
    }
    return groups
}

private func deriveManifestEnvelopes(from params: [ParamInfo]) -> [ManifestEnvelope] {
    struct EnvelopeAccumulator {
        var group: String?
        var roles: [UIEnvelopeRole: String] = [:]
    }

    var order: [String] = []
    var byName: [String: EnvelopeAccumulator] = [:]

    for param in params {
        guard let env = param.env else { continue }
        if byName[env] == nil {
            order.append(env)
            byName[env] = EnvelopeAccumulator()
        }
        if let group = param.group {
            byName[env]?.group = group
        }
        if let role = param.role {
            byName[env]?.roles[role] = param.name
        }
    }

    return order.compactMap { name in
        guard let envelope = byName[name] else { return nil }
        return ManifestEnvelope(
            name: name,
            group: envelope.group,
            roles: ManifestEnvelopeRoles(
                attack: envelope.roles[.attack],
                decay: envelope.roles[.decay],
                sustain: envelope.roles[.sustain],
                release: envelope.roles[.release]
            )
        )
    }
}

private func mapTensorMetadata(
    evaluatorTensors: [TensorInfo],
    tensorInitPairs: [(Int, [Float])]
) -> [ManifestTensor] {
    var searchStart = 0
    return evaluatorTensors.enumerated().compactMap { index, info in
        let expectedCount = info.shape.reduce(1, *)
        var matchIndex: Int? = nil

        for i in searchStart..<tensorInitPairs.count {
            let data = tensorInitPairs[i].1
            guard data.count == expectedCount else { continue }
            if let expectedData = info.data, expectedData.count == data.count {
                let prefixMatches = zip(expectedData.prefix(16), data.prefix(16)).allSatisfy { abs($0 - $1) < 0.000001 }
                if !prefixMatches { continue }
            }
            matchIndex = i
            break
        }

        guard let i = matchIndex else { return nil }
        searchStart = i + 1
        return ManifestTensor(
            name: info.name.isEmpty ? "tensor\(index)" : info.name,
            cellOffset: tensorInitPairs[i].0,
            shape: info.shape,
            kind: info.kind,
            mutable: info.mutable,
            sourceFile: info.sourceFile,
            sourceSampleRate: info.sourceSampleRate
        )
    }
}

func writeManifest(_ manifest: PatchManifest, to dir: String, name: String) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(manifest)
    let jsonString = String(data: data, encoding: .utf8)!

    let path = "\(dir)/\(name).json"
    try jsonString.write(toFile: path, atomically: true, encoding: .utf8)

    return jsonString
}

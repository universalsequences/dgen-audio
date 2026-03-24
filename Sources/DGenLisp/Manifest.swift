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
    let dylib: String
    let cSourcePath: String
    let sampleRate: Float
    let maxFrameCount: Int
    let voiceCount: Int
    let voiceCellId: Int?
    let totalMemorySlots: Int
    let params: [ManifestParam]
    let inputs: [ManifestInput]
    let outputs: [ManifestOutput]
    let modulators: [ManifestModulator]
    let modDestinations: [ManifestModDestination]
    let tensorInitData: [ManifestTensorInit]
}

struct ManifestParam: Codable {
    let name: String
    let cellId: Int
    let defaultValue: Float  // JSON key: "default"
    let min: Float?
    let max: Float?
    let unit: String?
    let hidden: Bool?

    enum CodingKeys: String, CodingKey {
        case name, cellId
        case defaultValue = "default"
        case min, max, unit, hidden
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

struct ManifestModDestination: Codable {
    let name: String
    let paramCellId: Int
    let mode: String
    let sourceCellId: Int
    let depthCellId: Int
    let min: Float
    let max: Float
    let unit: String?
    let depthMin: Float?
    let depthMax: Float?
}

struct ManifestTensorInit: Codable {
    let offset: Int
    let data: [Float]
}

// MARK: - Manifest generation

func generateManifest(
    compilerResult: CompilerResult,
    evaluator: LispEvaluator,
    options: CompilerOptions
) -> PatchManifest {
    let cellMappings = compilerResult.compilationResult.cellAllocations.cellMappings

    // Map param cell IDs to physical cell IDs
    let manifestParams = evaluator.params.map { param -> ManifestParam in
        let physicalCellId: Int
        if let logicalId = param.cellId {
            physicalCellId = cellMappings[logicalId] ?? logicalId
        } else {
            physicalCellId = -1
        }
        return ManifestParam(
            name: param.name,
            cellId: physicalCellId,
            defaultValue: param.defaultValue,
            min: param.min,
            max: param.max,
            unit: param.unit,
            hidden: param.hidden ? true : nil
        )
    }

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

    let paramsByName = Dictionary(uniqueKeysWithValues: evaluator.params.map { ($0.name, $0) })
    let manifestModDestinations = evaluator.params.compactMap { param -> ManifestModDestination? in
        guard let mode = param.modulationMode,
              let sourceName = param.modulationSourceParamName,
              let depthName = param.modulationDepthParamName,
              let min = param.min,
              let max = param.max,
              let paramCell = param.cellId,
              let sourceParam = paramsByName[sourceName],
              let depthParam = paramsByName[depthName],
              let sourceCell = sourceParam.cellId,
              let depthCell = depthParam.cellId
        else {
            return nil
        }

        return ManifestModDestination(
            name: param.name,
            paramCellId: cellMappings[paramCell] ?? paramCell,
            mode: mode.rawValue,
            sourceCellId: cellMappings[sourceCell] ?? sourceCell,
            depthCellId: cellMappings[depthCell] ?? depthCell,
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

    return PatchManifest(
        version: 1,
        dylib: "\(options.name).dylib",
        cSourcePath: compilerResult.cSourcePath,
        sampleRate: options.sampleRate,
        maxFrameCount: options.maxFrames,
        voiceCount: options.voiceCount,
        voiceCellId: compilerResult.compilationResult.voiceCellId.flatMap { cellMappings[$0] ?? $0 },
        totalMemorySlots: compilerResult.compilationResult.totalMemorySlots,
        params: manifestParams,
        inputs: manifestInputs,
        outputs: manifestOutputs,
        modulators: manifestModulators,
        modDestinations: manifestModDestinations,
        tensorInitData: manifestTensorInit
    )
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

// Compiler - CompilationPipeline + clang → dylib
//
// Takes evaluated DGenLazy graph, compiles to C source via CompilationPipeline,
// then invokes clang to produce a shared dylib.

import DGen
import DGenLazy
import Foundation

struct CompilerResult {
    let dylibPath: String
    let cSourcePath: String
    let compilationResult: CompilationResult
    let cSource: String
}

struct CompilerOptions {
    let outputDir: String
    let name: String
    let sampleRate: Float
    let maxFrames: Int
    let voiceCount: Int
    let debug: Bool
}

func compilePatch(
    graph: LazyGraph,
    outputs: [OutputInfo],
    options: CompilerOptions
) throws -> CompilerResult {
    // 1. Add outputs to graph
    for output in outputs {
        graph.addOutput(output.signal, channel: output.channel)
    }

    // 2. Compile graph to C source
    let result = try graph.compileOnly(frameCount: options.maxFrames, voiceCount: options.voiceCount)

    // 3. Extract combined C source
    let cSource = result.kernels.map { $0.source }.joined(separator: "\n\n")

    // 4. Create output directory
    let fm = FileManager.default
    try fm.createDirectory(atPath: options.outputDir, withIntermediateDirectories: true)

    // 5. Write C source to temp file
    let cFilePath = "\(options.outputDir)/\(options.name).c"
    try cSource.write(toFile: cFilePath, atomically: true, encoding: .utf8)

    // 6. Invoke clang
    let dylibPath = "\(options.outputDir)/\(options.name).dylib"

    let compile = Process()
    let invocation = try DGenToolchainPolicy.compileInvocation(
        outputPath: dylibPath,
        sourcePath: cFilePath)
    compile.launchPath = invocation.executable
    let arguments = invocation.arguments

    let errorPipe = Pipe()
    compile.standardError = errorPipe

    var errorData = Data()
    let errorReadQueue = DispatchQueue(label: "dgenlisp.clang.stderr")
    errorPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if !data.isEmpty {
            errorReadQueue.sync { errorData.append(data) }
        }
    }

    compile.arguments = arguments
    compile.launch()
    compile.waitUntilExit()

    errorPipe.fileHandleForReading.readabilityHandler = nil
    let remainingData = errorPipe.fileHandleForReading.readDataToEndOfFile()
    errorReadQueue.sync { errorData.append(remainingData) }

    guard compile.terminationStatus == 0 else {
        let errorStr = String(data: errorData, encoding: .utf8) ?? "Unknown error"
        throw DGenLazyError.compilationFailed("clang failed (exit \(compile.terminationStatus)): \(errorStr)")
    }

    try DGenBinaryAudit.audit(dylibPath)

    if options.debug {
        let errorStr = String(data: errorData, encoding: .utf8) ?? ""
        if !errorStr.isEmpty {
            fputs("[debug] clang warnings: \(errorStr)\n", stderr)
        }
    }

    return CompilerResult(
        dylibPath: dylibPath,
        cSourcePath: cFilePath,
        compilationResult: result,
        cSource: cSource
    )
}

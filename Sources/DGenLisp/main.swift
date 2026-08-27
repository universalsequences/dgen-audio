// DGenLisp - Lisp-to-Dylib Compiler CLI
//
// Usage: dgenlisp compile [<file.lisp>] [options]
//   -o, --output <dir>       Output directory (default: .)
//   --name <name>            Output name (default: patch)
//   --sample-rate <rate>     Sample rate (default: 44100)
//   --max-frames <count>     Max frame count (default: 512)
//   --voices <count>         Voice count for polyphony (default: 1)
//   --asset-base <dir>        Base directory for relative tensor/wavetable files
//   --toolchain-root <dir>   Staged DGen toolchain root (embedded clang/lld)
//   --skip-inline-audit      Skip the post-compile binary audit (host audits)
//   --audit-tool <path>      Explicit audit script for the inline audit
//   --debug                  Debug output
//   -                        Read from stdin (also default if no file given)

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

// The train subcommand has its own protocol-owning argument parser and
// never returns (it owns stdout and the exit code). Route before the
// compile-oriented parser can touch the arguments.
if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "train" {
    TrainCommand.run(
        arguments: Array(CommandLine.arguments.dropFirst(2)),
        realTrainer: RealTrainer.run
    )
}
// Hidden render helper spawned by the trainer (realize() must not share a
// process with backward()).
if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "train-render" {
    TrainRenderCommand.run(arguments: Array(CommandLine.arguments.dropFirst(2)))
}

// MARK: - Argument parsing

struct CLIArgs {
    var command: String = "compile"
    var inputFile: String? = nil
    var outputDir: String = "."
    var name: String = "patch"
    var sampleRate: Float = 44100
    var maxFrames: Int = 512
    var voiceCount: Int = 1
    var assetBase: String? = nil
    var toolchainRoot: String? = nil
    var skipInlineAudit: Bool = false
    var auditTool: String? = nil
    var debug: Bool = false
    var readStdin: Bool = false
}

func parseArgs(_ args: [String]) -> CLIArgs {
    var cli = CLIArgs()
    var i = 1  // skip program name

    while i < args.count {
        let arg = args[i]
        switch arg {
        case "compile":
            cli.command = "compile"
        case "-o", "--output":
            i += 1
            if i < args.count { cli.outputDir = args[i] }
        case "--name":
            i += 1
            if i < args.count { cli.name = args[i] }
        case "--sample-rate":
            i += 1
            if i < args.count { cli.sampleRate = Float(args[i]) ?? 44100 }
        case "--max-frames":
            i += 1
            if i < args.count { cli.maxFrames = Int(args[i]) ?? 512 }
        case "--voices":
            i += 1
            if i < args.count { cli.voiceCount = Int(args[i]) ?? 1 }
        case "--asset-base":
            i += 1
            if i < args.count { cli.assetBase = args[i] }
        case "--toolchain-root":
            i += 1
            if i < args.count { cli.toolchainRoot = args[i] }
        case "--skip-inline-audit":
            cli.skipInlineAudit = true
        case "--audit-tool":
            i += 1
            if i < args.count { cli.auditTool = args[i] }
        case "--debug":
            cli.debug = true
        case "-":
            cli.readStdin = true
        case "--help", "-h":
            printUsage()
            exit(0)
        default:
            if !arg.hasPrefix("-") && cli.inputFile == nil {
                cli.inputFile = arg
            } else {
                fputs("Unknown option: \(arg)\n", stderr)
            }
        }
        i += 1
    }

    // Default to stdin if no file given
    if cli.inputFile == nil {
        cli.readStdin = true
    }

    return cli
}

func printUsage() {
    let usage = """
        Usage: dgenlisp compile [<file.lisp>] [options]
               dgenlisp train --patch <dsp.lisp> --target <sample.wav> \\
                              --seed-params <seed.json> --job-dir <dir> \\
                              [--mode direction] [--epochs N] \\
                              [--gate-frames N] [--pitch-hz F] [--plan-only] \\
                              [--multistart-candidates N] [--multistart-lanes N] \\
                              [--multistart-batch N] [--multistart-steps N] \\
                              [--search legacy|cma-es] [--cma-generations N] \\
                              [--cma-population N] [--cma-sigma F] [--cma-seed N] \\
                              [--cma-forward-batch N] [--cma-continue N] \\
                              [--local-epochs N] [--cma-refine-epochs N] \\
                              [--cma-final-epochs N] [--cma-refine-mode MODE]

        Options:
          -o, --output <dir>       Output directory (default: .)
          --name <name>            Output name (default: patch)
          --sample-rate <rate>     Sample rate (default: 44100)
          --max-frames <count>     Max frame count (default: 512)
          --voices <count>         Voice count for polyphony (default: 1)
          --asset-base <dir>        Base directory for relative tensor/wavetable files
          --toolchain-root <dir>   Staged DGen toolchain root (overrides
                                   DGEN_TOOLCHAIN_STAGE_ROOT)
          --skip-inline-audit      Skip the post-compile binary audit; the
                                   host process audits the artifact itself
          --audit-tool <path>      Audit script for the inline audit
                                   (overrides DGEN_BINARY_AUDIT_TOOL and the
                                   dgen-checkout fallback)
          --debug                  Debug output
          -                        Read from stdin (also default if no file given)
          -h, --help               Show this help
        """
    print(usage)
}

// MARK: - Main

func main() throws {
    let cli = parseArgs(CommandLine.arguments)

    guard cli.command == "compile" else {
        fputs("Unknown command: \(cli.command). Only 'compile' is supported.\n", stderr)
        exit(1)
    }

    // Read source
    let source: String
    var inputDirectory = FileManager.default.currentDirectoryPath
    if cli.readStdin {
        source = readStdin()
    } else if let file = cli.inputFile {
        guard FileManager.default.fileExists(atPath: file) else {
            fputs("Error: File not found: \(file)\n", stderr)
            exit(1)
        }
        inputDirectory = URL(fileURLWithPath: file).deletingLastPathComponent().path
        source = try String(contentsOfFile: file, encoding: .utf8)
    } else {
        fputs("Error: No input file specified and stdin is empty\n", stderr)
        exit(1)
    }

    guard !source.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        fputs("Error: Empty input\n", stderr)
        exit(1)
    }

    // Configure DGen
    DGenConfig.backend = .c
    DGenConfig.sampleRate = cli.sampleRate
    DGenConfig.maxFrameCount = cli.maxFrames
    DGenConfig.debug = cli.debug
    if ProcessInfo.processInfo.environment["DGEN_NO_REUSE"] != nil {
        DGenConfig.enableBufferReuse = false
    }

    // Reset graph
    LazyGraphContext.reset()
    let graph = LazyGraphContext.current

    // Evaluate lisp source
    let assetBase = cli.assetBase ?? inputDirectory
    let evaluator = LispEvaluator(sourceDirectory: URL(fileURLWithPath: assetBase, isDirectory: true))
    do {
        let parsedNodes = try parseSource(source)
        let loweredNodes = try lowerModulation(in: parsedNodes)
        try evaluator.evaluate(nodes: loweredNodes)
    } catch let error as LispError {
        fputs("Error: \(error.message)\n", stderr)
        exit(1)
    }

    guard !evaluator.outputs.isEmpty else {
        fputs("Error: No outputs defined. Use (out <signal> <channel>) to define outputs.\n", stderr)
        exit(1)
    }

    // Compile
    let options = CompilerOptions(
        outputDir: cli.outputDir,
        name: cli.name,
        sampleRate: cli.sampleRate,
        maxFrames: cli.maxFrames,
        voiceCount: cli.voiceCount,
        toolchainRoot: cli.toolchainRoot,
        skipInlineAudit: cli.skipInlineAudit,
        auditToolPath: cli.auditTool,
        debug: cli.debug
    )

    let compilerResult: CompilerResult
    do {
        compilerResult = try compilePatch(
            graph: graph,
            outputs: evaluator.outputs,
            options: options
        )
    } catch {
        fputs("Compilation error: \(error)\n", stderr)
        exit(1)
    }

    // Generate manifest
    let manifest = generateManifest(
        compilerResult: compilerResult,
        evaluator: evaluator,
        options: options
    )

    // Write manifest to file and print to stdout
    do {
        let jsonString = try writeManifest(manifest, to: cli.outputDir, name: cli.name)
        print(jsonString)
    } catch {
        fputs("Error writing manifest: \(error)\n", stderr)
        exit(1)
    }

    if cli.debug {
        fputs(
            "[debug] Wrote \(cli.outputDir)/\(cli.name)."
                + "\(DGenToolchainPolicy.artifactExtension)\n", stderr)
        fputs("[debug] Wrote \(cli.outputDir)/\(cli.name).json\n", stderr)
        fputs("[debug] Total memory slots: \(compilerResult.compilationResult.totalMemorySlots)\n", stderr)
        fputs("[debug] Kernels: \(compilerResult.compilationResult.kernels.count)\n", stderr)
    }
}

func readStdin() -> String {
    var lines: [String] = []
    while let line = readLine(strippingNewline: false) {
        lines.append(line)
    }
    return lines.joined()
}

// Run
do {
    try main()
} catch {
    fputs("Fatal error: \(error)\n", stderr)
    exit(1)
}

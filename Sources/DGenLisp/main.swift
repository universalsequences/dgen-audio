// DGenLisp - Lisp-to-Dylib Compiler CLI
//
// Usage: dgenlisp compile [<file.lisp>] [options]
//   -o, --output <dir>       Output directory (default: .)
//   --name <name>            Output name (default: patch)
//   --sample-rate <rate>     Sample rate (default: 44100)
//   --max-frames <count>     Max frame count (default: 4096)
//   --voices <count>         Voice count for polyphony (default: 1)
//   --debug                  Debug output
//   -                        Read from stdin (also default if no file given)

import DGenLazy
import Foundation

// MARK: - Argument parsing

struct CLIArgs {
    var command: String = "compile"
    var inputFile: String? = nil
    var outputDir: String = "."
    var name: String = "patch"
    var sampleRate: Float = 44100
    var maxFrames: Int = 4096
    var voiceCount: Int = 1
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
            if i < args.count { cli.maxFrames = Int(args[i]) ?? 4096 }
        case "--voices":
            i += 1
            if i < args.count { cli.voiceCount = Int(args[i]) ?? 1 }
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

        Options:
          -o, --output <dir>       Output directory (default: .)
          --name <name>            Output name (default: patch)
          --sample-rate <rate>     Sample rate (default: 44100)
          --max-frames <count>     Max frame count (default: 4096)
          --voices <count>         Voice count for polyphony (default: 1)
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
    if cli.readStdin {
        source = readStdin()
    } else if let file = cli.inputFile {
        guard FileManager.default.fileExists(atPath: file) else {
            fputs("Error: File not found: \(file)\n", stderr)
            exit(1)
        }
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

    // Reset graph
    LazyGraphContext.reset()
    let graph = LazyGraphContext.current

    // Evaluate lisp source
    let evaluator = LispEvaluator()
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
        fputs("[debug] Wrote \(cli.outputDir)/\(cli.name).dylib\n", stderr)
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

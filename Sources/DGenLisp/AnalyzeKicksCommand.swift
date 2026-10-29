// `dgenlisp analyze-kicks`: thin CLI over DGenAnalysis.KickDataset.
//
//   dgenlisp analyze-kicks --dataset DIR -o analysis.json
//       [--noise-bands N] [--noise-lo HZ] [--noise-hi HZ] [--max-modes K]
//       [--no-second-drop] [--no-mode-drop] [--independent-noise] [--noise-order 2|4]
//       [--check-length N]
//       [--resynth-dir DIR] [--debug-dir DIR]

import DGenAnalysis
import Foundation

enum AnalyzeKicksCommand {
    static func usage() -> String {
        """
        Usage: dgenlisp analyze-kicks --dataset DIR -o analysis.json [options]
          --noise-bands N     noise bands (default 16)
          --noise-lo HZ       lowest band edge (default 40)
          --noise-hi HZ       highest band edge (default 16000)
          --max-modes K       modal slots incl. the fundamental (default 24)
          --no-second-drop    fit a single pitch-drop term
          --no-mode-drop      keep G = 0 for every non-fundamental mode
          --no-stabilize      keep every matrix-pencil mode (no stabilisation check)
          --stability F,D     stabilisation tolerances: relative frequency, relative decay
          --noise-order 2|4   band-pass order of each noise band (default 2, SPEC §3.1)
          --independent-noise one white per noise band (default: one shared white, SPEC §3.1)
          --check-length N    zero-pad length for the check metrics (default 32768)
          --resynth-dir DIR   write <name>_{target,resynth,modal,noise}.wav
          --debug-dir DIR     write per-sample IF tracks / envelopes (JSON)
        """
    }

    static func run(arguments: [String]) -> Never {
        var opts = KickAnalysisOptions()
        var dataset: String? = nil
        var output: String? = nil
        var resynth: String? = nil
        var debugDir: String? = nil
        var i = 0
        func value() -> String {
            i += 1
            guard i < arguments.count else {
                fputs("missing value for \(arguments[i - 1])\n", stderr)
                exit(2)
            }
            return arguments[i]
        }
        func number(_ s: String) -> Double {
            guard let v = Double(s) else {
                fputs("not a number: \(s)\n", stderr)
                exit(2)
            }
            return v
        }
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset": dataset = value()
            case "-o", "--output": output = value()
            case "--noise-bands": opts.noiseBandCount = Int(number(value()))
            case "--noise-lo": opts.noiseLoHz = number(value())
            case "--noise-hi": opts.noiseHiHz = number(value())
            case "--max-modes": opts.maxModes = Int(number(value()))
            case "--no-second-drop": opts.secondDrop = false
            case "--no-mode-drop": opts.modeDropTest = false
            case "--no-stabilize": opts.stabilize = false
            case "--stability": // relF,relD
                let v = value().split(separator: ",").map { number(String($0)) }
                guard v.count == 2 else {
                    fputs("--stability takes relF,relD\n", stderr)
                    exit(2)
                }
                opts.stabilityRelF = v[0]
                opts.stabilityRelD = v[1]
            case "--independent-noise": opts.sharedNoiseWhite = false
            case "--noise-order":
                let v = Int(number(value()))
                guard v == 2 || v == 4 else {
                    fputs("--noise-order must be 2 or 4\n", stderr)
                    exit(2)
                }
                opts.noiseSections = v / 2
            case "--check-length": opts.checkLength = Int(number(value()))
            case "--resynth-dir": resynth = value()
            case "--debug-dir": debugDir = value()
            case "-h", "--help":
                print(usage())
                exit(0)
            default:
                fputs("unknown option: \(arguments[i])\n\(usage())\n", stderr)
                exit(2)
            }
            i += 1
        }
        guard let ds = dataset, let out = output else {
            fputs(usage() + "\n", stderr)
            exit(2)
        }
        do {
            let t0 = Date()
            let result = try KickDataset.run(
                dataset: URL(fileURLWithPath: ds), options: opts,
                resynthDir: resynth.map { URL(fileURLWithPath: $0) },
                debugDir: debugDir.map { URL(fileURLWithPath: $0) }
            ) { fputs($0 + "\n", stderr) }
            try KickDataset.encode(result).write(to: URL(fileURLWithPath: out))
            let s = result.summary
            fputs(String(format: "analysed %d samples in %.1fs: band median %.3f p90 %.3f, mr median %.4f p90 %.4f -> %@\n",
                         Int(s["count"] ?? 0), Date().timeIntervalSince(t0), s["band_median"] ?? .nan,
                         s["band_p90"] ?? .nan, s["mr_median"] ?? .nan, s["mr_p90"] ?? .nan, out), stderr)
            exit(0)
        } catch {
            fputs("analyze-kicks: \(error)\n", stderr)
            exit(1)
        }
    }
}

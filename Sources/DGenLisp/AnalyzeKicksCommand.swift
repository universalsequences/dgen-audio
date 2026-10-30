// `dgenlisp analyze-kicks` / `analyze-drums`: thin CLI over DGenAnalysis.KickDataset.
//
//   dgenlisp analyze-kicks --dataset DIR -o analysis.json
//   dgenlisp analyze-drums --kind snare --dataset DIR -o analysis.json   (KickAnalysisOptions.snare())
//       [--noise-bands N] [--noise-lo HZ] [--noise-hi HZ] [--max-modes K]
//       [--no-second-drop] [--no-mode-drop] [--independent-noise] [--noise-order 2|4]
//       [--check-length N]
//       [--resynth-dir DIR] [--debug-dir DIR]

import DGenAnalysis
import Foundation

enum AnalyzeKicksCommand {
    static func usage() -> String {
        """
        Usage: dgenlisp analyze-kicks|analyze-drums --dataset DIR -o analysis.json [options]
          --kind kick|snare   option preset (default kick); later options override it
          --noise-bands N     noise bands (default 16)
          --noise-lo HZ       lowest band edge (default 40)
          --noise-hi HZ       highest band edge (default 16000)
          --max-modes K       modal slots incl. the fundamental (default 24)
          --no-second-drop    fit a single pitch-drop term
          --no-mode-drop      keep G = 0 for every non-fundamental mode
          --relative-weight F relative-error weighting floor for slot 0 / joint LS (default 0 = plain LS)
          --no-fundamental-tail  no second (slower-decay) mode on the fundamental's law
          --no-stabilize      keep every matrix-pencil mode (no stabilisation check)
          --no-rescue         do not re-admit stabilisation-rejected modes / slot-0 harmonics
          --v3                voice v3 analysis: strike sines, crackle, noise events (implies 4th order)
          --strikes N         strike sines (v3, default 4)
          --max-events N      noise events (v3, default 2)
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
        if let k = arguments.firstIndex(of: "--kind"), k + 1 < arguments.count {
            switch arguments[k + 1] {
            case "kick": break
            case "snare": opts = KickAnalysisOptions.snare()
            default:
                fputs("--kind must be kick or snare\n", stderr)
                exit(2)
            }
        }
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
            case "--kind": _ = value()
            case "--dataset": dataset = value()
            case "-o", "--output": output = value()
            case "--noise-bands": opts.noiseBandCount = Int(number(value()))
            case "--noise-lo": opts.noiseLoHz = number(value())
            case "--noise-hi": opts.noiseHiHz = number(value())
            case "--max-modes": opts.maxModes = Int(number(value()))
            case "--no-second-drop": opts.secondDrop = false
            case "--no-mode-drop": opts.modeDropTest = false
            case "--no-stabilize": opts.stabilize = false
            case "--v3": opts.v3 = true
            case "--no-rescue": opts.rescueModes = false
            case "--strikes": opts.strikeCount = Int(number(value()))
            case "--max-events": opts.maxEvents = Int(number(value()))
            case "--no-fundamental-tail": opts.fundamentalTail = false
            case "--fundamental-tail-gain": opts.fundamentalTailGain = number(value())
            case "--relative-weight": opts.relativeWeightFloor = number(value())
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
        if opts.v3 { opts.noiseSections = 2 }
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

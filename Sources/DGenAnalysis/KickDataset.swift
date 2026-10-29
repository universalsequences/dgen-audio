// Dataset driver: analyse every WAV of a folder and write analysis.json
// (+ optional resynth WAVs and per-sample debug JSON).

import Foundation

public struct DatasetAnalysis: Codable {
    public var version: Int = 1
    public var conventions: [String: String]
    public var options: KickAnalysisOptions
    public var noiseEdges: [Double]
    public var samples: [KickAnalysis]
    public var summary: [String: Double]
}

public enum KickDataset {
    public static let conventions: [String: String] = [
        "time_origin": "t = 0 at sample 0 of the prepared target (2 ms before the -40 dB onset); prep.onsetSeconds is the onset in that frame",
        "prepared_target": "mono = mean of channels; onset = first |x| > peak*10^(-40/20); crop at onset - round(0.002*sr); peak-normalised to 0.9; no padding (checks zero-pad to checkLength)",
        "voice": "SPEC §3.1 v1, see DGenAnalysis/Voice.swift: f_k = F(1+G e^{-t/T}) [slot0 + F G2 e^{-t/T2}], env = A e^{-t/D}(1-e^{-t/R}), y = Σ env sin(phase+phi)",
        "modes": "modes[0] is slot 0 (fundamental, drop law from IF fit). Others from matrix pencil on the tail, G = 0 unless the per-mode drop test adopted the fundamental's (G,T)",
        "noise_bands": "NB log-spaced edges [lo, hi]; band b = RBJ band-pass (constant 0 dB peak), fc = sqrt(lo_b hi_b), Q = fc/(hi_b-lo_b), DF1 zero state, input uniform white [-1,1), normalised to unit expected RMS by 1/(sqrt(1/3) sqrt(sum h^2)); N_b(t) = L[m e^{-t/d1} + (1-m) e^{-t/d2}](1-e^{-t/r_n})",
        "gate": "gate(t) = 1 - smoothstep((t - E + Fd)/Fd)",
        "chain": "analysis assumes a linear chain: level 1, mix 0 (clipper off)",
        "check": "band = kick_metric.band_score, mr = e1 metric.score windows 64..4096; both on zero-padded checkLength signals; noise realised with the fixed seed",
        "units": "seconds, Hz, radians, amplitudes in prepared-target units",
    ]

    public static func listWavs(_ dir: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { ["wav", "wave"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    public static func run(dataset: URL, options: KickAnalysisOptions, resynthDir: URL?, debugDir: URL?,
                           log: (String) -> Void) throws -> DatasetAnalysis
    {
        let files = try listWavs(dataset)
        guard !files.isEmpty else { throw WavError.notWave("no .wav files in \(dataset.path)") }
        var results = [KickAnalysis?](repeating: nil, count: files.count)
        var errors = [String?](repeating: nil, count: files.count)
        let lock = NSLock()
        for d in [resynthDir, debugDir].compactMap({ $0 }) {
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        }
        DispatchQueue.concurrentPerform(iterations: files.count) { i in
            let url = files[i]
            let name = url.deletingPathExtension().lastPathComponent
            do {
                let audio = try WAV.read(url)
                var dbg = KickDebug()
                let (a, prep) = KickAnalyzer.analyze(name: name, audio: audio, options: options, debug: &dbg)
                if let rd = resynthDir { try writeResynth(a, prep: prep, options: options, dir: rd) }
                if let dd = debugDir {
                    let enc = JSONEncoder()
                    enc.nonConformingFloatEncodingStrategy = .convertToString(
                        positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
                    try enc.encode(dbg).write(to: dd.appendingPathComponent("\(name).json"))
                }
                lock.lock()
                results[i] = a
                lock.unlock()
                if let c = a.check {
                    log(String(format: "%@  F0 %.1f→%.1f Hz  modes %d  band %.2f  mr %.3f", name,
                               a.drop.fStartHz, a.drop.F, a.modes.count, c.band, c.mr))
                }
            } catch {
                lock.lock()
                errors[i] = "\(name): \(error)"
                lock.unlock()
            }
        }
        for e in errors.compactMap({ $0 }) { log("error: \(e)") }
        let samples = results.compactMap { $0 }
        func stat(_ v: [Double], _ q: Double) -> Double {
            let s = v.filter { $0.isFinite }.sorted()
            guard !s.isEmpty else { return .nan }
            let pos = q * Double(s.count - 1)
            let lo = Int(pos.rounded(.down)), hi = min(s.count - 1, lo + 1)
            return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
        }
        let band = samples.compactMap { $0.check?.band }, mr = samples.compactMap { $0.check?.mr }
        let summary: [String: Double] = [
            "count": Double(samples.count),
            "band_median": stat(band, 0.5), "band_p90": stat(band, 0.9),
            "mr_median": stat(mr, 0.5), "mr_p90": stat(mr, 0.9),
            "band_modal_median": stat(samples.compactMap { $0.check?.bandModal }, 0.5),
            "mr_modal_median": stat(samples.compactMap { $0.check?.mrModal }, 0.5),
            "modes_median": stat(samples.map { Double($0.modes.count) }, 0.5),
        ]
        let edges = NoiseBands.edges(count: options.noiseBandCount, lo: options.noiseLoHz,
                                     hi: min(options.noiseHiHz, 0.45 * (samples.first?.sampleRate ?? 44100)))
        return DatasetAnalysis(conventions: conventions, options: options, noiseEdges: edges, samples: samples,
                               summary: summary)
    }

    public static func writeResynth(_ a: KickAnalysis, prep: PreparedTarget, options: KickAnalysisOptions,
                                    dir: URL) throws
    {
        let sr = a.sampleRate
        let n = prep.signal.count
        let p = a.voiceParams
        let full = KickVoice.render(p, count: n, sampleRate: sr, seed: options.noiseSeed)
        var modalOnly = p
        modalOnly.noiseBands = []
        let modal = KickVoice.render(modalOnly, count: n, sampleRate: sr)
        var noiseOnly = p
        noiseOnly.modes = []
        let noise = KickVoice.render(noiseOnly, count: n, sampleRate: sr, seed: options.noiseSeed)
        try WAV.writeFloat32(prep.signal, sampleRate: sr, to: dir.appendingPathComponent("\(a.name)_target.wav"))
        try WAV.writeFloat32(full, sampleRate: sr, to: dir.appendingPathComponent("\(a.name)_resynth.wav"))
        try WAV.writeFloat32(modal, sampleRate: sr, to: dir.appendingPathComponent("\(a.name)_modal.wav"))
        try WAV.writeFloat32(noise, sampleRate: sr, to: dir.appendingPathComponent("\(a.name)_noise.wav"))
    }

    public static func encode(_ d: DatasetAnalysis) throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf",
                                                                  nan: "nan")
        return try enc.encode(d)
    }
}

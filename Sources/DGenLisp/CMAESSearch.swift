import DGenTrainProtocol
import Foundation

/// Product-facing CMA basin search using the generic tensor-lane Lisp renderer.
enum CMAESSearch {
    struct Result {
        var bestZ: [Float]
        var bestScore: Float
        var elites: [[Float]]
        var eliteScores: [Float]
        var seedScore: Float
        var report: [String: Any]
    }

    static func search(
        options: TrainOptions, transforms: [DirectionTrainer.TransformedParam],
        seedZ: [Float], parameterValues: [String: Float], nodes: [ASTNode],
        target: [Float], sampleRate: Float,
        crop: Int, sink: TrainEventSink, jobDir: JobDir
    ) throws -> Result {
        let dimension = transforms.count
        let defaultPopulation = max(32, 4 + Int(floor(3 * log(Double(max(2, dimension))))))
        let population = options.cmaPopulation > 0 ? options.cmaPopulation : defaultPopulation
        let batchSize = min(population, options.cmaForwardBatch > 0 ? options.cmaForwardBatch : population)
        var optimizer = CMAES(
            mean: seedZ.map(Double.init), sigma: options.cmaSigma,
            population: population, seed: UInt64(bitPattern: Int64(options.cmaSeed)))
        let scorer = try TrainSpectralScorer(
            target: target, windows: DirectionTrainer.spectralWindows,
            epsilon: DirectionTrainer.logEpsilon)
        try sink.emit(.stage(StageEvent(name: "cma-es", total: options.cmaGenerations)))

        // The seed is an external baseline, not a member of the adaptive CMA
        // population. Score and archive it independently so every candidate
        // passed to `tell` remains a genuine draw from the optimizer.
        let seedScore = try BatchMultistart.score(
            candidates: [seedZ], transforms: transforms,
            parameterValues: parameterValues, nodes: nodes,
            scorer: scorer, sampleRate: sampleRate, crop: crop,
            batchSize: 1, options: options)[0]
        var bestScore = seedScore
        var bestZ = seedZ
        var trace = [[String: Any]]()
        var archive: [(z: [Float], score: Float)] = [(seedZ, seedScore)]
        var evaluations = 1
        var stopReason = "generation_limit"

        for generation in 0..<options.cmaGenerations {
            let candidates = optimizer.ask()
            if candidates.isEmpty { stopReason = optimizer.stopReason?.rawValue ?? "numerical_failure"; break }
            let floatCandidates = candidates.map { $0.values.map(Float.init) }

            let evaluationStart = Date()
            var renderSeconds = 0.0
            var deinterleaveSeconds = 0.0
            var scoringSeconds = 0.0
            let scores = try BatchMultistart.score(
                candidates: floatCandidates, transforms: transforms,
                parameterValues: parameterValues, nodes: nodes,
                scorer: scorer, sampleRate: sampleRate, crop: crop,
                batchSize: batchSize, options: options
            ) { render, deinterleave, scoring in
                renderSeconds = render
                deinterleaveSeconds = deinterleave
                scoringSeconds = scoring
            }
            let evaluationSeconds = Date().timeIntervalSince(evaluationStart)
            evaluations += scores.count

            let rankedIndices = candidates.indices.sorted {
                let lhs = scores[$0].isFinite ? scores[$0] : .infinity
                let rhs = scores[$1].isFinite ? scores[$1] : .infinity
                return lhs == rhs ? candidates[$0].index < candidates[$1].index : lhs < rhs
            }
            let generationBest = scores[rankedIndices[0]]
            if generationBest < bestScore {
                bestScore = generationBest
                bestZ = floatCandidates[rankedIndices[0]]
            }
            for index in rankedIndices.prefix(min(population / 2, 16)) where scores[index].isFinite {
                archive.append((floatCandidates[index], scores[index]))
            }

            let updateStart = Date()
            optimizer.tell(ranked: rankedIndices.map { candidates[$0] })
            let updateMS = Date().timeIntervalSince(updateStart) * 1000
            let finiteScores = scores.filter(\.isFinite).sorted()
            let reflected = candidates.reduce(0) { $0 + $1.reflectedCoordinates }
            trace.append([
                "generation": generation,
                "best": jsonNumber(generationBest),
                "median": jsonNumber(finiteScores.isEmpty ? .infinity : finiteScores[finiteScores.count / 2]),
                "mean": jsonNumber(finiteScores.isEmpty ? .infinity : finiteScores.reduce(0, +) / Float(finiteScores.count)),
                "sigma": jsonNumber(optimizer.sigma),
                "condition_number": jsonNumber(optimizer.conditionNumber),
                "reflected_fraction": Double(reflected) / Double(population * dimension),
                // `forward_seconds` is retained for report-schema v1 and is
                // explicitly the whole population evaluation, not GPU-only.
                "forward_seconds": evaluationSeconds,
                "render_seconds": renderSeconds,
                "readback_deinterleave_seconds": deinterleaveSeconds,
                "scoring_seconds": scoringSeconds,
                "cma_update_ms": updateMS,
            ])
            try writeCandidate(
                bestZ, score: bestScore,
                to: jobDir.file(String(format: "cma_best_generation_%04d.json", generation)))
            try JSONEncoder.sorted.encode(optimizer).write(
                to: jobDir.file("cma_es_state.json"), options: .atomic)
            FileHandle.standardError.write(Data(String(
                format: "[train] cma generation=%d best=%.6g all_best=%.6g sigma=%.4g reflected=%.3f forward=%.3fs update=%.3fms\n",
                generation, generationBest, bestScore, optimizer.sigma,
                Double(reflected) / Double(population * dimension), evaluationSeconds, updateMS).utf8))
            try sink.emit(.optimizationProgress(OptimizationProgressEvent(
                current: generation + 1, total: options.cmaGenerations,
                losses: rankedIndices.prefix(5).compactMap {
                    scores[$0].isFinite ? Double(scores[$0]) : nil
                })))

            if optimizer.stopReason != nil {
                stopReason = optimizer.stopReason!.rawValue
                break
            }
        }

        // No explicit append of (bestZ, bestScore): the generation winner is
        // always inside the top min(population/2, 16) archived above (population
        // >= 4 is validated at parse time), and if no generation ever improved
        // then bestZ == seedZ == archive[0].z. Appending it again would put an
        // exact duplicate at indices 0/1 of the sorted archive.
        archive.sort { $0.score == $1.score ? lexicographic($0.z, $1.z) : $0.score < $1.score }
        let candidateVectors = archive.map(\.z)
        let archiveScores = archive.map(\.score)
        let eliteCount = min(max(options.cmaContinue, 1), candidateVectors.count)
        let eliteIndices = BatchMultistart.diverseSelection(
            candidates: candidateVectors, scores: archiveScores, count: eliteCount,
            mandatory: [0])
        let elites = eliteIndices.map { candidateVectors[$0] }
        let eliteScores = eliteIndices.map { archiveScores[$0] }
        let bestParams = DirectionTrainer.naturalValues(z: bestZ, transforms: transforms)
            .mapValues(Double.init)
        let report: [String: Any] = [
            "algorithm": "cma-es", "version": 1, "dimension": dimension,
            "population": population, "generations_completed": trace.count,
            "evaluations": evaluations, "seed": options.cmaSeed,
            "initial_sigma": options.cmaSigma, "stop_reason": stopReason,
            "generation_trace": trace, "seed_score": jsonNumber(seedScore),
            "cma_best_score": jsonNumber(bestScore), "best_z": bestZ.map(Double.init),
            "best_params": bestParams, "continued_candidates": [],
            "local_seed_outcome": [:] as [String: Any],
            "global_outcome": ["pre_refine_score": jsonNumber(bestScore)],
        ]
        try writeReport(report, jobDir: jobDir)
        return Result(
            bestZ: bestZ, bestScore: bestScore, elites: elites,
            eliteScores: eliteScores, seedScore: seedScore, report: report)
    }

    static func writeReport(_ report: [String: Any], jobDir: JobDir) throws {
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: jobDir.file("cma_es_report.json"), options: .atomic)
    }

    private static func writeCandidate(_ z: [Float], score: Float, to url: URL) throws {
        let object: [String: Any] = ["score": jsonNumber(score), "z": z.map(Double.init)]
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
            .write(to: url, options: .atomic)
    }

    private static func jsonNumber(_ value: Float) -> Any {
        value.isFinite ? Double(value) : NSNull()
    }

    /// JSONSerialization aborts the process (rather than throwing) on
    /// non-finite doubles, so raw optimizer state must be sanitized too.
    private static func jsonNumber(_ value: Double) -> Any {
        value.isFinite ? value : NSNull()
    }

    private static func lexicographic(_ a: [Float], _ b: [Float]) -> Bool {
        for (x, y) in zip(a, b) where x != y { return x < y }
        return false
    }
}

private extension JSONEncoder {
    static var sorted: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

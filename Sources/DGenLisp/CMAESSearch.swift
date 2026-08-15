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

        var bestScore = Float.infinity
        var bestZ = seedZ
        var seedScore = Float.infinity
        var trace = [[String: Any]]()
        var archive: [(z: [Float], score: Float)] = []
        var evaluations = 0
        var staleGenerations = 0
        var stopReason = "generation_limit"

        for generation in 0..<options.cmaGenerations {
            var anchors = [[Double]]()
            if generation == 0, population >= 8 {
                // Deterministic diagonal strata are immigrants, not RNG draws,
                // so they do not perturb resume sampling.
                let immigrants = min(4, population - 4)
                for i in 0..<immigrants {
                    anchors.append((0..<dimension).map { d in
                        (Double((i + d) % immigrants) + 0.5) / Double(immigrants)
                    })
                }
            }
            if generation == 0 {
                anchors.append(seedZ.map(Double.init))
                anchors.append([Double](repeating: 0.5, count: dimension))
            }
            anchors.append(bestZ.map(Double.init))
            anchors.append(optimizer.mean.map(CMAES.reflect))
            let candidates = optimizer.ask(anchors: anchors)
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
            if generation == 0 {
                // Seed is always the fourth-from-last anchor in generation 0.
                if let index = candidates.indices.first(where: { candidates[$0].values == seedZ.map(Double.init) }) {
                    seedScore = scores[index]
                }
            }

            let rankedIndices = candidates.indices.sorted {
                let lhs = scores[$0].isFinite ? scores[$0] : .infinity
                let rhs = scores[$1].isFinite ? scores[$1] : .infinity
                return lhs == rhs ? candidates[$0].index < candidates[$1].index : lhs < rhs
            }
            let generationBest = scores[rankedIndices[0]]
            let previousBest = bestScore
            let tolerance = previousBest.isFinite
                ? max(1e-7, abs(previousBest) * 1e-7) : 0
            let meaningfulImprovement = generationBest < previousBest - tolerance
            if generationBest < bestScore {
                bestScore = generationBest
                bestZ = floatCandidates[rankedIndices[0]]
            }
            staleGenerations = meaningfulImprovement ? 0 : staleGenerations + 1
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
                "sigma": optimizer.sigma,
                "condition_number": optimizer.conditionNumber,
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

            if optimizer.stopReason != nil {
                stopReason = optimizer.stopReason!.rawValue
                break
            }
            if staleGenerations >= 5 {
                stopReason = "no_improvement"
                break
            }
        }

        archive.append((bestZ, bestScore))
        archive.sort { $0.score == $1.score ? lexicographic($0.z, $1.z) : $0.score < $1.score }
        let candidateVectors = archive.map(\.z)
        let archiveScores = archive.map(\.score)
        let eliteCount = min(max(options.cmaContinue, 1), candidateVectors.count)
        let eliteIndices = BatchMultistart.diverseSelection(
            candidates: candidateVectors, scores: archiveScores, count: eliteCount)
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

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

/// Generic tensor-lane population search for an already-lowered dgenlisp
/// patch. Patch authors keep writing scalar Lisp; LispEvaluator lifts params,
/// phasors, history cells, and output into independent [B] lanes.
enum BatchMultistart {
    struct Result {
        let bestZ: [Float]
        let bestScore: Float
        let seedPostScore: Float
        let report: [String: Any]
    }

    private struct RNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func uniform() -> Float {
            Float(Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0))
        }
        mutating func gaussian() -> Float {
            let u1 = max(uniform(), Float.ulpOfOne)
            let u2 = uniform()
            return Foundation.sqrt(-2 * Foundation.log(u1))
                * Foundation.cos(2 * Float.pi * u2)
        }
    }

    static func search(
        options: TrainOptions, transforms: [DirectionTrainer.TransformedParam],
        seedZ: [Float], nodes: [ASTNode], target: [Float], sampleRate: Float,
        crop: Int, sink: TrainEventSink, jobDir: JobDir
    ) throws -> Result {
        let count = options.multistartCandidates
        let laneCount = min(options.multistartLanes, count)
        let forwardBatch = min(options.multistartBatch, count)
        var rng = RNG(state: UInt64(bitPattern: Int64(options.multistartSeed)))
        var candidates = makeCandidates(
            count: count, dimensions: transforms.count, seed: seedZ, rng: &rng)
        let scorer = try TrainSpectralScorer(
            target: target, windows: DirectionTrainer.spectralWindows,
            epsilon: DirectionTrainer.logEpsilon)

        let forwardStart = Date()
        let initialScores = try score(
            candidates: candidates, transforms: transforms, nodes: nodes,
            scorer: scorer, sampleRate: sampleRate, crop: crop,
            batchSize: forwardBatch, options: options)
        let forwardSeconds = Date().timeIntervalSince(forwardStart)
        let retainedIndices = diverseSelection(
            candidates: candidates, scores: initialScores, count: laneCount)
        candidates = retainedIndices.map { candidates[$0] }
        let retainedInitial = retainedIndices.map { initialScores[$0] }

        try sink.emit(.stage(StageEvent(name: "multistart", total: options.multistartSteps)))
        let refineStart = Date()
        let refined = try refine(
            candidates: candidates, transforms: transforms, nodes: nodes,
            target: target, crop: crop, sampleRate: sampleRate,
            steps: options.multistartSteps, options: options)
        let refineSeconds = Date().timeIntervalSince(refineStart)
        let postStart = Date()
        let postScores = try score(
            candidates: refined, transforms: transforms, nodes: nodes,
            scorer: scorer, sampleRate: sampleRate, crop: crop,
            batchSize: laneCount, options: options)
        let postScoreSeconds = Date().timeIntervalSince(postStart)
        let best = postScores.indices.min { postScores[$0] < postScores[$1] } ?? 0
        let correlation = pearson(retainedInitial, postScores)

        let report: [String: Any] = [
            "candidate_count": count,
            "forward_batch": forwardBatch,
            "refine_lanes": laneCount,
            "refine_steps": options.multistartSteps,
            "forward_seconds": forwardSeconds,
            "refine_seconds": refineSeconds,
            "post_score_seconds": postScoreSeconds,
            "initial_best_score": initialScores.min() ?? Float.infinity,
            "post_best_score": postScores[best],
            "seed_initial_score": initialScores[0],
            "seed_post_score": postScores[0],
            "initial_post_pearson": correlation,
            "retained_source_indices": retainedIndices,
            "retained_initial_scores": retainedInitial,
            "retained_post_scores": postScores,
            "best_lane": best,
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: jobDir.file("multistart_report.json"), options: .atomic)
        FileHandle.standardError.write(Data(String(
            format: "[train] multistart candidates=%d lanes=%d steps=%d initial_best=%.6g post_best=%.6g seed_post=%.6g corr=%.3f forward=%.2fs refine=%.2fs\n",
            count, laneCount, options.multistartSteps, initialScores.min() ?? .infinity,
            postScores[best], postScores[0], correlation, forwardSeconds, refineSeconds).utf8))
        return Result(
            bestZ: refined[best], bestScore: postScores[best],
            seedPostScore: postScores[0], report: report)
    }

    private static func makeCandidates(
        count: Int, dimensions: Int, seed: [Float], rng: inout RNG
    ) -> [[Float]] {
        var result = [[Float]](repeating: [Float](repeating: 0.5, count: dimensions), count: count)
        result[0] = seed
        if count > 1 { result[1] = [Float](repeating: 0.5, count: dimensions) }
        let jitterCount = max(0, (count - 2) / 2)
        if jitterCount > 0 {
            for i in 0..<jitterCount {
                let sigma: Float = i < jitterCount / 2 ? 0.03 : 0.12
                for d in 0..<dimensions {
                    result[i + 2][d] = min(1, max(0, seed[d] + sigma * rng.gaussian()))
                }
            }
        }
        let start = 2 + jitterCount
        let stratifiedCount = count - start
        if stratifiedCount > 0 {
            for d in 0..<dimensions {
                var strata = Array(0..<stratifiedCount)
                for i in stride(from: strata.count - 1, through: 1, by: -1) {
                    let j = Int(rng.next() % UInt64(i + 1))
                    strata.swapAt(i, j)
                }
                for i in 0..<stratifiedCount {
                    result[start + i][d] = (Float(strata[i]) + rng.uniform()) / Float(stratifiedCount)
                }
            }
        }
        return result
    }

    private static func naturalTensors(
        z: [Tensor], transforms: [DirectionTrainer.TransformedParam]
    ) -> [String: EvalResult] {
        var values: [String: EvalResult] = [:]
        let one = Signal.constant(1)
        for (i, transform) in transforms.enumerated() {
            let frameValue = z[i] * one
            if transform.useLog {
                values[transform.name] = .signalTensor(DGenLazy.exp(
                    frameValue * transform.span + Foundation.log(transform.min)))
            } else {
                values[transform.name] = .signalTensor(
                    frameValue * transform.span + transform.min)
            }
        }
        return values
    }

    private static func evaluate(
        z: [Tensor], transforms: [DirectionTrainer.TransformedParam], nodes: [ASTNode]
    ) throws -> SignalTensor {
        let evaluator = LispEvaluator(
            batchLaneCount: z[0].shape[0],
            batchParameterValues: naturalTensors(z: z, transforms: transforms))
        do { try evaluator.evaluate(nodes: nodes) }
        catch let error as LispError {
            throw TrainProtocolError("batched patch evaluation failed: \(error.message)")
        }
        guard let output =
            (evaluator.tensorOutputs.first { $0.channel == 0 } ?? evaluator.tensorOutputs.first)?.signal
        else { throw TrainProtocolError("batched patch produced no tensor output") }
        return output
    }

    private static func score(
        candidates: [[Float]], transforms: [DirectionTrainer.TransformedParam],
        nodes: [ASTNode], scorer: TrainSpectralScorer, sampleRate: Float, crop: Int,
        batchSize: Int, options: TrainOptions
    ) throws -> [Float] {
        var scores = [Float](repeating: .infinity, count: candidates.count)
        var start = 0
        while start < candidates.count {
            let actualCount = min(batchSize, candidates.count - start)
            var batch = Array(candidates[start..<(start + actualCount)])
            while batch.count < batchSize { batch.append(batch.last!) }
            DirectionTrainer.configureRuntime(options: options, sampleRate: sampleRate, crop: crop)
            LazyGraphContext.reset()
            let z = transforms.indices.map { d in
                Tensor(batch.map { $0[d] }, requiresGrad: true)
            }
            let output = try evaluate(z: z, transforms: transforms, nodes: nodes)
            let flat = try output.realize(frames: crop)
            for lane in 0..<actualCount {
                var audio = [Float](repeating: 0, count: crop)
                for frame in 0..<crop { audio[frame] = flat[frame * batchSize + lane] }
                scores[start + lane] = scorer.score(audio)
            }
            start += actualCount
        }
        return scores
    }

    private static func refine(
        candidates: [[Float]], transforms: [DirectionTrainer.TransformedParam],
        nodes: [ASTNode], target: [Float], crop: Int, sampleRate: Float,
        steps: Int, options: TrainOptions
    ) throws -> [[Float]] {
        let lanes = candidates.count
        DirectionTrainer.configureRuntime(options: options, sampleRate: sampleRate, crop: crop)
        LazyGraphContext.reset()
        let z = transforms.indices.map { d in Tensor(candidates.map { $0[d] }, requiresGrad: true) }
        var m = [[Float]](repeating: [Float](repeating: 0, count: lanes), count: z.count)
        var v = m
        let targetTensor = Tensor(target)
        let ones = Tensor([Float](repeating: 1, count: lanes))
        for step in 1...steps {
            let output = try evaluate(z: z, transforms: transforms, nodes: nodes)
            let targetSignal = ones * targetTensor.toSignal(maxFrames: crop)
            let loss = DirectionTrainer.multiResolutionSpectralLoss(
                synth: output, target: targetSignal, frames: crop)
                * Signal.constant(Float(lanes))
            _ = try loss.backward(frames: crop)
            let progress = Float(step - 1) / Float(max(steps - 1, 1))
            let lr = DirectionTrainer.transformedLR
                * (0.05 + 0.95 * 0.5 * (1 + Foundation.cos(Float.pi * progress)))
            for p in z.indices {
                guard let gradients = z[p].grad?.getData(), let values = z[p].getData() else { continue }
                var updated = values
                for lane in 0..<lanes {
                    let g = min(DirectionTrainer.gradClip, max(-DirectionTrainer.gradClip, gradients[lane]))
                    m[p][lane] = 0.9 * m[p][lane] + 0.1 * g
                    v[p][lane] = 0.999 * v[p][lane] + 0.001 * g * g
                    let mh = m[p][lane] / (1 - Foundation.pow(0.9, Float(step)))
                    let vh = v[p][lane] / (1 - Foundation.pow(0.999, Float(step)))
                    updated[lane] = min(1, max(0, values[lane] - lr * mh / (sqrt(vh) + 1e-8)))
                }
                z[p].updateDataLazily(updated)
                z[p].grad = nil
            }
        }
        let data = z.map { $0.getData() ?? [Float](repeating: 0.5, count: lanes) }
        return (0..<lanes).map { lane in data.map { $0[lane] } }
    }

    private static func diverseSelection(
        candidates: [[Float]], scores: [Float], count: Int
    ) -> [Int] {
        var selected = Array(0..<min(2, count))
        let ranked = scores.indices.sorted { scores[$0] < scores[$1] }
        for index in ranked where selected.count < count && !selected.contains(index) {
            let distance = selected.map { other -> Float in
                zip(candidates[index], candidates[other]).reduce(Float(0)) { sum, pair in
                    let delta = pair.0 - pair.1
                    return sum + delta * delta
                }.squareRoot()
            }.min() ?? 1
            if distance >= 0.05 { selected.append(index) }
        }
        for index in ranked where selected.count < count && !selected.contains(index) {
            selected.append(index)
        }
        return selected
    }

    private static func pearson(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, a.count > 1 else { return 0 }
        let am = a.reduce(0, +) / Float(a.count)
        let bm = b.reduce(0, +) / Float(b.count)
        var numerator: Float = 0
        var ad: Float = 0
        var bd: Float = 0
        for i in a.indices {
            numerator += (a[i] - am) * (b[i] - bm)
            ad += (a[i] - am) * (a[i] - am)
            bd += (b[i] - bm) * (b[i] - bm)
        }
        return numerator / max(1e-12, sqrt(ad * bd))
    }
}

private extension DirectionTrainer {
    static func multiResolutionSpectralLoss(
        synth: SignalTensor, target: SignalTensor, frames: Int
    ) -> Signal {
        var total = Signal.constant(0)
        for window in spectralWindows where window <= frames {
            let hop = max(1, window / 4)
            total = total + spectralLossFFT(
                synth, target, windowSize: window, useHannWindow: true,
                useLogMagnitude: true, lossMode: .l1, hop: hop, normalize: true)
            total = total + spectralLossFFT(
                synth, target, windowSize: window, useHannWindow: true,
                useLogMagnitude: false, lossMode: .l1, hop: hop, normalize: true)
                * linearMagnitudeWeight
        }
        return total
    }
}

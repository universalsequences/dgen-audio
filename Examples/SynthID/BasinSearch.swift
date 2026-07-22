import Accelerate
import DGen
import DGenLazy
import Foundation

// MARK: - CPU MR-STFT scorer (Swift port of scripts/compare.py mrstft)
//
// Ranks batched candidates with the SAME objective family the corrected
// rung-3 protocol already uses for restart selection: the independent CPU
// metric from scripts/compare.py. Windows 256/512/1024/2048, hop = w/4,
// symmetric Hann (np.hanning), coherent-gain magnitude scale sum(w)/2,
// log(mag + 1e-3), L1 mean over bins, mean over frames, sum over windows.
// Verified against compare.py to 7 significant digits on seed-6 audio.
final class CPUSpectralScorer {
  struct Plan {
    let size: Int
    let hop: Int
    let hann: [Float]
    let scale: Float
    let setup: vDSP_DFT_Setup
    let starts: [Int]
    let bins: Int
    var targetLog: [Float]  // starts.count * bins, frame-major
  }

  private(set) var plans: [Plan] = []
  let epsilon: Float

  init(target: [Float], windows: [Int] = [256, 512, 1024, 2048], epsilon: Float = 1e-3) throws {
    self.epsilon = epsilon
    for size in windows {
      guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(size), .FORWARD) else {
        throw SynthIDError.message("vDSP DFT setup failed for window \(size)")
      }
      // np.hanning: symmetric Hann, denominator (size - 1).
      var hann = [Float](repeating: 0, count: size)
      for i in 0..<size {
        hann[i] = 0.5 - 0.5 * Foundation.cos(2.0 * Float.pi * Float(i) / Float(size - 1))
      }
      let scale = max(hann.reduce(0, +) / 2.0, 1e-12)
      let hop = max(1, size / 4)
      var starts: [Int] = []
      var s = 0
      while s + size <= target.count {
        starts.append(s)
        s += hop
      }
      var plan = Plan(
        size: size, hop: hop, hann: hann, scale: scale, setup: setup,
        starts: starts, bins: size / 2 + 1, targetLog: [])
      plan.targetLog = logSpectra(target, plan: plan)
      plans.append(plan)
    }
  }

  deinit {
    for plan in plans {
      vDSP_DFT_DestroySetup(plan.setup)
    }
  }

  /// Frame-major log magnitudes for every analysis frame of `signal`.
  func logSpectra(_ signal: [Float], plan: Plan) -> [Float] {
    let size = plan.size
    let bins = plan.bins
    var inReal = [Float](repeating: 0, count: size)
    let inImag = [Float](repeating: 0, count: size)
    var outReal = [Float](repeating: 0, count: size)
    var outImag = [Float](repeating: 0, count: size)
    var out = [Float](repeating: 0, count: plan.starts.count * bins)
    for (frame, start) in plan.starts.enumerated() {
      signal.withUnsafeBufferPointer { sig in
        plan.hann.withUnsafeBufferPointer { win in
          inReal.withUnsafeMutableBufferPointer { dst in
            vDSP_vmul(sig.baseAddress! + start, 1, win.baseAddress!, 1,
                      dst.baseAddress!, 1, vDSP_Length(size))
          }
        }
      }
      vDSP_DFT_Execute(plan.setup, inReal, inImag, &outReal, &outImag)
      let base = frame * bins
      for b in 0..<bins {
        let mag = Foundation.sqrt(outReal[b] * outReal[b] + outImag[b] * outImag[b]) / plan.scale
        out[base + b] = Foundation.log(mag + epsilon)
      }
    }
    return out
  }

  /// compare.py `mrstft(audio, target)` up to float accumulation order.
  func score(_ audio: [Float]) -> Float {
    var total = 0.0
    for plan in plans {
      let logA = logSpectra(audio, plan: plan)
      let bins = plan.bins
      var frameSum = 0.0
      for frame in 0..<plan.starts.count {
        var binSum = 0.0
        let base = frame * bins
        for b in 0..<bins {
          binSum += Double(abs(logA[base + b] - plan.targetLog[base + b]))
        }
        frameSum += binSum / Double(bins)
      }
      total += frameSum / Double(max(1, plan.starts.count))
    }
    return Float(total)
  }
}

// MARK: - Batched stratified basin search (policy v2)
//
// Policy v2 changes from the first (failed) wide scan:
//  1. Candidates are sampled UNIFORMLY IN TRANSFORMED COORDINATES over the
//     full declared bounds — the same coordinate system the E1 generative
//     prior (PatchValues.sample) draws from. The v1 raw-unit prior made
//     low-fAmt truths ~0.2% tail events.
//  2. Iterated stratified resampling: after the uniform round, each stratum's
//     best candidates seed Gaussian resampling rounds with shrinking sigma —
//     derivative-free local search across the coupled shape/filter ridge
//     that Adam cannot cross (see E1_POLICY_AUDIT_FINDING.md).
enum BasinSearch {

  static let searchParamNames = [
    "shape", "pw", "fBase", "fAmt", "fDecay", "res",
    "attackTime", "decayTime", "sustain", "releaseTime", "drive", "outGain",
  ]

  struct StratumResult: Codable {
    var stratum: String
    var score: Float
    var fBase: Float
    var shape: Float
    var pw: Float
    var res: Float
    var fAmt: Float
    var fDecay: Float
  }

  struct Scored {
    var transformed: [Float]
    var candidate: SubtractiveCandidate
    var score: Float
  }

  /// fBase octave bands matching the declared deployment range.
  static let fBaseBandEdges: [Float] = [60, 120, 240, 480, 960, 2000.01]

  static func stratumKey(fBase: Float, shape: Float) -> Int? {
    guard let band = (0..<(fBaseBandEdges.count - 1)).first(where: {
      fBase >= fBaseBandEdges[$0] && fBase < fBaseBandEdges[$0 + 1]
    }) else { return nil }
    return band * 2 + (shape >= 0.5 ? 1 : 0)
  }

  static func stratumLabel(_ key: Int) -> String {
    let band = key / 2
    let lo = Int(fBaseBandEdges[band])
    let hi = Int(fBaseBandEdges[band + 1])
    let half = key % 2 == 0 ? "shape<0.5" : "shape>=0.5"
    return "fBase[\(lo),\(hi))/\(half)"
  }

  static func gaussianPair(_ rng: inout SplitMix64) -> (Float, Float) {
    let u1 = max(rng.uniform(0.0, 1.0), 1e-12)
    let u2 = rng.uniform(0.0, 1.0)
    let r = Foundation.sqrt(-2.0 * Foundation.log(u1))
    return (r * Foundation.cos(2.0 * Float.pi * u2), r * Foundation.sin(2.0 * Float.pi * u2))
  }

  static func run(options: [String: String]) throws {
    var config = try loadConfig(url: options["config"].map { URL(fileURLWithPath: $0) })
    config.profile = "subtractive-bass"
    try config.applyCLI(options)
    config.profile = "subtractive-bass"
    config.enableNoiseFilter = true
    config.applyRuntime()

    guard
      let targetPath = options["target"],
      let outPath = options["out"]
    else {
      throw SynthIDError.message("basin-search requires --target <wav> and --out <dir>")
    }
    let count = try options["count"].map { try parseInt($0, "--count") } ?? 8192
    let batchSize = try options["batch"].map { try parseInt($0, "--batch") } ?? 256
    let seed = try options["seed"].map { try parseInt($0, "--seed") } ?? 6
    let resampleRounds = try options["rounds"].map { try parseInt($0, "--rounds") } ?? 2
    let parentsPerStratum = try options["parents"].map { try parseInt($0, "--parents") } ?? 8
    let childrenPerParent = try options["children"].map { try parseInt($0, "--children") } ?? 96
    // Fixed SplitMix64 stream derived from the self-inversion seed, matching
    // the audit's population-search convention. Overridable for replication.
    let searchSeed = try options["search-seed"].map { UInt64(try parseInt($0, "--search-seed")) }
      ?? (UInt64(seed) &* 0x9E37_79B9_7F4A_7C15 &+ 0xE1BA_51)

    let outDir = URL(fileURLWithPath: outPath)
    try ensureDirectory(outDir)
    let elitesDir = outDir.appendingPathComponent("elites")
    try ensureDirectory(elitesDir)

    let (rawTarget, targetRate) = try AudioFile.load(url: URL(fileURLWithPath: targetPath))
    if abs(targetRate - config.sampleRate) > 0.5 {
      print("warning: target sampleRate=\(targetRate) config sampleRate=\(config.sampleRate)")
    }
    let targetSamples = fitOrPad(
      peakNormalized(rawTarget, peak: config.peakNormalizeTo), frames: config.frames)
    let scorer = try CPUSpectralScorer(target: targetSamples)

    if let scoreWavPath = options["score-wav"] {
      // Debug: score an existing WAV (fit/padded, not renormalized) against
      // the normalized target, for cross-checking with compare.py mrstft.
      let (wav, _) = try AudioFile.load(url: URL(fileURLWithPath: scoreWavPath))
      print("score-wav \(scoreWavPath): \(scorer.score(fitOrPad(wav, frames: config.frames)))")
      return
    }

    // Base values fill the non-searched PatchValues fields; every searched
    // field is overwritten, so any deterministic subtractive params work.
    let basePath = options["base-params"]
      ?? "output/e1_subtractive_fresh_seeds_6_7/seed-6/initial_params.json"
    let baseValues = try loadPatchValues(from: URL(fileURLWithPath: basePath))

    let specs = try searchParamNames.map { name -> ParameterSpec in
      guard let spec = KickParamSpecs.byName[name] else {
        throw SynthIDError.message("missing ParameterSpec for \(name)")
      }
      return spec
    }
    // Full declared transformed bounds with the coordinate-search inset.
    let lows = specs.map { $0.transformedBounds.min + ($0.transformedBounds.max - $0.transformedBounds.min) * 0.001 }
    let highs = specs.map { $0.transformedBounds.max - ($0.transformedBounds.max - $0.transformedBounds.min) * 0.001 }
    let spans = zip(highs, lows).map { $0 - $1 }

    func candidate(fromTransformed vec: [Float]) -> SubtractiveCandidate {
      var v = baseValues
      for (i, spec) in specs.enumerated() {
        v[spec.name] = spec.inverse(vec[i])
      }
      return SubtractiveCandidate(v)
    }

    var rng = SplitMix64(seed: searchSeed)

    // One graph, one compile; the same realized SignalTensor is reused for
    // every batch (see BatchBench.runTimingSweep for why rebuilding between
    // realizes is unsafe with fullCompilationCache).
    LazyGraphContext.reset()
    config.applyRuntime()
    let params = BatchBench.makeParams(batchSize: batchSize)
    let audio = BatchBench.buildAudio(params: params, config: config, frozen: baseValues)

    func scoreAll(_ vecs: [[Float]], round: Int) throws -> [Scored] {
      let cands = vecs.map { candidate(fromTransformed: $0) }
      var scores = [Float](repeating: .greatestFiniteMagnitude, count: cands.count)
      let batches = (cands.count + batchSize - 1) / batchSize
      let start = DispatchTime.now()
      for batch in 0..<batches {
        let lo = batch * batchSize
        let hi = min(lo + batchSize, cands.count)
        var slice = Array(cands[lo..<hi])
        while slice.count < batchSize { slice.append(slice[slice.count - 1]) }
        params.update(candidates: slice)
        let flat = try audio.realize(frames: config.frames)
        let lanes = BatchBench.deinterleave(flat, frames: config.frames, batchSize: batchSize)
        scores.withUnsafeMutableBufferPointer { buf in
          DispatchQueue.concurrentPerform(iterations: hi - lo) { i in
            buf[lo + i] = scorer.score(lanes[i])
          }
        }
        if (batch + 1) % 8 == 0 || batch == batches - 1 {
          let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
          print("  round \(round) batch \(batch + 1)/\(batches)"
            + " best=\(String(format: "%.6f", scores[0..<hi].min() ?? .nan))"
            + " elapsed=\(String(format: "%.1f", elapsed))s")
        }
      }
      return zip(vecs, zip(cands, scores)).map {
        Scored(transformed: $0, candidate: $1.0, score: $1.1)
      }
    }

    func bestPerStratum(_ archive: [Scored], topK: Int) -> [Int: [Scored]] {
      var byStratum: [Int: [Scored]] = [:]
      for s in archive {
        guard let key = stratumKey(fBase: s.candidate.fBase, shape: s.candidate.shape) else {
          continue
        }
        byStratum[key, default: []].append(s)
      }
      return byStratum.mapValues { Array($0.sorted { $0.score < $1.score }.prefix(topK)) }
    }

    // Round 0: uniform in transformed coordinates over the full bounds.
    var uniformVecs: [[Float]] = []
    uniformVecs.reserveCapacity(count)
    for _ in 0..<count {
      uniformVecs.append((0..<specs.count).map { d in rng.uniform(lows[d], highs[d]) })
    }
    print("basin-search v2: round 0 uniform count=\(count) searchSeed=\(searchSeed)")
    var archive = try scoreAll(uniformVecs, round: 0)

    // Resampling rounds: Gaussian children around each stratum's parents,
    // sigma shrinking by half each round.
    for round in 1...max(0, resampleRounds) where resampleRounds > 0 {
      let sigmaFraction = 0.08 / Foundation.pow(2.0, Float(round - 1))
      let parents = bestPerStratum(archive, topK: parentsPerStratum)
      var vecs: [[Float]] = []
      for key in parents.keys.sorted() {
        for parent in parents[key]! {
          for _ in 0..<childrenPerParent {
            var child = [Float](repeating: 0, count: specs.count)
            var d = 0
            while d < specs.count {
              let (g1, g2) = gaussianPair(&rng)
              child[d] = min(highs[d], max(lows[d],
                parent.transformed[d] + g1 * sigmaFraction * spans[d]))
              d += 1
              if d < specs.count {
                child[d] = min(highs[d], max(lows[d],
                  parent.transformed[d] + g2 * sigmaFraction * spans[d]))
                d += 1
              }
            }
            vecs.append(child)
          }
        }
      }
      print("round \(round): resampling \(vecs.count) children"
        + " (sigma=\(String(format: "%.3f", sigmaFraction)) of span)")
      archive.append(contentsOf: try scoreAll(vecs, round: round))
    }

    // Final stratified elites over the whole archive + 2 global extras.
    let finalStrata = bestPerStratum(archive, topK: 1)
    var elites: [Scored] = finalStrata.keys.sorted().compactMap { finalStrata[$0]?.first }
    let globalOrder = archive.sorted { $0.score < $1.score }
    var extras = 0
    for s in globalOrder {
      if elites.contains(where: { $0.score == s.score && $0.transformed == s.transformed }) {
        continue
      }
      elites.append(s)
      extras += 1
      if extras == 2 { break }
    }

    var strataResults: [StratumResult] = []
    for key in finalStrata.keys.sorted() {
      guard let s = finalStrata[key]?.first else { continue }
      let c = s.candidate
      strataResults.append(StratumResult(
        stratum: stratumLabel(key), score: s.score,
        fBase: c.fBase, shape: c.shape, pw: c.pw, res: c.res, fAmt: c.fAmt, fDecay: c.fDecay))
    }

    print("\nstratified elites (post-resampling):")
    for r in strataResults {
      print("  \(r.stratum): score=\(String(format: "%.6f", r.score))"
        + " fBase=\(String(format: "%.2f", r.fBase)) shape=\(String(format: "%.4f", r.shape))"
        + " pw=\(String(format: "%.4f", r.pw)) res=\(String(format: "%.3f", r.res))"
        + " fAmt=\(String(format: "%.1f", r.fAmt)) fDecay=\(String(format: "%.4f", r.fDecay))")
    }

    for (rank, s) in elites.enumerated() {
      let values = s.candidate.patchValues(basedOn: baseValues).clamped()
      try writeJSON(values, to: elitesDir.appendingPathComponent(
        String(format: "elite-%02d.json", rank)))
    }

    var report: [String: Any] = [:]
    report["policy"] = "v2: transformed-coordinate uniform prior + stratified Gaussian resampling"
    report["count"] = count
    report["batchSize"] = batchSize
    report["seed"] = seed
    report["searchSeed"] = String(searchSeed)
    report["resampleRounds"] = resampleRounds
    report["parentsPerStratum"] = parentsPerStratum
    report["childrenPerParent"] = childrenPerParent
    report["totalEvaluations"] = archive.count
    report["scoringObjective"] =
      "compare.py mrstft: windows 256/512/1024/2048, hop w/4, hann, log(mag+1e-3), L1"
    report["eliteScores"] = elites.map(\.score)
    report["strata"] = strataResults.map { r -> [String: Any] in
      [
        "stratum": r.stratum, "score": r.score,
        "fBase": r.fBase, "shape": r.shape, "pw": r.pw, "res": r.res,
        "fAmt": r.fAmt, "fDecay": r.fDecay,
      ]
    }
    report["globalTop20"] = globalOrder.prefix(20).map { s -> [String: Any] in
      let c = s.candidate
      return [
        "score": s.score, "fBase": c.fBase, "shape": c.shape,
        "pw": c.pw, "res": c.res, "fAmt": c.fAmt, "fDecay": c.fDecay,
      ]
    }
    let data = try JSONSerialization.data(
      withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: outDir.appendingPathComponent("basin_search_report.json"))
    print("\nwrote \(elites.count) elites to \(elitesDir.path)")
    print("wrote \(outDir.appendingPathComponent("basin_search_report.json").path)")
  }
}

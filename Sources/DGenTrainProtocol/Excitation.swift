// Excitation.swift — excitation convention measurement (patch-learn-spec §6).
//
// Single trigger at t=0, gate hard-coded on for N frames, pitch frozen to
// the CPU-estimated f0, velocity 1.0. This file measures the defaults from
// the target sample; CLI overrides always win. Pure CPU/Foundation code so
// Phase A/B tests run without Metal.

import Foundation

public enum Excitation {
    /// Steady f0 estimate: median over a few analysis windows of an
    /// MPM-style normalized autocorrelation (lags before the correlation's
    /// first negative crossing are ignored, which rejects the small-lag
    /// ridge that fools a global argmax on wide search bands — the reason
    /// PitchTrack.extract, tuned for narrow drum bands, is not reused
    /// verbatim here). nil when no confident pitch is found.
    public static func estimatePitchHz(
        samples: [Float], sampleRate: Float,
        minHz: Float = 30, maxHz: Float = 1000
    ) -> Double? {
        let windowSize = min(4096, largestPowerOfTwo(atMost: samples.count))
        guard windowSize >= 2048 else { return nil }
        let lagMin = max(2, Int(sampleRate / maxHz))
        let lagMax = min(windowSize - 2, Int(sampleRate / minHz))
        guard lagMin < lagMax else { return nil }

        var estimates: [Float] = []
        for fraction in [0.1, 0.35, 0.6] {
            let start = min(
                Int(Double(samples.count) * fraction),
                samples.count - windowSize)
            let frame = Array(samples[start..<(start + windowSize)])
            if let hz = framePitch(
                frame: frame, sampleRate: sampleRate, lagMin: lagMin, lagMax: lagMax)
            {
                estimates.append(hz)
            }
        }
        guard !estimates.isEmpty else { return nil }
        let sorted = estimates.sorted()
        return Double(sorted[sorted.count / 2])
    }

    private static func framePitch(
        frame: [Float], sampleRate: Float, lagMin: Int, lagMax: Int
    ) -> Float? {
        let n = frame.count
        var mean: Float = 0
        for v in frame { mean += v }
        mean /= Float(n)
        let x = frame.map { $0 - mean }
        var energy: Float = 0
        for v in x { energy += v * v }
        guard energy > 1e-7 else { return nil }

        var corr = [Float](repeating: 0, count: lagMax + 1)
        for lag in 1...lagMax {
            var num: Float = 0
            var lagEnergy: Float = 0
            for i in 0..<(n - lag) {
                num += x[i] * x[i + lag]
                lagEnergy += x[i + lag] * x[i + lag]
            }
            corr[lag] = num / max((energy * lagEnergy).squareRoot(), 1e-12)
        }

        // MPM-style: only consider lags after the first negative crossing.
        guard let firstNegative = (1...lagMax).first(where: { corr[$0] < 0 }) else {
            return nil
        }
        let searchStart = max(lagMin, firstNegative)
        guard searchStart < lagMax else { return nil }
        var bestLag = searchStart
        for lag in searchStart...lagMax where corr[lag] > corr[bestLag] {
            bestLag = lag
        }
        guard corr[bestLag] > 0.5, bestLag > 1, bestLag < lagMax else { return nil }

        // Parabolic refinement around the peak.
        let prev = corr[bestLag - 1]
        let peak = corr[bestLag]
        let next = corr[bestLag + 1]
        let denom = prev - 2 * peak + next
        let offset = abs(denom) > 1e-9 ? 0.5 * (prev - next) / denom : 0
        let lag = Float(bestLag) + max(-0.5, min(0.5, offset))
        return sampleRate / lag
    }

    /// Release-point estimate: gate stays on until the RMS amplitude
    /// envelope falls below `threshold` x peak (and stays there), after the
    /// peak. For a sustained sample this is near the sample end; for a
    /// one-shot drum it is early in the decay.
    public static func gateFrames(
        samples: [Float], window: Int = 1024, hop: Int = 256,
        threshold: Float = 0.15
    ) -> Int {
        guard samples.count > window else { return max(samples.count, 1) }
        var envelope: [Float] = []
        var start = 0
        while start + window <= samples.count {
            var sum: Float = 0
            for i in start..<(start + window) {
                sum += samples[i] * samples[i]
            }
            envelope.append((sum / Float(window)).squareRoot())
            start += hop
        }
        guard let peak = envelope.max(), peak > 0 else { return samples.count }
        let peakIndex = envelope.firstIndex(of: peak) ?? 0
        let floorLevel = threshold * peak

        // First post-peak index where the envelope drops below threshold and
        // never comes back above it (release point).
        var release = envelope.count
        var i = envelope.count - 1
        while i > peakIndex {
            if envelope[i] >= floorLevel { break }
            release = i
            i -= 1
        }
        if release == envelope.count { return samples.count }
        return min(samples.count, release * hop + window / 2)
    }

    /// Crop length from the sample, bounded by trainer frame limits.
    public static func cropFrames(sampleCount: Int, maxFrames: Int = 65536) -> Int {
        max(1024, min(sampleCount, maxFrames))
    }

    static func largestPowerOfTwo(atMost n: Int) -> Int {
        var p = 1
        while p * 2 <= n { p *= 2 }
        return p
    }
}

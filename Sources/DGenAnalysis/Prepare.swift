// Target preparation (the convention the trainer uses for its targets):
// mono = mean of channels; onset = first |x| above peak·10^(-40/20);
// crop at onset - preroll (2 ms); peak-normalise to 0.9.
// The analysis time origin is sample 0 of this prepared signal.

import Foundation

public struct PreparedTarget {
    public var name: String
    public var sampleRate: Double
    public var signal: [Double]        // prepared, native length (no padding)
    public var cropStart: Int          // raw sample index of prepared sample 0
    public var onsetRawSample: Double  // sub-sample threshold crossing, raw index
    public var onsetSeconds: Double    // onset in prepared time
    public var rawPeak: Double
    public var gain: Double            // prepared = gain * raw mono
    public var prerollSamples: Int
    public var channels: Int
    public var rawFrames: Int
}

public enum Prepare {
    public static func prepare(name: String, audio: WavAudio, thresholdDB: Double = -40,
                               prerollSeconds: Double = 0.002, peak: Double = 0.9) -> PreparedTarget {
        let x = audio.mono
        let sr = audio.sampleRate
        let pk = x.map { abs($0) }.max() ?? 0
        let thr = pk * pow(10, thresholdDB / 20)
        let i = x.firstIndex { abs($0) > thr } ?? 0
        var sub = Double(i)
        if i > 0 {
            let a = abs(x[i - 1]), b = abs(x[i])
            if b > a { sub = Double(i - 1) + (thr - a) / (b - a) }
        }
        let pre = Int((prerollSeconds * sr).rounded())
        let s = max(0, i - pre)
        let y = Array(x[s...])
        let ypk = y.map { abs($0) }.max() ?? 1
        let g = ypk > 0 ? peak / ypk : 1
        return PreparedTarget(name: name, sampleRate: sr, signal: y.map { $0 * g }, cropStart: s,
                              onsetRawSample: sub, onsetSeconds: (sub - Double(s)) / sr, rawPeak: pk, gain: g,
                              prerollSamples: pre, channels: audio.channels.count, rawFrames: audio.frameCount)
    }
}

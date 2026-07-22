import DGen
import XCTest

@testable import DGenLazy

/// Does the tinygrad-style rebuild-per-epoch loop return the same loss for
/// identical params? Uses the monologue-style SVF voice (Signal.history()
/// cells + statefulPhasor + accum) against a fixed tensor target.
final class EpochRebuildDriftScratchTests: XCTestCase {
  override func setUp() {
    super.setUp()
    DGenConfig.maxFrameCount = 32768
  }

  func testLossStableAcrossEpochRebuilds() throws {
    let frames = 26624
    let root = "/Users/alecresende/code/swift/dgen/output/monologue_bass"
    let (targetRaw, _) = try AudioFile.load(
      url: URL(fileURLWithPath: "\(root)/mono_rung2_target.wav"))
    let target = Array(targetRaw.prefix(frames))
    DGenSpectralConfig.logMagnitudeEpsilon = 1e-3

    func buildVoice() -> Signal {
      let sr = Signal.constant(44100.0)
      let t = Signal.accum(Signal.constant(1.0 / 44100.0), reset: 0.0, min: 0.0, max: 1.0)
      let f0 = Signal.constant(35.3678)
      let phase1 = Signal.statefulPhasor(f0)
      let phase2 = mod(phase1 - t * 0.65, 1.0)
      let dt = (f0 / sr).clip(0.000001, 0.5)
      func osc(_ p: Signal) -> Signal {
        let blep: (Signal) -> Signal = { q in
          let lX = q / dt
          let l = 2.0 * lX - lX * lX - 1.0
          let rX = (q - 1.0) / dt
          let r = rX * rX + 2.0 * rX + 1.0
          return (q < dt) * l + (q > (1.0 - dt)) * r
        }
        let saw = (p * 2.0 - 1.0) - blep(p)
        let falling = mod(p - 0.55, 1.0)
        let rawPulse = (p < 0.55) * 2.0 - 1.0
        let pulse = rawPulse + blep(p) - blep(falling)
        return 0.4 * saw + 0.6 * pulse
      }
      let mixed = (osc(phase1) + 0.6 * osc(phase2)) / 1.6
      let y = 1.8 * mixed + 0.12
      let pre = y + 0.15 * (y * y) + 0.2 * (y * y * y)
      let cutoff = (Signal.constant(300.0) + 900.0 * DGenLazy.exp(-t / 0.15)).clip(20.0, 20000.0)
      let g = DGenLazy.tan(Signal.constant(Float.pi) * cutoff / sr)
      let kDamp = Signal.constant(1.0 / 1.2)
      let a1 = 1.0 / (1.0 + g * (g + kDamp))
      let a2 = g * a1
      let ic1 = Signal.history()
      let ic2 = Signal.history()
      let kSat = Signal.constant(1.2)
      let s1 = ic1.read / (1.0 + DGenLazy.abs(kSat * ic1.read))
      let s2 = ic2.read / (1.0 + DGenLazy.abs(kSat * ic2.read))
      let v3 = pre - s2
      let v1 = a1 * s1 + a2 * v3
      let ic1New = ic1.write(2.0 * v1 - s1)
      let v1PT = (ic1New + s1) * 0.5
      let v2 = s2 + g * v1PT
      let ic2New = ic2.write(2.0 * v2 - s2)
      let lp = (ic2New + s2) * 0.5
      let attack = 1.0 - DGenLazy.exp(-t / 0.005)
      let decay = DGenLazy.exp(-t / 0.2)
      let release = 1.0 / (1.0 + DGenLazy.exp((t - 0.5) / 0.02))
      let driven = lp * attack * decay * release * 2.5
      return (driven / (1.0 + DGenLazy.abs(driven))) * 0.45
    }

    var losses: [Float] = []
    LazyGraphContext.reset()
    let dummy = Signal.param(0.45)  // a grad target so backward machinery engages
    for _ in 0..<4 {
      let synth = buildVoice() * (dummy / 0.45)
      let targetSig = Tensor(target).toSignal(maxFrames: frames)
      var total = Signal.constant(0.0)
      for w in [256, 512, 1024, 2048] {
        total = total + spectralLossFFT(
          synth, targetSig, windowSize: w, useHannWindow: true,
          useLogMagnitude: true, lossMode: .l1, hop: max(1, w / 4), normalize: true)
      }
      let vals = try total.backward(frames: frames)
      losses.append(vals.reduce(0, +))
      dummy.grad = nil
      LazyGraphContext.current.clearComputationGraph()
    }
    print("epoch losses: \(losses)")
    for l in losses.dropFirst() {
      XCTAssertEqual(l, losses[0], accuracy: max(1e-3, abs(losses[0]) * 1e-3),
                     "loss drifts across epoch rebuilds")
    }
  }
}

import XCTest

@testable import DGen
@testable import DGenLazy

/// End-to-end proof of the soothe-class spectral substrate: the two pieces we
/// built — `cumsum` (O(N) wide spectral smoothing) and hop-gated `TensorHistory`
/// (per-bin attack/release env follower) — working together inside a real STFT
/// pipeline: input → buffer → FFT → per-bin processing → IFFT → overlapAdd.
///
/// The processing is a minimal soothe: compare each magnitude bin to its local
/// spectral average (a box filter computed via cumsum differences) and attenuate
/// bins that stick out above the neighborhood (resonances), smoothing the gain
/// over time with a one-pole follower that ticks once per hop.
///
/// What this proves:
///  1. The full path compiles and runs (cumsum + hop-history live in one FFT
///     hop pipeline together).
///  2. The hop-gated env follower is genuinely driven by the cumsum-smoothed
///     gain: at the resonant bin it releases from 1.0 toward the computed
///     attenuation across successive hops.
///  3. The soothe attenuates the resonant tone — output energy is lower than a
///     bypass (identical pipeline, gain = 1).
final class SootheSubstrateTests: XCTestCase {

  override func setUp() {
    super.setUp()
    DGenConfig.sampleRate = 16  // fs == N: a k Hz tone lands exactly in bin k.
    DGenConfig.maxFrameCount = 256
    LazyGraphContext.reset()
  }

  private let N = 16          // FFT size / bins
  private let hop = 8         // STFT hop (50% overlap)
  private let w = 2           // box-filter radius (window = 2w+1 = 5 bins)
  private let peakBin = 6     // interior resonant bin
  private let frames = 96     // ~12 hops — enough for the follower to settle

  private enum Readout { case output, envAtPeak }

  /// Builds the soothe substrate and realizes the requested terminal.
  /// `sootheEnabled == false` is the bypass control (pure FFT→IFFT→OLA).
  private func runSoothe(backend: Backend, sootheEnabled: Bool, readout: Readout) throws -> [Float] {
    DGenConfig.backend = backend
    LazyGraphContext.reset()

    // Steady resonant tone at bin `peakBin`.
    let tone = sin(Signal.phasor(Float(peakBin)) * Signal.constant(2 * .pi))

    // STFT analysis (hop-gated window → FFT).
    let frame = tone.buffer(size: N, hop: hop).reshape([N])
    let (re, im) = signalTensorFFT(frame, N: N)

    // Per-bin magnitude.
    let mag = sqrt(re * re + im * im)  // [N]

    // Local spectral average via cumsum box filter:
    //   boxsum[i] = c[i+w] - c[i-w-1]   (prefix-sum difference, O(N))
    // Edges zero-fill; the gain clamp below keeps them well-behaved.
    let c = mag.cumsum()  // [N]
    let right = c.pad([(0, w)]).shrink([(w, w + N)])        // c[i+w]
    let left = c.pad([(w + 1, 0)]).shrink([(0, N)])         // c[i-w-1]
    let smooth = (right - left) * Signal.constant(1.0 / Float(2 * w + 1))

    // Soothe gain: attenuate bins above their neighborhood average. Flat bins
    // get ~1; a bin sticking out by a peak gets smooth/mag < 1.
    let eps = Signal.constant(1e-6)
    let rawGain = smooth / (mag + eps)
    let gain = min(max(rawGain, 0.0), 1.0)  // [N], clamped to [0,1]

    // Hop-gated per-bin env follower (one-pole release), state held once per hop.
    // Init to 1.0 so it RELEASES downward into attenuation over hops.
    let env = TensorHistory(shape: [N], hop: hop, data: [Float](repeating: 1.0, count: N))
    let prev = env.read()
    let a = Signal.constant(0.7)  // release coefficient
    let newEnv = prev * a + gain * Signal.constant(0.3)
    let applied = env.write(newEnv)  // pass-through; keeps feedback in the graph

    if readout == .envAtPeak {
      // Read the follower state at the resonant bin (hop-gated: valid on hop frames).
      return try applied.shrink([(peakBin, peakBin + 1)]).sum().realize(frames: frames)
    }

    // The follower value is hop-GATED (defined only on hop frames). The IFFT is a
    // per-frame consumer, so hold the gain across the hop before applying it —
    // otherwise the spectrum reads 0 between hops. This is the canonical way to
    // consume hop-gated state inside an FFT region.
    let heldGain = applied.hopHold(hop: hop)

    // Apply the (soothe or bypass) gain and resynthesize.
    let gReal = sootheEnabled ? re * heldGain : re
    let gImag = sootheEnabled ? im * heldGain : im
    let out = signalTensorIFFT(gReal, gImag, N: N).overlapAdd(hop: hop)
    return try out.realize(frames: frames)
  }

  private func rms(_ x: ArraySlice<Float>) -> Float {
    guard !x.isEmpty else { return 0 }
    let ss = x.reduce(Float(0)) { $0 + $1 * $1 }
    return (ss / Float(x.count)).squareRoot()
  }

  // MARK: - 1. Full path runs on C

  func testFullPathRunsOnC() throws {
    let c = try runSoothe(backend: .c, sootheEnabled: true, readout: .output)

    XCTAssertEqual(c.count, frames)
    XCTAssertTrue(c.allSatisfy { $0.isFinite }, "output must be finite")
    XCTAssertGreaterThan(rms(c[(frames - 32)...]), 0, "output must be non-zero")
  }

  // MARK: - 2. Hop-gated follower is driven by the cumsum-smoothed gain

  func testEnvFollowerReleasesPeakBinAcrossHops() throws {
    let c = try runSoothe(backend: .c, sootheEnabled: true, readout: .envAtPeak)

    // The follower value is defined on hop frames (hop-gated). Sample those,
    // skipping the very first hop while the analysis ring fills.
    let hopVals = stride(from: 0, to: frames, by: hop).map { c[$0] }
    print("=== peak-bin env per hop: \(hopVals)")
    let settled = Array(hopVals.dropFirst())  // drop ring-fill transient

    // Started at 1.0; must release downward (attenuation increasing) ...
    XCTAssertLessThan(settled.last!, settled.first!, "follower did not release over hops")
    XCTAssertLessThan(settled.last!, 0.6, "peak bin should end clearly attenuated")
    // ... and stay in a sane attenuated range (gain at the peak ≈ 1/window = 0.2).
    XCTAssertGreaterThan(settled.last!, 0.1)
    // Monotone non-increasing release across settled hops.
    for k in 1..<settled.count {
      XCTAssertLessThanOrEqual(
        settled[k], settled[k - 1] + 1e-4, "release should be monotone at hop \(k)")
    }
  }

  // MARK: - 3. Soothe attenuates the resonant tone vs bypass

  func testSootheAttenuatesResonanceVsBypass() throws {
    let soothed = try runSoothe(backend: .c, sootheEnabled: true, readout: .output)
    let bypass = try runSoothe(backend: .c, sootheEnabled: false, readout: .output)

    // Compare settled-region energy (after follower release + OLA latency).
    let tail = (frames - 32)..<frames
    let soothedRMS = rms(soothed[tail])
    let bypassRMS = rms(bypass[tail])
    print("=== soothed RMS \(soothedRMS) vs bypass RMS \(bypassRMS)")

    XCTAssertGreaterThan(bypassRMS, 0)
    XCTAssertLessThan(
      soothedRMS, bypassRMS * 0.9,
      "soothe should reduce the resonant tone's energy relative to bypass")
  }
}

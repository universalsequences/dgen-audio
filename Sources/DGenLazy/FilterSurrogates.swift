import DGen
import Foundation

/// A differentiable, frequency-sampled approximation of the Cytomic/TPT SVF.
///
/// The cutoff, Q and mode are sampled once per STFT hop. The filter's closed-form
/// magnitude response is applied to a full, symmetric complex spectrum, then
/// returned to the time domain with overlap-add. This deliberately has zero
/// phase; it is intended for magnitude-spectral training losses, not rendering.
///
/// Modes match the patch SVF convention: 0=lowpass, 1=bandpass, 2=highpass,
/// 3=notch, 4=peak, 5=allpass.
public func svfFrequencySampled(
  _ input: Signal,
  cutoff: Signal,
  q: Signal,
  mode: Signal,
  window n: Int = 1024,
  hop: Int = 256,
  sampleRate: Float
) -> Signal {
  precondition(n >= 2 && n.nonzeroBitCount == 1, "SVF surrogate window must be a power of two")
  precondition(hop > 0 && hop <= n && n % hop == 0, "SVF surrogate hop must divide window")
  precondition(sampleRate > 0, "SVF surrogate sample rate must be positive")

  // The buffer is [1,N]; flatten it before the FFT. Creating all constant
  // tensors here also ensures that they belong to the current lazy graph.
  let analysisWindow = hann(n)
  let frame = input.buffer(size: n, hop: hop).reshape([n]) * analysisWindow
  let spectrum = signalTensorFFT(frame, N: n)

  // Full-spectrum, mirrored bin frequencies keep the IFFT real. Clamp both
  // ends away from tan's singular/degenerate points.
  let binPrewarp = Tensor((0..<n).map { i in
    let mirrored = Swift.min(i, n - i)
    let hz = Swift.min(Swift.max(Float(mirrored) * sampleRate / Float(n), 1), 0.499 * sampleRate)
    return Foundation.tan(Float.pi * hz / sampleRate)
  })

  // Sample the controls once per hop (matching the buffer's hop grid). This
  // also tags them hop-rate so the whole mask construction — and its backward
  // — is hop-gated and hop-slot allocated instead of demoted to per-frame.
  let safeCutoff = cutoff.clip(1, Double(0.49 * sampleRate)).hopHold(hop: hop)
  let g = DGenLazy.tan(safeCutoff * (Float.pi / sampleRate))
  let safeQ = DGenLazy.max(q, Signal.constant(0.001)).hopHold(hop: hop)
  // Lift each scalar control once. Keeping all response arithmetic tensor-shaped
  // avoids multiple independent broadcast reductions into the same scalar.
  let tensorOnes = Tensor([Float](repeating: 1, count: n))
  let gTensor = tensorOnes * g
  let kTensor = tensorOnes * (1.0 / safeQ)
  let w = SignalTensor.lift(binPrewarp) / gTensor
  let w2 = w * w
  let oneMinusW2 = 1.0 - w2
  let denominator = oneMinusW2 * oneMinusW2 + w2 * kTensor * kTensor

  let lp2 = 1.0 / denominator
  let bp2 = w2 / denominator
  let hp2 = w2 * w2 / denominator
  let notch2 = oneMinusW2 * oneMinusW2 / denominator
  let onePlusW2 = w2 + Float(1.0)
  let peak2 = onePlusW2 * onePlusW2 / denominator
  let shape = [n]
  let modeT = SignalTensor.lift(mode, shape: shape)
  let zero = SignalTensor.lift(Signal.constant(0), shape: shape)
  let one = SignalTensor.lift(Signal.constant(1), shape: shape)

  // Equality masks mirror selector-style mode dispatch. An unknown mode falls
  // back to allpass rather than muting the signal.
  let m0 = modeT.eq(SignalTensor.lift(Signal.constant(0), shape: shape))
  let m1 = modeT.eq(SignalTensor.lift(Signal.constant(1), shape: shape))
  let m2 = modeT.eq(SignalTensor.lift(Signal.constant(2), shape: shape))
  let m3 = modeT.eq(SignalTensor.lift(Signal.constant(3), shape: shape))
  let m4 = modeT.eq(SignalTensor.lift(Signal.constant(4), shape: shape))
  let m5 = modeT.eq(SignalTensor.lift(Signal.constant(5), shape: shape))
  let selected = m0 + m1 + m2 + m3 + m4 + m5
  let magnitudeSquared =
    m0 * lp2 + m1 * bp2 + m2 * hp2 + m3 * notch2 + m4 * peak2
    + (m5 + (one - selected)) * one + zero
  let mask = DGenLazy.sqrt(magnitudeSquared)

  let reconstructed = signalTensorIFFT(spectrum.re * mask, spectrum.im * mask, N: n)
  // A periodic Hann's overlap sum is N/(2H). We use one analysis window and
  // no synthesis window, as specified for v0.1.
  let colaGain = Float(n) / Float(2 * hop)
  return reconstructed.overlapAdd(hop: hop) / colaGain
}

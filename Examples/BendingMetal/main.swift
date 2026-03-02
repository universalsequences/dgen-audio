// BendingMetal — Time-varying nonlinear plate resonator
//
// Physical model of a struck metal plate being slowly bent.
// The bending changes the tension field across the plate,
// which shifts wave propagation speed, which shifts resonant
// frequencies, which changes how subsequent reflections interfere.
// The plate's own vibration energy feeds back into the tension
// field (structural feedback), so the system is never in the
// same state twice.
//
// H(z,t) = H0(z) + DeltaH(z, t, s(t))
//
// where s(t) is the plate's tension state vector.

import DGenLazy
import Foundation

// ── Configuration ──────────────────────────────────────────────

let sampleRate: Float = 44100.0
let duration: Float = 3.0
let frameCount = Int(sampleRate * duration)
let N = 16  // grid points per side

// Physical parameters
let baseTension: Float = 0.12        // base c² (wave speed squared)
let damping: Float = 0.00001         // very light damping → ~1.5s energy half-life
let tensionCoupling: Float = 0.0003  // vibration energy → tension feedback
let tensionRelax: Float = 0.9998     // tension relaxation rate toward base
let bendDepth: Float = 0.035         // external bend modulation amplitude

// Wave equation coefficients
let twoMinusD: Float = 2.0 - damping
let oneMinusD: Float = 1.0 - damping
let twoPi = Float.pi * 2.0

// ── Setup ──────────────────────────────────────────────────────

DGenConfig.backend = .metal
DGenConfig.sampleRate = sampleRate
DGenConfig.maxFrameCount = frameCount
DGenConfig.enableBufferReuse = false
LazyGraphContext.reset()

// ── Excitation ─────────────────────────────────────────────────
// Strike slightly off-center for asymmetric mode excitation.
// An off-center strike excites more modes than a center strike,
// producing a richer, more metallic timbre.

var exciteData = [Float](repeating: 0.0, count: N * N)
exciteData[5 * N + 7] = 0.8
exciteData[5 * N + 8] = 0.5
exciteData[6 * N + 7] = 0.5
exciteData[6 * N + 8] = 0.3
let excitationPattern = Tensor(exciteData).reshape([N, N])

let click = Signal.click()
let gatedExcite = excitationPattern * click  // fires on frame 0 only

// ── Initial tension field (uniform) ────────────────────────────

let initialTension = [Float](repeating: baseTension, count: N * N)

// ── Bend gesture: spatial tension gradients ────────────────────
// Two incommensurate modulation rates so the combined tension
// pattern never repeats. This creates ever-evolving phase
// relationships between the plate's resonant modes.

let bendPhase1 = Signal.phasor(0.3)   // slow horizontal bend
let bendPhase2 = Signal.phasor(0.17)  // slower diagonal bend (incommensurate)
let bendMod1 = sin(bendPhase1 * twoPi)
let bendMod2 = sin(bendPhase2 * twoPi)

// Horizontal gradient: top gets tighter, bottom gets looser
var horizGradData = [Float](repeating: 0.0, count: N * N)
for row in 0..<N {
  let v = (Float(row) / Float(N - 1) - 0.5) * 2.0  // -1..+1
  for col in 0..<N { horizGradData[row * N + col] = v }
}
let horizGrad = Tensor(horizGradData).reshape([N, N])

// Diagonal gradient: top-left vs bottom-right
var diagGradData = [Float](repeating: 0.0, count: N * N)
for row in 0..<N {
  for col in 0..<N {
    let v = (Float(row + col) / Float(2 * (N - 1)) - 0.5) * 2.0
    diagGradData[row * N + col] = v
  }
}
let diagGrad = Tensor(diagGradData).reshape([N, N])

// Combined bend field (Tensor * Signal → SignalTensor)
let bendField = horizGrad * bendMod1 * bendDepth
             + diagGrad * bendMod2 * (bendDepth * 0.6)

// ── State histories ────────────────────────────────────────────

let stateHistory = TensorHistory(shape: [N, N])
let prevStateHistory = TensorHistory(shape: [N, N])
let tensionHistory = TensorHistory(shape: [N, N], data: initialTension)

// ── Read state ─────────────────────────────────────────────────

let state_t_raw = stateHistory.read()
let state_t_1 = prevStateHistory.read()
let tension_t = tensionHistory.read()

// Inject excitation on first frame
let state_t = state_t_raw + gatedExcite

// ── Laplacian (discrete ∇²) via conv2d ─────────────────────────
// Dirichlet boundary conditions (fixed edges at 0) via zero-padding.

let laplacianKernel = Tensor([
  [0.0,  1.0, 0.0],
  [1.0, -4.0, 1.0],
  [0.0,  1.0, 0.0],
])

let paddedState = state_t.pad([(1, 1), (1, 1)])
let laplacian = paddedState.conv2d(laplacianKernel)

// ── Variable-tension wave equation ─────────────────────────────
// state_{t+1} = (2-d)*state_t - (1-d)*state_{t-1} + T(x,t)*∇²state_t
//
// The key: T(x,t) varies in space AND time. Each grid point has
// its own local wave speed. Waves passing through regions of
// different tension arrive back with shifted phase — structural
// feedback through a changing material.

let scaledLaplacian = laplacian * tension_t
let state_next = state_t * twoMinusD - state_t_1 * oneMinusD + scaledLaplacian

// ── Tension dynamics ───────────────────────────────────────────
// The plate's vibration energy feeds back into the tension field.
// When you hold a piece of sheet metal and let it ring, the
// vibration itself changes the stress distribution. The sound
// travels through different stiffness at different points.

let velocity = state_t - state_t_1
let localEnergy = velocity * velocity

// Relax toward base tension + energy feedback + external bend
let relaxed = tension_t * tensionRelax + baseTension * (1.0 - tensionRelax)
let tension_unclamped = relaxed + localEnergy * tensionCoupling + bendField

// Clamp tension for CFL stability: 2D 5-point Laplacian requires c² < 0.25
let tension_next = max(min(tension_unclamped, 0.24), 0.01)

// ── Write state back ───────────────────────────────────────────

prevStateHistory.write(state_t)
stateHistory.write(state_next)
tensionHistory.write(tension_next)

// ── Output: asymmetric pickup points ───────────────────────────
// Three pickup positions at different locations on the plate.
// Each hears the same modes but with different phase relationships.
// When summed, this creates natural chorusing — the interference
// geometry of the material rendered as sound. Not an effect.
// A structural principle.

var pickupData = [Float](repeating: 0.0, count: N * N)
pickupData[3  * N + 4]  = 1.0   // near top-left
pickupData[5  * N + 11] = 0.8   // right of center
pickupData[11 * N + 6]  = 0.7   // lower-left area
let pickupMask = Tensor(pickupData).reshape([N, N])

let output = (state_next * pickupMask).sum()

// ── Render ─────────────────────────────────────────────────────

print("Rendering \(duration)s of bending metal plate (\(N)x\(N) grid, \(frameCount) frames)...")

let samples = try output.realize(frames: frameCount)

// Normalize to -0.8..+0.8 peak
let peak = samples.map { Swift.abs($0) }.max() ?? 1.0
let gain: Float = peak > 0 ? 0.8 / peak : 1.0
let normalized = samples.map { $0 * gain }

let outputPath = "bending_metal.wav"
let outputURL = URL(fileURLWithPath: outputPath)
try AudioFile.save(url: outputURL, samples: normalized, sampleRate: sampleRate)

print("Peak amplitude before normalization: \(peak)")
print("Saved to \(outputURL.path)")

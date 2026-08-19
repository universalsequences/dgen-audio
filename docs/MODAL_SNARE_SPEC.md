# Spec: Modal snare — latent drum model on a modal + noise substrate

Status: PROPOSED — 2026-08-17

## Summary

Fit a corpus of real snare one-shots with a differentiable **modal bank +
filtered-noise** drum synth, then train a DeepSDF-style **auto-decoder** over
the fits so the whole corpus becomes an 8-ish-knob latent instrument: drag the
latent, hear the snare morph; record a new snare, fit its latent in seconds,
get back an editable patch. Product shape: "every snare you own, as one knob-
controllable drum synth," with the trained artifact being plain DSP parameters
— nothing neural in the audio path.

This is Track C + the auto-decoder core of Track B from the direction doc,
built as a SynthID-style gated ladder. The strategic reason to run it first:
the genuinely new machinery (auto-decoder trainer, latent staging, export) gets
built on the **best-conditioned substrate we have** — a linear sum of damped
sinusoids — so a bad interpolation at M2 means "trainer bug," not "maybe the
gradients are bad." The subtractive latent track then inherits a proven
trainer.

## Why this is mostly recombination, not new build

| Component | Status in repo |
|---|---|
| Fixed-frequency sinusoid bank `[K]` | tensor `statefulPhasor` + `.sum(axis:)` — the DDSPE2E harmonic bank, proven through R0–R7 prep |
| Per-mode exponential decay envelopes | elementwise `exp` on a time accumulator × `[K]` param tensors — no recurrence, no BPTT |
| Filtered noise | DDSPE2E 15-tap time-domain FIR branch (won the R2 bake-off vs FD) |
| Multi-scale spectral loss | `spectralLossFFT`, lin+log, `logMagnitudeEpsilon = 1e-3` (post scale-bug, post hop-gate fixes) |
| Selection metric | CPU MR-STFT scorer (`Examples/DDSPE2E/BestMetric.swift`), calibrated against listening at R0/R6 |
| Decoder MLP | `matmul` (GEMMPass-fused fwd+bwd), relu/sigmoid, validated by `TransformerOpsTests` |
| Latents as trainable params | `Tensor.param` (auto-decoder = DeepSDF; latents trained by descent, no encoder) |
| Optional room/shell tail | `learnedReverb` (R5: exact overlap-add conv, FD cosine ~1.0) |
| Trainer scaffolding | DDSPE2E Trainer/Checkpoint/Dataset patterns, incl. the R6 pinned-chunk selection fix |

The one deliberately **avoided** component: recursive resonators during
training. The training form is closed-form additive (below), so the mode bank
has zero recurrence — no BPTT cost, no high-Q vanishing/exploding, and the
whole render is frame-parallel on Metal. Recursive form is export-only.

## Synthesis model

### Training form (closed-form, fully parallel)

```
y(t) = Σ_k  g_k · exp(-t / τ_k) · sin(2π f_k t)              (modal branch)
     +      n_g · exp(-t / n_τ) · FIR15(white(t))            (noise branch)
```

- `t` in seconds from onset (scalar time accumulator, broadcast against `[K]`).
- **Frequencies `f_k` are FIXED on a log-spaced grid** — not learned. K modes
  from 120 Hz to 14 kHz (covers snare fundamental ~160–250 Hz through wire
  sizzle). This is the permutation-ambiguity fix AND the local-minima fix:
  CLAUDE.md's resolution math says frequency gradients inside a bin are a
  trap, and the direction doc's mode-matching problem disappears when mode k
  means the same frequency in every sample. What we train is a **learned
  filterbank**: per-mode gain + decay on a shared grid.
- `g_k`: per-mode gain, sigmoid-squashed, `[K]`.
- `τ_k`: per-mode decay, **log-parameterized** in [5 ms, 2 s], `[K]`.
- Noise branch: learned 15 FIR taps (static per patch to start), gain, log-τ.
  Noise τ is **capped shorter** than mode τ floor is long — see degeneracy
  guard below.
- Optional, OFF by default: per-mode detune `δ_k = tanh(·) × (half local grid
  spacing)` — bounded so modes cannot cross or alias, evaluated only if the M1
  gate shows grid quantization is the binding residual. Phases fixed at 0
  (loss is magnitude-STFT; phase-blind is fine for this).

Per-patch parameter count at K=64: 64 gains + 64 decays + 15 taps + 2 noise +
1 global gain ≈ **146 params**. At K=128 ≈ 274.

### Export form (analytic, no training)

Each damped sinusoid maps exactly to a coupled-form recursive oscillator
(2 mul + 2 add, no transcendental) or a two-pole resonator; the FIR noise
branch ships as-is. ~5 flops/mode/sample → 200 modes ≈ 50 MFLOPS, <1% core.
The mapping is closed-form, so export is a serializer, not a training step.
(The repo's `SignalTensor.biquad` bandpass path is the fallback runtime form
if coupled-form isn't wired in eseq yet; forward + BPTT both validated, though
we shouldn't need gradients there.)

## The ladder

House rules apply: fixed seed, deterministic runs, numeric gate + listen gate
(`listen` skill) per rung, artifact trail (loss CSV, preview WAVs, checkpoint)
under `runs/`. A failed gate stops the ladder and becomes a diagnosis task.
Gate on **sound**, not parameters (the SynthID/E3 lesson) — except M0, where
parameters are the whole point.

### M0 — Synthetic recovery (no network, no real data)

Render a known drum with the training-form synth itself (K=32, e.g. inharmonic
mode stack loosely from membrane ratios + noise burst), then recover `g, τ,
taps, n_g, n_τ` from audio alone via MR-STFT loss, params initialized flat.

Gates:
- fdcheck on a small config (K=4, short render): cosine > 0.999 for every
  param group. Cheap and it pins the `exp(-t/τ)` log-parameterization
  gradients before anything real depends on them.
- Loss collapses ≥100× toward the render-noise floor.
- Recovered τ within 10% for every mode with `g_k` above −40 dBFS; gain vector
  cosine ≥ 0.99. (Inaudible modes are allowed to be wrong — that's the
  quasi-degenerate-direction lesson from E3: don't gate on perceptually null
  params.)

Wall-clock expectation: minutes. This rung exists purely for failure
attribution — after it, a bad M1 fit cannot be a gradient bug in the
substrate.

### M1 — Single real snare

One snare one-shot → full training-form fit. Also the rung where the model
class gets sized.

Protocol:
- Data prep (see §Data prep): 44.1 kHz, onset at sample 0, peak-normalized,
  fixed 0.75 s render.
- Sweep K ∈ {32, 64, 128}. Sweep is cheap; the additive bank is parallel.
- **Ablation A/B: modal-only vs modal+noise.** The noise branch must earn its
  place — if modal-only matches, the wires are being eaten by high modes and
  the degeneracy guard below isn't working.
- Init: `g_k` from the target's spectral envelope sampled at the grid
  frequencies (warm start — the wavetable trick from the direction doc,
  adapted); τ flat at ~150 ms; taps ~lowpass; noise τ ~80 ms.

Gates:
- Numeric: CPU MR-STFT selection score in the "sounds like the target" band.
  Calibrate first on this material exactly as R0 did (score the target vs
  itself, vs white noise, vs a wrong snare) — the flute calibration
  (0.67 exact / 2.98 noise) does not transfer to transients. Provisional gate:
  score within 1.5× of the self-noise floor; tighten after calibration.
- Listen: A/B against the target — "same drum, convincing snare." Also listen
  to the two branches solo'd: modal branch should sound like a damped shell
  tone, noise branch like the wires. If the split sounds wrong but the sum
  sounds right, the M2 latent space will be built on a degenerate
  decomposition — treat as a soft fail and strengthen the guard.
- Record: smallest K that passes, per-branch energy split, wall-clock.

### M2 — Auto-decoder on 30 snares (the go/no-go rung)

The direction doc's B10.1 milestone, on modal. New build: the auto-decoder
trainer (§Trainer), reusable afterward for the subtractive track.

- Latent `z_i ∈ R^4` per snare, `Tensor.param`, init N(0, 0.01²).
- Decoder MLP `4 → 128 → 128 → P` (P from M1's winning K), outputs squashed
  into the same ranges as M1 (sigmoid gains, log-space τ).
- Loss: `L_mrstft + λ₂‖z‖²`, λ₂ ≈ 1e-4. No mod matrix here, so no L1 term —
  the modal substrate's canonicalization is structural (fixed grid), which is
  exactly why this rung isolates the trainer.
- Schedule: joint decoder+latents to plateau, then latent-only polish (freeze
  decoder, few hundred steps). Two stages, not three — there is no modulator
  stage on this substrate.
- Batching: start serial with shuffle (the R6 recipe, incl. pinned-selection
  fix); move to `[B]` lanes only if wall-clock demands it.

Gates:
1. **Decoder tax**: per-sample fitted loss within 1.5× of that sample's solo
   M1-style fit. (Measures what the shared decoder costs; if it's huge, latent
   dim or capacity is wrong.)
2. **Interpolation listen**: 10 random pairs, render z-midpoints. Every
   midpoint must sound like a plausible snare — no combing, no mush, no
   silence. This is the rung's entire reason to exist.
3. **Round-trip**: hold out 3 snares. Fit each by latent-only descent (freeze
   decoder, few hundred steps on z). Render, listen: recognizably that drum.

Fail modes route by symptom exactly as the direction doc's B9 table (averaged
→ latent dim/λ₂; great fits + bad interpolation → dim too high; plateau high
→ corpus exceeds K, raise K or narrow corpus).

### M3 — Full corpus + export

- Scale to the full snare library (hundreds). Latent dim sweep {4, 8, 16};
  pick by gate-2/gate-3 quality, not fit loss.
- Val split grouped by source pack/recording (the R7 leakage lesson).
- **Export**: patch = `{ latent, f_grid (shared), g[K], τ[K], taps[15], n_g,
  n_τ, gain }` as JSON + a tiny offline render CLI so a patch can be auditioned
  without the trainer. Decoder + latents checkpoint separately — the decoder
  IS the product; patches are samples from it.
- Demo target (the productizable moment): a CLI that takes a directory of
  snares in and gives back (a) the latent instrument, (b) per-snare patches,
  (c) a `morph a.wav b.wav 0.35` command. eseq integration (decoder on knob
  move → param block over the lock-free queue) is out of scope here but the
  export format should be designed against it.

### M4 (optional, gated on M3 residuals) — expressiveness upgrades

Only build what a failed listen names:
- Time-varying noise filter (F frames of taps, `sampleRow` upsampling — the
  DDSPE2E machinery) if static taps can't do the wire "shhh→tick" evolution.
- Bounded per-mode detune (the flag from §Synthesis) if renders sound
  quantized/choired vs targets.
- `learnedReverb` tail if room ambience in the corpus audibly separates
  renders from targets (R5 identifiability caveat applies: on short transients
  the IR is *better* identified than on quasi-stationary tones, so this may
  work well here).
- Velocity conditioning as a decoder input (NOT into z) if the corpus has
  multi-velocity samples of the same drum.

## Data prep

- Corpus: snares only — narrow and coherent per the direction doc. One-shots.
- Resample 44.1 kHz mono. Trim to onset (first sample exceeding −40 dBFS of
  peak, minus 32 samples of pre-roll), zero-pad/truncate to 0.75 s.
- Peak-normalize; record original peak as metadata (loudness is normalized
  OUT of the model — global gain param is fixed at 1.0 during training so
  gains and amp don't go redundant; per direction doc §6).
- Validity filter: reject non-one-shots (RMS re-rise after 300 ms → probably a
  loop), DC offset, clipped files.
- 30 Hz highpass on targets (kills the DC/sub-rumble that R3 flagged as a
  synth head-bias attractor; snares have no content there).
- Script: `Examples/ModalDrum/scripts/prepare_snares.sh`, patterned on
  `prepare_flute_multidynamics.sh` (dataset-wide decisions logged to a
  manifest JSON).

## Loss

- `spectralLossFFT`, windows [64, 128, 256, 512, 1024, 2048], hop = window/4,
  linear + log magnitude, `logMagnitudeEpsilon = 1e-3`. 64/128 windows matter
  more here than for the flute — the attack lives there; do not drop them.
- Aux loudness-envelope term (frame RMS, L1) at small weight — transient decay
  shape is exactly where lin-only spectral loss underweights, and the
  R0 open question about MSE-vs-spectral stability should be re-tested here on
  the post-scale-bug gradients (drums are the best case for a time-domain
  term). If it destabilizes, drop it and note it.
- CPU MR-STFT selection score for checkpoints, recalibrated on percussion
  (M1 gate).

## Known degeneracy & its guard

The one representational ambiguity this substrate retains: **high modes vs
noise** — a dense enough mode bank can fake the wire noise, and broadband
noise can fake unresolved high modes. Guard, from the start (not bolted on):

1. Hard prior split: noise τ ∈ [10 ms, 250 ms]; mode τ floor 5 ms but modes
   above 6 kHz get a *gain* budget via a mild L1 on `g_k` restricted to the
   top octave (λ ≈ 1e-3, tuned at M1's ablation).
2. M1's solo-branch listen gate is the detector: if either branch solo'd
   sounds like it's doing the other's job, raise the guard before M2.

This is deliberately the same move as the direction doc's L1-on-mod-matrix:
a canonicalization term that is simultaneously regularization and what makes
the export readable (a patch whose mode section is the shell and whose noise
section is the wires is editable; a smeared one isn't).

## Engineering notes / repo pitfalls that WILL bite otherwise

- Graph rebuild per epoch, params via `Tensor.param`, metrics captured before
  `zeroGrad()` — CLAUDE.md lifecycle rules.
- All `Tensor`s created AFTER `LazyGraphContext.reset()` (stale-nodeId
  aliasing; the fixed frequency grid tensor is the classic candidate).
- Constants that change across steps (schedule weights, if any) must be
  data-backed (`Tensor([w]).peek`) — the R0 stale-constant cache lesson; the
  fingerprint fix is in but data-backed avoids the AGX variant-limit wall.
- The additive bank should compile to frame-parallel kernels (no history
  cells in the modal branch at all — verify no accidental recurrence sneaks in
  via the time accumulator; a `statefulPhasor` at fixed freq is fine, it's the
  proven harmonic-bank path).
- Shuffled multi-sample training must use pinned-chunk selection
  (Trainer.swift fix from R6).
- 4096-window spectral remains Metal-blocked; window set above stays ≤2048.

## Where the code lives

`Examples/ModalDrum/` — new target, importing/copying DDSPE2E's `Losses`,
`BestMetric`, `Checkpoint`, `Dataset` patterns (same call as DDSP open
question 5: keep DDSPE2E frozen as baseline). The auto-decoder trainer goes in
its own file (`AutoDecoder.swift`) with nothing snare-specific in it — it is
the piece the subtractive track will import later.

## Explicit non-goals (v1)

- Toms/kicks/cymbals (one corpus, one model; the *method* generalizes, run it
  again).
- Learned mode frequencies end-to-end (grid + bounded detune only).
- Encoder network (auto-decoder has no amortization gap; latent-only descent
  is the round-trip mechanism).
- Realtime/eseq integration beyond designing the export format for it.
- MAP-Elites / unsupervised exploration (Track A skipped by decision
  2026-08-17).

## Sequencing vs DDSP revival

Independent lanes; no shared blocking state. Shared substrate risk is nil (the
modal branch uses only R0-proven ops). The open static-`tensorFFT`-adjoint bug
(`StaticTensorFrameGradProbeTests`) does not touch this ladder — no static
tensor feeds `tensorFFT` here; if M4 adds `learnedReverb` it uses the DFT-
matmul route that sidesteps it, same as R5.

## References

- Diaz et al., *Differentiable Modal Resonators*, ICASSP 2023 — read before
  M3; their parallel-vs-cascade finding is the one structural alternative to
  the flat bank worth knowing about.
- Park et al., *DeepSDF*, CVPR 2019 — auto-decoder formulation.
- Direction doc (2026-08-17 conversation) — Tracks B/C, failure table B9,
  shared components §9.

# Spec: DDSP revival — full paper implementation as a gated ladder

Status: PROPOSED — 2026-08-14

## Summary

Resurrect the long-standing goal of a paper-faithful DDSP (Engel et al., ICLR
2020) implementation in DGen. Not as a reboot of `Examples/DDSPE2E`, but as a
SynthID-style rung ladder where every rung has a target provably inside the
model class and a numeric exit gate, so a failure is attributable to a specific
component rather than "undertrained model vs. library bug".

The thesis: DDSPE2E (Feb 2026) did not stall for missing machinery — it stalled
on optimization dynamics and failure attribution, and both of those have been
addressed since. Several of its failure modes are now root-caused as actual
bugs or misconfigurations that were live during that campaign.

## Post-mortem of DDSPE2E (what we are not repeating)

`Examples/DDSPE2E` built and validated: batched `[B,K]` harmonic phasor bank
with Nyquist masking and harmonic renormalization, time-domain FIR filtered
noise, frame-to-sample control upsampling via `accum` + `sampleRow`, a full
transformer decoder (deliberate GRU substitute), multi-scale spectral loss with
log-magnitude, checkpointing, and a 3-stage orchestrator. Design docs:
`docs/DDSP_E2E_ROADMAP.md`, `docs/DDSP_PAPER_ALIGNMENT_PLAN.md`.

Documented stall reasons:

1. **Control collapse / plateau.** Framewise heads found cheap local minima
   (one dominant harmonic, flat gain). Spectral-only loss plateaued at ~10%
   reduction; the best single-chunk overfit probe improved 3.27e-4 → 2.35e-4
   (~28%), never reaching the roadmap's >50% gate.
2. **Non-monotonic training.** Loss rebounded after early minima; A/B/C
   staging and best-checkpoint selection were workarounds, not fixes.
3. **Attribution ambiguity** (the decisive one, per `Examples/SynthID/SPEC.md`):
   research-scale problem, failure ambiguous between library bug and
   undertrained model. SynthID was created explicitly to replace it.

## What changed since Feb 2026 (why the plateau is suspect)

Known defects and misconfigurations that were live during the DDSPE2E campaign:

- **Spectral gradient scale bug** — the (N·numBins)^1.5 scaling error was
  fixed after DDSPE2E; every LR tuned in that campaign was tuned against
  wrongly-scaled gradients.
- **`DGenSpectralConfig.logMagnitudeEpsilon` never set** — DDSPE2E trained
  against the 1e-8 default, which creates a large irreducible loss floor from
  empty bins. `DirectionTrainer` now uses 1e-3.
- **Phasor-frequency gradient corruption** — fixed during the Monologue work
  (trainable-phasor-freq grads, `2bae8eb`); tensor phasor/accum suffix
  adjoints now compose correctly with tensor-history BPTT (`cd3aa3b`), pinned
  by `TensorTemporalGradientCompositionTests`.

New capability that attacks the failure modes directly:

- **CMA-ES + tensor-lane batched evaluation** (`Sources/DGenLisp/CMAES.swift`,
  `docs/DGENLISP_CMA_ES_SPEC.md`): global basin search at ~ms/candidate versus
  DDSPE2E's single gradient trajectory per run. Control collapse is a basin
  problem.
- **Wall-clock** — hop-peel BlockFormation + Metal training kernels took the
  SVF benchmark from 42.9 s to under 1 s/epoch; the iteration loop that made
  the old campaign a multi-day slog is ~50x faster.
- **SynthID ladder methodology** — the library is now validated rung-by-rung
  on recoverable targets, so a DDSP failure would mean something.

## Capability inventory (verified 2026-08-14)

Differentiable and proven:

- Harmonic bank: tensor `statefulPhasor` with trainable frequencies
  (`Gradients.swift` temporalIncrementGradient), `.sum(axis:)` reductions,
  batched `[B,K]` lanes.
- Control upsampling: `sampleRow`/`peekRow` linear interpolation with
  dedicated grad-write/reduce ops; `Signal.mix` lerp.
- FFT path: `tensorFFT`/`tensorIFFT` (Cooley-Tukey from view+arithmetic
  primitives, generic autograd), `overlapAdd` with two-phase backward,
  `buffer(size:hop:)` grads, per-bin complex multiply. Forward FFT filtering
  proven in `ConvolutionReverbTests`.
- Multi-scale spectral loss: `spectralLossFFT` (+ batched `[B]` variant),
  linear + log/smooth-log magnitude, L1/L2, hop, summed over window sizes.
- NN layers: `matmul` (view-decomposed, differentiable; GEMMPass fuses forward
  and backward), broadcast bias, relu/sigmoid/tanh/softmax, userland
  LayerNorm + causal attention validated by `TransformerOpsTests`.
- Recurrence substrate: `TensorHistory` BPTT via grad-carry cells
  (`BPTTTests.testLearnIIRCoefficients`).

Forward-only (must not appear in trainable paths):

- `acceleratedFFT`/`acceleratedIFFT`, `partitionedSpectralConvolve`, `conv1d`,
  `conv2dSame`, `cumsum`, `gather`. (`conv2d` via the pool/asStrided path IS
  differentiable — it's what the old FIR noise branch used.)

## The ladder

Every rung: fixed seed, deterministic run, numeric gate, artifact trail
(loss CSV + preview WAV + checkpoint). A rung that fails its gate stops the
ladder and becomes a diagnosis task, not a tuning task.

### R0 — Re-run the DDSPE2E overfit probe under fixed gradients

Cheapest, most informative experiment. Re-run the single-chunk overfit probe
(same clip, same transformer decoder, same config) with only:

- current gradient code (post scale-bug, post phasor-freq fixes),
- `logMagnitudeEpsilon = 1e-3`,
- LR re-sweep (old LRs are invalid by construction).

Gate: >50% spectral loss reduction on the probe (the original M3b exit
criterion). Also record whether training is now monotone-ish without A/B/C
staging.

Interpretation: pass ⇒ the 2026-02 stall is fully explained by since-fixed
defects; proceed. Fail ⇒ the residual is model-class or optimization-shape;
diagnose before building anything new.

### R1 — Harmonic branch parameter recovery (synthetic, no network)

Ground-truth harmonic synth (K=32–64 harmonics, known per-frame amplitudes,
f0 track) rendered by DGen itself; recover the control tensors directly as
parameters (no decoder) from the audio alone via multi-scale spectral loss.
Use CMA-ES init + Adam polish for the f0-adjacent params.

Gate: recovered controls within tolerance; final loss within a small factor of
the render-noise floor. This isolates loss + gradients from network capacity.

### R2 — Frequency-sampled noise branch (the missing paper component)

Build the paper's `frequency_impulse_response` filtered noise:

1. Network/params emit `[F, nBins]` half-spectrum magnitudes per frame.
2. Convert to zero-phase (or linear-phase) IR via `tensorIFFT`; apply Hann
   window in time domain; back via `tensorFFT` (frequency-sampling method).
3. Multiply noise-frame spectra per-bin; `tensorIFFT` + `overlapAdd`.

All on the differentiable `tensorFFT` path. Validate first as a standalone
recovery task: known filter-magnitude trajectory, recover from filtered-noise
audio. Gate: magnitude-trajectory recovery + fdcheck on a small config.

Deliverable: a reusable `Synth.filteredNoiseFD(...)` alongside the existing
time-domain FIR branch, plus an A/B on the R0 probe clip.

### R3 — Decoder-driven harmonic+noise on one real clip

Full DDSP autoencoder minus z-encoder minus reverb: (f0, loudness) features →
decoder → harmonic bank + FD-filtered noise. Start with the existing
transformer decoder (paper deviation, already built and tested); single
sustained monophonic clip (TinySOL, as before).

Gate: >50% spectral loss reduction AND qualitative listen check (use the
`listen` skill) — the SynthID lesson: gate on sound, not parameters.

### R4 — Recurrent core (micro-GRU) [optional if R3 transformer suffices]

Implement a minimal gated recurrent cell on `TensorHistory` (the
`DDSP_PAPER_ALIGNMENT_PLAN.md` Stage 4 sketch). Validate on a toy sequence
task first (`BPTTTests` style), then swap into R3 and A/B against the
transformer at matched parameter count. This rung exists for paper
faithfulness; skip if the transformer matches or beats it.

### R5 — Learned reverb

Paper's learned-IR reverb, on the `tensorFFT` × IR-spectrum × `tensorIFFT` ×
`overlapAdd` route (`ConvolutionReverbTests` already proves the forward path).
Do NOT use `partitionedSpectralConvolve`/`acceleratedFFT` (forward-only).
Start with short IRs (≤0.5 s); gate on a known-IR recovery task before joining
the full model.

### R6 — Multi-clip training (small dataset)

Scale R3(+R4/R5) from single-clip overfit to a small TinySOL subset
(one instrument, ~10–20 clips). This is the first rung where "undertrained
model" is back on the table — but by now every component below it has a passed
gate, so residuals are attributable. Batched lanes can serve as data
parallelism here (B = clips) if wall-clock demands it.

Out-of-ladder / later: z-encoder (paper M5), CREPE-quality f0, timbre
transfer demos.

## Non-goals for v1

- z-encoder and timbre transfer (defer until R6 passes).
- Polyphony.
- Real-time inference.
- Matching the paper's exact dataset scale (NSynth); TinySOL subset suffices
  to claim the mechanism.
- GRU as a general library op (userland cell on `TensorHistory` is enough).

## Open questions

1. **Decoder capacity at R3** — reuse DDSPE2E's transformer config as-is, or
   re-sweep depth/width now that epochs are ~50x cheaper?
2. **CMA-ES scope** — CMA-ES is proven on synth params (20–100 dims), not
   network weights. For R3+, use it only for synth-side/init params and
   schedule scalars; network weights stay Adam-only. Confirm this split works
   or drop CMA-ES above R1.
3. **MSE aux loss** — DDSPE2E found even 0.1-weight MSE destabilizing next to
   spectral loss. Was that the scale bug? Re-test at R0; default to
   spectral + loudness-envelope only otherwise.
4. **Window set** — DDSPE2E used [64..1024]; DirectionTrainer uses
   [256..2048]. Paper uses [64..2048]. Pick per-rung based on f0 range and the
   CLAUDE.md resolution rule; 4096 remains Metal-blocked.
5. **Where the code lives** — new `Examples/DDSP/` reusing DDSPE2E modules as
   a library, vs. resurrecting DDSPE2E in place. Leaning new directory with
   imports, keeping DDSPE2E frozen as the baseline for R0 comparison.

## Pointers

- Prior attempt: `Examples/DDSPE2E/` (README is thorough), scoped-down
  successor `Examples/HarmonicE2E/`.
- Old design docs: `docs/DDSP_E2E_ROADMAP.md` (M0–M7),
  `docs/DDSP_PAPER_ALIGNMENT_PLAN.md` (paper gap analysis; Stage 4 has the
  micro-GRU sketch).
- Methodology template: `Examples/SynthID/SPEC.md` (rung ladder rationale).
- Search machinery: `docs/DGENLISP_CMA_ES_SPEC.md`,
  `Sources/DGenLisp/{CMAES,CMAESSearch,BatchMultistart,DirectionTrainer}.swift`.
- FFT-filtering forward proof: `Tests/DGenLazyTests/ConvolutionReverbTests.swift`.
- BPTT substrate: `Tests/DGenLazyTests/BPTTTests.swift`,
  `TensorTemporalGradientCompositionTests.swift`.

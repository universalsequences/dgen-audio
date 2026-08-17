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

**RESULT (2026-08-15/16): PASSED, decisively.**

- eps=1e-3, spectral-only probe: `7.63e-4 → 8.02e-5` = **89.5% reduction**,
  whole chain in ~15 min (the 2026-02 campaign peaked at ~28% over days).
- eps=1e-8 baseline, directly comparable to the historical best of
  `2.3486e-4`: **`9.948e-5`**, i.e. 2.36× better on the identical metric with
  no schedule retuning. The gradient fixes alone account for the old stall.
- Perceptual: the trained render is indistinguishable from the target by ear —
  the first perceptually convincing DDSP resynthesis in this repo. Renders and
  target are in `runs/listen_r0/`.
- Loss-to-perception calibration on this clip (eps=1e-3): ~3.6e-4 sounds like
  noise, ~8e-5 sounds like the target. The knee is steep; small absolute
  differences are large perceptual ones.

Two systemic bugs were found and fixed while running R0, both of which had
silently corrupted the 2026-02 campaign:

1. **Stale-constant compile cache** (`Sources/DGenLazy/Realize.swift`): the
   compile fingerprint keyed only on node/tensor counts, so a rebuilt graph
   that changed only a constant reused stale kernels with the old literal
   baked in. Every scheduled weight ramp in the old campaign was therefore
   fictional (stage A's "ramp to 0.05" actually trained at 0.00125 forever).
   Fixed by hashing constant values into the fingerprint; pinned by
   `Tests/DGenLazyTests/CompilationCacheTests.swift`. Making ramps real then
   hit the AGX driver's compiled-variants limit, so scheduled weights are now
   data-backed (`Tensor([w]).peek(...)`) instead of baked literals, and the
   runtime cache is FIFO-bounded.
2. **Ramp-blind best-checkpoint selection**: `model_best` was chosen by
   minimum *combined scheduled* loss, which is not comparable across steps
   while weights ramp. It selected step 10 of 500 and the pipeline then
   polished a near-random model to noise while discarding the excellent one.
   Fixed with a schedule-independent CPU MR-STFT scorer
   (`Examples/DDSPE2E/BestMetric.swift`, `--best-metric spectral`, default).
   Its ranking matches listening tests: exact-sounding 0.67, noise 2.98,
   near-random 4.89. Not yet ported to the batched path (logs a warning).

R1 is **folded** — R0 passed perceptually with the full decoder on real audio,
which strictly dominates R1's synthetic no-network recovery. R1 existed as a
fallback diagnostic for an ambiguous R0 failure that did not occur.

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

**Piece 1 (operator + gate): DONE 2026-08-16.**
`Examples/DDSPE2E/NoiseFD.swift` implements the full paper path — half-spectrum
magnitudes → constant mirror matmul → zero-phase IR via `tensorIFFT` → wrapped
Hann IR window (bounds IR length, which is what prevents circular-convolution
time aliasing) → `tensorFFT` → per-bin complex multiply against the windowed
noise frame → `tensorIFFT` → `overlapAdd`.

Gates, all passing:
- `Tests/DGenLazyTests/FilteredNoiseFDTests.swift` — static per-bin magnitude
  recovery through `tensorFFT`/`tensorIFFT`; loss collapses >100×, every bin
  within 0.05. Confirms the differentiable FFT path carries gradients.
- `Tests/DGenLazyTests/FilteredNoiseFDTrajectoryTests.swift` — (a) forward
  check that a lowpass response passes far less first-difference energy than a
  highpass one; (b) learning a per-frame magnitude *trajectory* from audio
  error alone: `0.080 → 0.018` (77.5%), monotone.

Two findings worth carrying into piece 2:
- The output is **linear in the magnitude parameters**, so waveform MSE is
  convex but ill-conditioned: a fixed step overshoots near the optimum and the
  loss drifts back up. Mild LR decay fixes it; expect the same sensitivity when
  the decoder drives these controls.
- A brickwall target is **not representable** by a bounded-length windowed IR,
  and trying to fit one produces a plateau that looks like an optimizer
  failure. Targets and expectations must respect the IR-length budget.

**Piece 2 (synth wiring + A/B): DONE 2026-08-16** (commit 9f1ac4e).
`--noise-filter-mode fir|fd` (plus `--noise-fd-{fft-size,hop,ir-length}`) wires
the frequency-sampled branch into the trainer, reusing the decoder's existing
sigmoid noise head sized to the bin count. FIR path verified bit-identical
after the refactor. The batched renderer has no fd path; training rejects
batch-size > 1 in fd mode. The operator was subsequently generalized into
`spectralFilter` (a438a7c): the magnitude→response chain is linear, so it
collapses to one constant `[nBins, fftSize]` CPU matrix — exact, and it
removes both per-hop transforms.

**RESULT (2026-08-16): R2 CLOSED.** The paper's frequency-sampled method was
faithfully built, gated, and measured — and found **not to be the better
engineering choice at this scale**. FIR stays the default; FD stays available
behind the flag.

Fair A/B on the R0 overfit clip (post hop-gated-gradient fixes 1572b8d +
f9b5af4; the pre-fix comparison was invalid — FD's hop=32 path had corrupted
adjoints while FIR's hop=1 path did not):

| | probe loss | selection score |
|---|---|---|
| FIR (baseline) | 8.023e-5 | 0.660 |
| FD, correct gradient | 8.632e-5 | 0.677 |
| FD, broken gradient | 9.080e-5 | 0.759 |

Fixing the gradient bought FD ~5% (about half its deficit), confirming the bug
was a real handicap — but on a correct gradient FD still trails the 15-tap
time-domain FIR by ~7.6% at ~2.5x the per-step compute.

Flute A/B (breathy material, FD's best-case): partial — the FD chain was
stopped twice for machine contention. At the matched stage-A step 499, FIR
selection score 0.746 vs FD 1.623 (2.2x worse), FD stage B essentially flat
through step 190. Not a clean verdict (FD ran FIR-tuned LRs and a loud
near-allpass sigmoid-0.5 init it must first suppress), but consistent with the
overfit-clip result.

This answers the February roadmap's question ("graduate to option 2 if
frequency resolution is insufficient"): frequency resolution is **not**
insufficient at this scale. If FD deserves a fairer shake later, the one
experiment worth running is its own init and LR — in particular a negative
`b_filter` bias so the noise branch starts quiet — a five-minute change and
one chain. The natural hybrid follow-up: frequency-domain *parameterization*
with time-domain *rendering*, keeping the paper's insight about what the
network should control without convolving in the frequency domain.

### R3 — Decoder-driven harmonic+noise on one real clip

Full DDSP autoencoder minus z-encoder minus reverb: (f0, loudness) features →
decoder → harmonic bank + FD-filtered noise. Start with the existing
transformer decoder (paper deviation, already built and tested); single
sustained monophonic clip (TinySOL, as before).

Gate: >50% spectral loss reduction AND qualitative listen check (use the
`listen` skill) — the SynthID lesson: gate on sound, not parameters.

**RESULT (2026-08-17): PASSED.** The banked `flute_fir` A/B/C chain already
satisfied R3's definition — (f0, loudness) → transformer decoder (2 layers,
d_model 64) → 64-harmonic bank + 15-tap FIR noise, real flute clip
(`.ddsp_cache_flute`, chunk_00000000), spectral loss only:

- Numeric gate: `6.251e-4 → 8.787e-5` = **85.9% reduction** (≫50%).
- Listen gate: best checkpoint rendered
  (`runs/listen_r3_flute/`) and A/B'd against the target — **confirmed by ear
  "sounds pretty much the same."** Analysis agrees: fundamental exact at
  369.1 Hz, all harmonics on-grid, every band 0–8 kHz within ±0.5 dB, RMS
  envelope tracks segment-by-segment including the breath dip, harmonic
  energy fraction 0.993 vs target 0.999.

Two non-blocking observations: the render carries a +0.018 DC offset
(target +0.0003) — the synth emits a small 0 Hz component worth a highpass or
head-bias check eventually; and the target's vibrato/breath sidebands (split
peaks at 740/758 Hz) come out as single peaks — the classic DDSP smoothing
signature, slightly more static than the real flute.

The transformer suffices ⇒ **R4 is skippable** per its own criterion.

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

**RESULT (2026-08-17): PASSED.** Dataset: `.ddsp_cache_flute_r6` — 18 TinySOL
flute ordinario clips (C4–B5 at mf, plus pp/ff at C4 and G5), 216 chunks,
194 train / 22 val. Chain: the R3 recipe with `SHUFFLE=true FIXED_BATCH=false`
(new script env overrides), 2000/800/800 steps, batch 1, eps=1e-3, FIR noise.

- Numeric: pinned-eval-chunk selection score `2.86 → 1.09` = **62% reduction**.
- Val (all 22 chunks, CPU MR-STFT): mean **1.452**, worst 1.838 — on the
  calibrated scale where "exact" ≈ 0.67 (R3 overfit 0.736) and "noise" ≈ 2.98.
- Listen gate (best/typical/worst val pairs, `runs/listen_r6/`): **passed by
  ear** — best (F5) "pretty spot on"; typical (D5) "a tiny bit brighter than
  the original"; worst pair's *target* (G5 ff) itself has wild pitch
  fluctuations, i.e. the hardest material is hard for a reason.

Two systemic fixes landed while running R6, both worth keeping in mind:

1. **Shuffle-blind best-checkpoint selection** (Trainer.swift): the spectral
   scorer evaluated the *current* training chunk, which changes every step
   under shuffle — scores were incomparable across steps and selection would
   freeze on whichever chunk scored easiest. Selection is now pinned to one
   fixed chunk (first entry of the split) whenever training is shuffled
   multi-chunk. Same disease as the ramp-blind selection bug from R0, new
   vector.
2. **f0 clamped to 500 Hz** (Synth.swift, non-batched render path): every
   note above 500 Hz trained *and* rendered at exactly 500 Hz — the first R6
   run's val scores split cleanly at the octave-4/5 boundary (2.2–2.6 ≈ noise
   for octave 5). Now clamps to Nyquist; harmonic aliasing was already
   handled by the downstream Nyquist mask. This clamp was live through R0–R3
   but never bit because that material sat below 500 Hz.

Known residuals (not gate-blocking): slight brightness excess on typical
val material; per-note vibrato/breath micro-structure smoothed (the classic
DDSP signature, also seen at R3).

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

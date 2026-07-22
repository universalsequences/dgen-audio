# Monologue Bass — first real-world subtractive target (STATUS)

Target: `Assets/monologue-bass.wav`. Goal: recover a `subtractive-bass`
patch (SUBTRACTIVE_SPEC.md topology) that recreates the sample, judged by
the independent CPU MR-STFT gate (≥80% vs deterministic midpoint+pitch-fit
baseline, absolute distance always reported) and by ear (`ab.wav`).

This is the first non-self-inversion target for the subtractive voice. The
frozen pipeline from the E-series applies: v2 stratified basin search →
`batch-refine --mode polish --schedule s4` (LR ×2.0, 150 smooth + 500
production steps, cosine per phase, 6 restarts/elite jitter 0.05, B≥32) →
`scripts/refine_rung3.py` coordinate descent on the CPU metric.

## Phase 1 — measurements (2026-07-21)

Script: `scripts/analysis/analyze_monologue_bass.py` → JSON at
`output/monologue_bass/phase1_measurements.json`.

### Housekeeping / provenance

- 44.1 kHz stereo Int16, 38,072 frames (0.863 s). Channels are duplicates
  (correlation 0.99999998, max |L−R| = 2 LSB) → mono-average is lossless.
- Peak −5.49 dBFS, 0 clipped samples, DC 5.4e-6. Sub-30 Hz energy fraction
  6.4e-5 (no junk).
- **Provenance check passed**: spectrum is flat through 20.5–22.05 kHz
  (−50.8 dB mean, no codec shelf); a 320 kbps MP3 decode would cut ~20.5 kHz.
  The sibling `monologue-bass.mp3` is an *export of* this wav, not its
  source. Treated as a lossless capture (Ableton `.asd` sidecars present);
  instrument is a Korg Monologue per the filename (see
  `MONOLOGUE_FEASIBILITY.md`).
- Noise floor: tail RMS −84.5 dBFS vs Int16 quantization −90.3 dB —
  effectively dead; no capture-floor restart or noise source needed.

### Pitch

- **f0 = 35.33 Hz, dead steady** (harmonic-score track over 100 ms windows:
  35.33 Hz on every frame 0.05–0.53 s, std 0.05 Hz; last frame 34.99 as the
  release fades). ≈ C#1+34c / 10 Hz below D1. No glide, no vibrato → the
  pitch trio collapses to one frozen f0, as the subtractive voice assumes.
- Harmonic peaks from a 2^18 FFT of the sustain sit at 35.33, 70.32,
  106.32, 141.31, 176.98, 211.80, 247.63 Hz … — n·f0 with ±0.3 Hz pulling,
  consistent with unresolved two-oscillator doublets (see beating).

### Amplitude envelope

- Attack: −3 dB point at ~15 ms, envelope peak at 15 ms. Fast but not
  clicky.
- **Decay-only, no sustain plateau**: 30 ms-window RMS falls ≈ linearly in
  dB from −8 dB (0.05 s) to −25 dB (0.4 s) ≈ −42 dB/s → single exponential
  with τ ≈ 0.20 s (VCA decayTime ≈ 0.2 s, sustain ≈ 0).
- **Note-off ≈ 0.45–0.50 s**: the decay steepens sharply (−39 dB avg in
  0.45–0.55 s → −72 dB by 0.55–0.65 s); release τ ≈ 30–40 ms.
- Effective length 0.570 s (last −60 dB-rel-peak crossing). Rest of the
  file is silence.

### Harmonic structure / oscillator

- **Odd-dominant**: late even-to-odd energy ratio −7.4 dB; H2 sits
  −13…−16 dB below H1 while H3 ≈ H1. Square/pulse-dominant oscillator mix,
  pw near (not exactly) 0.5 — H2's presence and its deep beat null indicate
  a second, detuned source contributing even/odd content.
- **Beating is real and strong**: per-harmonic tracks are non-monotonic
  with deep dip-and-recover nulls — H2 dips 22 dB (null at 0.13 s), H7
  21 dB (null 0.09 s), H4 11 dB (null 0.03 s), plus 3–10 dB wiggles on
  H6/H8/H10–H16. This is two-VCO beating (order ~0.5–1 Hz detune at the
  fundamental, i.e. tens of cents); nulls do not fit a single clean n·Δf
  ladder, consistent with the two VCOs carrying different waveform mixes.
  The single-oscillator v1 topology cannot express this; it is declared
  expected residual for v1 and a candidate zero-default detune pair for v2,
  to be probed against the CPU metric in NumPy first (playbook Phase 5).

### Filter

- Spectral centroid falls 580 Hz (onset) → 250 Hz (0.35 s) with excess-
  over-floor decaying ~e-fold per 0.15 s → **filter EG confirmed**:
  fBase ≈ 250–350 Hz territory, fAmt several hundred Hz, fDecay ≈ 0.15 s.
  High harmonics decay faster than low ones early (H11 −83 dB/s vs H1
  −37 dB/s over 0.14–0.32 s) — classic closing lowpass.
- No visible resonance ridge in the sustain spectrum; res likely low-mid.

## Decisions locked from Phase 1 (playbook bass section)

1. **30 Hz comparator high-pass: KEEP.** `compare.py`'s cosine transition
   spans 24–36 Hz; at f0 = 35.33 Hz the response is 0.992 (−0.07 dB).
   Sub-30 Hz target energy is 6e-5 of total. Changing the metric buys
   nothing and risks a three-script mirror bug.
2. **Envelope shape: decay-only ADSR is sufficient.** The existing
   `sustain + (1−sustain)·exp(−t/decayTime)` VCA fits (expect sustain → ~0,
   decayTime ≈ 0.2 s). No new envelope capacity needed.
3. **Crop: fit at 26,624 frames (0.604 s)** = effective length 0.570 s
   rounded up to a multiple of the 2048 window. Captures attack + full
   decay + release; never fits digital silence.
4. **f0 = frozen scalar from the CPU pitch fit** (playbook: steady bass →
   pitch trio collapses). `noteOff` becomes a **fixed documented scalar
   ≈ 0.47 s** from measurement (currently hardcoded 0.6 in three renderers
   — must be threaded, see Phase 3 notes). Trainable surface stays the
   validated 12 params.
5. **Windows: keep 256/512/1024/2048 + full-band coverage.** 4096 is
   Metal-blocked (threadgroup memory); training at reduced sample rate is
   forbidden (S5r reopened the drive degeneracy). f0 is frozen from the
   CPU fit so fundamental-resolution pressure on the loss is low.
6. **Detune/beating: measured and real, but NOT in v1.** Single-osc fit
   first; residual grid + NumPy probe decides whether a symmetric detune
   pair earns its scalars (SUBTRACTIVE_SPEC zero-default rule).

## Phase 3 — wiring + validation gates (2026-07-21)

Implemented (all legacy-inert; `subF0`/`subNoteOff` frozen documented
scalars, defaults 110 Hz / 0.6 s preserve every existing artifact):

- `Params.swift`: `PatchValues.subF0/.subNoteOff` (+Codable, dictionary,
  subscript); `SubtractiveBassVoiceSignals.f0/.noteOff` frozen constants.
- `Patch.swift` `SubtractiveBassVoice`: f0/noteOff from params (was 110/0.6
  hardcoded). Same in `BatchRefine.buildStudent` and `BatchBench.buildAudio`
  (threaded from base/elite `PatchValues`); `BasinSearch` passes its
  `--base-params` values.
- `render_reference.py`: `params.get("subF0", 110.0)` /
  `params.get("subNoteOff", 0.6)`.
- `refine_rung3.py`: `BOUNDS_SUBTRACTIVE_BASS` (mirrors spec table, new
  `log1p` mode for fAmt) + subtractive coordinate order (filter → VCA →
  tone/output, 8 passes × 19 steps). `score_params.py`: profile choice.

Gates:

- **Renderer equivalence**: legacy subtractive params (no subF0 keys)
  → 110 Hz render, max abs err vs NumPy 3.9e-5; new-style (subF0 35.33)
  → 35.33 Hz render, 5.4e-5; 808 midpoint 3.5e-6 (all ≤1e-3 gate).
- **fdcheck, envelope-driven cutoff at target config** (26,624 frames,
  f0 35.37, smooth log-L2 probe, representative sweep point fAmt=800
  fDecay=0.15): fAmt 4.0e-4 (eps 3e-3), fDecay 1.2e-4–3.5e-3 (all eps),
  res 2.6e-3–7.2e-3, fBase 9.6e-3 at eps 3e-3 (FD scatters ±2–6%
  symmetrically around a stable autograd at other eps — float32
  loss-readback noise on a 17×-smaller gradient, FDCHECK_FINDING issue 3;
  E0's protocol previously passed fBase). **PASS.**

Target preparation: `rung3 --prepare-only --frames 26624` → onset crop
4.58 ms, 26,624 frames, no resample/normalization (peak 0.53).
CPU pitch fit (`scripts/analysis/prepare_subtractive_initial.py`):
**f0 = 35.3678 Hz** (magnitude-weighted LS over H1–H8 peak positions),
noteOff = 0.50 s (release-logistic center vs body-decay extrapolation;
knee detector said 0.427 = onset of steepening, center chosen from the
−6 dB release-attenuation point; release τ ≈ 0.02 s in-bounds).

**Deterministic baseline** (spec midpoints + pitch fit, 30 Hz HP):
MR-STFT distance **0.418476**. The 80% gate requires learned ≤ 0.083695.

## Bug found during rung-2 (fixed)

`BatchRefine.scoreLanes` rendered lanes via `BatchBench.buildAudio` without
the `frozen:` base params, so every CPU selection score (pre-refine ranking,
final lane selection, reported per-elite results) rendered at the legacy
110 Hz subF0 while training ran correctly at the target f0. Invisible on
every E-series run (all at 110 Hz); on the first non-110 Hz target the
pre-refine scores came back 0.48–0.64 for elites basin-search had scored
at 0.079–0.23. Fixed by passing `frozen: base` (BatchRefine.swift:476);
the first rung-2 polish run was discarded and re-run.

## Synthetic bass rung-2 — PASS (2026-07-21)

Known params in the measured regime (shape 0.8, pw 0.45, fBase 250,
fAmt 900, fDecay 0.15, res 1.3, decay-only VCA, subF0 35.3678,
subNoteOff 0.5) rendered at 26,624 frames and recovered with the frozen
pipeline (basin-search defaults → `batch-refine --mode polish --schedule
s4`, B=72): global best lane CPU MR-STFT **0.00094**, **12/12 params
within spec tolerance**, most to 3–4 digits (pw 0.005%, fAmt 0.078%,
fBase 0.175%); only the known sustain/decayTime quasi-degenerate trade
visible (4.3%/2.5%). Population mechanism again did real work (winning
elite-09: unjittered 0.0164 → jittered lane 0.00094). Artifacts:
`output/monologue_bass/rung2/`.

## Real-sample fit — result (2026-07-21)

Frozen pipeline, no per-target tuning: basin-search (defaults, best elite
0.2576) → S4 polish (B=72, best lane 0.21856, elite-10) →
`refine_rung3.py --profile subtractive-bass` (8 passes → 0.208678).

**Gate: FAIL.** Improvement **50.13%** vs the required 80%
(deterministic midpoint+pitch-fit baseline 0.418476 → learned absolute
MR-STFT **0.208678**). Artifacts: `output/monologue_bass/real/`
(`refined_params.json`, `learned.wav`, `ab.wav`, `compare.json/png`).
NumPy↔Swift parity on the winner: 2.6e-3 max abs (above the 1e-3 gate,
localized to the first 200 ms — float32 divergence of the two biquad
implementations ringing at res=6; drops to 2e-5 by the tail).

Recovered patch (compensated — see diagnosis): shape 0.34, pw 0.62,
fBase 173.3, fAmt 3.4 (filter EG abandoned), fDecay 0.049, **res 6.0
(bound)**, attack 0.001, decayTime 0.134, sustain 0, release 0.011,
**drive 8.0 (bound)**, outGain 0.469, subF0 35.3678, subNoteOff 0.5.

### Diagnosis (playbook Phase 5, all probed on the CPU metric in NumPy)

Residual grid (2048-window): ~50% of remaining distance sits in
**500–3000 Hz × 0–0.3 s** — the upper-harmonic region while the filter is
open, where the target's two-VCO doublets (2nΔ spacing, resolved above
n≈16) and analog waveform texture live. Spectral overlay: learned
*undershoots* 345–1800 Hz by 2–17 dB even at res/drive bounds; the
res=6+drive=8 pinning is the voice fabricating brightness it cannot
otherwise produce (classic pinned-param compensation).

Probes (all starting points and bounds documented in
`scripts/analysis/probe_monologue_residual.py` + session notes):

| Probe | Result | Verdict |
|---|---|---|
| Symmetric detune pair, refit from winner (Δ 0.2–1.0 Hz) | 0.233–0.242 (worse than 0.2087) | rejected |
| Detune from measurement-informed sane start, 4 passes | Δ=0: 0.2369 / Δ=0.65: 0.2392 | no gain |
| Widen drive bound to 30 | drive stays 8.0, 0.2086 | not bound-limited |
| Harmonic correction bank (H2–40 × sin/cos × decay {0,25}/s, 93 nonzero coeffs) | 0.1903 (54.5%) | +4.4 pts only — residual is not coherent-series level error |
| True second VCO (independent shape2/pw2/level/detune) | 0.2084, vco2Level→0.02 | optimizer turns VCO2 off |

Conclusion: **~0.208 (50%) is a robust capacity ceiling of the 12-scalar
single-osc subtractive topology on this target.** The measured beating,
while physically real, does not move the phase-blind MR-STFT metric; the
deficit is analog waveform/drive texture across harmonics 10–80 that
neither a steady harmonic bank nor a second polyblep VCO expresses. This
is SUBTRACTIVE_SPEC E3's "Partial (40–60%): validated as
direction-finding" tier, consistent with the E3-closure finding that
residuals live in perceptually-quasi-null / texture directions.

### Ear check (analysis; user A/B pending on `ab.wav`)

Same note/gesture (f0, fast attack, decay arc). Differences: learned is
darker (centroid 286 vs 481 Hz), more compressed (crest 2.4 vs 4.4),
carries a static ~175 Hz resonant emphasis instead of the target's
brighter hollow-square timbre, no beating movement, hangs ~10 dB hotter
into the release, and has a 0.031 DC offset (drive-8 softsign on a
pw 0.62 pulse; invisible to the 30 Hz-HP metric).

### Paths forward (user decision)

1. **Accept as direction-finding** (spec's partial tier): the patch sheet
   is a deployable Monologue-style starting point; gate-on-sound per the
   E3 closure recommendation.
2. **Additive fallback** (`hoodie-bass` profile, existing machinery):
   expected to land near the hoodie result (~70%) — still likely below
   80% on a real analog target.
3. **Hybrid capacity** (subtractive + pruned correction bank in-graph,
   909-style): NumPy probe caps the near-term gain at ~54–55%; full joint
   training might exceed the probe but the bank's marginal value here is
   measured small.

## v2 — circuit-modeling voice (`monologue-bass` profile, 2026-07-22)

User direction after A/B ("less hi-res version of the target"): model the
circuit rather than fall back to additive — multi-stage chain with
parametric nonlinearities between layers, backprop finds the coefficients.

**Topology** (20 trainable scalars + frozen subF0/subNoteOff; every stage
inert at defaults): VCO1 + VCO2 (vco2Level, vco2Detune — VCO2 phase is the
closed-form offset `wrap(phase1 − t·detune)`), mixer → asymmetric
polynomial saturator (`satGain, satBias, satA2, satA3, satA5`; DC-comped at
the bias operating point) → **ZDF SVF lowpass built from Signal.history()
primitives** (Cytomic form, per-sample `g = tan(π·fc/sr)`, envelope-driven
cutoff, softsign-saturated integrator states `filtSat`; filtSat=0 is the
exact linear SVF; mirrors the eseq deployment `svf` macro) → VCA → softsign
drive. NumPy mirror `render_monologue_bass`; equivalence 9e-8.

**Circuit probes first** (`scripts/analysis/probe_monologue_circuit.py`,
NumPy coordinate refits): V1 pre-filter drive 39%, V2 poly pre-sat 50.2%,
V3 sat sandwich 43%, V4 feedback sat 50.2% (filtSat→0). Read as
inconclusive-negative: coordinate descent stalls where the GPU pipeline
does not (rung-2 evidence: 0.24 vs 0.00094), so the in-graph test decides.

**Library bugs found**:

1. Hand-rolled history feedback with dangling writes silently truncates
   BPTT (documented Part-B condition) — the SVF uses pass-through writes
   (`v = (write_out + s)/2`, `v2 = s2 + g·v1`).
2. **A trainable `statefulPhasor` frequency corrupts gradients of
   UNRELATED params** (all ~10× too small — this had made every monologue
   fdcheck fail). Pinned reproducer:
   `SVFBPTTScratchTests.testFullVoiceManyTargetsGradientsWithTrainableDetune`
   (XCTExpectFailure). Voices must keep trainable frequency out of
   statefulPhasor inputs — hence the closed-form detune phase.

**fdcheck (target config, smooth log-L2, eps 3e-3)**: fBase 1.3e-3,
fAmt 3.1e-3, fDecay 1.1e-3, res 5.0e-3, filtSat 9.3e-4, satGain 5.9e-3,
satA3 6.0e-3, satA5 1.8e-2, drive/outGain/decayTime ~1-2e-3 — PASS.
satA2/satBias/vco2Level/shape fail the log-loss FD instrument but all pass
the well-conditioned MSE scratch harness
(`SVFBPTTScratchTests.testFullVoiceManyTargetsGradients`, all ≤0.5% except
pw ~10% = the documented E0 conditioning case) — adjoints verified healthy.

**Training path**: scalar rung3 restart trainer (batched lane machinery
for the SVF voice deferred — tensor-history BPTT unvalidated). Trainer
wiring: monologue pitch-fit branch (steady median, measured 35.3683 vs
true 35.3678 on the synthetic), `--sub-note-off` config override,
subtractive-style restart templates, optimizer groups (circuit shapers in
the slow oscillator group). refine_rung3.py `BOUNDS_MONOLOGUE_BASS` +
coordinate order; score_params profile added.

Shakedown (200 epochs, synthetic self-target): loss 3.64 → 2.06, params
converging (vco2Level 0.607 vs true 0.6, filtSat 1.25 vs 1.2).

**Search pipeline for the 20-param space** (single-restart Adam recovers
only 9/20, CPU 0.29; coordinate refine 0.19 — population search needed):
`scripts/analysis/basin_search_monologue.py` — basin-search v2's algorithm
with the candidate batch vectorized per sample in NumPy (6.7 ms/candidate,
matches the canonical renderer to 4.5e-8). Synthetic best elite 0.068.

**THIRD (real) bug — `TrainableKickParams.naturalValues()` dropped
non-spec frozen scalars**: it roundtripped through
`frozenNaturalValues.dictionary`, which contains spec-table params only,
so every checkpoint/recovered JSON silently reset subF0 → 110 Hz and
subNoteOff → 0.6. Training was correct in-graph the whole time; only the
saved patches re-rendered at the wrong fundamental. This masqueraded as
"GPU loss anticorrelated with CPU metric" for three polish rounds (an
offline-replica investigation falsely indicted the production log-L1 op —
tensor-level tests later showed both GPU losses rank correctly; that
detour is retracted). Fixed in `naturalValues()`; old checkpoints are
salvageable by re-attaching subF0/subNoteOff.

**Synthetic rung-2 (monologue voice): PASS on sound.** Pipeline = NumPy
vectorized basin search → GPU Adam polish (600 epochs, `--no-linear-mag`)
→ refine_rung3 coordinate descent: **94.01% improvement, absolute CPU
MR-STFT 0.0131** on the synthetic self-target. Param recovery: identifying
params essentially exact (vco2Detune 0.00%, vco2Level 0.3%, shape 0.7%,
pw 0.2%, decayTime 6%); the saturator polynomial family
(satGain/satBias/satA2/A3/A5) and filter-character params
(fBase/res/filtSat, drive·outGain) trade along compensating ridges —
sound-identical, param-different, the E3 quasi-degeneracy class amplified
by the deliberately overcomplete shaper. Gate on sound per the E3-closure
recommendation.

## v2 real-target result (2026-07-22)

Pipeline (fixed): NumPy vectorized basin search (best elite 0.259) → GPU
Adam polish, 4 elites × 600 epochs `--no-linear-mag` (winner elite-11
0.2297) → refine_rung3 coordinate descent → **absolute CPU MR-STFT
0.2176** (52.51% vs the v2 deterministic baseline 0.4597; v1 comparison is
by absolute distance: v1 = 0.2087). Artifacts: `real_v2_refined.json`,
`real_v2_learned.wav`, `real_v2_ab.wav`; NumPy↔Swift parity 2.4e-7 (the
SVF avoids v1's high-Q biquad float32 divergence).

**Metric-tied with v1, but a fundamentally more physical patch**: the
filter EG is restored and matches Phase 1 (fAmt 1113 Hz, fDecay 0.118 s vs
measured τ≈0.15 s), VCO2 fully engaged (level 1.41, detune 1.29 Hz → real
beating), heavy asymmetric pre-sat (satGain 3.49, satA2 0.87, satA3
−0.58), SVF feedback sat 1.70, drive relaxed to 2.92 (v1 pinned 8.0).
Perceptual stats: centroid 241 Hz (v1 223, target 348), crest 2.98 (v1
2.44, target 4.40). Remaining pins: res 6.0 + fBase 30 (floor); a res≤14
probe chases the bound for +0.09pp — perceptually-null ridge, bound kept.

Both topologies plateau at ~0.21 absolute. The unreached ~0.21 → ≤0.09
(80% gate) span lives in analog texture the phase-blind metric measures
but neither forward model expresses (VCO waveform fine structure, drift,
capture chain). Levers not yet pulled: population-scale search for the
monologue voice (S4-style batched lanes — needs the tensor-history SVF
build), and gate-on-sound acceptance per the E3 closure.

## Plan of record

1. ~~Phase 1 measurement~~ (this section).
2. Phase 3 wiring: profile-threaded f0/noteOff (Patch.swift,
   BatchRefine.buildStudent, render_reference.py, refine_rung3.py bounds,
   score_params.py), fdcheck at target config, renderer equivalence.
3. **Synthetic bass rung-2 first**: render known params with f0 = 35.33 Hz
   / noteOff = 0.47 s at 26,624 frames, recover via the frozen pipeline
   (basin search → S4 polish). Gate: rung-1-style recovery quality on the
   sound (production-loss ratio ≪ baseline), params within tolerance except
   known quasi-degenerate directions (res/shape, fAmt·fDecay,
   sustain/decayTime — perceptually null, do not chase).
4. Real fit: CPU pitch fit → basin search → S4 polish → refine_rung3.py.
5. Gate report: % improvement vs deterministic midpoint+pitch-fit baseline
   + absolute learned MR-STFT distance + `ab.wav`.

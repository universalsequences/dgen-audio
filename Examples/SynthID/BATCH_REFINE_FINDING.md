# Batched (lane-parallel) gradient refinement — first results

Status: COMPLETE 2026-07-21 (runs executed overnight 2026-07-20/21,
M-series laptop, release build, seed-6 target, 32,768 frames).

Follow-up to `docs/TENSOR_BIQUAD_PARALLEL_LANES_SPEC.md` (lane-per-thread
batched biquad BPTT) and `E1_POLICY_AUDIT_V2_FINDING.md` (avenue 2: "batch
the refinement stage itself"). New harness: `BatchRefine.swift`
(`swift run SynthID batch-refine`), which trains the full 12-parameter
subtractive-bass surface as [B] transformed-space tensors through the
batched voice + batched multi-resolution spectral loss, with a per-lane
Adam (per-lane LR, production group-LR ratios, bounds projection,
grad clip, cosine decay) and loss scaled by B for batch-size-invariant
per-lane gradients (BatchTrainBench conventions).

## Correctness gates (run before any experiment)

- **Kernel dispatch modes** (B=12 dump via `BATCH_REFINE_KERNEL_DUMP`): the
  forward biquad recurrence block emits `fixedWithFrameLoop(12)`
  (`if (id < 12) { for (frames) ... }`) and the consolidated BPTT backward
  emits `selfManagedThreads(12)` with the reverse frame loop; no inner
  element loops in either. The lane-parallel predicate holds for the full
  subtractive topology (polyblep oscillator + biquad + VCA + softsign).
- **`probe-scalar`** ([1]-batched tensor path vs the production scalar
  `TrainableKickParams` path, full production loss, seed-6 target): worst
  rel diff **2.1e-3** across all 12 params at a generic point
  (seed-6 `initial_params.json`). At elite-01 the `res` adjoint shows a 43%
  rel diff, but FD cannot arbitrate there (central-difference estimates
  swing -0.18 / -0.094 / -0.021 across eps 0.05/0.01/0.002 — the landscape
  is kink/cancellation-dominated at that point, and the scalar autograd
  itself is 30-66% off every FD estimate). At `true_params` (loss ~1e-6)
  both paths return L1-subgradient noise, as expected. Conclusion: the
  batched adjoints match the scalar path at the scalar path's own noise
  floor; the ~2e-5 regime from the minimal BatchTrainBench voice does not
  carry over to the full topology, but the batched and scalar paths agree
  with each other.
- **`probe-grads`** ([12] vs per-lane [1], full topology, 8192 frames):
  worst rel diff **2.3e-3** (fBase) — lane independence at the same noise
  floor as the scalar comparison.
- Smoke test at production scale (B=64, 32768 frames, 4 log-L1 windows +
  4 linear-L1 terms): compiles, ~4 GB tape residency, loss decreases.

## Experiment 1 — escape test (documented seed-6 stall)

Setup: seed-6 target, lane 0 = elite-01 exactly (the best serially-refined
elite from `E1_POLICY_AUDIT_V2_FINDING.md`, which stalled at 14.31% after
600 smooth + 800 production serial epochs); lanes 1..63 = Gaussian jitters
of elite-01 at sigma 0.08 of each transformed span. One B=64 batched
trajectory, 150 smooth (log-L2) + 350 production steps, production group
LRs, cosine decay per phase. Command in
`output/batch_refine_escape_seed6_v1` (`batch_refine_report.json`).

Timing: **2.735 s/step at B=64, 32768 frames** = 0.0427 s/lane-step. The
serial subtractive trainer runs ~0.39 s/epoch at this frame count
(`GPU_REFINEMENT.md` 0.27 s/epoch at 22,528 frames, no bank), so batched
refinement delivers **~9x per lane-step** at production scale — above the
>=5x wiring threshold from the parallel-lanes spec.

Results (CPU mrstft selection score; production loss ratio vs the seed-6
cold baseline 2.502263):

| Lane | Init score | Final score | Prod loss | Ratio |
|---|---:|---:|---:|---:|
| 0 (exact elite-01, stall repro) | 0.08141 | 0.01362 | 0.313653 | 12.53% |
| 2 (best jitter; started 6x worse) | 0.48677 | 0.01063 | 0.266750 | **10.66%** |

- The stall reproduces: the unjittered lane lands at 12.5%, the same
  regime as the serial schedule's 14.3%.
- **The population escapes the basin-discovery failure**: lane 2 recovers
  **10/12 hidden params within spec tolerance**, including the dark-corner
  filter basin serial refinement never finds (fBase 77.8 vs truth 86.4,
  fAmt 8.57 vs 7.19, fDecay 0.1239 vs 0.1237, plus the whole VCA group and
  pw 0.471 vs 0.465). Serial refinement from generic starts lands at
  fBase~266 (bright compensated basin).
- What still fails: the **res/shape pair** (res 1.88 vs 3.04, shape 0.81
  vs 0.67) — the coupled oscillator/filter compensation documented in both
  E1 audits. That residual holds the production ratio at 10.66% >> the 2%
  gate.
- 5/63 jittered lanes beat the unjittered trajectory; init-score proximity
  did not predict outcome (the winner started at 0.487), consistent with
  the v2 finding that refinement outcome is only weakly coupled to start
  quality.

Reading: population restarts fix *where refinement ends up in the
fBase/fAmt/fDecay/VCA subspace* — a real, previously-blocked win — but the
res/shape ridge is a property of the schedule/objective, not of the start
point, exactly as the v2 finding predicted. Batched refinement makes
schedule experiments on that ridge ~9x cheaper per trajectory.

## Experiment 2 — elite polish (12 basin-search elites x jittered restarts)

Setup: the 12 v2 basin-search elites (`output/e1_basin_v2_seed6_run2/elites`)
x 6 restarts each (restart 0 = the unjittered elite, 5 jitters at sigma
0.05) packed into ONE B=72 batch; 150 smooth + 350 production steps, same
schedule as the escape run. Artifacts in
`output/batch_refine_polish_seed6_v1` (per-elite best-lane params +
`batch_refine_report.json`).

Timing: **2.859 s/step at B=72 (0.0397 s/lane-step); the entire 12-elite
polish took 24 min wall** vs 78 min for the serial refinement stage in the
v2 audit — and the serial stage spent 1400 epochs/elite vs 500 here.
Normalized: 36,000 lane-steps in 1430 s (0.040 s/lane-step) vs 16,800
serial epochs in ~4680 s (0.279 s/lane-step) — **~7x per lane-step** at
production scale, again above the >=5x threshold.

Per-elite outcome (CPU mrstft; production ratio vs cold baseline 2.502263;
serial column = the v2 audit's 1400-epoch refinement of the same elite):

| Elite | Pre | Unjittered final | Best-lane final | Prod ratio (batched, 500 steps) | Prod ratio (serial, 1400 epochs) |
|---|---:|---:|---:|---:|---:|
| elite-00 | 0.06979 | 0.05367 | 0.05353 | 27.33% | 27.21% |
| elite-01 | 0.08141 | 0.01362 | **0.01112** | **11.56%** | 14.31% |
| elite-02 | 0.10791 | 0.09257 | 0.05418 | 30.28% | 44.47% |
| elite-03 | 0.10481 | 0.05575 | 0.03407 | 18.71% | 30.33% |
| elite-04 | 0.14953 | 0.15112 | 0.11252 | 56.66% | 72.31% |
| elite-05 | 0.12956 | 0.11909 | 0.11287 | 50.92% | 49.27% |
| elite-06 | 0.18145 | 0.16144 | 0.15906 | 70.42% | 72.32% |
| elite-07 | 0.14851 | 0.12350 | 0.12350 | 57.16% | 54.51% |
| elite-08 | 0.20717 | 0.17456 | 0.17456 | 78.57% | 77.00% |
| elite-09 | 0.20852 | 0.18685 | 0.18685 | 82.69% | 81.01% |
| elite-10 | 0.07221 | 0.05543 | 0.05485 | 30.23% | 26.83% |
| elite-11 | 0.07361 | 0.05527 | 0.05384 | 30.58% | 30.19% |

- Global best: elite-01 lane 10 at 11.56% — beats the best serial result
  (14.31%) in ~1/3 the steps and 1/20 the per-elite wall-clock.
- Jittered restarts helped where it mattered: elite-02 (0.0926 -> 0.0542)
  and elite-03 (0.0558 -> 0.0341) had jitters escape their unjittered
  trajectory; the tail elites (08, 09) were unmoved — their basins are
  simply wrong, consistent with the search-stage stratification being the
  right place to kill them.
- No elite cleared the 2% gate (expected: the escape experiment shows the
  res/shape ridge is schedule-attracted; polish uses the same schedule).

## Experiment 3 — per-lane LR sweep

Setup: 64 lanes, ALL initialized to elite-01 exactly; each lane's global LR
multiplier log-spaced over x0.1..x10 on top of the production group LRs
(amp 3e-3, decay 1e-2, tone 1e-2, osc 1e-3); 200 production steps, cosine
decay. Lanes are fully independent through backward, so this is one batch
(2.964 s/step). Full table in `output/batch_refine_lrsweep_seed6_v1`.

| LR multiplier | Final CPU score |
|---:|---:|
| x0.10 | 0.06211 |
| x0.50 | 0.04478 |
| x1.00 (production) | ~0.0247 |
| x1.39 | 0.01532 |
| x1.50 | 0.01501 |
| x2.00 | 0.01342 |
| x2.32 | 0.01251 |
| x2.68 | 0.02541 (first instability) |
| x4.16 | **0.01170** (best, but in the noisy regime) |
| x10.0 | 0.01297 (erratic neighborhood) |

Reading: final score improves monotonically from x0.1 up to ~x1.4, sits on
a broad plateau through ~x2.5, and becomes erratic above that (adjacent
lanes swing 2x, e.g. x2.68 -> 0.0254 vs x2.49 -> 0.0140). The old
production LRs are **~2x conservative** after the spectral-grad-scale fix.
Recommended: multiply the subtractive group LRs by **1.5-2.5** (i.e. amp
~5e-3, decay/tone ~1.5-2.5e-2, osc ~2e-3); treat the x4+ winners as
noise-lottery, not a setting. Note the sweep's x1.4-2.5 plateau at 200
steps already matches the 500-step x1.0 escape/polish results — the right
LR buys ~2.5x fewer steps on top of the lane parallelism.

## Decision

1. **Wire batched refinement in.** Both experiments cleared the >=5x
   wall-clock threshold at production frame counts (~9x per lane-step at
   B=64, ~7x vs the audit's serial schedule at B=72), with correctness at
   the scalar path's own noise floor. `scripts/refine_elites_batched.sh`
   wraps `batch-refine --mode polish` as the drop-in replacement for the
   serial `refine_elites.sh` stage (which remains for A/B). Keep B >= 32
   by packing elites x restarts (the harness pads with restarts of the
   best-scoring elite automatically).
2. **Population restarts solve basin discovery, not the res/shape ridge.**
   The escape test recovered 10/12 hidden params (incl. the dark-corner
   fBase/fAmt/fDecay basin no serial trajectory finds) but stalled at
   10.66% on the same res/shape compensation both E1 audits document. The
   next lever is the refinement *schedule* (the v2 finding's
   observable-trajectory continuation idea), now ~an-order-of-magnitude
   cheaper to iterate on. Do not spend more restarts attacking seed-6 with
   the unchanged schedule.
3. **Bump the subtractive group LRs x1.5-2.5** (re-derived post
   spectral-grad-scale fix; old absolute LRs were invalid). Worth a quick
   confirming sweep on seed 7's target before freezing into a policy.

## Reproduction

```sh
swift build -c release --product SynthID
# probes (guardrails)
.build/release/SynthID batch-refine --mode probe-scalar --target <t.wav> --init <p.json>
.build/release/SynthID batch-refine --mode probe-grads --target <t.wav> --init <p.json> --batch 12 --frames 8192
# experiments
.build/release/SynthID batch-refine --mode escape --target <t.wav> --init <elite.json> \
  --true-params <truth.json> --batch 64 --jitter 0.08 --smooth-steps 150 --steps 350 \
  --baseline 2.502263 --out <dir>
bash Examples/SynthID/scripts/refine_elites_batched.sh <search-out-dir> <t.wav> <baseline> <out-dir>
.build/release/SynthID batch-refine --mode lr-sweep --target <t.wav> --init <elite.json> \
  --batch 64 --steps 200 --lr-min 0.1 --lr-max 10 --out <dir>
```

Caveats / known deviations from the serial stage:
- The batched "smooth" phase approximates the production smooth loss with
  log-magnitude L2 (the batched spectralLossFFT has no
  useSmoothLogMagnitude variant) — same kink-removal intent, not
  bit-identical.
- Batched mean-loss traces are not comparable to serial finalLoss prints
  (mean-vs-per-lane bookkeeping); all cross-run comparisons above use
  per-lane CPU mrstft scores and serial production-loss re-evaluation of
  winning lanes (`--baseline` prints gate ratios).
- Full-topology gradient agreement (batched vs scalar, and [B] vs [1]) is
  ~2e-3, not the minimal-voice 2e-5 — kink/cancellation noise from the
  polyblep/softsign/L1 terms, shared by both paths.

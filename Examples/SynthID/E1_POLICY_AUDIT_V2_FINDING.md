# SynthID E1 Optimizer-Policy Audit — v2 (Stratified Batched Basin Search) Finding

## Status: BLOCKED — v2 policy also fails the seed-6 gate

Executed 2026-07-20 after `docs/TENSOR_BIQUAD_SPEC.md` landed (batched,
filter-active `SignalTensor.biquad`), which unblocked running the v2 search
at its declared production scale. Frozen policy: `output/e1_basin_v2_seed6_run2/POLICY.md`.
E0/E2 remain PASS. This is the second attempt at breaking the compensation-
ridge basin-selection failure documented in `E1_POLICY_AUDIT_FINDING.md`
("v1" below); it does not clear the gate either, and the untouched seeds 7,
8, 9 remain sealed. E3 remains blocked.

## What changed from v1

v1 (`output/e1_basin_search_seed6/`) sampled candidates uniformly in *raw*
parameter units, single round, no stratification, and failed at 17.10%
best-elite ratio. v2 changed exactly two things (see `POLICY.md`), both
target-independent:

1. Uniform sampling in **transformed** coordinates over the full declared
   bounds (v1's raw-unit prior made the true low-`fAmt` basin a ~0.2% tail
   event).
2. Two rounds of **stratified Gaussian resampling** — 10 strata over
   `fBase` octave bands × `shape` halves, top-8 parents per stratum each
   spawning 96 children, sigma 0.08 then 0.04 of each transformed span —
   specifically to keep dark/coupled-basin corners alive against the
   dominant bright basin that global top-N ranking collapses into.

Refinement schedule (stage 3) is unchanged from the audit's best-known
schedule: 600 smooth-loss epochs from each elite, then 800 production
epochs from the smooth result, selection by production `finalLoss` only.

## Search stage: throughput and result

23,552 batched forward evaluations (8,192 uniform + 7,680 + 7,680 resampled)
completed in 320.1s at B=256, using the now-fixed filter-active tensor
biquad:

| Round | Evals | Time | ms/candidate | evals/hour |
|---|---:|---:|---:|---:|
| round 0 (uniform, transformed coords) | 8,192 | 108.9s | 13.29 | 270,810 |
| round 1 (σ=0.08 resample) | 7,680 | 104.6s | 13.62 | 264,321 |
| round 2 (σ=0.04 resample) | 7,680 | 106.6s | 13.88 | 259,362 |
| **total** | **23,552** | **320.1s** | **13.59** | **264,877** |

This matches `docs/TENSOR_BIQUAD_SPEC.md`'s B=256 micro-benchmark
(14.09 ms/candidate, 255,493 evals/hour) to within ~4% at 46x the candidate
count — the correctness/throughput claim generalizes to the real search
scale, not just the 8-candidate probe it was measured on.

The stratified selection worked as designed: 10 of 12 elites' pre-refine
scores (CPU `mrstft`, same objective as v1's ranking) improved monotonically
with `fBase`, and the best raw candidate overall (0.0698, `fBase=94.6 Hz`,
low `shape`) sits in the correct dark-corner neighborhood — closer to hidden
seed-6 truth (`fBase=86.44 Hz`) than anything v1's global-only ranking
surfaced.

## Refinement stage: all 12 elites FAIL

Gate: `finalLoss / coldBaseline <= 0.02`, cold baseline `2.502263`
(unchanged from the v1 audit).

| Elite | Stratum (fBase band / shape half) | Pre-refine score | Post-refine finalLoss | Ratio | Gate |
|---|---|---:|---:|---:|:---:|
| elite-01 | [60,120) / shape≥0.5 | 0.08141 | 0.357990 | 14.307% | FAIL |
| elite-10 | [960,2000) / shape≥0.5 (top-2 overall) | — | 0.671282 | 26.827% | FAIL |
| elite-00 | [60,120) / shape<0.5 | 0.06979 | 0.680764 | 27.206% | FAIL |
| elite-11 | [960,2000) / shape<0.5 (top-2 overall) | — | 0.755429 | 30.190% | FAIL |
| elite-03 | [120,240) / shape≥0.5 | 0.10481 | 0.758847 | 30.326% | FAIL |
| elite-02 | [120,240) / shape<0.5 | 0.10791 | 1.112783 | 44.471% | FAIL |
| elite-05 | [240,480) / shape≥0.5 | 0.12956 | 1.232924 | 49.272% | FAIL |
| elite-07 | [480,960) / shape≥0.5 | 0.14851 | 1.363929 | 54.508% | FAIL |
| elite-04 | [240,480) / shape<0.5 | 0.14953 | 1.809380 | 72.310% | FAIL |
| elite-06 | [480,960) / shape<0.5 | 0.18145 | 1.809627 | 72.320% | FAIL |
| elite-08 | [960,2000) / shape<0.5 | 0.20717 | 1.926648 | 76.996% | FAIL |
| elite-09 | [960,2000) / shape≥0.5 | 0.20852 | 2.027184 | 81.014% | FAIL |

Best result: **elite-01 at 14.307%**, marginally *worse* than v1's best
(13.09%, the population-search-plus-refinement candidate from the original
audit) despite starting from a better-diversified, better-scoring search.
The two elites seeded from the correct dark-corner stratum (`fBase<120`,
`elite-00`/`elite-01`) refined to 27.2% and 14.3% respectively — better than
the rest of the pool, but neither cleared 2%, and starting `fBase` proximity
to truth did not reliably predict refinement outcome (`elite-00` started
closer in raw score but refined worse than `elite-01`, whose starting
`shape=0.739` happened to be closer to hidden `shape=0.668` than `elite-00`'s
`shape=0.067`).

## Interpretation

This is a negative result for the search-diversity hypothesis specifically,
not a new root cause. The v1 audit already established that gradient
optimization locally recovers the correct basin when told the truth (exact
solution scores `9.023061e-7` through the same production path) but standard
Adam refinement pulls generic starting points toward the dominant
bright/compensated basin. v2 tested whether *feeding refinement a
better-diversified, better-scoring set of starting points* was enough to
survive that pull. It was not: even the elite seeded almost exactly in the
correct `fBase` octave still drifted to 14-27% after the identical 600+800
epoch schedule that produces >70% ratios from worse starting points. Search
quality and refinement outcome were only weakly correlated across the pool.

The practical conclusion sharpens the v1 audit's framing: this is not (or
not only) a basin-*discovery* problem that better sampling can solve. The
1400-epoch production+smooth Adam schedule itself has a strong basin of
attraction toward the compensated solution that swallows most starting
points regardless of how close they start, and stratified sampling's
information is largely lost by the time refinement finishes. The refinement
*schedule*, not the *initialization policy*, is now the more likely lever —
consistent with the v1 audit's untried recommendation of "a fixed
multi-resolution continuation that first fits observable summary
trajectories... before enabling the full waveform loss," which this
experiment did not test (refinement stage was deliberately left unchanged
to isolate the initialization-policy variable).

## Decision and next steps

v2 policy is rejected under the same protocol as v1: seeds 7, 8, 9 remain
sealed, E3 remains blocked. No further tuning of this policy against seed 6
or 7 is licensed by this result.

Two independent next avenues, not mutually exclusive:

1. **Change the refinement schedule**, not the search — e.g. the audit's
   suggested multi-resolution/observable-trajectory continuation, frozen and
   tested against seeds 6 and 7 only per protocol.
2. **Batch the refinement stage itself.** All 12 elites here were refined
   *serially* (one 1400-epoch Adam trajectory at a time, ~78 minutes total
   wall time vs. 5.3 minutes for the entire 23,552-candidate search) because
   `SignalTensor.biquad` backward passes are unimplemented and explicitly
   guarded against (`docs/TENSOR_BIQUAD_GRADIENT_SPEC.md`, written
   2026-07-20, work in progress on a separate track). Batched gradients
   would not by themselves fix the basin-attraction problem this finding
   documents, but they would make trying many more refinement-schedule
   variants (per avenue 1) or many more elites per search round cheap enough
   to iterate on quickly, the same way batched forward eval made the search
   stage itself cheap.

## Reproduction commands

```sh
swift build -c release --product SynthID

# Stage 1+2: stratified batched search (defaults match the frozen v2 policy)
.build/release/SynthID basin-search \
  --target output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  --out output/e1_basin_v2_seed6_run2 \
  --seed 6

# Stage 3: refine all 12 elites (audit's best-known schedule)
bash Examples/SynthID/scripts/refine_elites.sh \
  output/e1_basin_v2_seed6_run2 \
  output/e1_subtractive_fresh_seeds_6_7/seed-6/target.wav \
  6 \
  2.502263
```

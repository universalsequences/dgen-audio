# Schedule-continuation experiments — results (SCHEDULE_SPEC.md)

Status: COMPLETE 2026-07-21 (D0 + S4/S1/S3 matrix + S5 escalation, seed-6
target, one afternoon as budgeted; seed-7 confirmation run same day).
Artifacts: `output/schedule_spec_seed6/`, `output/schedule_spec_seed7/`.

## TL;DR

The res/shape ridge survives every schedule change, but **S4 (LR ×2.0,
production phase extended to 500 steps) cut the gate ratio from 10.66% to
7.70% and recovered 12/12 hidden params within spec tolerance** — the
first full-recovery lane on seed 6. The curriculum (S1) and homotopy (S3)
variants failed exactly as the D0 diagnostic predicted: the ridge is a
*coupled 12-D valley*, not a 2-D trap, so schedules that decouple res/shape
from the other params make things worse. The realizable S5 (hi-res
objective) failed for an unrelated, instructive reason (drive degeneracy,
below). **The user A/B'd S4's best lane against the target and could not
distinguish them** — the 2% gate is stricter than the perceptual bar on
this target.

## D0 — ridge cross-section (full notes: `output/schedule_spec_seed6/d0/D0_NOTES.md`)

Two 33×33 production-loss grids over res × shape centered on truth (~2 min
each):

1. **Others clamped to truth**: no barrier — the valley floor descends
   monotonically from the stall point to truth (0.635 → 0.517 → 0.484 →
   0.322 → 9e-7), and the cross-ridge gradient at the stall (dL/dshape
   0.90, dL/dz_res 0.016) is 450×/8× the 2e-3 noise floor → per the branch
   rule, the schedule matrix proceeded. The valley is a narrow rotated
   diagonal (floor slope d(shape)/d(z_res) ≈ −0.28) lined with pocket
   minima that require *coordinated* res+shape moves to escape.
2. **Others clamped to the escape winner's recovered values**: the slice's
   global minimum **moves to the ridge point** (0.337) and truth becomes
   3× worse (0.960). The tiny residuals in the 10 "recovered" params
   reprice the slice so the wrong res/shape is genuinely optimal there.
   The ridge is a property of the *joint* geometry: res/shape cannot be
   resolved against any fixed slightly-wrong background.

## Schedule matrix (escape-control semantics: elite-01 + 63 jitters σ0.08, B=64, seed 6)

| ID | Schedule | Best-lane gate ratio | Params recovered | Verdict |
|---|---|---:|---:|---|
| S0 | control (150 smooth + 350 prod) | 10.66% | 10/12 | baseline |
| S4 | LR ×2.0, prod extended to 500 (650 total) | **7.70%** | **12/12** | **winner** |
| S1 | freeze-ridge curriculum (A: freeze res/shape; B: freeze others; C: all free, LR×0.3) | 18.54% | 10/12 (res 1.12, worse than control) | rejected |
| S3 | homotopy: λ-blend smooth→prod over steps 100–300 | 11.66% | 9/12 | rejected |
| S5 | 4096-window log-mag term ×0.5 | — | — | **hardware-blocked** (needs 65,528 B threadgroup vs 32,768 Metal max) |
| S5r | S4 objective at 22,050 Hz (2048 window → 10.8 Hz bins, the S5 resolution goal) | 306% at 44.1k | 9/12 (drive→8.0 bound) | rejected |

- **S4**: best lane 9 — res 2.40 (truth 3.04), shape 0.749 (truth 0.668),
  both *within tolerance*; all other params essentially exact. The "were
  we just undertrained at too-low LR" null hypothesis carries most of the
  achievable gain: more LR + more production steps traverses about half
  the remaining valley. Audio: `s4/best_lane_09.wav` vs `s4/target_ref.wav`
  — perceptually indistinguishable per user A/B.
- **S1**: phase B (train res/shape against frozen others) drove res the
  *wrong way* (1.88 → 1.12), and the best lane was the unjittered one —
  the curriculum also neutralized the population advantage. Direct
  confirmation of D0 grid 2: with the background fixed, the slice minimum
  IS the ridge.
- **S3**: statistically indistinguishable from control. The step-150
  discontinuity was not what shoves lanes onto the ridge.
- **S5r**: scored well on its own 22 kHz objective (CPU 0.033) but 306% on
  the 44.1k gate. Winner: drive 8.0 (bound; truth 2.04), fAmt 17.3 (truth
  7.19), res 0.89. **Dropping the 11–22 kHz octave reopens the drive
  product degeneracy** — the top octave is what penalizes saturation
  harmonics and regularizes drive. Any future hi-res objective must *add*
  low-frequency resolution without *removing* high-frequency coverage
  (true 4096 window needs a chunked/half-precision Metal FFT —
  Sources/DGen work).

## Decision (per the spec's rule)

Partial → escalated to S5 → realizable S5 failed → the spec's fail-branch
question ("is the 2% gate the right bar?") was put to the ear: **best-lane
audio is indistinguishable from the target**. Conclusions:

1. **Freeze S4 as the new control schedule** (LR ×2.0 group multiplier,
   150 smooth + 500 production, cosine per phase). Implemented as
   `batch-refine --mode escape --schedule s4`.
2. The remaining 7.7%→2% gap is objective-identifiability at a level below
   perceptual relevance for this target. The honest gate for E3 is either
   (a) params-within-spec-tolerance (S4 achieves 12/12), or (b) a ratio
   bar calibrated against audible difference — not the current 2%.
3. Next protocol step: run S4 unmodified on seed 7 (needs its own
   basin-search elites + cold baseline) before any decision about seeds
   8/9. Only after that, revisit the gate definition with both seeds'
   evidence on the table.

## Seed-7 confirmation (S4 unmodified)

Full pipeline run cold on seed 7, no per-seed tuning: frozen v2 basin-search
policy (`--seed 7` only; `output/schedule_spec_seed7/basin`), then
`batch-refine --mode polish --schedule s4` over all 12 elites × 6 restarts
(B=72, jitter σ0.05), gate vs seed-7's own cold baseline **19.8281**
(baseline method validated by reproducing seed-6's 2.502263 to 6 digits).

**S4 clears seed 7 decisively: four elites under the 2% gate**, global best
elite-11 lane 71 at **0.1393%** (prod loss 0.027613); also elite-04 0.4475%,
elite-06 0.5542%, elite-10 0.6944%. Recovery is near-exact — res 1.1195 vs
truth 1.1201, shape 0.4229 exact, fBase 2345.9 vs 2347.8, pw 0.1821 vs
0.1820 — with only the fAmt/fDecay pair trading along the filter-envelope
direction (151/0.105 vs 188/0.082) at negligible loss. Jittered restarts
mattered: elite-11's unjittered lane stalled at CPU 0.136 while a jitter
reached 0.00148 (same pattern for elites 04 and 10).

Seed-7's truth sits in a bright low-resonance corner (fBase 2348, res 1.12)
with no res/shape ridge — supporting the reading that the seed-6 ridge is a
property of that target's parameter point (high-res dark corner), not of
the schedule or search method.

**Protocol status**: the spec's letter requires seed 6 ≤2% before the
seed-7 run counts as a pass, and seed 6 stands at 7.70% (perceptually
exact, 12/12 in tolerance). Seeds 8/9 therefore stay sealed pending a
decision on the gate definition — with both design seeds now in evidence:
seed 7 passes the 2% bar outright, seed 6 fails it only on a
perceptually-null identifiability ridge.

## Implementation notes

`BatchRefine.swift` gained: per-phase freeze masks (frozen params skip
Adam entirely and get fresh m/v/bias-correction clocks on unfreeze),
λ-blended smooth/production loss, per-phase LR multipliers, named
schedules (`--schedule s1|s3|s4|s5`), phase labels in trace + report.
`hiResTerm` carries the blocked S5 objective for when a big-window Metal
FFT exists. No Sources/DGen changes.

Repro commands: see `output/schedule_spec_seed6/*/run.log` headers; matrix
driver preserved the escape-control lane set (same rng seed → identical
jitters across variants).

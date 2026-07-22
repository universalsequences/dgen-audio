# SynthID Schedule-Continuation Experiments (mini spec)

## Status: PROPOSED — next phase after batched refinement landed

Successor to the two rejected initialization policies
(`E1_POLICY_AUDIT_FINDING.md`, `E1_POLICY_AUDIT_V2_FINDING.md`) and the
batched-refinement wiring (`BATCH_REFINE_FINDING.md`). Target: the last
seed-6 blocker — the **res/shape compensation ridge**.

## Problem statement

The escape test settled the diagnosis. From 64 jittered restarts, the best
lane recovers 10/12 hidden params including the dark-corner
fBase/fAmt/fDecay basin, then **every lane** converges to the same wrong
point on the res/shape ridge (res 1.88 vs truth 3.04, shape 0.81 vs 0.67),
holding the production ratio at 10.66% vs the 2% gate. When 64 independent
starts land on one wrong answer under one schedule, the attractor is the
schedule/objective, not the start. Both audits predicted this; neither
tested a schedule change (deliberately, to isolate the init variable).

Known anchors:
- The production loss DOES separate truth from the ridge point
  (exact-truth loss `9.02e-7` vs ridge `0.267` — a 5-order gap), so the
  information exists; the descent path just doesn't reach it.
- Batched refinement makes one full schedule variant (B=64, 500 steps)
  cost ~25 min wall. The whole matrix below fits in an afternoon.

## Protocol (inherited, unchanged)

- Design and freeze using **seeds 6 and 7 only**. Seeds 8, 9 stay sealed.
- Selection metric: production `finalLoss / coldBaseline` (seed-6 cold
  baseline `2.502263`), best lane per run. CPU mrstft only for
  intra-run lane selection, never for cross-schedule comparison.
- A winning schedule must clear seed 6 at ≤2%, then clear seed 7
  **without modification**, before seeds 8/9 are unsealed (E3).
- No per-seed tuning. One frozen schedule for all.

## D0 — ridge cross-section (diagnostic, run first, ~minutes)

Before any training: map the ridge with the existing `loss-sweep` tool.
Grid `res × shape` around hidden truth with the other 10 params **clamped
to truth**, production loss; repeat with the other 10 clamped to the
escape winner's recovered values (lane 2 of
`output/batch_refine_escape_seed6_v1`).

- 2D grid: `--param res --radius ~2.0 --points 33 --param2 shape
  --radius2 0.35 --points2 33`, both centered on truth.
- Read out: (a) is there a monotone descent path from the lane-2 endpoint
  to truth, or a genuine barrier? (b) how steep is the valley across vs
  along the ridge (curvature ratio)?

Branch: if the cross-ridge gradient at the stall point is at or below the
2e-3 grad noise floor, no schedule fixes this — jump to S5 (objective
change). Otherwise proceed to the schedule matrix.

## Schedule matrix (one batched run each, B=64, seed 6)

All variants run through `batch-refine --mode escape` semantics: elite-01
+ 63 jitters (σ 0.05), 500 total steps, per-lane selection — identical to
the escape control except for the schedule under test. Control numbers
already exist (S0).

| ID | Idea | Schedule |
|---|---|---|
| S0 | control (done) | 150 smooth + 350 production, cosine LR, all params free — best lane 10.66% |
| S1 | freeze-ridge curriculum | Phase A (200): res+shape **frozen** at init, others train. Phase B (150): res+shape free, others frozen. Phase C (150): all free, LR ×0.3 |
| S2 | observable-trajectory continuation (v1 audit's untried recommendation) | Phase A (150): fit summary trajectories only — output RMS envelope + spectral-centroid (brightness) trajectory. Phase B (150): blend λ·summary + (1−λ)·smooth, λ 1→0. Phase C (200): production |
| S3 | gradual homotopy | replace the hard smooth→production handoff at step 150 with a linear λ-blend over steps 100–300 (loss = λ·smooth + (1−λ)·production); no other change |
| S4 | LR-corrected control | S0 with the sweep's ×2.0 group-LR multiplier and production phase extended to 500 (650 total); cheapest variant, isolates "were we just undertrained at too-low LR" |
| S5 | resonance-sensitized objective (only if D0 shows a barrier or matrix fails) | add a high-resolution (4096-window) log-mag term ×0.5 to sharpen the resonance peak the 2048 window smears; rerun best surviving schedule |

Rationale in one line each: S1 — settle the 10 unambiguous knobs first so
the ridge pair resolves against a fixed background instead of compensating
for everything at once. S2 — brightness trajectory pins the filter story
(cutoff sweep + resonance emphasis) before the full-spectrum loss lets
shape compensate. S3 — the step-150 discontinuity may be what shoves lanes
onto the ridge; remove the shove. S4 — null hypothesis; must run so S1–S3
gains aren't just "more LR". S5 — if the loss geometrically can't see
across the ridge, change the loss, not the path.

## Implementation deltas (BatchRefine.swift)

Small, all inside `Examples/SynthID` (no `Sources/DGen` changes → no
guardrail-test risk):

1. **Per-phase freeze masks**: `PerLaneAdam.step` already iterates named
   params; add a `frozen: Set<String>` per phase that zeroes those grads
   (S1). Bounds projection unchanged.
2. **λ-blended loss**: `buildBatchedLoss` gains a `blend: Float` — build
   both smooth and production window terms, weight and sum before the ×B
   rescale (S3, S2 phase B).
3. **Summary-trajectory loss** (S2 only, the one real build): per-window
   RMS is `sum(mag²)`; spectral centroid is `sum(bin·mag)/sum(mag)` over
   the existing DFT bins — both expressible with current tensor ops
   (multiply by a constant bin-index tensor + reduce). L2 on both
   trajectories vs target. If centroid autograd fights the division,
   fall back to unnormalized `sum(bin·mag)` (brightness-weighted energy)
   — acceptable for a continuation loss.
4. **Schedule as data**: `--schedule s1|s2|s3|s4` mapping to a
   `[(steps, loss, frozen, lrScale)]` table; log the active phase in the
   trace so plots show seams.

Order of execution: D0 → S4 (cheapest null) → S1 → S3 → S2 → (S5 only if
needed). Stop early if any variant's best lane clears 2%.

## Success criteria & decision rule

- **Pass**: best lane ≤2% ratio on seed 6 AND res/shape inside spec
  tolerance (not just a lower loss on the same ridge). Then freeze, run
  seed 7 (its own cold baseline, jitters of its best elite), require ≤2%.
  Both pass → write finding, unseal seeds 8/9, run E3 untouched.
- **Partial** (ratio improves but >2%): keep the best schedule as the new
  control, escalate to S5.
- **Fail** (nothing beats 10.66% meaningfully): the ridge is an
  objective-identifiability problem; the honest next question becomes
  whether the 2% gate is the right bar — render best-lane audio vs target
  and A/B by ear before spending more optimizer effort.

## Budget

D0 ~5 min; S1–S4 ~25–30 min each (B=64, 500–650 steps at 2.7 s/step);
S2 adds ~half a day of loss implementation. Whole phase ≤1 day compute,
≤1 day code.

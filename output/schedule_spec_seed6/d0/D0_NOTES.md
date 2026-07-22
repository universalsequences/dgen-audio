# D0 — res/shape ridge cross-section (SCHEDULE_SPEC.md diagnostic)

Date: 2026-07-21. Tool: `loss-sweep`, production loss, seed-6 target,
33x33 grid in transformed space centered on hidden truth
(`--param res --radius 2.0 --param2 shape --radius2 0.35`), ~2 min/grid.

Stall point (escape winner lane 2): res z-delta -0.478, shape delta +0.139
from truth (res 1.885 vs 3.041, shape 0.806 vs 0.668).

## Grid 1 — other 10 params clamped to truth (`grid_truth.csv`)

- Truth is the global min of the slice at 9.0e-7; the stall grid point
  (-0.5, +0.131) sits at 0.635.
- **No genuine barrier along the valley floor**: per-res-row minima are
  monotone decreasing from the stall to truth
  (0.635 -> 0.517 -> 0.484 -> 0.322 -> 0). The information is there.
- **But the valley is a narrow curved diagonal**: the floor shifts ~2 shape
  grid steps (0.044) per res step (0.125), slope d(shape)/d(z_res) ~ -0.28.
  At grid resolution the stall point is an 8-neighbor local minimum — its
  along-valley escape requires a *coordinated* res+shape move; axis or
  diagonal moves hit the valley wall. Three such pocket minima line the
  valley (-0.5, -0.75, -0.875).
- Cross-ridge gradient at the stall point: dL/d(shape) = 0.90,
  dL/d(z_res) = 0.016 — 450x / 8x the 2e-3 grad noise floor. Per the spec's
  branch rule: **schedule matrix proceeds** (no jump to S5).
- Curvature at truth: d2L/dz_res2 = 78, d2L/dshape2 = 2133 (raw units);
  span-normalized these are comparable (~1250 vs ~1045), i.e. the
  conditioning problem is the valley's *rotation* (compensation coupling),
  not a per-axis scale mismatch.

## Grid 2 — other 10 clamped to the escape winner's recovered values
(`grid_lane2.csv`, center file `lane2_center.json`)

- **The slice's global minimum moves to the ridge point**: (-0.375, +0.109)
  at loss 0.337, adjacent to the stall; truth's res/shape scores 0.960 in
  this background — ~3x worse than the ridge.
- All four interior local minima lie on the ridge; truth is not even a
  local min of this slice.
- Reading: lane 2's res/shape endpoint is (locally) *optimal given the
  small residual errors in the other 10 recovered params*. The stall is not
  a failure of descent within the res/shape plane — the other params'
  residuals reprice the slice so the wrong res/shape wins. Reaching truth
  requires coordinated movement of (at least some of) the other 10 together
  with res/shape: a genuinely coupled 12-D valley.

## Implications for the schedule matrix

- S1 phase B (freeze others, train res/shape alone) is predicted to fail
  from a lane-2-like background: the frozen background pins the slice
  minimum at the ridge. Phase C (all free, low LR) carries S1's chances.
- S3 (homotopy) and S4 (more LR + steps) act on the full coupled valley
  and are not contradicted by D0.
- If the matrix fails, D0 grid 2 is direct evidence for S5: the production
  objective's res/shape valley is mispriced by ~0.6-unit-scale backgrounds;
  a sharper (4096-window) resonance term is the right lever, or the
  identifiability question escalates per the spec's fail branch.

# M1 finding — single real snare fit (dgen-70v.3)

Status: **NUMERIC GATE NOT MET** — 0.1491 achieved against a 0.1409 gate.
Recorded 2026-08-19 by review of commit b431b28. Spec: `docs/MODAL_SNARE_SPEC.md` §M1.

Everything below was produced by `swift run ModalDrum fit-real` on artifacts
under `runs/modal_m1*` (gitignored). Target: `SNARE 1.wav`, wrong-snare
control: `SNARE 6.wav`, both from an 11-file kit prepared with
`scripts/prepare_snares.sh` (11 accepted, 0 rejected).

## Score calibration on percussion

CPU MR-STFT, windows 64–2048, hop = w/4, linear + log magnitude, eps 1e-3,
over the fixed 0.75 s window.

| control | score |
|---|---|
| target vs itself | 0.0000 |
| **wrong snare** (SNARE 6) | **0.2114** |
| silence | 0.3409 |
| white noise, RMS-matched | 7.5341 |
| derived gate = min(negatives) / 1.5 | **0.1409** |

Across all 10 other snares in the kit the wrong-snare control spans
0.177–0.241, so 0.2114 is typical, not a lucky draw.

**The flute calibration does not transfer, and not only in scale.** On a 0.75 s
percussion window the target is below −100 dB after 0.19 s, so ~75% of the
frames are silence-vs-silence and contribute nothing. That compresses the
metric hard: *pure silence* scores 0.3409, only 1.6× worse than a completely
different snare. The usable band between "different drum" and "identical" is
0.21 wide, and a gate derived only from the loud controls (white noise) would
have landed at 5.02 — above silence, i.e. passable by rendering nothing. The
silence control is in the negative set for exactly this reason.

## K sweep and noise ablation (300 steps, λ=1e-3)

| K | modal-only | modal+noise | noise gain | modal/noise energy | wall-clock |
|---|---|---|---|---|---|
| 32 | 0.2373 | **0.1664** | −29.9% | 93.0 / 7.0 | 295 / 304 s |
| 64 | 0.2201 | **0.1604** | −27.1% | 93.0 / 7.0 | 298 / 309 s |
| 128 | 0.2173 | **0.1687** | −22.4% | 95.2 / 4.8 | 296 / 305 s |

**Ablation: PASS at every K.** The noise branch earns its place by 22–30%,
which is far outside any run-to-run variation. It is not being faked by high
modes.

**Smallest passing K: none.** Best K is **64**; K=128 is *worse* than K=64, so
the model class saturates at 64 and capacity is not the binding constraint.
M2 should take **P from K=64** (64 gains + 64 decays + 15 taps + 2 noise).

## The gate failure is a model-class limit, not a tuning miss

Three independent levers were swept and none reaches 0.1409:

- **Steps.** K=64 modal+noise at 1200 steps reaches **0.1491** and is flat to
  five decimals over the last 200 steps (0.149070 → 0.149068). 4× the compute
  buys 7%, then nothing. Modal-only at 1200 steps: 0.2170 (vs 0.2201 at 300).
- **λ (high-mode L1).** At K=64, 300 steps, modal+noise: λ=0 → 0.1605,
  1e-4 → 0.1604, 1e-3 → 0.1604, 3e-3 → 0.1604. **Zero effect across 30×.**
  On modal-only it costs 2.7% at 3e-3 (0.2175 → 0.2233) — the guard only bites
  when there is no noise branch to take the wires. λ=1e-3 is therefore
  confirmed as a good default: it canonicalizes the split at no fit cost.
- **K.** See above; 128 < 64.

Best achieved: **0.1491**, which is 1.42× closer to the target than a different
snare is. The gate asks for 1.5×. The fit misses by 5.8%.

## Listen gate (via the listen skill's spectral analysis, `Assets/analyze_wav.py`)

Best fit = K=64, modal+noise, 1200 steps.

**Branch split — PASS, and the guard is working.**

| branch | centroid | character |
|---|---|---|
| modal solo | 227.8 Hz, tonal F0=162.7 Hz | damped shell tone |
| noise solo | 3785 Hz, ZCR 0.099, no periodicity | the wires |

This is the correct decomposition, not a smeared one: the modal branch is not
reaching into the wire band and the noise branch is not carrying pitch. **Not
a soft fail** — no guard-strengthening bead is needed before M2.

**Sum vs target — same drum, but duller.**

| | target | fit |
|---|---|---|
| F0 | 165.3 Hz | 162.7 Hz (one grid step; log-grid quantization) |
| RMS | −20.9 dB | −22.0 dB |
| **centroid** | **2703 Hz** | **2193 Hz** |
| decay to −95 dB | 0.19 s | 0.19 s |

Pitch, level and decay all match. The residual is entirely a **high-frequency
deficit** — 510 Hz of missing centroid, which is what the numeric gate is
measuring too. (The fit also clips 3 samples at +0.6 dB peak; cosmetic.)

## Diagnosis: the noise branch is the bottleneck

The noise branch is one *static* 15-tap FIR times one exponential, so it can
produce exactly one spectral shape with one decay rate. A real snare's wires
are a bright crack that darkens as it decays. The optimizer resolves that
conflict by favouring the crack: **noise τ converges to 12.4 ms, pinned against
its 10 ms floor**, in every modal+noise run. It buys the transient and gives up
the sizzle tail — and the modal bank cannot cover for it, because the guard
(correctly) stops high modes from impersonating noise.

This is not a degeneracy and not a gradient problem. It is the noise branch
being under-parameterized for this material, and it is the finding M1 exists to
produce. Filed as the follow-up bead.

## Reproduce

```bash
scripts/prepare_snares.sh <kit-dir> <prepared-dir>
swift run -c release ModalDrum fit-real \
  --target <prepared-dir>/SNARE\ 1.wav \
  --wrong-snare <prepared-dir>/SNARE\ 6.wav \
  --k 32,64,128 --steps 300 --render-every 25 --out runs/modal_m1
```

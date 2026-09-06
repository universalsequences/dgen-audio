# E-mu Orbit-9090 tom "66.wav" — Handoff (2026-09-03)

Target: `Assets/emu-orbit9090-tom-66.wav`, copied from the eseq sample library
entry titled `66.wav` with tags EMU / EMU Orbit-9090, sha256
`1f29dd660fe4dbc8b69915baee9ff47881db5dd808097f99469d9beb6999824c`.
(There are several library samples titled `66.wav`; the Sonic CD one is an
8-bit 16 kHz file and is NOT this target.) 44.1 kHz 16-bit mono, 9,791 frames
(222 ms), peak −0.7 dBFS, no DC, −60 dB end at 219 ms (the file ends in
zeros). eseq bead: `eseq-q0v1`. Fit: `scripts/fit_emu_tom.py`.

## Measurements

- Onset: 65 samples (1.5 ms) of silence, then the sweep opens directly — no
  separate click, the first cycle is the 3 kHz start of the sweep.
- Instantaneous frequency (analytic signal): ~3.5 kHz at 2 ms, 1.5 kHz at
  5.5 ms, 560 Hz at 8 ms, ~300 Hz at 10 ms, ~155 Hz by 40 ms. From 22 ms on
  it wobbles with an 11.5–12 ms period between ~65 and ~330 Hz. The
  zero-crossing tracker and the one/two-exponential harmonic-ladder script
  both fail on it (rms log error 0.99): this is not a swept sine with
  harmonics.
- Long-window partials (dB, per window):
  - 25–50 ms: 84.5 (−20) 177 (−17) 270 (−20) 362 (−28) 458 (−40)
  - 40–80 ms: 73 (−24) 160 (−20) 246.5 (−23) 332 (−31) 415 (−40) 498 (−47)
  - 100–200 ms: 68 (−38) 155 (−34) 242 (−37) 328.5 (−45) 415.5 (−55) 502 (−65)
  Evenly spaced (~87 Hz in the tail, ~92 Hz at 25–50 ms) but offset from
  zero: FM sidebands fc + n·fm with fc 177 → 155 Hz and fm 92 → 87 Hz. The
  upper sidebands fall −3 / −11 / −21 / −32 dB (a Bessel ladder at index
  ~1.3), one lower sideband at −4 dB; the second lower sideband would fold
  through 0 Hz and is not visible, so the bank is left free rather than
  Bessel-locked. Sideband levels relative to the carrier stay roughly
  constant across the note (extra decays ≈ 0).
- Amplitude: flat for ~20 ms, then log-RMS slope −14.9 / −17.1 / −18.7 /
  −27.9 s⁻¹ over 20–60 / 60–120 / 120–170 / 170–215 ms — one exponential
  with a steepening curve (ampCurve ≈ −43 s⁻², ampDecay ≈ −11.6 s⁻¹).
- Recording texture: >4 kHz RMS is −27 dBFS in the first 20 ms (the sweep
  passing through), −62 dBFS at 20–50 ms and −83 and below after 50 ms. A
  16-bit ROMpler: no hiss block.

## Voice (`fit_emu_tom.py::render`)

```
t'      = max(t − onset, 0), gated at t ≥ onset
φc(t')  = cEnd t' + ca1/cr1 (e^{cr1 t'} − 1) + ca2/cr2 (e^{cr2 t'} − 1)
φm(t')  = mEnd t' + ma1/mr1 (e^{mr1 t'} − 1)
bank    Σ_k h_k e^{d_k t'} sin 2π frac(φc + (k−1) φm + p_k),  k ∈ {0,1,2,3,4,5,6}
        (k = 1 carrier: h = bodyAmp, d = 0, p = 0; k = 0 lower sideband)
env     attack(t'; attackTime) · exp(ampDecay t' + ampCurve t'²)
click   clickAmp sin(2π clickFreq t') e^{clickDecay t'}          (cap 0.05)
noise   noiseAmp LP(noise, noiseCutoff) e^{noiseDecay t'}         (cap 0.05)
out     tanh(drive · mix) / drive · outGain
```

39 scalars. Partial-track loss: heterodyne along the MEASURED seed's
φc + (k−1)φm tracks with a two-modulator-period window (the sidebands are
fm apart, so the window must span ≥ 2/fm ≈ 23 ms), from 6 ms on.

## Runs

(filled per round)

| run | gate (baseline) | train parts | what changed / what it showed |
|---|---|---|---|
| `output/emu_tom_v1` | **0.1555** (0.3809) | spectral 0.173, harmonic 0.156, pooled 1.31 | 39 scalars, onset + two-exp carrier + one-exp modulator + 6-partial bank. Tail (40 ms on) within ±1 dB on all seven partial tracks. Onset landed 1.2 ms late, 0–8 ms windows 40–50 dB short; the fitter parked a 2.7 kHz "click" at its cap for 50 ms as a stand-in for the sweep's start. The instantaneous frequency sits on a ~2.9 kHz plateau from 2 to 4.5 ms before falling — an exponential from onset cannot do that. |
| `output/emu_tom_v2` | 0.1510 (0.3684) | spectral 0.168, harmonic 0.206, pooled 2.78 | + `hold` (SNAP holds, then falls), + second modulator exponential (`ma2`,`mr2`), click/noise caps tightened. Gate barely moved; hold collapsed to 0.2 ms and `cEnd`/`ca1` never left the seed. Cause: the pooled ≥2.5 kHz term (0.3 × 2.78 of a 1.06 total) was chasing the sample's −85 dBFS 16-bit floor and pinned `noiseAmp` at its cap. The pooled term is the hiss objective; this target has no hiss. |
| `output/emu_tom_v3` | 0.1417 (0.3684) | spectral 0.157, harmonic 0.180, pooled off | pooled weight 0, sweep reseeded at the measured plateau (hold 2.8 ms, −470/s). Onset now within 2 samples. Sweep still too slow (555 Hz target vs 1182 Hz learned at 8 ms; 330 vs 812 at 10 ms) — `ca1`/`cr1` are coupled and coordinate descent cannot move them together. Noise pinned at its cap as a stand-in for early high energy. Target's early amplitude dips every 1–1.5 ms ⇒ modulator ≈ 700–900 Hz at 5 ms, and the sideband spacing / carrier is ≈0.65 at 5 ms, 0.5–0.56 from 8 ms on: the modulator is ratio-locked to the carrier (FM tom, one pitch envelope). |
| `output/emu_tom_v4` | 0.1530 (0.3608) | spectral 0.171, harmonic 0.139 | Modulator ratio-locked to the carrier (`ratio` 0.561 + a small own fall), carrier seeded from a least-squares pre-fit of the instantaneous-frequency track (`output/emu_tom_carrier_prefit.json`: hold 4 ms, −1120/s cliff, 482 Hz drop at −81/s). Early deficits collapse (8–15 ms: +4/+7 dB, was +27/+17) but the gate slips: the locked modulator swings the first 8 ms far wider than the target, whose first cycles are nearly a bare sine — the modulation index rises over the first ms. |
| `output/emu_tom_v5` | **0.1505** (0.3622) | spectral 0.168, harmonic 0.137 | + `sbAttack` (sideband rise, 3.1 ms fitted). Converged with v4 (train 0.209 vs 0.212, same basin, sweep unchanged). Tail exact from 40 ms on; 8–30 ms partials within ±4 dB except P3/P5 spot misses; 0–3 ms still 10–24 dB short above 200 Hz with the noise burst pinned at its 0.05 cap. |
| `output/emu_tom_v6` | 0.1466 (0.3616) | spectral 0.164, harmonic 0.137 | `--only` refine of the transient block (noise / click / attack / sbAttack / hold) from v5 with the noise cap widened to 0.15. Noise pinned at the new cap again and the 0–3 ms deficit did not move: the gate is rewarding a 10 ms 3.7 kHz noise stand-in, not the sound. Not shipped as the default; it is the third hit in the A/B for the ear. |

## Where it stands (2026-09-03)

- Shipped default = **v5** (`output/emu_tom_v5/recovered_params.json`, gate
  0.1505 vs 0.3622 midpoint baseline, 58.5 %). The tail from 40 ms on matches
  on all seven partial tracks to within ±1 dB; 8–30 ms within ±4 dB with spot
  misses on P3/P5; the first 3 ms are 10–24 dB short above 200 Hz.
- What the first 3 ms still lack: the target's first cycles show a 2·fc
  component (2881 / 5776 Hz at 2 ms) at near full scale — the carrier is
  saturating hard while it sits on the plateau (analytic amplitude −1 dB flat
  for 15 ms with only shallow beat dips). A level-dependent drive (higher
  early) or an early-only drive envelope would be the next capacity; the
  fitter cannot reach it because `drive` / `bodyAmp` / `outGain` are coupled
  under the 0.9 peak normalisation.
- Coordinate descent could not move the coupled sweep (`ca1`,`cr1`,`hold`)
  from a wrong basin (v1–v3); the direct least-squares pre-fit of the
  instantaneous-frequency track (`output/emu_tom_carrier_prefit.json`) plus
  the ratio-locked modulator is what fixed the early 15 ms. Do that first on
  the next FM-shaped sample.
- A/B for the ear: `output/emu_tom_ab_target_v5_v6.wav` (target / v5 / v6,
  twice) and `output/emu_tom_ab_target_v4_v5.wav`.

## Port

`content/instruments/Drums/Orbit Tom 66/{dsp,ui}.lisp` +
`Drums/Orbit Tom 66.presets` in eseq, from
`.claude/skills/identify-drum/fm-tom-dsp-template.lisp` (new template for the
FM-sideband topology). Parity via `tools/audition/synthid_port_check.py`
against v5: max abs 1.45e-3 (float32 phase drift on the 2.9 kHz plateau),
gate instrument 0.1504 vs learned.wav 0.1505. Layout test
`metal_seq_fx_lisp_lays_out_orbit_tom_66_controls` passes (51 params).

Final v5 params:
```
{
 "onset": 0.00158,
 "hold": 0.004,
 "cEnd": 154.0,
 "ca1": 2016.43249,
 "cr1": -1138.85619,
 "ca2": 466.63161,
 "cr2": -81.0,
 "ratio": 0.561,
 "ma1": 0.1542,
 "mr1": -1489.83032,
 "attackTime": 0.00224,
 "ampDecay": -10.9499,
 "ampCurve": -40.71278,
 "bodyAmp": 0.73623,
 "clickFreq": 4709.00492,
 "clickAmp": 0.00885,
 "clickDecay": -300.0,
 "noiseCutoff": 5786.4813,
 "noiseAmp": 0.05,
 "noiseDecay": -100.0,
 "drive": 1.28515,
 "outGain": 0.66,
 "h0": 0.77084,
 "d0": -2.24688,
 "p0": 0.35358,
 "h2": 0.78051,
 "d2": -0.89125,
 "p2": 0.01944,
 "h3": 0.29394,
 "d3": -0.90147,
 "p3": 0.06768,
 "h4": 0.08854,
 "d4": -0.5,
 "p4": 0.12312,
 "h5": 0.02681,
 "d5": -0.26646,
 "p5": 0.30955,
 "h6": 0.02758,
 "d6": -7.11048,
 "p6": 0.17763,
 "sbAttack": 0.00307
}
```

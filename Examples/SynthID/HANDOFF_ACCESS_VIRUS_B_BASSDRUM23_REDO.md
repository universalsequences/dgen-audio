# Access Virus B BassDrum_23 — SynthID redo (2026-09-03)

Supersedes `HANDOFF_ACCESS_VIRUS_B_BASSDRUM23.md` (v1–v3). The v3 result was a
polished sine kick: the fundamental and its envelope matched, the ladder of
swept harmonics did not, and the MR-STFT gate could not tell. This redo
measured that ladder first, built the voice around it, and added a harmonic
diagnostic the gate lacks.

Target: `Assets/access-virus-b-bassdrum-23.wav` (32.5 kHz, 461 ms), fitted at
48 kHz. Fit script: `scripts/fit_virus_kick.py` (NumPy voice + coordinate
descent, same protocol as `fit_clap.py`).

## What the sound actually is

Heterodyne each harmonic k along a zero-crossing phase track of the target
(two-period Hann window, so neighbours sit on the window's null — a fixed 4 ms
window leaks between harmonics once f0 reaches 48 Hz and fakes a flat
"pulse-like" spectrum). Findings:

- Harmonics 1..20 of the sweep account for the target below 1 kHz to within
  0.1 dB in every time window (band table target vs resynthesis vs residual).
  Nothing inharmonic matters; the residual above 1 kHz after 50 ms is
  -65..-75 dBFS texture, under the gate's -60 dB floor.
- The ladder relative to H1, by window (4–15 / 15–40 / 40–80 / 80–150 /
  150–250 ms):
  H2 -26/-25/-26/-34/-34, H3 -37/-42/-44/-48/-48, H4 -51/-46/-42/-44/-45,
  H5 -54/-49/-53/-50/-50, H6 -59/-55/-57/-69/-75, H7 -63/-61/-48/-47/-47,
  H8 ~-65, H9/H10 -60..-75.
- The harmonics decay with H1 (dB/s slope of Hk−H1 ≈ -13 for H2/H3, ≈0 for
  H5/H7), not with its power: they are oscillator content, not a static
  waveshaper. tanh saturation would make H3 ∝ H1³ — it does not.
- The tail (>100 ms) is a nearly pure 48.3 Hz sine (H2 -41 dB).
- Pitch: the single-exponential model sat 30–40 % low from 30 to 70 ms
  (model 109 Hz vs measured 152 Hz at 30 ms, 68 vs 100 at 40 ms). A
  two-exponential fit to the zero-crossing track has 0.013 rms log error
  versus 0.139: fEnd 48.06, 1458·e^{-157t} + 573·e^{-61t}.
- Amplitude: rises to -0.4 dBFS at 60 ms, then -1.9 @100, -7.3 @160,
  -13.6 @200, -29.6 @300 ms (log-quadratic).

## Voice (`fit_virus_kick.py::render`)

    phi(t) = fEnd t + a1/r1 (e^{r1 t}−1) + a2/r2 (e^{r2 t}−1)
    env(t) = attack(t; attackTime, normalised at 50 ms) · exp(ampDecay t + ampCurve t²)
    bank   = sin(2π frac φ) + Σ_{k=2..10} h_k e^{d_k t} sin(2π k frac φ)
    y      = tanh(drive · (bodyAmp·env·bank + click + noise)) / drive · outGain

35 bounded scalars (5 pitch, 4 envelope, 9 levels, 9 extra decays, 3 click,
3 noise, drive, outGain). Phase is wrapped before the sines so the eseq port
stays float32-exact. The saturator is gain-normalised so `drive` sets shape
only; without that, coordinate descent could not trade drive against outGain.

## Loss

Training = per-bin log-magnitude MR-STFT on windows 256..4096 (tonal sound:
no band pooling) + 0.3 × harmonic-track loss, where the track loss is the mean
|log(A_k(t)+1e-3) − log(Â_k(t)+1e-3)| over harmonics 1..10 and 2 ms steps to
300 ms, heterodyned on the *target's* phase track. The gate stays
`compare.py`. The run log prints the synth-minus-target harmonic table; read
it, not just the gate.

## Runs

| Run | Train (spectral / harmonic) | Gate (48 kHz) | Notes |
|---|---|---|---|
| v3 (old profile, resampled) | — / 0.355 | 0.0688 | H4 err 7.9 dB, H7 err 6.4 dB, H3 +5.8 dB (40–250 ms mean) |
| `output/virus_kick_v4` | 0.0635 / 0.169 | 0.0567 | 30 restarts + measured seed; measured seed won every round |
| `output/virus_kick_v5` | 0.0621 / 0.151 | 0.0554 | drive floor 0.5→0.05, clickDecay to -8000 |
| **`output/virus_kick_v6`** | 0.0621 / 0.151 | **0.0554** | gain-normalised saturator; identical to v5 → converged |

Gate improvement vs midpoint baseline 67.8 % (0.1721 → 0.0554); vs v3's
render 19.5 %. The 80 % gate is not met and, as v3 showed, the gate number
is not the point for this sound.

Harmonic-track mean |dB error| over 40–250 ms, v3 → v6:
H1 1.6→0.6, H2 4.2→3.8, H3 5.8→3.3, H4 7.9→1.3, H5 1.6→1.7, H6 0.9→0.7,
H7 6.4→0.8. Remaining misses are in the first 30 ms of the sweep (f0 > 300 Hz,
where a small pitch error moves a harmonic out of its own heterodyne window)
and at levels within 10 dB of the -60 dBFS floor.

Pinned: `d3` at -80/s (H3 is -37 dB early and -48 dB late relative; the
bank wants it to fade faster than the body). Checked and rejected: forcing a
linear output (drive 0.05) raises the harmonic loss 0.151→0.162 and leaves the
gate flat, so the mild saturation is real, not a leak.

## Final scalars (v6)

fEnd 48.326; a1 1271.98, r1 -138.92; a2 454.76, r2 -61.4; attackTime 0.0839 s;
ampDecay -8.196/s; ampCurve -23.27/s²; bodyAmp 0.6464;
h2 0.0889 (d2 -74.5), h3 0.0354 (d3 -80), h4 0.0102 (d4 -3.0), h5 0.0030
(d5 +0.4), h6 0.0010 (d6 -11.7), h7 0.0048 (d7 -0.4), h8 0.0011 (d8 -3.8),
h9 0.00027 (d9 -5.3), h10 0.0011 (d10 0);
clickFreq 619.8 Hz, clickAmp 0.1008, clickDecay -2818/s;
noiseCutoff 2169 Hz, noiseAmp 0.00294, noiseDecay -12.43/s;
drive 0.744; outGain 2.599 (2.246 in the instrument after 0.864 peak normalisation).

## eseq port

`content/instruments/Drums/Virus B BassDrum 23/{dsp.lisp,ui.lisp}` +
`.presets`, filled from
`.claude/skills/identify-drum/harmonic-kick-dsp-template.lisp` by
`tools/audition/synthid_port_check.py`. At 48 kHz, host pitch 261.63 Hz:
parity max abs 1.32e-3 (float32 clock drift, grows smoothly with time,
no sample shift, -47 dB relative to the signal), gate 0.0554 for both renders.

## Listen

- `output/virus_kick_v6/ab.wav` — target / v6, twice
- `output/virus_kick_v6/ab_target_v3_v6.wav` — target / v3 / v6, twice
- `output/virus_kick_v6/compare_vs_v3.png` — spectrograms, v3 as "initial"

## Round 2 (same day): character — the tick that wasn't and the hiss that was

Ear feedback after v6: "the kick is great, the recording character is
missing — like a finger on a guitar cable, high-passed". Band-by-time table
of target minus v6 (`output/virus_kick_v6`) showed two things:

- **v6 opened with a tick the target does not have.** The fitted click had
  become a 0.095-amplitude impulse on sample one; the target's first
  millisecond is a -40 dB wiggle at 1.7 kHz that grows in. Removing the click
  costs 0.0003 on the gate. `clickAmp` is now capped at 0.02 and the seed is
  clamped into bounds (a seed outside a tightened bound was silently kept).
- **The target carries a broadband hiss at ~-70 dBFS**, dip at 4–5 kHz,
  plateau 6–10 kHz, nothing above the Virus B's 16.25 kHz Nyquist. >4 kHz RMS:
  -69 dBFS at onset, -72 @50 ms, -75 @100, -81 @150, -87 @200, then the
  recording's -90 dB floor. It sits ~-90 dB *per bin*, under the gate's
  -60 dB epsilon; worse, the per-bin log metric penalises any noise whose
  realisation differs from the target's, so every fit had pushed it to zero.

Added block: `hissAmp · HP(noise, hissCutoff) → two fixed 16 kHz LPs ·
exp(hissDecay t)`, zero-default. Fitting it needed its own objective: a
band-pooled log-power term restricted to bands ≥ 2.5 kHz with ε = 1e-6.
Letting that term into the full fit dragged the body parameters (v7/v8 went
to a worse basin and pushed the cutoff to 2 kHz to fake the upper ladder with
noise), so the final round refines *only* the three hiss scalars from the v6
body (`--only hissCutoff,hissAmp,hissDecay --pooled-weight 20`).

| Run | Gate | Notes |
|---|---|---|
| v7/v8 | 0.0553 / 0.0573 | pooled term in the full fit: wrong basin, rejected |
| v9 | 0.0595 | hiss-only refine, pooled weight 1: 4–5 dB short at 50–150 ms |
| **v10** | 0.0609 | hiss-only refine, pooled weight 20: hissCutoff 5300 Hz, hissAmp 4.51e-4, hissDecay -6.57/s |

v10 >4 kHz RMS, target / synth per 50 ms: -69/-65, -72/-75, -75/-78,
-81/-80, -87/-83, -90/-85, -90/-88, -90/-91. The gate rose 0.0554 → 0.0609
purely because the metric charges for a noise realisation it cannot match;
the harmonic table is unchanged. Instrument: `hiss` knob (0..4, 1 = as
fitted), presets "Dry" (hiss 0, noise 0) for the A/B. Port parity max abs
1.32e-3 (drift, as before), gate 0.0609 both renders.

Listen: `output/virus_kick_v10/ab.wav` (target / v10) and
`output/virus_kick_v10/ab_target_v3_v6_v10.wav`.

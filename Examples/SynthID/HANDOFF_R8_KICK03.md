# HANDOFF — Roland R-8 "Kick03" (modal kick), 2026-09-03

Target: `Assets/r8-kick03.wav` (eseq library title `Kick03.wav`, tags drums / kick /
Roland / Roland R8, sha256 `26eb639f7d6587382cdc95d297626bef8fac396f5b5ff5fc8892fb805042278e`).
Fit: `Examples/SynthID/scripts/fit_r8_kick.py`. Instrument: eseq
`content/instruments/Drums/R8 Kick 03/` via
`.claude/skills/identify-drum/modal-kick-dsp-template.lisp`. Bead `eseq-u13m`.

The user's brief: "a drum kick but it doesn't sound that electronic, it has a real feel
to it. likely gonna need modal to do this right." The measurements agree.

## 1. Measurements (before any DSP)

Housekeeping: 44,100 Hz, 8,032 frames = 182.1 ms, peak 0.00 dBFS, DC 0.001, mono
(both channels identical). No capture band-limit: the spectrum rolls off smoothly,
-70 dB per 1 kHz band at 2 kHz, -90 at 6 kHz, -100 above 8 kHz. The sample ends with
a fast fade over the last 30 ms (-600 dB/s), i.e. it was truncated in the machine.

Onset: opens with a 0.4 ms negative pre-swing (-0.1) then a positive push to +0.62
at sample 34 (0.77 ms) that holds ~0.5 for a millisecond — a pressure pulse, not a
click. Content "grows in" over ~0.7 ms.

Waveform: a train of positive humps whose spacing lengthens from ~16 ms (13→29 ms)
to ~25 ms (110→135 ms): the low body glides from ~65 Hz to ~40 Hz. Between 13 and
29 ms a smaller hump sits at 20 ms: a strong partial near 130 Hz that dies by 30 ms.

Pitch model: **not a swept sine.** Zero-crossing f0 is chaotic (406, 613, 150, 126,
130, 76, 171, 66 Hz …) because two partials of comparable level alternate. Ridge
tracking finds one partial 183→129 Hz over 2–15 ms (fast glide, ratio 1.42), then
~121 Hz at 25 ms, ~160 Hz at 40 ms, ~125–135 Hz from 100 ms on; the lowest partial
73→48 Hz over 5–35 ms then 45 Hz. Heterodyning along any one pitch track finds the
same inharmonic ratio set in every window (≈ 1 : 1.8 : 2.7 : 3.7 with the ratio-2.7
member the strongest), and from 65 to 130 ms every partial drifts down by the same
factor 1.14–1.15: a fixed inharmonic mode set under one shared multiplicative
(tension) glide. That is a membrane, so the voice is modal.

Low-band peaks (8192-window, target): 0 ms 64/117/141/164/211/264 Hz; 43 ms
53/111/141/164/193; 85 ms 47/100/141/170/182; 128 ms 47/88/100/123/141; 171 ms
41/82/123/158. Matrix pencil on the 40–180 ms tail: 46.7 Hz (T60 259 ms), 92.6
(147), 137.9 (154), 80.5 (214), 174.6, 195.5.

Ring modes (residual after removing the low partials, matrix pencil, 8–60 ms):
623 Hz T60 46 ms (-3.5 dB rel), 254 (86 ms), 2420 (17 ms), 2800 (8.5 ms), 396/365
(115–190 ms), 2920 (74 ms), 1120 (167 ms), 1010 (40 ms). The 2.3–3.0 kHz cluster is
visible in the spectrogram as horizontal lines to ~75 ms (-30 dB at 12 ms); the
300–400 Hz and ~620 Hz modes ring past 100 ms at -33..-45 dB. This cluster is the
"real" in the sound.

Click: >4 kHz RMS -15.5 dBFS over 0–5 ms, -27 at 5–15, -30 at 15–30, -48 at 30–60,
-62 at 60–100, -70 after. Broadband to ~6 kHz, T60 ≈ 5–8 ms (matrix pencil
2–20 ms: 2838/1071/5557/518/1953 Hz all at -730..-1370 /s).

Band table (dBFS, 30-100 / 100-200 / 200-500 / 500-1k / 1-2k / 2-4k / 4-8k / 8-16k):
0–20 ms -17 -3 -13 -27 -28 -19 -26 -36; 20–50 -8 -11 -27 -36 -36 -35 -50 -72;
50–100 -14 -18 -30 -46 -47 -54 -71 -75; 100–200 -21 -23 -44 -52 -60 -68 -73 -73.

## 2. Voice (fit_r8_kick.py::render)

    g(t)  = 1 + gA1 e^{gR1 t} + gA2 e^{gR2 t}        shared tension glide
    G(t)  = gA1/(-gR1)(1-e^{gR1 t}) + gA2/(-gR2)(1-e^{gR2 t})
    low   = Σ_k la_k e^{ld_k t} sin 2π frac(lf_k (t + lg_k G(t)))    k = 1..5
    mid   = Σ_j ma_j e^{md_j t} sin 2π frac(mf_j t)                  j = 1..8
    body  = (low + mid) (1 - e^{-t/attackTime})
    click = noiseAmp LP(noise, noiseCutoff) e^{noiseDecay t}
    hiss  = hissAmp HP(noise, hissCutoff) e^{hissDecay t}
    out   = tanh(drive · mix)/drive · outGain

57 scalars. Per-mode glide scale `lg_k` because the lowest partial glides more
(73→45, ratio 1.6) than the 130 Hz one (160→125 late, ratio 1.28). Mode
frequencies are bounded ±45 % (low) / ±30 % (mid) around the measured seeds and
random restarts keep them (3 % jitter): a random mode set never lands in the basin.
Loss = per-bin log magnitude (256…4096) + 0.5 × an 8192-window log-magnitude term
over 20–700 Hz (6 Hz resolution, places the membrane modes the short windows blur).
`HarmonicTracks` keeps its name so `deficit_table.py` prints the low-band peak
table; there is no harmonic ladder here.

## 3. Runs

| run | train (spec + 0.5·lowband) | gate | notes |
|---|---|---|---|
| v1 | 0.555 (0.366 + 0.379) | 0.319 (baseline 0.898) | wrong basin: drive pinned 3.9 → body hard-clipped at 0.29; noiseAmp 1.48 / noiseDecay -112 turned the click into a 60 ms mid pad; gA1 3.65 × lg 1.7–2.0 swept every mode in from >700 Hz; gR2 -6.5 acted as a constant offset (lf1 pinned at 30 Hz to compensate). Low band: the 141 Hz partial 10 dB short through 64 ms. |
| v2 | 0.565 (0.389 + 0.353) | 0.342 | bounds tightened (drive ≤ 1.2, gA1 ≤ 2, gR2 ≤ -12, noiseAmp ≤ 0.8, noiseDecay ≤ -150). Still clipping: drive, la5, ma6–8, noiseAmp, gA1 all pinned at max, onset flat-topped at 0.37. 100–200 Hz short 14 dB at 8–15 ms. Question raised: is the saturation real character (the target's humps are flat-ish, crest factor low) or a crest-factor crutch? → v3lin vs v3sat. |
| v3lin | 0.580 (0.413 + 0.335) | 0.358 | drive ≤ 0.3 forced. The fitter still pins la1/la2/la5/noiseAmp at max to drive the tanh (mix peaks ~6 into tanh(0.3·x)): the target's limiting is real character, not a crutch. Worse gate. |
| v3sat | 0.547 (0.375 + 0.344) | 0.330 | drive ≤ 1.2 kept, gA1 ≤ 3, la5/ma6–8/noiseAmp widened. Best train so far; la2, ma5, mf2, mf7 pinned → widen; 100–150 ms mid bands 10–18 dB short (ring tails at -63..-70 dBFS, under the gate floor). |
| v4 | 0.520 (0.363 + 0.314) | 0.321 | + per-mode initial phase lp1–5 (matters once the clipper shapes the sum), la ≤ 6, ma ≤ 1.5, mf ±50 %. Still the clipped basin: waveform flat at ±0.37 for 60 ms (target swings ±0.75). cmp.png: tail from 60 ms matches; the target's 0–50 ms low band is a dense gliding smear, and its 0.5–3 kHz is a diffuse cloud, neither of which sparse lines + a clipper reproduce honestly. |
| **v5** | 0.570 (0.402 + 0.336) | 0.352 | drive forced linear (≤ 0.15), la ≤ 1, ma ≤ 0.3, + rattle block. **The honest basin**: from 10 ms on the waveform humps match the target in shape and level (0.75 at 29 ms), tail matches; modes land at 45.7 / 89.6 / 127.7 / 167.7 / 244 Hz (1 : 1.96 : 2.8 : 3.67 : 5.3, the measured inharmonic set), glide 0.25 fast (-481/s) + 0.26 slow (-35/s) with per-mode depth 1.8–3.0. rattleAmp pinned at 0: the per-bin metric penalises noise it cannot phase-match (same as the Virus hiss) → fit alone (v6). The gate prefers v4's clipped pulse train by 0.03; that number is wrong about the sound (cmp.png). |
| **v6 (shipped)** | pooled 1.28 (texture-only) | 0.360 | v5 body frozen, `--only rattleAmp,rattleHp,rattleDecay,hissCutoff,hissAmp,hissDecay --harmonic-weight 0 --pooled-weight 20`. Hiss 4.5e-4 → 1.48e-3 at -2.9/s (a steady floor, hissCutoff pinned at the 2 kHz floor); rattle still 0 even against the pooled term — it stays in as inert capacity (the `rattle` knob does nothing at the identified sound until rattle_amp is raised). >4 kHz RMS learned -33/-68/-68 vs target -27/-61/-60 per 50 ms: the texture is still 6–8 dB under, the gate rose 0.352 → 0.360 as expected. |

## 4. Diagnostics after v6 (deficit_table, target − learned dB)

Bands 30-100 / 100-200 / 200-400 / 400-800 / 800-1.5k / 1.5-3k / 3-6k / 6-12k / 12-24k:
0–3 ms +0 +0 +0 +0 -7 -17 -5 +17 +8 (the first 3 ms: synth click too bright at 1.5–3 kHz, short above 6 kHz);
3–8 +0 +0 +7 +9 -1 +0 -1 +5 +6; 8–15 +0 +5 +5 -5 -3 +1 -2 +4 +3; 15–30 -2 -1 -4 +2 -2 -1 +4 -5 -8;
30–60 -0 +1 -7 +5 +1 +3 -1 -10 -6; 60–100 -1 -3 +5 +4 +2 +5 +8 +0 -6; 100–150 -1 -2 +10 +6 +1 +15 +6 -2 -5.
The 100–150 ms mid deficits are the target's ring tails at -48..-70 dBFS (under or at the gate
floor); the 200–400 Hz late deficit (+10) is a ring mode the fit lets decay too fast.
Low-band peak table (8192 win): synth 47/135/199 Hz late vs target 47/123/141/158 — the
120–160 Hz pair is one mode in the synth; a sixth low mode is the obvious next capacity.

## 5. What the gate got wrong (read this before the next modal fit)

v1–v4 all converged on a hard-clipped pulse train (drive pinned at max, body flat at ±0.37 for
60 ms, target swings ±0.75). The per-bin metric rewards it because a clipped pulse train has the
dense low-band comb the target's fast-gliding, fuzzy membrane shows in the spectrogram. Forcing
the tanh linear (v3lin) did not help while levels were unbounded — the fitter pinned every level
at max to saturate anyway. The honest basin needed **both** drive ≤ 0.15 and level caps
(la ≤ 1, ma ≤ 0.3, noiseAmp ≤ 0.8). Its gate is 0.03 *worse* than the clipped one. For modal /
real-drum targets: bound the saturator to its linear regime from round 1 and judge rounds on
`cmp.png` (waveform overlay + spectrograms) and the ear, not the gate delta.

## 6. Parity

`tools/audition/synthid_port_check.py` on v6: max abs 5.72e-4, 0 samples > 1e-3, gate 0.3600
for both renders, peak normalisation 0.0956 folded into `out_gain`. Instrument
`content/instruments/Drums/R8 Kick 03/` (dsp.lisp from the modal template, ui.lisp four
columns, presets R8 Kick 03 / Dry / Tight / Boomy / Woody / Sub), layout test
`metal_seq_fx_lisp_lays_out_r8_kick03_controls`.

Listen: `output/r8kick_v6/ab.wav` (target / learned ×2) and
`output/r8kick_v6/ab_versions.wav` (target / v4 clipped / v6 shipped, twice).

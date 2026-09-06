# R8 Kick03 acoustic rework — 2026-09-05

Bead: `eseq-u13m` (still in progress, user listening verdict required).
Previous attempt: `HANDOFF_R8_KICK03.md`. The user rejected that v6 as too basic
and asked for a genuinely realistic reconstruction plus boom-bap shaping controls.

## Status / honesty

An installed, validated **candidate**, not a perfect recreation. This is actual
synthesis: no sample playback, residual data, target-derived waveform arrays, or
learned FIR/EQ tables. The independent gate improved, but **does not meet the
80% improvement criterion**. Attack texture still differs; the user's ear is the
acceptance gate. Do not close the bead on the implementation or these numbers.

Listen in `output/r8_acoustic_v8/`:

- `ab-target-old-new-level-matched.wav`: target / old shipping synth / new shipping
  synth, three times; 700 ms per sound includes each synth's real tail.
- `ab-target-old-new.wav`: same order without individual loudness matching.
- `ab-report.json`: exact gains, active-window RMS, and comparable gate numbers.
- `validation/presets.wav`: nine presets in their file order, no per-preset
  normalization. Individual `validation/preset-*.wav` files are also present.
- `ui.png`: real `metal_seq capture`, opened and visually inspected.
- `diagnostics.png` / `diagnostics.txt`: raw-reference waveform, spectral and band
  comparison, all on shared scales. These are diagnostics, not a waveform loss.

## Provenance and diagnosis

Unchanged source: `Assets/r8-kick03.wav`, library title `Kick03.wav`, Roland R8
(tags drums / kick / Roland / Roland R8), SHA256
`26eb639f7d6587382cdc95d297626bef8fac396f5b5ff5fc8892fb805042278e`.
44,100 Hz, 8,032 frames (182.13 ms). See previous handoff for the initial
measurements; the new diagnostic uses explicit causal 3rd-order bandpasses,
so its RMS values are not interchangeable with old STFT-band tables.

The original voice collapsed low resonances, gave all modes one instantaneous
rise, and substituted clean sine rings for diffuse shell texture. Its RATTLE
macro multiplied an identified zero. Its fit ended at the recording boundary,
but the actual instrument kept sounding after it. Its coordinate search could
not move coupled frequency, initial phase and pitch-relaxation parameters together.

The important new evidence is the low head's **delayed growth**. Allowing that
mode a separate rise found roughly 22.7 ms versus sub-millisecond rises for the
upper head modes. This removed much of the excessive low-frequency onset without
saturation. Simply adding more modes/noise did not fix that envelope mismatch.

## Voice and fitting

`Examples/SynthID/scripts/fit_r8_acoustic.py`:

- Seven membrane modes: frequency, amplitude, decay, phase, independent rise,
  fast and slow tension depths. Two shared exponential relaxation times.
- Ten struck shell/beater sine modes, independently phased and damped.
- Six filtered contact-noise bands, each with fast amplitude/decay/rise and an
  independent quieter decay. Fixed centers/Q: 350/.9, 700/.9, 1400/1.2,
  2600/2, 4200/1.5 Hz (bandpass), and 6500/.707 Hz (highpass).
- The struck noise has a fitted, bounded negative-pressure modulation from the
  modal signal. It does not feed back into the modes; quiet texture is independent.
- A scalar start/end cubic fade represents the ROM cut. No embedded envelope table.
- 125 fitted, bounded ordinary scalars, not 125 independent waveform samples.
  Frequencies/positive rates/times/amplitudes use log coordinates; phases and
  dimensionless depths use linear coordinates. Full bounds in each report.
- No saturator in the fit; no gain/amplitude degeneracy and no hidden per-candidate
  peak normalization. `outGain=1` is fixed during identification.
- Torch CPU derivatives, one thread, scipy L-BFGS-B joint bounded optimization.
  NumPy/SciPy implement the independent reference. No DGen compiler changes.
- Training is phase-blind: centered MR-STFT windows 256..8192, low-band fine log
  magnitudes, log pooled band power, within-band spectral contrast, and a small
  linear-magnitude term. Contrast discourages replacing diffuse noise with one
  line having the same average band power. No waveform, slope or crossing loss.
- Training includes actual silence after the recording (240 ms total).

Run history (losses only comparable within a fixed objective/topology):

| run | train | independent gate, raw reference | finding |
| --- | ---: | ---: | --- |
| acoustic v1 | .47703 | .36347 | joint modal fit; insufficient diffuse mid texture |
| v2 | .45507 | .33450 | converged toward clean line substitutes |
| v3 | .52230 | .35217 | six noise bands; fine loss restricted to low band |
| v4 | .42791 | .31614 | spectral contrast + v6 measured membrane seed; better basin |
| v5 | .43977 | .41220 | noise-heavy restart; worse objective and gate, not selected |
| v6 | .42421 | .31922 | pressure-modulated contact; small improvement only |
| v7 | .39752 | .31216 | independent low-mode rise fixes low onset balance |
| **v8** | **.39747** | **.31221** | converged after 339 further iterations; selected |

v1/v2 used the old shared `render_reference.write_wav`, which clips float payloads
at 1. That was discovered in the diagnostics and corrected in v3 onward: output
now uses unclipped scipy float WAVs. The reported gate was always on the raw array,
but those two early WAV files must NOT be used for a precise waveform audit.

v8 midpoint gate .64771 -> .31221 = **51.80%**, not an 80% pass. Porting folds a
fixed .88252-ish headroom factor into output gain (the standard port check's .9
peak convention). Comparable **compiled** 240 ms gate: old .32197, new .30293.
Do not compare these to the old handoff's .360 without matching duration and gain.
The A/B loudness match changes only listening artifacts, not fitting or gates:
raw active-window RMS target .20033, old .19796, new .17873; total matched gains
are .94175 / .95302 / 1.05556. See `ab-report.json` for exact values.

### Remaining mismatches and bounds

Raw new minus target, representative causal band RMS differences:

- 200–400 Hz at 15–30 / 30–60 ms: old +8.2/+5.7 dB, new -2.0/-1.2 dB.
- 1.5–3 kHz at 100–150 ms: old -8.5 dB, new -0.8 dB.
- 6–12 kHz at 3–8 ms: old -13.0 dB, new -0.5 dB.
- **Unresolved:** 1.5–3 kHz at 0–3 ms is still +10.1 dB, then -5.1 dB at
  3–8 ms; 3–6 kHz at 15–30 ms is -7.9 dB. The transient's detailed time evolution
  is not represented exactly by these modal and exponential-noise envelopes.
- f17 reaches its lower bound (~4.45 kHz); fast glide depths g1/g2 reach their
  upper bounds; fadeStart reaches 170 ms; some quiet decay rates reach bounds.
  These are explicitly unresolved capacity/identifiability diagnostics, not
  grounds to blindly widen the ranges or claim exactness.
- Stochastic texture is not the recording's particular noise realization. Live
  retriggers advance the noise stream. No claim of sample-identical repeated hits.

## eseq port and musical surface

Files in eseq:

- `.claude/skills/identify-drum/acoustic-kick-dsp-template.lisp`
- `content/instruments/Drums/R8 Kick 03/{dsp,ui}.lisp`
- `content/instruments/Drums/R8 Kick 03.presets`
- `tools/audition/verify_r8_kick.py`
- `crates/sequencer/ui/capture-fixtures/r8-kick03.lisp`
- Only the R8 layout test in `crates/sequencer/src/ui/state_values/tests.rs` was
  intentionally edited; other concurrent changes in that file are not this task.

The coefficients are readable scalar macro arguments in the DSP. The UI no longer
shows a laboratory matrix: all 24 **musical** parameters appear exactly once.

| group | controls / behavior |
| --- | --- |
| Membrane | tune; lowest-mode weight; remaining head modes; natural decay; frequency-dependent damping; inharmonic spacing/stretch |
| Impact | bend depth/time; rise; independent finite length; early punch; velocity-to-brightness response |
| Contact | shell knock; shell transposition; shell/quiet-texture ring; struck beater level; felt-to-hard spectral tilt; contact duration |
| Print | quiet air; shell/texture keytracking; dry-at-zero drive; dark/bright tone; dry-at-zero quantization; level |

At C4 (261.63 Hz) and velocity 1, departures are neutral. Velocity is captured per
hit. Controls are smoothed with initialized history (no startup ramp). Phase is
wrapped/integrated using exact per-sample relaxation integrals, so pitch changes
do not multiply the entire elapsed phase. The small-x `1-exp(-x)` series avoids
float32 cancellation. Time is an integer sample counter, bounded after the longest
possible fade. Modes approaching Nyquist fade out; filter frequencies are clamped
below it. No tensors or compiler-fusion workarounds.

Presets: R8 Kick 03, Dry, Pillow Cut, Dusty Crate, Chest Hit, Hard Knocker,
Long Low, Basement 12, Paper Tight. Transposition is in `tune` only, not also in
`base_note_offset`. Chest Hit and Hard Knocker were trimmed after checking actual
44.1 kHz peaks, not just 48 kHz. These are starting voicings, not ear-approved finals.

## Verification

- Independent NumPy vs Torch reference before fit: ~3e-8 max absolute.
- Compiled shipping DSP vs normalized NumPy: **1.86e-5 max absolute**, zero samples
  above 1e-3; independent gate .3029 for both. `port-check.log`.
- `instrument_probe` through the real saved-factory compile/load/init path,
  MIDI 60, 44.1 kHz, 24,000 frames: peak .98296, RMS .10368, no non-finite samples
  or state. `host-probe.log`.
- `verify_r8_kick.py`: every control's endpoints demonstrably change the audio
  (dynamics tested below velocity 1; keytracking away from C4), finite endpoints,
  exact silence beyond length; all presets in bounds and with headroom at
  44.1/48/96 kHz; extreme retriggers/pitch automation finite and tail-silent;
  64/512-frame renders bit-identical. Results in `validation/validation.json`.
- Fusion checker: all compiled variants clean (zero scalar writes in tensor loops).
- Exact layout test passes: `state_values::tests::metal_seq_fx_lisp_lays_out_r8_kick03_controls`.
  It checks all 24 controls once, finite/nonzero geometry, and containment in the
  visible instrument panel on both axes. No cosmetic text assertions.
- Real production headless capture passed, PNG inspected; no standalone UI mock.
- No full package/workspace suite, no commits, no pushes, no stash.

## Reproduce

Python environment used: `/tmp/r8-identify-venv` with numpy, scipy, torch,
matplotlib. Recreate a venv if it is removed. From dgen:

```sh
python Examples/SynthID/scripts/fit_r8_acoustic.py \
  --start output/r8_acoustic_v7/recovered_params.json \
  --out output/r8_acoustic_v8 --iterations 1800
python Examples/SynthID/scripts/analysis/r8_acoustic_diagnostics.py output/r8_acoustic_v8
```

From eseq, with `DGEN_RUNTIME_INCLUDE` set to the fetched toolchain include dir
and `DGEN_BINARY_AUDIT_TOOL` set to dgen's `scripts/audit-dgen-dylib.sh`:

```sh
python tools/audition/synthid_port_check.py \
  --run ~/code/swift/dgen/output/r8_acoustic_v8 \
  --fit-module ~/code/swift/dgen/Examples/SynthID/scripts/fit_r8_acoustic.py \
  --template .claude/skills/identify-drum/acoustic-kick-dsp-template.lisp \
  --instrument 'content/instruments/Drums/R8 Kick 03'
python tools/audition/verify_r8_kick.py --out ~/code/swift/dgen/output/r8_acoustic_v8/validation
cargo nextest run -p sequencer --bin metal_seq \
  -E 'test(=state_values::tests::metal_seq_fx_lisp_lays_out_r8_kick03_controls)'
cargo run -p sequencer --bin metal_seq -- capture \
  --script crates/sequencer/ui/capture-fixtures/r8-kick03.lisp \
  --buffer fx --track 0 --width 1800 --height 500 --out /tmp/r8-kick03-panel.png
```

For the listening reel, from dgen with the same compiler environment:

```sh
python Examples/SynthID/scripts/analysis/r8_acoustic_ab.py \
  --eseq-root <absolute-eseq-root> \
  --old-instrument output/r8_acoustic_v8/old-instrument \
  --run output/r8_acoustic_v8
```

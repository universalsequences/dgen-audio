# R8 Kick03 — contact/diffuse-shell revision

2026-09-05. Bead `eseq-u13m`, **open pending the user's next listening verdict**.
The installed candidate is **`output/r8_contact_v3`**. Later experimental runs
v4–v7 are NOT shipping; see below. Nothing committed or pushed.

## User feedback and actual root causes

The user liked the new UI but rejected acoustic v8: RING still sounded synthetic
and comb-like, shaping felt weak, and Hard Knocker was hollow rather than hard.

The earlier safety tests all passed: changing every knob produced different
samples. That was insufficient. The actual flaws were:

1. RING prolonged ten clean stationary shell oscillators. Their strongest ten
   spectral bins held ~87% of the ring change's midrange energy.
2. DECAY/RING/CONTACT changed damping but ran into a shared 180 ms ROM cutoff.
   Extending an envelope could not extend the actual voice.
3. DAMP killed the membrane but not the shell/noise, leaving the synthetic ring
   more exposed and making the result hollow.
4. HARDNESS tilted/retuned texture without changing the strike mechanism.
5. KNOCK multiplied a quiet calibrated layer. PUNCH mostly applied gain to an
   attack whose lowest head mode still took ~23 ms to grow.

## Installed voice

`fit_r8_contact.py` and eseq's `contact-kick-dsp-template.lisp`:

- Keep seven low membrane modes and their independent rise times / two-rate
  tension relaxation. **Delete all ten stationary shell sine modes.**
- Add a finite half-sine force pulse, filtered by 180 Hz HP and 3.5 kHz LP,
  both Q .707. This is a struck pressure pulse, not a free-running or decaying
  shell oscillator. Reference width ~.96 ms; scalar amplitude ~.047.
- KNOCK directly controls that force amplitude, from zero to 3. The identified
  default is shown honestly near .05; moving it toward 1 really adds impact.
- Six broad contact/shell noise bands (350/.65, 700/.75, 1400/.85, 2600/1,
  4200/.8 Hz BP; 6500/.707 Hz HP). No high-Q resonance or comb feedback.
- Noise has a fixed 14 kHz exciter lowpass and sqrt(sr/48000) density scaling.
  Each band has separate struck and quiet-decay amplitudes/envelopes.
- RING controls only diffuse quiet-decay duration. With AIR=0, changing RING
  leaves the dry membrane and pressure impact **bit-identical** at neutral tone.
- Each layer scales its own cubic cutoff along with its decay: body uses
  LENGTH × DECAY; shell LENGTH × RING; contact LENGTH × CONTACT. LENGTH is the
  base cut, not a hidden global ceiling that defeats the other controls.
- After the latest layer cut, the tone-filter state gets a separate smooth
  20 ms exit. The source fade is not accidentally multiplied twice. The exact
  sample counter stops at 8 seconds, past every legal layer/exit duration.

Control revisions, preserving the same 24 positions and the liked UI:

- WEIGHT/HEAD use squared level ranges; BEND uses a squared depth range.
- DAMP applies frequency-dependent damping to membrane, contact and shell.
- PUNCH accelerates head energy transfer (up to 16× faster rise) plus a modest
  early body gain. It no longer just amplifies a slow-rising head.
- HARDNESS changes force duration and noise-contact duration. Felt reduces the
  exciter bandwidth; hard hits change the upper head excitation weighting.
  Shell band centers do **not** retune with hardness. An initial revision did
  that and pushed too much hard-strike energy above useful audible bands.
- BEATER has a 1.5-power gain range and AIR a squared range.
- TONE is a normalized bass/attack tilt around 420 Hz, rather than merely a
  1.4 kHz lowpass blend on a bass-dominated sound. Neutral remains dry.
- DRIVE reaches stronger saturation and CRUSH reaches 4-bit resolution; both
  remain exactly dry at zero. Strong boosts can exceed unity; use LEVEL.

This is still a compact analytical/stochastic synthesizer, not a full mechanical
simulation or a sample player. No target waveform, residual, learned FIR or
waveform table is embedded. The force-filter impulse used by the Torch fitter
is generated from the fixed IIR equations, never from the recording; the port
uses the actual biquads.

## Fitting / model selection

Target/provenance unchanged: `Assets/r8-kick03.wav`, Roland R8 `Kick03.wav`,
44.1 kHz / 8032 frames, SHA256
`26eb639f7d6587382cdc95d297626bef8fac396f5b5ff5fc8892fb805042278e`.

The new fitter reuses acoustic v8's phase-blind MR-STFT objective: low-band fine
magnitudes, pooled power, spectral contrast, and a small linear-mag term. No
waveform/slope loss. v1 optimizes 32 contact scalars with the measured membrane
fixed; v2 jointly refines the membrane; v3 relaxes the amplitude ceiling only
on the two delayed 1.4/2.6 kHz contact bands. All seed scalars, including frozen
ones, are now checked for finiteness and clamped to documented bounds.

| run | full objective | raw independent gate | decision |
| --- | ---: | ---: | --- |
| contact v1 | .444404 | .304107 | new topology / port shakedown |
| v2 | .443516 | .304539 | joint refinement; essentially stable |
| **v3** | **.443418** | **.304851** | selected articulation candidate |

Acoustic v8's old full objective was .397475. Removing line substitutes worsens
that objective, even though the audible complaint is precisely those lines.
Do not optimize them back into the voice to improve this number.

Midpoint gate for the new topology: .650431 -> .304851 (**53.13%**, NOT the
80% SynthID acceptance gate). Comparable compiled 240 ms renders, both at their
shipping defaults: old acoustic v8 .302931, new contact v3 .297125. The new
output headroom factor is .864444-ish. The gate delta is modest; the structural
and articulation changes, not that delta, are the reason for this revision.

### Unselected late-texture experiments

v4 tried a separate 250 Hz shell low cut; full objective .445372, raw gate
.305309. v5/v6 tried fitting quiet amplitudes/decays alone with late pooled
power (raw gates .300455 / .315485). v7 used longer late windows to reduce
leakage from the loud <300 Hz body (raw gate .298549). They did not establish
a meaningful improvement to justify the extra filter and fitting machinery;
the selected implementation remains the simpler, fully validated v3.

Those experiments are retained as research artifacts (`fit_source.py` per run,
v7 `template.lisp`), not shipped as half-finished layers. The canonical fitter
and template were returned to v3, with only seed validation added. Rechecked
canonical renderer vs shipping DSP after this selection: 2.63e-5 max absolute.

**Diagnostic caution:** a causal 3rd-order bandpass is not a brick wall. With a
very loud low body, its 400–800 Hz RMS can include attenuated lower modes.
Similarly short FFT windows leak the body into quiet midrange bands. Do not
add another ringing bank or boost noise solely to zero such a deficit table.
`r8_acoustic_diagnostics.py` now accepts `--old` so comparisons can explicitly
name the user's actual previous version rather than silently using the original
v6 from the first identification attempt.

The exact source attack remains imperfect. This revision fixes the reported
ring/control/preset problems structurally; it is **not a perfect-source-match
claim**, and listening is still required.

## Stronger validation than “the samples differ”

New `tools/audition/r8_articulation.py` measures gain-independent structure and
writes level-matched control demonstrations. Thresholds were not weakened to
pass the new voice. The old voice was run as a negative control and fails the
ring flatness assertion (.0109 vs required >.25).

| measurement / conditions | old | installed new |
| --- | ---: | ---: |
| Ring delta, top-10-bin share, 350–4500 Hz / 35–180 ms | .8717 | .2031 |
| Ring delta, within-band spectral flatness | .0109 | .5257 |
| Ring dry max difference, AIR=0 | .1603 | **0** |
| Isolated shell energy-duration ratio, RING .2 -> 4 | — | 18.53× |
| Isolated excitation brightness, HARDNESS -1 -> +1 | — | +12.96 dB |
| Full hit energy-duration ratio, DECAY .2 -> 4 | — | 24.93× |
| Full hit attack/tail change, DAMP 0 -> 1 | — | +45.17 dB |
| Full hit attack/tail change, PUNCH 0 -> 1 | — | +3.85 dB |
| Full hit low/mid balance, WEIGHT .2 -> 2 | — | +21.91 dB |
| Hard Knocker attack/tail ratio | 10.27 dB | 22.69 dB |
| Hard Knocker middle-90%-energy duration | 72.17 ms | 27.85 ms |
| Hard Knocker first-hit peak, 48 kHz | .8034 | .6987 |

Ring flatness/concentration also pass across five real retriggers that advance
the noise stream. They are not measurements of a single fortunate noise seed.
Isolation is explicitly documented: the hardness test mutes body/quiet air and
uses KNOCK=.8 / BEATER=.6; shell duration mutes body/force/struck noise. These
numbers do not pretend every full-mix control has the isolated-layer ratio.

Other checks passed:

- NumPy vs compiled defaults: max abs **2.63e-5**, 0 samples over 1e-3;
  independent gate .2971 for both.
- All 24 controls' endpoints finite, non-dead, and eventually silent.
- All nine presets within bounds and first-hit headroom at 44.1/48/96 kHz.
- Longest legal layer durations plus retrigger/pitch automation finite and
  tail-silent (test duration now derives from the actual layer bounds).
- 64/512-frame renders bit-identical, including retriggers.
- Fusion checks clean for every compiled variant.
- Actual saved-factory `instrument_probe` default, MIDI 60 / 44.1 kHz:
  peak .884129, RMS .100574, no non-finite output or state.
- Actual host Hard Knocker preset: peak .731563, RMS .040927, no non-finite
  output or state. This verifies the preset through the real storage/host path.
- Exact R8 layout test passes (one test; no full suite). Only its KNOCK default
  was changed in this pass; it still checks finite/nonzero controls once each,
  inside the panel on both axes.
- Production `metal_seq capture` passed; PNG opened/inspected. The UI source
  was deliberately left unchanged. KNOCK now honestly displays ~.05 by default.

## Presets / listening handoff

All nine were revoiced for the new meanings/ranges. Hard Knocker uses a shorter
hard contact, strong direct pressure knock, fast head transfer, shorter damped
body, and a short low-level diffuse tail. It is not “old hollow kick, louder.”
Dry also mutes the direct force; Long Low explicitly exploits the now-effective
decay extension. All transposition stays in `tune`, never duplicated in base note.

**Reload the factory instrument and reselect the preset.** An already-loaded
project can retain the old compiled/source instrument and old parameter values;
changing only a preset on that engine is not this new voice.

Under `output/r8_contact_v3/`:

- `articulation/hard-knocker-old-new.wav` — matched old/new Hard Knocker, ×3.
- `articulation/ring-old-new.wav` — matched old/new RING=4, ×3.
- `articulation/ring-short-long.wav`, `felt-wood.wav`, `decay-short-long.wav`,
  `open-muffled.wav`, `punch-neutral-full.wav`, `weight-light-heavy.wav`.
- `articulation/articulation.json` — exact measurements and A/B gains.
- `validation/presets.wav` and individual `preset-*.wav` files — shipping levels.
- `ab-target-old-new-level-matched.wav` — source / acoustic v8 / new default.
- `ab-target-old-new.wav`, `ab-report.json` — unmodified relative levels / gains.
- `ui.png`, `port-check.log`, `layout-test.log`, `host-default.log`,
  `host-hard-knocker.log`, `validation/run.log`, `negative-control.log`.
- `shipping-dsp.lisp`, `template.lisp` — exact selected voice snapshots.

The previous voice/presets are preserved under `output/r8_contact_before/`.

## Changed files / reproduction

eseq (this pass):

- `content/instruments/Drums/R8 Kick 03/dsp.lisp`
- `content/instruments/Drums/R8 Kick 03.presets`
- `.claude/skills/identify-drum/contact-kick-dsp-template.lisp` (new)
- `tools/audition/r8_articulation.py` (new)
- `tools/audition/verify_r8_kick.py` (legal-duration coverage)
- R8 KNOCK fixture value in `crates/sequencer/src/ui/state_values/tests.rs`

Other concurrent edits in either repo were preserved. No UI-source rewrite,
engine/scheduler workaround, commit, push, or stash in this pass.

Python environment: `/tmp/r8-identify-venv` (numpy, scipy, torch, matplotlib).
From dgen:

```sh
python Examples/SynthID/scripts/fit_r8_contact.py \
  --start output/r8_contact_v2/recovered_params.json \
  --out output/r8_contact_v3 --iterations 1600
```

From eseq, with the audition environment variables described in
`docs/instrument-audition-harness.md`:

```sh
python tools/audition/synthid_port_check.py \
  --run ~/code/swift/dgen/output/r8_contact_v3 \
  --fit-module ~/code/swift/dgen/Examples/SynthID/scripts/fit_r8_contact.py \
  --template .claude/skills/identify-drum/contact-kick-dsp-template.lisp \
  --instrument 'content/instruments/Drums/R8 Kick 03'
python tools/audition/verify_r8_kick.py \
  --out ~/code/swift/dgen/output/r8_contact_v3/validation
python tools/audition/r8_articulation.py \
  --old-instrument ~/code/swift/dgen/output/r8_contact_before/instrument \
  --old-presets ~/code/swift/dgen/output/r8_contact_before/presets.json \
  --out ~/code/swift/dgen/output/r8_contact_v3/articulation
cargo nextest run -p sequencer --bin metal_seq \
  -E 'test(=state_values::tests::metal_seq_fx_lisp_lays_out_r8_kick03_controls)'
```

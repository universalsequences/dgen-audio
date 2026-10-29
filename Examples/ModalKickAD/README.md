# ModalKickAD: the Latent Kick instruments

The kick counterpart of `../ModalSnareAD`. A learned 4-knob latent space is trained against
eseq's VILLAIN Kick and distilled into two realtime eseq instruments:

- **Latent Kick**: an MLP decoder (4 latent knobs + velocity -> 534 voice parameters) that runs once
  per hit. The voice is VILLAIN Kick's own kick-hit math with decoded per-kick parameters: a
  24-slot modal bank with VILLAIN's pitch-drop law and time warp, 5 transient sines, growl, burst,
  the body shaper, 18 fixed noise bands plus VILLAIN's hum, and the recording chain. VILLAIN's
  transform knobs, velocity curves and note tracking act on the decoded row, so they behave exactly
  as in VILLAIN Kick. Kick A, Kick B and a blend knob walk a straight line in latent space with a
  per-pair perceptual warp.
- **Latent Kick Morph**: the decoder re-runs every 128 samples while the kick morphs from a START to
  an END latent during the hit. The strike (transients, tick, attack) is fixed at the hit.

Unlike the snare, the template is VILLAIN Kick's voice itself: each training row is VILLAIN's
exact block-rate table for one render, so the decoder is trained in parameter space (numpy) and
only the noise is fitted. The Swift side (`swift build -c release --product ModalKickAD`) is the
trainer-side renderer: stem parity against VILLAIN, FD gradient checks, and optional Metal audio
stages, which did not improve on the parameter-space fit and are not used by the shipped
instrument.

## Swift sources

| File | What it is |
|---|---|
| `KickLayout.swift` | the decoder contract (segments of `$MODAL_AD_ROOT/kick/Latent Kick/decoder_layout.json`), teacher rows, the fixed noise tables, data paths |
| `KickVoice.swift` | the Latent Kick voice in DGenLazy, frame-parallel (mirrors `scripts/kick_np.py` `mode="trainer"`) |
| `main.swift` | `parity`, `fdcheck`, `train-audio`, and two profiling probes |
| `TrainAudio.swift` | `train-audio`: two-pass readout-only fine-tune on Metal against VILLAIN stems |

## scripts/

| File | What it does |
|---|---|
| `kick_common.py` | VILLAIN Kick loading through eseq's `tools/audition`, WAV IO, data paths |
| `make_stems_copy.py` | VILLAIN Kick copy whose kick-hit also returns 6 stem taps -> `kick/villain_stems/` |
| `render_kick_dataset.py` | the dataset: 33 kicks x 3 velocities, 15 transform knobs x 3 values x 3 kicks, pitches, 24 blends, 12 PC points (279 renders + stems) -> `kick/data/villain/` |
| `kick_table.py`, `extract_table.py` | VILLAIN Kick's block-rate section evaluated in Python -> `kick/data/kick_modes.json` |
| `villain_kick_np.py`, `_filt.c` | numpy twin of one VILLAIN Kick hit (C lfilter built on first use into `$MODAL_AD_ROOT/kick`) |
| `kick_layout.py` | the decoder contract: output segments, squashes, ranges, noise bands |
| `kick_np.py` | numpy twin of the Latent Kick voice (`mode="full"` = the instrument, `mode="trainer"` = the Swift renderer) |
| `targets.py` | teacher rows: VILLAIN block-rate values -> decoder-space u -> `kick/data/teacher.json` |
| `export_tables.py` | the fixed band-noise and hum signals the Swift trainer reads -> `kick/data/tables/` |
| `train_np.py` | parameter-space autodecoder pretrain (numpy) |
| `eval_run.py`, `kick_metric.py` | reconstruction vs VILLAIN (band-envelope dB, MR-STFT, cents, peak ratio) |
| `ablate.py` | which VILLAIN Kick stages matter, and the nearest-other-kick distance used to normalise scores -> `kick/data/ablations.json` |
| `knobspace.py` | re-express the codes as PCA-rotated knobs (z1 = largest axis), decoder output unchanged |
| `warp.py`, `blend_eval.py` | the per-pair perceptual blend warp (E6 method); latent blend vs VILLAIN's own blend |
| `gen_dsp.py` | the realtime instrument generator (dsp.lisp + decoder_layout.json) |
| `build_instrument.py`, `ui_template.lisp` | the installed **Latent Kick**: A/B/blend front end, weights, panel, family map, presets |
| `parity_np.py`, `parity_template.py`, `swift_parity.py`, `instrument_parity.py`, `installed_parity.py` | the parity chain: twin vs VILLAIN stems, template vs VILLAIN, Swift vs twin, compiled instrument vs twin |
| `prep_audio.py` | inputs for the optional `train-audio` stages |
| `morph/` | the **Latent Kick Morph**: `gen_morph.py` + `voice_mid.tmpl` (dsp), `build_stage.py` + `ui.tmpl.lisp` (stage/install), `survey.py`, `survey_pairs.py`, `presets.py` (preset picking), `make_refs.py` and `t_*.py` (checks), `bench.py` (CPU), `lib.py` (helpers) |

The kick scripts also import from the snare example: `ModalSnareAD/scripts/modal_ad.py` (paths),
`e1/metric.py` (MR-STFT), `e3proto/bench.py` (CPU bench) and `e4/build_family.py` (map envelopes).

## The data directory

As for the snare, everything the scripts produce goes to `$MODAL_AD_ROOT`, here under
`$MODAL_AD_ROOT/kick` (`data/`, `runs/`, `stage/`, `morph/`, `villain_stems/`). Paths such as
`kick/data/teacher.json` in docstrings are relative to `$MODAL_AD_ROOT`. Run paths given on the
command line (`runs/...`) are relative to the current directory, so run the commands from
`$MODAL_AD_ROOT/kick` as shown. `ModalKickAD` requires `MODAL_AD_ROOT` and exits with a message if
it is unset.

## Requirements and setup

As in `../ModalSnareAD/README.md`: macOS on Apple Silicon, Python 3 with numpy, and an eseq checkout
(`ESEQ_ROOT`) with its compiler and toolchain fetched. Then:

```bash
cd <this dgen checkout>
export MODAL_AD_ROOT=~/modal-ad          # can be shared with the snare pipeline
export ESEQ_ROOT=<eseq checkout>
K=$PWD/Examples/ModalKickAD/scripts
KB=$PWD/.build/release/ModalKickAD
export DGEN_RUNTIME_INCLUDE=$PWD/toolchain/include DGEN_BINARY_AUDIT_TOOL=$PWD/scripts/audit-dgen-dylib.sh
swift build -c release --product ModalKickAD
Examples/ModalSnareAD/scripts/make_dgenlisp_tool.sh   # -> $MODAL_AD_ROOT/tools/dgenlisp/DGenLisp
export ESEQ_DGENLISP_TOOL=$MODAL_AD_ROOT/tools/dgenlisp/DGenLisp
mkdir -p $MODAL_AD_ROOT/kick && cd $MODAL_AD_ROOT/kick
```

## Pipeline: from nothing to the installed instruments

### K0: dataset, exact table, template

```bash
python3 $K/make_stems_copy.py                # kick/villain_stems/
python3 $K/render_kick_dataset.py            # kick/data/villain/ (279 renders + stems, manifest.json)
python3 $K/extract_table.py                  # kick/data/kick_modes.json, kick/data/kick_stats.json
python3 $K/parity_np.py                      # numpy twin vs VILLAIN stems (output median 5.5e-6)
python3 $K/ablate.py                         # kick/data/ablations.json (stage ablations + nearest-other-kick floor the metrics normalise by)
python3 $K/gen_dsp.py --random-weights       # kick/Latent Kick/ (dsp.lisp, decoder_layout.json the trainer reads)
python3 $K/targets.py                        # kick/data/teacher.json
python3 $K/parity_template.py                # template vs VILLAIN (modal exact, band median 0.22 dB)
python3 $K/export_tables.py                  # kick/data/tables/ (bands.f32, hum.f32, tables.json)
```

### K1: trainer-side renderer checks (Swift)

```bash
IDS=kick_00_v100,kick_03_v100,kick_11_v100,kick_11_v040,kick_00_v070,sweep_11_tune_0.5,sweep_11_drop_time_3,sweep_11_tilt_1.5,sweep_11_spread_1.5,sweep_11_beat_3,sweep_00_decay_0.3,sweep_00_bend_2,sweep_00_attack_2.5,sweep_11_body_0.5,sweep_11_knock_2.5,sweep_11_tick_2.5,sweep_00_shape_2.5,sweep_11_growl_2.5,sweep_11_level_1.5,sweep_11_pitch_C5,sweep_00_pitch_C3
$KB parity --ids $IDS --backend c            # kick/data/swift_parity/<id>.<tap>.f32
python3 $K/swift_parity.py ${IDS//,/ }       # Swift vs twin: out median ~1e-5 relative RMS
$KB fdcheck --backend c                      # kick/data/fdcheck_c.json (optional)
```

### Training, export and the installed Latent Kick

```bash
python3 $K/train_np.py --out runs/pre_d4b --steps 20000      # kick/runs/pre_d4b (weights.json, codes.json)
python3 $K/eval_run.py runs/pre_d4b                          # band median 0.24 dB, 0.2 cents vs VILLAIN
python3 $K/knobspace.py runs/pre_d4b runs/final_d4           # kick/runs/final_d4 (knob-space codes)
python3 $K/warp.py runs/final_d4 runs/final_d4/warp.json --workers 7
python3 $K/blend_eval.py runs/final_d4                       # optional: latent blend vs VILLAIN's blend
python3 $K/build_instrument.py runs/final_d4 "stage/Latent Kick"   # kick/stage/Latent Kick/ + Latent Kick.presets
python3 $K/installed_parity.py runs/final_d4 "stage/Latent Kick"   # compiled vs twin (5e-5 median)
python3 $K/instrument_parity.py              # optional: pinned teacher rows and random weights, kick/data/instrument_parity.json
```

`kick/stage/Latent Kick/` and `kick/stage/Latent Kick.presets` are the instrument. Copy both into
the `Research/` folder of your eseq user instruments (`eseq paths` prints `user-instruments`) and
run `eseq instrument check "Research/Latent Kick"` with `ESEQ_DGENLISP_TOOL` exported.

The Metal audio stages were tried and not kept (noise MR 0.13 -> 0.10-0.12 but a worse band
maximum; the modal stage was a wash):

```bash
python3 $K/prep_audio.py runs/pre_d4b        # kick/data/tex_targets/, runs/pre_d4b/anchor.json
$KB train-audio --run runs/pre_d4b --out runs/noise2 --stage noise --steps 90 --group 11 --lr 1e-4 --anchor 30 --anchor-end 10
$KB train-audio --run runs/pre_d4b --out runs/modal1 --stage modal --steps 90 --group 11 --lr 1e-4 --anchor 30 --anchor-end 10
```

### Latent Kick Morph

```bash
python3 $K/morph/build_stage.py              # kick/morph/Latent Kick Morph/ (from kick/stage/Latent Kick)
python3 $K/morph/survey.py                   # kick/morph/survey_singles.json
python3 $K/morph/survey_pairs.py             # kick/morph/survey_pairs.json (morph quality of every pair)
python3 $K/morph/presets.py                  # kick/morph/Latent Kick Morph.presets, kick/morph/renders/
python3 $K/morph/build_stage.py --install "<eseq user-instruments>/Research"
python3 $K/morph/make_refs.py                # kick/morph/build/ (check builds)
python3 $K/morph/t_parity.py                 # START = END vs the Latent Kick: median 7e-7
python3 $K/morph/t_click.py                  # hop-boundary clicks vs a build that decodes every sample
python3 $K/morph/t_midhit.py                 # END knob and mod changes during a hit
python3 $K/morph/t_safety.py                 # every pair x 3 morph times: no blow-ups
python3 $K/morph/t_presets.py                # feature tracks of the preset renders
python3 $K/morph/bench.py                    # CPU vs the Latent Kick and the Latent Snare Morph
```

The morph scripts work in `$MODAL_AD_ROOT/kick/morph` whatever the current directory; `t_parity.py`
takes an optional build directory relative to it. `bench.py` also times `e8/Latent Snare Morph`, so it
needs the snare pipeline's E8 stage.

## Reproducibility

`build_instrument.py` and `morph/build_stage.py` are deterministic: on the trained run they
reproduce the shipped Latent Kick and Latent Kick Morph exactly, apart from comments. `train_np.py`
is CPU numpy; trained weights are not checked in.

## Not checked in

All audio (dataset renders, stems, texture targets, demo renders), the noise tables
(`kick/data/tables/*.f32`), teacher rows, runs and weights, and staged instruments. They are
regenerated by the steps above.

# TrainKick808 Experiments

Target:

`/Users/alecresende/code/learning/anthropic/eseq/crates/sequencer/samples/manufacturers/EMU/Kicky Fdz.wav`

Current renderer:

```sh
python3 Examples/TrainKick808/waveform_compare.py \
  --learned /tmp/dgen-train-kick808-click380/learned.wav \
  --target /tmp/dgen-train-kick808-click380/target.wav \
  --out /tmp/dgen-train-kick808-click380/compare.png
```

Best current render:

```sh
python3 Examples/TrainKick808/waveform_compare.py \
  --learned /tmp/dgen-train-kick808-fullres4/learned.wav \
  --target /tmp/dgen-train-kick808-fullres4/target.wav \
  --out /tmp/dgen-train-kick808-fullres4/compare.png
```

## Observations

The visual comparison is catching the important failure mode: spectral loss can improve
while the learned patch still looks like the wrong instrument. The target has a dense,
high-crossing transient in the first 30 ms; the early simple families produced only a
few smooth crossings.

Latest metric from `waveform_compare.py`:

| Branch | Output dir | Norm MSE | 30 ms MSE | Learned crossings | Target crossings |
| --- | --- | ---: | ---: | ---: | ---: |
| Smooth body + phase | `/tmp/dgen-train-kick808-family3-best` | 0.512692 | 0.743186 | 4 | 22 |
| Learnable envelopes | `/tmp/dgen-train-kick808-env1` | 0.542059 | 0.839098 | 4 | 22 |
| Independent front mode | `/tmp/dgen-train-kick808-front1` | 0.445888 | 0.590724 | 4 | 22 |
| Faster front mode | `/tmp/dgen-train-kick808-front2` | 0.453764 | 0.574459 | 6 | 22 |
| 380 Hz click/front bias | `/tmp/dgen-train-kick808-click380` | 0.442623 | 0.553274 | 8 | 22 |
| Modal bank | `/tmp/dgen-train-kick808-modal1` | 0.445886 | 0.560003 | 7 | 22 |
| Learnable frequency curve | `/tmp/dgen-train-kick808-freqcurve1` | 0.748728 | 1.056729 | 11 | 22 |
| Zero-crossing curve init | `/tmp/dgen-train-kick808-zc-init` | 0.472490 | 0.693389 | 14 | 22 |
| Zero-crossing curve staged | `/tmp/dgen-train-kick808-zc-stage1` | 0.454722 | 0.693554 | 14 | 22 |
| Learnable residual, 80 epochs | `/tmp/dgen-train-kick808-residual1` | 0.304721 | 0.266747 | 18 | 22 |
| Learnable residual, checkpointed | `/tmp/dgen-train-kick808-residual2b` | 0.274638 | 0.186952 | 24 | 22 |
| Resume residual + amp | `/tmp/dgen-train-kick808-residual3` | 0.257554 | 0.146055 | 30 | 22 |
| Resume amp, residual frozen | `/tmp/dgen-train-kick808-residual4-freeze` | 0.268433 | 0.179831 | 28 | 22 |
| Shorter residual | `/tmp/dgen-train-kick808-residual-short1` | 0.321243 | 0.296381 | 18 | 22 |
| Full residual upper bound | `/tmp/dgen-train-kick808-fullres1` | 0.074191 | 0.146768 | 26 | 22 |
| Full residual resume | `/tmp/dgen-train-kick808-fullres2` | 0.029806 | 0.069056 | 22 | 22 |
| Full residual refine | `/tmp/dgen-train-kick808-fullres3` | 0.017709 | 0.044848 | 22 | 22 |
| Full residual refine 2 | `/tmp/dgen-train-kick808-fullres4` | 0.012491 | 0.033353 | 22 | 22 |
| 512-point residual | `/tmp/dgen-train-kick808-residual512-1` | 0.076521 | 0.151156 | 16 | 22 |
| 512-point residual resume | `/tmp/dgen-train-kick808-residual512-2` | 0.026280 | 0.062324 | 8 | 22 |
| 1024-point residual | `/tmp/dgen-train-kick808-residual1024-1` | 0.086593 | 0.168664 | 16 | 22 |
| 1024-point residual resume | `/tmp/dgen-train-kick808-residual1024-2` | 0.027502 | 0.064973 | 10 | 22 |
| 1024-point residual + slope loss | `/tmp/dgen-train-kick808-residual1024-slope1` | 0.015948 | 0.041225 | 12 | 22 |
| 1024-point residual + slope/freq loss | `/tmp/dgen-train-kick808-residual1024-freqslope1` | 0.011928 | 0.032246 | 12 | 22 |

## Current Diagnosis

The model needs more transient topology, not just better amplitude envelopes. The
target front is not explained by a single body oscillator plus amplitude wobble. A
separate front component helps, but the oscillator body still needs a better way to
match the target's pitch and phase contour after the first few cycles.

The learnable residual branch is the first branch that makes the image visibly move
toward the target. It is also a useful warning: letting the residual keep training
can lower loss while making the attack too busy. The checkpointed 1536-frame
residual run currently looks like the best proof of concept; the resumed amp run
has better scalar metrics but overshoots the target crossing count.

The full-length residual family is intentionally sample-like, but it is a strong
upper-bound result for the training loop. It proves the loss, renderer, checkpoint,
and optimizer path can drive the generated waveform very close to the target:
`fullres4` reaches 0.012491 normalized MSE with matched 30 ms zero crossings. The
next honest synth step is to compress that learned residual into a lower-dimensional
transient model rather than ship a 4096-sample correction table.

The `--residual-points` option separates residual duration from residual parameter
count. A 512- or 1024-point residual stretched over the full 4096-frame target can
match the body and tail well, but currently smooths away the highest-frequency
attack motion. Adding `--slope-weight` improves normalized MSE and makes the visual
render closer, but it still undercounts early zero crossings. The next patch-family
step should add an explicit attack oscillator/noise/exciter path rather than asking
the compressed residual to learn all transient topology.

The most useful next branch is likely one of:

- A learnable frequency curve improves early crossing count, but the current
  initialization is too front-loaded and gives worse MSE. It probably needs staged
  fitting: phase/frequency over the first 30 ms first, then body decay.
- A zero-crossing-derived frequency curve gets closer on crossing count than the
  guessed exponential curve, but it produces too many equal-amplitude early cycles
  and still misses the target phase/envelope. Curve-only staged training improves
  its scalar loss but does not materially improve the visual match.
- A short bank of 3-5 independently phased damped modes between roughly 250 and
  900 Hz was tried with tensorized phasors; it was slower than expected and did
  not improve the visual match.
- A deterministic noise/exciter component with a trainable decay and tone filter, if
  the loss emphasizes spectral/transient shape more than exact sample phase.
- A staged optimizer schedule: first fit phase/frequency/transient crossings with
  waveform-heavy loss over the first 30 ms, then fit body decay with spectral loss.
- A slope/first-difference loss is available as `--slope-weight`. It is useful for
  the visual shape but not sufficient by itself to recover the missing early cycles.
- A checkpoint/resume flow is now in the example. By default each run writes
  `checkpoint.json`, and `--checkpoint-in <json>` resumes matching scalar and
  tensor params.

## Runtime Notes

- Basic smooth family: roughly 12 ms per epoch at 4096 frames/window 512.
- Envelope/front family: roughly 48 ms per epoch with envelope tensors.
- Independent front mode branch currently compiles/runs around 296 ms per epoch,
  so it needs pruning or staging before long sweeps.
- Tensorized modal bank branch ran around 437 ms per epoch and is not worth
  continuing in its current form.
- Residual branch runs around 49 ms per epoch at 4096 frames/window 512, so a
  220-epoch fit takes roughly 11 seconds plus warmup/build time.
- Full residual runs around 50 ms per epoch at 4096 frames/window 512. The staged
  `fullres1` -> `fullres4` sequence took under a minute of training time and moved
  normalized MSE from 0.074191 to 0.012491.
- Slope-loss runs are slightly slower, around 52 ms per epoch in this example.

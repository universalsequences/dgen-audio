# SynthID Rung 3 Blocker

## What we are stuck on

The retained model improves the independent MR-STFT distance by **68.94%**, but
Rung 3 requires **80%**. Oracle splices now localize the entire gap to the
attack: replacing only the first `100 ms` of the learned render with the target
scores `81.87%`, and replacing `200 ms` scores `86.10%`. The sustain and tail
are therefore good enough.

Inside the first `200 ms`, the 2048-window residual is concentrated in the
`45–350 Hz` sweep region and below `20 Hz`. High-passing target, initialization,
and learned audio at `30 Hz` lifts the retained score to `72.94%`, showing about
four points are tied to the sub-bass/DC component.

## Why this is difficult

The remaining mismatch is localized but is still not attributable to one safe
model parameter. A new high-resolution attack ridge used a zero-phase
`25–350 Hz` band and Hilbert instantaneous frequency, then fitted the pitch
contour directly on the CPU. It selected a single exponential (`fStart=80.0`,
`fEnd=48.991`, `pitchDecay=-45.0`) rather than earning the optional curvature
term. Freezing that direct fit in the matched `120+60` pilot regressed badly:
`30.43%` improvement and `0.029485` learned distance, versus the established
pilot's `61.01%` and `0.020554`. No full run was justified, and the candidate
code was removed.

Pitch curvature, body asymmetry, a fixed second harmonic, phase, loss-scale
normalization, and now direct CPU pitch fitting have all failed pilot or full
validation. The multi-window L1 loss is jagged for fine pitch/phase changes,
and a contour that looks plausible in isolation can break the attack phase
alignment that the joint optimizer finds.

The core pipeline is not the blocker: preprocessing, autograd, the independent
comparator, five-restart training, and Rung 2 renderer equivalence are working.

## Next step

Determine whether the target's sub-`20 Hz` component is recording rumble/DC
drift or intentional 808 waveform asymmetry. Inspect and audition a `25 Hz`
low-pass copy, and measure its time-domain shape and correlation with the body.
If it is capture noise, make an explicit preprocessing/comparator high-pass spec
change and re-run the gate. If it is coherent with the 808 body, model it as a
target-independent waveform term. Do not resume whole-sound parameter tuning.

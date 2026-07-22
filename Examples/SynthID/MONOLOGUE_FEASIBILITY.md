# Feasibility: SynthID Method on Monologue Bass Samples

**Question:** Can the three-rung SynthID method (see `output/pdf/synthid_three_rung_paper.tex`, `SPEC.md`) recreate a monophonic bass synth sample from a Korg Monologue? Is this a drum method or a general synth method?

**Answer:** The method transfers — it's a synth method that happened to be validated on a drum. But the current SynthID patch won't do it as-is, because the patch *is* the model, and the existing one is a drum topology.

## Why the method fits a Monologue bass note

- **Same problem shape.** System identification of a low-dimensional synth from one recording: known topology (2 VCOs with shape, 2-pole filter, drive, EG, LFO), unknown knob settings. Output is a patch sheet of interpretable scalars you can dial into hardware or a plugin — the "14 interpretable scalars, no residual" framing carries over directly.
- **Easier than the 808 in key ways.**
  - Pitch is stable — no swept-phasor gradient machinery (one of the nastiest Rung 1 bugs).
  - The note is longer, so MR-STFT windows see plenty of stationary signal.
  - Pitch extraction is trivial (monophonic, constant).
- **Loss is well suited.** The phase-blind multi-resolution STFT loss handles harmonically rich sustained tones well; filter cutoff and resonance appear as smooth spectral-envelope changes with good gradients. DGen's biquad already has working BPTT.
- **Rung 3 capture-chain lessons apply unchanged.** Symmetric high-pass / capture-model discipline for interface coloration, compression, etc.

## What does NOT transfer

- **Oscillators.** DGen's `phasor` is a 0–1 ramp (the kick voice wraps it in `sin` for its sine body), so a naive saw is just `2·phasor−1` and a pulse is the difference of two phase-offset saws — no new op required for the naive case. What's missing is band-limiting and a differentiable shape path: naive saw/square alias, and the shape parameter needs a smooth blend. Standard fixes:
  - Additive/harmonic oscillator (differentiable for free, DDSP-style), or
  - PolyBLEP with a differentiable shape blend.

  Either is a new op (or two), each needing a Rung-2-style independent-renderer equivalence check.
- **Time-varying character.** The interesting part of a Monologue bass is often EG sweeping the cutoff and drive interacting with resonance — more parameters and more identifiability traps than the kick (cutoff-vs-osc-mix, drive-vs-level degeneracies, analogous to the drive×amplitude product issue from Rung 1). Score invariant products, as before.
- **Sync / ring mod.** Oscillator sync is a discontinuity that is genuinely painful to differentiate through. A plain saw-through-filter bass should work well; a sync-lead-style patch is a research problem, not a port.

## Drum method or synth method?

Nothing in the ladder logic (self-inversion → independent renderer → real audio) or the loss is percussion-specific. Drums were arguably the *hard* case: the transient-dominated first 100–200 ms is where the remaining 808/909 Rung 3 residual lives. A sustained bass note moves most signal energy into the regime where the spectral loss works best.

**Caveat:** the 84.55% result is one recording of one instrument whose topology the patch was designed around. The Monologue requires a new differentiable patch: 2 osc + shape (+ optional sync/ring) + 2-pole filter + EG + drive.

## Suggested path ("Rung 0.5" first)

1. **New op:** additive band-limited saw oscillator with a shape parameter; finite-difference gradient check.
2. **Rung 1 self-inversion** on a minimal saw + biquad + EG patch (hidden DGen-generated target).
3. **Rung 2** independent NumPy renderer for the new oscillator; float32 equivalence gate.
4. **Rung 3** on the real Monologue sample, with the same symmetric capture preprocessing and independent CPU comparator.

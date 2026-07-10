# SynthID rung 3 targets

Place real TR-808 kick WAVs here for `swift run SynthID rung3`, or pass a WAV
from elsewhere in the repository with `--target`.

Preferred source format: mono WAV, 44.1 kHz or higher, trimmed to onset, shorter
than 1 second. The Rung 3 preprocessor mixes supported WAVs to mono, aligns the
onset, uses windowed-sinc resampling when needed, and crops or pads a copy to the
configured training length. The original file is never modified.

The repository's `Assets/808kicklong.wav` is the first development target. It is
a mono 32.5 kHz recording, so its run must retain `preprocessing.json` to document
the conversion to 44.1 kHz.

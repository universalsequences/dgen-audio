#!/usr/bin/env bash
# Build the C4-C7 chromatic TinySOL flute ordinario dataset at pp/mf/ff.
set -euo pipefail

ROOT="${TINYSOL_ROOT:-datasets/tinysol}"
SOURCE="$ROOT/Winds/Flute/ordinario"
STAGING="${STAGING:-.flute_multidynamics_wavs}"
CACHE="${CACHE:-.ddsp_cache_flute_multidynamics}"
BIN="${BIN:-./.build/debug/DDSPE2E}"

rm -rf "$STAGING" "$CACHE"
mkdir -p "$STAGING"

python3 - "$SOURCE" "$STAGING" <<'PY'
from pathlib import Path
import re
import sys

source, staging = map(Path, sys.argv[1:])
note_pc = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
           "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
pattern = re.compile(r"^Fl-ord-([A-G]#?)(\d)-(pp|mf|ff)-.*\.wav$")
selected = {}
for path in sorted(source.glob("*.wav")):
    match = pattern.match(path.name)
    if not match:
        continue
    note, octave, dynamic = match.groups()
    midi = 12 * (int(octave) + 1) + note_pc[note]
    if 60 <= midi <= 96:  # C4 through C7, inclusive
        key = (midi, dynamic)
        if key in selected:
            raise SystemExit(f"duplicate TinySOL flute sample for {key}: {path}")
        selected[key] = path

expected = {(midi, dynamic) for midi in range(60, 97)
            for dynamic in ("pp", "mf", "ff")}
missing = sorted(expected - selected.keys())
if missing:
    raise SystemExit(f"missing {len(missing)} required pitch/dynamic samples: {missing}")

for path in selected.values():
    (staging / path.name).symlink_to(path.resolve())
print(f"Staged {len(selected)} clips (37 pitches x 3 dynamics) in {staging}")
PY

if [[ ! -x "$BIN" ]]; then
  swift build -c debug --product DDSPE2E
fi

# Use one gain for the whole dataset: per-file peak normalization would erase
# pp/mf/ff differences, while leaving TinySOL's very low raw levels unchanged
# would invalidate the R6 loss epsilon/scale. Keep each source recording in one
# split so overlapping chunks cannot leak into validation.
"$BIN" preprocess \
  --input "$STAGING" \
  --cache "$CACHE" \
  --normalize-to 0.99 \
  --global-normalize true \
  --max-f0 2400 \
  --split-by-source true \
  --shuffle true \
  --seed "${SEED:-1337}" \
  --train-split "${TRAIN_SPLIT:-0.9}"

"$BIN" inspect-cache --cache "$CACHE" --limit 3

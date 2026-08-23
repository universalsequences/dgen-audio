#!/usr/bin/env bash
# Build the first instrument-conditioned proof set: flute + Bb clarinet,
# matched over C4-G6 chromatic at pp/mf/ff.
set -euo pipefail

ROOT="${TINYSOL_ROOT:-datasets/tinysol}"
STAGING="${STAGING:-.flute_clarinet_wavs}"
CACHE="${CACHE:-.ddsp_cache_flute_clarinet}"
BIN="${BIN:-./.build/debug/DDSPE2E}"
REFERENCE_FEATURES="${REFERENCE_FEATURES:-64}"
DYNAMICS="${DYNAMICS:-pp,mf,ff}"

rm -rf "$STAGING" "$CACHE"
mkdir -p "$STAGING/flute" "$STAGING/clarinet"

python3 - "$ROOT" "$STAGING" "$DYNAMICS" <<'PY'
from pathlib import Path
import re, sys
root, staging = map(Path, sys.argv[1:3])
dynamics = tuple(sys.argv[3].split(","))
sources = {
    "flute": root / "Winds/Flute/ordinario",
    "clarinet": root / "Winds/Clarinet_Bb/ordinario",
}
pc = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,"F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}
pattern = re.compile(r"^[^-]+-ord-([A-G]#?)(\d)-(pp|mf|ff)-.*\.wav$")
for label, source in sources.items():
    selected = {}
    for path in sorted(source.glob("*.wav")):
        match = pattern.match(path.name)
        if not match: continue
        note, octave, dynamic = match.groups()
        midi = 12 * (int(octave) + 1) + pc[note]
        if 60 <= midi <= 91 and dynamic in dynamics:  # C4 through G6
            key = (midi, dynamic)
            if key in selected:
                raise SystemExit(f"duplicate {label} sample for {key}: {path}")
            selected[key] = path
    expected = {(midi, dynamic) for midi in range(60, 92) for dynamic in dynamics}
    missing = sorted(expected - selected.keys())
    if missing: raise SystemExit(f"{label} is missing {len(missing)} samples: {missing}")
    for path in selected.values():
        (staging / label / path.name).symlink_to(path.resolve())
    print(f"Staged {len(selected)} {label} clips")
PY

if [[ ! -x "$BIN" ]]; then swift build -c debug --product DDSPE2E; fi

"$BIN" preprocess \
  --input "$STAGING" --cache "$CACHE" \
  --label-by-top-level true --reference-features "$REFERENCE_FEATURES" \
  --normalize-to 0.99 --global-normalize true \
  --max-f0 2400 --split-by-source true \
  --shuffle true --seed "${SEED:-1337}" --train-split "${TRAIN_SPLIT:-0.9}"

"$BIN" inspect-cache --cache "$CACHE" --limit 3

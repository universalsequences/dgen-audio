#!/bin/bash
# Refine basin-search elites with the audit's best-performing schedule:
# 600 smooth epochs from each elite, then 800 production epochs from the
# smooth result. Selection uses production finalLoss only; the gate ratio
# is computed against the seed's canonical COLD baseline (never the elite's
# own start).
#
# Usage: refine_elites.sh <search-out-dir> <target.wav> <seed> <cold-baseline-loss>
set -euo pipefail

SEARCH_DIR="$1"
TARGET="$2"
SEED="$3"
BASELINE="$4"
BIN=".build/release/SynthID"

for elite in "$SEARCH_DIR"/elites/elite-*.json; do
  k="$(basename "$elite" .json)"
  refdir="$SEARCH_DIR/refine/$k"
  echo "=== $k: smooth 600 ==="
  "$BIN" train --profile subtractive-bass --seed "$SEED" \
    --target "$TARGET" --out "$refdir/smooth" \
    --epochs 600 --restarts 1 --smooth-training-loss \
    --initial-params "$elite" \
    --log-every 200 --checkpoint-every 300
  echo "=== $k: production 800 ==="
  "$BIN" train --profile subtractive-bass --seed "$SEED" \
    --target "$TARGET" --out "$refdir/production" \
    --epochs 800 --restarts 1 \
    --initial-params "$refdir/smooth/recovered_params.json" \
    --log-every 200 --checkpoint-every 400
done

echo
echo "=== summary (gate: finalLoss / coldBaseline <= 0.02) ==="
python3 - "$SEARCH_DIR" "$BASELINE" <<'EOF'
import glob, json, sys
search_dir, baseline = sys.argv[1], float(sys.argv[2])
rows = []
for path in sorted(glob.glob(f"{search_dir}/refine/elite-*/production/report.json")):
    name = path.split("/")[-3]
    final = json.load(open(path))["finalLoss"]
    rows.append((final, name))
rows.sort()
for final, name in rows:
    ratio = final / baseline
    gate = "PASS" if ratio <= 0.02 else "FAIL"
    print(f"{name}: finalLoss={final:.6f} ratio={ratio:.4%} {gate}")
if rows:
    best, name = rows[0]
    print(f"\nbest: {name} finalLoss={best:.6f} ratio={best/baseline:.4%}")
EOF

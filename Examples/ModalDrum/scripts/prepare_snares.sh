#!/usr/bin/env bash
# Prepare a deterministic 44.1 kHz mono snare one-shot corpus and manifest.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 INPUT_WAV_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/prepare_snares.py" "$1" "$2"

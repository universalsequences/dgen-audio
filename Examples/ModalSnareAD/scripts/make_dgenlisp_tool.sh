#!/bin/bash
# Assemble this checkout's DGenLisp as the compiler distribution eseq's tools/audition expects
# (binary + scripts/audit-dgen-dylib.sh + toolchain/{abi,include} side by side) in
# $MODAL_AD_ROOT/tools/dgenlisp. The Latent instruments need it: the compiler eseq pins predates
# the dgen fixes they rely on. modal_ad.py finds it there unless ESEQ_DGENLISP_TOOL is set.
set -e
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${MODAL_AD_ROOT:?set MODAL_AD_ROOT}/tools/dgenlisp"
(cd "$REPO" && swift build -c release --product DGenLisp)
rm -rf "$OUT"
mkdir -p "$OUT/scripts" "$OUT/toolchain"
cp "$REPO/.build/release/DGenLisp" "$OUT/"
cp "$REPO/scripts/audit-dgen-dylib.sh" "$OUT/scripts/"
cp -R "$REPO/toolchain/abi" "$REPO/toolchain/include" "$OUT/toolchain/"
echo "$OUT/DGenLisp"

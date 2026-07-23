#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output_root=${DGEN_VECTOR_BENCH_ROOT:-"${repo_root}/.toolchain/vector-math"}
mkdir -p "${output_root}"

/usr/bin/clang \
  -target arm64-apple-macos11.0 \
  -O3 \
  -ffast-math \
  -fno-math-errno \
  -fno-trapping-math \
  -ffp-contract=fast \
  -mcpu=apple-m1 \
  -framework Accelerate \
  "${repo_root}/toolchain/benchmarks/vector_math_benchmark.c" \
  -o "${output_root}/vector-math-benchmark"

"${output_root}/vector-math-benchmark" |
  tee "${output_root}/vector-math-results.csv"

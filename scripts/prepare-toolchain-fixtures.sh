#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
input_dir=${1:-"${repo_root}/toolchain/generated/current"}
output_dir=${2:-"${repo_root}/toolchain/generated/hermetic"}

mkdir -p "${output_dir}"

for input_path in "${input_dir}"/*.c; do
  fixture_name=$(basename "${input_path}")
  output_path="${output_dir}/${fixture_name}"
  if ! grep -q '^#include "dgen_runtime.h"$' "${input_path}"; then
    echo "error: generated fixture does not use dgen_runtime.h: ${input_path}" >&2
    exit 1
  fi
  cp "${input_path}" "${output_path}"
done

echo "Copied generated runtime-ABI fixtures into ${output_dir}"

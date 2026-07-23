#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
input_dir=${1:-"${repo_root}/toolchain/generated/current"}
output_dir=${2:-"${repo_root}/toolchain/generated/hermetic"}

mkdir -p "${output_dir}"

for input_path in "${input_dir}"/*.c; do
  fixture_name=$(basename "${input_path}")
  output_path="${output_dir}/${fixture_name}"
  temporary_path="${output_path}.tmp"

  {
    echo '#include "phase1_compat.h"'
    sed '/^#include </d' "${input_path}"
  } > "${temporary_path}"
  mv "${temporary_path}" "${output_path}"
done

echo "Prepared fixture-only no-SDK C sources in ${output_dir}"

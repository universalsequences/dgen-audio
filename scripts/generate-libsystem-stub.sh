#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
allowlist=${DGEN_LIBSYSTEM_ALLOWLIST:-"${repo_root}/toolchain/abi/libsystem-symbols-v1.txt"}
output=${1:-"${repo_root}/toolchain/lib/libSystem.tbd"}

if [ ! -f "${allowlist}" ]; then
  echo "error: libSystem symbol allowlist is missing: ${allowlist}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")"
{
  echo "--- !tapi-tbd"
  echo "tbd-version:     4"
  echo "targets:         [ arm64-macos ]"
  echo "install-name:    '/usr/lib/libSystem.B.dylib'"
  echo "current-version: 1.0.0"
  echo "compatibility-version: 1.0.0"
  echo "exports:"
  echo "  - targets:         [ arm64-macos ]"
  echo "    symbols:"
  LC_ALL=C sort -u "${allowlist}" | while IFS= read -r symbol; do
    case "${symbol}" in
      ""|\#*) continue ;;
    esac
    printf '      - %s\n' "${symbol}"
  done
  echo "..."
} > "${output}"

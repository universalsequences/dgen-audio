#!/bin/sh
set -eu

# Build the publishable x86_64-unknown-linux-gnu DGen toolchain stage inside a
# pinned container image, then hand off to scripts/build-toolchain.sh unchanged.
#
# WHY A CONTAINER AT ALL. The stage is a *distribution*: it is fetched onto
# machines nobody building it controls. Its glibc floor is therefore a property
# of the distribution and must be chosen, not inherited from whatever the build
# host happens to run. Building natively on a rolling-release distribution
# produces binaries that refuse to start on Ubuntu LTS -- a silent packaging
# regression that only shows up on a user's machine.
#
# ubuntu:22.04 carries glibc 2.35, which is the floor the published Linux
# DGenLisp distribution already targets (content/dgenlisp.lock in the ESeq
# host records the same 2.35). Keeping both halves on one floor means one
# statement covers the whole Linux install rather than two that can drift.
# scripts/build-toolchain.sh reads the achieved floor back off the staged
# binaries into VERSION.json, so the number below is an input and VERSION.json
# is the evidence.
#
# A native `scripts/build-toolchain.sh` run on a Linux host is still supported
# and produces a correct stage; it is just not publishable, because its floor
# is the build host's.

base_image=${DGEN_TOOLCHAIN_BASE_IMAGE:-ubuntu:22.04}
build_image=${DGEN_TOOLCHAIN_BUILD_IMAGE:-dgen-toolchain-build:jammy}
docker=${DOCKER:-docker}

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

# The repository is bind-mounted at a distinctive path rather than at its host
# path. build-toolchain.sh scans the staged binaries for any string naming its
# own repo/build root, and a short generic mount point would make that scan
# either useless or trigger-happy.
container_repo=/dgen-src

command -v "${docker}" >/dev/null 2>&1 || {
  echo "error: required tool not found: ${docker}" >&2
  echo "A publishable Linux stage is built inside a pinned container image." >&2
  exit 1
}

host_arch=$(uname -m)
if [ "${host_arch}" != "x86_64" ]; then
  echo "error: this builds an x86_64 stage and does not cross-compile;" >&2
  echo "the host is ${host_arch}." >&2
  exit 1
fi

echo "Building the toolchain build image (${build_image}) from ${base_image}..."
"${docker}" build --tag "${build_image}" - <<DOCKERFILE
FROM ${base_image}
ENV DEBIAN_FRONTEND=noninteractive
# Exactly the tools scripts/build-toolchain.sh requires: a C++ host compiler,
# cmake/ninja, python3 for the deterministic tar step, binutils for strings /
# strip / readelf, and xz/curl for the pinned LLVM source archive.
RUN apt-get update \\
 && apt-get install --yes --no-install-recommends \\
      binutils \\
      build-essential \\
      ca-certificates \\
      cmake \\
      curl \\
      file \\
      ninja-build \\
      patch \\
      python3 \\
      xz-utils \\
 && rm -rf /var/lib/apt/lists/*
DOCKERFILE

echo "Running scripts/build-toolchain.sh inside ${build_image}..."
# Run as the invoking user so the staged prefix, archive and regenerated
# metadata land in the working tree owned by whoever started the build.
exec "${docker}" run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env DGEN_TOOLCHAIN_JOBS \
  --env DGEN_TOOLCHAIN_LINK_JOBS \
  --volume "${repo_root}:${container_repo}" \
  --workdir "${container_repo}" \
  "${build_image}" \
  "${container_repo}/scripts/build-toolchain.sh" "$@"

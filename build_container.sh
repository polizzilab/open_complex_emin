#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/protonator.sif"

echo "Building Singularity container: $SIF"
singularity build --fakeroot --force "$SIF" "${SCRIPT_DIR}/protonator.def"
echo "Done: $SIF"

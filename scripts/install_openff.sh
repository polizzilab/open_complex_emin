#!/usr/bin/env bash
# Install the openff stack from GitHub source.
# These packages are not currently available on PyPI (yanked / require py3.12+).
# Run this once after `uv sync`:
#
#   uv sync
#   bash scripts/install_openff.sh
#
set -euo pipefail

VENV="$(dirname "$0")/../.venv"
PYTHON="$VENV/bin/python"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

patch_version() {
    # Remove versioningit / setuptools-scm and inject a static version string.
    local toml="$1/pyproject.toml"
    python3 - "$toml" <<'EOF'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
c = p.read_text()
c = re.sub(r'"versioningit[^"]*"', '', c)
c = re.sub(r'"setuptools-scm[^"]*"', '', c)
c = re.sub(r'\ndynamic\s*=\s*\[.*?"version".*?\]', '\nversion = "0.0.1"', c)
p.write_text(c)
EOF
}

install_from_git() {
    local name="$1" url="$2" tag="$3"
    echo "==> Installing $name @ $tag"
    local dst="$TMP/$name"
    git clone --quiet --depth 1 --branch "$tag" "$url" "$dst"
    patch_version "$dst"
    "$PYTHON" -m pip install --quiet --no-deps "$dst"
}

# Order matters: utilities → units → toolkit
install_from_git openff-utilities \
    https://github.com/openforcefield/openff-utilities.git main
install_from_git openff-units \
    https://github.com/openforcefield/openff-units.git main
install_from_git openff-toolkit \
    https://github.com/openforcefield/openff-toolkit.git main

echo "==> openff stack installed successfully."

# Install pip + setuptools so editable installs work, then install this package
"$PYTHON" -m ensurepip --upgrade --default-pip 2>/dev/null || true
"$PYTHON" -m pip install --quiet pip setuptools
"$PYTHON" -m pip install --quiet -e "$(dirname "$0")/.." --no-build-isolation
echo "==> protonator installed."

#!/usr/bin/env bash
# Install the bunsalyze dependency (not on PyPI) into the *active* conda env.
#
# bunsalyze provides the protein/ligand donor-acceptor identification and the
# is_valid_hbond geometry used by the fast track (protonator-fast).  We install
# it from source so that as bunsalyze improves, the fast track improves too.
# `pip install .` pulls bunsalyze's only extra runtime dep (freesasa) from PyPI;
# numpy/prody/scipy/networkx/rdkit are already provided by the conda env.
#
# Usage:
#   conda activate protonator
#   ./setup.sh                 # clone (if needed) + install
#
# Override the source/branch/location if desired:
#   BUNSALYZE_URL=...  BUNSALYZE_REF=main  BUNSALYZE_DIR=~/programs/bunsalyze  ./setup.sh
set -euo pipefail

BUNSALYZE_URL="${BUNSALYZE_URL:-https://github.com/polizzilab/bunsalyze}"
BUNSALYZE_REF="${BUNSALYZE_REF:-main}"
BUNSALYZE_DIR="${BUNSALYZE_DIR:-$HOME/programs/bunsalyze}"

if ! command -v python >/dev/null 2>&1; then
    echo "error: no python on PATH — activate the conda env first (conda activate protonator)." >&2
    exit 1
fi

if [ -d "$BUNSALYZE_DIR/.git" ]; then
    echo "bunsalyze already present at $BUNSALYZE_DIR — fetching $BUNSALYZE_REF"
    git -C "$BUNSALYZE_DIR" fetch --quiet origin "$BUNSALYZE_REF"
    git -C "$BUNSALYZE_DIR" checkout --quiet "$BUNSALYZE_REF"
    git -C "$BUNSALYZE_DIR" pull --quiet --ff-only origin "$BUNSALYZE_REF" || true
else
    echo "cloning bunsalyze ($BUNSALYZE_REF) -> $BUNSALYZE_DIR"
    git clone --branch "$BUNSALYZE_REF" "$BUNSALYZE_URL" "$BUNSALYZE_DIR"
fi

echo "installing bunsalyze into the active environment ($(python -c 'import sys;print(sys.prefix)'))"
python -m pip install "$BUNSALYZE_DIR"

python - <<'PY'
import sys
from bunsalyze.utils.graph import is_valid_hbond                     # noqa: F401
from bunsalyze.utils.calc_protein_dons_accs import get_protein_polar_atoms  # noqa: F401
from bunsalyze.utils.calc_ligand_dons_accs import get_ligand_polar_atoms    # noqa: F401
import bunsalyze  # exercises the full package (incl. burial_calc) import path
if "torch" in sys.modules:
    sys.exit(
        "ERROR: importing bunsalyze pulled in torch. You cloned a revision that "
        "still depends on torch. Use the torch-free branch (BUNSALYZE_REF)."
    )
print("bunsalyze installed, importable, and torch-free.")
PY

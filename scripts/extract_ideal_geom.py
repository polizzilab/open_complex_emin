"""
One-off extraction of LASErMPNN's ideal protonated residue geometry.

Run ONCE, in an environment that has torch + the LASErMPNN repo importable
(e.g. the `lasermpnn2` conda env):

    python scripts/extract_ideal_geom.py \
        --lasermpnn-root ~/programs/LASErMPNN \
        --out protonator/data/ideal_residue_h.json

The output JSON is vendored into the package so the runtime `protonator` env
never needs torch or LASErMPNN.  It contains, per amino acid (3-letter code):

    ideal_coords          {atom_name: [x, y, z]}  ideal all-atom template incl. H
    hydrogen_triads       [[ [hA,hB,hC], [h_name, ...] ], ...]
                          heavy-atom triad -> hydrogens placed by superposing the
                          ideal template's triad onto the observed heavy atoms
                          (LASErMPNN add_nonrotatable_hydrogens recipe).
    optional_hydrogens    list of H names that are titratable/removable (His, Cys)

Plus global backbone amide-H geometry (bond length / angle).

Provenance: bond/angle/template values are LASErMPNN's
utils/constants.py + files/ideal_aa_coords_prot.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lasermpnn-root", type=Path,
                    default=Path.home() / "programs" / "LASErMPNN",
                    help="Path to the LASErMPNN repo (its parent is added to sys.path).")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "protonator" / "data" / "ideal_residue_h.json")
    args = ap.parse_args()

    root = args.lasermpnn_root.expanduser().resolve()
    sys.path.insert(0, str(root.parent))  # so `import LASErMPNN.utils...` resolves
    from LASErMPNN.utils import constants as C  # type: ignore

    aa_idx_to_short = C.aa_idx_to_short
    atom_order = C.hydrogen_extended_dataset_atom_order
    ideal = C.ideal_prot_aa_coords  # (20, MAX_PROT_ATOMS, 3) tensor
    triad_map = C.hydrogen_alignment_coord_map
    optional = C.optional_hydrogen_map  # {short: [H names]}

    residues: dict = {}
    for idx in range(20):
        short = aa_idx_to_short[idx]
        long = C.aa_short_to_long[short]
        names = atom_order[short]
        coords_t = ideal[idx]

        ideal_coords = {}
        for ai, name in enumerate(names):
            xyz = coords_t[ai]
            # ideal template is padded with NaN beyond the residue's real atoms
            if bool(xyz.isnan().any().item()):
                continue
            ideal_coords[name] = [float(v) for v in xyz.tolist()]

        triads = []
        for triad_names, h_names in triad_map.get(short, {}).items():
            triads.append([list(triad_names), list(h_names)])

        residues[long] = {
            "ideal_coords": ideal_coords,
            "hydrogen_triads": triads,
            "optional_hydrogens": list(optional.get(short, [])),
        }

    out = {
        "_provenance": "Extracted from LASErMPNN utils/constants.py + "
                       "files/ideal_aa_coords_prot.pt via scripts/extract_ideal_geom.py",
        "backbone_amide_h": {
            # extend from (C, CA, N): bond N-H, angle CA-N-H, dihedral C-CA-N-H = phi + 180
            "bond_length": 1.0,
            "bond_angle_deg": 120.0,
            "dihedral_offset_deg": 180.0,
        },
        "residues": residues,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    n_h = sum(
        sum(1 for a in r["ideal_coords"] if a.startswith("H") or a[0:1] == "H")
        for r in residues.values()
    )
    print(f"Wrote {args.out}  ({len(residues)} residues, ~{n_h} ideal H atoms)")


if __name__ == "__main__":
    main()

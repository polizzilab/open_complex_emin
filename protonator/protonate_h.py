"""
Torch-less geometric placement of protein hydrogens on fixed sidechain geometry.

Adapted from LASErMPNN's hydrogen-placement method (utils/build_rotamers.py:
add_nonrotatable_hydrogens + impute_backbone_nh_coords).  We reuse only the
*method* and the *ideal residue templates* (vendored into data/ideal_residue_h.json
by scripts/extract_ideal_geom.py), reimplemented in numpy so the runtime needs
no torch.

Heavy atoms are never moved.  Non-rotatable H are placed by superposing the
ideal template's heavy-atom triads onto the observed heavy atoms; the backbone
amide H is placed from the phi dihedral; rotatable hydroxyl/thiol H get an
arbitrary initial dihedral (the H-bond sweep orients them afterwards).

Histidine ring N-H are NOT placed here (left to the tautomer step); a default
HIE (HE2 only) is placed so the structure is valid if the tautomer step is off.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import prody as pr

from .geometry import (
    apply_transform,
    dihedral,
    extend_coordinate,
    internal_coords,
    kabsch,
)

_DATA = Path(__file__).parent / "data" / "ideal_residue_h.json"

# Only polar (H-bond donor) hydrogens are needed for the fast interface-H-bond
# track, so we skip all C-H placement.  This is both simpler and avoids the
# distortion that 3-atom triad superposition can introduce for branched C-H
# (e.g. Leu HG, Val HB) when the observed sidechain deviates from ideal.
# Donor H names are RESIDUE-SPECIFIC (e.g. "HG" is a thiol donor on Cys but a
# nonpolar methine on Leu), mirroring bunsalyze aa_to_sc_hbond_donor_to_heavy_atom.
_POLAR_H_BY_RES = {
    "SER": {"HG"}, "CYS": {"HG"}, "THR": {"HG1"}, "TYR": {"HH"},
    "LYS": {"HZ1", "HZ2", "HZ3"},
    "ASN": {"HD21", "HD22"}, "GLN": {"HE21", "HE22"},
    "ARG": {"HE", "HH11", "HH12", "HH21", "HH22"},
    "TRP": {"HE1"},
    "HIS": {"HD1", "HE2"},  # placed by the tautomer step
}

# Rotatable terminal hydroxyl/thiol H: (H name, X heavy atom, neighbour B, next-neighbour A)
# H is placed off X; A-B-X-H dihedral is arbitrary (set by the sweep afterwards).
_ROTATABLE_OH = {
    "SER": ("HG", "OG", "CB", "CA"),
    "THR": ("HG1", "OG1", "CB", "CA"),
    "TYR": ("HH", "OH", "CZ", "CE1"),
    "CYS": ("HG", "SG", "CB", "CA"),
}


@lru_cache(maxsize=1)
def _templates() -> dict:
    return json.loads(_DATA.read_text())


def _residue_coords(residue: pr.Selection) -> dict[str, np.ndarray]:
    names = residue.getNames()
    coords = residue.getCoords()
    return {n: c for n, c in zip(names, coords)}


def place_his_ring_h(obs: dict[str, np.ndarray], which: str) -> np.ndarray | None:
    """
    Place a histidine ring N-H ('HD1' on ND1 or 'HE2' on NE2) by superposing the
    ideal imidazole onto the observed ring atoms.  Returns the H coord or None.
    """
    tmpl = _templates()["residues"]["HIS"]["ideal_coords"]
    ring = ["CG", "ND1", "CE1", "NE2", "CD2"]
    if not all(a in obs for a in ring) or not all(a in tmpl for a in ring):
        return None
    if which not in tmpl:
        return None
    mob = np.array([tmpl[a] for a in ring])
    fix = np.array([obs[a] for a in ring])
    r, mc, fc = kabsch(mob, fix)
    return apply_transform(np.array(tmpl[which])[None, :], r, mc, fc)[0]


def place_protein_hydrogens(prot_ag: pr.AtomGroup,
                            default_his: str = "HE2") -> pr.AtomGroup:
    """
    Return a new prody AtomGroup = ``prot_ag`` heavy atoms + placed hydrogens.

    Heavy-atom coordinates are copied unchanged.  Existing H in the input are
    dropped and re-placed.  The returned AtomGroup is ready for
    bunsalyze.get_protein_polar_atoms.
    """
    tmpl = _templates()
    res_tmpl = tmpl["residues"]
    bb = tmpl["backbone_amide_h"]

    out_names: list[str] = []
    out_elems: list[str] = []
    out_coords: list[np.ndarray] = []
    out_resnum: list[int] = []
    out_resname: list[str] = []
    out_chid: list[str] = []
    out_icode: list[str] = []

    residues = list(prot_ag.iterResidues())
    prev_c: np.ndarray | None = None
    prev_key: tuple | None = None

    for ri, residue in enumerate(residues):
        resname = residue.getResname()
        chid = residue.getChids()[0]
        resnum = int(residue.getResnums()[0])
        icode = residue.getIcodes()[0]
        obs = _residue_coords(residue)

        def emit(name: str, coord: np.ndarray) -> None:
            out_names.append(name)
            out_elems.append("H" if name.startswith("H") else name[0])
            out_coords.append(np.asarray(coord, dtype=float))
            out_resnum.append(resnum)
            out_resname.append(resname)
            out_chid.append(chid)
            out_icode.append(icode)

        # 1. Emit all observed heavy atoms unchanged (skip any pre-existing H).
        for name, coord in zip(residue.getNames(), residue.getCoords()):
            if name.startswith("H"):
                continue
            emit(name, coord)

        info = res_tmpl.get(resname)

        # contiguity for backbone phi
        contiguous = (
            prev_c is not None and prev_key is not None
            and prev_key[0] == chid and resnum == prev_key[1] + 1
        )

        if info is not None:
            ideal = {k: np.array(v) for k, v in info["ideal_coords"].items()}

            # 2. Non-rotatable polar (donor) H via triad superposition.
            polar = _POLAR_H_BY_RES.get(resname, set())
            for triad_names, h_names in info["hydrogen_triads"]:
                wanted = [h for h in h_names if h in polar and h in ideal]
                if not wanted:
                    continue
                if not all(a in obs for a in triad_names):
                    continue
                if not all(a in ideal for a in triad_names):
                    continue
                mob = np.array([ideal[a] for a in triad_names])
                fix = np.array([obs[a] for a in triad_names])
                r, mc, fc = kabsch(mob, fix)
                for h in wanted:
                    emit(h, apply_transform(ideal[h][None, :], r, mc, fc)[0])

            # 3. Rotatable hydroxyl/thiol H — arbitrary initial dihedral (180°).
            if resname in _ROTATABLE_OH:
                hname, xn, bn, an = _ROTATABLE_OH[resname]
                if all(a in obs for a in (xn, bn, an)) and hname in ideal and xn in ideal and bn in ideal:
                    bl, ang = internal_coords(ideal[xn], ideal[bn], None, ideal[hname])
                    coord = extend_coordinate(obs[an], obs[bn], obs[xn], bl, ang, np.pi)
                    emit(hname, coord)

            # 4. Histidine ring H — default tautomer only (HIE = HE2).
            if resname == "HIS" and default_his:
                hc = place_his_ring_h(obs, default_his)
                if hc is not None:
                    emit(default_his, hc)

        # 5. Backbone amide H (all non-Pro residues with a preceding residue).
        if resname != "PRO" and contiguous and all(a in obs for a in ("N", "CA", "C")):
            phi = dihedral(prev_c, obs["N"], obs["CA"], obs["C"])
            coord = extend_coordinate(
                obs["C"], obs["CA"], obs["N"],
                bb["bond_length"], np.deg2rad(bb["bond_angle_deg"]),
                phi + np.deg2rad(bb["dihedral_offset_deg"]),
            )
            emit("H", coord)

        prev_c = obs.get("C")
        prev_key = (chid, resnum)

    ag = pr.AtomGroup("protein_h")
    ag.setCoords(np.array(out_coords))
    ag.setNames(np.array(out_names, dtype=object))
    ag.setElements(np.array(out_elems, dtype=object))
    ag.setResnums(np.array(out_resnum))
    ag.setResnames(np.array(out_resname, dtype=object))
    ag.setChids(np.array(out_chid, dtype=object))
    ag.setIcodes(np.array(out_icode, dtype=object))
    ag.setOccupancies(np.ones(len(out_names)))
    ag.setBetas(np.zeros(len(out_names)))
    return ag

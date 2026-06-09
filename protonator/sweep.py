"""
Greedy maximization of protein-ligand interface hydrogen bonds.

No force field, no minimization: heavy atoms and non-rotatable groups stay put.
We only reorient the degrees of freedom that don't move heavy-atom scaffolding:

  * protein Ser/Thr/Tyr/Cys hydroxyl/thiol and Lys ammonium H  (dihedral sweep)
  * ligand terminal polar-H rotors                              (dihedral sweep)
  * protein Asn/Gln and ligand primary-amide flips             (180 deg)
  * protein His tautomer  (HID / HIE / HIP)                    (discrete)

Scoring uses bunsalyze.is_valid_hbond verbatim (so as bunsalyze's H-bond
definition improves, this does too).  A single greedy pass visits each group
in turn, picking the orientation that maximizes the *total* number of valid
protein-ligand H-bonds given everything committed so far.

For the protein hydroxyl/thiol/ammonium rotors, ties on that count are broken
by (a) avoiding H-H clashes with the rest of the protein and (b) the most
linear D-H...A geometry (angle closest to 180°, the bunsalyze ideal), measured
against both ligand and protein acceptors.  This keeps a rotor that can't reach
the ligand from sitting at its arbitrary initial dihedral — it instead points
at a backbone/sidechain acceptor with good geometry rather than clashing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import prody as pr

from bunsalyze.utils.constants import (
    PolarAtom,
    H_TO_H_CLASH_DIST,
    MIN_HBOND_ANGLE,
    MIN_HBOND_DISTANCE,
    ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF,
    ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF,
    S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF,
)
from bunsalyze.utils.graph import is_valid_hbond
from bunsalyze.utils.calc_protein_dons_accs import get_protein_polar_atoms
from bunsalyze.utils.calc_ligand_dons_accs import (
    compute_ligand_capacity,
    get_ligand_polar_atoms,
)

from .geometry import rotate_about_axis
from .ligand_h import find_rotatable_groups
from .protonate_h import place_his_ring_h

_MAXD = S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF

# Protein rotatable polar-H rotors: resname -> (donor, pivot, [H...])
_PROT_ROTORS = {
    "SER": ("OG", "CB", ["HG"]),
    "THR": ("OG1", "CB", ["HG1"]),
    "TYR": ("OH", "CZ", ["HH"]),
    "CYS": ("SG", "CB", ["HG"]),
    "LYS": ("NZ", "CE", ["HZ1", "HZ2", "HZ3"]),
}
# Protein amide flips: resname -> (carbonyl, ref, oxygen, nitrogen, [H...])
_PROT_FLIPS = {
    "ASN": ("CG", "CB", "OD1", "ND2", ["HD21", "HD22"]),
    "GLN": ("CD", "CG", "OE1", "NE2", ["HE21", "HE22"]),
}


def _hbond_distance_cutoff(donor_elem: str, acc_elem: str) -> float | None:
    """Donor-heavy to acceptor-heavy distance cutoff (Å) for an X-H...A bond,
    mirroring bunsalyze.is_valid_hbond.  None if the pair can't H-bond."""
    on = ("N", "O")
    if donor_elem in on and acc_elem in on:
        return ON_ON_HYDROGEN_BOND_DISTANCE_CUTOFF
    if donor_elem == "S" and acc_elem == "S":
        return S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF
    if (donor_elem == "S" and acc_elem in on) or (donor_elem in on and acc_elem == "S"):
        return ON_S_HYDROGEN_BOND_DISTANCE_CUTOFF
    return None


def _best_donor_hbond_angle(
    donor_coord: np.ndarray, donor_elem: str, h_coords: list[np.ndarray],
    acc_coords: np.ndarray, acc_elems: list[str],
) -> float:
    """Largest D-H...A angle (deg, 180° = linear) over all valid donor→acceptor
    pairs, using the same distance/angle criteria as bunsalyze.is_valid_hbond.

    Returns -inf when no candidate forms a valid H-bond, so it can be used as a
    geometry tie-breaker (higher = closer to ideal linear geometry).
    """
    best = -np.inf
    if not len(acc_coords):
        return best
    for ai in range(len(acc_coords)):
        ac = acc_coords[ai]
        cutoff = _hbond_distance_cutoff(donor_elem, acc_elems[ai])
        if cutoff is None:
            continue
        da = np.linalg.norm(donor_coord - ac)
        if da > cutoff or da < MIN_HBOND_DISTANCE:
            continue
        for h in h_coords:
            d_to_h = h - donor_coord
            a_to_h = h - ac
            denom = np.linalg.norm(d_to_h) * np.linalg.norm(a_to_h)
            if denom < 1e-9:
                continue
            angle = np.rad2deg(np.arccos(np.clip(np.dot(d_to_h, a_to_h) / denom, -1.0, 1.0)))
            if angle >= MIN_HBOND_ANGLE and angle > best:
                best = angle
    return best


def _reset(polars) -> None:
    """Restore full donor/acceptor capacity and clear H engagement in place."""
    for p in polars:
        p.donor_count = p.max_donor_count
        p.acceptor_count = p.max_acceptor_count
        for h in p.donor_hydrogens:
            h.is_engaged = False
            h.engaged_to = None


def count_interface_hbonds(protein_polars, ligand_polars) -> int:
    """Number of valid protein<->ligand H-bonds (bunsalyze geometry).

    Capacities/engagement are reset in place each call (cheaper than deep-copying
    every evaluation); coordinates are never modified here.  is_valid_hbond
    mutates the reset state during a single count, which is exactly the greedy
    donor/acceptor assignment bunsalyze uses.
    """
    _reset(protein_polars)
    _reset(ligand_polars)
    n = 0
    for p in protein_polars:
        pc = np.asarray(p.coord)
        for l in ligand_polars:
            if np.linalg.norm(pc - np.asarray(l.coord)) > _MAXD:
                continue
            for h in sorted(p.donor_hydrogens, key=lambda x: np.linalg.norm(x.coord - l.coord)):
                if is_valid_hbond(p, h, l, hydrogen_clash_check=True):
                    n += 1
            for h in sorted(l.donor_hydrogens, key=lambda x: np.linalg.norm(x.coord - p.coord)):
                if is_valid_hbond(l, h, p, hydrogen_clash_check=True):
                    n += 1
    return n


@dataclass
class _Atom:
    name: str
    element: str
    resname: str
    resnum: int
    icode: str
    chain: str
    coord: np.ndarray


class InterfaceModel:
    """Mutable protein+ligand model for the greedy interface-H-bond optimizer."""

    def __init__(self, protein_ag: pr.AtomGroup, ligand_ag: pr.AtomGroup,
                 lig_mol, ncaa_dict: dict | None = None):
        self.ncaa_dict = ncaa_dict or {}
        self.lig_mol = lig_mol

        # Output-truth atom tables (mutated on commit).
        self.prot_atoms = self._table(protein_ag)
        self.lig_atoms = self._table(ligand_ag)

        # Coordinate lookups for static reference atoms (axes/pivots).
        self.prot_xyz = {(a.resnum, a.icode, a.name): a.coord for a in self.prot_atoms}
        self.lig_xyz = {a.name: a.coord for a in self.lig_atoms}

        # bunsalyze polar-atom scoring state (built once, mutated in place).
        self.protein_polars = get_protein_polar_atoms(protein_ag, ncaa_dict=self.ncaa_dict)
        lig_cap = compute_ligand_capacity(lig_mol)
        self.ligand_polars = get_ligand_polar_atoms(lig_cap, ligand_ag, lig_mol)

        self._index()
        self._compute_interface()

    # Protein polar atoms more than this far (heavy-heavy) from any ligand polar
    # atom can never form an interface H-bond; skip them when scoring.  Generous
    # enough to survive the ~2.5 A an Asn/Gln flip can move OD1/ND2.
    _INTERFACE_CUTOFF = 8.0

    def _compute_interface(self) -> None:
        lig_xyz = np.array([np.asarray(l.coord) for l in self.ligand_polars])
        if len(lig_xyz) == 0:
            self.interface_polars = []
            return
        keep = []
        for p in self.protein_polars:
            if np.min(np.linalg.norm(lig_xyz - np.asarray(p.coord), axis=1)) <= self._INTERFACE_CUTOFF:
                keep.append(p)
        self.interface_polars = keep

    @staticmethod
    def _table(ag: pr.AtomGroup) -> list[_Atom]:
        return [
            _Atom(n, e, rn, int(rs), ic, ch, np.asarray(c, dtype=float))
            for n, e, rn, rs, ic, ch, c in zip(
                ag.getNames(), ag.getElements(), ag.getResnames(),
                ag.getResnums(), ag.getIcodes(), ag.getChids(), ag.getCoords())
        ]

    def _index(self) -> None:
        # protein polar atom by (resnum, icode, name); ligand by name
        self.p_polar = {
            (p.parent_group_identifier[2], p.parent_group_identifier[3], p.name): p
            for p in self.protein_polars
        }
        self.l_polar = {p.name: p for p in self.ligand_polars}

    # ----- coordinate mutation helpers (update scoring + output in sync) -----

    def _set_prot_coord(self, resnum, icode, name, coord):
        self.prot_xyz[(resnum, icode, name)] = coord
        for a in self.prot_atoms:
            if a.resnum == resnum and a.icode == icode and a.name == name:
                a.coord = coord

    def _set_prot_polar_heavy(self, resnum, icode, name, coord):
        pa = self.p_polar.get((resnum, icode, name))
        if pa is not None:
            pa.coord = coord

    def _set_prot_donor_h(self, resnum, icode, heavy_name, h_name, coord):
        pa = self.p_polar.get((resnum, icode, heavy_name))
        if pa is not None:
            for h in pa.donor_hydrogens:
                if h.name == h_name:
                    h.coord = coord

    def _set_lig_coord(self, name, coord):
        self.lig_xyz[name] = coord
        for a in self.lig_atoms:
            if a.name == name:
                a.coord = coord

    def score(self) -> int:
        return count_interface_hbonds(self.interface_polars, self.ligand_polars)

    # ------------------------------ sweeps -------------------------------

    def optimize(self, step_deg: float = 10.0, sweep_protein: bool = True,
                 sweep_ligand: bool = True, do_flips: bool = True,
                 his_tautomers: bool = True, allow_hip: bool = False) -> dict:
        report = {"start": self.score()}
        if sweep_protein:
            self._sweep_protein_rotors(step_deg)
        if sweep_ligand:
            self._sweep_ligand_rotors(step_deg)
        if do_flips:
            self._do_protein_flips()
        if his_tautomers:
            self._do_his_tautomers(allow_hip=allow_hip)
        report["final"] = self.score()
        return report

    def _sweep_protein_rotors(self, step_deg: float) -> None:
        angles = np.deg2rad(np.arange(0.0, 360.0, step_deg))
        seen = set()
        for a in self.prot_atoms:
            key = (a.resnum, a.icode, a.resname)
            if key in seen or a.resname not in _PROT_ROTORS:
                continue
            seen.add(key)
            resnum, icode, resname = key
            donor, pivot, hnames = _PROT_ROTORS[resname]
            o = self.prot_xyz.get((resnum, icode, donor))
            piv = self.prot_xyz.get((resnum, icode, pivot))
            if o is None or piv is None:
                continue
            pa = self.p_polar.get((resnum, icode, donor))
            if pa is None:
                continue
            h0 = {h.name: h.coord.copy() for h in pa.donor_hydrogens if h.name in hnames}
            if not h0:
                continue
            axis = o - piv

            # Other protein H coords for H-H clash detection.  Built fresh each
            # residue so it reflects H positions committed earlier in this sweep.
            other_h_list = [
                b.coord for b in self.prot_atoms
                if b.element == "H" and not (b.resnum == resnum and b.icode == icode)
            ]
            other_h = np.array(other_h_list, dtype=float) if other_h_list else np.empty((0, 3))

            # Acceptors near this donor, for the geometry tie-breaker:
            #   * protein polar atoms on other residues (backbone/sidechain H-bonds)
            #   * ligand polar atoms (so interface H-bonds also prefer linear geometry)
            # Only atoms with acceptor capacity count — donor-only atoms (backbone
            # amide N, Lys NZ, Arg guanidinium N) can't accept, so steering an H
            # toward them would be a phantom H-bond.  Gathered once per residue
            # (acceptor heavy atoms are fixed during the rotor sweep; only H moves).
            acc_coords_list: list[np.ndarray] = []
            acc_elems: list[str] = []
            for p in self.protein_polars:
                if (p.max_acceptor_count > 0
                        and not (p.parent_group_identifier[2] == resnum
                                 and p.parent_group_identifier[3] == icode)
                        and np.linalg.norm(np.asarray(p.coord) - o) < _MAXD):
                    acc_coords_list.append(np.asarray(p.coord))
                    acc_elems.append(p.element)
            lig_near = False
            for lp in self.ligand_polars:
                lc = np.asarray(lp.coord)
                if np.linalg.norm(lc - o) < _MAXD:
                    lig_near = True
                    if lp.max_acceptor_count > 0:
                        acc_coords_list.append(lc)
                        acc_elems.append(lp.element)
            acc_coords = np.array(acc_coords_list, dtype=float) if acc_coords_list else np.empty((0, 3))

            # Greedy choice over all dihedral angles, ranked lexicographically by:
            #   1. n_lig    — protein-ligand H-bond count (the module's purpose)
            #   2. no_clash — trial H clear of other protein H (bunsalyze clash dist)
            #   3. geom     — best D-H...A angle (→180° linear); the geometry the
            #                 user asked for, and what orients a non-interface
            #                 rotor toward a protein acceptor instead of leaving
            #                 it at the arbitrary initial dihedral.
            # theta=0 (initial placement) is always among the angles, so n_lig
            # never drops below its starting value: the sweep is non-decreasing.
            donor_elem = pa.element
            best_key = None
            best_th = 0.0
            for th in angles:
                trial = [rotate_about_axis(c0, o, axis, th) for c0 in h0.values()]
                for hn, tc in zip(h0, trial):
                    self._set_prot_donor_h(resnum, icode, donor, hn, tc)

                n_lig = self.score() if lig_near else 0

                no_clash = True
                if len(other_h):
                    for tc in trial:
                        if np.linalg.norm(other_h - tc, axis=1).min() < H_TO_H_CLASH_DIST:
                            no_clash = False
                            break

                geom = _best_donor_hbond_angle(o, donor_elem, trial, acc_coords, acc_elems)

                cand = (n_lig, no_clash, geom)
                if best_key is None or cand > best_key:
                    best_key, best_th = cand, th

            for hn, c0 in h0.items():
                c = rotate_about_axis(c0, o, axis, best_th)
                self._set_prot_donor_h(resnum, icode, donor, hn, c)
                self._set_prot_coord(resnum, icode, hn, c)

    def _sweep_ligand_rotors(self, step_deg: float) -> None:
        angles = np.deg2rad(np.arange(0.0, 360.0, step_deg))
        for g in find_rotatable_groups(self.lig_mol):
            o = self.lig_xyz.get(g.donor)
            piv = self.lig_xyz.get(g.pivot)
            pa = self.l_polar.get(g.donor)
            if o is None or piv is None or pa is None:
                continue
            h0 = {h.name: h.coord.copy() for h in pa.donor_hydrogens if h.name in g.hydrogens}
            if not h0:
                continue
            axis = o - piv
            best_score, best = self.score(), None
            for th in angles:
                for hn, c0 in h0.items():
                    for h in pa.donor_hydrogens:
                        if h.name == hn:
                            h.coord = rotate_about_axis(c0, o, axis, th)
                s = self.score()
                if s > best_score:
                    best_score, best = s, th
            th = best if best is not None else 0.0
            for hn, c0 in h0.items():
                c = rotate_about_axis(c0, o, axis, th)
                for h in pa.donor_hydrogens:
                    if h.name == hn:
                        h.coord = c
                self._set_lig_coord(hn, c)

    def _do_protein_flips(self) -> None:
        seen = set()
        for a in self.prot_atoms:
            key = (a.resnum, a.icode, a.resname)
            if key in seen or a.resname not in _PROT_FLIPS:
                continue
            seen.add(key)
            resnum, icode, resname = key
            carbonyl, ref, oxy, nit, hnames = _PROT_FLIPS[resname]
            cc = self.prot_xyz.get((resnum, icode, carbonyl))
            rr = self.prot_xyz.get((resnum, icode, ref))
            o_pa = self.p_polar.get((resnum, icode, oxy))
            n_pa = self.p_polar.get((resnum, icode, nit))
            if cc is None or rr is None or o_pa is None or n_pa is None:
                continue
            axis = rr - cc

            def snapshot():
                hs = {h.name: h.coord.copy() for h in n_pa.donor_hydrogens}
                return (o_pa.coord.copy(), n_pa.coord.copy(), hs)

            def apply(o_c, n_c, hcoords):
                o_pa.coord = o_c
                n_pa.coord = n_c
                self._set_prot_polar_heavy(resnum, icode, oxy, o_c)
                self._set_prot_coord(resnum, icode, oxy, o_c)
                self._set_prot_coord(resnum, icode, nit, n_c)
                for h in n_pa.donor_hydrogens:
                    if h.name in hcoords:
                        h.coord = hcoords[h.name]
                        self._set_prot_coord(resnum, icode, h.name, hcoords[h.name])

            base = snapshot()
            s_base = self.score()
            flipped = (
                rotate_about_axis(base[0], cc, axis, np.pi),
                rotate_about_axis(base[1], cc, axis, np.pi),
                {k: rotate_about_axis(v, cc, axis, np.pi) for k, v in base[2].items()},
            )
            apply(*flipped)
            if self.score() <= s_base:
                apply(*base)  # revert

    def _do_his_tautomers(self, allow_hip: bool = False) -> None:
        seen = set()
        for a in self.prot_atoms:
            key = (a.resnum, a.icode, a.resname)
            if key in seen or a.resname != "HIS":
                continue
            seen.add(key)
            resnum, icode, _ = key
            self._optimize_one_his(resnum, icode, allow_hip=allow_hip)

    def _optimize_one_his(self, resnum, icode, allow_hip: bool = False) -> None:
        obs = {a.name: a.coord for a in self.prot_atoms
               if a.resnum == resnum and a.icode == icode and not a.name.startswith("H")}
        hd1 = place_his_ring_h(obs, "HD1")
        he2 = place_his_ring_h(obs, "HE2")
        if hd1 is None or he2 is None:
            return
        candidates = {"HID": ["HD1"], "HIE": ["HE2"]}
        if allow_hip:
            candidates["HIP"] = ["HD1", "HE2"]
        coords = {"HD1": hd1, "HE2": he2}

        # Atoms that define this His residue (heavy + backbone H), minus ring H.
        residue_atoms = [a for a in self.prot_atoms
                         if a.resnum == resnum and a.icode == icode
                         and a.name not in ("HD1", "HE2")]

        others = [p for p in self.interface_polars
                  if not (p.parent_group_identifier[2] == resnum
                          and p.parent_group_identifier[3] == icode)]
        best = (None, -1, None)  # (which, score, polar_atoms)
        for which, hset in candidates.items():
            polars = self._his_polar_atoms(residue_atoms, hset, coords)
            s = count_interface_hbonds(others + polars, self.ligand_polars)
            # prefer HIE on ties (matches the addHydrogens/LASErMPNN default)
            if s > best[1] or (s == best[1] and which == "HIE"):
                best = (which, s, polars)

        which, _, polars = best
        # commit: replace this His's polar atoms + rewrite output ring H
        self.protein_polars = [
            p for p in self.protein_polars
            if not (p.parent_group_identifier[2] == resnum
                    and p.parent_group_identifier[3] == icode)
        ] + polars
        self._index()
        self._compute_interface()
        self.prot_atoms = [a for a in self.prot_atoms
                           if not (a.resnum == resnum and a.icode == icode
                                   and a.name in ("HD1", "HE2"))]
        chain = next(a.chain for a in residue_atoms)
        for hn in candidates[which]:
            self.prot_atoms.append(_Atom(hn, "H", "HIS", resnum, icode, chain, coords[hn]))

    def _his_polar_atoms(self, residue_atoms, hset, coords) -> list[PolarAtom]:
        """Build a single-residue prody AtomGroup for the given His tautomer and
        run bunsalyze on it so capacities/aromatic flags are correct."""
        atoms = list(residue_atoms) + [
            _Atom(hn, "H", "HIS", residue_atoms[0].resnum, residue_atoms[0].icode,
                  residue_atoms[0].chain, coords[hn]) for hn in hset
        ]
        ag = pr.AtomGroup("his")
        ag.setCoords(np.array([a.coord for a in atoms]))
        ag.setNames(np.array([a.name for a in atoms], dtype=object))
        ag.setElements(np.array([a.element for a in atoms], dtype=object))
        ag.setResnums(np.array([a.resnum for a in atoms]))
        ag.setResnames(np.array([a.resname for a in atoms], dtype=object))
        ag.setChids(np.array([a.chain for a in atoms], dtype=object))
        ag.setIcodes(np.array([a.icode for a in atoms], dtype=object))
        return get_protein_polar_atoms(ag, ncaa_dict=self.ncaa_dict)

    # ------------------------------ output -------------------------------

    def write_pdb(self, path) -> None:
        atoms = self.prot_atoms + self.lig_atoms
        ag = pr.AtomGroup("complex")
        ag.setCoords(np.array([a.coord for a in atoms]))
        ag.setNames(np.array([a.name for a in atoms], dtype=object))
        ag.setElements(np.array([a.element for a in atoms], dtype=object))
        ag.setResnums(np.array([a.resnum for a in atoms]))
        ag.setResnames(np.array([a.resname for a in atoms], dtype=object))
        ag.setChids(np.array([a.chain for a in atoms], dtype=object))
        ag.setIcodes(np.array([a.icode for a in atoms], dtype=object))
        ag.setOccupancies(np.ones(len(atoms)))
        ag.setBetas(np.zeros(len(atoms)))
        pr.writePDB(str(path), ag)

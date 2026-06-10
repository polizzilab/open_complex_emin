"""
Greedy maximization of protein-ligand interface hydrogen bonds.

No force field, no minimization.  Two classes of degrees of freedom are
optimized:

  Single greedy pass (_sweep_protein_rotors):
  * protein Ser/Thr/Tyr/Cys: chi1 rotamer search {-60°, +60°, 180°, current}
    integrated with H dihedral sweep — both scored together so the chi1
    decision is made with previously-committed H atoms already in place.
    A heavy-atom clash gate (< 2.4 Å) filters clashing chi1 candidates.
  * protein Lys ammonium H                                       (H sweep only)

  Additional passes:
  * ligand terminal polar-H rotors                              (dihedral sweep)
  * protein Asn/Gln amide flips                                (180 deg)
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

from .geometry import rotate_about_axis, dihedral
from .ligand_h import find_rotatable_groups
from .protonate_h import place_his_ring_h

_MAXD = S_TO_S_HYDROGEN_BOND_DISTANCE_CUTOFF

# Minimum allowed distance (Å) between a chi1-rotated heavy atom and any
# other heavy atom.  Filters clearly clashing chi1 candidates before H scoring.
_HEAVY_CLASH_DIST = 2.4

# Chi1 rotor definitions: resname -> donor atom that gains H-bond access via chi1.
# chi1 = N-CA-CB-Xγ rotation; moves everything bonded to CB except the backbone core.
_CHI1_ROTORS = {
    "SER": "OG",
    "THR": "OG1",
    "TYR": "OH",
    "CYS": "SG",
}

# Backbone + pivot atom names that are fixed during chi1 rotation.
# Everything else in the residue (beta H, sidechain atoms) rotates.
_BACKBONE_NAMES = frozenset({
    "N", "CA", "C", "O", "OXT", "CB",
    "H", "H1", "H2", "H3", "HA", "HA2", "HA3",
})

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
                 his_tautomers: bool = True, allow_hip: bool = False,
                 chi1: bool = True) -> dict:
        report = {"start": self.score()}
        if sweep_protein:
            self._sweep_protein_rotors(step_deg, chi1=chi1)
        if sweep_ligand:
            self._sweep_ligand_rotors(step_deg)
        if do_flips:
            self._do_protein_flips()
        if his_tautomers:
            self._do_his_tautomers(allow_hip=allow_hip)
        report["final"] = self.score()
        return report

    def _sweep_h_dihedral(
        self,
        resnum: int, icode: str, donor: str,
        h0: dict,
        o: np.ndarray,
        axis: np.ndarray,
        donor_elem: str,
        lig_near: bool,
        other_h: np.ndarray,
        acc_coords: np.ndarray,
        acc_elems: list,
        angles: np.ndarray,
    ) -> tuple:
        """Sweep the H dihedral over all angles and return (best_key, best_theta).

        Ranks each angle by (n_lig, no_clash, geom) — the lexicographic key
        used throughout this module.  Leaves H coords at the last evaluated
        angle; caller is responsible for committing best_theta or reverting.
        """
        best_key: tuple | None = None
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
        return best_key, best_th

    def _sweep_protein_rotors(self, step_deg: float, chi1: bool = True) -> None:
        """Greedy per-residue sweep of rotatable polar H dihedrals.

        For Ser/Thr/Tyr/Cys (when chi1=True) the chi1 rotamer search is
        integrated into this pass: each of the four canonical chi1 angles
        (current, -60°, +60°, 180°) is tried, and for each the H dihedral is
        fully swept with the same (n_lig, no_clash, geom) ranking used for
        H-only residues.  The best (chi1, H) combination is committed.

        Running chi1 and H optimisation in a single greedy pass means each
        residue's chi1 decision is made with previously-visited residues
        already at their optimal H positions — the same scoring context the
        H sweep would have had if chi1 were disabled.
        """
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

            # Other protein H for clash detection — rebuilt each residue so it
            # reflects H positions committed earlier in this sweep.
            other_h_list = [b.coord for b in self.prot_atoms
                            if b.element == "H"
                            and not (b.resnum == resnum and b.icode == icode)]
            other_h = (np.array(other_h_list, dtype=float)
                       if other_h_list else np.empty((0, 3)))

            # Acceptors for geometry tie-breaking.  Use CB as the distance
            # reference (stable across chi1 rotamers) so the list stays valid
            # when the donor moves during the chi1 loop.
            _cb = self.prot_xyz.get((resnum, icode, "CB"))
            ref = _cb if _cb is not None else o
            acc_coords_list: list[np.ndarray] = []
            acc_elems: list[str] = []
            for p in self.protein_polars:
                if (p.max_acceptor_count > 0
                        and not (p.parent_group_identifier[2] == resnum
                                 and p.parent_group_identifier[3] == icode)
                        and np.linalg.norm(np.asarray(p.coord) - ref) < _MAXD):
                    acc_coords_list.append(np.asarray(p.coord))
                    acc_elems.append(p.element)
            lig_near = False
            for lp in self.ligand_polars:
                lc = np.asarray(lp.coord)
                if np.linalg.norm(lc - ref) < _MAXD:
                    lig_near = True
                    if lp.max_acceptor_count > 0:
                        acc_coords_list.append(lc)
                        acc_elems.append(lp.element)
            acc_coords = (np.array(acc_coords_list, dtype=float)
                          if acc_coords_list else np.empty((0, 3)))
            donor_elem = pa.element

            is_interface = any(
                p.parent_group_identifier[2] == resnum
                and p.parent_group_identifier[3] == icode
                for p in self.interface_polars
            )
            if chi1 and resname in _CHI1_ROTORS and is_interface:
                # ---- chi1 + H sweep ----
                n_xyz  = self.prot_xyz.get((resnum, icode, "N"))
                ca_xyz = self.prot_xyz.get((resnum, icode, "CA"))
                cb_xyz = self.prot_xyz.get((resnum, icode, "CB"))
                if n_xyz is None or ca_xyz is None or cb_xyz is None:
                    # Backbone incomplete — fall through to H-only.
                    pass
                else:
                    moving = [(a2.name, a2.element, a2.coord.copy())
                              for a2 in self.prot_atoms
                              if a2.resnum == resnum and a2.icode == icode
                              and a2.name not in _BACKBONE_NAMES]
                    h0_orig = {h.name: h.coord.copy()
                               for h in pa.donor_hydrogens if h.name in hnames}
                    if moving and h0_orig:
                        orig_state = {nm: c.copy() for nm, _el, c in moving}
                        moving_heavy_names = [nm for nm, el, _ in moving if el != "H"]
                        other_heavy_list = (
                            [a2.coord for a2 in self.prot_atoms
                             if a2.element != "H"
                             and not (a2.resnum == resnum and a2.icode == icode)]
                            + [a2.coord for a2 in self.lig_atoms if a2.element != "H"]
                        )
                        all_other_heavy = (np.array(other_heavy_list, dtype=float)
                                           if other_heavy_list else np.empty((0, 3)))
                        chi1_axis = cb_xyz - ca_xyz
                        current_chi1 = dihedral(n_xyz, ca_xyz, cb_xyz, o)

                        overall_best_key: tuple | None = None
                        overall_best_delta = 0.0
                        overall_best_th = 0.0

                        for target_chi1 in (current_chi1,
                                            np.deg2rad(-60.0),
                                            np.deg2rad(60.0),
                                            np.deg2rad(180.0)):
                            delta = target_chi1 - current_chi1
                            new_pos = {nm: rotate_about_axis(c, cb_xyz, chi1_axis, delta)
                                       for nm, _el, c in moving}

                            # Heavy-atom clash gate.
                            if moving_heavy_names and len(all_other_heavy):
                                moved_hvy = np.array(
                                    [new_pos[nm] for nm in moving_heavy_names], dtype=float
                                )
                                if (np.linalg.norm(
                                        all_other_heavy[:, None] - moved_hvy[None], axis=2
                                    ).min() < _HEAVY_CLASH_DIST):
                                    continue

                            new_donor = new_pos[donor]
                            for nm, c in new_pos.items():
                                self._set_prot_coord(resnum, icode, nm, c)
                            self._set_prot_polar_heavy(resnum, icode, donor, new_donor)

                            # H-rotation axis: use the (possibly chi1-moved) pivot.
                            # pivot is CB for SER/THR/CYS (not in moving); CZ for
                            # TYR (in moving, so new_pos carries it correctly).
                            new_piv = new_pos.get(pivot, piv)
                            new_h_axis = new_donor - new_piv
                            new_h0 = {nm: new_pos[nm] for nm in hnames if nm in new_pos}
                            if not new_h0:
                                for nm, c in orig_state.items():
                                    self._set_prot_coord(resnum, icode, nm, c)
                                self._set_prot_polar_heavy(resnum, icode, donor, orig_state[donor])
                                continue

                            best_key, best_th = self._sweep_h_dihedral(
                                resnum, icode, donor, new_h0, new_donor, new_h_axis,
                                donor_elem, lig_near, other_h, acc_coords, acc_elems, angles,
                            )

                            if overall_best_key is None or best_key > overall_best_key:
                                overall_best_key = best_key
                                overall_best_delta = delta
                                overall_best_th = best_th

                            # Revert for next chi1 candidate.
                            for nm, c in orig_state.items():
                                self._set_prot_coord(resnum, icode, nm, c)
                            self._set_prot_polar_heavy(resnum, icode, donor, orig_state[donor])
                            for hn, c0 in h0_orig.items():
                                self._set_prot_donor_h(resnum, icode, donor, hn, c0)

                        # Commit the winning (chi1, H) combination.
                        best_pos = {nm: rotate_about_axis(c, cb_xyz, chi1_axis, overall_best_delta)
                                    for nm, _el, c in moving}
                        best_donor = best_pos[donor]
                        for nm, c in best_pos.items():
                            self._set_prot_coord(resnum, icode, nm, c)
                        self._set_prot_polar_heavy(resnum, icode, donor, best_donor)
                        best_piv = best_pos.get(pivot, piv)
                        best_h_axis = best_donor - best_piv
                        best_h0 = {nm: best_pos[nm] for nm in hnames if nm in best_pos}
                        for hn, c0 in best_h0.items():
                            c = rotate_about_axis(c0, best_donor, best_h_axis, overall_best_th)
                            self._set_prot_donor_h(resnum, icode, donor, hn, c)
                            self._set_prot_coord(resnum, icode, hn, c)
                        continue  # next residue

            # ---- H-only sweep (Lys, chi1=False, or backbone-incomplete chi1) ----
            h0 = {h.name: h.coord.copy() for h in pa.donor_hydrogens if h.name in hnames}
            if not h0:
                continue
            axis = o - piv
            best_key, best_th = self._sweep_h_dihedral(
                resnum, icode, donor, h0, o, axis,
                donor_elem, lig_near, other_h, acc_coords, acc_elems, angles,
            )
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

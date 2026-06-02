"""
Protein-ligand H-bond detection for use as OpenMM flat-bottom restraints.

Geometry criteria and PolarAtom/DonorHydrogen dataclasses are adapted from
https://github.com/polizzilab/bunsalyze (MIT licence).  The torch/burial/SASA
machinery is not used — only the distance+angle validation logic and the
donor/acceptor identification rules.
"""
from __future__ import annotations

import io
import math
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import prody as pr
from rdkit import Chem

# ---------------------------------------------------------------------------
# Constants  (from bunsalyze/utils/constants.py)
# ---------------------------------------------------------------------------

ON_ON_DIST   = 3.3
ON_S_DIST    = 4.0
S_S_DIST     = 4.3
MAX_DIST     = S_S_DIST   # widest cutoff for initial neighbour search
MIN_DIST     = 1.5
MIN_ANGLE    = 110        # D-H···A angle in degrees
H_CLASH_DIST = 1.5
MAX_ARO_PLANAR_ANGLE = 60

DEFAULT_NCAA_DICT = {
    "SEP": {"N": (1, ["H"]), "O": (2, []), "OG": (2, []), "O1P": (2, []), "O2P": (2, []), "O3P": (2, [])},
    "PTR": {"N": (1, ["H"]), "O": (2, []), "OH": (2, []), "O1P": (2, []), "O2P": (2, []), "O3P": (2, [])},
    "DJD": {"N": (0, ["H"]), "O": (2, []), "N03": (1, []), "N04": (1, []), "N05": (1, []), "N06": (1, [])},
}

AA_LONG = {"CYS","ASP","SER","GLN","LYS","ILE","PRO","THR","PHE","ASN",
           "GLY","HIS","LEU","ARG","TRP","ALA","VAL","GLU","TYR","MET","XAA"}
AA_SHORT = {"C":"CYS","D":"ASP","S":"SER","Q":"GLN","K":"LYS","I":"ILE",
            "P":"PRO","T":"THR","F":"PHE","N":"ASN","G":"GLY","H":"HIS",
            "L":"LEU","R":"ARG","W":"TRP","A":"ALA","V":"VAL","E":"GLU",
            "Y":"TYR","M":"MET","X":"XAA"}
AA_L2S = {v: k for k, v in AA_SHORT.items()}

DONOR_MAP = {
    "G": {"CA": [("HA2","1HA"),("HA3","2HA")]},
    "C": {"SG": [("HG",)]},
    "H": {"ND1": [("HD1",)], "NE2": [("HE2",)]},
    "K": {"NZ": [("HZ1","1HZ"),("HZ2","2HZ"),("HZ3","3HZ")]},
    "N": {"ND2": [("HD21","1HD2"),("HD22","2HD2")]},
    "Q": {"NE2": [("HE21","1HE2"),("HE22","2HE2")]},
    "R": {"NE": [("HE",)], "NH1": [("HH11","1HH1"),("HH12","2HH1")],
          "NH2": [("HH21","1HH2"),("HH22","2HH2")]},
    "S": {"OG": [("HG",)]},
    "T": {"OG1": [("HG1",)]},
    "W": {"NE1": [("HE1",)]},
    "Y": {"OH": [("HH",)]},
}
for _aa in AA_SHORT:
    if _aa == "X":
        continue
    DONOR_MAP.setdefault(_aa, {})["N"] = [("H",),("1H","H1"),("2H","H2"),("3H","H3")]

ACCEPTOR_MAP = {
    "G":["O"],"A":["O"],"S":["O","OG"],"C":["O","SG"],"T":["O","OG1"],
    "P":["O"],"V":["O"],"M":["O","SD"],"N":["O","OD1"],"I":["O"],"L":["O"],
    "D":["O","OD1","OD2"],"E":["O","OE1","OE2"],"K":["O"],"Q":["O","OE1"],
    "H":["O","ND1","NE2"],"F":["O"],"R":["O"],"Y":["O","OH"],"W":["O"],
}
for _aa, _lst in ACCEPTOR_MAP.items():
    _lst.append("OXT")

ARO_BONDED = {"H": {"ND1": ["CE1","CG"], "NE2": ["CE1","CD2"]}}


# ---------------------------------------------------------------------------
# Dataclasses  (from bunsalyze/utils/constants.py)
# ---------------------------------------------------------------------------

@dataclass
class DonorHydrogen:
    name: str
    coord: np.ndarray
    is_engaged: bool = False
    engaged_to: Optional[object] = None

    def engage(self, other):
        self.is_engaged = True
        self.engaged_to = other


@dataclass
class BondedHeavyAtom:
    name: str
    element: str
    coord: np.ndarray


@dataclass
class PolarAtom:
    name: str
    coord: np.ndarray
    donor_count: int
    acceptor_count: int
    parent_group_identifier: tuple
    element: str
    is_ligand_atom: bool
    donor_hydrogens: list
    is_aromatic_planar: bool
    covalent_bonded_heavy_atoms: list
    is_weak_acceptor: bool = False
    openmm_index: Optional[int] = None     # OpenMM topology atom index
    max_donor_count: int = field(init=False)
    max_acceptor_count: int = field(init=False)
    is_buried: Optional[bool] = None

    def __post_init__(self):
        self.max_donor_count = self.donor_count
        self.max_acceptor_count = self.acceptor_count


# ---------------------------------------------------------------------------
# H-bond validation  (from bunsalyze/utils/graph.py — no torch)
# ---------------------------------------------------------------------------

def _norm(v):
    return v / max(np.linalg.norm(v), 1e-6)


def is_valid_hbond(
    donor: PolarAtom, h: DonorHydrogen, acceptor: PolarAtom,
    clash_check: bool = True,
) -> bool:
    if h.is_engaged:
        return h.engaged_to is acceptor

    if donor.element in ("N","O") and acceptor.element in ("N","O"):
        cutoff = ON_ON_DIST
    elif set([donor.element, acceptor.element]) in ({"N","S"},{"O","S"}):
        cutoff = ON_S_DIST
    elif donor.element == "S" and acceptor.element == "S":
        cutoff = S_S_DIST
    elif "C" in (donor.element, acceptor.element):
        cutoff = 3.7  # Ca-H donor
    else:
        return False

    if not (donor.donor_count > 0 and acceptor.acceptor_count > 0):
        return False

    d = np.linalg.norm(donor.coord - acceptor.coord)
    if d > cutoff or d < MIN_DIST:
        return False

    dh = h.coord - donor.coord
    ah = h.coord - acceptor.coord
    cos = np.dot(dh, ah) / (np.linalg.norm(dh) * np.linalg.norm(ah) + 1e-9)
    if np.rad2deg(np.arccos(np.clip(cos, -1, 1))) < MIN_ANGLE:
        return False

    if clash_check:
        for ah2 in acceptor.donor_hydrogens:
            if np.linalg.norm(h.coord - ah2.coord) < H_CLASH_DIST:
                return False

    if acceptor.is_aromatic_planar and len(acceptor.covalent_bonded_heavy_atoms) == 2:
        v1 = _norm(acceptor.covalent_bonded_heavy_atoms[0].coord - acceptor.coord)
        v2 = _norm(acceptor.covalent_bonded_heavy_atoms[1].coord - acceptor.coord)
        lp = _norm(-(v1 + v2))
        adh = _norm(h.coord - acceptor.coord)
        if np.rad2deg(np.arccos(np.clip(np.dot(lp, adh), -1, 1))) > MAX_ARO_PLANAR_ANGLE:
            return False

    donor.donor_count -= 1
    acceptor.acceptor_count -= 1
    h.engage(acceptor)
    return True


# ---------------------------------------------------------------------------
# Protein polar atoms  (from bunsalyze/utils/calc_protein_dons_accs.py)
# ---------------------------------------------------------------------------

def _flat(nested):
    return [x for sub in nested for x in sub]


def _protein_polar_atoms(ag: pr.AtomGroup, ncaa_dict: dict, pos_to_idx: dict) -> list[PolarAtom]:
    ncaa = deepcopy(DEFAULT_NCAA_DICT)
    ncaa.update(deepcopy(ncaa_dict))
    result = []

    for res in ag.iterResidues():
        resname = res.getResnames()[0]
        pgid = (str(res.getChids()[0]), resname,
                int(res.getResnums()[0]), str(res.getIcodes()[0]))

        if resname in ncaa:
            for atom_name, (acc_cnt, don_list) in ncaa[resname].items():
                sel = res.select(f"name {atom_name}")
                if sel is None:
                    continue
                coord = sel.getCoords()[0]
                dhs = []
                for hn in don_list:
                    hs = res.select(f"name {hn}")
                    if hs is not None:
                        dhs.append(DonorHydrogen(hn, hs.getCoords()[0]))
                result.append(PolarAtom(
                    name=atom_name, coord=coord, donor_count=len(dhs),
                    acceptor_count=acc_cnt, parent_group_identifier=pgid,
                    element=sel.getElements()[0], is_ligand_atom=False,
                    donor_hydrogens=dhs, is_aromatic_planar=False,
                    covalent_bonded_heavy_atoms=[], is_buried=True,
                    openmm_index=pos_to_idx.get(tuple(coord)),
                ))
            continue

        if resname not in AA_L2S:
            continue
        aa = AA_L2S[resname]
        names = res.getNames()
        coords = res.getCoords()
        polar_set = (set(DONOR_MAP.get(aa, {})) |
                     set(ACCEPTOR_MAP.get(aa, [])))

        for pname, pcoord in zip(names, coords):
            if pname not in polar_set:
                continue
            elem = next((c for c in pname if c.isalpha()), "C")

            dhs = []
            if pname in DONOR_MAP.get(aa, {}):
                all_hnames = set(_flat(DONOR_MAP[aa][pname]))
                hmask = np.array([n in all_hnames for n in names])
                for hn, hc in zip(names[hmask], coords[hmask]):
                    dhs.append(DonorHydrogen(hn, hc))

            acc_cnt = 0
            is_weak = False
            if pname in ACCEPTOR_MAP.get(aa, []):
                acc_cnt = 2
                if aa == "M":
                    acc_cnt, is_weak = 1, True
                if aa == "C":
                    if not dhs:
                        continue
                    acc_cnt = 1
                if aa == "H" and pname in ("ND1","NE2"):
                    acc_cnt = 0 if dhs else 1

            is_aro = (aa == "H" and pname in ("ND1","NE2")
                      and acc_cnt > 0 and not dhs)
            cov = []
            if is_aro:
                for bn in ARO_BONDED["H"][pname]:
                    bs = res.select(f"name {bn}")
                    if bs is not None:
                        cov.append(BondedHeavyAtom(bn, bn[0], bs.getCoords()[0]))

            result.append(PolarAtom(
                name=pname, coord=pcoord, donor_count=len(dhs),
                acceptor_count=acc_cnt, parent_group_identifier=pgid,
                element=elem, is_ligand_atom=False, donor_hydrogens=dhs,
                is_aromatic_planar=is_aro, covalent_bonded_heavy_atoms=cov,
                is_weak_acceptor=is_weak, is_buried=True,
                openmm_index=pos_to_idx.get(tuple(pcoord)),
            ))

    return result


# ---------------------------------------------------------------------------
# Ligand polar atoms  (from bunsalyze/utils/calc_ligand_dons_accs.py)
# ---------------------------------------------------------------------------

_tbl = Chem.GetPeriodicTable()


def _num_lp(atom) -> int:
    v = _tbl.GetNOuterElecs(atom.GetAtomicNum())
    c = atom.GetFormalCharge()
    b = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
    return max(0, math.floor(0.5 * (v - c - b)))


def _ligand_capacity(rdmol: Chem.Mol) -> dict:
    cap = {}
    for atom in rdmol.GetAtoms():
        if atom.GetAtomicNum() not in (7, 8):
            continue
        ri = atom.GetPDBResidueInfo()
        aname = ri.GetName().strip() if ri else f"{atom.GetSymbol()}{atom.GetIdx()}"
        nH = atom.GetTotalNumHs()
        acc = 0
        is_weak = False
        if atom.GetAtomicNum() == 8:
            if atom.GetTotalValence() <= 2 and atom.GetFormalCharge() <= 0:
                acc = _num_lp(atom)
        elif atom.GetAtomicNum() == 7:
            deg = atom.GetTotalDegree()
            hyb = atom.GetHybridization()
            if deg < 4 and atom.GetFormalCharge() == 0:
                if hyb == Chem.rdchem.HybridizationType.SP2 and deg < 3:
                    acc = _num_lp(atom)
                    if atom.GetIsAromatic() and nH == 0:
                        is_weak = True
                if hyb == Chem.rdchem.HybridizationType.SP3 and deg == 3:
                    acc = _num_lp(atom)
        cap[aname] = {"donor": nH, "acceptor": acc, "is_weak": is_weak}
    return cap


def _ligand_polar_atoms(
    cap: dict, lig_ag: pr.AtomGroup, lig_mol: Chem.Mol,
    pos_to_idx: dict, cov_h_dist: float = 1.2,
) -> list[PolarAtom]:
    name_to_rdatom = {
        a.GetPDBResidueInfo().GetName().strip(): a
        for a in lig_mol.GetAtoms()
        if a.GetPDBResidueInfo() is not None
    }
    conf = lig_mol.GetConformer()
    result = []

    for aname, don_acc in cap.items():
        dc, ac = don_acc["donor"], don_acc["acceptor"]
        if dc == 0 and ac == 0:
            continue

        dhs = []
        if dc > 0:
            hs = lig_ag.select(f"element H within {cov_h_dist} of (name {aname})")
            if hs is not None:
                for hn, hc in zip(hs.getNames(), hs.getCoords()):
                    dhs.append(DonorHydrogen(hn, hc))

        sel = lig_ag.select(f"name {aname}")
        if sel is None:
            continue
        coord = sel.getCoords()[0]
        pgid = (str(sel.getChids()[0]), str(sel.getResnames()[0]),
                int(sel.getResnums()[0]), str(sel.getIcodes()[0]))
        elem = sel.getElements()[0]

        cov = []
        rdatom = name_to_rdatom.get(aname)
        if rdatom is not None:
            for bond in rdatom.GetBonds():
                other = bond.GetOtherAtom(rdatom)
                if other.GetAtomicNum() > 1:
                    ori = other.GetPDBResidueInfo()
                    on = ori.GetName().strip() if ori else f"{other.GetSymbol()}{other.GetIdx()}"
                    oc = np.array(conf.GetAtomPosition(other.GetIdx()))
                    cov.append(BondedHeavyAtom(on, other.GetSymbol(), oc))

            is_aro = (
                len(cov) == 2
                and rdatom.GetHybridization() == Chem.rdchem.HybridizationType.SP2
                and dc == 0 and ac > 0
            )
        else:
            is_aro = False

        result.append(PolarAtom(
            name=aname, coord=coord, donor_count=dc, acceptor_count=ac,
            parent_group_identifier=pgid, element=elem, is_ligand_atom=True,
            donor_hydrogens=dhs, is_aromatic_planar=is_aro,
            covalent_bonded_heavy_atoms=cov,
            is_weak_acceptor=don_acc.get("is_weak", False),
            is_buried=True,
            openmm_index=pos_to_idx.get(tuple(coord)),
        ))

    return result


# ---------------------------------------------------------------------------
# Cross-interface detection
# ---------------------------------------------------------------------------

def _find_cross_interface_hbonds(
    prot_polars: list[PolarAtom],
    lig_polars: list[PolarAtom],
    clash_check: bool = True,
) -> list[tuple[PolarAtom, PolarAtom, float]]:
    """
    Return (donor_heavy, acceptor_heavy, d0_Å) for every protein-ligand H-bond.
    Donor and acceptor can be on either molecule.
    """
    pairs = []

    def _check(p_atom, l_atom):
        d = np.linalg.norm(p_atom.coord - l_atom.coord)
        if d > MAX_DIST:
            return
        for h in sorted(p_atom.donor_hydrogens,
                        key=lambda x: np.linalg.norm(x.coord - l_atom.coord)):
            if is_valid_hbond(p_atom, h, l_atom, clash_check):
                pairs.append((p_atom, l_atom, d))
        for h in sorted(l_atom.donor_hydrogens,
                        key=lambda x: np.linalg.norm(x.coord - p_atom.coord)):
            if is_valid_hbond(l_atom, h, p_atom, clash_check):
                pairs.append((l_atom, p_atom, d))

    for pp in prot_polars:
        for lp in lig_polars:
            _check(pp, lp)

    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_protein_ligand_hbonds(
    combined_pdb_str: str,
    positions_ang: np.ndarray,    # (N_atoms, 3) in Å, OpenMM atom order
    smiles: str,
    ligand_resname: str = "LIG",
    ncaa_dict: dict | None = None,
    clash_check: bool = True,
    rdmol: Chem.Mol | None = None,   # pre-parsed mol (preferred; skips SMILES reassignment)
) -> list[tuple[int, int, float]]:
    """
    Detect protein–ligand H-bonds using bunsalyze geometry criteria.

    Parameters
    ----------
    combined_pdb_str:
        PDB string of the protonated protein + ligand complex.
    positions_ang:
        OpenMM atom positions in Angstroms, indexed by OpenMM atom index.
    smiles:
        Ligand SMILES (used to assign bond orders for capacity calculation).
    ligand_resname:
        Residue name of the ligand (default "LIG").

    Returns
    -------
    List of (donor_heavy_omm_idx, acceptor_omm_idx, d0_Å) for every
    detected protein-ligand H-bond.  These indices are into the OpenMM
    topology / positions arrays.
    """
    if ncaa_dict is None:
        ncaa_dict = {}

    # Build a coord→OpenMM-index lookup (positions are unique to <0.01 Å)
    pos_to_idx = {tuple(np.round(pos, 3)): i for i, pos in enumerate(positions_ang)}

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(combined_pdb_str)
        tmp_path = f.name

    ag = pr.parsePDB(tmp_path)
    prot_sel = ag.select("protein")
    lig_sel  = ag.select(f"resname {ligand_resname}")

    if prot_sel is None or lig_sel is None:
        return []

    prot_ag = prot_sel.copy()
    lig_ag  = lig_sel.copy()

    prot_polars = _protein_polar_atoms(prot_ag, ncaa_dict, pos_to_idx)

    # Build RDKit mol with PDB residue info for capacity calc.
    # Prefer the caller-supplied mol (from LigandParams, already correctly parsed
    # from SDF) so bond orders are always right.  Only fall back to re-parsing
    # from prody+SMILES if no mol is provided.
    if rdmol is not None:
        lig_mol = _align_rdmol_to_prody(rdmol, lig_ag)
    else:
        lig_mol = _rdmol_from_prody(lig_ag, smiles)
    lig_cap = _ligand_capacity(lig_mol)
    lig_polars = _ligand_polar_atoms(lig_cap, lig_ag, lig_mol, pos_to_idx)

    raw_pairs = _find_cross_interface_hbonds(prot_polars, lig_polars, clash_check)

    results = []
    for donor, acceptor, d0 in raw_pairs:
        di = donor.openmm_index
        ai = acceptor.openmm_index
        if di is not None and ai is not None:
            results.append((di, ai, d0))

    return results


def _align_rdmol_to_prody(rdmol: Chem.Mol, lig_ag: pr.AtomGroup) -> Chem.Mol:
    """
    Return a copy of rdmol with its conformer replaced by the prody coordinates.

    Atoms are matched by nearest-neighbour position (both structures share the
    same heavy-atom geometry; only H positions may differ slightly after pdb2pqr).
    The prody AtomGroup has H atoms placed by pdb2pqr; we need those H coords
    for accurate donor-hydrogen detection.  We therefore rebuild an RDKit mol
    that has:
      - bond orders / aromaticity from the original rdmol (correct chemistry)
      - 3-D coordinates from lig_ag (pdb2pqr-placed positions)
    """
    from rdkit.Chem import AllChem, RWMol

    mol = Chem.RWMol(Chem.RenumberAtoms(rdmol, list(range(rdmol.GetNumAtoms()))))
    conf = mol.GetConformer()

    # Build a coord→prody-name map from prody
    prody_coords = lig_ag.getCoords()      # (N, 3) Å
    prody_names  = lig_ag.getNames()
    prody_elems  = lig_ag.getElements()

    # For each rdmol atom, find the closest prody atom of the same element
    # and update its position + PDB residue info
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        rdcoord = np.array(conf.GetAtomPosition(atom.GetIdx()))

        # Restrict candidates to same element
        mask = np.array([e.upper() == elem.upper() for e in prody_elems])
        if not mask.any():
            continue
        cands  = prody_coords[mask]
        cnames = prody_names[mask]
        dists  = np.linalg.norm(cands - rdcoord, axis=1)
        best   = int(np.argmin(dists))

        # Update conformer coordinate
        pc = cands[best]
        conf.SetAtomPosition(atom.GetIdx(), (float(pc[0]), float(pc[1]), float(pc[2])))

        # Attach PDB residue info so atom name lookup works downstream
        ri = atom.GetPDBResidueInfo()
        if ri is None:
            ri = Chem.AtomPDBResidueInfo()
        ri.SetName(f" {cnames[best]:<3s}")
        atom.SetMonomerInfo(ri)

    # Now add any prody atoms (H) that aren't in the original rdmol
    # as bare atoms so DonorHydrogen coords can be found by prody selection.
    # (We don't need them in the rdmol; get_ligand_polar_atoms fetches H
    #  coords via a prody proximity search, not from the rdmol conformer.)
    return mol.GetMol()


def _rdmol_from_prody(lig_ag: pr.AtomGroup, smiles: str) -> Chem.Mol:
    """Build an RDKit mol from a prody ligand AtomGroup with bond orders from SMILES."""
    pdb_str = io.StringIO()
    pr.writePDBStream(pdb_str, lig_ag)
    raw_mol = Chem.MolFromPDBBlock(pdb_str.getvalue(), removeHs=False, sanitize=False)
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return raw_mol
    try:
        mol = Chem.AllChem.AssignBondOrdersFromTemplate(template, raw_mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return raw_mol

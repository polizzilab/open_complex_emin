"""
RDKit detection of the ligand degrees of freedom the fast track optimizes:

  * terminal rotatable polar-H groups  (hydroxyl, carboxyl, thiol, sp3 amine/
    ammonium)  -- the H(s) rotate about the single bond to their one heavy
    neighbour.

Ligand amide flips are intentionally NOT handled (only protein Asn/Gln flips are).

Groups are described by ATOM NAME (the common key across the RDKit mol's
PDBResidueInfo, the prody ligand AtomGroup, and bunsalyze PolarAtom/DonorHydrogen
objects), so the sweep can look coordinates up by name without tracking three
sets of indices.

The input mol must carry PDBResidueInfo atom names and explicit H.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass
class RotatableGroup:
    """Terminal polar-H rotor: H atoms rotate about the pivot->donor bond axis."""
    donor: str            # heavy atom bearing the H(s); rotation origin
    pivot: str            # its single heavy neighbour; defines the axis
    hydrogens: list[str]


def _name(atom: Chem.Atom) -> str:
    ri = atom.GetPDBResidueInfo()
    if ri is not None:
        return ri.GetName().strip()
    return f"{atom.GetSymbol()}{atom.GetIdx()}"


def find_rotatable_groups(mol: Chem.Mol) -> list[RotatableGroup]:
    groups: list[RotatableGroup] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (7, 8, 16):  # N, O, S
            continue
        hs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if not hs or len(heavy) != 1:
            continue
        pivot = heavy[0]
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), pivot.GetIdx())
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue  # e.g. imine =N-H is not a free rotor
        # sp2 amide / conjugated N-H is handled by the flip path (or is rigid),
        # so exclude an sp2 N whose neighbour carries a double bond to O/N.
        if atom.GetAtomicNum() == 7 and atom.GetHybridization() == Chem.HybridizationType.SP2:
            conjugated = any(
                b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(pivot).GetAtomicNum() in (7, 8)
                for b in pivot.GetBonds()
            )
            if conjugated:
                continue
        groups.append(RotatableGroup(
            donor=_name(atom), pivot=_name(pivot),
            hydrogens=[_name(h) for h in hs],
        ))
    return groups

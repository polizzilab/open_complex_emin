"""Ligand parameter preparation: GFN2-xTB partial charges + GAFF2 FFXML."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

_BOHR_PER_ANG = 1.8897259886


@dataclass
class LigandParams:
    """
    Per-ligand-identity parameters.  Picklable — safe for multiprocessing.

    All expensive computation (xTB charges, antechamber/GAFF2) runs once in
    prepare_ligand().  Workers call minimize_complex() with the same
    LigandParams instance and pay no per-structure parameterisation cost.
    """

    smiles: str
    charges: list[float]   # GFN2-xTB Mulliken charges, atom order = rdmol atom order
    mol_block: str         # MDL V2000 mol block, removeHs=False
    gaff_xml: str          # pre-computed GAFF2 FFXML; loaded directly into ForceField

    @property
    def mol(self) -> Chem.Mol:
        return Chem.MolFromMolBlock(self.mol_block, removeHs=False, sanitize=True)

    @property
    def n_atoms(self) -> int:
        return len(self.charges)


def prepare_ligand(mol_source: str, *, is_file: bool = False) -> LigandParams:
    """
    Compute GFN2-xTB partial charges and GAFF2 parameters for the ligand.

    Parameters
    ----------
    mol_source:
        SMILES string, or path to a .sdf / .mol file when is_file=True.
    is_file:
        Treat mol_source as a file path.

    Notes
    -----
    When loading from a file the 3-D coordinates are used as-is.
    When loading from SMILES a conformer is generated via ETKDGv3 + MMFF94.

    Both the xTB charge calculation (tblite) and the GAFF2 parameterisation
    (antechamber subprocess via openmmforcefields) run here.  Call this once
    per ligand identity and reuse the result across all structures.
    """
    if is_file:
        mol = Chem.MolFromMolFile(mol_source, removeHs=False, sanitize=True)
        if mol is None:
            raise ValueError(f"Could not parse mol file: {mol_source}")
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    else:
        mol = Chem.MolFromSmiles(mol_source)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {mol_source}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("RDKit could not generate a 3-D conformer from SMILES.")
        AllChem.MMFFOptimizeMolecule(mol)
        smiles = mol_source

    charges = _xtb_charges(mol)

    from .gaff import build_gaff2_ffxml
    gaff_xml = build_gaff2_ffxml(mol, charges)

    return LigandParams(
        smiles=smiles,
        charges=charges,
        mol_block=Chem.MolToMolBlock(mol),
        gaff_xml=gaff_xml,
    )


def _xtb_charges(mol: Chem.Mol) -> list[float]:
    from tblite.interface import Calculator

    conf = mol.GetConformer()
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    positions = conf.GetPositions() * _BOHR_PER_ANG
    total_charge = int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))

    calc = Calculator("GFN2-xTB", numbers, positions, charge=total_charge)
    calc.set("verbosity", 0)
    return calc.singlepoint().get("charges").tolist()

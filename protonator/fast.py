"""
Fast track: protonate + maximize protein-ligand interface H-bonds, no OpenMM.

Pipeline (per structure):
  1. split chains  (A = protein heavy atoms, B = ligand)
  2. place polar protein H geometrically          (protonate_h, torch-less)
  3. build the ligand RDKit mol + prody AtomGroup  (ligand.build_ligand_mol)
  4. greedy interface-H-bond optimization          (sweep.InterfaceModel)
  5. write the protonated, optimized complex PDB

No force field is built, so no xTB charges and no GAFF2/antechamber are needed.
Heavy atoms and non-rotatable groups never move.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import prody as pr
from rdkit import Chem

from .ligand import build_ligand_mol
from .protonate_h import place_protein_hydrogens
from .sweep import InterfaceModel


def _name_ligand_mol(mol: Chem.Mol, resname: str = "LIG", chain: str = "B") -> Chem.Mol:
    """Assign unique element+index PDB names + residue info to every atom."""
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        el = atom.GetSymbol()
        counts[el] = counts.get(el, 0) + 1
        name = f"{el}{counts[el]}"
        ri = Chem.AtomPDBResidueInfo()
        ri.SetName(name if len(name) >= 4 else f" {name:<3s}")
        ri.SetResidueName(resname)
        ri.SetResidueNumber(1)
        ri.SetChainId(chain)
        ri.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(ri)
    return mol


def _ligand_atomgroup(mol: Chem.Mol, resname: str = "LIG", chain: str = "B") -> pr.AtomGroup:
    conf = mol.GetConformer()
    names, elems, coords = [], [], []
    for atom in mol.GetAtoms():
        names.append(atom.GetPDBResidueInfo().GetName().strip())
        elems.append(atom.GetSymbol())
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    n = len(names)
    ag = pr.AtomGroup("ligand")
    ag.setCoords(np.array(coords))
    ag.setNames(np.array(names, dtype=object))
    ag.setElements(np.array(elems, dtype=object))
    ag.setResnums(np.ones(n, dtype=int))
    ag.setResnames(np.array([resname] * n, dtype=object))
    ag.setChids(np.array([chain] * n, dtype=object))
    ag.setIcodes(np.array([""] * n, dtype=object))
    return ag


def fast_protonate(
    pdb_path: str | Path,
    output_path: str | Path,
    *,
    smiles: str | None = None,
    ligand_file: str | Path | None = None,
    apo: bool = False,
    step_deg: float = 10.0,
    sweep_protein: bool = True,
    sweep_ligand: bool = True,
    do_flips: bool = True,
    his_tautomers: bool = True,
    ncaa_dict: dict | None = None,
) -> dict:
    """Run the fast protonation/H-bond-optimization track.  Returns a small
    report dict (H-bond counts before/after)."""
    pdb_path = Path(pdb_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    complex_ag = pr.parsePDB(str(pdb_path))
    prot_in = complex_ag.select("chain A")
    if prot_in is None:
        raise ValueError("No chain A (protein) found in input PDB.")
    protein_ag = place_protein_hydrogens(prot_in.copy())

    if apo:
        pr.writePDB(str(output_path), protein_ag)
        return {"mode": "apo", "n_atoms": protein_ag.numAtoms()}

    # ---- ligand ----
    if ligand_file is not None:
        mol, _ = build_ligand_mol(str(ligand_file), is_file=True)
    elif smiles is not None:
        lig_block = _extract_chain_block(pdb_path, "B")
        mol, _ = build_ligand_mol(smiles, pdb_ligand_block=lig_block)
    else:
        raise ValueError("Holo mode needs --smiles or --ligand-file (or use apo=True).")

    mol = _name_ligand_mol(mol)
    ligand_ag = _ligand_atomgroup(mol)

    model = InterfaceModel(protein_ag, ligand_ag, mol, ncaa_dict=ncaa_dict)
    report = model.optimize(
        step_deg=step_deg, sweep_protein=sweep_protein, sweep_ligand=sweep_ligand,
        do_flips=do_flips, his_tautomers=his_tautomers,
    )
    model.write_pdb(output_path)
    report["mode"] = "holo"
    return report


def _extract_chain_block(pdb_path: Path, chain_id: str) -> str:
    lines = [
        line for line in pdb_path.read_text().splitlines()
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) > 21 and line[21] == chain_id
    ]
    return "\n".join(lines) + "\nEND\n"

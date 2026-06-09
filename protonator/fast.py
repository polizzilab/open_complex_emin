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
from rdkit.Chem import AllChem

from .ligand import build_ligand_mol
from .protonate_h import place_protein_hydrogens
from .sweep import InterfaceModel


def _name_ligand_mol(mol: Chem.Mol, resname: str = "LIG", chain: str = "B") -> Chem.Mol:
    """Assign unique element+index PDB names + residue info to every atom (fallback)."""
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


def _remap_lig_names(
    mol: Chem.Mol,
    ref_pdb: Path,
    smiles: str,
    resname: str = "LIG",
    chain: str = "B",
) -> Chem.Mol:
    """
    Apply the original ligand atom names from chain B of ref_pdb to mol's heavy atoms.

    Uses symmetry-aware substructure matching (same approach as
    calc_symmetry_aware_rmsd.py) so that symmetrically-equivalent heavy atoms
    get their correct original names.  The best match is chosen by minimising
    the coordinate RMSD between matched heavy atom pairs.

    H atoms always receive fresh element+index names (they are absent from the
    original AF3 PDB and have no canonical naming convention).

    Falls back to element+index naming for all atoms if parsing or matching fails.
    """
    # Build reference mol from chain B of the original input PDB.
    lig_lines = [
        line for line in ref_pdb.read_text().splitlines()
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) > 21 and line[21] == chain
    ]
    if not lig_lines:
        return _name_ligand_mol(mol, resname=resname, chain=chain)

    ref_pdb_block = "\n".join(lig_lines) + "\nEND\n"
    ref_raw = Chem.MolFromPDBBlock(ref_pdb_block, removeHs=True, sanitize=False)
    smi_mol = Chem.MolFromSmiles(smiles)
    if ref_raw is None or smi_mol is None:
        return _name_ligand_mol(mol, resname=resname, chain=chain)

    try:
        ref_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, ref_raw)
        Chem.SanitizeMol(ref_mol)
    except Exception:
        return _name_ligand_mol(mol, resname=resname, chain=chain)

    mol_noH = Chem.RemoveHs(mol)
    if ref_mol.GetNumAtoms() != mol_noH.GetNumAtoms():
        return _name_ligand_mol(mol, resname=resname, chain=chain)

    # ref_mol.GetSubstructMatches(mol_noH): for each match m,
    #   m[i] = index in ref_mol that corresponds to mol_noH atom i.
    matches = ref_mol.GetSubstructMatches(mol_noH, uniquify=False)
    if not matches:
        return _name_ligand_mol(mol, resname=resname, chain=chain)

    # Build coordinate arrays for RMSD-based best-match selection.
    ref_conf = ref_mol.GetConformer()
    noH_conf = mol_noH.GetConformer()
    ref_xyz = np.array([[*ref_conf.GetAtomPosition(i)] for i in range(ref_mol.GetNumAtoms())])
    noH_xyz = np.array([[*noH_conf.GetAtomPosition(i)] for i in range(mol_noH.GetNumAtoms())])

    best_rmsd, best_match = np.inf, matches[0]
    for m in matches:
        rmsd = float(np.sqrt(((ref_xyz[list(m)] - noH_xyz) ** 2).sum(1).mean()))
        if rmsd < best_rmsd:
            best_rmsd, best_match = rmsd, m

    # Original name for each ref_mol atom index.
    ref_idx_to_name: list[str] = []
    for atom in ref_mol.GetAtoms():
        ri = atom.GetPDBResidueInfo()
        ref_idx_to_name.append(ri.GetName().strip() if ri else f"{atom.GetSymbol()}{atom.GetIdx()}")

    # mol_noH atom i  →  original name  →  mol atom index (heavy atoms in order).
    noH_to_name = {i: ref_idx_to_name[ref_idx] for i, ref_idx in enumerate(best_match)}
    heavy_mol_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    mol_idx_to_name = {heavy_mol_idxs[i]: name for i, name in noH_to_name.items()}

    # Apply: original names for heavy atoms, fresh element+index for H.
    h_counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        el = atom.GetSymbol()
        if idx in mol_idx_to_name:
            name = mol_idx_to_name[idx]
        else:
            h_counts[el] = h_counts.get(el, 0) + 1
            name = f"{el}{h_counts[el]}"
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
    allow_hip: bool = False,
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
        mol, lig_smiles = build_ligand_mol(str(ligand_file), is_file=True)
    elif smiles is not None:
        lig_block = _extract_chain_block(pdb_path, "B")
        mol, lig_smiles = build_ligand_mol(smiles, pdb_ligand_block=lig_block)
        lig_smiles = smiles  # keep the user's SMILES (canonical form may differ)
    else:
        raise ValueError("Holo mode needs --smiles or --ligand-file (or use apo=True).")

    # Preserve the original chain-B atom names from the input PDB so that the
    # fast-track output is directly comparable to the rigorous-track output by
    # bunsalyze and other tools that expect protenix-style atom names.
    mol = _remap_lig_names(mol, pdb_path, lig_smiles)
    ligand_ag = _ligand_atomgroup(mol)

    model = InterfaceModel(protein_ag, ligand_ag, mol, ncaa_dict=ncaa_dict)
    report = model.optimize(
        step_deg=step_deg, sweep_protein=sweep_protein, sweep_ligand=sweep_ligand,
        do_flips=do_flips, his_tautomers=his_tautomers, allow_hip=allow_hip,
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

"""
OpenMM energy minimisation of a protein–ligand complex.

Protein backbone and ligand heavy atoms are harmonically restrained;
protein sidechains and all hydrogens (including ligand H) are free.
Implicit solvent: GBn2.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

from .gaff import make_gaff2_generator
from .ligand import LigandParams, prepare_ligand

_BACKBONE = {"N", "CA", "C", "O", "OXT"}


def minimize_complex(
    pdb_path: str | Path,
    ligand_params: LigandParams,
    output_path: str | Path,
    *,
    recompute_ligand: bool = False,
    restraint_k: float = 50.0,   # kcal/mol/Å²
    ph: float = 7.4,
    tolerance: float = 10.0,     # kJ/mol/nm — convergence criterion
    max_iterations: int = 0,     # 0 = until convergence
    freeze_ligand: bool = True,  # restrain ligand heavy atoms; False lets ligand relax
    sweep_hbonds: bool = True,   # post-minimisation hydroxyl sweep for SER/THR
) -> None:
    """
    Protonate, flip-optimise, and energy-minimise a protein–ligand complex.

    Parameters
    ----------
    pdb_path:
        Input PDB.  Chain A = protein (no H), Chain B = ligand (H already placed).
    ligand_params:
        Pre-computed per-ligand parameters from prepare_ligand().
    output_path:
        Destination PDB for the relaxed structure.
    recompute_ligand:
        If True, recompute xTB charges and GAFF2 template from scratch.
    restraint_k:
        Force constant for harmonic position restraints (kcal/mol/Å²).
    ph:
        pH for protonation state assignment (passed to Modeller.addHydrogens).
    tolerance:
        L-BFGS convergence threshold (kJ/mol/nm).
    max_iterations:
        Maximum minimisation steps; 0 means run until convergence.
    freeze_ligand:
        If True (default), restrain ligand heavy atoms in addition to backbone.
    """
    pdb_path = Path(pdb_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if recompute_ligand:
        ligand_params = prepare_ligand(ligand_params.smiles)

    # ------------------------------------------------------------------
    # 1. Split chains
    # ------------------------------------------------------------------
    pdb_text = pdb_path.read_text()
    protein_text = _extract_chain(pdb_text, "A")
    ligand_text  = _extract_chain(pdb_text, "B")

    # ------------------------------------------------------------------
    # 2. Load protein (no H) + ligand into OpenMM
    #    Always write the ligand from ligand_params.mol so that explicit H
    #    atoms are present even when the input PDB has none.  The mol's heavy-
    #    atom coordinates match the input (loaded from SDF or via
    #    AssignBondOrdersFromTemplate); H coords come from RDKit AddHs.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        prot_path = Path(td) / "protein.pdb"
        lig_path  = Path(td) / "ligand.pdb"
        prot_path.write_text(protein_text)
        lig_path.write_text(_mol_to_ligand_pdb(ligand_params.mol))
        prot_pdb = app.PDBFile(str(prot_path))
        lig_pdb  = app.PDBFile(str(lig_path))

    # ------------------------------------------------------------------
    # 3. Build force field (needed before addHydrogens so it can evaluate
    #    H-bond geometry for His tautomer selection)
    # ------------------------------------------------------------------
    ff = app.ForceField("amber/ff14SB.xml", "implicit/gbn2.xml")
    ff.registerTemplateGenerator(
        make_gaff2_generator(ligand_params.gaff_xml)
    )

    # ------------------------------------------------------------------
    # 4. Combine and add protein hydrogens
    #    Modeller.addHydrogens sees the full complex (protein + ligand)
    #    and uses H-bond geometry to pick His tautomers (HID vs HIE),
    #    so it will correctly identify H-bonds to ligand acceptors.
    # ------------------------------------------------------------------
    modeller = app.Modeller(prot_pdb.topology, prot_pdb.positions)
    modeller.add(lig_pdb.topology, lig_pdb.positions)
    modeller.addHydrogens(ff, pH=ph)

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )

    # ------------------------------------------------------------------
    # 5. Position restraints: backbone heavy atoms + ligand heavy atoms
    # ------------------------------------------------------------------
    _add_restraints(
        system, modeller.topology, modeller.positions,
        restraint_k, freeze_ligand=freeze_ligand,
    )

    # ------------------------------------------------------------------
    # 6. Minimise
    # ------------------------------------------------------------------
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    import os
    cpu_threads = str(int(os.environ.get("OMP_NUM_THREADS", 1)))
    platform = openmm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(modeller.topology, system, integrator, platform,
                         {"Threads": cpu_threads})
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(
        tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )

    # ------------------------------------------------------------------
    # 7. Optional: sweep SER/THR hydroxyl orientations to prefer ligand H-bonds
    # ------------------------------------------------------------------
    state = sim.context.getState(getPositions=True)
    final_pos = state.getPositions()

    if sweep_hbonds:
        pos_nm = np.array([[v.x, v.y, v.z] for v in final_pos.value_in_unit(unit.nanometer)])
        pos_nm = _sweep_ser_thr(modeller.topology, pos_nm, rdmol=ligand_params.mol)
        final_pos = unit.Quantity(
            [openmm.Vec3(*row) for row in pos_nm], unit.nanometer
        )

    # ------------------------------------------------------------------
    # 8. Write output
    # ------------------------------------------------------------------
    with open(output_path, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, final_pos, fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_chain(pdb_text: str, chain_id: str) -> str:
    lines = [
        line for line in pdb_text.splitlines()
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) > 21 and line[21] == chain_id
    ]
    lines.append("END")
    return "\n".join(lines)


def _add_restraints(
    system: openmm.System,
    topology: app.Topology,
    positions,
    k: float,
    freeze_ligand: bool = True,
) -> None:
    """Harmonic restraints on protein backbone (always) and optionally ligand heavy atoms."""
    k_val = (
        (k * unit.kilocalories_per_mole / unit.angstrom**2)
        .value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)
    )

    force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k_val)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    pos_nm = np.array(
        [[v.x, v.y, v.z] for v in positions.value_in_unit(unit.nanometer)]
    )

    for atom in topology.atoms():
        is_backbone = (
            atom.residue.chain.id == "A"
            and atom.name in _BACKBONE
            and atom.element is not None
            and atom.element.symbol != "H"
        )
        is_lig_heavy = (
            freeze_ligand
            and atom.residue.name == "LIG"
            and atom.element is not None
            and atom.element.symbol != "H"
        )
        if is_backbone or is_lig_heavy:
            x, y, z = pos_nm[atom.index]
            force.addParticle(atom.index, [x, y, z])

    system.addForce(force)


def _mol_to_ligand_pdb(rdmol, resname: str = "LIG", chain: str = "B") -> str:
    """
    Write an RDKit molecule as a PDB string with HETATM + CONECT records.

    Uses the molecule's existing conformer coordinates.  If H atoms are absent
    they are added with 3-D coordinates via AddHs(addCoords=True) before writing.
    This ensures the ligand always has explicit H regardless of the input PDB.
    """
    from rdkit.Chem import AllChem

    mol = rdmol
    if mol.GetNumAtoms() == sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1):
        # No H atoms present — add them with geometry
        mol = AllChem.AddHs(mol, addCoords=True)

    lines = []
    conf = mol.GetConformer()
    serial = 1
    serial_map: dict[int, int] = {}   # rdmol atom idx → PDB serial

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        elem = atom.GetSymbol()
        # Atom name: use PDB residue info name if available, else element+count
        ri = atom.GetPDBResidueInfo()
        if ri:
            name = ri.GetName().strip()
        else:
            name = f"{elem}{serial}"
        # PDB HETATM format
        name_field = f" {name:<3s}" if len(name) < 4 else name[:4]
        lines.append(
            f"HETATM{serial:5d} {name_field:<4s} {resname:3s} {chain}{1:4d}    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00          {elem:>2s}  "
        )
        serial_map[atom.GetIdx()] = serial
        serial += 1

    # CONECT records
    for bond in mol.GetBonds():
        i = serial_map[bond.GetBeginAtomIdx()]
        j = serial_map[bond.GetEndAtomIdx()]
        lines.append(f"CONECT{i:5d}{j:5d}")
        lines.append(f"CONECT{j:5d}{i:5d}")

    lines.append("END")
    return "\n".join(lines) + "\n"


def _add_conect_records(ligand_pdb_text: str, rdmol) -> str:
    """
    Append PDB CONECT records derived from rdmol bonds to a ligand PDB string.

    OpenMM cannot infer bonds for unknown HETATM residues; CONECT records
    supply the connectivity so the graph-isomorphism match in
    GAFFTemplateGenerator can succeed.
    """
    serials = [
        int(line[6:11])
        for line in ligand_pdb_text.splitlines()
        if line[:6].strip() in ("ATOM", "HETATM")
    ]

    adj: dict[int, list[int]] = {s: [] for s in serials}
    for bond in rdmol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[serials[i]].append(serials[j])
        adj[serials[j]].append(serials[i])

    conect = [
        f"CONECT{s:5d}" + "".join(f"{p:5d}" for p in partners)
        for s, partners in adj.items()
        if partners
    ]

    body = [l for l in ligand_pdb_text.splitlines() if not l.startswith("END")]
    return "\n".join(body + conect + ["END"])


# ---------------------------------------------------------------------------
# SER / THR hydroxyl sweep
# ---------------------------------------------------------------------------

# Hydroxyl donor atom for each residue, and the two heavy atoms that define
# the rotation axis (axis = bonded_to_O → O_atom).
_HYDROXYL = {
    "SER": ("OG",  "HG",  "CB"),
    "THR": ("OG1", "HG1", "CB"),
    "TYR": ("OH",  "HH",  "CZ"),  # included for completeness
}
# Ligand elements that can act as H-bond acceptors (fluorine excluded per bunsalyze rules)
_ACCEPTOR_ELEMENTS = {"O", "N", "S"}


def _rotate(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate *point* around *axis* through *origin* by *theta* radians (Rodrigues)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = point - origin
    return origin + (v * np.cos(theta)
                     + np.cross(axis, v) * np.sin(theta)
                     + axis * np.dot(axis, v) * (1 - np.cos(theta)))


def _build_lig_acceptor_info(
    rdmol: "Chem.Mol",
    topology: app.Topology,
    pos_nm: np.ndarray,
) -> dict[int, tuple[str, str, bool, list]]:
    """
    For each ligand acceptor atom in the topology, return a dict:
        topology_atom_idx → (atom_name, element, is_aromatic_planar, covalent_bonded_heavy_atoms)

    is_aromatic_planar and covalent_bonded_heavy_atoms follow bunsalyze's
    calc_ligand_dons_accs.py: is_aromatic_planar is True for sp2 N with exactly
    2 heavy-atom bonds and no H (lone-pair geometry check needed in is_valid_hbond).
    Bonded heavy atom coords come from the current topology positions (post-
    minimisation) so the geometry is consistent with the sweep.
    """
    from rdkit import Chem as _Chem
    from .hbonds import BondedHeavyAtom

    ANG = 10.0  # nm → Å

    # Match topology ligand atoms to rdmol atoms by nearest position
    mol_pos_ang = rdmol.GetConformer().GetPositions()   # (N, 3) Å
    topo_lig = [
        a for a in topology.atoms()
        if a.residue.name == "LIG" and a.element is not None
    ]
    topo_pos_ang = np.array([pos_nm[a.index] * ANG for a in topo_lig])

    # Build topo_atom_index → rdmol_atom_index by nearest position
    topo_to_rdmol: dict[int, int] = {}
    for ti, ta in enumerate(topo_lig):
        dists = np.linalg.norm(mol_pos_ang - topo_pos_ang[ti], axis=1)
        topo_to_rdmol[ta.index] = int(np.argmin(dists))

    # Build rdmol_idx → topo_atom for reverse lookup
    rdmol_to_topo: dict[int, int] = {v: k for k, v in topo_to_rdmol.items()}

    result: dict[int, tuple] = {}
    for topo_atom in topo_lig:
        elem = topo_atom.element.symbol
        if elem not in _ACCEPTOR_ELEMENTS:
            continue

        rdidx = topo_to_rdmol[topo_atom.index]
        rdatom = rdmol.GetAtomWithIdx(rdidx)

        n_heavy_bonds = sum(
            1 for b in rdatom.GetBonds()
            if b.GetOtherAtom(rdatom).GetAtomicNum() > 1
        )
        hyb = rdatom.GetHybridization()
        nH  = rdatom.GetTotalNumHs()

        # Bunsalyze rule: sp2, exactly 2 heavy neighbours, pure acceptor
        is_planar = (
            n_heavy_bonds == 2
            and hyb == _Chem.rdchem.HybridizationType.SP2
            and nH == 0
        )

        # covalent_bonded_heavy_atoms using current topology positions
        cov = []
        for bond in rdatom.GetBonds():
            other = bond.GetOtherAtom(rdatom)
            if other.GetAtomicNum() <= 1:
                continue
            other_topo_idx = rdmol_to_topo.get(other.GetIdx())
            if other_topo_idx is None:
                continue
            cov.append(BondedHeavyAtom(
                name=other.GetSymbol() + str(other.GetIdx()),
                element=other.GetSymbol(),
                coord=pos_nm[other_topo_idx] * ANG,
            ))

        ri = rdatom.GetPDBResidueInfo()
        aname = ri.GetName().strip() if ri else f"{elem}{rdidx}"

        result[topo_atom.index] = (aname, elem, is_planar, cov)

    return result


def _sweep_ser_thr(
    topology: app.Topology,
    pos_nm: np.ndarray,                  # (N_atoms, 3) in nm
    rdmol: "Chem.Mol | None" = None,     # ligand mol for is_aromatic_planar lookup
    step_deg: float = 10.0,
    search_radius_nm: float = 0.45,      # 4.5 Å candidate search sphere
) -> np.ndarray:
    """
    For each SER/THR/TYR residue, sweep the hydroxyl dihedral and replace the
    current H position with the orientation that best H-bonds a ligand atom.

    Uses bunsalyze's is_valid_hbond with correct is_aromatic_planar and
    covalent_bonded_heavy_atoms for sp2 N acceptors.  Fluorine excluded.
    Returns a copy of pos_nm with updated H positions.
    """
    from .hbonds import is_valid_hbond, PolarAtom, DonorHydrogen

    ANG = 10.0   # nm → Å

    pos    = pos_nm.copy()
    angles = np.deg2rad(np.arange(0, 360, step_deg))

    atom_idx: dict[tuple, int] = {
        (a.residue.index, a.name): a.index for a in topology.atoms()
    }

    # Pre-build acceptor info (includes is_aromatic_planar and bonded heavy atoms)
    acc_info: dict[int, tuple] = {}
    if rdmol is not None:
        acc_info = _build_lig_acceptor_info(rdmol, topology, pos_nm)
    else:
        # Fallback: element only, no aromatic planar check
        for a in topology.atoms():
            if a.residue.name == "LIG" and a.element and a.element.symbol in _ACCEPTOR_ELEMENTS:
                acc_info[a.index] = (a.name, a.element.symbol, False, [])

    for residue in topology.residues():
        if residue.name not in _HYDROXYL:
            continue
        o_name, h_name, ax_name = _HYDROXYL[residue.name]

        o_idx  = atom_idx.get((residue.index, o_name))
        h_idx  = atom_idx.get((residue.index, h_name))
        ax_idx = atom_idx.get((residue.index, ax_name))
        if any(x is None for x in [o_idx, h_idx, ax_idx]):
            continue

        o_pos  = pos[o_idx]
        h_pos  = pos[h_idx]
        axis   = o_pos - pos[ax_idx]

        # Candidates within search radius
        cands = [
            (ai, info) for ai, info in acc_info.items()
            if np.linalg.norm(pos[ai] - o_pos) <= search_radius_nm
        ]
        if not cands:
            continue

        best_ha_dist = np.inf
        best_h_pos   = None

        for theta in angles:
            h_trial_nm  = _rotate(h_pos, o_pos, axis, theta)
            h_trial_ang = h_trial_nm * ANG

            donor_h  = DonorHydrogen(h_name, h_trial_ang)
            donor_pa = PolarAtom(
                name=o_name,
                coord=o_pos * ANG,
                donor_count=1,
                acceptor_count=1,
                parent_group_identifier=("A", residue.name, residue.index, ""),
                element="O",
                is_ligand_atom=False,
                donor_hydrogens=[donor_h],
                is_aromatic_planar=False,
                covalent_bonded_heavy_atoms=[],
                is_buried=True,
            )

            for acc_idx, (aname, acc_elem, is_planar, cov) in cands:
                # cov coords already reflect post-minimisation positions
                # (set by _build_lig_acceptor_info); only H atoms move in the sweep.
                acc_pa = PolarAtom(
                    name=aname,
                    coord=pos[acc_idx] * ANG,
                    donor_count=0,
                    acceptor_count=2,
                    parent_group_identifier=("B", "LIG", 1, ""),
                    element=acc_elem,
                    is_ligand_atom=True,
                    donor_hydrogens=[],
                    is_aromatic_planar=is_planar,
                    covalent_bonded_heavy_atoms=cov,
                    is_buried=True,
                )

                if is_valid_hbond(donor_pa, donor_h, acc_pa, clash_check=False):
                    ha_dist = np.linalg.norm(h_trial_ang - pos[acc_idx] * ANG)
                    if ha_dist < best_ha_dist:
                        best_ha_dist = ha_dist
                        best_h_pos   = h_trial_nm
                    break

        if best_h_pos is not None:
            pos[h_idx] = best_h_pos

    return pos

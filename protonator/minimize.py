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
    # 2. Load protein (no H) + ligand (with H) into OpenMM
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        prot_path = Path(td) / "protein.pdb"
        lig_path  = Path(td) / "ligand.pdb"
        prot_path.write_text(protein_text)
        lig_path.write_text(_add_conect_records(ligand_text, ligand_params.mol))
        prot_pdb = app.PDBFile(str(prot_path))
        lig_pdb  = app.PDBFile(str(lig_path))

    # ------------------------------------------------------------------
    # 3. Build force field (needed before addHydrogens so it can evaluate
    #    H-bond geometry for His tautomer selection)
    # ------------------------------------------------------------------
    ff = app.ForceField("amber/ff14SB.xml", "implicit/gbn2.xml")
    ff.registerTemplateGenerator(
        make_gaff2_generator(ligand_params.mol, ligand_params.charges)
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
    platform = openmm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(
        tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=max_iterations,
    )

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    state = sim.context.getState(getPositions=True)
    with open(output_path, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh)


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

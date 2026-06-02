from __future__ import annotations
from protonator.initialize import _init_worker
_init_worker(1)  # Set thread-count env vars for the main process before any library is imported

from pathlib import Path
from typing import Optional
from multiprocessing import Pool

import typer
from rich.progress import track
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from protonator.minimize import minimize_complex, minimize_apo, _init_worker_ff

app = typer.Typer(add_completion=False)


def minimize_apo_pool(inputs_dict: dict) -> None:
    """Wrapper for minimize_apo to be used in multiprocessing Pool."""
    minimize_apo(**inputs_dict)


def minimize_complex_pool(inputs_dict: dict) -> None:
    """Wrapper for minimize_complex to be used in multiprocessing Pool."""
    minimize_complex(**inputs_dict)


@app.command()
def protonate_batch(
    input_pdbs_txt_file: Path = typer.Argument(..., help="Text file with one PDB path per line (chain A = protein, chain B = ligand with H)"),
    smiles: str = typer.Argument(..., help="One ligand SMILES string representing all structures in the batch. See protonator/scripts/run_batch.py for a more flexible alternative that allows different ligands per structure."),
    output_dir: Path = typer.Option(Path("batch_emin_output"), "--output-dir", "-o", help="Output directory"),
    apo: bool = typer.Option(False, "--apo", help="Minimise protein only — strip ligand and ignore --smiles"),
    ph: float = typer.Option(7.4, "--ph", help="pH for protonation state assignment"),
    restraint_k: float = typer.Option(50.0, "--restraint-k", help="Backbone/ligand restraint (kcal/mol/Å²)"),
    tolerance: float = typer.Option(30.0, "--tolerance", help="Minimisation convergence (kJ/mol/nm)"),
    freeze_ligand: bool = typer.Option(True, "--freeze-ligand/--no-freeze-ligand", help="Restrain ligand heavy atoms during minimisation"),
    sweep_hbonds: bool = typer.Option(True, "--sweep-hbonds/--no-sweep-hbonds", help="Post-minimisation sweep of SER/THR/TYR hydroxyl orientations to prefer ligand H-bonds (default: on)"),
    max_iterations: int = typer.Option(0, "--max-iterations", help="Max minimisation steps; 0 = run until convergence"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Custom suffix for output files (default: _emin.pdb or _emin_apo.pdb if --apo)"),
    n_workers: int = typer.Option(1, "--n-workers", help="Number of parallel workers to use (default: 1, i.e. no parallelism)"),
):
    """
    Protonate and energy-minimise a batch of protein-ligand complexes from a text file containing paths to PDBs with the same ligand.
    """
    # Prep output directory and read input paths
    all_targets = [Path(line.strip()) for line in input_pdbs_txt_file.read_text().splitlines() if line.strip() and Path(line.strip()).is_file()]
    if not all_targets:
        typer.echo(f"No valid PDB paths found in {input_pdbs_txt_file}.", err=True)
        raise typer.Exit(1)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    print(f"Found {len(all_targets)} valid PDB paths in {input_pdbs_txt_file}.")

    # Sanity check that all file stems are unique since they are used for output naming:
    stems = [p.stem for p in all_targets]
    if len(set(stems)) != len(stems):
        duplicates = set(s for s in stems if stems.count(s) > 1)
        typer.echo(f"Error: non-unique file stems found in input PDB paths: {duplicates}", err=True)
        raise typer.Exit(1)

    # Sanity check that ligand can be loaded
    ligand_mol = Chem.MolFromSmiles(smiles)
    if ligand_mol is None or ligand_mol.GetNumHeavyAtoms() == 0:
        typer.echo(f"Error: invalid SMILES string: {smiles}", err=True)
        raise typer.Exit(1)

    inputs = []
    if apo:
        for pdb_path in all_targets:
            name = pdb_path.stem
            out_path = output_dir / (name + (suffix if suffix else "_emin_apo.pdb"))
            inputs.append({
                "pdb_path": str(pdb_path),
                "output_path": str(out_path),
                "ph": ph,
                "restraint_k": restraint_k,
                "tolerance": tolerance,
                "max_iterations": max_iterations
            })
        with Pool(n_workers, initializer=_init_worker_ff, initargs=(1,)) as pool:
            for _ in track(pool.imap_unordered(minimize_apo_pool, inputs), total=len(inputs), description=f"Running apo energy minimization…"):
                pass
    else:
        for pdb_path in all_targets:
            name = pdb_path.stem
            out_path = output_dir / (name + (suffix if suffix else "_emin.pdb"))
            inputs.append({
                "pdb_path": str(pdb_path),
                "smiles": smiles,
                "output_path": str(out_path),
                "ph": ph,
                "restraint_k": restraint_k,
                "tolerance": tolerance,
                "freeze_ligand": freeze_ligand,
                "sweep_hbonds": sweep_hbonds,
                "max_iterations": max_iterations,
            })

        with Pool(n_workers, initializer=_init_worker_ff, initargs=(1,)) as pool:
            for _ in track(pool.imap_unordered(minimize_complex_pool, inputs), total=len(inputs), description=f"Running ligand-bound energy minimization…"):
                pass

from __future__ import annotations
from protonator.initialize import _init_worker
_init_worker(1)  # Set thread-count env vars for the main process before any library is imported

import shutil
from pathlib import Path
from typing import Optional
from multiprocessing import Pool

import prody as pr
import typer
from rich.progress import track
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from protonator.minimize import minimize_complex, minimize_apo, _init_worker_ff

app = typer.Typer(add_completion=False)


def _write_apo_fallback(pdb_path: str, output_path: Path) -> None:
    ag = pr.parsePDB(pdb_path)
    pr.writePDB(str(output_path), ag.select('protein'))


def minimize_apo_pool(inputs_dict: dict) -> None:
    """Wrapper for minimize_apo to be used in multiprocessing Pool."""
    try:
        minimize_apo(**inputs_dict)
    except Exception as e:
        print(f"Warning: apo minimization failed for {inputs_dict['pdb_path']} ({e}); writing input structure as fallback.", flush=True)
        _write_apo_fallback(inputs_dict['pdb_path'], Path(inputs_dict['output_path']))


def minimize_complex_pool(inputs_dict: dict) -> None:
    """Wrapper for minimize_complex to be used in multiprocessing Pool."""
    try:
        minimize_complex(**inputs_dict)
    except Exception as e:
        print(f"Warning: holo minimization failed for {inputs_dict['pdb_path']} ({e}); writing input structure as fallback.", flush=True)
        shutil.copy2(inputs_dict['pdb_path'], inputs_dict['output_path'])


def minimize_two_state(inputs_dict: dict) -> None:
    """Minimize both holo and apo states for one structure in a single pool worker.

    Expects inputs_dict to contain all minimize_complex keys plus 'apo_output_path'
    for the apo output file. Each state is skipped individually if its output already
    exists, so partial runs (holo done, apo not) resume correctly with -r.
    """
    apo_output_path = Path(inputs_dict['apo_output_path'])
    holo_inputs = {k: v for k, v in inputs_dict.items() if k != 'apo_output_path'}

    if not Path(holo_inputs['output_path']).exists():
        try:
            minimize_complex(**holo_inputs)
        except Exception as e:
            print(f"Warning: holo minimization failed for {inputs_dict['pdb_path']} ({e}); writing input structure as fallback.", flush=True)
            shutil.copy2(inputs_dict['pdb_path'], holo_inputs['output_path'])

    if not apo_output_path.exists():
        try:
            minimize_apo(
                pdb_path=inputs_dict['pdb_path'],
                output_path=apo_output_path,
                ph=inputs_dict['ph'],
                restraint_k=inputs_dict['restraint_k'],
                tolerance=inputs_dict['tolerance'],
                max_iterations=inputs_dict['max_iterations'],
            )
        except Exception as e:
            print(f"Warning: apo minimization failed for {inputs_dict['pdb_path']} ({e}); writing input structure as fallback.", flush=True)
            _write_apo_fallback(inputs_dict['pdb_path'], apo_output_path)


@app.command()
def protonate_batch(
    input_pdbs_txt_file: Path = typer.Argument(..., help="Text file with one PDB path per line (chain A = protein, chain B = ligand with H)"),
    smiles: str = typer.Argument(..., help="One ligand SMILES string representing all structures in the batch. See protonator/scripts/run_batch.py for a more flexible alternative that allows different ligands per structure."),
    output_dir: Path = typer.Option(Path("batch_emin_output"), "--output-dir", "-o", help="Output directory"),
    apo: bool = typer.Option(False, "--apo", help="Minimise protein only — strip ligand and ignore --smiles"),
    two_state: bool = typer.Option(False, "--two-state", help="Minimise both apo and holo states (default: off), Overrides --apo if both are set."),
    ph: float = typer.Option(7.4, "--ph", help="pH for protonation state assignment"),
    restraint_k: float = typer.Option(50.0, "--restraint-k", help="Backbone/ligand restraint (kcal/mol/Å²)"),
    tolerance: float = typer.Option(30.0, "--tolerance", help="Minimisation convergence (kJ/mol/nm)"),
    freeze_ligand: bool = typer.Option(True, "--freeze-ligand/--no-freeze-ligand", help="Restrain ligand heavy atoms during minimisation"),
    sweep_hbonds: bool = typer.Option(True, "--sweep-hbonds/--no-sweep-hbonds", help="Post-minimisation sweep of SER/THR/TYR hydroxyl orientations to prefer ligand H-bonds (default: on)"),
    max_iterations: int = typer.Option(0, "--max-iterations", help="Max minimisation steps; 0 = run until convergence"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Custom suffix for output files (default: _emin.pdb or _emin_apo.pdb if --apo)"),
    n_workers: int = typer.Option(1, "--n-workers", help="Number of parallel workers to use (default: 1, i.e. no parallelism)"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Skip structures whose output file(s) already exist"),
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

    complex_template = {
        "smiles": smiles, "ph": ph, "restraint_k": restraint_k,
        "tolerance": tolerance, "freeze_ligand": freeze_ligand, "sweep_hbonds": sweep_hbonds, "max_iterations": max_iterations,
    }

    apo_template = {
        "ph": ph, "restraint_k": restraint_k, "tolerance": tolerance, "max_iterations": max_iterations,
    }

    already_complete_count = 0
    if two_state:
        inputs_two_state = []
        for pdb_path in all_targets:
            name = pdb_path.stem
            out_path_holo = output_dir / (name + (suffix if suffix else "_emin.pdb"))
            apo_suffix = Path(suffix).stem + "_apo" + Path(suffix).suffix if suffix else "_emin_apo.pdb"
            out_path_apo = output_dir / (name + apo_suffix)
            if resume and out_path_holo.exists() and out_path_apo.exists():
                already_complete_count += 1
                continue
            entry = {**complex_template, "pdb_path": str(pdb_path), "output_path": str(out_path_holo), "apo_output_path": str(out_path_apo)}
            inputs_two_state.append(entry)
        print(f"Prepared {len(inputs_two_state)} structures for two-state minimization." + (f" Skipping {already_complete_count} already complete structures due to --resume." if already_complete_count else ""))
        with Pool(n_workers, initializer=_init_worker_ff, initargs=(1,)) as pool:
            for _ in track(pool.imap_unordered(minimize_two_state, inputs_two_state), total=len(inputs_two_state), description="Running two-state energy minimization…"):
                pass
    else:
        inputs = []
        if apo:
            for pdb_path in all_targets:
                name = pdb_path.stem
                out_path = output_dir / (name + (suffix if suffix else "_emin_apo.pdb"))
                if resume and out_path.exists():
                    already_complete_count += 1
                    continue
                apo_template.update({
                    "pdb_path": str(pdb_path),
                    "output_path": str(out_path),
                })
                inputs.append(apo_template.copy())
            print(f"Prepared {len(inputs)} structures for apo minimization." + (f" Skipping {already_complete_count} already complete structures due to --resume." if already_complete_count else ""))
            with Pool(n_workers, initializer=_init_worker_ff, initargs=(1,)) as pool:
                for _ in track(pool.imap_unordered(minimize_apo_pool, inputs), total=len(inputs), description=f"Running apo energy minimization…"):
                    pass
        else:
            for pdb_path in all_targets:
                name = pdb_path.stem
                out_path = output_dir / (name + (suffix if suffix else "_emin.pdb"))
                if resume and out_path.exists():
                    already_complete_count += 1
                    continue
                complex_template.update({
                    "pdb_path": str(pdb_path),
                    "output_path": str(out_path),
                })
                inputs.append(complex_template.copy())
            print(f"Prepared {len(inputs)} structures for holo minimization." + (f" Skipping {already_complete_count} already complete structures due to --resume." if already_complete_count else ""))
            with Pool(n_workers, initializer=_init_worker_ff, initargs=(1,)) as pool:
                for _ in track(pool.imap_unordered(minimize_complex_pool, inputs), total=len(inputs), description=f"Running ligand-bound energy minimization…"):
                    pass

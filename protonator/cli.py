"""Command-line interface for the protonator pipeline."""
from __future__ import annotations
from protonator.initialize import _init_worker
_init_worker(1)  # Set thread-count env vars for the main process before any library is imported

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pdb: Path = typer.Argument(..., help="Input PDB (chain A = protein, chain B = ligand with H)"),
    smiles: Optional[str] = typer.Option(None, "--smiles", "-s", help="Ligand SMILES string"),
    ligand_file: Optional[Path] = typer.Option(None, "--ligand-file", "-l", help="Ligand .sdf / .mol file"),
    apo: bool = typer.Option(False, "--apo", help="Minimise protein only — strip ligand and ignore --smiles/--ligand-file"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Output directory"),
    ph: float = typer.Option(7.4, "--ph", help="pH for protonation state assignment"),
    restraint_k: float = typer.Option(50.0, "--restraint-k", help="Backbone/ligand restraint (kcal/mol/Å²)"),
    tolerance: float = typer.Option(30.0, "--tolerance", help="Minimisation convergence (kJ/mol/nm)"),
    freeze_ligand: bool = typer.Option(True, "--freeze-ligand/--no-freeze-ligand", help="Restrain ligand heavy atoms during minimisation"),
    sweep_hbonds: bool = typer.Option(True, "--sweep-hbonds/--no-sweep-hbonds", help="Post-minimisation sweep of SER/THR/TYR hydroxyl orientations to prefer ligand H-bonds (default: on)"),
    max_iterations: int = typer.Option(0, "--max-iterations", help="Max minimisation steps; 0 = run until convergence"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Custom suffix for output file (default: _emin.pdb or _emin_apo.pdb if --apo)"),
) -> None:
    """
    Protonate and energy-minimise a protein-ligand complex from an AF3-style PDB.

    Exactly one of --smiles or --ligand-file must be supplied, unless --apo is
    used in which case the ligand is ignored and only chain A is processed.
    """
    from .minimize import minimize_complex, minimize_apo

    if not suffix:
        output_path = output_dir / (pdb.stem + ("_emin.pdb" if not apo else "_emin_apo.pdb"))
    else:
        output_path = output_dir / (pdb.stem + suffix)

    if apo:
        typer.echo(f"Protonating and minimising {pdb.name} (apo) …")
        minimize_apo(
            pdb,
            output_path,
            ph=ph,
            restraint_k=restraint_k,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        typer.echo(f"Written → {output_path}")
        return

    # Holo path — ligand required
    if smiles is None and ligand_file is None:
        typer.echo("Error: supply --smiles or --ligand-file (or use --apo).", err=True)
        raise typer.Exit(1)
    if smiles is not None and ligand_file is not None:
        typer.echo("Error: supply only one of --smiles or --ligand-file.", err=True)
        raise typer.Exit(1)

    if ligand_file is not None:
        from rdkit import Chem as _Chem
        _mol = _Chem.MolFromMolFile(str(ligand_file), removeHs=True, sanitize=True)
        if _mol is None:
            typer.echo(f"Error: could not parse {ligand_file}", err=True)
            raise typer.Exit(1)
        smiles = _Chem.MolToSmiles(_mol)

    typer.echo(f"Protonating and minimising {pdb.name} …")
    minimize_complex(
        pdb,
        smiles,
        output_path,
        ph=ph,
        restraint_k=restraint_k,
        tolerance=tolerance,
        freeze_ligand=freeze_ligand,
        sweep_hbonds=sweep_hbonds,
        max_iterations=max_iterations,
    )
    typer.echo(f"Written → {output_path}")


def _extract_ligand_pdb_block(pdb_path: Path) -> str:
    """Return a PDB block string for chain B (ligand) from the input PDB."""
    lines = [
        line for line in pdb_path.read_text().splitlines()
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) > 21 and line[21] == "B"
    ]
    return "\n".join(lines) + "\nEND\n"

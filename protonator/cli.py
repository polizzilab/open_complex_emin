"""Command-line interface for the protonator pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pdb: Path = typer.Argument(..., help="Input PDB (chain A = protein, chain B = ligand with H)"),
    smiles: Optional[str] = typer.Option(None, "--smiles", "-s", help="Ligand SMILES string"),
    ligand_file: Optional[Path] = typer.Option(None, "--ligand-file", "-l", help="Ligand .sdf / .mol file"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Output directory"),
    ph: float = typer.Option(7.4, "--ph", help="pH for pdb2pqr fallback protonation"),
    restraint_k: float = typer.Option(50.0, "--restraint-k", help="Backbone/ligand restraint (kcal/mol/Å²)"),
    tolerance: float = typer.Option(10.0, "--tolerance", help="Minimisation convergence (kJ/mol/nm)"),
    recompute_ligand: bool = typer.Option(False, "--recompute-ligand", help="Recompute GAFF2/xTB per structure"),
    freeze_ligand: bool = typer.Option(True, "--freeze-ligand/--no-freeze-ligand", help="Restrain ligand heavy atoms during minimisation"),
    sweep_hbonds: bool = typer.Option(True, "--sweep-hbonds/--no-sweep-hbonds", help="Post-minimisation sweep of SER/THR/TYR hydroxyl orientations to prefer ligand H-bonds (default: on)"),
    max_iterations: int = typer.Option(0, "--max-iterations", help="Max minimisation steps; 0 = run until convergence"),
) -> None:
    """
    Protonate and energy-minimise a protein–ligand complex from an AF3-style PDB.

    Exactly one of --smiles or --ligand-file must be supplied.
    Protein protonation uses REDUCE (with ligand-aware His flip optimisation)
    if available, otherwise falls back to pdb2pqr.
    """
    from .ligand import prepare_ligand
    from .minimize import minimize_complex, _extract_chain

    if smiles is None and ligand_file is None:
        typer.echo("Error: supply either --smiles or --ligand-file.", err=True)
        raise typer.Exit(1)
    if smiles is not None and ligand_file is not None:
        typer.echo("Error: supply only one of --smiles or --ligand-file.", err=True)
        raise typer.Exit(1)

    typer.echo("Computing GFN2-xTB ligand charges …")
    if ligand_file is not None:
        ligand_params = prepare_ligand(str(ligand_file), is_file=True)
    else:
        # Extract chain B from the input PDB and assign bond orders from SMILES,
        # preserving the AF3-predicted 3-D pose rather than generating a new conformer.
        pdb_ligand_block = _extract_ligand_pdb_block(pdb)
        ligand_params = prepare_ligand(smiles, pdb_ligand_block=pdb_ligand_block)

    output_path = output_dir / (pdb.stem + "_relaxed.pdb")
    typer.echo(f"Protonating and minimising {pdb.name} …")
    minimize_complex(
        pdb,
        ligand_params,
        output_path,
        ph=ph,
        restraint_k=restraint_k,
        tolerance=tolerance,
        recompute_ligand=recompute_ligand,
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

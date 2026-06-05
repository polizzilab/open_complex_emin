"""Command-line interface for the fast (force-field-free) protonation track."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pdb: Path = typer.Argument(..., help="Input PDB (chain A = protein, chain B = ligand)"),
    smiles: Optional[str] = typer.Option(None, "--smiles", "-s", help="Ligand SMILES string"),
    ligand_file: Optional[Path] = typer.Option(None, "--ligand-file", "-l", help="Ligand .sdf / .mol file"),
    apo: bool = typer.Option(False, "--apo", help="Protein only — place polar H, no ligand, no sweep"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir", "-o", help="Output directory"),
    step_deg: float = typer.Option(10.0, "--sweep-step-deg", help="Rotatable-H dihedral sweep increment (deg)"),
    sweep_protein: bool = typer.Option(True, "--sweep-protein/--no-sweep-protein", help="Sweep protein Ser/Thr/Tyr/Cys/Lys polar H"),
    sweep_ligand: bool = typer.Option(True, "--sweep-ligand/--no-sweep-ligand", help="Sweep ligand terminal rotatable polar H"),
    flips: bool = typer.Option(True, "--flips/--no-flips", help="Evaluate Asn/Gln amide flips"),
    his_tautomers: bool = typer.Option(True, "--his-tautomers/--no-his-tautomers", help="Select His tautomer (HID/HIE/HIP) to maximize ligand H-bonds"),
) -> None:
    """
    Fast protonation: place polar protein hydrogens geometrically (no OpenMM,
    no xTB, no GAFF2) and reorient rotatable polar H / Asn-Gln flips / His
    tautomers to maximize the number of protein-ligand interface hydrogen bonds.
    Heavy atoms and non-rotatable groups never move.
    """
    from .fast import fast_protonate

    output_path = output_dir / (pdb.stem + "_relaxed.pdb")

    if not apo and smiles is None and ligand_file is None:
        typer.echo("Error: supply --smiles or --ligand-file (or use --apo).", err=True)
        raise typer.Exit(1)
    if smiles is not None and ligand_file is not None:
        typer.echo("Error: supply only one of --smiles or --ligand-file.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Fast-protonating {pdb.name} …")
    report = fast_protonate(
        pdb, output_path,
        smiles=smiles, ligand_file=ligand_file, apo=apo,
        step_deg=step_deg, sweep_protein=sweep_protein, sweep_ligand=sweep_ligand,
        do_flips=flips, his_tautomers=his_tautomers,
    )
    if report.get("mode") == "holo":
        typer.echo(f"Interface H-bonds: {report['start']} → {report['final']}")
    typer.echo(f"Written → {output_path}")

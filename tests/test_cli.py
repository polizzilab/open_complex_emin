"""
Integration tests for the protonator CLI.

Tests use debug/ data with --max-iterations 10 to run a short minimisation
(exercises every code path without waiting for full convergence).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from protonator.cli import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEBUG = Path(__file__).parent.parent / "debug"

INPUT_PDB = DEBUG / "nise_iter_012-197_sample_0.pdb"
LIGAND_SDF = DEBUG / "temp_lig.sdf"
LIGAND_SMILES = "CC1=NC(=Cc2cc(F)c([O-])c(F)c2)C(=O)N1C"

runner = CliRunner()


def _count_h(pdb_path: Path) -> int:
    """Count hydrogen ATOM/HETATM records in a PDB file."""
    return sum(
        1 for l in pdb_path.read_text().splitlines()
        if l[:6].strip() in ("ATOM", "HETATM") and l[76:78].strip() == "H"
    )


def _count_h_input() -> int:
    return _count_h(INPUT_PDB)


def _ligand_heavy_coords(pdb_path: Path) -> np.ndarray:
    """Return (N, 3) array of ligand heavy atom coordinates."""
    coords = []
    for l in pdb_path.read_text().splitlines():
        if l[:6].strip() == "HETATM" and l[76:78].strip() != "H":
            coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return np.array(coords)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSdfInput:
    """CLI with --ligand-file (SDF path)."""

    def test_runs_and_writes_output(self, tmp_path):
        result = runner.invoke(app, [
            str(INPUT_PDB),
            "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        assert result.exit_code == 0, result.output
        out = tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb"
        assert out.exists()

    def test_hydrogens_added_to_protein(self, tmp_path):
        runner.invoke(app, [
            str(INPUT_PDB),
            "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        out = tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb"
        assert _count_h(out) > _count_h_input(), \
            "Output should have more H atoms than the unprotonated input"

    def test_ligand_heavy_atoms_restrained(self, tmp_path):
        runner.invoke(app, [
            str(INPUT_PDB),
            "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        out = tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb"
        inp_lig = _ligand_heavy_coords(INPUT_PDB)
        out_lig = _ligand_heavy_coords(out)
        assert inp_lig.shape == out_lig.shape
        rmsd = np.sqrt(((inp_lig - out_lig) ** 2).sum(1).mean())
        assert rmsd < 0.5, f"Ligand heavy RMSD too large: {rmsd:.3f} Å"


class TestSmilesInput:
    """CLI with --smiles (uses PDB conformer coordinates)."""

    def test_runs_and_writes_output(self, tmp_path):
        result = runner.invoke(app, [
            str(INPUT_PDB),
            "--smiles", LIGAND_SMILES,
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        assert result.exit_code == 0, result.output
        out = tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb"
        assert out.exists()

    def test_uses_pdb_conformer_not_fresh_embed(self, tmp_path):
        """--smiles output should have the same ligand pose as --ligand-file."""
        sdf_dir = tmp_path / "sdf"
        smi_dir = tmp_path / "smi"
        sdf_dir.mkdir(); smi_dir.mkdir()

        runner.invoke(app, [
            str(INPUT_PDB), "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(sdf_dir), "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        runner.invoke(app, [
            str(INPUT_PDB), "--smiles", LIGAND_SMILES,
            "--output-dir", str(smi_dir), "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])

        sdf_lig = _ligand_heavy_coords(sdf_dir / "nise_iter_012-197_sample_0_relaxed.pdb")
        smi_lig = _ligand_heavy_coords(smi_dir / "nise_iter_012-197_sample_0_relaxed.pdb")
        rmsd = np.sqrt(((sdf_lig - smi_lig) ** 2).sum(1).mean())
        assert rmsd < 0.1, (
            f"--smiles and --ligand-file should produce the same ligand pose "
            f"(RMSD {rmsd:.3f} Å suggests a fresh conformer was generated)"
        )

    def test_hydrogens_added_to_protein(self, tmp_path):
        runner.invoke(app, [
            str(INPUT_PDB),
            "--smiles", LIGAND_SMILES,
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        out = tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb"
        assert _count_h(out) > _count_h_input()


class TestUnprotonatedLigand:
    """Pipeline should work even when chain B has no hydrogen atoms."""

    @pytest.fixture()
    def stripped_pdb(self, tmp_path):
        """Input PDB with H atoms removed from chain B."""
        stripped = tmp_path / "no_lig_h.pdb"
        lines = INPUT_PDB.read_text().splitlines()
        stripped.write_text("\n".join(
            l for l in lines
            if not (l[:6].strip() in ("ATOM", "HETATM")
                    and len(l) > 78 and l[21] == "B" and l[76:78].strip() == "H")
        ) + "\n")
        return stripped

    def _count_lig_h(self, pdb_path: Path) -> int:
        return sum(
            1 for l in pdb_path.read_text().splitlines()
            if l[:6].strip() == "HETATM" and l[76:78].strip() == "H"
        )

    def test_ligand_file_runs_when_ligand_h_absent(self, tmp_path, stripped_pdb):
        result = runner.invoke(app, [
            str(stripped_pdb),
            "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        assert result.exit_code == 0, result.output
        out = tmp_path / "no_lig_h_relaxed.pdb"
        assert out.exists()
        assert self._count_lig_h(out) > 0, "Output should contain ligand H atoms"

    def test_smiles_runs_when_ligand_h_absent(self, tmp_path, stripped_pdb):
        result = runner.invoke(app, [
            str(stripped_pdb),
            "--smiles", LIGAND_SMILES,
            "--output-dir", str(tmp_path),
            "--max-iterations", "10",
            "--no-sweep-hbonds",
        ])
        assert result.exit_code == 0, result.output
        out = tmp_path / "no_lig_h_relaxed.pdb"
        assert out.exists()
        assert self._count_lig_h(out) > 0, "Output should contain ligand H atoms"


class TestErrorHandling:
    """Bad inputs should fail with helpful messages."""

    def test_neither_smiles_nor_file_raises(self, tmp_path):
        result = runner.invoke(app, [str(INPUT_PDB), "--output-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_both_smiles_and_file_raises(self, tmp_path):
        result = runner.invoke(app, [
            str(INPUT_PDB),
            "--smiles", LIGAND_SMILES,
            "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code != 0

    def test_bad_smiles_raises(self, tmp_path):
        result = runner.invoke(app, [
            str(INPUT_PDB),
            "--smiles", "not_a_valid_smiles!!!",
            "--output-dir", str(tmp_path),
            "--max-iterations", "1",
        ])
        assert result.exit_code != 0

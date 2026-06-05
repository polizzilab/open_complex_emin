"""
Integration tests for the fast (force-field-free) protonation track.

Fast mode does no minimization, so these run in well under a second each.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import prody as pr
import pytest
from typer.testing import CliRunner

from protonator.cli_fast import app
from protonator.fast import fast_protonate

DEBUG = Path(__file__).parent.parent / "examples" / "debug"
INPUT_PDB = DEBUG / "nise_iter_012-197_sample_0.pdb"
LIGAND_SDF = DEBUG / "temp_lig.sdf"
LIGAND_SMILES = "CC1=NC(=Cc2cc(F)c([O-])c(F)c2)C(=O)N1C"

runner = CliRunner()


def _heavy_by_name(ag, sel):
    s = ag.select(sel)
    return {n: c for n, c in zip(s.getNames(), s.getCoords())} if s is not None else {}


def _count_h(ag, sel):
    s = ag.select(sel)
    return 0 if s is None else int((s.getElements() == "H").sum())


class TestFastCLI:
    def test_runs_and_writes(self, tmp_path):
        result = runner.invoke(app, [
            str(INPUT_PDB), "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb").exists()

    def test_protein_gets_polar_h(self, tmp_path):
        runner.invoke(app, [str(INPUT_PDB), "--ligand-file", str(LIGAND_SDF),
                            "--output-dir", str(tmp_path)])
        out = pr.parsePDB(str(tmp_path / "nise_iter_012-197_sample_0_relaxed.pdb"))
        assert _count_h(out, "chain A") > 0
        # backbone amide H present
        assert out.select("chain A and name H") is not None


class TestFrozenScaffold:
    """Heavy atoms and the backbone must never move in fast mode."""

    def test_backbone_and_ligand_heavy_frozen(self, tmp_path):
        out_path = tmp_path / "o.pdb"
        fast_protonate(INPUT_PDB, out_path, ligand_file=LIGAND_SDF)
        inp = pr.parsePDB(str(INPUT_PDB))
        out = pr.parsePDB(str(out_path))
        # CA atoms unmoved
        ca_i = inp.select("chain A and name CA").getCoords()
        ca_o = out.select("chain A and name CA").getCoords()
        assert np.abs(ca_i - ca_o).max() < 1e-3
        # ligand heavy atoms unmoved (compare sorted coords; names may differ)
        li = np.sort(inp.select("chain B and not hydrogen").getCoords(), axis=0)
        lo = np.sort(out.select("chain B and not hydrogen").getCoords(), axis=0)
        assert li.shape == lo.shape
        assert np.abs(li - lo).max() < 1e-2


class TestHBondMaximization:
    def test_hbonds_non_decreasing(self, tmp_path):
        rep = fast_protonate(INPUT_PDB, tmp_path / "o.pdb", ligand_file=LIGAND_SDF)
        assert rep["final"] >= rep["start"]

    def test_debug_case_improves(self, tmp_path):
        """On the bundled debug complex the sweep finds extra interface H-bonds."""
        rep = fast_protonate(INPUT_PDB, tmp_path / "o.pdb", ligand_file=LIGAND_SDF)
        assert rep["final"] > rep["start"]

    def test_no_sweep_flags_disable_optimization(self, tmp_path):
        rep = fast_protonate(
            INPUT_PDB, tmp_path / "o.pdb", ligand_file=LIGAND_SDF,
            sweep_protein=False, sweep_ligand=False, do_flips=False, his_tautomers=False,
        )
        assert rep["final"] == rep["start"]


class TestSmilesMatchesSdf:
    def test_same_ligand_pose(self, tmp_path):
        fast_protonate(INPUT_PDB, tmp_path / "sdf.pdb", ligand_file=LIGAND_SDF)
        fast_protonate(INPUT_PDB, tmp_path / "smi.pdb", smiles=LIGAND_SMILES)
        sdf = np.sort(pr.parsePDB(str(tmp_path / "sdf.pdb")).select("chain B and not hydrogen").getCoords(), axis=0)
        smi = np.sort(pr.parsePDB(str(tmp_path / "smi.pdb")).select("chain B and not hydrogen").getCoords(), axis=0)
        assert np.abs(sdf - smi).max() < 0.1


class TestHisTautomer:
    def test_his_has_ring_h(self, tmp_path):
        out_path = tmp_path / "o.pdb"
        fast_protonate(INPUT_PDB, out_path, ligand_file=LIGAND_SDF)
        out = pr.parsePDB(str(out_path))
        his = out.select("resname HIS")
        if his is None:
            pytest.skip("no His in debug structure")
        for resnum in set(his.getResnums().tolist()):
            ring = out.select(f"resname HIS and resnum {resnum} and name HD1 HE2")
            assert ring is not None and ring.numAtoms() >= 1


class TestApo:
    def test_apo_no_ligand(self, tmp_path):
        out_path = tmp_path / "apo.pdb"
        rep = fast_protonate(INPUT_PDB, out_path, apo=True)
        assert rep["mode"] == "apo"
        out = pr.parsePDB(str(out_path))
        assert out.select("hetatm") is None or out.select("chain B") is None
        assert _count_h(out, "chain A") > 0


class TestErrors:
    def test_holo_requires_ligand(self, tmp_path):
        result = runner.invoke(app, [str(INPUT_PDB), "--output-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_both_ligand_sources_rejected(self, tmp_path):
        result = runner.invoke(app, [
            str(INPUT_PDB), "--smiles", LIGAND_SMILES, "--ligand-file", str(LIGAND_SDF),
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code != 0

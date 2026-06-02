"""
GAFF2 residue template generator for OpenMM ForceField.

build_gaff2_ffxml() runs antechamber once and returns the FFXML string.
make_gaff2_generator() takes that pre-built string and returns a fast closure
that simply loads it into whatever ForceField it encounters — no subprocess
calls at minimisation time.
"""
from __future__ import annotations
from protonator.initialize import _init_worker
_init_worker(1)  # Set thread-count env vars for the main process before any library is imported

import io

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def build_gaff2_ffxml(rdmol: Chem.Mol, charges: list[float]) -> str:
    """
    Run antechamber + parmchk2 via GAFFTemplateGenerator and return the
    GAFF2 FFXML string for the ligand.

    This is the expensive step (antechamber subprocess).  Call it once per
    ligand identity, store the result in LigandParams.gaff_xml, and reuse
    across all binder structures.
    """
    from openff.toolkit import Molecule as OFFMol
    from openff.units import unit as off_unit
    from openmmforcefields.generators import GAFFTemplateGenerator

    off_mol = OFFMol.from_rdkit(rdmol, allow_undefined_stereo=True)
    off_mol.partial_charges = np.array(charges) * off_unit.elementary_charge

    gaff = GAFFTemplateGenerator(molecules=[off_mol], forcefield="gaff-2.11")
    return gaff.generate_residue_template(off_mol)


def make_gaff2_generator(gaff_xml: str, lig_resname: str = "LIG"):
    """
    Return a callable for ForceField.registerTemplateGenerator() that loads
    a pre-computed GAFF2 FFXML string.  No subprocess calls at runtime.
    """
    def generator(forcefield, residue):
        if residue.name != lig_resname:
            return False
        forcefield.loadFile(io.StringIO(gaff_xml))
        return True

    return generator

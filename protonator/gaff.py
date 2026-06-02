"""
GAFF2 residue template generator for OpenMM ForceField.

Wraps GAFFTemplateGenerator (openmmforcefields) but injects caller-supplied
partial charges so AM1-BCC is never called.  The filter on lig_resname avoids
spurious graph-isomorphism checks against every protein residue.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem


def make_gaff2_generator(rdmol: Chem.Mol, charges: list[float], lig_resname: str = "LIG"):
    """
    Return a callable for ForceField.registerTemplateGenerator().

    Parameters
    ----------
    rdmol:
        RDKit molecule with explicit H and a 3-D conformer matching PDB coords.
    charges:
        Partial charges (elementary charge) in rdmol atom order.
    lig_resname:
        Residue name used in the PDB / OpenMM topology (default "LIG").
    """
    from openff.toolkit import Molecule as OFFMol
    from openff.units import unit as off_unit
    from openmmforcefields.generators import GAFFTemplateGenerator

    off_mol = OFFMol.from_rdkit(rdmol, allow_undefined_stereo=True)
    # Pre-set charges → GAFFTemplateGenerator skips AM1-BCC and uses these
    off_mol.partial_charges = np.array(charges) * off_unit.elementary_charge

    gaff = GAFFTemplateGenerator(molecules=[off_mol], forcefield="gaff-2.11")

    def generator(forcefield, residue):
        if residue.name != lig_resname:
            return False
        return gaff.generator(forcefield, residue)

    return generator

# protonator

Protonates and energy-minimises AF3-style protein–ligand complexes as a drop-in replacement for Rosetta's hydrogen placement + sidechain relaxation step.

**Input**: unprotonated PDB (chain A = protein, chain B = ligand with explicit H) + ligand SMILES or SDF.  
**Output**: protonated, minimised PDB with relaxed sidechains and optimised hydrogen positions.

---

## How it works

1. **Ligand parameterisation** — GFN2-xTB partial charges (via `tblite`) replace the slow AM1-BCC step. GAFF2-2.11 bonded parameters are assigned by `openmmforcefields`. This runs once per ligand identity and is reused across thousands of binder designs.

2. **Protonation** — `Modeller.addHydrogens()` (OpenMM) sees the full protein + ligand complex when placing hydrogens. For neutral histidines it evaluates H-bond geometry against all atoms — including the GAFF2-parameterised ligand — to pick the correct HID/HIE tautomer. No external tools required.

3. **Energy minimisation** — ff14SB (protein) + GAFF2 (ligand) + GBn2 implicit solvent. Backbone heavy atoms and ligand heavy atoms are harmonically restrained; sidechains and all hydrogens are free.

---

## Installation

### Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for environment management
- `antechamber` and `parmchk2` (AmberTools) on `PATH` — used by `openmmforcefields` for GAFF2 atom typing

### Steps

```bash
cd 00_simple

# 1. Create venv and install PyPI dependencies
uv sync

# 2. Install the openff stack from GitHub source
#    (openff-toolkit, openff-units, openff-utilities are not on PyPI)
bash scripts/install_openff.sh

# 3. Install this package in editable mode
uv run python -m pip install -e . --no-build-isolation
```

> **Note**: `uv sync` will wipe the openff packages if re-run. Re-run `bash scripts/install_openff.sh` to restore them.

---

## Usage

### Command line

```bash
# Ligand supplied as SDF file
uv run protonator input.pdb --ligand-file ligand.sdf --output-dir out/

# Ligand supplied as SMILES string
uv run protonator input.pdb --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --output-dir out/
```

Output is written to `out/<input_stem>_relaxed.pdb`.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--ligand-file` / `--smiles` | — | Ligand source (one required) |
| `--output-dir` | `output/` | Directory for relaxed PDB |
| `--ph` | `7.4` | pH for protonation state assignment |
| `--restraint-k` | `50.0` | Backbone + ligand restraint force constant (kcal/mol/Å²) |
| `--tolerance` | `10.0` | Minimisation convergence (kJ/mol/nm) |
| `--no-freeze-ligand` | off | Allow ligand heavy atoms to move during minimisation |
| `--recompute-ligand` | off | Recompute xTB charges + GAFF2 per structure instead of reusing |

### Python API — batch parallelisation

The design is optimised for evaluating thousands of binder designs against a fixed ligand. The expensive ligand parameterisation step runs once; `LigandParams` is picklable and safe to pass across `multiprocessing` workers.

```python
import multiprocessing
from pathlib import Path
from protonator.ligand import prepare_ligand
from protonator.minimize import minimize_complex

# Run once per ligand identity
ligand_params = prepare_ligand("ligand.sdf", is_file=True)

def relax(pdb_path):
    out = Path("out") / (pdb_path.stem + "_relaxed.pdb")
    minimize_complex(pdb_path, ligand_params, out)

with multiprocessing.Pool(32) as pool:
    pool.map(relax, list(design_pdbs))
```

---

## Input format

The input PDB must follow AF3 conventions:

- **Chain A** — protein, no hydrogen atoms
- **Chain B** — ligand, with explicit hydrogens already placed and the correct protonation state

The ligand protonation state must be known in advance (encoded in the SMILES or SDF). The pipeline does not attempt to predict it.

---

## Force field summary

| Component | Force field |
|-----------|-------------|
| Protein bonded + vdW | AMBER ff14SB |
| Protein solvation | GBn2 implicit solvent |
| Ligand bonded + vdW | GAFF2-2.11 |
| Ligand charges | GFN2-xTB (Mulliken) |
| Restrained atoms | Protein backbone (N/Cα/C/O) + ligand heavy atoms |
| Free atoms | Protein sidechains + all hydrogens |

---

## Dependencies not on PyPI

`openff-toolkit`, `openff-units`, and `openff-utilities` must be installed from GitHub source via `scripts/install_openff.sh`. All other dependencies are in `pyproject.toml`.

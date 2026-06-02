# protonator

Protonates and energy-minimises AF3-style protein–ligand complexes as a drop-in replacement for Rosetta's hydrogen placement + sidechain relaxation step.

**Input**: unprotonated PDB (chain A = protein, chain B = ligand with explicit H) + ligand SMILES or SDF.  
**Output**: protonated, minimised PDB with relaxed sidechains and optimised hydrogen positions.

---

## How it works

1. **Ligand parameterisation** — GFN2-xTB partial charges (via `tblite`) replace the slow AM1-BCC step. GAFF2-2.11 bonded parameters are assigned by `openmmforcefields`. Charges are computed from the 3-D coordinates of chain B in each input PDB, so the AF3-predicted pose determines the charge distribution for every structure.

2. **Protonation** — `Modeller.addHydrogens()` (OpenMM) sees the full protein + ligand complex when placing hydrogens. For neutral histidines it evaluates H-bond geometry against all atoms — including the GAFF2-parameterised ligand — to pick the correct HID/HIE tautomer. No external tools required.

3. **Energy minimisation** — ff14SB (protein) + GAFF2 (ligand) + GBn2 implicit solvent. Backbone heavy atoms and ligand heavy atoms are harmonically restrained; sidechains and all hydrogens are free.

---

## Installation

Installation uses **conda** (miniforge / mambaforge recommended). The OpenFF
stack and AmberTools are only distributed through conda-forge, not PyPI, so
conda manages the whole environment.

```bash
cd 00_simple

# 1. Create the conda environment (conda-forge: ambertools, openff stack,
#    openmm, rdkit, tblite, and all other runtime deps)
conda env create -f environment.yml

# 2. Install openmmforcefields and this package (--no-deps: conda already
#    provides all runtime dependencies; avoids pulling yanked PyPI packages)
conda activate protonator
pip install --no-deps openmmforcefields
pip install -e . --no-deps
```

### Singularity container

A [`protonator.def`](protonator.def) builds a self-contained image. Use the provided script (no root required):

```bash
bash build_container.sh
singularity run protonator.sif input.pdb --ligand-file ligand.sdf --output-dir out/
```

`build_container.sh` calls `singularity build --fakeroot protonator.sif protonator.def` — it installs miniforge3, creates the conda environment, and installs the `protonator` package inside the image.

---

## Usage

### Command line

With the `protonator` conda environment activated (`conda activate protonator`):

**Option 1 — SDF / MOL file** (recommended when available)

```bash
protonator input.pdb --ligand-file ligand.sdf --output-dir out/
```

Bond orders, formal charges, and 3-D coordinates are read directly from the
mol file.  The protonation state encoded in the SDF is used as-is.

**Option 2 — SMILES string**

```bash
protonator input.pdb \
    --smiles "CC1=NC(=Cc2cc(F)c([O-])c(F)c2)C(=O)N1C" \
    --output-dir out/
```

Bond orders are assigned to the ligand coordinates already present in chain B
of the input PDB via `AssignBondOrdersFromTemplate` — the AF3-predicted 3-D
pose is preserved.  The SMILES must encode the correct protonation state (e.g.
`[O-]` for a phenoxide).  A fresh conformer is **not** generated.

Output is written to `out/<input_stem>_relaxed.pdb`.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--ligand-file` / `--smiles` | — | Ligand source (exactly one required) |
| `--output-dir` | `output/` | Directory for relaxed PDB |
| `--ph` | `7.4` | pH for protonation state assignment |
| `--restraint-k` | `50.0` | Backbone + ligand restraint force constant (kcal/mol/Å²) |
| `--tolerance` | `30.0` | Minimisation convergence (kJ/mol/nm) |
| `--no-freeze-ligand` | off | Allow ligand heavy atoms to move during minimisation |
| `--no-sweep-hbonds` | off | Disable post-minimisation SER/THR/TYR hydroxyl sweep |
| `--max-iterations` | `0` | Max minimisation steps (0 = run until convergence) |

### Batch CLI — same ligand, many structures

`protonator_batch` processes a list of PDB files sharing the same ligand identity using a multiprocessing pool. Each worker computes xTB charges independently from chain B of its own PDB, so the AF3 pose is preserved per structure.

```bash
# Write one PDB path per line
ls designs/*.pdb > pdbs.txt

protonator_batch pdbs.txt "CC1=NC(=Cc2cc(F)c([O-])c(F)c2)C(=O)N1C" \
    --output-dir emin_out/ \
    --n-workers 32

# Apo mode (protein only, --smiles ignored)
protonator_batch pdbs.txt "" --apo --output-dir emin_apo/ --n-workers 32
```

| Flag | Default | Description |
|------|---------|-------------|
| `smiles` | (required) | Ligand SMILES shared by all structures |
| `--output-dir` | `batch_emin_output/` | Output directory |
| `--apo` | off | Protein-only mode; strips ligand |
| `--n-workers` | `1` | Parallel worker processes |
| `--ph` | `7.4` | pH for protonation state assignment |
| `--restraint-k` | `50.0` | Backbone + ligand restraint (kcal/mol/Å²) |
| `--tolerance` | `30.0` | Convergence criterion (kJ/mol/nm) |
| `--no-freeze-ligand` | off | Allow ligand heavy atoms to move |
| `--no-sweep-hbonds` | off | Disable hydroxyl orientation sweep |
| `--suffix` | auto | Override output filename suffix |
| `--max-iterations` | `0` | Max minimisation steps (0 = until convergence) |

Output filenames default to `{stem}_emin.pdb` (holo) or `{stem}_emin_apo.pdb` (apo).

### Batch script — different ligands per target

`scripts/run_batch.py` handles datasets where each target has its own ligand SDF. It expects one subdirectory per target, each containing `{name}_protein.pdb` and `{name}_ligand.sdf`:

```bash
python scripts/run_batch.py dataset_dir/ --jobs 32
python scripts/run_batch.py dataset_dir/ --jobs 32 --apo
```

Failures are logged to `dataset_dir/failures_holo.tsv` (or `failures_apo.tsv`).

### Python API

`minimize_complex` takes a SMILES string and handles parameterisation internally — xTB charges are computed from chain B of each input PDB, preserving the AF3-predicted pose.

```python
from protonator.minimize import minimize_complex, _init_worker_ff
import multiprocessing
from pathlib import Path

def relax(pdb_path):
    out = Path("out") / (Path(pdb_path).stem + "_emin.pdb")
    minimize_complex(pdb_path, "CC1=NC(=Cc2cc(F)c([O-])c(F)c2)C(=O)N1C", out)

with multiprocessing.Pool(
    32,
    initializer=_init_worker_ff,
    initargs=(1,),          # 1 CPU thread per worker
) as pool:
    pool.map(relax, design_pdbs)
```

`_init_worker_ff` sets all thread-count environment variables (including `OPENMM_CPU_THREADS`) and pre-loads the base ff14SB + GBn2 ForceField so the expensive XML read happens once per worker rather than once per structure.

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

## Dependencies

All dependencies are resolved by `conda env create -f environment.yml` from
conda-forge. The OpenFF stack (`openff-toolkit`, `openff-units`,
`openff-utilities`) and AmberTools are not available on PyPI in a usable state —
conda-forge is their official distribution channel — which is why the project
uses a conda environment rather than a pip/uv virtualenv. `pyproject.toml`
documents the Python-level dependencies and defines the `protonator` package.

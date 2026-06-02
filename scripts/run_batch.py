"""
Batch-process a directory of protein-ligand targets with protonator.

Each subdirectory must contain:
    {name}_protein.pdb
    {name}_ligand.sdf   (holo mode only)

Holo output: {dir}/{name}_relaxed.pdb
 Apo output: {dir}/{name}_apo_relaxed.pdb
Failures are logged to {dataset_dir}/failures.tsv.
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import traceback
from pathlib import Path


def _init_worker(threads_per_worker: int) -> None:
    """Set thread-count env vars before any library is imported in this worker."""
    import os
    t = str(threads_per_worker)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = t


def _process_target(args: tuple) -> tuple[str, str | None]:
    """
    Worker: (target_dir, apo, threads_per_worker) -> (name, error_or_None).
    Imported here so it is picklable on all platforms.
    """
    target_dir, apo, _ = args
    target_dir = Path(target_dir)
    name = target_dir.name
    pdb = target_dir / f"{name}_protein.pdb"

    if not pdb.exists():
        return name, f"missing protein PDB: {pdb}"

    if apo:
        out = target_dir / f"{name}_apo_relaxed.pdb"
        try:
            from protonator.minimize import minimize_apo
            minimize_apo(pdb, out)
            return name, None
        except Exception:
            return name, traceback.format_exc()
    else:
        sdf = target_dir / f"{name}_ligand.sdf"
        out = target_dir / f"{name}_relaxed.pdb"
        if not sdf.exists():
            return name, f"missing ligand SDF: {sdf}"
        try:
            from protonator.ligand import prepare_ligand
            from protonator.minimize import minimize_complex
            ligand_params = prepare_ligand(str(sdf), is_file=True)
            minimize_complex(pdb, ligand_params, out, tolerance=30.0)
            return name, None
        except Exception:
            return name, traceback.format_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path,
                        help="Directory containing one subdirectory per target.")
    parser.add_argument("--apo", action="store_true",
                        help="Run apo (protein-only) minimisation instead of holo.")
    parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="Parallel workers (default: 8).")
    parser.add_argument("-t", "--threads-per-worker", type=int, default=1,
                        help="CPU threads each worker may use (default: 1).")
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        sys.exit(f"Not a directory: {dataset_dir}")

    targets = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not targets:
        sys.exit("No subdirectories found.")

    tpw  = args.threads_per_worker
    mode = "apo" if args.apo else "holo"
    print(f"Found {len(targets)} targets — {mode} mode, "
          f"{args.jobs} workers ({tpw} thread(s) each)")

    work = [(str(t), args.apo, tpw) for t in targets]

    failures: list[tuple[str, str]] = []
    completed = 0

    with multiprocessing.Pool(args.jobs,
                              initializer=_init_worker,
                              initargs=(tpw,)) as pool:
        for name, error in pool.imap_unordered(_process_target, work):
            completed += 1
            if error:
                failures.append((name, error))
                print(f"[{completed:3d}/{len(targets)}] FAIL  {name}")
            else:
                print(f"[{completed:3d}/{len(targets)}] OK    {name}")

    log_path = dataset_dir / f"failures_{mode}.tsv"
    if failures:
        with open(log_path, "w") as fh:
            fh.write("target\terror\n")
            for name, err in failures:
                fh.write(f"{name}\t{err.replace(chr(10), ' | ')}\n")
        print(f"\n{len(failures)}/{len(targets)} failures — details in {log_path}")
    else:
        print(f"\nAll {len(targets)} {mode} targets completed successfully.")
        log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

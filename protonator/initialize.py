def _init_worker(threads_per_worker: int) -> None:
    """Set thread-count env vars before any library is imported in this worker.

    OPENMM_CPU_THREADS is essential: OpenMM's CPU platform ignores OMP_NUM_THREADS
    and reads this variable instead, defaulting to the hardware thread count when
    unset.  Modeller.addHydrogens() builds an internal Context with no explicit
    "Threads" property, so without this it spawns one thread per core (e.g. 64)
    before minimizeEnergy is ever reached — the source of CPU oversubscription.
    """
    import os
    t = str(threads_per_worker)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OPENMM_CPU_THREADS"):
        os.environ[var] = t

def _init_worker(threads_per_worker: int) -> None:
    """Set thread-count env vars before any library is imported in this worker.

    OPENMM_CPU_THREADS is essential: OpenMM's CPU platform ignores OMP_NUM_THREADS
    and reads this variable instead, defaulting to the hardware thread count when
    unset.  Modeller.addHydrogens() builds an internal Context with no explicit
    "Threads" property, so without this it spawns one thread per core (e.g. 64)
    before minimizeEnergy is ever reached — the source of CPU oversubscription.

    omp_set_num_threads() is called directly via ctypes because libgomp reads
    OMP_NUM_THREADS only at library initialization — setting the env var after
    libgomp is already loaded (e.g. pulled in by openmm) has no effect at
    runtime.  tblite (GFN2-xTB) links against libgomp and would otherwise use
    all available cores.
    """
    import os
    t = str(threads_per_worker)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OPENMM_CPU_THREADS"):
        os.environ[var] = t

    # Prevent OpenMM from initializing the CUDA/OpenCL platform plugins at
    # import time. Even when using the CPU platform exclusively, importing
    # openmm probes all available platforms and allocates a CUDA context per
    # worker process — holding GPU memory for the entire run with 0% utilization.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import ctypes, ctypes.util
        _gomp = ctypes.util.find_library("gomp")
        if _gomp:
            ctypes.CDLL(_gomp).omp_set_num_threads(threads_per_worker)
    except Exception:
        pass

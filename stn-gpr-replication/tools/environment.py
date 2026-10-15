import os, platform, sys
import numpy as np, scipy

def info():
    print("CPU      :", platform.processor() or platform.machine())
    print("Cores    :", os.cpu_count())
    try:
        import psutil
        print("Physical :", psutil.cpu_count(logical=False))
        print("Memory   :", round(psutil.virtual_memory().total / 2**30), "GB")
    except ImportError:
        print("Memory   : install psutil")
    print("OS       :", platform.platform())
    print("Python   :", sys.version.split()[0])
    print("NumPy    :", np.__version__)
    print("SciPy    :", scipy.__version__)
    try:
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            print("BLAS     :", p["internal_api"], p["version"], "threads =", p["num_threads"])
    except ImportError:
        print("BLAS     : install threadpoolctl")

info()
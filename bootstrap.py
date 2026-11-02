"""Run before importing NumPy in every new kernel."""
import os
import sys

def configure(threads=8):
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['NUMBA_NUM_THREADS'] = str(threads)
    os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
    os.environ['ARROW_NUM_THREADS'] = '2'
    if 'numpy' in sys.modules:
        from threadpoolctl import threadpool_limits
        global _limits
        _limits = threadpool_limits(limits=1)
    return threads
configure(int(os.environ.get('ASSEMBLY_THREADS', '8')))

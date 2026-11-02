from pathlib import Path
import subprocess
import json
import sys
root=Path(__file__).resolve().parent
if sys.maxsize<=2**32:raise SystemExit('64-bit Python is required.')
if sys.version_info[:2] not in ((3,11),(3,12)):
    raise SystemExit('Use Python 3.11 or 3.12 for the tested Numba / binary-wheel combination.')
subprocess.check_call([sys.executable,'-m','pip','install','--upgrade','pip'])
subprocess.check_call([sys.executable,'-m','pip','install','-r',str(root/'requirements-tested.txt')])
subprocess.check_call([sys.executable,'-m','pip','check'])
subprocess.check_call([sys.executable,'-m','ipykernel','install','--user','--name','global-knowledge-python',
                       '--display-name','Python - Global Knowledge Assembly'])
from jupyter_client.kernelspec import KernelSpecManager
path=Path(KernelSpecManager().get_kernel_spec('global-knowledge-python').resource_dir)/'kernel.json'
spec=json.loads(path.read_text(encoding='utf-8'))
threads=str(json.loads((root/'config.json').read_text(encoding='utf-8-sig'))['threads'])
spec.setdefault('env',{}).update({name:'1' for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'OMP_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
spec['env'].update(NUMBA_NUM_THREADS=threads,NUMBA_THREADING_LAYER='workqueue',ARROW_NUM_THREADS='2',ASSEMBLY_THREADS=threads)
path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
print('Kernel-start thread limits installed. Select Python - Global Knowledge Assembly. Restart existing kernels.')

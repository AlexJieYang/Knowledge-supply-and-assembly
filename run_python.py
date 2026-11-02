"""Fresh processes per stage prevent accumulation of indexes and mapped caches."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--stage',choices=['metadata','references','metrics','networks','descriptions','figures'])
    p.add_argument('--force',action='store_true',help='Recompute selected stages. Existing raw input is never changed.')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    import json
    c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    os.environ['ASSEMBLY_THREADS']=str(c['threads'])
    import bootstrap
    bootstrap.configure(c['threads'])
    from assembly.runtime import config,run,STAGES
    if a.worker:
        run(config(a.config),a.stage,force=a.force);return
    stages=[a.stage] if a.stage else STAGES
    for stage in stages:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--config',str(Path(a.config).resolve()),'--worker','--stage',stage]
        if a.force:cmd.append('--force')
        subprocess.check_call(cmd,cwd=Path(__file__).resolve().parent)
    print('Python stages complete. Run 06_R_Conditional_Associations.ipynb separately.')

if __name__=='__main__':main()

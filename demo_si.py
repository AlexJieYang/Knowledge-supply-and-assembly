"""Run a small synthetic SI workflow; its results are never empirical evidence."""
from pathlib import Path
import argparse
import os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='_si_synthetic_check')
    parser.add_argument('--audit-demo',action='store_true',help='Also exercise audit aggregation on synthetic DataFrame batches; not Parquet I/O.')
    args=parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
    from si_tests.synthetic_inputs import create
    from si_support.core import Context
    from si_support.analysis import build_tables
    from si_support.figures import render_all
    config,sample=create(Path(args.output))
    ctx=Context(config,sample_file=sample,synthetic=True)
    build_tables(ctx)
    if args.audit_demo:
        import numpy as np
        import pandas as pd
        from si_support.analysis import audit_frames
        a=pd.read_csv(sample);a['eligible']=True
        a.loc[a.index%9==0,['eligible','known_fields']]=[False,0]
        a.loc[a.index%13==0,'known_countries']=0
        a.loc[a.known_countries==0,'eligible']=False
        kinds=np.arange(len(a),dtype=np.int8)%3;ids=a.paper_id.to_numpy()
        audit_frames(ctx,(a.iloc[i:i+333].copy() for i in range(0,len(a),333)),kinds,ids,expected_rows=len(a))
        ctx.event('tables','synthetic_audit_aggregation','complete','Synthetic DataFrame batches; no Parquet validation claimed.')
    render_all(ctx);ctx.finish()
    print('SYNTHETIC functional/layout checks only:',ctx.out)


if __name__=='__main__':main()

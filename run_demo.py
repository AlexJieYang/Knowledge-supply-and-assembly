from pathlib import Path
import argparse
import json
import bootstrap
bootstrap.configure(2)
from assembly.synthetic import create,configuration
from assembly.runtime import run,STAGES

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=6000)
    parser.add_argument('--output',default='_synthetic_check');parser.add_argument('--figures',action='store_true')
    parser.add_argument('--joint-semantic',action='store_true');args=parser.parse_args()
    project=Path(__file__).resolve().parent;root=project/args.output;root.mkdir(parents=True,exist_ok=True)
    create(root/'input',args.n);c=configuration(root/'input',root/'results',project);c['synthetic']=True
    if args.joint_semantic:
        c['joint_semantic'].update(enabled=True,index_parquet=str(root/'input/embedding_index.parquet'),matrix_npy=str(root/'input/embeddings.npy'),max_papers=300)
    (root/'config.json').write_text(json.dumps({k:v for k,v in c.items() if k!='_project_dir'},indent=2),encoding='utf-8')
    for stage in STAGES:
        if stage=='figures' and not args.figures:continue
        run(c,stage,force=True)
    print('SYNTHETIC functional run only. Results are not empirical evidence:',root)
